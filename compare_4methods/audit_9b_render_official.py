#!/usr/bin/env python3
"""AUDIT-9b 用 gsplat 忠实复刻 DA3 官方 render_3dgs，做三档对照。

官方 render_3dgs 的关键调用（model/utils/gs_renderer.py）：
  rasterization(means, quats, scales, opacities, colors=sh(N,n,3),
                viewmats=w2c, Ks=K, backgrounds=0, render_mode="RGB+D",
                width=w, height=h, packed=False, sh_degree=degree)
K 由归一化内参经 fov 反算，主点强制 (w/2, h/2)。

三档：
  A 官方等价：未缩放相机(t/sf) + 全 SH      ← 官方视频导出的真实路径
  B 米制等价：缩放相机(t)    + 全 SH      ← 应与 A 完全一致（全局相似变换）
  C 我们的口径：缩放相机(t)  + 仅 DC      ← fig20 / R-3 用的口径
"""
import json
import os
import re

import numpy as np
import torch
from gsplat import rasterization
from PIL import Image

NPZ = os.environ["NPZ"]
OUT = os.environ["OUT"]
VIEWS = os.environ.get("VIEWS", "0,8,16")
SH_MODE = os.environ.get("SH_MODE", "full")   # full | dc
dev = "cuda"


def log(*a):
    print(*a, flush=True)


d = np.load(NPZ)
means = torch.from_numpy(d["means"]).float().to(dev)
scales = torch.from_numpy(d["scales"]).float().to(dev)
quats = torch.from_numpy(d["quats"]).float().to(dev)
opac = torch.from_numpy(d["opacities"].reshape(-1)).float().to(dev)
harm = d["harmonics"]                        # (N,3,n)
V, H, W = d["depth"].shape
K = d["intrinsics"].astype(np.float32)
GT = d["processed_images"]
n_sh = harm.shape[-1]
degree = int(round(n_sh ** 0.5)) - 1
log(f"📦 {len(means):,} 高斯  harmonics {harm.shape} → degree {degree}  "
    f"图像 {W}x{H}  scale_factor={float(d['scale_factor']):.4f}")

# 官方把 harmonics (b,g,3,n) rearrange 成 (b,g,n,3)；我们这里是 (N,3,n) → (N,n,3)
sh_full = torch.from_numpy(np.transpose(harm, (0, 2, 1)).copy()).float().to(dev)
sh_dc = torch.from_numpy(np.transpose(harm[:, :, :1], (0, 2, 1)).copy()).float().to(dev)

# 顺手核对：本次前向的内参 vs 训练/评测一直在用的 cameras.npz 内参
old = os.environ.get("OLD_NPZ", "")
if old and os.path.exists(old):
    o = np.load(old)
    Ko = o["intrinsics"].astype(np.float64)
    d_fx = K[:, 0, 0] / Ko[:, 0, 0] - 1
    d_fy = K[:, 1, 1] / Ko[:, 1, 1] - 1
    log(f"🔍 与旧 npz 的内参比：fx {np.median(d_fx)*100:+.3f}%  fy {np.median(d_fy)*100:+.3f}%  "
        f"（旧 fx {Ko[0,0,0]:.2f} / 新 {K[0,0,0]:.2f}；fy {Ko[0,1,1]:.2f} / {K[0,1,1]:.2f}）")
    if "extrinsics" in o.files and abs(float(o["scale_factor"]) - float(d["scale_factor"])) > 1e-3:
        log(f"⚠️ scale_factor 不同：旧 {float(o['scale_factor']):.4f} 新 {float(d['scale_factor']):.4f}")


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 10 * np.log10(255.0 ** 2 / max(mse, 1e-12))


def render(idx, Ks, viewmat, sh, deg):
    vm = np.asarray(viewmat, dtype=np.float32)
    if vm.shape == (3, 4):                       # 官方容器常用 (V,3,4)，gsplat 要 4x4
        vm = np.vstack([vm, np.array([0, 0, 0, 1], np.float32)])
    with torch.no_grad():
        img, alpha, _ = rasterization(
            means[idx], quats[idx], scales[idx], opac[idx],
            sh[idx], torch.from_numpy(vm).float().to(dev)[None],
            torch.from_numpy(Ks).float().to(dev)[None], W, H,
            sh_degree=deg, packed=False,
        )
    return (img[0].clamp(0, 1).mul(255).byte().cpu().numpy(),
            alpha[0].squeeze(-1).cpu().numpy())


# 复刻 save_gaussian_ply 的裁剪：边框 8/256 + 逐视角深度 q90
dep_np = d["depth"]
th, tw = int(8 / 256 * H), int(8 / 256 * W)
keep = np.zeros((V, H, W), bool)
keep[:, th:H - th, tw:W - tw] = True
keep &= dep_np <= np.quantile(dep_np.reshape(V, -1), 0.90, axis=1).reshape(V, 1, 1)
idx_keep = torch.from_numpy(keep.reshape(-1)).to(dev)
idx_all = torch.ones(len(means), dtype=torch.bool, device=dev)
log(f"✂️ 裁剪后保留 {int(idx_keep.sum()):,} / {len(means):,}"
    f"（{float(idx_keep.float().mean())*100:.1f}%）")


rows = []
view_list = list(range(V)) if VIEWS.strip() == "all" else [int(x) for x in VIEWS.split(",")]
CASES = (
    ("A_official", "extrinsics_unscaled", sh_full, degree, idx_all),
    ("A_dc", "extrinsics_unscaled", sh_dc, 0, idx_all),
    ("A_pruned_full", "extrinsics_unscaled", sh_full, degree, idx_keep),
    ("A_pruned_dc", "extrinsics_unscaled", sh_dc, 0, idx_keep),
    ("B_metric_full", "extrinsics", sh_full, degree, idx_all),
)
for v in view_list:
    gt = GT[v]
    outs = {}
    for tag, extkey, sh, deg, sel in CASES:
        img, a = render(sel, K[v], d[extkey][v], sh, deg)
        outs[tag] = img
        rows.append(dict(view=v, tag=tag, psnr=psnr(img, gt),
                         alpha_med=float(np.median(a)), alpha_mean=float(a.mean())))
        if len(view_list) <= 8:
            Image.fromarray(img).save(os.path.join(OUT, f"{tag}_view{v:03d}.png"))
    if len(view_list) <= 8 or v in (0, 8, 16):
        ab = psnr(outs["A_official"], outs["B_metric_full"])
        log(f"  view {v:2d}  GT: A {rows[-5]['psnr']:.2f} | A_dc {rows[-4]['psnr']:.2f} | "
            f"A_pr_full {rows[-3]['psnr']:.2f} | A_pr_dc {rows[-2]['psnr']:.2f} | "
            f"B {rows[-1]['psnr']:.2f}   ‖A−B‖ {ab:.1f} dB   α中位 {rows[-5]['alpha_med']:.3f}")

agg = {}
for tag, *_ in CASES:
    agg[tag] = float(np.mean([r["psnr"] for r in rows if r["tag"] == tag]))
log(f"\n均值 PSNR（{len(view_list)} 视角，对模型实际看到的输入帧）：")
for k, v in agg.items():
    log(f"  {k:<16} {v:.2f} dB")

dst = os.path.join(OUT, "audit_official_render.json")
with open(dst, "w", encoding="utf-8", newline="\n") as f:
    json.dump(dict(npz=NPZ, views=VIEWS, mean_psnr=agg, rows=rows,
                   note="A=官方(未缩放相机+全SH) B=米制相机+全SH C=米制相机+仅DC"),
              f, indent=1, ensure_ascii=False)
log(f"\n💾 {dst}")
