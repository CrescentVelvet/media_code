#!/usr/bin/env python3
"""AUDIT-13 · I-01/I-02 归位验证：把「PLY 交付口径」抬回官方水平。

为什么要这步：`audit_9` 已证明**内存高斯**按官方口径渲染是 22.11 dB，而我们的 PLY 只有
18.97 dB。差 3.14 dB 全在导出选项（裁剪 −2.34 / 仅 DC −1.15）。但训练与分发走的是 PLY，
所以必须证明「导出成 PLY 之后仍然拿得到 22.11」，否则归位只是纸面结论。

三档 PLY（由 `98_da3_ff_init.py` 的 `EXPORT_OFFICIAL=1` 在同一次前向导出）：

  i01_fullpts_dc  不裁剪 + 仅 DC      → 预期 = 内存 B 档 20.96
  i02_pruned_sh   裁剪   + 全 SH      → 预期 = 内存 C 档 19.77
  i12_fullpts_sh  不裁剪 + 全 SH      → 预期 = 内存 A 档 22.11（官方等价）
  （另加既有 gs_ply/0000.ply = 裁剪 + DC → 预期 18.97，作历史口径对照）

判据（跑前定死，见 IMPROVEMENTS.md §2）：
  P1 官方档 PLY 的均值 PSNR 与内存 A 档之差 < 0.05 dB（导出无损）
  P2 四档的排序与内存版完全一致，且每档差值 < 0.05 dB
  P3 i12 相对历史 PLY 口径 ≥ +2.9 dB（台账登记 +3.14，留 0.24 容差）

一个必须自己防的坑（E-07 的同类）：3DGS PLY 的 `f_rest` 是 **channel-major**
（`harmonics[...,1:].flatten(start_dim=1)`，即 [ch0 的 K 个, ch1 的 K 个, ch2 的 K 个]），
读回必须 `reshape(N,3,K).transpose(0,2,1)` 才是 gsplat 要的 (N,K,3)。排错会得到
「颜色像但方向反」的奇怪结果——P1 判据就是它的兜底。
"""
import json
import os
import re

import numpy as np
import torch
from gsplat import rasterization
from PIL import Image
from skimage.metrics import structural_similarity as ssim

SC = os.environ["SC"]                       # 场景目录（含 official_export/ 与 gs_ply/）
CAM = os.environ.get("CAM", "")             # 相机 npz（默认 SC/cameras.npz）；指定外部则同时做位姿对账
MEM_NPZ = os.environ.get("MEM_NPZ", "")     # 内存高斯 dump（official_dump.npz），作参照 + GT
OUT = os.environ.get("OUT", os.path.join(SC, "official_export"))
VIEWS = os.environ.get("VIEWS", "all")
dev = "cuda"


def log(*a):
    print(*a, flush=True)


# ── PLY 读取：按 header 真实偏移，不猜 4096（audit_8 第一版的教训） ─────────────
def read_ply(path):
    with open(path, "rb") as f:
        head = b""
        while b"end_header" not in head:
            head += f.read(1)
        # 补到行尾
        tail = f.read(1)
        while tail not in (b"\n", b""):
            head += tail
            tail = f.read(1)
        head += tail
        offset = len(head)
        text = head.decode("ascii", "replace")
        n = int(re.search(r"element vertex (\d+)", text).group(1))
        names = re.findall(r"^property float (\w+)$", text, re.M)
        dt = np.dtype([(nm, "<f4") for nm in names])
        f.seek(offset)
        raw = np.fromfile(f, dtype=dt, count=n, offset=0)
    if raw.shape[0] != n:
        raise SystemExit(f"❌ {path}: 读到 {raw.shape[0]} 顶点，头里写 {n}")
    means = np.stack([raw["x"], raw["y"], raw["z"]], 1).astype(np.float32)
    scales = np.exp(np.stack([raw[f"scale_{i}"] for i in range(3)], 1)).astype(np.float32)
    opac = (1.0 / (1.0 + np.exp(-raw["opacity"].astype(np.float64)))).astype(np.float32)
    quats = np.stack([raw[f"rot_{i}"] for i in range(4)], 1).astype(np.float32)
    f_dc = np.stack([raw[f"f_dc_{i}"] for i in range(3)], 1).astype(np.float32)      # (N,3)
    rest_names = [nm for nm in names if nm.startswith("f_rest_")]
    n_rest = len(rest_names)
    k = n_rest // 3
    if n_rest % 3 != 0:
        raise SystemExit(f"❌ f_rest 个数 {n_rest} 不是 3 的倍数")
    if k:
        f_rest = np.stack([raw[f"f_rest_{i}"] for i in range(n_rest)], 1)            # (N,3K)
        f_rest = f_rest.reshape(len(means), 3, k).transpose(0, 2, 1)                  # (N,K,3)
        sh = np.concatenate([f_dc[:, None, :], f_rest], 1)                            # (N,1+K,3)
    else:
        sh = f_dc[:, None, :]
    return dict(means=means, scales=scales, opac=opac, quats=quats, sh=sh,
                n=int(n), n_rest=n_rest, deg=int(round((1 + k) ** 0.5)) - 1)


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 10 * np.log10(255.0 ** 2 / max(mse, 1e-12))


def sharp(rgb):
    import cv2
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


# ── 相机与 GT ────────────────────────────────────────────────────────────────
cam = np.load(CAM or os.path.join(SC, "cameras.npz"))
ext = cam["extrinsics"].astype(np.float32)      # (V,3,4) w2c，米制（与高斯同帧）
K = cam["intrinsics"].astype(np.float32)
depth = cam["depth"]
V, H, W = depth.shape
log(f"📷 {V} 视角  {W}x{H}  fx={K[0,0,0]:.2f} fy={K[0,1,1]:.2f}  "
    f"scale_factor={float(cam['scale_factor']):.4f}  gs_scale_applied={int(cam['gs_scale_applied'])}")

# 两次前向一致性闸门：外部给了 CAM 而本次产物另有 cameras.npz，就互相比一次。
# 若不一致，"PLY 与内存口径的差"里会混进"两次前向不同"，P2 判据立刻失真。
gate_pose = None
own_cam = os.path.join(SC, "cameras.npz")
if CAM and os.path.exists(own_cam):
    o = np.load(own_cam)
    dR = float(np.abs(o["extrinsics"][:, :, :3] - cam["extrinsics"][:, :, :3]).max())
    dt_ = float(np.abs(o["extrinsics"][:, :, 3] - cam["extrinsics"][:, :, 3]).max())
    dK = float(np.abs(o["intrinsics"] - cam["intrinsics"]).max())
    gate_pose = dict(dR=dR, dt=dt_, dK=dK)
    log(f"🔁 本次前向 vs 参照前向：ΔR {dR:.2e}  Δt {dt_:.2e} m  ΔK {dK:.2e} px  "
        f"{'✅ 同一世界系' if (dR < 1e-6 and dt_ < 1e-6) else '⚠️ 位姿有差，P2 需打折看'}")

GT = None
if MEM_NPZ and os.path.exists(MEM_NPZ):
    mem = np.load(MEM_NPZ)
    GT = mem["processed_images"]
    log(f"🖼️ GT 取自内存 dump（与模型实际看到的输入帧同源）{GT.shape}")

CASES = (
    ("i12_fullpts_sh", "official_export/i12_fullpts_sh/0000.ply", "官方等价（不裁剪+全SH）"),
    ("i01_fullpts_dc", "official_export/i01_fullpts_dc/0000.ply", "I-01 单独（不裁剪+DC）"),
    ("i02_pruned_sh", "official_export/i02_pruned_sh/0000.ply", "I-02 单独（裁剪+全SH）"),
    ("hist_pruned_dc", "gs_ply/0000.ply", "历史 PLY 口径（裁剪+DC）"),
)

view_list = list(range(V)) if VIEWS.strip() == "all" else [int(x) for x in VIEWS.split(",")]
rows, agg, meta = [], {}, {}


def render(means, quats, scales, opac, sh, v, deg):
    vm = np.vstack([ext[v], np.array([0, 0, 0, 1], np.float32)])
    with torch.no_grad():
        img, alpha, _ = rasterization(
            means, quats, scales, opac, sh,
            torch.from_numpy(vm).float().to(dev)[None],
            torch.from_numpy(K[v]).float().to(dev)[None], W, H,
            sh_degree=deg, packed=False,
        )
    return (img[0].clamp(0, 1).mul(255).byte().cpu().numpy(),
            alpha[0].squeeze(-1).cpu().numpy())


for tag, rel, note in CASES:
    path = os.path.join(SC, rel)
    if not os.path.exists(path):
        log(f"⚠️ 缺 {tag}：{path}（跳过）")
        continue
    g = read_ply(path)
    log(f"\n📦 {tag}: {g['n']:,} 点  SH degree {g['deg']}  "
        f"α中位 {np.median(g['opac']):.4f}  σ中位(m) {np.round(np.median(g['scales'],0),6)}  "
        f"[{note}]")
    means = torch.from_numpy(g["means"]).to(dev)
    scales = torch.from_numpy(g["scales"]).to(dev)
    quats = torch.from_numpy(g["quats"]).to(dev)
    opac = torch.from_numpy(g["opac"]).to(dev)
    sh = torch.from_numpy(g["sh"]).to(dev)
    ps, ss, sp = [], [], []
    for v in view_list:
        img, a = render(means, quats, scales, opac, sh, v, g["deg"])
        if GT is not None:
            ps.append(psnr(img, GT[v]))
            ss.append(float(ssim(img, GT[v], channel_axis=-1, data_range=255)))
        sp.append(sharp(img))
        rows.append(dict(tag=tag, view=v, psnr=ps[-1] if ps else None,
                         ssim=ss[-1] if ss else None,
                         sharp=sp[-1], alpha_med=float(np.median(a))))
        if v in (0, 8, 16):
            Image.fromarray(img).save(os.path.join(OUT, f"ply_{tag}_view{v:03d}.png"))
    agg[tag] = dict(n=g["n"], deg=g["deg"],
                    psnr=float(np.mean(ps)) if ps else None,
                    ssim=float(np.mean(ss)) if ss else None,
                    sharp=float(np.mean(sp)),
                    alpha_med=float(np.median(g["opac"])))
    meta[tag] = dict(path=path, note=note, n=g["n"], n_rest=g["n_rest"])
    log(f"   ⇒ 均值 PSNR {agg[tag]['psnr']:.2f} dB  SSIM {agg[tag]['ssim']:.3f}  "
        f"锐度 {agg[tag]['sharp']:.0f}")
    del means, scales, quats, opac, sh, g
    torch.cuda.empty_cache()

log("\n" + "=" * 78)
log(f"{'档':<18}{'点数':>10}{'deg':>5}{'PSNR':>9}{'SSIM':>8}{'锐度':>8}")
for tag, *_ in CASES:
    if tag not in agg:
        continue
    a = agg[tag]
    log(f"{tag:<18}{a['n']:>10,}{a['deg']:>5}{a['psnr']:>9.2f}{a['ssim']:>8.3f}{a['sharp']:>8.0f}")

# ── 判据自检 ─────────────────────────────────────────────────────────────────
gate = {}
if "i12_fullpts_sh" in agg and "hist_pruned_dc" in agg:
    gate["P3_delta_vs_hist"] = round(agg["i12_fullpts_sh"]["psnr"]
                                    - agg["hist_pruned_dc"]["psnr"], 3)
    gate["P3_pass"] = bool(gate["P3_delta_vs_hist"] >= 2.9)
MEM_REF = {"i12_fullpts_sh": 22.11, "i01_fullpts_dc": 20.96,
           "i02_pruned_sh": 19.77, "hist_pruned_dc": 18.97}
gate["vs_mem"] = {k: round(agg[k]["psnr"] - MEM_REF[k], 3) for k in agg}
gate["P2_pass"] = bool(all(abs(v) < 0.05 for v in gate["vs_mem"].values()))
log("\n判据：")
log(f"  P2 PLY 与内存口径逐档差 < 0.05 dB ? {'✅' if gate['P2_pass'] else '❌'}  {gate['vs_mem']}")
if "P3_delta_vs_hist" in gate:
    log(f"  P3 i12 相对历史 PLY 口径 ≥ +2.9 dB ? "
        f"{'✅' if gate['P3_pass'] else '❌'}  {gate['P3_delta_vs_hist']:+.2f} dB")

dst = os.path.join(OUT, "ply_metrics.json")
with open(dst, "w", encoding="utf-8", newline="\n") as f:
    json.dump(dict(scene=SC, cam=CAM, mem_npz=MEM_NPZ, views=VIEWS, cases=meta,
                   pose_gate=gate_pose, mean=agg, mem_ref=MEM_REF, gate=gate, rows=rows),
              f, indent=1, ensure_ascii=False)
log(f"\n💾 {dst}")
