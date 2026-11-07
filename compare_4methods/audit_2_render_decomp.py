#!/usr/bin/env python3
"""AUDIT-2：把「重影/模糊」拆成「自视角贡献」与「跨视角污染」两部分（只读诊断）

## 为什么要拆

每个高斯都能追溯它来自哪个视角（DA3 的 GS 头是逐像素预测，导出顺序 = view-major，
屏蔽规则可精确复刻）。于是同一个渲染器可以做三组对照：

  all    全部高斯            —— 我们看到的成品
  self   只留本视角的高斯     —— 该视角"自己的"几何，等于自一致性上限
  other  去掉本视角的高斯     —— 只剩"别的视角搬过来的几何"

若 self 明显好于 all，说明糊/重影来自**跨视角几何不一致**（模型侧）；
若 self 本身也糊，说明问题在**高斯头的表达能力或参数空间**（缩放/朝向/SH/opacity）。

注意：self 组只覆盖部分像素（每视角约 0.68 个高斯/像素），直接算 PSNR 会被空洞拖低。
所以主指标统一在 **self 覆盖到的像素集合**（alpha_self>0.5）上算，三组同口径。

## 防再犯

本脚本自带一条闸门：`all` 组必须与 94 的既有输出逐像素一致（PSNR>45 dB），
否则说明两条渲染路径已经分叉，先停下来对齐，不要用它的结论。

用法（WSL，querysplat env，经 94_render.sh 的同一套 CUDA 环境）：
  PLY=...cameras.npz NPZ=... OUT=... [VIEWS=0,8,16] \
    python compare_4methods/audit_2_render_decomp.py
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from gsplat import rasterization

PLY = Path(os.path.expanduser(os.environ["PLY"]))
NPZ = Path(os.path.expanduser(os.environ["NPZ"]))
OUT = Path(os.path.expanduser(os.environ["OUT"]))
VIEWS = [int(x) for x in os.environ.get("VIEWS", "0,8,16,24").split(",")]
SIZE = os.environ.get("RENDER_SIZE", "378x504")
REF_DIR = Path(os.path.expanduser(os.environ.get(
    "REF", "~/output/da3_results/_captures/佳佳抱枕/images")))


def log(s):
    print(s, flush=True)


def load_ply(path):
    with open(path, "rb") as f:
        raw = f.read(65536)
        end = raw.index(b"end_header") + len(b"end_header") + 1
        head = raw[:end].decode("ascii", "replace")
        n = int(re.search(r"element vertex (\d+)", head).group(1))
        props = [l.split()[2] for l in head.splitlines() if l.startswith("property")]
        f.seek(end)
        arr = np.fromfile(f, dtype=np.float32).reshape(n, len(props))
    nm = {k: i for i, k in enumerate(props)}
    means = arr[:, [nm["x"], nm["y"], nm["z"]]]
    sh = arr[:, [nm["f_dc_0"], nm["f_dc_1"], nm["f_dc_2"]]]
    rest_idx = sorted((int(k.split("_")[-1]), i) for k, i in nm.items() if k.startswith("f_rest_"))
    if rest_idx:
        rest = arr[:, [i for _, i in rest_idx]]
        n_rest = rest.shape[1] // 3
        rest = rest.reshape(n, 3, n_rest).transpose(0, 2, 1)
        sh = np.concatenate([sh[:, None, :], rest], axis=1)
    else:
        sh = sh[:, None, :]
    return (means, sh, arr[:, nm["opacity"]],
            arr[:, [nm["scale_0"], nm["scale_1"], nm["scale_2"]]],
            arr[:, [nm["rot_0"], nm["rot_1"], nm["rot_2"], nm["rot_3"]]])


def view_ids_for_ply(path, npz_path, xyz=None, op_logit=None,
                     min_opacity=0.005, max_dist=5.0):
    """复刻 save_gaussian_ply 的屏蔽（border + depth q90），返回每个顶点的来源视角。

    若顶点数不等于屏蔽数，再尝试复刻 98b_ply_to_3dgs_ckpt 的剪枝
    （opacity>min_opacity 且 距最近相机<max_dist）—— 3DGS 在 densify 关闭时
    只做参数更新、不增删点，所以训练产物的点集/顺序 = ckpt，可直接套用。
    """
    d = np.load(npz_path)
    dep = d["depth"]
    V, H, W = dep.shape
    th, tw = int(8 / 256 * H), int(8 / 256 * W)
    m = np.zeros((V, H, W), bool)
    m[:, th:H - th, tw:W - tw] = True
    m &= dep <= np.quantile(dep.reshape(V, -1), 0.90, axis=1).reshape(V, 1, 1)
    cnt = m.reshape(V, -1).sum(1)
    with open(path, "rb") as f:
        head = f.read(4096).decode("ascii", "replace")
    n = int(re.search(r"element vertex (\d+)", head).group(1))
    vid_full = np.repeat(np.arange(V), cnt)
    if n == len(vid_full):
        return vid_full, "导出屏蔽"

    if xyz is None or op_logit is None:
        raise SystemExit(f"❌ 顶点数 {n:,} ≠ 屏蔽 {len(vid_full):,}，且未提供 xyz/opacity 用于复刻剪枝")
    # ⚠️ 剪枝必须用**初始化时**的 opacity/xyz（98b 就是拿 gs_ply 剪的）；
    #    训练后 opacity 已变，用它复刻会得到不同的集合。
    init_ply = os.environ.get("INIT_PLY", "")
    if init_ply:
        xyz, _, op_logit, _, _ = load_ply(Path(os.path.expanduser(init_ply)))
        if len(xyz) != len(vid_full):
            raise SystemExit(f"❌ INIT_PLY 顶点数 {len(xyz):,} ≠ 屏蔽 {len(vid_full):,}")
    e = d["extrinsics"].astype(np.float64)
    C = -np.einsum("nij,nj->ni", e[:, :, :3].transpose(0, 2, 1), e[:, :, 3])
    op = 1.0 / (1.0 + np.exp(-op_logit.astype(np.float64)))
    keep = op > min_opacity
    dd = np.sqrt(((xyz[:, None, :].astype(np.float64) - C[None]) ** 2).sum(-1)).min(1)
    keep &= dd < max_dist
    if int(keep.sum()) != n:
        raise SystemExit(f"❌ 复刻剪枝得 {int(keep.sum()):,}，与实际 {n:,} 仍不符 → 顺序假设不成立")
    return vid_full[keep], f"导出屏蔽+98b剪枝({int(keep.sum()):,}/{len(vid_full):,})"


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def main():
    import cv2
    dev = "cuda"
    means_np, sh_np, op_np, sc_np, q_np = load_ply(PLY)
    N = len(means_np)
    log(f"📦 {PLY.name}: {N:,} 高斯")
    vid, how = view_ids_for_ply(PLY, NPZ, xyz=means_np, op_logit=op_np)
    log(f"✅ 来源视角复原（{how}）：每视角 {np.bincount(vid).min()}~{np.bincount(vid).max()} 点")

    sh_degree = int(round(sh_np.shape[1] ** 0.5)) - 1
    means = torch.from_numpy(means_np).to(dev)
    sh = torch.from_numpy(sh_np).to(dev)
    opacity = torch.sigmoid(torch.from_numpy(op_np).to(dev))
    scales = torch.exp(torch.from_numpy(sc_np).to(dev))
    quats = torch.from_numpy(q_np).to(dev)
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    vid_t = torch.from_numpy(vid).to(dev)

    d = np.load(NPZ)
    w2c4 = np.concatenate([d["extrinsics"], np.tile(np.array([[[0, 0, 0, 1]]], np.float32),
                                                    (d["extrinsics"].shape[0], 1, 1))], axis=1)
    Ks0 = d["intrinsics"].astype(np.float32).copy()
    H0, W0 = d["depth"].shape[-2:]
    W, H = (int(v) for v in SIZE.lower().replace("*", "x").split("x"))
    Ks = Ks0.copy()
    Ks[:, 0, :] *= W / W0
    Ks[:, 1, :] *= H / H0
    log(f"📷 {SIZE}  （npz {W0}x{H0}）  SH degree={sh_degree}")

    refs = sorted([p for p in REF_DIR.iterdir()
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg")])

    def render(idx, view, K, Wr, Hr):
        with torch.no_grad():
            img, alpha, _ = rasterization(
                means[idx], quats[idx], scales[idx], opacity[idx], sh[idx],
                torch.from_numpy(w2c4[view]).float().to(dev)[None],
                torch.from_numpy(K).float().to(dev)[None], Wr, Hr, sh_degree=sh_degree,
            )
        return (img[0].clamp(0, 1).mul(255).byte().cpu().numpy(),
                alpha[0].cpu().numpy())

    rows = []
    for v in VIEWS:
        ref = cv2.cvtColor(cv2.imread(str(refs[v])), cv2.COLOR_BGR2RGB)
        ref = cv2.resize(ref, (W, H), interpolation=cv2.INTER_AREA)
        K = Ks[v]
        out = {}
        for tag, idx in (("all", torch.arange(N, device=dev)),
                         ("self", torch.nonzero(vid_t == v).flatten()),
                         ("other", torch.nonzero(vid_t != v).flatten())):
            img, a = render(idx, v, K, W, H)
            out[tag] = (img, a)
            Path(OUT, f"render_{tag}").mkdir(parents=True, exist_ok=True)
            from PIL import Image
            Image.fromarray(img).save(OUT / f"render_{tag}" / f"view_{v:03d}.png")

        # 闸门：all 必须与 94 的既有输出一致（设 REF94 指向任一 94 输出目录）
        gate = None
        ref94 = Path(os.path.expanduser(os.environ["REF94"])) if os.environ.get("REF94") else None
        if ref94 is not None:
            p94 = ref94 / f"view_{v:03d}.png"
            if p94.exists():
                a94 = cv2.cvtColor(cv2.imread(str(p94)), cv2.COLOR_BGR2RGB)
                if a94.shape[:2] == (H, W):
                    gate = psnr(out["all"][0], a94)

        mask = np.squeeze(out["self"][1]) > 0.5
        if mask.shape != (H, W):
            raise SystemExit(f"❌ alpha 形状异常：{out['self'][1].shape}")
        cov = float(mask.mean())
        # 反预乘版"仅本视角"：rgb/alpha，用来看本视角几何本身是否正确（暗是 alpha 没叠满）
        a_self = np.squeeze(out["self"][1])[..., None].astype(np.float64)
        sn = np.clip(out["self"][0].astype(np.float64) / np.maximum(a_self, 0.05), 0, 255)
        out["self_n"] = (sn.astype(np.uint8), out["self"][1])
        Path(OUT, "render_self_n").mkdir(parents=True, exist_ok=True)
        from PIL import Image
        Image.fromarray(out["self_n"][0]).save(OUT / "render_self_n" / f"view_{v:03d}.png")
        psnr_self_n = psnr(out["self_n"][0][mask], ref[mask])
        if mask.sum() < 100:
            log(f"  view {v}: self 覆盖过低（{cov:.3f}），跳过 PSNR 口径")
            continue
        r = dict(view=v, coverage=float(cov),
                 psnr_all=psnr(out["all"][0], ref),
                 psnr_self=psnr(out["self"][0], ref),
                 psnr_other=psnr(out["other"][0], ref),
                 psnr_all_masked=psnr(out["all"][0][mask], ref[mask]),
                 psnr_self_masked=psnr(out["self"][0][mask], ref[mask]),
                 psnr_self_n_masked=psnr_self_n,
                 psnr_other_masked=psnr(out["other"][0][mask], ref[mask]),
                 gate_vs_94=gate)
        rows.append(r)
        log(f"  view {v:2d}  cover(self)={cov:.3f}  "
            f"全图 PSNR all/self/other = {r['psnr_all']:.2f}/{r['psnr_self']:.2f}/{r['psnr_other']:.2f}"
            f"  |  self覆盖集 all/self/self÷α/other = {r['psnr_all_masked']:.2f}/"
            f"{r['psnr_self_masked']:.2f}/{r['psnr_self_n_masked']:.2f}/{r['psnr_other_masked']:.2f}"
            + (f"  |  闸门 vs94 {gate:.1f} dB" if gate is not None else ""))

    if rows:
        agg = {k: float(np.mean([r[k] for r in rows]))
               for k in ("psnr_all", "psnr_self", "psnr_other", "psnr_all_masked",
                         "psnr_self_masked", "psnr_self_n_masked", "psnr_other_masked")}
        log("\n汇总（均值）：")
        log(f"  全图        all {agg['psnr_all']:.2f}  self {agg['psnr_self']:.2f}  "
            f"other {agg['psnr_other']:.2f}")
        log(f"  self覆盖集  all {agg['psnr_all_masked']:.2f}  self {agg['psnr_self_masked']:.2f}  "
            f"self÷α {agg['psnr_self_n_masked']:.2f}  other {agg['psnr_other_masked']:.2f}")
        log(f"  ⇒ 跨视角污染（all 相对 self÷α 的损失）: "
            f"{agg['psnr_self_n_masked'] - agg['psnr_all_masked']:+.2f} dB")
    else:
        agg = {}

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "audit_render_decomp.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(dict(ply=str(PLY), npz=str(NPZ), size=SIZE, sh_degree=sh_degree,
                       n_gauss=N, per_view=rows, mean=agg), f, ensure_ascii=False, indent=1)
    log(f"💾 {OUT / 'audit_render_decomp.json'}")


if __name__ == "__main__":
    main()
