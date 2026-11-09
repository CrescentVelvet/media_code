#!/usr/bin/env python3
"""07_eval_da3_gs.py — 评估官方 DA3 GSDPT 高斯头在本场景的渲染质量。

两个必做前置（否则数字无效，见 compare_4methods/ERRATA.md）：
  1) scale_factor 扫掠（E2/E-01）：DA3 的 gaussians 不随 depth/extrinsics 乘
     scale_factor，必须扫出峰位再定量（studio_human 实测 1.8666）。
  2) 双分辨率口径（E6）：官方口径 378x504 vs 交付口径 1440x1920，两者不可混用。

两种用法：
  sweep  扫 scale_factor（378x504，7 视角，~2 min）
  eval   正式评估 125 视角 × 两个口径（需 SF；PLY 已米制时传 SF=1.0）

Env:
  PLY   高斯 PLY 路径（默认 da3_inference_gs/.../gs_ply/0000.ply = DC-only 官方导出）
  NPZ   相机 npz（默认 exports/mini_npz/results.npz；97 导出的用 cameras.npz）
  SF    米制折算倍数；97 已折算过 → SF=1.0
"""
import math
import os
import re
import sys

import cv2
import numpy as np
import torch
from gsplat import rasterization

GS_DIR = "/home/velvet/output/da3_human_results/da3_inference_gs/studio_human"
PLY = os.environ.get("PLY", f"{GS_DIR}/gs_ply/0000.ply")
NPZ = os.environ.get("NPZ", f"{GS_DIR}/exports/mini_npz/results.npz")
IMGS = "/home/velvet/output/vggt_human_results/source/images"
DEV = "cuda"
SWEEP_VIEWS = [0, 20, 40, 60, 80, 100, 120]


def load_ply(path):
    with open(path, "rb") as f:
        raw = f.read(65536)
        end = raw.index(b"end_header") + len(b"end_header") + 1
        head = raw[:end].decode("ascii", "replace")
        n = int(re.search(r"element vertex (\d+)", head).group(1))
        props = [ln.split()[2] for ln in head.splitlines() if ln.startswith("property")]
        f.seek(end)
        arr = np.fromfile(f, dtype=np.float32).reshape(n, len(props))
    nm = {k: i for i, k in enumerate(props)}
    means = arr[:, [nm["x"], nm["y"], nm["z"]]]
    op = arr[:, nm["opacity"]]
    sc = arr[:, [nm["scale_0"], nm["scale_1"], nm["scale_2"]]]
    q = arr[:, [nm["rot_0"], nm["rot_1"], nm["rot_2"], nm["rot_3"]]]
    dc = arr[:, [nm["f_dc_0"], nm["f_dc_1"], nm["f_dc_2"]]]
    rest = sorted((int(k.split("_")[-1]), i) for k, i in nm.items() if k.startswith("f_rest_"))
    if rest:
        r = arr[:, [i for _, i in rest]]
        k = r.shape[1] // 3
        r = r.reshape(n, 3, k).transpose(0, 2, 1)  # E8: channel-major → (N,K,3)
        sh = np.concatenate([dc[:, None, :], r], axis=1)
    else:
        sh = dc[:, None, :]
    has_normal = "nx" in nm
    normals = arr[:, [nm["nx"], nm["ny"], nm["nz"]]] if has_normal else None
    return means, sh, op, sc, q, normals, props


def _gt_paths():
    """与 08 同口径的图像列表（三种扩展名分别排序后合并再排序）。"""
    import pathlib

    fs = []
    for e in ("*.jpg", "*.jpeg", "*.png"):
        fs.extend(sorted(pathlib.Path(IMGS).glob(e)))
    return sorted(fs)


GT_PATHS = _gt_paths()
VI_MAP = None  # NPZ 里第 i 个视角 → sorted(images) 的索引；抽帧时必须用


def load_gt(vi, W, H):
    gi = int(VI_MAP[vi]) if VI_MAP is not None else vi
    p = GT_PATHS[gi]
    im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(im).permute(2, 0, 1)
    if (t.shape[1], t.shape[2]) != (H, W):
        t = torch.nn.functional.interpolate(t[None], size=(H, W), mode="area")[0]
    return t.to(DEV)


def main():
    global VI_MAP
    mode = sys.argv[1] if len(sys.argv) > 1 else "sweep"

    means_np, sh_np, op_np, sc_np, q_np, normals, props = load_ply(PLY)
    print(f"📦 PLY: {len(means_np):,} 高斯 | 字段 {len(props)} | 法向 {'有' if normals is not None else '无'}")
    print(f"   SH 阶数: {sh_np.shape[1]} → degree {int(round(sh_np.shape[1] ** 0.5)) - 1}")

    d = np.load(NPZ)
    ext = d["extrinsics"]
    nv = len(ext)
    w2c = np.concatenate([ext, np.tile(np.array([[[0.0, 0.0, 0.0, 1.0]]], np.float32), (nv, 1, 1))], axis=1)
    K0 = d["intrinsics"].astype(np.float32).copy()
    H0, W0 = d["depth"].shape[-2:]
    if "view_indices" in d:
        VI_MAP = np.asarray(d["view_indices"])
        assert len(VI_MAP) == nv, f"view_indices {len(VI_MAP)} != 视角数 {nv}"
        print(f"🗺️ 视角映射来自 NPZ：{list(VI_MAP[:8])}…（共 {nv} 个）")
    elif nv != len(GT_PATHS):
        sys.exit(f"❌ NPZ 只有 {nv} 个视角但图像目录有 {len(GT_PATHS)} 张，"
                 f"且没有 view_indices → 无法对齐 GT（08 已修，重跑 08 或补 view_indices）")
    print(f"📷 {nv} 视角 | npz 处理分辨率 {W0}x{H0}")

    sh_deg = int(round(sh_np.shape[1] ** 0.5)) - 1
    sh = torch.from_numpy(sh_np).to(DEV)
    opac = torch.sigmoid(torch.from_numpy(op_np).to(DEV))
    quats = torch.from_numpy(q_np).to(DEV)
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    scales0 = torch.exp(torch.from_numpy(sc_np).to(DEV))
    means0 = torch.from_numpy(means_np).to(DEV)
    print(f"  opacity 中位 {opac.median():.4f} | scale(exp) 中位 {scales0.median(0).values.cpu().numpy().round(5)}")

    def render(means, scales, W, H, K, vi, deg=sh_deg):
        vm = torch.from_numpy(w2c[vi]).float().to(DEV)[None]
        kk = torch.from_numpy(K[vi]).float().to(DEV)[None]
        with torch.no_grad():
            img, alpha, _ = rasterization(
                means, quats, scales, opac, sh, vm, kk, W, H, sh_degree=deg)
        return img[0].clamp(0, 1), alpha[0, ..., 0]

    # ---------------- 阶段 1：scale_factor 扫掠（378x504，官方口径） ----------------
    if mode == "sweep":
        Ksmall = K0.copy()
        print("\n🔍 scale_factor 扫掠（378x504，7 视角）")
        print(f"{'sf':>5} {'PSNR':>7}  per-view")
        best = (None, -1e9)
        for sf in [1.0, 1.2, 1.5, 1.8, 2.0, 2.2, 2.5]:
            m = (means0 * sf).contiguous()
            s = (scales0 * sf).contiguous()
            ps = []
            for vi in SWEEP_VIEWS:
                img, _ = render(m, s, W0, H0, Ksmall, vi)
                gt = load_gt(vi, W0, H0)
                mse = ((img.permute(2, 0, 1) - gt) ** 2).mean().item()
                ps.append(-10 * math.log10(max(mse, 1e-12)))
            print(f"{sf:5.2f} {np.mean(ps):7.2f}  {np.round(ps, 1)}")
            if np.mean(ps) > best[1]:
                best = (sf, np.mean(ps))
        print(f"\n✅ 峰位 sf = {best[0]:.2f} → PSNR {best[1]:.2f} dB")
        return

    # ---------------- 阶段 2：正式评估（两个口径） ----------------
    sf = float(os.environ.get("SF", "1.0"))
    m = (means0 * sf).contiguous()
    s = (scales0 * sf).contiguous()
    print(f"\n📊 正式评估（sf={sf}）")

    for tag, W, H, Ksrc in [("官方口径 504p", W0, H0, K0), ("交付口径 1440x1920", 1440, 1920, None)]:
        if Ksrc is None:
            K = K0.copy()
            K[:, 0, :] *= 1440 / W0
            K[:, 1, :] *= 1920 / H0
        else:
            K = Ksrc
        ps, ssims = [], []
        for vi in range(nv):
            img, alpha = render(m, s, W, H, K, vi)
            gt = load_gt(vi, W, H)
            r = img.permute(2, 0, 1)
            mse = ((r - gt) ** 2).mean().item()
            ps.append(-10 * math.log10(max(mse, 1e-12)))
            ssims.append(ssim_torch(r, gt).item())
            if vi in (0, 30, 60, 90):
                arr = (r.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                out = f"/tmp/gs_{tag.split()[0]}_v{vi:03d}.png"
                cv2.imwrite(out, arr[:, :, ::-1])
        print(f"  {tag}: PSNR {np.mean(ps):.2f} | SSIM {np.mean(ssims):.4f} | "
              f"per-view p10 {np.percentile(ps,10):.1f} p90 {np.percentile(ps,90):.1f} "
              f"min {min(ps):.1f} max {max(ps):.1f}")


def ssim_torch(a, b):
    import torch.nn.functional as F

    C = a.shape[0]
    size, sigma = 11, 1.5
    g = torch.exp(-((torch.arange(size, dtype=torch.float32, device=a.device) - size // 2) ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w = (g.view(1, 1, size) * g.view(1, size, 1)).expand(C, 1, size, size).contiguous()
    a, b = a.unsqueeze(0), b.unsqueeze(0)
    mu1 = F.conv2d(a, w, padding=5, groups=C)
    mu2 = F.conv2d(b, w, padding=5, groups=C)
    s1 = F.conv2d(a * a, w, padding=5, groups=C) - mu1 * mu1
    s2 = F.conv2d(b * b, w, padding=5, groups=C) - mu2 * mu2
    s12 = F.conv2d(a * b, w, padding=5, groups=C) - mu1 * mu2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) /
            ((mu1 * mu1 + mu2 * mu2 + c1) * (s1 + s2 + c2))).mean()


if __name__ == "__main__":
    main()
