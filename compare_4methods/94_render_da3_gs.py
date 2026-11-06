#!/usr/bin/env python3
"""渲染 DA3 原生 GS 头导出的 gaussians.ply（官方 gs_ply 格式）。

用途：验证「DA3 一次前向直出高斯」的产物是否真的能看，供四方法对比图使用。
不依赖 DA3 仓（da3_human env 没装 gsplat），改用 querysplat env 的 gsplat 光栅化。

官方导出约定（depth_anything_3/utils/gsply_helpers.py save_gaussian_ply）：
  - opacity 存 inverse_sigmoid → 渲染前要 sigmoid
  - scale   存 log               → 渲染前要 exp（标准 3DGS ply 约定）
  - 只保留 SH DC（save_sh_dc_only=True）
  - 坐标 = DA3 预测相机空间（world），相机来自 mini_npz 的 extrinsics(w2c, opencv)

用法（WSL，querysplat env）：
  python compare_4methods/94_render_da3_gs.py <ply> <mini_npz> <out_dir> [frames]
Env:
  SCALE_MODE  exp(默认) | raw
  OPACITY_MIN sigmoid 后的剪枝下限（默认 0.0 不剪）
"""
import os
import re
import sys

import numpy as np
import torch
from gsplat import rasterization

PROPS_ORDER = None


def load_ply(path):
    with open(path, "rb") as f:
        raw = f.read(16384)
        end = raw.index(b"end_header") + len(b"end_header") + 1
        head = raw[:end].decode("ascii", "replace")
        n = int(re.search(r"element vertex (\d+)", head).group(1))
        props = [l.split()[2] for l in head.splitlines() if l.startswith("property")]
        f.seek(end)
        arr = np.fromfile(f, dtype=np.float32).reshape(n, len(props))
    nm = {k: i for i, k in enumerate(props)}
    means = arr[:, [nm["x"], nm["y"], nm["z"]]]
    sh = arr[:, [nm["f_dc_0"], nm["f_dc_1"], nm["f_dc_2"]]]           # (N,3) DC
    opacity = arr[:, nm["opacity"]]
    scales = arr[:, [nm["scale_0"], nm["scale_1"], nm["scale_2"]]]
    quats = arr[:, [nm["rot_0"], nm["rot_1"], nm["rot_2"], nm["rot_3"]]]
    return means, sh, opacity, scales, quats


def main():
    ply, npz, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    frames = [int(x) for x in (sys.argv[4].split(",") if len(sys.argv) > 4 else ["0", "8", "16"])]
    scale_mode = os.environ.get("SCALE_MODE", "exp")
    opacity_min = float(os.environ.get("OPACITY_MIN", "0.0"))

    dev = "cuda"
    means_np, sh_np, op_np, sc_np, q_np = load_ply(ply)
    print(f"📦 高斯: {len(means_np)} 个")

    means = torch.from_numpy(means_np).to(dev)
    sh = torch.from_numpy(sh_np).to(dev).reshape(-1, 1, 3)             # (N,1,3) sh_degree=0
    opacity = torch.sigmoid(torch.from_numpy(op_np).to(dev))           # logit → [0,1]
    scales = torch.from_numpy(sc_np).to(dev)
    if scale_mode == "exp":
        scales = torch.exp(scales)
    quats = torch.from_numpy(q_np).to(dev)
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-9)

    if opacity_min > 0:
        keep = opacity >= opacity_min
        print(f"✂️ 剪枝 opacity<{opacity_min}: {int(keep.sum())} / {len(opacity)}")
        means, sh, opacity, scales, quats = means[keep], sh[keep], opacity[keep], scales[keep], quats[keep]

    print(f"📐 scale(exp 后) median: {scales.median(0).values.cpu().numpy().round(5)}")

    d = np.load(npz)
    w2c4 = np.concatenate([d["extrinsics"], np.tile(np.array([[[0, 0, 0, 1]]], np.float32),
                                                   (d["extrinsics"].shape[0], 1, 1))], axis=1)
    Ks = d["intrinsics"]
    H, W = d["depth"].shape[-2:]
    print(f"📷 相机 {w2c4.shape[0]} 个, 分辨率 {W}x{H}")

    os.makedirs(out_dir, exist_ok=True)
    for i in frames:
        if i >= w2c4.shape[0]:
            continue
        vm = torch.from_numpy(w2c4[i]).float().to(dev)[None]            # (1,4,4)
        K = torch.from_numpy(Ks[i]).float().to(dev)[None]               # (1,3,3)
        with torch.no_grad():
            img, alpha, _ = rasterization(
                means, quats, scales, opacity, sh, vm, K, W, H, sh_degree=0,
            )
        img = img[0].clamp(0, 1).mul(255).byte().cpu().numpy()
        from PIL import Image
        Image.fromarray(img).save(os.path.join(out_dir, f"view_{i:03d}.png"))
        print(f"🖼️ view_{i:03d}.png  alpha mean={float(alpha.mean()):.3f}")


if __name__ == "__main__":
    main()
