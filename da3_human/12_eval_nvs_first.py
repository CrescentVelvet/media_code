#!/usr/bin/env python3
"""12_eval_nvs_first.py — 干净的 NVS 口径：ref=first 保证两次推理世界系一致，SH 无需旋转。

与 10_eval_nvs_32to125.py 的差别：
  10 走 COLMAP 做公证世界系，需要 Sim(3) 把 DA3 系转过去；若参考视角不同，R 可达 83.6°，
     而 SH 与方向绑定，不转 SH 会白丢约 1.2 dB。
  12 走官方 REF_STRATEGY=first：32 视角与 125 视角两次推理都以「第 0 张图」定义世界系，
     实测 s=0.9985 / R 与 I 夹角 0.10°，于是只需一个均匀缩放（不改方向），SH 原样可用。

Env:
  PLY       32 视角那次前向导出的高斯 PLY（米制）
  NPZ32     32 视角的 cameras.npz（提供 view_indices）
  NPZ125    125 视角的 cameras.npz（提供渲染用的全部相机）
  IMAGES_DIR GT
  W,H       渲染分辨率
"""
import math
import os
import re
from pathlib import Path

import cv2
import numpy as np
import torch


def log(m):
    print(m, flush=True)


def load_ply(path):
    with open(path, "rb") as f:
        raw = f.read(1 << 20)
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
        r = r.reshape(n, 3, k).transpose(0, 2, 1)
        sh = np.concatenate([dc[:, None, :], r], axis=1)
    else:
        sh = dc[:, None, :]
    return means, sh, op, sc, q


def w2c44(ext):
    return np.concatenate(
        [ext.astype(np.float32),
         np.tile(np.array([[[0, 0, 0, 1]]], np.float32), (len(ext), 1, 1))], axis=1)


def main():
    from gsplat import rasterization

    ply = os.environ["PLY"]
    npz32 = np.load(os.environ["NPZ32"])
    npz125 = np.load(os.environ["NPZ125"])
    images_dir = os.path.expanduser(os.environ["IMAGES_DIR"])
    W, H = int(os.environ.get("W", "1440")), int(os.environ.get("H", "1920"))
    dev = "cuda"

    means_np, sh_np, op_np, sc_np, q_np = load_ply(ply)
    sh_deg = int(round(sh_np.shape[1] ** 0.5)) - 1
    log(f"📦 PLY {len(means_np):,} 高斯 | SH degree {sh_deg}")

    ext32, ext125 = npz32["extrinsics"], npz125["extrinsics"]
    idx = npz32["view_indices"].astype(int)
    # ⚠️ PLY 与 extrinsics 各自都已被自己的 scale_factor 折成米制，两者应直接用；
    #    额外乘 sf32/sf125 等于把两次推理的估计误差当成真实尺度差，实测掉 2 dB。
    s = float(os.environ.get("EXTRA_SCALE", "1.0"))
    log(f"🔗 世界系一致性：s={s:.4f}（R≈I，SH 不旋转） | 32→125 视角 {list(idx[:8])}...")

    means = means_np.astype(np.float64) * s
    scales = np.exp(sc_np.astype(np.float64)) * s

    means = torch.from_numpy(means.astype(np.float32)).to(dev)
    scales = torch.from_numpy(scales.astype(np.float32)).to(dev)
    quats = torch.from_numpy(q_np.astype(np.float32)).to(dev)
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    opac = torch.sigmoid(torch.from_numpy(op_np).to(dev))
    sh = torch.from_numpy(sh_np).to(dev)

    all_imgs = []
    for e in ("*.jpg", "*.jpeg", "*.png"):
        all_imgs.extend(sorted(Path(images_dir).glob(e)))
    all_imgs = sorted(all_imgs)
    assert len(all_imgs) == len(ext125), f"图数 {len(all_imgs)} != 相机数 {len(ext125)}"

    K0 = npz125["intrinsics"].astype(np.float32).copy()
    H0, W0 = npz125["depth"].shape[-2:]
    K = K0.copy()
    if (W, H) != (W0, H0):
        K[:, 0, :] *= W / W0
        K[:, 1, :] *= H / H0
    log(f"📷 内参 {W0}x{H0} → {W}x{H} | fx={K[0,0,0]:.1f}")

    seen_set = set(idx.tolist())
    ps, ps_seen, ps_unseen = [], [], []
    for vi in range(len(ext125)):
        p = all_imgs[vi]
        g = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        gt = torch.from_numpy(g).permute(2, 0, 1).to(dev)
        if (gt.shape[1], gt.shape[2]) != (H, W):
            gt = torch.nn.functional.interpolate(gt[None], size=(H, W), mode="area")[0]
        with torch.no_grad():
            img, _, _ = rasterization(
                means, quats, scales, opac, sh,
                torch.from_numpy(w2c44(ext125[vi:vi + 1])).to(dev),
                torch.from_numpy(K[vi:vi + 1]).to(dev),
                W, H, sh_degree=sh_deg)
        r_ = img[0].clamp(0, 1).permute(2, 0, 1)
        mse = ((r_ - gt) ** 2).mean().item()
        v = -10 * math.log10(max(mse, 1e-12))
        ps.append(v)
        (ps_seen if vi in seen_set else ps_unseen).append(v)

    log(f"\n📊 NVS 口径（ref=first，{len(ps)} 视角，{W}x{H}）")
    log(f"  全部       PSNR {np.mean(ps):.2f}")
    log(f"  seen({len(ps_seen):3d})   PSNR {np.mean(ps_seen):.2f}")
    log(f"  unseen({len(ps_unseen):3d}) PSNR {np.mean(ps_unseen):.2f}")
    log(f"  per-view p10 {np.percentile(ps,10):.1f} p50 {np.percentile(ps,50):.1f} "
        f"p90 {np.percentile(ps,90):.1f}")


if __name__ == "__main__":
    main()
