#!/usr/bin/env python3
"""face_metrics.py — 人脸区指标评估（方案一验收，Task #18）。

对比两个渲染目录（基线 30k vs finetune）在同一组相机上的人脸区域质量：
  1. Laplacian 方差（人脸区内，越高越锐）
  2. 梯度能量 |Sobel| 均值（人脸区内，越高越锐）
  3. LPIPS(render, 原图) 人脸区均值（对原图而非增强图，防偏离真实内容；
     注意 HYPIR 本身会改变人脸纹理，此项预期小幅上升，看的是"别崩"）
  4. PSNR / SSIM 人脸区（参考项，非主指标——会奖励模糊平均解，仅记录）

用法:
  python face_metrics.py \
      --render_a  $RESULTS_DIR/render_baseline \
      --render_b  $RESULTS_DIR/render_face \
      --ref_dir   $RESULTS_DIR/03_source/images \
      --masks_dir $RESULTS_DIR/06b_face_masks \
      --out       $RESULTS_DIR/face_metrics.json

命名约定: 三个目录中同一 stem 的文件视为同一相机。mask 用 .alpha.png（软权重，
与 loss mask 同源）；无人脸帧跳过。输出 JSON + 控制台表格。

多人 (SAM3 p-mask): 若 {stem}.alpha.png 不存在, 自动找 {stem}.p*.alpha.png
并取逐像素最大值 (并集) —— 人脸区 = 任意人的脸, 指标语义与单人一致。
"""
import os
import sys
import glob
import json
import argparse

import numpy as np
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

_lpips_model = None


def load_lpips():
    global _lpips_model
    if _lpips_model is None:
        # 优先用 lpips pip 包（权重可能已缓存），回退到 lpipsPyTorch（GS 仓自带）
        import torch
        try:
            import lpips as lpips_pkg
            _net = lpips_pkg.LPIPS(net="alex").cuda()
            _lpips_model = lambda a, b: _net(
                torch.from_numpy(a).permute(2, 0, 1)[None].float().cuda(),
                torch.from_numpy(b).permute(2, 0, 1)[None].float().cuda(),
            ).item()
        except Exception:
            from lpipsPyTorch import lpips
            _lpips_model = lambda a, b: lpips(
                torch.from_numpy(a).permute(2, 0, 1)[None].float().cuda(),
                torch.from_numpy(b).permute(2, 0, 1)[None].float().cuda(),
            ).item()
    return _lpips_model


def list_images(d):
    return {os.path.splitext(f)[0]: f for f in os.listdir(d)
            if os.path.splitext(f)[1].lower() in IMG_EXTS}


def imread_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def sharpness(gray):
    """(laplacian_var, grad_energy) on a grayscale face crop."""
    # cv2.Laplacian/Sobel 要求 src 与 ddepth 匹配；统一转 float64
    gray = np.asarray(gray, dtype=np.float64)
    try:
        import cv2
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return float(lap.var()), float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))
    except ImportError:
        k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
        from numpy.lib.stride_tricks import sliding_window_view
        pad = np.pad(gray, 1)
        lap = (sliding_window_view(pad, (3, 3)) * k).sum(axis=(2, 3))
        sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64) / 8.0
        sy = sx.T
        gx = (sliding_window_view(pad, (3, 3)) * sx).sum(axis=(2, 3))
        gy = (sliding_window_view(pad, (3, 3)) * sy).sum(axis=(2, 3))
        return float(lap.var()), float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))


def face_crop_stats(img, alpha, min_frac=0.001):
    """Stats on the face region (alpha-weighted); returns dict or None."""
    a = alpha / max(alpha.max(), 1e-8)
    if (a > 0.1).mean() < min_frac:      # 人脸区过小 → 视为无人脸
        return None
    ys, xs = np.where(a > 0.1)
    y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = img[y1:y2, x1:x2]
    m = a[y1:y2, x1:x2][..., None]
    gray = (0.299 * crop[..., 0] + 0.587 * crop[..., 1] + 0.114 * crop[..., 2]) * (m[..., 0] ** 0.5)
    lap_var, grad_e = sharpness(gray)
    return {"bbox": [int(x1), int(y1), int(x2), int(y2)],
            "frac": float((a > 0.1).mean())}


def load_face_alpha(masks_dir, stem, w, h):
    """Load soft alpha for a stem: single {stem}.alpha.png, or union of
    multi-person {stem}.p{pid}.alpha.png (pixelwise max). None if absent."""
    single = os.path.join(masks_dir, f"{stem}.alpha.png")
    if os.path.isfile(single):
        parts = [single]
    else:
        parts = sorted(glob.glob(os.path.join(masks_dir, f"{stem}.p*.alpha.png")))
        if not parts:
            return None
    alphas = []
    for p in parts:
        a = Image.open(p).convert("L")
        if a.size != (w, h):
            a = a.resize((w, h), Image.LANCZOS)
        alphas.append(np.asarray(a, dtype=np.float32))
    if len(alphas) == 1:
        return alphas[0]
    return np.maximum.reduce(alphas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render_a", required=True, help="baseline renders dir")
    ap.add_argument("--render_b", required=True, help="finetuned renders dir")
    ap.add_argument("--ref_dir", required=True, help="original training images dir")
    ap.add_argument("--masks_dir", required=True, help="face_masks.py output dir")
    ap.add_argument("--out", default="", help="output JSON path (optional)")
    args = ap.parse_args()

    gs_dir = os.environ.get("GS_DIR") or os.path.expanduser("~/repos/gaussian-splatting")
    if gs_dir not in sys.path:
        sys.path.insert(0, gs_dir)

    A, B, R = list_images(args.render_a), list_images(args.render_b), list_images(args.ref_dir)
    common = sorted(set(A) & set(B) & set(R))
    if not common:
        sys.exit("❌ no common stems across the three dirs")

    lpips_fn = load_lpips()
    rows = []
    for stem in common:
        img_r = imread_rgb(os.path.join(args.ref_dir, R[stem]))
        h, w = img_r.shape[:2]
        alpha = load_face_alpha(args.masks_dir, stem, w, h)
        if alpha is None:
            continue
        ia = imread_rgb(os.path.join(args.render_a, A[stem]))
        ib = imread_rgb(os.path.join(args.render_b, B[stem]))
        for x in (ia, ib):
            if x.shape[:2] != (h, w):
                sys.exit(f"❌ {stem}: render size {x.shape[:2]} != ref {(h, w)}")

        fa = face_crop_stats(ia, alpha)
        if fa is None:
            continue
        ys, xs = np.where(alpha > 0.1)
        y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        m = alpha[y1:y2, x1:x2][..., None] / 255.0

        def crop(x):
            c = x[y1:y2, x1:x2]
            return c

        ca, cb, cr = crop(ia), crop(ib), crop(img_r)

        def region_lpips(a, b):
            return lpips_fn(a * m, b * m)

        def psnr_ssim(a, b):
            mse = float(np.mean(((a - b) * m) ** 2))
            p = -10 * np.log10(max(mse, 1e-10))
            # 简易全局 SSIM（人脸 crop 上）
            from math import prod
            mu_a, mu_b = (a * m).sum() / m.sum(), (b * m).sum() / m.sum()
            va = ((a - mu_a) ** 2 * m).sum() / m.sum()
            vb = ((b - mu_b) ** 2 * m).sum() / m.sum()
            cov = ((a - mu_a) * (b - mu_b) * m).sum() / m.sum()
            c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
            s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
            return p, float(s)

        la_lap, la_grad = None, None  # （历史占位，实际锐度用 sharp() 重算）
        def sharp(c):
            g = 0.299 * c[..., 0] + 0.587 * c[..., 1] + 0.114 * c[..., 2]
            g = g * (m[..., 0] ** 0.5)
            return sharpness(g)

        sa_lap, sa_grad = sharp(ca)
        sb_lap, sb_grad = sharp(cb)

        rows.append({
            "stem": stem, "face_frac": fa["frac"],
            "laplacian_var_A": sa_lap, "laplacian_var_B": sb_lap,
            "grad_energy_A": sa_grad, "grad_energy_B": sb_grad,
            "lpips_A_ref": region_lpips(ca, cr), "lpips_B_ref": region_lpips(cb, cr),
            "psnr_A": psnr_ssim(ca, cr)[0], "psnr_B": psnr_ssim(cb, cr)[0],
            "ssim_A": psnr_ssim(ca, cr)[1], "ssim_B": psnr_ssim(cb, cr)[1],
        })
        print(f"  {stem}: lapVar {sa_lap:.1f}→{sb_lap:.1f}  gradE {sa_grad:.2f}→{sb_grad:.2f} "
              f" lpips {rows[-1]['lpips_A_ref']:.4f}→{rows[-1]['lpips_B_ref']:.4f}")

    if not rows:
        sys.exit("❌ no frames with face masks found")

    def avg(k):
        vals = [r[k] for r in rows]
        return float(np.mean(vals))

    summary = {
        "n_frames": len(rows),
        "laplacian_var_A": avg("laplacian_var_A"), "laplacian_var_B": avg("laplacian_var_B"),
        "grad_energy_A": avg("grad_energy_A"), "grad_energy_B": avg("grad_energy_B"),
        "lpips_A_ref": avg("lpips_A_ref"), "lpips_B_ref": avg("lpips_B_ref"),
        "psnr_A": avg("psnr_A"), "psnr_B": avg("psnr_B"),
        "ssim_A": avg("ssim_A"), "ssim_B": avg("ssim_B"),
    }
    print("\n════════ 人脸区指标汇总（A=基线30k, B=finetune）════════")
    print(f"  frames            : {summary['n_frames']}")
    print(f"  Laplacian 方差    : {summary['laplacian_var_A']:.1f} → {summary['laplacian_var_B']:.1f}  (越高越锐)")
    print(f"  梯度能量          : {summary['grad_energy_A']:.3f} → {summary['grad_energy_B']:.3f}  (越高越锐)")
    print(f"  LPIPS vs 原图     : {summary['lpips_A_ref']:.4f} → {summary['lpips_B_ref']:.4f}  (防偏离, 越低越近)")
    print(f"  PSNR(参考)        : {summary['psnr_A']:.2f} → {summary['psnr_B']:.2f}")
    print(f"  SSIM(参考)        : {summary['ssim_A']:.4f} → {summary['ssim_B']:.4f}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "per_frame": rows}, f, indent=2, ensure_ascii=False)
        print(f"\n  saved → {args.out}")


if __name__ == "__main__":
    main()
