#!/usr/bin/env python3
"""eval_fg_psnr.py — mask 加权 A/B 的前景区域专项评测

对比同一场景两个模型（baseline vs mask 加权）在 test 视角上的：
  - 全图 PSNR
  - 前景（SAM3 person+doll mask 内）PSNR   ← 本实验真正关心的指标
并导出「GT | baseline | mask模型」前景包围盒裁剪对比图，供目视检查手部。

用法（WSL，deformable_human env）：
  python deformable_human/eval_fg_psnr.py \
      --scene_dir /mnt/d/output/deformable_human_results/hand_motion/colmap_scene \
      --model_a ~/output/deformable_human_results/hand_motion/model \
      --model_b ~/output/deformable_human_results/hand_motion/model_mask \
      --out_dir  /mnt/d/output/deformable_human_results/hand_motion/mask_ab_compare

依赖：numpy + PIL（env 里随 torchvision 自带，无新增依赖）。
test 视角选取口径与训练一致：排序后每 8 帧取 1（llffhold=8）。
"""
import argparse
import math
import os

import numpy as np
from PIL import Image


def load_rgb(path, size=None):
    img = Image.open(path).convert("RGB")
    if size is not None and img.size != size:
        img = img.resize(size, Image.LANCZOS)
    return np.asarray(img, dtype=np.float64) / 255.0


def psnr(a, b, mask=None):
    diff = (a - b) ** 2  # H,W,3
    if mask is not None:
        m = mask[..., None]
        mse = (diff * m).sum() / max(m.sum() * diff.shape[2], 1.0)
    else:
        mse = diff.mean()
    if mse <= 1e-12:
        return float("inf")
    return -10.0 * math.log10(mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True, help="colmap_scene 目录（含 images/ 与 masks/）")
    ap.add_argument("--model_a", required=True, help="baseline 模型目录（含 test/ours_10000/renders）")
    ap.add_argument("--model_b", required=True, help="mask 加权模型目录")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--iteration", default="ours_10000")
    ap.add_argument("--llffhold", type=int, default=8)
    args = ap.parse_args()

    images_dir = os.path.join(args.scene_dir, "images")
    masks_dir = os.path.join(args.scene_dir, "masks")
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(images_dir)
                   if f.lower().endswith((".jpg", ".png")))
    test_stems = stems[:: args.llffhold]

    ra = os.path.join(args.model_a, "test", args.iteration, "renders")
    rb = os.path.join(args.model_b, "test", args.iteration, "renders")

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for i, stem in enumerate(test_stems):
        rpath_a = os.path.join(ra, f"{i:05d}.png")
        rpath_b = os.path.join(rb, f"{i:05d}.png")
        if not (os.path.exists(rpath_a) and os.path.exists(rpath_b)):
            print(f"[skip] {stem}: render 缺失")
            continue
        A = Image.open(rpath_a)
        size = A.size  # 渲染分辨率（-r 2 后的尺寸）
        gt = load_rgb(os.path.join(images_dir, stem + ".jpg"), size)
        mpath = os.path.join(masks_dir, stem + ".png")
        mask = None
        if os.path.exists(mpath):
            m = Image.open(mpath).convert("L").resize(size, Image.NEAREST)
            mask = np.asarray(m, dtype=np.float64) > 127
        a = load_rgb(rpath_a)
        b = load_rgb(rpath_b)
        row = dict(
            stem=stem,
            full_a=psnr(a, gt), full_b=psnr(b, gt),
            fg_a=psnr(a, gt, mask) if mask is not None else float("nan"),
            fg_b=psnr(b, gt, mask) if mask is not None else float("nan"),
            fg_ratio=float(mask.mean()) if mask is not None else float("nan"),
        )
        rows.append(row)
        print(f"{stem}: full {row['full_a']:.2f}->{row['full_b']:.2f} | "
              f"fg {row['fg_a']:.2f}->{row['fg_b']:.2f} (fg占比 {row['fg_ratio']*100:.1f}%)")

        # 对比图：前景包围盒裁剪（外扩 15%），GT|A|B 横排
        if mask is not None and mask.any():
            ys, xs = np.where(mask)
            h, w = mask.shape
            y0, y1 = ys.min(), ys.max()
            x0, x1 = xs.min(), xs.max()
            pad_y = int((y1 - y0) * 0.15)
            pad_x = int((x1 - x0) * 0.15)
            y0 = max(0, y0 - pad_y); y1 = min(h, y1 + pad_y)
            x0 = max(0, x0 - pad_x); x1 = min(w, x1 + pad_x)
            crop = lambda im: im[y0:y1, x0:x1]
            strip = np.concatenate([crop(gt), crop(a), crop(b)], axis=1)
            Image.fromarray((strip * 255).astype(np.uint8)).save(
                os.path.join(args.out_dir, f"cmp_{stem}.jpg"), quality=92)

    valid = [r for r in rows if not math.isnan(r["fg_a"])]
    if valid:
        mf_a = np.mean([r["full_a"] for r in valid]); mf_b = np.mean([r["full_b"] for r in valid])
        mg_a = np.mean([r["fg_a"] for r in valid]);  mg_b = np.mean([r["fg_b"] for r in valid])
        print("\n=== 汇总（%d 个 test 视角）===" % len(valid))
        print(f"全图 PSNR : {mf_a:.3f} -> {mf_b:.3f}  (Δ {mf_b - mf_a:+.3f})")
        print(f"前景 PSNR: {mg_a:.3f} -> {mg_b:.3f}  (Δ {mg_b - mg_a:+.3f})")
    print(f"\n对比图输出: {args.out_dir}/cmp_*.jpg")


if __name__ == "__main__":
    main()
