#!/usr/bin/env python3
"""渲染图 vs 输入帧的定量体检（P0 各项共用，口径固定便于横向比）。

三个指标，各有明确用途：
  PSNR / SSIM   与"该视角的输入帧"比。注意这是**自一致性**指标（该帧在前向输入里），
                不是 NVS holdout；用来做 A/B 的方向判断，不能当绝对质量分。
  sharpness     渲染图 Laplacian 方差（只算渲染，ref 的锐度一并给，便于看"离原图多远"）
  coverage      非白像素占比（近似 alpha 覆盖），看视角 16 人物消失这类问题

用法（WSL，querysplat env 已有 cv2/skimage）：
  python compare_4methods/99_eval_render_metrics.py \
      --render ~/output/da3_gs_results/佳佳抱枕_shfull/render_full \
      --ref    ~/output/da3_results/_captures/佳佳抱枕/images \
      --views  0,8,16 --tag full_sh
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def load_ref(ref_dir, idx):
    files = sorted([p for p in Path(ref_dir).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    return cv2.cvtColor(cv2.imread(str(files[idx])), cv2.COLOR_BGR2RGB)


def sharpness(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def coverage(rgb):
    # 背景是白（渲染无 alpha，取 250 以上为"空"）
    return float((rgb.min(axis=2) < 248).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", required=True, help="渲染图目录（view_%03d.png）")
    ap.add_argument("--ref", required=True, help="输入帧目录（按文件名排序）")
    ap.add_argument("--views", default="0,8,16")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default="", help="json 输出路径（默认写到 render 目录下）")
    a = ap.parse_args()

    views = [int(v) for v in a.views.split(",")]
    rows = []
    for v in views:
        rp = Path(a.render) / f"view_{v:03d}.png"
        if not rp.exists():
            print(f"⚠️ 缺渲染: {rp}")
            continue
        rgb = cv2.cvtColor(cv2.imread(str(rp)), cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        ref = load_ref(a.ref, v)
        ref = cv2.resize(ref, (w, h), interpolation=cv2.INTER_AREA)
        mse = float(np.mean((rgb.astype(np.float64) - ref.astype(np.float64)) ** 2))
        psnr = 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
        s = float(ssim(ref, rgb, channel_axis=2, data_range=255))
        rows.append(dict(view=v, psnr=round(psnr, 2), ssim=round(s, 4),
                         sharp=round(sharpness(rgb), 1), sharp_ref=round(sharpness(ref), 1),
                         coverage=round(coverage(rgb), 4), size=[w, h]))
        print(f"  view {v:2d}  PSNR {psnr:6.2f}  SSIM {s:.4f}  "
              f"sharp {sharpness(rgb):7.1f} (ref {sharpness(ref):7.1f})  cover {coverage(rgb):.3f}")

    if rows:
        avg = {k: round(float(np.mean([r[k] for r in rows])), 4) for k in ("psnr", "ssim", "sharp", "coverage")}
        print(f"  MEAN     PSNR {avg['psnr']:6.2f}  SSIM {avg['ssim']:.4f}  sharp {avg['sharp']:7.1f}  cover {avg['coverage']:.3f}")
    else:
        avg = {}

    out = Path(a.out) if a.out else Path(a.render) / f"metrics_{a.tag}.json"
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"tag": a.tag, "render": str(a.render), "ref": a.ref,
                   "views": views, "per_view": rows, "mean": avg}, f, indent=4, ensure_ascii=False)
    print(f"✅ {out}")


if __name__ == "__main__":
    main()
