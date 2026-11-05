# 03b_compare_renders.py — 03/04 基线同视角渲染三联对比（gt | COLMAP init | DA3 init）
# 目的：阶段 B 定性对照——同事观察的 floater/重影现象在两个基线上的差异
# 用法: python 03b_compare_renders.py（路径经环境变量注入，见 03b_compare_renders.sh）
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

VIEWS = [0, 30, 60, 90]  # 覆盖轨迹不同位置（开弧采集）


def label(img: Image.Image, text: str) -> Image.Image:
    img = img.copy()
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 190, 44], fill=(0, 0, 0))
    d.text((10, 10), text, fill=(255, 255, 0))
    return img


def main():
    results = Path(os.environ["RESULTS_DIR"])
    r03 = results / "baseline_3dgs_colmap/train/ours_30000"
    r04 = results / "baseline_3dgs_da3init/train/ours_30000"
    out = results / "baseline_compare"
    out.mkdir(parents=True, exist_ok=True)

    n_gt = len(sorted((r03 / "gt").glob("*.png")))
    picks = [v for v in VIEWS if v < n_gt]
    for v in picks:
        gt = Image.open(r03 / "gt" / f"{v:05d}.png")
        a = Image.open(r03 / "renders" / f"{v:05d}.png")
        b = Image.open(r04 / "renders" / f"{v:05d}.png")
        w, h = gt.size
        row = Image.new("RGB", (w * 3 + 8, h))
        row.paste(label(gt, "GT"), (0, 0))
        row.paste(label(a, "03 COLMAP init"), (w + 4, 0))
        row.paste(label(b, "04 DA3 init"), (w * 2 + 8, 0))
        row.save(out / f"compare_{v:05d}.png")
        print(f"saved {out}/compare_{v:05d}.png ({w}x{h}x3)")


if __name__ == "__main__":
    main()
