#!/usr/bin/env python3
"""05_make_compare.py — 生成两个模型渲染结果的手部对比图（r2 加速验收用）。

产物（OUT 目录）：
  hand_strip.png      手部时间条带：行 = [r2↑, base, GT]，列 = STRIP_FRAMES
  full_f<idx>.png     整帧三联：r2↑ | base | GT（缩到等高）

r2 渲染是半分辨率，统一 ×2 LANCZOS 上采样到 GT 尺寸再裁切，保证像素对齐。

Env vars:
  BASE_DIR   全分辨率模型 renders 目录（0000.png 命名）
  R2_DIR     半分辨率模型 renders 目录
  GT_DIR     GT images 目录（00000.png 命名）
  OUT        输出目录
  CROP       裁切框 "x0,y0,x1,y1"（GT 全分辨率像素，默认挥手手区域）
  STRIP_FRAMES  逗号分隔帧号（默认 8,10,12,14,16,18,20,22,24）
  FULL_FRAMES   逗号分隔帧号（默认 16,64）
"""
import os
import sys

from PIL import Image, ImageDraw

BASE_DIR = os.environ.get("BASE_DIR", "")
R2_DIR = os.environ.get("R2_DIR", "")
GT_DIR = os.environ.get("GT_DIR", "")
OUT = os.environ.get("OUT", ".")
CROP = tuple(int(v) for v in os.environ.get("CROP", "719,480,1151,960").split(","))
STRIP_FRAMES = [int(v) for v in os.environ.get("STRIP_FRAMES", "8,10,12,14,16,18,20,22,24").split(",") if v.strip()]
FULL_FRAMES = [int(v) for v in os.environ.get("FULL_FRAMES", "16,64").split(",") if v.strip()]


def load_triplet(idx):
    """返回 (r2_up, base, gt)，三者同尺寸。"""
    gt = Image.open(os.path.join(GT_DIR, "{:05d}.png".format(idx))).convert("RGB")
    base = Image.open(os.path.join(BASE_DIR, "{:04d}.png".format(idx))).convert("RGB")
    r2 = Image.open(os.path.join(R2_DIR, "{:04d}.png".format(idx))).convert("RGB")
    if r2.size != gt.size:
        r2 = r2.resize(gt.size, Image.LANCZOS)
    if base.size != gt.size:
        base = base.resize(gt.size, Image.LANCZOS)
    return r2, base, gt


def label(img, text):
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 110, 22], fill=(0, 0, 0))
    d.text((6, 4), text, fill=(255, 255, 0))
    return img


def main():
    os.makedirs(OUT, exist_ok=True)
    # 手部时间条带
    rows = []
    for name, pos in (("r2 x2up", 0), ("base", 1), ("GT", 2)):
        cells = []
        for idx in STRIP_FRAMES:
            t = load_triplet(idx)[pos].crop(CROP)
            if pos == 0 and idx == STRIP_FRAMES[0]:
                t = label(t, name)
            cells.append(t)
        sep = 4
        w = sum(c.size[0] for c in cells) + sep * (len(cells) - 1)
        row = Image.new("RGB", (w, cells[0].size[1]), (32, 32, 32))
        x = 0
        for c in cells:
            row.paste(c, (x, 0))
            x += c.size[0] + sep
        rows.append(row)
    rw = max(r.size[0] for r in rows)
    strip = Image.new("RGB", (rw, sum(r.size[1] for r in rows) + 8), (32, 32, 32))
    y = 0
    for r in rows:
        strip.paste(r, (0, y))
        y += r.size[1] + 4
    strip_path = os.path.join(OUT, "hand_strip.png")
    strip.save(strip_path)
    print("✅ saved:", strip_path)

    # 整帧三联
    for idx in FULL_FRAMES:
        r2, base, gt = load_triplet(idx)
        target_h = 960
        scaled = []
        for name, im in (("r2 x2up", r2), ("base", base), ("GT", gt)):
            w2 = int(im.size[0] * target_h / im.size[1])
            scaled.append(label(im.resize((w2, target_h), Image.LANCZOS), name))
        w = sum(s.size[0] for s in scaled) + 8
        trip = Image.new("RGB", (w, target_h), (32, 32, 32))
        x = 0
        for s in scaled:
            trip.paste(s, (x, 0))
            x += s.size[0] + 4
        p = os.path.join(OUT, "full_f{:03d}.png".format(idx))
        trip.save(p)
        print("✅ saved:", p)


if __name__ == "__main__":
    main()
