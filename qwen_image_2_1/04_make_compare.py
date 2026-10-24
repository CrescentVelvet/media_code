#!/usr/bin/env python3
"""批量去反光效果对比图生成（Windows/WSL 均可跑，只依赖 Pillow）。

产物：
  1) 逐帧左右对比图：左=原图，右=去反光，顶部带文件名标注
  2) 网格拼图：所有帧按文件名序拼成 contact sheet（快速翻页看跨视角一致性）

用法（env 或位置参数均可，位置参数优先）：
  python 04_make_compare.py [SRC_DIR] [DST_DIR] [OUT_DIR]
Env vars:
  SRC_DIR / DST_DIR / OUT_DIR   目录（默认见下）
  NAMES                         只处理这些文件名（逗号分隔，不含则全量交集）
  MAX_W                         拼接图单栏最大宽度（默认 960，等比缩放）
  GRID_COLS                     网格列数（默认 4）
"""
import os
import sys

SRC_DIR = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
    "SRC_DIR", r"/mnt/d/dataset/物体反光数据/耳机白/image")
DST_DIR = sys.argv[2] if len(sys.argv) > 2 else os.environ.get(
    "DST_DIR", r"/mnt/d/output/qwen_image_2_1_dereflect/耳机白/image")
OUT_DIR = sys.argv[3] if len(sys.argv) > 3 else os.environ.get(
    "OUT_DIR", r"/mnt/d/output/qwen_image_2_1_dereflect/耳机白/compare")
NAMES = os.environ.get("NAMES", "")
MAX_W = int(os.environ.get("MAX_W", "960"))
GRID_COLS = int(os.environ.get("GRID_COLS", "4"))

LABEL_H = 40
GAP = 8


def main():
    from PIL import Image, ImageDraw

    if not os.path.isdir(SRC_DIR) or not os.path.isdir(DST_DIR):
        sys.exit("❌ 目录不存在: " + SRC_DIR + " / " + DST_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)
    pair_dir = os.path.join(OUT_DIR, "pairs")

    if NAMES:
        names = [n.strip() for n in NAMES.split(",") if n.strip()]
    else:
        src_names = set(os.listdir(SRC_DIR))
        names = sorted(n for n in os.listdir(DST_DIR)
                       if n in src_names and not n.startswith("_"))
    if not names:
        sys.exit("❌ 无交集文件")

    print("🚀 对比图生成: " + str(len(names)) + " 帧")
    print("  SRC: " + SRC_DIR)
    print("  DST: " + DST_DIR)
    print("  OUT: " + OUT_DIR)

    os.makedirs(pair_dir, exist_ok=True)
    made = []
    for n in names:
        a = Image.open(os.path.join(SRC_DIR, n)).convert("RGB")
        b = Image.open(os.path.join(DST_DIR, n)).convert("RGB")
        if b.size != a.size:
            b = b.resize(a.size, Image.LANCZOS)
        scale = min(1.0, MAX_W / a.width)
        if scale < 1.0:
            sz = (int(a.width * scale), int(a.height * scale))
            a, b = a.resize(sz, Image.LANCZOS), b.resize(sz, Image.LANCZOS)
        w, h = a.size
        canvas = Image.new("RGB", (w * 2 + GAP, h + LABEL_H), (24, 24, 24))
        canvas.paste(a, (0, LABEL_H))
        canvas.paste(b, (w + GAP, LABEL_H))
        d = ImageDraw.Draw(canvas)
        d.text((8, 12), "ORIGINAL  " + n, fill=(255, 200, 80))
        d.text((w + GAP + 8, 12), "DEREFLECT", fill=(120, 220, 120))
        out = os.path.join(pair_dir, os.path.splitext(n)[0] + "_pair.jpg")
        canvas.save(out, quality=92)
        made.append((n, canvas))

    # 网格拼图（只放去反光结果，按序排列——快速判断跨视角一致性）
    cell_w = MAX_W // 2
    cell_h = int(cell_w * made[0][1].height / (made[0][1].width / 2 - GAP / 2))
    cols = GRID_COLS
    rows = (len(names) + cols - 1) // cols
    grid = Image.new("RGB", (cols * cell_w, rows * (cell_h + 24)), (16, 16, 16))
    d = ImageDraw.Draw(grid)
    for i, n in enumerate(names):
        im = Image.open(os.path.join(DST_DIR, n)).convert("RGB")
        im.thumbnail((cell_w, cell_h), Image.LANCZOS)
        x = (i % cols) * cell_w
        y = (i // cols) * (cell_h + 24)
        grid.paste(im, (x, y + 24))
        d.text((x + 4, y + 6), n, fill=(255, 200, 80))
    grid_path = os.path.join(OUT_DIR, "grid_dereflect.jpg")
    grid.save(grid_path, quality=92)
    print("✅ pairs: " + pair_dir + "（" + str(len(made)) + " 张）")
    print("✅ grid : " + grid_path)


if __name__ == "__main__":
    main()
