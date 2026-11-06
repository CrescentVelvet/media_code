#!/usr/bin/env python3
"""FlashWorld 单图生成结果展示图（供 PPT）。

四联：单图输入（唯一条件） + 轨迹上三个生成视角。
强调口径：FlashWorld 只看一张图，其余视角是生成（幻觉补全），不是重建。

输出：compare_4methods/ppt_assets/fig5_flashworld.png
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WSL = Path.home() / "output"
OUT = Path(__file__).resolve().parent / "ppt_assets"
FONT = "/mnt/c/Windows/Fonts/msyh.ttc"

FW = WSL / "flashworld_results/03_gaussians/jiajia/render"
cells = [
    (WSL / "flashworld_results/_input/jiajia.png",
     "单图输入（唯一条件）", "1439×1919 首帧"),
    (FW / "render_00.png", "生成视角 A", "单目生成 · 24 帧环绕轨迹"),
    (FW / "render_08.png", "生成视角 B", "未见内容全部为模型生成"),
    (FW / "render_16.png", "生成视角 C", "4.06M 高斯（剪枝后 74 万）"),
]

H = 500
imgs = []
for p, t, s in cells:
    if not p.exists():
        raise SystemExit(f"❌ 缺图: {p}")
    im = Image.open(p).convert("RGB")
    imgs.append((im.resize((int(im.width * H / im.height), H), Image.LANCZOS), t, s))

pad, label_h = 14, 64
W = sum(im.width for im, _, _ in imgs) + pad * (len(imgs) + 1)
canvas = Image.new("RGB", (W, H + label_h), "white")
draw = ImageDraw.Draw(canvas)
f_t = ImageFont.truetype(FONT, 24)
f_s = ImageFont.truetype(FONT, 16)
x = pad
for im, t, s in imgs:
    canvas.paste(im, (x, label_h))
    draw.text((x + (im.width - draw.textlength(t, font=f_t)) / 2, 8), t, fill=(20, 20, 20), font=f_t)
    draw.text((x + (im.width - draw.textlength(s, font=f_s)) / 2, 40), s, fill=(110, 110, 110), font=f_s)
    x += im.width + pad
canvas.save(OUT / "fig5_flashworld.png", quality=92)
print("✅ fig5_flashworld.png", canvas.size)
