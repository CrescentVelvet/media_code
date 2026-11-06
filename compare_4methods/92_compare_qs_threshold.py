#!/usr/bin/env python3
"""QuerySplat 两档保存阈值对比图（供 PPT）。

左：阈值 0.05（我们首跑口径，22.2 万高斯）
中：阈值 0.005（论文 yaml 口径，37.1 万高斯）
右：GT 参考帧

输出：compare_4methods/ppt_assets/fig4_qs_threshold.png
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WSL = Path.home() / "output"
OUT = Path(__file__).resolve().parent / "ppt_assets"
OUT.mkdir(exist_ok=True)
FONT = "/mnt/c/Windows/Fonts/msyh.ttc"

cells = [
    (WSL / "querysplat_results/佳佳抱枕/rendered/render_view0.png",
     "opacity ≥ 0.05（首跑口径）", "22.2 万高斯 · 20 MB"),
    (WSL / "querysplat_results/佳佳抱枕_paper/rendered/render_view0.png",
     "opacity ≥ 0.005（论文口径）", "37.1 万高斯 · 33 MB"),
    (WSL / "infinisplat_v2_results/_captures/佳佳抱枕_32v/images/000_5093309607791922.png",
     "GT 参考帧", "同视角原图"),
]

H = 560
imgs = []
for p, t, s in cells:
    if not p.exists():
        raise SystemExit(f"❌ 缺图: {p}")
    im = Image.open(p).convert("RGB")
    imgs.append((im.resize((int(im.width * H / im.height), H), Image.LANCZOS), t, s))

pad, label_h = 14, 66
W = sum(im.width for im, _, _ in imgs) + pad * (len(imgs) + 1)
canvas = Image.new("RGB", (W, H + label_h), "white")
draw = ImageDraw.Draw(canvas)
f_t = ImageFont.truetype(FONT, 24)
f_s = ImageFont.truetype(FONT, 17)
x = pad
for im, t, s in imgs:
    canvas.paste(im, (x, label_h))
    draw.text((x + (im.width - draw.textlength(t, font=f_t)) / 2, 8), t, fill=(20, 20, 20), font=f_t)
    draw.text((x + (im.width - draw.textlength(s, font=f_s)) / 2, 42), s, fill=(110, 110, 110), font=f_s)
    x += im.width + pad
canvas.save(OUT / "fig4_qs_threshold.png", quality=92)
print("✅ fig4_qs_threshold.png", canvas.size)
