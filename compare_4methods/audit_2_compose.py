#!/usr/bin/env python3
"""把 AUDIT-2 的三组渲染拼成一张对照图（GT | all | self | other），便于目检。"""
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

AUD = Path(os.path.expanduser(os.environ["AUD"]))
REFS = Path(os.path.expanduser(os.environ.get(
    "REF", "~/output/da3_results/_captures/佳佳抱枕/images")))
VIEW = int(os.environ.get("VIEW", "0"))
OUT = Path(os.environ.get("OUT", "/mnt/c/code/media_code/_tmp_view/audit2_view.png"))
CROP = os.environ.get("CROP", "")          # "x,y,w,h" 可选，裁人脸

refs = sorted([p for p in REFS.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
gt = Image.open(refs[VIEW]).convert("RGB")
tiles = [("GT", gt)]
for tag in ("all", "self", "other"):
    p = AUD / f"render_{tag}" / f"view_{VIEW:03d}.png"
    if p.exists():
        tiles.append((tag, Image.open(p).convert("RGB")))
w, h = tiles[1][1].size
gt = gt.resize((w, h), Image.LANCZOS)
tiles[0] = ("GT", gt)
if CROP:
    x, y, cw, ch = (int(v) for v in CROP.split(","))
else:
    x, y, cw, ch = 0, 0, w, h
sc = max(1, int(round(360 / cw)))
pad = 6
W = len(tiles) * (cw * sc + pad) + pad
H = ch * sc + 26 + pad
canvas = Image.new("RGB", (W, H), (255, 255, 255))
dr = ImageDraw.Draw(canvas)
for i, (name, im) in enumerate(tiles):
    t = im.crop((x, y, x + cw, y + ch)).resize((cw * sc, ch * sc), Image.LANCZOS)
    ox = pad + i * (cw * sc + pad)
    canvas.paste(t, (ox, 22))
    dr.text((ox + 2, 6), name, fill=(0, 0, 0))
OUT.parent.mkdir(parents=True, exist_ok=True)
canvas.save(OUT)
print(f"💾 {OUT}  ({canvas.size[0]}x{canvas.size[1]})")
