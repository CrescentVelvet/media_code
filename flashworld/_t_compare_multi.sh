#!/usr/bin/env bash
# 1cond vs 2cond vs 3cond vs 真值：逐帧 PSNR + 对比拼图
SCRIPT_DIR=/mnt/c/code/media_code/flashworld
export SCRIPT_DIR
source "$SCRIPT_DIR/_env.sh"

python - << 'PYEOF'
import json
import math
import os

import numpy as np
from PIL import Image, ImageDraw

FRAMES = [0, 6, 11, 17, 23]
PNG_DIR = "/mnt/d/output/flashworld_input/jiabao_png"
RENDERS = {
    "1cond": "/mnt/d/output/flashworld_results/04_render_narrow",
    "2cond": "/mnt/d/output/flashworld_results/06_multi_cond/render_2cond",
    "3cond": "/mnt/d/output/flashworld_results/06_multi_cond/render_3cond",
}
COND_FRAMES = {"1cond": {0}, "2cond": {0, 23}, "3cond": {0, 11, 23}}
OUT = "/mnt/d/output/flashworld_results/06_multi_cond/compare_psnr.png"
W, H = 480, 704


def load_real(i):
    import glob
    p = glob.glob(os.path.join(PNG_DIR, "%03d_*.png" % i))[0]
    im = Image.open(p).convert("RGB")
    w, h = im.size
    scale = max(H / h, W / w)
    nw, nh = int(W / scale), int(H / scale)
    im = im.crop(((w - nw) // 2, (h - nh) // 2, (w - nw) // 2 + nw, (h - nh) // 2 + nh))
    return np.asarray(im.resize((W, H)), dtype=np.float64)


def psnr(a, b):
    mse = ((a - b) ** 2).mean()
    return 99.0 if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


rows = []
print("%-6s %-8s %-8s %-8s" % ("帧", "1cond", "2cond", "3cond"))
for fi in FRAMES:
    real = load_real(fi)
    row = [Image.fromarray(real.astype(np.uint8))]
    scores = []
    for name in ("1cond", "2cond", "3cond"):
        p = os.path.join(RENDERS[name], "render_%02d.png" % fi)
        r = np.asarray(Image.open(p).convert("RGB").resize((W, H)), dtype=np.float64)
        s = psnr(real, r)
        scores.append(s)
        row.append(Image.fromarray(r.astype(np.uint8)))
    tag = "".join("*" if fi in COND_FRAMES[n] else " " for n in ("1cond", "2cond", "3cond"))
    print("%-6d %-8.2f %-8.2f %-8.2f  %s" % (fi, scores[0], scores[1], scores[2], tag))
    rows.append(row)

# 拼图：行=帧，列=真值/1cond/2cond/3cond
sc = 0.5
tw, th = int(W * sc), int(H * sc)
bar, pad = 26, 6
labels = ["真值(采集帧)", "1cond", "2cond(0,23)", "3cond(0,11,23)"]
canvas = Image.new("RGB", (pad + 4 * (tw + pad), pad + len(FRAMES) * (th + bar + pad)), "white")
d = ImageDraw.Draw(canvas)
for r, row in enumerate(rows):
    y = pad + r * (th + bar + pad)
    for c, im in enumerate(row):
        x = pad + c * (tw + pad)
        canvas.paste(im.resize((tw, th)), (x, y + bar))
        d.rectangle([x, y, x + tw, y + bar], fill=(30, 30, 30))
        d.text((x + 6, y + 7), "%s | 帧%02d" % (labels[c], FRAMES[r]), fill="white")
canvas.save(OUT)
print("✅", OUT)
PYEOF
