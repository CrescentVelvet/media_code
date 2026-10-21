#!/usr/bin/env bash
# 条件帧数扫描评估：全 24 帧 PSNR（vs 采集包真值），含 1/2/3/5/8/12/24cond
SCRIPT_DIR=/mnt/c/code/media_code/flashworld
export SCRIPT_DIR
source "$SCRIPT_DIR/_env.sh"

python - << 'PYEOF'
import glob
import json
import math
import os

import numpy as np
from PIL import Image

PNG_DIR = "/mnt/d/output/flashworld_input/jiabao_png"
BASE = "/mnt/d/output/flashworld_results"
RENDERS = {
    "1cond":  BASE + "/04_render_narrow",
    "2cond":  BASE + "/06_multi_cond/render_2cond",
    "3cond":  BASE + "/06_multi_cond/render_3cond",
    "5cond":  BASE + "/06_multi_cond/render_5cond",
    "8cond":  BASE + "/06_multi_cond/render_8cond",
    "12cond": BASE + "/06_multi_cond/render_12cond",
    "24cond": BASE + "/06_multi_cond/render_24cond",
}
COND_SETS = {
    "1cond": {0},
    "2cond": {0, 23},
    "3cond": {0, 11, 23},
    "5cond": {0, 6, 11, 17, 23},
    "8cond": {0, 3, 6, 10, 13, 16, 20, 23},
    "12cond": {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 21, 23},
    "24cond": set(range(24)),
}
W, H = 480, 704
OUT_TXT = BASE + "/06_multi_cond/sweep_psnr.txt"

_real_cache = {}
def load_real(i):
    if i not in _real_cache:
        p = glob.glob(os.path.join(PNG_DIR, "%03d_*.png" % i))[0]
        im = Image.open(p).convert("RGB")
        w, h = im.size
        scale = max(H / h, W / w)
        nw, nh = int(W / scale), int(H / scale)
        im = im.crop(((w - nw) // 2, (h - nh) // 2, (w - nw) // 2 + nw, (h - nh) // 2 + nh))
        _real_cache[i] = np.asarray(im.resize((W, H)), dtype=np.float64)
    return _real_cache[i]

def psnr(a, b):
    mse = ((a - b) ** 2).mean()
    return 99.0 if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)

names = list(RENDERS)
table = {n: [] for n in names}
for fi in range(24):
    real = load_real(fi)
    for n in names:
        p = os.path.join(RENDERS[n], "render_%02d.png" % fi)
        r = np.asarray(Image.open(p).convert("RGB").resize((W, H)), dtype=np.float64)
        table[n].append(psnr(real, r))

lines = []
hdr = "帧   " + "".join("%9s" % n for n in names)
lines.append(hdr)
for fi in range(24):
    row = "%02d%s " % (fi, "*" if any(fi in COND_SETS[n] for n in names) else " ")
    for n in names:
        v = table[n][fi]
        row += "%8.2f%s" % (v, "*" if fi in COND_SETS[n] else " ")
    lines.append(row)
lines.append("-" * len(hdr))
mean_row = "均值 "
for n in names:
    mean_row += "%9.2f" % np.mean(table[n])
lines.append(mean_row)
# 非条件帧均值（对 24cond 无意义，用 12cond 的条件集合外的帧）
nc = [fi for fi in range(24) if fi not in COND_SETS["12cond"]]
row = "非条件均值(12cond口径) "
lines.append("")
for n in names:
    lines.append("  %-7s 条件帧均值 %6.2f | 非条件帧均值 %6.2f" % (
        n,
        np.mean([table[n][fi] for fi in sorted(COND_SETS[n])]),
        np.mean([table[n][fi] for fi in range(24) if fi not in COND_SETS[n]])))
txt = "\n".join(lines)
print(txt)
with open(OUT_TXT, "w", encoding="utf-8", newline="\n") as f:
    f.write(txt + "\n")
print("\n✅ %s" % OUT_TXT)
PYEOF
