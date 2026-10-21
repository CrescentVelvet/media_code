#!/usr/bin/env python3
"""长轨迹条件实验评估：逐帧 PSNR + 拉普拉斯锐度，按「共享采集帧」对齐。

traj41 的槽位 i ↔ 采集帧 F41[i]；traj82 的槽位 i ↔ 采集帧 i。
公平比较只在 41 个共享采集帧上进行（traj82 只取这些槽位）。

输出：sweep2_psnr.txt（表格）+ 打印摘要。
"""
import glob
import json
import os
import re
import sys

import numpy as np
from PIL import Image

ROOT = "/mnt/d/output/flashworld_results/07_long_cond"
JSON_DIR = "/mnt/d/output/flashworld_input/json_lc"
PNG_DIR = "/mnt/d/output/flashworld_input/jiabao_png"

F41 = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36,
       38, 40, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73,
       75, 77, 79, 81]

# 实验名 -> (轨迹槽位->采集帧 映射)
EXPS = {
    "traj41_cond12": F41,
    "traj41_cond41": F41,
    "traj82_cond24": list(range(82)),
    "traj82_cond41": list(range(82)),
    "traj82_cond82": list(range(82)),
}


def load_rgb(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float64)


def crop_like_render(arr):
    """把 1440x1920 真值按 app.py 的 center-crop+resize 逻辑变到 480x704。"""
    H, W = arr.shape[:2]
    scale = max(480 / W, 704 / H)
    nw, nh = int(W * scale + 0.5), int(H * scale + 0.5)
    im = Image.fromarray(arr.astype(np.uint8)).resize((nw, nh), Image.BILINEAR)
    x0, y0 = (nw - 480) // 2, (nh - 704) // 2
    return np.asarray(im.crop((x0, y0, x0 + 480, y0 + 704)), dtype=np.float64)


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return 99.0 if mse < 1e-12 else 10 * np.log10(255.0 ** 2 / mse)


def lap_var(a):
    """灰度拉普拉斯方差（锐度）。"""
    g = a.mean(axis=2)
    lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1]
           + g[1:-1, :-2] + g[1:-1, 2:])
    return float(lap.var())


def cond_slots(name):
    d = json.load(open(os.path.join(JSON_DIR, name + ".json"), encoding="utf-8"))
    return set(p["index"] for p in d.get("image_prompts", []))


# 预载 41 个共享帧真值（crop 后）
gt = {}
for fi in F41:
    p = glob.glob(os.path.join(PNG_DIR, "%03d_*.png" % fi))[0]
    gt[fi] = crop_like_render(load_rgb(p))

rows = {}
for name, slot2frame in EXPS.items():
    rdir = os.path.join(ROOT, "render_" + name)
    cond = cond_slots(name)
    per = {}
    for slot, fi in enumerate(slot2frame):
        if fi not in gt:
            continue
        rp = os.path.join(rdir, "render_%02d.png" % slot)
        if not os.path.exists(rp):
            continue
        r = load_rgb(rp)
        per[fi] = (psnr(r, gt[fi]), lap_var(r), slot in cond)
    rows[name] = per

# 锐度基线：真值
gt_lap = {fi: lap_var(a) for fi, a in gt.items()}

names = list(EXPS)
hdr = "采集帧 | " + " | ".join("%-16s" % n for n in names) + " | GT锐度"
lines = [hdr, "-" * len(hdr)]
for fi in F41:
    cells = []
    for n in names:
        if fi in rows[n]:
            p, l, isc = rows[n][fi]
            cells.append("%5.2f%s / %6.1f" % (p, "*" if isc else " ", l))
        else:
            cells.append("-")
    lines.append("%5d  | %s | %6.1f" % (fi, " | ".join("%-16s" % c for c in cells),
                                         gt_lap[fi]))
lines.append("-" * len(hdr))
for n in names:
    per = rows[n]
    ps = [v[0] for v in per.values()]
    ls = [v[1] for v in per.values()]
    cp = [v[0] for v in per.values() if v[2]]
    np_ = [v[0] for v in per.values() if not v[2]]
    lines.append("%-14s PSNR均值 %5.2f | 条件帧 %5.2f | 非条件 %s | 锐度均值 %6.1f (GT %6.1f)"
                 % (n, np.mean(ps), np.mean(cp) if cp else float("nan"),
                    "%5.2f" % np.mean(np_) if np_ else "  -  ",
                    np.mean(ls), np.mean(list(gt_lap.values()))))

out = os.path.join(ROOT, "sweep2_psnr.txt")
with open(out, "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines[-8:]))
print("✅", out)
