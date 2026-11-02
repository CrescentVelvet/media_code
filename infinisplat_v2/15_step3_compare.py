#!/usr/bin/env python3
"""Step3 视觉对照：同一视角下 GT / 原始 / 全量精调 / 只改颜色 / 只改几何。

手部框沿用 SAM3 的实测位置（归一化 x 0.395–0.601、y 0.416–0.487），
外扩到 (0.34,0.36)-(0.66,0.54) 保证把手和抱枕边缘都框进来。
"""
import glob
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

BASE = Path("/mnt/d/output/infinisplat_v2_results_20261101/jiajia")
FT = Path(os.path.expanduser("~/output/infinisplat_v2_results/_s3ft"))
IMGS = Path("/mnt/d/output/infinisplat_v2_results_20261101/_captures/jiajia/images")
OUT = Path("/mnt/d/output/infinisplat_v2_results_20261102/_diag")
OUT.mkdir(parents=True, exist_ok=True)


def load_preprocess(path, w, h):
    """与 render_check.model_preprocess 同口径（中心裁剪缩放）。"""
    im = Image.open(path).convert("RGB")
    scale = max(w / im.width, h / im.height)
    left = max(0.0, (im.width - w / scale) / 2)
    top = max(0.0, (im.height - h / scale) / 2)
    box = (left, top, min(float(im.width), left + w / scale),
           min(float(im.height), top + h / scale))
    return im.resize((w, h), Image.Resampling.BICUBIC, box=box)


views = [0, 10, 21]
srcs = [("GT", None),
        ("original", BASE / "eval"),
        ("ft-all", FT / "full" / "eval"),
        ("ft-sh only", FT / "sh" / "eval"),
        ("ft-geom only", FT / "geom" / "eval")]

for v in views:
    bt = sorted(glob.glob(str(IMGS / "*.png")))[v]
    ref = Image.open(str((FT / "full" / "eval") / f"render_{v:03d}.png"))
    W, H = ref.size
    gt = load_preprocess(bt, W, H)
    box = (int(W * 0.34), int(H * 0.36), int(W * 0.66), int(H * 0.54))
    for scale, name in ((2, "crop"), (1, "full")):
        panels = []
        for label, d in srcs:
            if d is None:
                im = gt
            else:
                p = d / f"render_{v:03d}.png"
                im = Image.open(str(p)).convert("RGB")
            if scale > 1:
                im = im.crop(box)
            im = im.resize((im.width * scale, im.height * scale), Image.Resampling.NEAREST)
            panels.append((label, im))
        pw, ph = panels[0][1].size
        canvas = Image.new("RGB", (pw * len(panels) + 8 * (len(panels) - 1), ph + 22), (18, 18, 18))
        dr = ImageDraw.Draw(canvas)
        for k, (label, im) in enumerate(panels):
            x = k * (pw + 8)
            canvas.paste(im, (x, 22))
            dr.text((x + 4, 6), label, fill=(255, 255, 255))
        p = OUT / f"s3_view{v}_{name}.png"
        canvas.save(p)
        print("写出", p, canvas.size)

# ---- 更紧的手部特写（4x）：这一步才是判断"真变清楚"还是"变斑点"的地方 ----
for v in (10, 21):
    bt = sorted(glob.glob(str(IMGS / "*.png")))[v]
    ref = Image.open(str((FT / "full" / "eval") / f"render_{v:03d}.png"))
    W, H = ref.size
    gt = load_preprocess(bt, W, H)
    boxz = (int(W * 0.38), int(H * 0.385), int(W * 0.62), int(H * 0.505))
    panels = []
    for label, d in srcs:
        im = gt if d is None else Image.open(str(d / f"render_{v:03d}.png")).convert("RGB")
        im = im.crop(boxz)
        im = im.resize((im.width * 4, im.height * 4), Image.Resampling.NEAREST)
        panels.append((label, im))
    pw, ph = panels[0][1].size
    canvas = Image.new("RGB", (pw * len(panels) + 8 * (len(panels) - 1), ph + 22), (18, 18, 18))
    dr = ImageDraw.Draw(canvas)
    for k, (label, im) in enumerate(panels):
        x = k * (pw + 8)
        canvas.paste(im, (x, 22))
        dr.text((x + 4, 6), label, fill=(255, 255, 255))
    p = OUT / f"s3_view{v}_zoom4.png"
    canvas.save(p)
    print("写出", p, canvas.size)
