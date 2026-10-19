#!/usr/bin/env python3
"""看 FlashWorld PLY 里高斯沿「输入帧相机视线」的深度分布。
重影=多深度副本 → 直方图应在人所在区域呈双峰。PLY 坐标系 = json cameras[0] 相机系，
所以深度就是 xyz 的 -Z（OpenGL 看 -Z）。只看画面中央 ROI 内的高斯。
"""
import json
import math
import re
import sys

import numpy as np

ply_path, json_path = sys.argv[1], sys.argv[2]

with open(ply_path, "rb") as f:
    raw = f.read(8192)
    end = raw.index(b"end_header") + len(b"end_header") + 1
    head = raw[:end].decode("ascii", "replace")
    n = int(re.search(r"element vertex (\d+)", head).group(1))
    props = [l.split()[2] for l in head.splitlines() if l.startswith("property")]
    nm = {k: i for i, k in enumerate(props)}
    f.seek(end)
    d = np.fromfile(f, dtype=np.float32).reshape(n, len(props))

xyz = d[:, 0:3].astype(np.float64)
op = 1.0 / (1.0 + np.exp(-d[:, nm["opacity"]].astype(np.float64)))

meta = json.load(open(json_path))
_, H, W = meta["resolution"]
c0 = meta["cameras"][0]
# json 内参是原图的，crop 换算与 03_render_ply 一致；这里用相对量即可
fx, fy = c0["fx"], c0["fy"]
cx, cy = c0["cx"], c0["cy"]
iw, ih = 1440.0, 1920.0
scale = max(H / ih, W / iw)
fx, fy = fx * scale, fy * scale
cx = (cx - (iw - int(W / scale)) // 2) * scale
cy = (cy - (ih - int(H / scale)) // 2) * scale

# 相机系 = cameras[0] 系（OpenGL）：深度 = -z
depth = -xyz[:, 2]
front = depth > 0.05
u = fx * xyz[:, 0] / np.maximum(depth, 1e-6) + cx
v = fy * xyz[:, 1] / np.maximum(depth, 1e-6) + cy
# 中央 ROI：人所在区域（画面中央 40%~60% 宽、20%~80% 高）
roi = front & (u > W * 0.3) & (u < W * 0.7) & (v > H * 0.2) & (v < H * 0.8) & (op > 0.3)
dep = depth[roi]
print("ROI 内高斯(op>0.3): %d" % len(dep))
hist, edges = np.histogram(dep, bins=30, range=(0, np.percentile(dep, 99)))
for h, e in zip(hist, edges):
    print("%5.2f~%5.2f | %s %d" % (e, e + (edges[1] - edges[0]), "#" * (h * 60 // max(hist.max(), 1)), h))
