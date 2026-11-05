#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""01c_check_gt.py — flash_human GT 渲染产物质检(深度/法线数值 + 相机一致性)

用 Blender 内置 OpenImageIO 读 EXR(本机 cv2 的 EXR 被编译禁用,OIIO 最稳)。

用法:
  "<BLENDER>/blender.exe" -b --python 01c_check_gt.py -- \
      --dir "D:/dataset/human_fbx_gt/rosemary_smoke"

检查项:
  1. depth EXR: 最小值 ≈ 相机距离 - 人体半径(>0.5m 量级),背景为极大值
  2. normal EXR: 有效像素法线模长 ≈ 1
  3. cameras.json: cam0 到 target 距离 ≈ meta.distance_m
  4. rgb/depth/normal 文件数一致
  5. rgb 色彩 sanity: 品红污染像素占比(2026-11-05 事故:无图环境纹理节点
     输出占位品红,全场偏紫;几何/深度检查全部 PASS 但色彩全废)
"""
import json
import os
import sys

import bpy  # noqa: F401  (确保在 Blender 环境)
import OpenImageIO as oiio
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:]
BASE = argv[argv.index("--dir") + 1] if "--dir" in argv else ""

meta = json.load(open(os.path.join(BASE, "meta.json"), encoding="utf-8"))
cams = json.load(open(os.path.join(BASE, "cameras.json"), encoding="utf-8"))
dist = meta["distance_m"]

ok = True
for sub in ("rgb", "depth", "normal"):
    n_files = len(os.listdir(os.path.join(BASE, sub)))
    print(f"[{sub}] 文件数: {n_files}")
    ok &= n_files == meta["n_views"]

# 深度/法线逐视角抽查(首/中/尾)
for i in (1, meta["n_views"] // 2, meta["n_views"]):
    p = os.path.join(BASE, "depth", f"{i:06d}.exr")
    inp = oiio.ImageInput.open(p)
    d = inp.read_image(format="float")[..., 0]
    finite = d[(d > 0.1) & (d < 1000)]
    dmin = float(finite.min()) if finite.size else -1
    in_range = finite.size / d.size
    print(f"[depth {i:06d}] min={dmin:.3f}m 中位={float(np.median(finite)):.3f}m "
          f"有效占比={in_range:.3f} (相机距离 {dist:.3f}m)")
    ok &= 0.5 < dmin < dist          # 最近表面应显著近于相机距离

    p = os.path.join(BASE, "normal", f"{i:06d}.exr")
    inp = oiio.ImageInput.open(p)
    n = np.linalg.norm(inp.read_image(format="float")[..., :3], axis=-1)
    nv = n[n > 0.01]
    nmed = float(np.median(nv)) if nv.size else 0
    print(f"[normal {i:06d}] 模长中位={nmed:.4f}")
    ok &= abs(nmed - 1.0) < 0.02

# 色彩 sanity:人物像素(alpha>200)里品红(R、B 显著高于 G)占比应≈0
for i in (1, meta["n_views"] // 2, meta["n_views"]):
    p = os.path.join(BASE, "rgb", f"{i:06d}.png")
    inp = oiio.ImageInput.open(p)
    rgba = inp.read_image(format="float")
    rgb, a = rgba[..., :3], rgba[..., 3]
    person = a > 0.8
    if person.sum() == 0:
        print(f"[color {i:06d}] 无人物像素(全透明)?")
        ok = False
        continue
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    magenta = person & (r > 0.45) & (b > 0.45) & (r > g + 0.25) & (b > g + 0.25)
    frac = float(magenta.sum()) / float(person.sum())
    print(f"[color {i:06d}] 品红像素占比={frac:.4f}")
    ok &= frac < 0.01

c0 = np.array(cams["cameras"][0]["transform_cam2world"])
tgt = np.array(meta["target"])
print(f"[geom] cam0→target = {float(np.linalg.norm(c0[:3, 3] - tgt)):.3f}m "
      f"vs distance_m = {dist:.3f}m")
ok &= abs(float(np.linalg.norm(c0[:3, 3] - tgt)) - dist) < 0.05

print("CHECK_RESULT:", "PASS" if ok else "FAIL")
