#!/usr/bin/env python3
"""位姿独立验证：把采集包 pcd.ply 按 transforms.json 位姿+内参投影回真实帧。

判据：投影点应与图像内容（椅子/人轮廓）对齐。对齐 → 位姿与坐标变换正确，
FlashWorld 重影就与位姿无关；错位 → 位姿/转换有问题。

w2c_cv = inv(M @ diag(1,-1,-1,1))（OpenGL c2w → OpenCV w2c，与 som_human 口径一致）

用法（WSL flashworld env）:
  python _t_pose_check.py <pkg_dir> <png_dir> <out_dir> [帧序号...]
"""
import json
import os
import re
import sys

import numpy as np
from PIL import Image

pkg, png_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
frames_idx = [int(x) for x in sys.argv[4:]] or [0, 12, 23, 41, 80]
os.makedirs(out_dir, exist_ok=True)

meta = json.load(open(os.path.join(pkg, "transforms.json")))
fr = meta["frames"]

# pcd.ply：解析 header，读 xyz+rgb（binary_little_endian）
with open(os.path.join(pkg, "pcd.ply"), "rb") as f:
    raw = f.read(65536)
    end = raw.index(b"end_header") + len(b"end_header") + 1
    head = raw[:end].decode("ascii", "replace")
n = int(re.search(r"element vertex (\d+)", head).group(1))
props = [l.split()[2] for l in head.splitlines() if l.startswith("property")]
nm = {k: i for i, k in enumerate(props)}
with open(os.path.join(pkg, "pcd.ply"), "rb") as f:
    f.seek(end)
    d = np.fromfile(f, dtype=np.uint8).reshape(n, 15)  # 3 float(12B) + 3 uchar
xyz = d[:, :12].copy().view(np.float32).reshape(n, 3).astype(np.float64)
# rgb 可能是 float 存的 uchar
has_rgb = True
rgb = d[:, 12:15].astype(np.float64)
print("pcd: %d points, xyz range x[%.2f,%.2f] y[%.2f,%.2f] z[%.2f,%.2f]" % (
    n, *xyz.min(0)[0:1], *xyz.max(0)[0:1], xyz[:, 1].min(), xyz[:, 1].max(),
    xyz[:, 2].min(), xyz[:, 2].max()))

D = np.diag([1.0, -1.0, -1.0, 1.0])

for fi in frames_idx:
    f = fr[fi]
    M = np.array(f["transform_matrix"], dtype=np.float64)  # OpenGL c2w
    w2c = np.linalg.inv(M @ D)                             # OpenCV w2c
    fx, fy, cx, cy = f["fl_x"], f["fl_y"], f["cx"], f["cy"]

    pngs = [p for p in os.listdir(png_dir) if p.startswith("%03d_" % fi)]
    if not pngs:
        print("  ⚠️ 帧 %03d 无 png，跳过" % fi)
        continue
    img = Image.open(os.path.join(png_dir, pngs[0])).convert("RGB")
    W, H = img.size
    a = np.asarray(img).astype(np.float64)

    # 内参按 png 分辨率换算（transforms.json 是原始分辨率）
    sw, sh = W / float(f["w"]), H / float(f["h"])
    kx, ky, kcx, kcy = fx * sw, fy * sh, cx * sw, cy * sh

    pw = np.concatenate([xyz, np.ones((n, 1))], 1) @ w2c.T
    z = pw[:, 2]
    ok = (z > 0.05)
    u = kx * pw[:, 0] / z + kcx
    v = ky * pw[:, 1] / z + kcy
    ok &= (u >= 3) & (u < W - 4) & (v >= 3) & (v < H - 4)
    uu, vv = u[ok].astype(int), v[ok].astype(int)
    print("帧 %03d: %d/%d 点落在画面内" % (fi, ok.sum(), n))

    # 不透明 5x5 点，错位一眼可见；左原图右覆盖
    ov = a.copy()
    for du in range(-2, 3):
        for dv in range(-2, 3):
            ov[vv + dv, uu + du] = rgb[ok]
    side = np.concatenate([a, ov], 1)
    Image.fromarray(side.astype(np.uint8)).save(
        os.path.join(out_dir, "pose_check_%03d.png" % fi))
    print("  ✅ pose_check_%03d.png" % fi)
print("输出:", out_dir)
