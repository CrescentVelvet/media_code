#!/usr/bin/env python3
"""把 3DGS 的 PLY 按多个正交视角投影成 PNG 拼图，用于快速肉眼验证重建形状。

不做真正的 splatting 光栅化（那要 CUDA 栅格器），只把高斯中心按深度排序画散点，
颜色取 SH 的 DC 项还原。够用来判断"是不是那个物体""有没有穿帮/塌陷"。

Env vars:
  PLY            输入 .ply
  OUT            输出 png（默认 <ply 同名>_preview.png）
  PREVIEW_SIZE   单视角画布边长（默认 480）

注意别用 SIZE 当变量名：conda-forge 的 gcc/binutils activate 脚本会把
SIZE 注入成 x86_64-conda-linux-gnu-size 的路径，取到的是垃圾值。
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from plyfile import PlyData

# 标准 3DGS：color = 0.5 + SH_C0 * f_dc，SH_C0 = 1/(2*sqrt(pi))
SH_C0 = 0.28209479177387814


def load_ply(path):
    v = PlyData.read(path)["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1).astype(np.float32)
    opacity = v["opacity"].astype(np.float32)
    rgb = np.clip(0.5 + SH_C0 * dc, 0.0, 1.0)
    alpha = 1.0 / (1.0 + np.exp(-opacity))          # opacity 存的是 logit
    return xyz, rgb, alpha


def render_view(xyz, rgb, alpha, az_deg, el_deg, size, radius):
    """正交投影：绕 Y 轴 az，再绕 X 轴 el；半径 radius 映射到画布半宽。"""
    a, e = np.radians(az_deg), np.radians(el_deg)
    Ry = np.array([[np.cos(a), 0, np.sin(a)],
                   [0, 1, 0],
                   [-np.sin(a), 0, np.cos(a)]], np.float32)
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]], np.float32)
    p = xyz @ Ry.T @ Rx.T

    half = size / 2.0
    u = (p[:, 0] / radius * half + half).astype(np.int32)
    v = (-p[:, 1] / radius * half + half).astype(np.int32)   # 图像 y 向下
    inside = (u >= 0) & (u < size) & (v >= 0) & (v < size)
    u, v, col, alp = u[inside], v[inside], rgb[inside], alpha[inside]

    # 按 alpha 做加权累积（近似 alpha 混合的收敛结果），而不是"近的覆盖远的"。
    # 后者在 1M 高斯下会失真：大量低不透明度小高斯本该叠加成不透明，
    # 单一覆盖只取最后落点，画出来稀疏发暗（实测 1M 熊猫比 262k 还淡）。
    flat_idx = v * size + u
    acc_c = np.zeros((size * size, 3), np.float32)
    acc_a = np.zeros(size * size, np.float32)
    np.add.at(acc_c, flat_idx, col * alp[:, None])
    np.add.at(acc_a, flat_idx, alp)
    canvas = (acc_c / np.maximum(acc_a, 1e-6)[:, None]).reshape(size, size, 3)
    canvas *= np.clip(acc_a, 0, 1).reshape(size, size, 1)      # 没被覆盖到的像素保持黑

    img = Image.fromarray((np.clip(canvas, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    d.text((8, 6), f"az={az_deg:+.0f} el={el_deg:+.0f}", fill=(255, 255, 255))
    return img


def main():
    ply = os.environ.get("PLY", "")
    if not ply or not os.path.isfile(ply):
        sys.exit("❌ PLY 未指定或不存在（env PLY）")
    out = os.environ.get("OUT", os.path.splitext(ply)[0] + "_preview.png")
    size = int(os.environ.get("PREVIEW_SIZE", "480"))

    xyz, rgb, alpha = load_ply(ply)
    lo, hi = xyz.min(0), xyz.max(0)
    center = (lo + hi) / 2
    xyz = xyz - center                                 # 居中，别让原点偏移把物体推出画布
    radius = float(np.percentile(np.abs(xyz), 99)) or 1.0

    print(f"🔍 {os.path.basename(ply)}")
    print(f"  🔢 高斯数: {len(xyz)}")
    print(f"  📐 bbox: {np.round(hi - lo, 3)}  中心偏移: {np.round(center, 3)}")
    print(f"  🎚️ opacity>0.5 占比: {(alpha > 0.5).mean() * 100:.1f}%")

    views = [(0, 0), (90, 0), (180, 0), (270, 0), (0, 60), (0, -30)]
    tiles = [render_view(xyz, rgb, alpha, az, el, size, radius) for az, el in views]

    sheet = Image.new("RGB", (size * 3, size * 2), (18, 18, 18))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % 3) * size, (i // 3) * size))
    sheet.save(out)
    print(f"  🖼️ 预览: {out}")


if __name__ == "__main__":
    main()
