#!/usr/bin/env python3
"""把 02 导出的 scene.ply 渲染成轨道视频（scene_splatter env 内运行）。

⚠️ 【待实测】本脚本是骨架阶段的备用件：02 流水线本身每轮就导出
render_video_{i}.mp4（最终轮 = 最终高斯的轨迹渲染），横评首选用它。
本脚本用于「拿 PLY 离线在自定义新轨迹上渲染」的场景。

渲染走官方仓自带的 gaussianSplatting 光栅器（GL camera 约定：
world_view_transform = w2c.T，full_proj = P @ world_view_transform）。

用法（WSL，scene_splatter env）：
  python 03_render_ply.py <ply> <out_mp4>

Env vars:
  NUM_FRAMES  渲染帧数，默认 120
  YAW_SPAN    环绕角度（度），默认 360
  W/H         输出分辨率，默认 1024x576
  FOV_Y       垂直视场角（度），默认 60
  SS_DIR      官方仓路径（引 gaussianSplatting 用，_env.sh 已默认）
"""
import math
import os
import sys

import numpy as np
import torch

PLY = sys.argv[1] if len(sys.argv) > 1 else ""
OUT = sys.argv[2] if len(sys.argv) > 2 else ""
NUM_FRAMES = int(os.environ.get("NUM_FRAMES", "120"))
YAW_SPAN = float(os.environ.get("YAW_SPAN", "360"))
W = int(os.environ.get("W", "1024"))
H = int(os.environ.get("H", "576"))
FOV_Y = float(os.environ.get("FOV_Y", "60"))
SS_DIR = os.environ.get("SS_DIR", os.path.expanduser("~/repos/Scene-Splatter"))

if not PLY or not OUT:
    sys.exit("❌ 用法: python 03_render_ply.py <ply> <out_mp4>")

sys.path.insert(0, SS_DIR)
from gaussianSplatting.scene.gaussian_model import GaussianModel  # noqa: E402
from gaussianSplatting.gaussian_renderer import render  # noqa: E402
from argparse import Namespace  # noqa: E402

device = "cuda"
gaussians = GaussianModel(3)
gaussians.load_ply(PLY)
gaussians.to(device)


class MiniCam:
    """gaussianSplatting render() 需要的最小相机接口（仿原版 MiniCam）。"""

    def __init__(self, w2c, fovx, fovy, w, h):
        self.image_width = w
        self.image_height = h
        self.FoVx = fovx
        self.FoVy = fovy
        w2c_t = torch.tensor(w2c, dtype=torch.float32, device=device).transpose(0, 1)
        self.world_view_transform = w2c_t
        znear, zfar = 0.01, 100.0
        proj = _projection_matrix(fovx, fovy, znear, zfar).transpose(0, 1).to(device)
        self.full_proj_transform = (w2c_t.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
        self.camera_center = torch.linalg.inv(self.world_view_transform)[3, :3]


def _projection_matrix(fovx, fovy, znear, zfar):
    tan_x, tan_y = math.tan(fovx * 0.5), math.tan(fovy * 0.5)
    return torch.tensor([[1 / tan_x, 0, 0, 0],
                         [0, 1 / tan_y, 0, 0],
                         [0, 0, (zfar + znear) / (zfar - znear), -(2 * zfar * znear) / (zfar - znear)],
                         [0, 0, 1, 0]], dtype=torch.float32)


def orbit_w2c(i, n, yaw_span, radius=1.5):
    yaw = math.radians(yaw_span) * i / max(n - 1, 1)
    pos = np.array([radius * math.sin(yaw), 0.0, radius * math.cos(yaw)], dtype=np.float32)
    z = -pos / np.linalg.norm(pos)  # 看向原点
    x = np.cross([0, 1, 0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2] = x, y, z
    c2w[:3, 3] = pos
    return np.linalg.inv(c2w)


pipe = Namespace(debug=False, compute_cov3D_python=False, prefiltered=False)
bg = torch.zeros(3, device=device)
tan_y = math.tan(math.radians(FOV_Y) * 0.5)
tan_x = tan_y * W / H
fovy = 2 * math.atan(tan_y)
fovx = 2 * math.atan(tan_x)

os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
writer = None
try:
    import imageio
    writer = imageio.get_writer(OUT, fps=24, quality=8)
    for i in range(NUM_FRAMES):
        cam = MiniCam(orbit_w2c(i, NUM_FRAMES, YAW_SPAN), fovx, fovy, W, H)
        img = render(cam, gaussians, pipe, bg)["render"]
        frame = (img.permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()
        writer.append_data(frame)
        if i % 20 == 0:
            print(f"  rendered {i}/{NUM_FRAMES}")
finally:
    if writer is not None:
        writer.close()
print(f"🎉 渲染完成: {OUT}")
