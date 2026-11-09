#!/usr/bin/env python3
"""生成 Scene Splatter 的轨道相机轨迹 pickle（camera.pickle.gz）。

官方约定（scenesplatter.py prepare_data + flash3d data_to_c2w；2026-11-09 已用
--inspect-example 对照官方 assert/cameras/camera0.pickle.gz 实测确认）：
  camera_poses["intrinsics"]: list，每帧一个 np.float32 flat-4
      [fx_n, fy_n, cx_n, cy_n]（除以图像宽/高归一化，主点 0.5；
      官方示例 elem=[0.4725, 0.8400, 0.5, 0.5]，fy_n=0.84 ↔ fov_y≈61°，
      prepare_data 里 K_scale_target 会乘回 w/h）。⚠️ README 说 "4×4 matrix"
      与代码取值方式有出入，以 flat-4 为准。
  camera_poses["poses"]: list，每帧 **3×4** w2c（OpenCV 约定 x 右 y 下 z 前，
      末行 [0,0,0,1] 省略；data_to_c2w 取逆得 c2w）。
      官方示例 143 帧，首帧 R≈I、t=[0.102, 0.020, -0.488]（相机在 +z 看向原点）。

用法：
  python 02_gen_traj.py                          # 按 env 生成轨迹
  python 02_gen_traj.py --inspect-example        # 打印官方示例相机 pickle 的结构
  python 02_gen_traj.py --selftest               # 读回刚写的 OUT 自检

Env vars:
  OUT         输出 pickle.gz 路径（生成模式必填）
  NUM_FRAMES  总帧数，默认 100。流水线按 camera_sample_interval=2 抽样、
              每段 per_video_length=16（重叠 2）切视频：N=100 → 约 3~4 轮迭代
  YAW_SPAN    环绕总角度（度），默认 40（对齐 FlashWorld 默认 orbit 口径）
  RADIUS      轨道半径（世界单位），默认 1.0
  HEIGHT      相机高度偏移，默认 0.0（负值=相机上移，y 向下为正）
  FOV_Y       垂直视场角（度），默认 60
  W/H         内参目标分辨率，默认 1024x576（main() 里硬编码 resize 到该尺寸）
  DIRECTION   ccw（默认）| cw
  SS_DIR      官方仓路径（--inspect-example 用，_env.sh 已默认）
"""
import gzip
import math
import os
import pickle
import sys

import numpy as np

SS_DIR = os.environ.get("SS_DIR", os.path.expanduser("~/repos/Scene-Splatter"))


def build_orbit(num_frames, yaw_span, radius, height, fov_y, w, h, direction):
    """绕原点（0,0,0）水平轨道。OpenCV 相机系：x 右、y 下、z 前。"""
    fy = (h / 2.0) / math.tan(math.radians(fov_y) / 2.0)
    fx = fy  # 方形像素
    intr = np.array([fx / w, fy / h, 0.5, 0.5], dtype=np.float32)

    target = np.array([0.0, 0.0, 0.0])
    world_up = np.array([0.0, 1.0, 0.0])  # y 向下为正的世界

    sign = -1.0 if direction == "cw" else 1.0
    poses, intrinsics = [], []
    for i in range(num_frames):
        yaw = sign * math.radians(yaw_span) * i / max(num_frames - 1, 1)
        pos = target + np.array([radius * math.sin(yaw),
                                 height,
                                 radius * math.cos(yaw)])
        z_axis = target - pos
        z_axis /= np.linalg.norm(z_axis)
        x_axis = np.cross(world_up, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 0], c2w[:3, 1], c2w[:3, 2] = x_axis, y_axis, z_axis
        c2w[:3, 3] = pos
        w2c = np.linalg.inv(c2w).astype(np.float32)
        poses.append(w2c[:3])  # 官方格式 3×4（末行 [0,0,0,1] 省略）
        intrinsics.append(intr.copy())
    return {"intrinsics": intrinsics, "poses": poses}


def inspect_example():
    path = os.path.join(SS_DIR, "assert", "cameras", "camera0.pickle.gz")
    if not os.path.exists(path):
        sys.exit(f"❌ 示例不存在（官方仓未 clone？）: {path}")
    with gzip.open(path, "rb") as f:
        cam = pickle.load(f)
    print(f"📷 示例相机: {path}")
    print(f"   keys: {list(cam.keys())}")
    for k, v in cam.items():
        if isinstance(v, (list, tuple)):
            arr0 = np.asarray(v[0])
            print(f"   {k}: list[{len(v)}], elem shape={arr0.shape} dtype={arr0.dtype}")
            print(f"      elem[0]:\n{arr0}")
        else:
            print(f"   {k}: {type(v).__name__} = {v}")


def selftest(out):
    with gzip.open(out, "rb") as f:
        cam = pickle.load(f)
    n = len(cam["poses"])
    p0 = np.asarray(cam["poses"][0])
    print(f"📷 自检 {out}")
    print(f"   frames={n}  keys={list(cam.keys())}")
    print(f"   intr[0] (flat-4): {np.asarray(cam['intrinsics'][0]).ravel()}")
    print(f"   pose[0] (w2c 3×4, t_z 应≈-RADIUS，相机在 +z 看向原点):\n{p0}")
    print(f"   pose[-1]:\n{np.asarray(cam['poses'][-1])}")


def main():
    if "--inspect-example" in sys.argv:
        inspect_example()
        return
    if "--selftest" in sys.argv:
        out = os.environ["OUT"]
        selftest(out)
        return

    out = os.environ.get("OUT", "")
    if not out:
        sys.exit("❌ OUT 未设置（输出 pickle.gz 路径）")
    num_frames = int(os.environ.get("NUM_FRAMES", "100"))
    yaw_span = float(os.environ.get("YAW_SPAN", "40"))
    radius = float(os.environ.get("RADIUS", "1.0"))
    height = float(os.environ.get("HEIGHT", "0.0"))
    fov_y = float(os.environ.get("FOV_Y", "60"))
    w, h = int(os.environ.get("W", "1024")), int(os.environ.get("H", "576"))
    direction = os.environ.get("DIRECTION", "ccw")

    cam = build_orbit(num_frames, yaw_span, radius, height, fov_y, w, h, direction)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with gzip.open(out, "wb") as f:
        pickle.dump(cam, f)
    print(f"🎉 轨迹已写: {out}  ({num_frames} 帧, yaw_span={yaw_span}°, "
          f"r={radius}, fov_y={fov_y}°, {w}x{h}, {direction})")


if __name__ == "__main__":
    main()
