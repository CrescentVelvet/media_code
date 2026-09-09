#!/usr/bin/env python3
"""colmap_io.py — 极简 COLMAP 相机读取（阶段三/四共用）。

只解析本链路需要的：每帧的画幅、内参（fx/fy/cx/cy）、外参（qvec/tvec，w2c）。
不解析 3D 点云，也不做BIN 格式——输入来自 vggt_human 的 03_source（文本格式）。

返回的 qvec/tvec 是 **world→camera**（COLMAP 约定），可直接组 P = K[R|t]。
"""
import os
from pathlib import Path

import numpy as np


def qvec2rotmat(q):
    """COLMAP qvec (w,x,y,z) → R (3,3)。"""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotmat2qvec(R):
    """R (3,3) → COLMAP qvec (w,x,y,z)。"""
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(R).as_quat()  # (x,y,z,w)
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def _parse_cameras(path):
    """cameras.txt → {cam_id: (model, W, H, params)}。"""
    cams = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            cid = int(parts[0])
            model = parts[1]
            W, H = int(parts[2]), int(parts[3])
            params = [float(x) for x in parts[4:]]
            cams[cid] = (model, W, H, params)
    return cams


def _intrinsics(model, W, H, params):
    """各模型 → (fx, fy, cx, cy)。畸变参数本链路不用（图像已去畸变）。"""
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params[0], params[1], params[2]
        return f, f, cx, cy
    if model == "PINHOLE":
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        return fx, fy, cx, cy
    if model in ("SIMPLE_RADIAL", "RADIAL", "OPENCV", "FULL_OPENCV"):
        # 前 4 个都是 fx, fy, cx, cy（RADIAL/FULL_OPENCV 的 OPENCV 子集同理）
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        return fx, fy, cx, cy
    if model == "SIMPLE_RADIAL_FISHEYE":
        f, cx, cy = params[0], params[1], params[2]
        return f, f, cx, cy
    raise ValueError(f"未支持的相机模型: {model}")


def read_cameras(source_dir):
    """读 COLMAP sparse 文本模型 → list of dict。

    优先读 sparse/0，没有就退回 sparse/。
    每项：{stem, name, W, H, fx, fy, cx, cy, qvec, tvec, R, T}
    R/T 是 world→camera：X_cam = R @ X_world + T。
    """
    source_dir = Path(source_dir)
    sparse = None
    for cand in (source_dir / "sparse" / "0", source_dir / "sparse",
                 source_dir / "sparse" / "0" / "0"):
        if (cand / "cameras.txt").exists() and (cand / "images.txt").exists():
            sparse = cand
            break
    if sparse is None:
        raise FileNotFoundError(
            f"找不到 COLMAP 文本模型（cameras.txt/images.txt）: {source_dir}/sparse/0")

    cams = _parse_cameras(sparse / "cameras.txt")

    views = []
    with open(sparse / "images.txt") as f:
        lines = [ln.rstrip("\n") for ln in f if not ln.startswith("#")]
    for i in range(0, len(lines), 2):
        head = lines[i].split()
        if len(head) < 10:
            continue
        qvec = np.array([float(x) for x in head[1:5]], dtype=np.float64)
        tvec = np.array([float(x) for x in head[5:8]], dtype=np.float64)
        cam_id = int(head[8])
        name = " ".join(head[9:])
        if cam_id not in cams:
            continue
        model, W, H, params = cams[cam_id]
        fx, fy, cx, cy = _intrinsics(model, W, H, params)
        views.append({
            "stem": Path(name).stem,
            "name": name,
            "W": W, "H": H,
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "qvec": qvec, "tvec": tvec,
            "R": qvec2rotmat(qvec), "T": tvec,
        })
    return views


def view_index(views):
    return {v["stem"]: v for v in views}


def proj_matrix(v):
    """→ P (3,4) 世界→像素，K (3,3)。"""
    K = np.array([[v["fx"], 0, v["cx"]],
                  [0, v["fy"], v["cy"]],
                  [0, 0, 1.0]], dtype=np.float64)
    P = K @ np.hstack([v["R"], v["T"].reshape(3, 1)])
    return P, K
