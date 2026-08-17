#!/usr/bin/env python3
"""pose_adjust_colmap.py — 对 COLMAP 场景做 PoseAdjuster 一次性变换。

读 source/sparse/0/{cameras,images,points3D}.txt → 视线交点居中 + SVD 重力对齐
+ 尺度归一化 → 写 source_adjusted/sparse/0/ + 复制 images/。

变换公式（w2c 约定，与 pose_adjuster.py 一致）:
  相机:  new_w2c_t = scale × (w2c_t + w2c_r @ center)
         new_w2c_r = w2c_r @ pred_rot
  点云:  new_pts = scale × (pts - center) @ pred_rot

无需改 train.py — 03 直接用调整后的 source 训练。

Env vars (set by 02b_pose_adjust.sh):
  SOURCE_DIR, SOURCE_ADJUSTED_DIR, GRAVITY_PRIOR
"""
import os
import sys
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch

# Import PoseAdjuster from vggt_human (sibling directory)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
VGGT_HUMAN_DIR = os.path.join(REPO_DIR, "vggt_human")
sys.path.insert(0, VGGT_HUMAN_DIR)
from pose_adjuster import PoseAdjuster, quat_to_rotmat, rotmat_to_quat  # noqa: E402

SOURCE_DIR = os.environ.get("SOURCE_DIR", "")
SOURCE_ADJUSTED_DIR = os.environ.get("SOURCE_ADJUSTED_DIR", "")
GRAVITY_PRIOR = os.environ.get("GRAVITY_PRIOR", "0") == "1"


# ---------------------------------------------------------------------------
# COLMAP text format parsers (same as pi3_recon.py / render_novel.py).
# ---------------------------------------------------------------------------
def parse_cameras_txt(path):
    """Returns {cam_id: (model, W, H, [params])}."""
    cameras = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cameras[int(parts[0])] = (parts[1], int(parts[2]), int(parts[3]),
                                      [float(x) for x in parts[4:]])
    return cameras


def parse_images_txt(path):
    """Returns list of (img_id, [qw,qx,qy,qz], [tx,ty,tz], cam_id, name)."""
    images = []
    with open(path) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        parts = line.split()
        images.append((int(parts[0]),
                        [float(x) for x in parts[1:5]],
                        [float(x) for x in parts[5:8]],
                        int(parts[8]), parts[9]))
        i += 2  # skip points2D line
    return images


def parse_points3D_txt(path):
    """Returns (xyz (N,3), rgb (N,3))."""
    pts, cols = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            cols.append([int(parts[4]), int(parts[5]), int(parts[6])])
    return np.array(pts, dtype=np.float32), np.array(cols, dtype=np.uint8)


# ---------------------------------------------------------------------------
# COLMAP text format writers.
# ---------------------------------------------------------------------------
def write_cameras_txt(path, cameras_list):
    with open(path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(cameras_list)}\n")
        for cam_id, model, w, h, params in cameras_list:
            ps = " ".join(f"{p:.6f}" for p in params)
            f.write(f"{cam_id} {model} {w} {h} {ps}\n")


def write_images_txt(path, images_list):
    with open(path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write(f"# Number of images: {len(images_list)}\n")
        for img in images_list:
            img_id, qw, qx, qy, qz, tx, ty, tz, cam_id, name = img
            f.write(f"{img_id} {qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} "
                    f"{tx:.8f} {ty:.8f} {tz:.8f} {cam_id} {name}\n\n")


def write_points3D_txt(path, points_list):
    with open(path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR\n")
        f.write(f"# Number of points: {len(points_list)}\n")
        for pt in points_list:
            pid, x, y, z, r, g, b, err = pt
            f.write(f"{pid} {x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)} {err:.6f}\n")


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main():
    if not SOURCE_DIR:
        sys.exit("❌ SOURCE_DIR not set")
    if not SOURCE_ADJUSTED_DIR:
        sys.exit("❌ SOURCE_ADJUSTED_DIR not set")

    sparse_in = os.path.join(SOURCE_DIR, "sparse", "0")
    sparse_out = os.path.join(SOURCE_ADJUSTED_DIR, "sparse", "0")
    images_in = os.path.join(SOURCE_DIR, "images")
    images_out = os.path.join(SOURCE_ADJUSTED_DIR, "images")

    t0 = time.time()
    print("🔍 [1/4] parsing COLMAP scene")
    cameras_info = parse_cameras_txt(os.path.join(sparse_in, "cameras.txt"))
    images_info = parse_images_txt(os.path.join(sparse_in, "images.txt"))
    points, colors = parse_points3D_txt(os.path.join(sparse_in, "points3D.txt"))
    print(f"  {len(cameras_info)} cameras, {len(images_info)} images, {len(points)} points")

    # ── 2. Compute camera params for PoseAdjuster ──────────────────────
    print("📐 [2/4] computing camera params")
    w2c_r_list, w2c_t_list, c2w_t_list, sight_dir_list = [], [], [], []
    for img in images_info:
        quat = img[1]  # [qw, qx, qy, qz]
        t = img[2]     # [tx, ty, tz]
        R = quat_to_rotmat(torch.tensor(quat, dtype=torch.float64)).numpy()
        T = np.array(t, dtype=np.float64)
        w2c_r_list.append(torch.tensor(R, dtype=torch.float64))
        w2c_t_list.append(torch.tensor(T, dtype=torch.float64))
        c2w_t = -R.T @ T
        c2w_t_list.append(torch.tensor(c2w_t, dtype=torch.float64))
        c2w_r = R.T
        sight = c2w_r[2, :]  # camera forward (look-at)
        sight_dir_list.append(torch.tensor(sight, dtype=torch.float64))

    # ── 3. PoseAdjuster ─────────────────────────────────────────────────
    print("🏋️ [3/4] PoseAdjuster: center + gravity-align + scale")
    adjuster = PoseAdjuster(
        w2c_r_list, w2c_t_list, c2w_t_list, sight_dir_list,
        enable_trans=True, enable_rotate=True, enable_scale=True,
        gravity_prior=GRAVITY_PRIOR, device="cpu")

    # Transform cameras
    adjusted_images = []
    for i, img in enumerate(images_info):
        new_r, new_t = adjuster.transform_camera(w2c_r_list[i], w2c_t_list[i])
        new_q = rotmat_to_quat(new_r)  # [w, x, y, z]
        adjusted_images.append((img[0], new_q.tolist(), new_t.tolist(),
                                 img[3], img[4]))

    # Transform points
    pts_tensor = torch.tensor(points, dtype=torch.float32)
    pts_adjusted = adjuster.transform_points(pts_tensor).numpy().astype(np.float32)

    # ── 4. Write adjusted COLMAP scene ──────────────────────────────────
    print(f"📝 [4/4] writing adjusted COLMAP -> {sparse_out}")
    os.makedirs(sparse_out, exist_ok=True)
    os.makedirs(images_out, exist_ok=True)

    # cameras.txt: same (intrinsics don't change, only extrinsics)
    cam_list = [(cid, m, w, h, p) for cid, (m, w, h, p) in sorted(cameras_info.items())]
    write_cameras_txt(os.path.join(sparse_out, "cameras.txt"), cam_list)

    # images.txt: adjusted quaternions + translations
    img_list = [(img[0], img[1][0], img[1][1], img[1][2], img[1][3],
                  img[2][0], img[2][1], img[2][2], img[3], img[4])
                for img in adjusted_images]
    write_images_txt(os.path.join(sparse_out, "images.txt"), img_list)

    # points3D.txt: adjusted XYZ (RGB unchanged)
    pt_list = [(i + 1, float(pts_adjusted[i, 0]), float(pts_adjusted[i, 1]),
                float(pts_adjusted[i, 2]),
                int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2]), 0.0)
               for i in range(len(pts_adjusted))]
    write_points3D_txt(os.path.join(sparse_out, "points3D.txt"), pt_list)

    # Copy images (unchanged)
    if images_in != images_out:
        for name in os.listdir(images_in):
            src = os.path.join(images_in, name)
            if os.path.isfile(src):
                shutil.copy(src, os.path.join(images_out, name))

    # Save PoseAdjuster JSON (for render_orbit.py to transform orbit cameras)
    adjuster.save(os.path.join(SOURCE_ADJUSTED_DIR, "pose_adjuster.json"))

    extent = pts_adjusted.max(0) - pts_adjusted.min(0)
    print(f"  ✅ cameras.txt  ({len(cam_list)} cameras, intrinsics unchanged)")
    print(f"  ✅ images.txt   ({len(img_list)} adjusted extrinsics)")
    print(f"  ✅ points3D.txt ({len(pt_list)} adjusted points; "
          f"extent=[{extent[0]:.2f}, {extent[1]:.2f}, {extent[2]:.2f}])")
    print(f"  ✅ images/      copied {len(os.listdir(images_out))} files")
    print(f"  ⏱️ {time.time() - t0:.1f}s")
    print(f"\n🎉 Done. Adjusted COLMAP: {SOURCE_ADJUSTED_DIR}")
    print(f"  Next: SOURCE_DIR={SOURCE_ADJUSTED_DIR} bash pdfgs_human/03_train_pdfgs.sh")


if __name__ == "__main__":
    main()
