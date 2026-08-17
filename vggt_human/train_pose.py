#!/usr/bin/env python3
"""train_pose.py — Wrapper for 3DGS train.py with PoseAdjuster + PoseRefineModule.

Replaces the original `python train.py -s ... -m ...` call when POSE_ADJUST=1 or
POSE_REFINE=1. When both are 0, falls back to standard 3DGS behavior (no modifications).

Flow:
  1. Load COLMAP scene (cameras, images, points3D) — same as 3DGS Scene but inline.
  2. PoseAdjuster (if POSE_ADJUST=1): center + gravity-align + scale.
  3. Create Camera objects; wrap with PoseRefinedCamera (if POSE_REFINE=1).
  4. GaussianModel.create_from_pcd + training_setup.
  5. Training loop (L1+SSIM + pose reg_loss + dual optimizer step + densification).
  6. Save PLY at save_iterations (+ pose_adjuster.json).
  7. End: reverse transform gaussians → save original-coords PLY.

Run inside GS_DIR (same as original train.py):
  ( cd $GS_DIR && python $VGGT_HUMAN_DIR/train_pose.py )

Env vars (set by 04/07 .sh):
  GS_DIR, SOURCE_DIR, GAUSSIAN_DIR, ITERATIONS, SH_DEGREE,
  POSE_ADJUST, POSE_REFINE, REFINE_INTRINSIC, POSE_REFINE_WEIGHT,
  POSE_REFINE_LR_Q, POSE_REFINE_LR_T, POSE_REFINE_LR_I,
  GRAVITY_PRIOR, WHITE_BG, RES, DEVICE
"""
import os
import sys
import math
import random
import time
import json
import shutil
from pathlib import Path
from argparse import Namespace

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

# ── Path setup ──────────────────────────────────────────────────────────────
GS_DIR = os.environ.get("GS_DIR", "../gaussian-splatting")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, GS_DIR)
sys.path.insert(0, SCRIPT_DIR)

from scene import GaussianModel  # noqa: E402
from scene.cameras import Camera  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from utils.loss_utils import l1_loss, ssim  # noqa: E402
from utils.graphics_utils import focal2fov, getProjectionMatrix  # noqa: E402

from pose_adjuster import PoseAdjuster, quat_to_rotmat, rotmat_to_quat  # noqa: E402
from pose_refine import PoseRefinedCamera  # noqa: E402
from render_novel import parse_cameras_txt, parse_images_txt  # noqa: E402

# ── Env vars ────────────────────────────────────────────────────────────────
SOURCE_DIR = os.environ.get("SOURCE_DIR", "")
GAUSSIAN_DIR = os.environ.get("GAUSSIAN_DIR", "")
ITERATIONS = int(os.environ.get("ITERATIONS", "30000"))
SH_DEGREE = int(os.environ.get("SH_DEGREE", "3"))
POSE_ADJUST = os.environ.get("POSE_ADJUST", "1") == "1"
POSE_REFINE = os.environ.get("POSE_REFINE", "1") == "1"
REFINE_INTRINSIC = os.environ.get("REFINE_INTRINSIC", "0") == "1"
POSE_REFINE_WEIGHT = float(os.environ.get("POSE_REFINE_WEIGHT", "0.01"))
LR_Q = float(os.environ.get("POSE_REFINE_LR_Q", "1e-3"))
LR_T = float(os.environ.get("POSE_REFINE_LR_T", "1e-3"))
LR_I = float(os.environ.get("POSE_REFINE_LR_I", "1e-4"))
GRAVITY_PRIOR = os.environ.get("GRAVITY_PRIOR", "0") == "1"
WHITE_BG = os.environ.get("WHITE_BG", "0") == "1"
DEVICE = os.environ.get("DEVICE", "cuda")

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif")


# ---------------------------------------------------------------------------
# COLMAP loaders
# ---------------------------------------------------------------------------
def parse_points3D_txt(path):
    """Returns (points_xyz, points_rgb) as numpy arrays."""
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
# Main
# ---------------------------------------------------------------------------
def main():
    if not SOURCE_DIR:
        sys.exit("❌ SOURCE_DIR not set")
    if not GAUSSIAN_DIR:
        sys.exit("❌ GAUSSIAN_DIR not set")

    t0 = time.time()
    sparse_dir = os.path.join(SOURCE_DIR, "sparse", "0")
    images_dir = os.path.join(SOURCE_DIR, "images")

    # ── 1. Load COLMAP ─────────────────────────────────────────────────────
    print("🔍 [1/6] loading COLMAP scene")
    cameras_info = parse_cameras_txt(os.path.join(sparse_dir, "cameras.txt"))
    images_info = parse_images_txt(os.path.join(sparse_dir, "images.txt"))
    points, colors = parse_points3D_txt(os.path.join(sparse_dir, "points3D.txt"))
    print(f"  {len(cameras_info)} cameras, {len(images_info)} images, {len(points)} points")

    # ── 2. Compute camera params (w2c_r, w2c_t, c2w_t, sight_dir) ──────────
    print("📐 [2/6] computing camera params")
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
        sight = c2w_r[2, :]  # camera forward in world (look-at direction)
        sight_dir_list.append(torch.tensor(sight, dtype=torch.float64))

    # ── 3. PoseAdjuster ────────────────────────────────────────────────────
    adjuster = None
    if POSE_ADJUST:
        print("🏋️ [3/6] PoseAdjuster: center + gravity-align + scale")
        adjuster = PoseAdjuster(
            w2c_r_list, w2c_t_list, c2w_t_list, sight_dir_list,
            enable_trans=True, enable_rotate=True, enable_scale=True,
            gravity_prior=GRAVITY_PRIOR, device=DEVICE)
        os.makedirs(GAUSSIAN_DIR, exist_ok=True)
        adjuster.save(os.path.join(GAUSSIAN_DIR, "pose_adjuster.json"))
        # Transform cameras
        for i in range(len(w2c_r_list)):
            new_r, new_t = adjuster.transform_camera(w2c_r_list[i], w2c_t_list[i])
            w2c_r_list[i] = new_r
            w2c_t_list[i] = new_t
        # Transform points
        pts_tensor = torch.tensor(points, dtype=torch.float32)
        pts_tensor = adjuster.transform_points(pts_tensor)
        points = pts_tensor.numpy().astype(np.float32)
    else:
        print("   [3/6] PoseAdjuster: OFF")

    # ── 4. Create cameras ──────────────────────────────────────────────────
    print(f"🖼️ [4/6] creating cameras (pose_refine={POSE_REFINE})")
    to_tensor = transforms.ToTensor()
    train_cameras = []

    for i, img_info in enumerate(images_info):
        cam_id = img_info[3]
        name = img_info[4]
        _, W, H, params = cameras_info[cam_id]
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]

        image_path = os.path.join(images_dir, name)
        if os.path.isfile(image_path):
            img_pil = Image.open(image_path).convert("RGB")
            image_tensor = to_tensor(img_pil)
        else:
            print(f"  ⚠️ image not found: {image_path}, using zeros")
            image_tensor = torch.zeros(3, H, W)

        FoVx = focal2fov(float(fx), W)
        FoVy = focal2fov(float(fy), H)
        R_np = w2c_r_list[i].float().numpy()
        T_np = w2c_t_list[i].float().numpy()

        cam = Camera(colmap_id=cam_id, R=R_np, T=T_np,
                     FoVx=FoVx, FoVy=FoVy,
                     image=image_tensor, gt_image_mask=None,
                     image_name=name, data_device=DEVICE)
        if POSE_REFINE:
            w2c_qvec = rotmat_to_quat(torch.tensor(R_np, dtype=torch.float32))
            w2c_tvec = torch.tensor(T_np, dtype=torch.float32)
            intrinsic = torch.tensor([fx, fy, cx, cy], dtype=torch.float32)
            cam = PoseRefinedCamera(
                cam, w2c_qvec, w2c_tvec, intrinsic,
                refine_intrinsic=REFINE_INTRINSIC, device=DEVICE,
                lr_q=LR_Q, lr_t=LR_T, lr_i=LR_I)
        train_cameras.append(cam)

    # ── 5. GaussianModel + training setup ──────────────────────────────────
    print("🏋️ [5/6] creating GaussianModel")
    gaussians = GaussianModel(SH_DEGREE)
    pts_t = torch.tensor(points, dtype=torch.float32, device=DEVICE)
    cols_t = torch.tensor(colors, dtype=torch.float32, device=DEVICE) / 255.0
    pcd = Namespace(points=pts_t, colors=cols_t, normals=None, metas=None)

    camera_extent = 10.0 if POSE_ADJUST else 1.0
    gaussians.create_from_pcd(pcd, camera_extent)
    gaussians.training_setup(camera_extent)

    # ── 6. Training loop ────────────────────────────────────────────────────
    print(f"🚀 [6/6] training: {ITERATIONS} iters, {len(train_cameras)} cams")
    print(f"  pose_adjust={POSE_ADJUST} pose_refine={POSE_REFINE} "
          f"refine_intrinsic={REFINE_INTRINSIC} weight={POSE_REFINE_WEIGHT}")

    bg = torch.ones(3, device=DEVICE) if WHITE_BG else torch.zeros(3, device=DEVICE)
    pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False, antialiasing=False)

    densify_from = 500
    densify_until = 15000
    densify_interval = 100
    opacity_reset = 3000
    save_iters = [7000, ITERATIONS]
    lambda_dssim = 0.2

    os.makedirs(os.path.join(GAUSSIAN_DIR, "point_cloud"), exist_ok=True)
    # Save cfg_args (for render.py compatibility)
    cfg_path = os.path.join(GAUSSIAN_DIR, "cfg_args")
    if not os.path.isfile(cfg_path):
        with open(cfg_path, "w") as f:
            f.write(f"Namespace(sh_degree={SH_DEGREE}, source_path='{SOURCE_DIR}', "
                    f"images='images', resolution=1, white_background={WHITE_BG}, "
                    f"data_device='{DEVICE}', eval=False, model_path='{GAUSSIAN_DIR}')")

    for iteration in range(1, ITERATIONS + 1):
        gaussians.update_learning_rate(iteration)
        viewpoint_cam = random.choice(train_cameras)

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image = render_pkg["render"]
        gt = viewpoint_cam.original_image

        loss = (1.0 - lambda_dssim) * l1_loss(image, gt) + \
               lambda_dssim * (1.0 - ssim(image, gt))
        if POSE_REFINE:
            loss += POSE_REFINE_WEIGHT * viewpoint_cam.pose_module.reg_loss()

        loss.backward()

        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)
        if POSE_REFINE:
            viewpoint_cam.pose_optimizer.step()
            viewpoint_cam.pose_optimizer.zero_grad(set_to_none=True)

        # Densification
        if iteration < densify_until:
            gaussians.add_densification_stats(
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"])
            if iteration > densify_from and iteration % densify_interval == 0:
                size_thr = 20 if iteration > opacity_reset else None
                try:
                    gaussians.densify_and_prune(
                        torch.tensor([0.0002]), 0.005, camera_extent, size_thr)
                except TypeError:
                    gaussians.densify_and_prune(
                        torch.tensor([0.0002]), 0.005, camera_extent)

        # Opacity reset
        if iteration == opacity_reset or iteration == opacity_reset + 1:
            gaussians.reset_opacity()

        # Save PLY
        if iteration in save_iters:
            pc_dir = os.path.join(GAUSSIAN_DIR, f"point_cloud/iteration_{iteration}")
            os.makedirs(pc_dir, exist_ok=True)
            gaussians.save_ply(os.path.join(pc_dir, "point_cloud.ply"))
            print(f"  [iter {iteration}] ✅ saved PLY")

        if iteration % 1000 == 0:
            print(f"  [iter {iteration}/{ITERATIONS}] loss={loss.item():.4f}")

    # ── End: reverse transform for original-coords PLY ─────────────────────
    if adjuster is not None:
        print("🔄 reverse transform for original-coords PLY...")
        orig_dir = os.path.join(GAUSSIAN_DIR, "point_cloud_original")
        os.makedirs(orig_dir, exist_ok=True)

        # Temporarily set gaussians to original coords and save
        with torch.no_grad():
            orig_pts = adjuster.reverse_points(gaussians.get_xyz.detach())
            # Save the adjusted PLY (already saved above), plus a manually-constructed original PLY
            # For a full reverse (scaling/rotation), patch the gaussian properties:
            save_gaussian_ply(
                os.path.join(orig_dir, "point_cloud.ply"),
                orig_pts,
                adjuster.reverse_scaling(gaussians.get_scaling.detach()),
                adjuster.reverse_rotation(gaussians.get_rotation.detach()),
                gaussians.get_opacity.detach(),
                gaussians.get_features.detach(),
            )
        print(f"  ✅ original-coords PLY: {orig_dir}/point_cloud.ply")

    print(f"\n🎉 Done. {time.time() - t0:.1f}s. Model: {GAUSSIAN_DIR}")


def save_gaussian_ply(path, xyz, scaling, rotation, opacity, features):
    """Save a 3DGS-format PLY with given gaussian attributes (original coords)."""
    from plyfile import PlyData, PlyElement
    n = xyz.shape[0]
    dtype_full = [(f"x", "f4"), (f"y", "f4"), (f"z", "f4"),
                  ("nx", "f4"), ("ny", "f4"), ("nz", "f4")]
    # SH: f_dc (3) + f_rest (rest)
    dc = features[:, 0, :]  # (N, 3) — DC component
    rest = features[:, 1:, :].reshape(n, -1)  # (N, 15) for SH degree 3
    for i in range(3):
        dtype_full.append((f"f_dc_{i}", "f4"))
    for i in range(rest.shape[1]):
        dtype_full.append((f"f_rest_{i}", "f4"))
    for i in range(3):
        dtype_full.append((f"opacity_{i}", "f4"))
    for i in range(3):
        dtype_full.append((f"scale_{i}", "f4"))
    for i in range(4):
        dtype_full.append((f"rot_{i}", "f4"))

    elements = np.empty(n, dtype=dtype_full)
    xyz_np = xyz.cpu().numpy()
    elements["x"] = xyz_np[:, 0]
    elements["y"] = xyz_np[:, 1]
    elements["z"] = xyz_np[:, 2]
    elements["nx"] = 0; elements["ny"] = 0; elements["nz"] = 0
    dc_np = dc.cpu().numpy()
    for i in range(3):
        elements[f"f_dc_{i}"] = dc_np[:, i]
    rest_np = rest.cpu().numpy()
    for i in range(rest.shape[1]):
        elements[f"f_rest_{i}"] = rest_np[:, i]
    op_np = opacity.cpu().numpy()
    for i in range(3):
        elements[f"opacity_{i}"] = op_np[:, 0]  # broadcast
    sc_np = scaling.cpu().numpy()
    for i in range(3):
        elements[f"scale_{i}"] = sc_np[:, i]
    rot_np = rotation.cpu().numpy()
    for i in range(4):
        elements[f"rot_{i}"] = rot_np[:, i]

    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(path)


if __name__ == "__main__":
    main()
