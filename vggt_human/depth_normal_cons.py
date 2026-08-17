#!/usr/bin/env python3
"""depth_normal_cons.py — Depth-normal consistency constraint (P1-1).

Improves reconstruction geometry by enforcing consistency between:
1. Normals estimated from the rendered depth map (Sobel gradients).
2. Normals estimated from the initial point cloud (KNN + PCA).

Pipeline:
  - estimate_normal_from_depth: Sobel on rendered depth → per-pixel camera-space normal
  - estimate_point_normals: KNN (k=20) neighborhoods → PCA → per-point normal
  - depth_normal_consistency_loss: project point normals to image, transform render
    normal to world, loss = 1 - dot(render_normal, point_normal) on valid pixels

Env vars:
  USE_DEPTH_NORMAL       : 1 = enable depth-normal consistency loss (default: 1)
  DEPTH_NORMAL_WEIGHT    : loss weight (default: 0.05)
"""
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Depth → normal (Sobel)
# ---------------------------------------------------------------------------
def estimate_normal_from_depth(depth, fx, fy, cx, cy):
    """Estimate per-pixel surface normals from a depth map via Sobel gradients.

    In camera coords: x = (u - cx) * z / fx, y = (v - cy) * z / fy.
    The surface normal is perpendicular to the depth-gradient tangents:
      nx = -dz/du / fx,  ny = -dz/dv / fy,  nz = 1.0
    then normalized.

    Args:
        depth: (1, H, W) or (H, W) tensor — rendered depth (camera z).
        fx, fy: focal lengths (floats).
        cx, cy: principal point (floats).
    Returns:
        (3, H, W) tensor — unit normals in camera coordinates.
    """
    # Normalize to (1, 1, H, W) for conv2d
    if depth.ndim == 2:
        d = depth.unsqueeze(0).unsqueeze(0)
    elif depth.ndim == 3:
        d = depth.unsqueeze(1)  # (1, 1, H, W) if (1, H, W)
    else:
        d = depth
    d = d.float()

    # Sobel kernels
    dtype = d.dtype
    device = d.device
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=dtype, device=device).view(1, 1, 3, 3) / 4.0
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=dtype, device=device).view(1, 1, 3, 3) / 4.0

    # Gradients (pad with replicate to avoid border zeros)
    dz_dx = F.conv2d(F.pad(d, (1, 1, 1, 1), mode="replicate"), sobel_x)  # (1,1,H,W)
    dz_dy = F.conv2d(F.pad(d, (1, 1, 1, 1), mode="replicate"), sobel_y)

    dz_dx = dz_dx.squeeze()  # (H, W)
    dz_dy = dz_dy.squeeze()
    if dz_dx.ndim == 0:  # single-pixel edge case
        dz_dx = dz_dx.unsqueeze(0)
        dz_dy = dz_dy.unsqueeze(0)

    # Normal = (-dz/dx / fx, -dz/dy / fy, 1)
    nx = -dz_dx / fx
    ny = -dz_dy / fy
    nz = torch.ones_like(dz_dx)
    normal = torch.stack([nx, ny, nz], dim=0)  # (3, H, W)
    normal = F.normalize(normal, dim=0, eps=1e-6)
    return normal


# ---------------------------------------------------------------------------
# Point cloud normals (KNN + PCA)
# ---------------------------------------------------------------------------
def estimate_point_normals(points, k=20):
    """Estimate per-point normals from KNN neighborhoods via PCA (SVD).

    For each point, finds k nearest neighbors, centers them, and takes the
    left singular vector with the smallest singular value as the normal.

    Args:
        points: (N, 3) torch tensor — point cloud coordinates.
        k: number of nearest neighbors (default: 20).
    Returns:
        (N, 3) torch tensor — unit normals (orientation unnormalized).
    """
    device = points.device if isinstance(points, torch.Tensor) else "cpu"
    pts_np = points.detach().cpu().numpy() if isinstance(points, torch.Tensor) else points

    try:
        from scipy.spatial import cKDTree
    except ImportError:
        sys.exit("❌ scipy not installed. Run: pip install scipy")

    tree = cKDTree(pts_np)
    _, indices = tree.query(pts_np, k=k)  # (N, k)
    indices = torch.tensor(indices, device=device, dtype=torch.long)

    # Gather neighbors: (N, k, 3)
    neighbors = points[indices]
    centered = neighbors - neighbors.mean(dim=1, keepdim=True)  # (N, k, 3)

    # Batched SVD: centered (N, k, 3) → U (N, k, k), S (N, 3), Vt (N, 3, 3)
    _, _, Vt = torch.linalg.svd(centered)
    normals = Vt[:, -1, :]  # (N, 3) — smallest singular value → normal

    normals = F.normalize(normals, dim=-1, eps=1e-6)
    print(f"  ✅ {len(points):,} point normals estimated (k={k})")
    return normals


# ---------------------------------------------------------------------------
# Camera params extraction
# ---------------------------------------------------------------------------
def _get_camera_params(camera, device):
    """Extract R, T, fx, fy, cx, cy, W, H from a 3DGS Camera object."""
    R = torch.tensor(camera.R, dtype=torch.float32, device=device)
    T = torch.tensor(camera.T, dtype=torch.float32, device=device)
    W = int(camera.image_width)
    H = int(camera.image_height)
    # Compute intrinsics from FoV (works for all Camera types)
    fx = float(W / (2 * math.tan(camera.FoVx / 2)))
    fy = float(H / (2 * math.tan(camera.FoVy / 2)))
    cx = float(W / 2.0)
    cy = float(H / 2.0)
    return R, T, fx, fy, cx, cy, W, H


# ---------------------------------------------------------------------------
# Depth-normal consistency loss
# ---------------------------------------------------------------------------
def depth_normal_consistency_loss(render_depth, render_normal, point_normals,
                                  points, camera):
    """Consistency loss between rendered-depth normals and point-cloud normals.

    1. Estimate normal from rendered depth (Sobel) → camera-space normal map.
    2. Transform render normal to world space (R^T @ normal_cam).
    3. Project point normals to image pixels (same projection as dynamic_filter).
    4. Orient point normals towards camera.
    5. loss = 1 - dot(render_normal_world, point_normal_world) on valid pixels.

    Args:
        render_depth: (1, H, W) tensor — rendered depth (differentiable → 3DGS).
        render_normal: unused (kept for API compatibility; we compute from depth).
        point_normals: (N, 3) tensor — precomputed point normals (world coords).
        points: (N, 3) tensor — 3D points (gaussians.get_xyz, differentiable).
        camera: 3DGS Camera object (with .R, .T, .FoVx, .FoVy, .image_width/height).
    Returns:
        scalar tensor — mean cosine distance loss on valid pixels.
    """
    device = render_depth.device
    R, T, fx, fy, cx, cy, W, H = _get_camera_params(camera, device)

    # 1. Render normal from depth (camera coords) → world coords
    normal_cam = estimate_normal_from_depth(render_depth, fx, fy, cx, cy)  # (3, H, W)
    normal_world = torch.einsum("ij,jhw->ihw", R.T, normal_cam)  # (3, H, W)
    normal_world = F.normalize(normal_world, dim=0, eps=1e-6)

    # 2. Project point normals to image
    P_cam = points @ R.T + T  # (N, 3) in camera coords
    z = P_cam[:, 2]
    valid = z > 1e-3
    z_safe = z.clamp(min=1e-3)
    u = (P_cam[:, 0] / z_safe * fx + cx).round().long()
    v = (P_cam[:, 1] / z_safe * fy + cy).round().long()
    in_image = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)

    # Orient point normals towards camera
    cam_pos = (-R.T @ T)  # (3,) camera center in world
    view_dir = cam_pos.unsqueeze(0) - points  # (N, 3)
    view_dir = F.normalize(view_dir, dim=-1, eps=1e-6)
    dot_view = (point_normals * view_dir).sum(dim=-1)
    pn_oriented = torch.where(
        (dot_view < 0).unsqueeze(-1), -point_normals, point_normals)

    # Place point normals at projected pixels
    point_normal_map = torch.zeros(3, H, W, device=device)
    has_normal = torch.zeros(H, W, device=device)
    u_v = u[in_image]
    v_v = v[in_image]
    pn_v = pn_oriented[in_image]
    # Sort by depth descending so nearest points overwrite (per-pixel nearest)
    sort_idx = torch.argsort(z[in_image], descending=True)
    point_normal_map[:, v_v[sort_idx], u_v[sort_idx]] = pn_v[sort_idx].T
    has_normal[v_v[sort_idx], u_v[sort_idx]] = 1.0

    # 3. Valid mask: pixels with both depth and point normal
    depth_valid = (render_depth.squeeze() > 1e-3).float()  # (H, W)
    valid_mask = depth_valid * has_normal  # (H, W)

    # 4. Loss: 1 - dot(render_normal, point_normal) on valid pixels
    dot = (normal_world * point_normal_map).sum(dim=0)  # (H, W)
    loss_per_pixel = 1.0 - dot
    n_valid = valid_mask.sum().clamp(min=1)
    loss = (valid_mask * loss_per_pixel).sum() / n_valid
    return loss
