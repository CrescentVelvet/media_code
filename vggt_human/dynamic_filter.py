#!/usr/bin/env python3
"""dynamic_filter.py — Filter 3D points that fall within dynamic object masks.

Multi-view projection voting: for each 3D point, project to all cameras,
check if the projected pixel falls in that camera's dilated dynamic mask.
If in_mask_views / valid_views > threshold, the point is a dynamic contaminant
and is removed from the point cloud before 3DGS initialization.

Projection (w2c convention, COLMAP / OpenCV):
  P_cam = P @ R_w2c^T + T_w2c           # world → camera (batch: (N,3) @ (3,3) + (3,))
  if P_cam.z < 1e-3: skip (behind camera)
  u = P_cam.x / P_cam.z * fx + cx
  v = P_cam.y / P_cam.z * fy + cy
  if (u, v) inside image AND mask[v, u] == 1: cnt_in_mask += 1

  dynamic_ratio = cnt_in_mask / num_valid_views
  if dynamic_ratio > threshold: remove point

Mask dilation: binary mask dilated by dilate_px pixels (max_pool2d) before
sampling, so edge points don't slip through.

Env vars:
  DYNAMIC_THRESHOLD  : dynamic ratio threshold (default: 0.3)
  DYNAMIC_DILATE_PX   : mask dilation in pixels (default: 5)

CLI:
  python dynamic_filter.py --points <ply> --sparse_dir <colmap/sparse/0> \
      --images_dir <colmap/images> --masks_dir <dynamic_mask_dir> [--threshold 0.3]
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

DYNAMIC_THRESHOLD = float(os.environ.get("DYNAMIC_THRESHOLD", "0.3"))
DYNAMIC_DILATE_PX = int(os.environ.get("DYNAMIC_DILATE_PX", "5"))


def _dilate_mask(mask_uint8, dilate_px, device):
    """Dilate a binary mask (H, W) uint8 by dilate_px pixels via max_pool2d."""
    if dilate_px <= 0:
        return torch.tensor(mask_uint8, dtype=torch.float32, device=device)
    t = torch.tensor(mask_uint8, dtype=torch.float32, device=device)
    t = t.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    kernel = 2 * dilate_px + 1
    t = F.max_pool2d(t, kernel_size=kernel, stride=1, padding=dilate_px)
    return t.squeeze(0).squeeze(0)  # (H, W)


def filter_dynamic_points(points, colors, cameras, dynamic_masks,
                          threshold=None, dilate_px=None):
    """Remove 3D points that fall within dynamic masks via multi-view voting.

    Args:
        points: (N, 3) torch.float32 — point cloud coords (world space).
        colors: (N, 3) torch.float32 — point cloud colors.
        cameras: list of dicts, each with keys:
            R (3x3 ndarray, w2c), T (3, ndarray, w2c),
            fx, fy, cx, cy (floats), W, H (ints), image_name (str).
        dynamic_masks: dict {image_name: mask(H,W) uint8}, 1=dynamic.
            None or empty → no filtering (return as-is).
        threshold: dynamic ratio threshold (default: env DYNAMIC_THRESHOLD, 0.3).
            Points with ratio > threshold are removed.
        dilate_px: mask dilation in pixels (default: env DYNAMIC_DILATE_PX, 5).

    Returns:
        (filtered_points, filtered_colors) — torch tensors, dynamic points removed.
    """
    if threshold is None:
        threshold = DYNAMIC_THRESHOLD
    if dilate_px is None:
        dilate_px = DYNAMIC_DILATE_PX

    device = points.device if isinstance(points, torch.Tensor) else "cpu"
    N = len(points)

    # No masks → no filtering
    if not dynamic_masks:
        print("  ⏭️ no dynamic masks, skipping filter")
        return points, colors

    # Ensure tensors
    if not isinstance(points, torch.Tensor):
        points = torch.tensor(points, dtype=torch.float32, device=device)
    if not isinstance(colors, torch.Tensor):
        colors = torch.tensor(colors, dtype=torch.float32, device=device)

    print(f"  🚀 filtering {N:,} points across {len(cameras)} cameras "
          f"(threshold={threshold}, dilate={dilate_px}px)")

    # Per-point accumulators
    in_mask_count = torch.zeros(N, dtype=torch.float32, device=device)
    valid_count = torch.zeros(N, dtype=torch.float32, device=device)

    t0 = time.time()
    cams_with_mask = 0

    for cam in cameras:
        name = cam["image_name"]
        has_mask = name in dynamic_masks

        R = torch.tensor(cam["R"], dtype=torch.float32, device=device)
        T = torch.tensor(cam["T"], dtype=torch.float32, device=device)
        fx = float(cam["fx"])
        fy = float(cam["fy"])
        cx = float(cam["cx"])
        cy = float(cam["cy"])
        W = int(cam["W"])
        H = int(cam["H"])

        # Batch projection: P_cam = P @ R^T + T  → (N, 3)
        P_cam = points @ R.T + T
        z = P_cam[:, 2]
        valid = z > 1e-3

        z_safe = z.clamp(min=1e-3)
        u = P_cam[:, 0] / z_safe * fx + cx
        v = P_cam[:, 1] / z_safe * fy + cy

        in_image = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        # 分母=所有可见视角（含无掩码的相机）。曾只在有掩码的相机上累加，
        # 导致"125 帧里只有 1 帧有掩码"时 ratio 恒为 1/1，过度删除。
        valid_count += in_image.float()

        if not has_mask:
            continue
        cams_with_mask += 1

        # Dilate mask (cached dilated masks could be precomputed, but
        # re-dilating per camera is cheap for typical mask sizes)
        mask = _dilate_mask(dynamic_masks[name], dilate_px, device)

        # Sample mask at projected pixel coords (nearest-neighbor)
        u_int = u.clamp(0, W - 1).round().long()
        v_int = v.clamp(0, H - 1).round().long()
        in_mask = mask[v_int, u_int] > 0

        in_mask_count += (in_image & in_mask).float()

    # Dynamic ratio = in_mask views / valid views
    ratio = in_mask_count / valid_count.clamp(min=1)
    keep = ratio <= threshold

    filtered_points = points[keep]
    filtered_colors = colors[keep]

    n_removed = N - len(filtered_points)
    elapsed = time.time() - t0
    print(f"  ✂️ {N:,} → {len(filtered_points):,} points "
          f"({n_removed:,} removed, {n_removed / max(N, 1) * 100:.1f}%) "
          f"in {elapsed:.1f}s ({cams_with_mask} cams with mask)")

    return filtered_points, filtered_colors


# ---------------------------------------------------------------------------
# CLI (standalone test: PLY + COLMAP + masks → filtered PLY)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Filter dynamic points from a point cloud using multi-view masks.")
    parser.add_argument("--points", required=True, help="input PLY file (world coords)")
    parser.add_argument("--sparse_dir", required=True, help="COLMAP sparse/0 dir (cameras.txt, images.txt)")
    parser.add_argument("--images_dir", required=True, help="COLMAP images dir (for mask matching)")
    parser.add_argument("--masks_dir", required=True, help="dynamic mask dir (PNGs, one per image)")
    parser.add_argument("--threshold", type=float, default=DYNAMIC_THRESHOLD,
                        help=f"dynamic ratio threshold (default: {DYNAMIC_THRESHOLD})")
    parser.add_argument("--dilate_px", type=int, default=DYNAMIC_DILATE_PX,
                        help=f"mask dilation in pixels (default: {DYNAMIC_DILATE_PX})")
    parser.add_argument("--output", default="", help="output PLY (default: <input>_filtered.ply)")
    args = parser.parse_args()

    from plyfile import PlyData, PlyElement
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, SCRIPT_DIR)
    from render_novel import parse_cameras_txt, parse_images_txt, quat_to_rotmat

    # Load PLY
    print(f"📦 loading {args.points}")
    plydata = PlyData.read(args.points)
    v = plydata["vertex"]
    points = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
    colors = np.stack([v["red"], v["green"], v["blue"]], axis=-1).astype(np.float32)
    print(f"  {len(points):,} points")

    # Load COLMAP
    cameras_info = parse_cameras_txt(os.path.join(args.sparse_dir, "cameras.txt"))
    images_info = parse_images_txt(os.path.join(args.sparse_dir, "images.txt"))
    print(f"  {len(cameras_info)} cameras, {len(images_info)} images")

    # Build camera dicts
    cameras = []
    for img in images_info:
        cam_id = img[3]
        name = img[4]
        _, W, H, params = cameras_info[cam_id]
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        quat = img[1]  # [qw,qx,qy,qz] scalar-first, w2c（COLMAP 约定）
        t = img[2]
        R = np.asarray(quat_to_rotmat(*quat), dtype=np.float64)
        T = np.array(t, dtype=np.float64)
        cameras.append({
            "R": R, "T": T, "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "W": W, "H": H, "image_name": name,
        })

    # Load masks
    from PIL import Image
    masks = {}
    for cam in cameras:
        name = cam["image_name"]
        stem = os.path.splitext(name)[0]
        mask_path = os.path.join(args.masks_dir, stem + ".png")
        if os.path.isfile(mask_path):
            m = np.array(Image.open(mask_path).convert("L"))
            masks[name] = (m > 127).astype(np.uint8)
    print(f"  {len(masks)} masks loaded from {args.masks_dir}")

    if not masks:
        print("⚠️ no masks found — output = input (no filtering)")
    else:
        pts_t = torch.tensor(points, dtype=torch.float32)
        cols_t = torch.tensor(colors, dtype=torch.float32)
        pts_t, cols_t = filter_dynamic_points(
            pts_t, cols_t, cameras, masks,
            threshold=args.threshold, dilate_px=args.dilate_px)
        points = pts_t.numpy()
        colors = cols_t.numpy()

    # Save output PLY
    out_path = args.output or os.path.splitext(args.points)[0] + "_filtered.ply"
    print(f"💾 saving {out_path}")
    # ⚠️ 官方 3DGS fetchPly 要求 nx/ny/nz 字段齐全，缺了会 ValueError。
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"),
             ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
             ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    el_arr = np.empty(len(points), dtype=dtype)
    el_arr["x"] = points[:, 0]
    el_arr["y"] = points[:, 1]
    el_arr["z"] = points[:, 2]
    el_arr["nx"] = 0.0
    el_arr["ny"] = 0.0
    el_arr["nz"] = 0.0
    el_arr["red"] = colors[:, 0].clip(0, 255).astype(np.uint8)
    el_arr["green"] = colors[:, 1].clip(0, 255).astype(np.uint8)
    el_arr["blue"] = colors[:, 2].clip(0, 255).astype(np.uint8)
    el = PlyElement.describe(el_arr, "vertex")
    PlyData([el]).write(out_path)
    print(f"✅ Done. {len(points):,} points → {out_path}")
