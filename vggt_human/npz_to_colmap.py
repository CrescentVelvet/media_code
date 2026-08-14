#!/usr/bin/env python3
"""npz_to_colmap.py — Convert VGGT-Omega predictions.npz to COLMAP text format.

Reads the npz saved by run_batch.py (step 01) and produces a COLMAP scene
that 3DGS train.py -s reads:
  source/images/<frame>.png
  source/sparse/0/cameras.txt      # PINHOLE intrinsics (from VGGT-Omega's actual)
  source/sparse/0/images.txt       # w2c (qw qx qy qz tx ty tz) — VGGT-Omega
                                   # outputs w2c directly (no c2w->w2c needed)
  source/sparse/0/points3D.txt     # adaptive-conf filtered + voxel-downsampled

VGGT-Omega extrinsic is world-to-camera [R | t] (OpenCV convention), same as
COLMAP. Pi3 outputs c2w (needs inversion); VGGT-Omega outputs w2c directly.

Adaptive confidence filtering: Otsu's method finds the natural threshold in
the confidence histogram (separates high-conf from low-conf points). If the
result is too sparse/dense, falls back to percentile-based.

Voxel downsampling: voxel size auto-computed from scene extent + target count.
Per voxel, the highest-confidence point is kept.

Coordinate alignment: scene is centered at the point-cloud centroid (camera
translations adjusted accordingly). Helps 3DGS training stability.

Env vars (set by 03_npz_to_colmap.sh):
  NPZ_PATH, FRAMES_DIR, SOURCE_DIR, TARGET_POINTS, ALIGN
"""
import os
import sys
import shutil
import math
import time
from pathlib import Path

import numpy as np

NPZ_PATH = os.environ.get("NPZ_PATH", "")
FRAMES_DIR = os.environ.get("FRAMES_DIR", "")
SOURCE_DIR = os.environ.get("SOURCE_DIR", "./source")
TARGET_POINTS = int(os.environ.get("TARGET_POINTS", "200000"))
ALIGN = os.environ.get("ALIGN", "1") == "1"

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp")


# ---------------------------------------------------------------------------
# Geometry helpers (from pi3_3dgs/pi3_recon.py — Shepperd's method).
# ---------------------------------------------------------------------------
def rotmat_to_quat(R):
    """R: 3x3 numpy rotation matrix -> (qw, qx, qy, qz) normalized quaternion."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n > 0:
        qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return qw, qx, qy, qz


# ---------------------------------------------------------------------------
# COLMAP text format writers (from pi3_3dgs/pi3_recon.py).
# ---------------------------------------------------------------------------
def write_cameras_txt(path, cameras_list):
    """cameras_list: [(cam_id, model, W, H, [fx, fy, cx, cy]), ...]"""
    with open(path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(cameras_list)}\n")
        for cam_id, model, w, h, params in cameras_list:
            params_str = " ".join(f"{p:.6f}" for p in params)
            f.write(f"{cam_id} {model} {w} {h} {params_str}\n")


def write_images_txt(path, images_list):
    """images_list: [(img_id, qw, qx, qy, qz, tx, ty, tz, cam_id, name, pts2d), ...]"""
    with open(path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(images_list)}\n")
        for img in images_list:
            img_id, qw, qx, qy, qz, tx, ty, tz, cam_id, name, pts2d = img
            f.write(f"{img_id} {qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} "
                    f"{tx:.8f} {ty:.8f} {tz:.8f} {cam_id} {name}\n")
            f.write("\n")  # empty points2D line (3DGS doesn't need them)


def write_points3D_txt(path, points_list):
    """points_list: [(pid, x, y, z, r, g, b, error), ...]"""
    with open(path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        f.write(f"# Number of points: {len(points_list)}\n")
        for pt in points_list:
            pid, x, y, z, r, g, b, err = pt
            f.write(f"{pid} {x:.6f} {y:.6f} {z:.6f} "
                    f"{int(r)} {int(g)} {int(b)} {err:.6f}\n")


# ---------------------------------------------------------------------------
# Adaptive confidence filtering (Otsu's method).
# ---------------------------------------------------------------------------
def otsu_threshold(conf):
    """Find the threshold that maximizes between-class variance (Otsu)."""
    hist, bin_edges = np.histogram(conf, bins=256, range=(conf.min(), max(conf.max(), 1e-6)))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    total = len(conf)
    w0, sum0, sum_total = 0, 0.0, np.sum(hist * bin_centers)
    best_thresh, best_var = 0.0, 0.0
    for i in range(256):
        w0 += hist[i]
        if w0 == 0:
            continue
        w1 = total - w0
        if w1 == 0:
            break
        sum0 += hist[i] * bin_centers[i]
        m0 = sum0 / w0
        m1 = (sum_total - sum0) / w1
        var_between = w0 * w1 * (m0 - m1) ** 2
        if var_between > best_var:
            best_var = var_between
            best_thresh = bin_centers[i]
    return best_thresh


def adaptive_conf_filter(points, colors, conf):
    """Filter by adaptive confidence threshold (Otsu, with percentile fallback).

    Returns (filtered_points, filtered_colors, filtered_conf, threshold).
    """
    conf_flat = conf.reshape(-1)
    valid = np.isfinite(conf_flat) & (conf_flat > 1e-5)
    conf_valid = conf_flat[valid]

    if len(conf_valid) == 0:
        return points, colors, conf_flat, 0.0

    # Try Otsu first
    threshold = otsu_threshold(conf_valid)
    mask = valid & (conf_flat >= threshold)
    n = mask.sum()

    # Adjust if too sparse or too dense
    if n < len(conf_valid) * 0.05:
        # Otsu too strict — use 10th percentile (keep top 90%)
        threshold = float(np.percentile(conf_valid, 10))
        mask = valid & (conf_flat >= threshold)
    elif n > len(conf_valid) * 0.8:
        # Otsu too permissive — use 40th percentile (keep top 60%)
        threshold = float(np.percentile(conf_valid, 40))
        mask = valid & (conf_flat >= threshold)

    pts = points[mask]
    cols = colors[mask]
    confs = conf_flat[mask]
    return pts, cols, confs, threshold


# ---------------------------------------------------------------------------
# Voxel downsampling (numpy, no extra deps).
# ---------------------------------------------------------------------------
def voxel_downsample(points, colors, conf, target_count):
    """Downsample to ~target_count via voxel grid. Per voxel, keep highest-conf point."""
    n = len(points)
    if n <= target_count:
        return points, colors, conf

    mins = points.min(0)
    extent = points.max(0) - mins
    volume = np.prod(np.maximum(extent, 1e-6))
    voxel_size = max((volume / target_count) ** (1.0 / 3.0), 1e-6)

    voxel_idx = np.floor((points - mins) / voxel_size).astype(np.int64)
    dims = voxel_idx.max(0) + 1
    # Flatten to 1D key for uniqueness
    keys = voxel_idx[:, 0].astype(np.int64) * int(dims[1] * dims[2]) + \
           voxel_idx[:, 1].astype(np.int64) * int(dims[2]) + \
           voxel_idx[:, 2].astype(np.int64)

    # Sort by key, then by confidence descending (first occurrence = highest conf)
    order = np.lexsort((-conf, keys))
    sorted_keys = keys[order]
    _, first_idx = np.unique(sorted_keys, return_index=True)
    result_idx = order[first_idx]

    return points[result_idx], colors[result_idx], conf[result_idx]


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main():
    if not NPZ_PATH:
        sys.exit("❌ NPZ_PATH not set")
    if not os.path.isfile(NPZ_PATH):
        sys.exit(f"❌ NPZ not found: {NPZ_PATH}")

    source_dir = Path(SOURCE_DIR).resolve()
    images_dir = source_dir / "images"
    sparse_dir = source_dir / "sparse" / "0"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load predictions.npz ─────────────────────────────────────────────
    print(f"🔍 [1/5] loading {NPZ_PATH}")
    t0 = time.time()
    pred = np.load(NPZ_PATH, allow_pickle=True)
    keys = list(pred.keys())
    print(f"  keys: {keys}")

    extrinsic = pred["extrinsic"]      # (N, 3, 4) w2c [R | t]
    intrinsic = pred["intrinsic"]      # (N, 3, 3)
    world_points = pred["world_points_from_depth"]  # (N, H, W, 3)
    depth_conf = pred["depth_conf"]     # (N, H, W) or (N, H, W, 1)
    images = pred["images"]             # (N, 3, H, W) in [0,1] (preprocessed)

    if depth_conf.ndim == 4:
        depth_conf = depth_conf[..., 0]
    N = extrinsic.shape[0]
    npz_H, npz_W = images.shape[2], images.shape[3]
    print(f"  N={N} views, image_size={npz_W}x{npz_H} (preprocessed)")
    print(f"  extrinsic={extrinsic.shape}, intrinsic={intrinsic.shape}")
    print(f"  world_points={world_points.shape}, depth_conf={depth_conf.shape}")

    # ── 2. Copy images to source/images/ + scale intrinsics ─────────────────
    print(f"🖼️ [2/5] preparing images -> {images_dir}")
    # Prefer original images from frames/ (better quality for 3DGS training).
    frame_names = []
    if FRAMES_DIR and os.path.isdir(FRAMES_DIR):
        frame_files = sorted([f for f in os.listdir(FRAMES_DIR)
                              if os.path.splitext(f)[1].lower() in IMG_EXTS])
        if len(frame_files) == N:
            for f in frame_files:
                shutil.copy(os.path.join(FRAMES_DIR, f), str(images_dir / f))
            frame_names = frame_files
            # Get original image size for intrinsic scaling
            from PIL import Image as PILImage
            img0 = PILImage.open(os.path.join(FRAMES_DIR, frame_files[0]))
            orig_W, orig_H = img0.size
            scale_x = orig_W / npz_W
            scale_y = orig_H / npz_H
            if abs(scale_x - 1) > 0.01 or abs(scale_y - 1) > 0.01:
                print(f"  scaling intrinsics: {npz_W}x{npz_H} -> {orig_W}x{orig_H}")
                intrinsic = intrinsic.copy()
                intrinsic[:, 0, 0] *= scale_x  # fx
                intrinsic[:, 1, 1] *= scale_y  # fy
                intrinsic[:, 0, 2] *= scale_x  # cx
                intrinsic[:, 1, 2] *= scale_y  # cy
            W, H = orig_W, orig_H
        else:
            print(f"  ⚠️ frames/ has {len(frame_files)} files, expected {N}; using npz images")
    if not frame_names:
        # Fallback: save npz images as PNG
        from PIL import Image as PILImage
        for i in range(N):
            img = (images[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            PILImage.fromarray(img).save(str(images_dir / f"{i:06d}.png"))
        frame_names = [f"{i:06d}.png" for i in range(N)]
        W, H = npz_W, npz_H

    print(f"  {len(frame_names)} images, {W}x{H}")

    # ── 3. Adaptive confidence filtering ────────────────────────────────────
    print(f"✂️ [3/5] adaptive confidence filtering")
    pts_all = world_points.reshape(-1, 3).astype(np.float64)
    imgs_hwc = images.transpose(0, 2, 3, 1)  # (N, H, W, 3) in [0,1]
    cols_all = (imgs_hwc.reshape(-1, 3) * 255).clip(0, 255).astype(np.uint8)
    conf_all = depth_conf.reshape(-1).astype(np.float64)

    # Remove non-finite points first
    finite = np.isfinite(pts_all).all(axis=1) & np.isfinite(conf_all)
    pts_all = pts_all[finite]
    cols_all = cols_all[finite]
    conf_all = conf_all[finite]
    print(f"  finite points: {len(pts_all):,}")

    pts_f, cols_f, conf_f, threshold = adaptive_conf_filter(pts_all, cols_all, conf_all)
    print(f"  Otsu threshold: {threshold:.4f} -> {len(pts_f):,} points")

    # ── 4. Voxel downsampling to ~TARGET_POINTS ────────────────────────────
    print(f"📐 [4/5] voxel downsampling to ~{TARGET_POINTS:,}")
    pts_d, cols_d, conf_d = voxel_downsample(pts_f, cols_f, conf_f, TARGET_POINTS)
    print(f"  -> {len(pts_d):,} points")

    # ── 5. Coordinate alignment + COLMAP export ────────────────────────────
    print(f"🎯 [5/5] exporting COLMAP -> {sparse_dir}")

    # Center scene at point-cloud centroid (adjusts camera translations too)
    if ALIGN and len(pts_d) > 0:
        centroid = pts_d.mean(axis=0)
        pts_d = pts_d - centroid
        # Adjust camera translations: t_new = R @ centroid + t (see docstring)
        for i in range(N):
            R = extrinsic[i, :3, :3]
            t = extrinsic[i, :3, 3]
            extrinsic[i, :3, 3] = R @ centroid + t
        print(f"  centered at origin (centroid was [{centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}])")

    # cameras.txt — one PINHOLE per image (VGGT-Omega's actual intrinsics)
    cameras = []
    for i in range(N):
        fx = float(intrinsic[i, 0, 0])
        fy = float(intrinsic[i, 1, 1])
        cx = float(intrinsic[i, 0, 2])
        cy = float(intrinsic[i, 1, 2])
        cameras.append((i + 1, "PINHOLE", W, H, [fx, fy, cx, cy]))
    write_cameras_txt(sparse_dir / "cameras.txt", cameras)

    # images.txt — w2c quaternion + translation (directly from extrinsic, no c2w->w2c)
    images_list = []
    for i, name in enumerate(frame_names):
        R = extrinsic[i, :3, :3]
        t = extrinsic[i, :3, 3]
        qw, qx, qy, qz = rotmat_to_quat(R)
        images_list.append((i + 1, qw, qx, qy, qz,
                             float(t[0]), float(t[1]), float(t[2]),
                             i + 1, name, []))
    write_images_txt(sparse_dir / "images.txt", images_list)

    # points3D.txt — filtered + downsampled + aligned
    points3D = [(i + 1, float(p[0]), float(p[1]), float(p[2]),
                 c[0], c[1], c[2], 0.0)
                for i, (p, c) in enumerate(zip(pts_d, cols_d))]
    write_points3D_txt(sparse_dir / "points3D.txt", points3D)

    extent = pts_d.max(0) - pts_d.min(0) if len(pts_d) else np.zeros(3)
    print(f"  ✅ cameras.txt  ({N} PINHOLE cameras)")
    print(f"  ✅ images.txt   ({N} w2c extrinsics)")
    print(f"  ✅ points3D.txt ({len(pts_d):,} init points; "
          f"extent=[{extent[0]:.2f}, {extent[1]:.2f}, {extent[2]:.2f}])")
    print(f"  ⏱️ {time.time() - t0:.1f}s")
    print(f"\n🎉 Done. COLMAP source: {source_dir}")
    print(f"  Next: bash vggt_human/04_train_3dgs.sh")


if __name__ == "__main__":
    main()
