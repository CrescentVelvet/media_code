#!/usr/bin/env python3
"""colmap_ba.py — COLMAP Bundle Adjustment on step 03's output.

Fixes intrinsics (refine_focal_length=False, refine_principal_point=False,
refine_extra_params=False), only refines extrinsics (poses) + 3D points.
This is the safe alternative to pose_refine (which can't get gradients
through diff_gaussian_rasterization's CUDA rasterizer).

Env vars:
  SOURCE_DIR    : COLMAP sparse model dir (input, e.g. .../source/sparse/0)
  OUTPUT_DIR     : output dir for refined model (e.g. .../source_ba/sparse/0)
  BA_VERBOSE     : 1 = print Ceres solver progress (default 0)
"""
import os
import sys
import shutil
from pathlib import Path

import pycolmap

SOURCE_DIR = os.environ.get("SOURCE_DIR", "")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "")
VERBOSE = os.environ.get("BA_VERBOSE", "0") == "1"


def main():
    if not SOURCE_DIR:
        sys.exit("SOURCE_DIR not set")
    if not OUTPUT_DIR:
        sys.exit("OUTPUT_DIR not set")

    src = Path(SOURCE_DIR)
    out = Path(OUTPUT_DIR)
    print(f"📂 input:  {src}")
    print(f"💾 output: {out}")

    if not (src / "cameras.txt").exists():
        sys.exit(f"cameras.txt not found in {src}")

    # Load existing reconstruction
    print("\n📥 loading COLMAP reconstruction...")
    rec = pycolmap.Reconstruction(str(src))
    n_cams = len(rec.cameras)
    n_imgs = len(reg := rec.reg_images())
    n_pts = len(rec.points3D)
    print(f"  cameras: {n_cams}, registered images: {n_imgs}, 3D points: {n_pts}")

    # Snapshot pre-BA stats for comparison
    pre_errors = []
    for img in reg.values():
        if img.num_points3D > 0:
            pre_errors.append(img.reprojection_error if hasattr(img, 'reprojection_error') else None)
    pre_mean_reproj = rec.compute_mean_reprojection_error() if hasattr(rec, 'compute_mean_reprojection_error') else None
    print(f"  pre-BA mean reprojection error: {pre_mean_reproj}")

    # Configure BA: fix intrinsics, only refine poses + 3D points
    print("\n⚙️ configuring bundle adjustment (intrinsics FIXED)...")
    options = pycolmap.BundleAdjustmentOptions()
    options.refine_focal_length = False      # fix fx, fy
    options.refine_principal_point = False   # fix cx, cy
    options.refine_extra_params = False       # fix distortion (none for PINHOLE anyway)
    options.refine_extrinsics = True if hasattr(options, 'refine_extrinsics') else None
    options.verbose = VERBOSE

    config = pycolmap.BundleAdjustmentConfig()
    for img_id in rec.reg_image_ids():
        config.add_image(img_id)

    # Snapshot pre-BA poses for diff
    pre_poses = {}
    for img_id in rec.reg_image_ids():
        img = rec.images[img_id]
        pre_poses[img_id] = (img.cam_from_world.rotation.quat, img.cam_from_world.translation)

    # Run BA
    print("\n🔧 running bundle adjustment...")
    ba = pycolmap.create_default_ceres_bundle_adjuster(options, config, rec)
    summary = ba.solve()

    # Post-BA stats
    post_mean_reproj = rec.compute_mean_reprojection_error() if hasattr(rec, 'compute_mean_reprojection_error') else None
    print(f"  post-BA mean reprojection error: {post_mean_reproj}")
    if pre_mean_reproj and post_mean_reproj is not None:
        change = (pre_mean_reproj - post_mean_reproj) / pre_mean_reproj * 100
        print(f"  improvement: {change:.2f}%")

    # Compute pose changes
    pose_diffs = []
    for img_id in rec.reg_image_ids():
        img = rec.images[img_id]
        post_q = img.cam_from_world.rotation.quat
        post_t = img.cam_from_world.translation
        pre_q, pre_t = pre_poses[img_id]
        # quaternion angular distance (degrees)
        dot = abs(sum(a*b for a, b in zip(pre_q, post_q)))
        dot = min(1.0, max(-1.0, dot))
        angle_deg = 2 * __import__('math').degrees(__import__('math').acos(dot))
        # translation distance
        t_dist = sum((a-b)**2 for a, b in zip(pre_t, post_t)) ** 0.5
        pose_diffs.append((angle_deg, t_dist))

    angles = [d[0] for d in pose_diffs]
    t_dists = [d[1] for d in pose_diffs]
    print(f"\n  pose changes (rot deg, trans units):")
    print(f"    rotation: mean={sum(angles)/len(angles):.4f}°, max={max(angles):.4f}°")
    print(f"    translation: mean={sum(t_dists)/len(t_dists):.6f}, max={max(t_dists):.6f}")

    # Write refined model
    out.mkdir(parents=True, exist_ok=True)
    rec.write(str(out))
    print(f"\n💾 wrote refined model to {out}")

    # Copy images dir (BA doesn't touch images, just link/copy for completeness)
    src_images = src.parent / "images"
    out_images = out.parent / "images"
    if src_images.exists() and not out_images.exists():
        # Symlink to save space; fallback to copy
        try:
            out_images.symlink_to(src_images)
            print(f"  symlinked images -> {src_images}")
        except (OSError, NotImplementedError):
            shutil.copytree(src_images, out_images)
            print(f"  copied images -> {out_images}")

    print("\n✅ BA complete.")


if __name__ == "__main__":
    main()
