#!/usr/bin/env python3
"""colmap_ba.py — COLMAP Bundle Adjustment with VGGT poses (fixed intrinsics).

PREVIOUS APPROACH (colmap_ba_with_features.py) WAS WRONG:
  It ran full incremental_mapping (from-scratch SfM) with CameraMode.AUTO,
  which let COLMAP estimate its own intrinsics (fx/fy=2304 via heuristic
  max(W,H)*1.2) and its own poses — completely ignoring VGGT predictions.
  Result: fx/fy=2304 (vs VGGT 1389), different coordinate system, different
  poses. This was NOT "fix intrinsics + refine poses" — it was a full SfM
  replacement of VGGT.

CORRECT APPROACH (this script):
  VGGT reconstruction already has: cameras (PINHOLE, VGGT-predicted intrinsics)
  + images (with VGGT-predicted poses) + points3D (VGGT point cloud, 7163 pts).
  BUT: images have num_points3D=0 (no 2D-3D observations) because VGGT directly
  predicts poses+pointcloud without feature matching.

  pycolmap.triangulate_points(reconstruction, database, images, output):
    - Takes the VGGT reconstruction (with known poses) as INPUT
    - Runs feature extraction + matching on the images
    - Triangulates new 3D points from matched features using VGGT poses
    - Creates 2D-3D observations (POINTS2D in images.txt)
    - Runs BA to refine poses + 3D points (intrinsics fixed via options)

  This is the correct "fix intrinsics, refine extrinsics + 3D points" that
  the user asked for. VGGT poses are the initial; BA refines them using
  reprojection error from real feature observations.

Env vars:
  SOURCE_DIR     : VGGT COLMAP scene (sparse/0 in text format)
  IMAGES_DIR     : original images (default: $SOURCE_DIR/images)
  OUTPUT_DIR     : BA-refined output (default: sibling of SOURCE_DIR, _ba suffix)
  BA_VERBOSE     : 1 = print progress
"""
import os
import sys
import shutil
from pathlib import Path

import numpy as np
import pycolmap

SOURCE_DIR_ENV = os.environ.get("SOURCE_DIR", "")
IMAGES_DIR_ENV = os.environ.get("IMAGES_DIR", "")
OUTPUT_DIR_ENV = os.environ.get("OUTPUT_DIR", "")
VERBOSE = os.environ.get("BA_VERBOSE", "0") == "1"


def main():
    source_dir = SOURCE_DIR_ENV
    images_dir = IMAGES_DIR_ENV
    output_dir = OUTPUT_DIR_ENV

    if not source_dir:
        sys.exit("SOURCE_DIR not set (VGGT COLMAP scene with sparse/0/)")
    if not images_dir:
        images_dir = os.path.join(source_dir, "images")
        print(f"  IMAGES_DIR not set, defaulting to {images_dir}")
    if not output_dir:
        # Default: $RESULTS_DIR/source_ba (sibling of source)
        parent = os.path.dirname(source_dir.rstrip("/\\"))
        output_dir = os.path.join(parent, "source_ba")
        print(f"  OUTPUT_DIR not set, defaulting to {output_dir}")

    sparse_in = Path(source_dir) / "sparse" / "0"
    if not sparse_in.exists():
        # Try sparse/ directly (some layouts)
        sparse_in = Path(source_dir) / "sparse"
        if not sparse_in.exists():
            sys.exit(f"❌ no sparse/0 or sparse/ in {source_dir}")

    out_dir = Path(output_dir)
    db_path = out_dir / "database.db"
    sparse_out = out_dir / "sparse" / "0"

    print(f"📂 source:   {sparse_in}")
    print(f"📂 images:   {images_dir}")
    print(f"💾 output:   {out_dir}")
    print(f"🔒 intrinsics: FIXED (VGGT-predicted, no refinement)")
    print(f"🎯 refine:   extrinsics (poses) + 3D points only")
    print()

    out_dir.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    shutil.rmtree(sparse_out, ignore_errors=True)
    sparse_out.mkdir(parents=True, exist_ok=True)

    # ── 1. Load VGGT reconstruction (text format) ──────────────────────
    print("📥 [1/4] loading VGGT reconstruction...")
    rec = pycolmap.Reconstruction(str(sparse_in))
    n_cams = len(rec.cameras)
    n_imgs = len(rec.images)
    n_pts = len(rec.points3D)
    print(f"  cameras: {n_cams}, images: {n_imgs}, points3D: {n_pts}")

    # Verify VGGT poses are present
    posed = sum(1 for img in rec.images.values() if img.has_pose)
    print(f"  images with pose: {posed}/{n_imgs}")
    if posed == 0:
        sys.exit("❌ no images have poses — VGGT reconstruction invalid")

    # Verify intrinsics (show first camera)
    for cid, cam in list(rec.cameras.items())[:1]:
        print(f"  sample camera {cid}: model={cam.model_name}, params={cam.params}")

    # ── 2. Create database + import images + extract features + match ────
    # triangulate_points needs a database with features+matches.
    # We must run feature extraction and matching ourselves first.
    print("\n🔍 [2/4] feature extraction + matching...")
    with pycolmap.Database.open(str(db_path)) as db:
        pass

    # Import images (SINGLE camera mode = reuse VGGT intrinsics per-image)
    # Actually: CameraMode.AUTO creates one camera per image, which matches
    # VGGT's layout (each image has its own camera with slightly different fx/fy).
    reader_opts = pycolmap.ImageReaderOptions()
    reader_opts.camera_model = "PINHOLE"
    pycolmap.import_images(
        database_path=str(db_path),
        image_path=images_dir,
        camera_mode=pycolmap.CameraMode.AUTO,
        options=reader_opts,
    )

    # IMPORTANT: import_images creates NEW cameras with COLMAP heuristic
    # intrinsics (fx/fy = max(W,H)*1.2). We must OVERWRITE them with VGGT
    # intrinsics so feature extraction + triangulation use VGGT values.
    with pycolmap.Database.open(str(db_path)) as db:
        db_imgs = {img.name: img for img in db.read_all_images()}

    # Build name -> VGGT camera mapping
    vgg_cams_by_name = {}
    for img in rec.images.values():
        vgg_cams_by_name[img.name] = rec.cameras[img.camera_id]

    # Overwrite each db camera with VGGT intrinsics (match by image name)
    updated = 0
    with pycolmap.Database.open(str(db_path)) as db:
        for img_name, db_img in db_imgs.items():
            vgg_cam = vgg_cams_by_name.get(img_name)
            if vgg_cam is None:
                print(f"  ⚠️ no VGGT match for {img_name}, skipping")
                continue
            db_cam = db.read_camera(db_img.camera_id)
            db_cam.params = list(vgg_cam.params)
            db_cam.width = vgg_cam.width
            db_cam.height = vgg_cam.height
            # model stays PINHOLE (both VGGT and COLMAP default use PINHOLE)
            db.update_camera(db_cam)
            updated += 1
    print(f"  updated {updated}/{len(db_imgs)} db cameras with VGGT intrinsics")

    # Also overwrite image poses in db with VGGT poses (so triangulate_points
    # uses VGGT poses as initial). Actually: triangulate_points takes the
    # reconstruction object (which already has VGGT poses). The db is only
    # for features/matches. So we don't need to write poses to db.

    # Feature extraction
    fopts = pycolmap.FeatureExtractionOptions()
    pycolmap.extract_features(
        database_path=str(db_path),
        image_path=images_dir,
        extraction_options=fopts,
    )

    # Exhaustive matching
    mopts = pycolmap.FeatureMatchingOptions()
    popts = pycolmap.ExhaustivePairingOptions()
    pycolmap.match_exhaustive(
        database_path=str(db_path),
        matching_options=mopts,
        pairing_options=popts,
    )

    with pycolmap.Database.open(str(db_path)) as db:
        n_matches = db.num_verified_image_pairs()
    print(f"  matched pairs: {n_matches}")

    # ── 3. Triangulate points using VGGT poses + run BA ─────────────────
    print("\n🏗️ [3/4] triangulate_points (VGGT poses → observations → BA)...")
    iopts = pycolmap.IncrementalPipelineOptions()
    iopts.ba_refine_focal_length = False       # fix intrinsics (VGGT provides)
    iopts.ba_refine_principal_point = False
    iopts.ba_refine_extra_params = False

    # triangulate_points: takes reconstruction (with VGGT poses), runs feature
    # matching triangulation, creates observations, runs BA.
    # It returns a NEW reconstruction (refined).
    rec_ba = pycolmap.triangulate_points(
        reconstruction=rec,
        database_path=str(db_path),
        image_path=images_dir,
        output_path=str(sparse_out),
        clear_points=True,
        options=iopts,
        refine_intrinsics=False,   # fix intrinsics
    )

    n_imgs_ba = len(rec_ba.images)
    n_pts_ba = len(rec_ba.points3D)
    # Count observations
    total_obs = sum(img.num_points3D for img in rec_ba.images.values())
    mean_obs = total_obs / max(n_imgs_ba, 1)
    print(f"  BA result: {n_imgs_ba} images, {n_pts_ba} 3D points")
    print(f"  observations: {total_obs} total, {mean_obs:.1f} per image")

    # Mean reprojection error
    try:
        reproj = rec_ba.compute_mean_reprojection_error()
        print(f"  mean reprojection error: {reproj:.4f}")
    except Exception as e:
        print(f"  (reproj error unavailable: {e})")

    # ── 4. Write output (text format for 3DGS) ─────────────────────────
    print(f"\n💾 [4/4] writing output to {sparse_out}...")
    # triangulate_points already wrote binary to sparse_out via output_path.
    # Also write text format (3DGS needs text).
    text_out = out_dir / "sparse" / "0_text"
    text_out.mkdir(parents=True, exist_ok=True)
    rec_ba.write_text(str(text_out))
    print(f"  binary: {sparse_out}")
    print(f"  text:   {text_out}")

    # Copy/symlink images
    out_images = out_dir / "images"
    if not out_images.exists():
        try:
            out_images.symlink_to(images_dir)
        except (OSError, NotImplementedError):
            shutil.copytree(images_dir, out_images)

    print("\n✅ COLMAP BA complete (VGGT poses refined, intrinsics fixed).")
    print(f"  → Use with 04_train_3dgs.sh: SOURCE_DIR={output_dir}")


if __name__ == "__main__":
    main()
