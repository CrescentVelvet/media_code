#!/usr/bin/env python3
"""denoise_images.py — Stage 2: denoise rendered novel views + AdaIN + augmented COLMAP.

Reads novel_renders/*.png + novel_alpha/*.png + novel_poses.json from render_novel.py
(stage 1). For each novel view:
  1. Check alpha — if avg alpha < ALPHA_THRESH, the region is sparse (needs denoising).
  2. Denoise the rendered image (DENOISER=diffbir|swinir|nafnet|none).
  3. AdaIN color correction — match denoised image's color stats to the nearest
     training image (or global mean).
  4. Save corrected image.

Then writes an augmented COLMAP scene (SOURCE_AUG_DIR):
  - images/: original training images + denoised novel images
  - sparse/0/cameras.txt: original cameras + novel cameras (same intrinsics as nearest)
  - sparse/0/images.txt: original images + novel images (w2c quaternion + translation)
  - sparse/0/points3D.txt: copy of original (unchanged)

Env vars (set by 04_denoise_novel.sh):
  DENOISER, ALPHA_THRESH, ADAIN_REF, SOURCE_DIR, SOURCE_AUG_DIR, RESULTS_DIR, DEVICE
"""
import os
import sys
import shutil
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

DENOISER = os.environ.get("DENOISER", "none")
ALPHA_THRESH = float(os.environ.get("ALPHA_THRESH", "0.3"))
ADAIN_REF = os.environ.get("ADAIN_REF", "nearest")
SOURCE_DIR = os.environ.get("SOURCE_DIR", "")
SOURCE_AUG_DIR = os.environ.get("SOURCE_AUG_DIR", "")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "")
DEVICE = os.environ.get("DEVICE", "cuda")

# Import COLMAP writers + parsers from sibling scripts.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from npz_to_colmap import write_cameras_txt, write_images_txt, write_points3D_txt, rotmat_to_quat  # noqa: E402
from render_novel import parse_cameras_txt, parse_images_txt  # noqa: E402
from denoisers import get_denoiser  # noqa: E402


# ---------------------------------------------------------------------------
# AdaIN color transfer.
# ---------------------------------------------------------------------------
def adain_color_transfer(source, target):
    """Transfer color statistics (mean, std) from target to source.

    source: denoised image (H, W, 3) in [0, 1]
    target: reference image (H, W, 3) in [0, 1]
    """
    s_mean = source.reshape(-1, 3).mean(axis=0)
    s_std = source.reshape(-1, 3).std(axis=0)
    t_mean = target.reshape(-1, 3).mean(axis=0)
    t_std = target.reshape(-1, 3).std(axis=0)
    result = (source - s_mean) / (s_std + 1e-6) * t_std + t_mean
    return np.clip(result, 0, 1)


def load_image_as_float(path):
    """Load image as (H, W, 3) float in [0, 1]."""
    img = Image.open(path).convert("RGB")
    return np.array(img).astype(np.float32) / 255.0


def save_float_as_png(image, path):
    """Save (H, W, 3) float [0,1] as PNG."""
    Image.fromarray((image * 255).clip(0, 255).astype(np.uint8)).save(path)


# ---------------------------------------------------------------------------
# Main pipeline.
# ---------------------------------------------------------------------------
def main():
    if not SOURCE_DIR:
        sys.exit("❌ SOURCE_DIR not set")
    if not SOURCE_AUG_DIR:
        sys.exit("❌ SOURCE_AUG_DIR not set")
    if not RESULTS_DIR:
        sys.exit("❌ RESULTS_DIR not set")

    poses_path = os.path.join(RESULTS_DIR, "novel_poses.json")
    renders_dir = os.path.join(RESULTS_DIR, "novel_renders")
    alpha_dir = os.path.join(RESULTS_DIR, "novel_alpha")

    if not os.path.isfile(poses_path):
        sys.exit(f"❌ {poses_path} not found — run render_novel.py first")
    with open(poses_path) as f:
        novel_data = json.load(f)
    novel_cameras = novel_data["novel_cameras"]

    # Parse original COLMAP scene
    sparse_dir = os.path.join(SOURCE_DIR, "sparse", "0")
    cameras_info = parse_cameras_txt(os.path.join(sparse_dir, "cameras.txt"))
    images_info = parse_images_txt(os.path.join(sparse_dir, "images.txt"))

    t0 = time.time()
    print(f"🚀 [stage 2] denoise + AdaIN + augmented COLMAP")
    print(f"  🤖 denoiser:     {DENOISER}")
    print(f"  📐 alpha_thresh: {ALPHA_THRESH}")
    print(f"  🎨 adain_ref:    {ADAIN_REF}")
    print(f"  📂 novel views:  {len(novel_cameras)}")
    print("")

    # Load denoiser (lazy)
    denoiser = get_denoiser(DENOISER) if DENOISER != "none" else None

    # Prepare augmented scene directories
    aug_images_dir = Path(SOURCE_AUG_DIR) / "images"
    aug_sparse_dir = Path(SOURCE_AUG_DIR) / "sparse" / "0"
    aug_images_dir.mkdir(parents=True, exist_ok=True)
    aug_sparse_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy original images
    print("📂 copying original training images...")
    orig_images_dir = os.path.join(SOURCE_DIR, "images")
    for name in os.listdir(orig_images_dir):
        if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            shutil.copy(os.path.join(orig_images_dir, name), str(aug_images_dir / name))
    print(f"  ✅ copied {len(os.listdir(aug_images_dir))} images")

    # 2. Denoise + AdaIN novel views
    novel_cameras_used = []  # only views that passed the alpha threshold
    print(f"\n✂️ denoising novel views (alpha < {ALPHA_THRESH} = sparse, needs denoising)...")
    for i, nv in enumerate(novel_cameras):
        render_path = os.path.join(renders_dir, nv["name"])
        alpha_path = os.path.join(alpha_dir, nv["name"])
        if not os.path.isfile(render_path) or not os.path.isfile(alpha_path):
            print(f"  [{i+1}] ⚠️ {nv['name']} — render or alpha not found, skipping")
            continue

        alpha = load_image_as_float(alpha_path)
        avg_alpha = float(alpha.mean())
        if avg_alpha >= ALPHA_THRESH:
            print(f"  [{i+1}] ⏭️ {nv['name']}  alpha={avg_alpha:.3f} ≥ {ALPHA_THRESH} (well-covered, skip)")
            continue

        # Load rendered image
        rendered = load_image_as_float(render_path)

        # Denoise
        if denoiser is not None:
            print(f"  [{i+1}] 🧹 {nv['name']}  alpha={avg_alpha:.3f} — denoising with {DENOISER}...")
            denoised = denoiser(rendered, device=DEVICE)
        else:
            print(f"  [{i+1}] ⏭️ {nv['name']}  alpha={avg_alpha:.3f} — DENOISER=none, no denoising")
            denoised = rendered

        # AdaIN color correction
        if ADAIN_REF == "nearest":
            nearest_idx = nv.get("nearest_idx", 0)
            if nearest_idx < len(images_info):
                ref_name = images_info[nearest_idx][4]
                ref_path = os.path.join(SOURCE_DIR, "images", ref_name)
                if os.path.isfile(ref_path):
                    ref_img = load_image_as_float(ref_path)
                    corrected = adain_color_transfer(denoised, ref_img)
                    print(f"       🎨 AdaIN → nearest: {ref_name}")
                else:
                    corrected = denoised
                    print(f"       ⚠️ nearest ref not found: {ref_path}, skip AdaIN")
            else:
                corrected = denoised
        elif ADAIN_REF == "mean":
            # Use mean color of all training images
            t_mean, t_std = _compute_mean_color_stats(SOURCE_DIR, images_info, max_samples=20)
            s_mean = denoised.reshape(-1, 3).mean(axis=0)
            s_std = denoised.reshape(-1, 3).std(axis=0)
            corrected = np.clip((denoised - s_mean) / (s_std + 1e-6) * t_std + t_mean, 0, 1)
            print(f"       🎨 AdaIN → mean color")
        else:
            corrected = denoised

        # Save to augmented images dir
        save_float_as_png(corrected, str(aug_images_dir / nv["name"]))
        novel_cameras_used.append(nv)

    print(f"\n  ✅ {len(novel_cameras_used)}/{len(novel_cameras)} novel views added")

    # 3. Write augmented COLMAP scene
    print("\n📝 writing augmented COLMAP scene...")

    # cameras.txt: original + novel (each novel camera = same intrinsics as nearest)
    aug_cameras = []
    max_cam_id = max(cameras_info.keys())
    for cam_id in sorted(cameras_info.keys()):
        model, W, H, params = cameras_info[cam_id]
        aug_cameras.append((cam_id, model, W, H, params))
    for nv in novel_cameras_used:
        max_cam_id += 1
        nv["aug_cam_id"] = max_cam_id
        aug_cameras.append((max_cam_id, "PINHOLE", nv["W"], nv["H"],
                            [nv["fx"], nv["fy"], nv["cx"], nv["cy"]]))
    write_cameras_txt(aug_sparse_dir / "cameras.txt", aug_cameras)

    # images.txt: original + novel
    aug_images = []
    max_img_id = max(img[0] for img in images_info)
    for img_id, qw, qx, qy, qz, tx, ty, tz, cam_id, name, pts2d in [
        (img[0], *_parse_image_full(img)) for img in images_info
    ]:
        aug_images.append((img_id, qw, qx, qy, qz, tx, ty, tz, cam_id, name, []))
    for nv in novel_cameras_used:
        max_img_id += 1
        R = np.array(nv["R"])
        T = np.array(nv["T"])
        qw, qx, qy, qz = rotmat_to_quat(R)
        aug_images.append((max_img_id, qw, qx, qy, qz,
                           float(T[0]), float(T[1]), float(T[2]),
                           nv["aug_cam_id"], nv["name"], []))
    write_images_txt(aug_sparse_dir / "images.txt", aug_images)

    # points3D.txt: copy original
    shutil.copy(os.path.join(sparse_dir, "points3D.txt"),
                str(aug_sparse_dir / "points3D.txt"))

    print(f"  ✅ cameras.txt  ({len(aug_cameras)} cameras = {len(cameras_info)} orig + {len(novel_cameras_used)} novel)")
    print(f"  ✅ images.txt   ({len(aug_images)} images)")
    print(f"  ✅ points3D.txt (copied)")
    print(f"  ⏱️ {time.time() - t0:.1f}s")
    print(f"\n🎉 Done. Augmented COLMAP: {SOURCE_AUG_DIR}")
    print(f"  Next: bash vggt_human/05_train_denoise.sh")


def _parse_image_full(img_tuple):
    """Convert images_info tuple to (qw, qx, qy, qz, tx, ty, tz, cam_id, name, pts2d)."""
    _, quat, t, cam_id, name = img_tuple
    return quat[0], quat[1], quat[2], quat[3], t[0], t[1], t[2], cam_id, name, []


def _compute_mean_color_stats(source_dir, images_info, max_samples=20):
    """Compute mean color statistics across training images (sampled)."""
    import random
    means, stds = [], []
    sampled = random.sample(images_info, min(max_samples, len(images_info)))
    for _, _, _, _, name in sampled:
        path = os.path.join(source_dir, "images", name)
        if os.path.isfile(path):
            img = load_image_as_float(path)
            means.append(img.reshape(-1, 3).mean(axis=0))
            stds.append(img.reshape(-1, 3).std(axis=0))
    if not means:
        return np.array([0.5, 0.5, 0.5]), np.array([0.2, 0.2, 0.2])
    return np.mean(means, axis=0), np.mean(stds, axis=0)


if __name__ == "__main__":
    main()
