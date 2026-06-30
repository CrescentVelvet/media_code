#!/usr/bin/env python3
"""Batch VGGT-Omega reconstruction over scenes (folders of images or videos).

For each scene in INPUT_DIR, run camera+depth inference and produce:
  - scene.ply       : merged, confidence-filtered colored point cloud (raw
                      world coords; view in MeshLab/SuperSplat or feed to 03).
  - predictions.npz : raw model outputs (depth, depth_conf, extrinsic,
                       intrinsic, world_points_from_depth, images, pose_enc) —
                       same keys the official demo_gradio saves.
  - scene.glb       : official visualization (point cloud + camera frustums),
                       built via visual_util.predictions_to_glb  [needs trimesh]
  - frames/         : the images actually fed to the model (copied/extracted),
                       so predictions.npz is self-contained.

The model is loaded ONCE and reused across scenes. INPUT_DIR may be:
  - a folder of images        -> one scene (named after the folder)
  - a video file (.mp4/.mov)  -> one scene (frames extracted at VIDEO_FPS)
  - a folder of scene folders -> one reconstruction per subfolder (batch)
  - a folder of videos        -> one reconstruction per video (batch)

Env vars (set by 02_run_inference.sh):
  VGGT_DIR, MODEL_DIR, INPUT_DIR, OUTPUT_DIR, VARIANT, RESOLUTION, MODE,
  CONF_THRES, MAX_POINTS, MASK_SKY, MASK_BLACK_BG, MASK_WHITE_BG, VIDEO_FPS, DEVICE
"""
import glob
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

VGGT_DIR = os.environ.get("VGGT_DIR", "../vggt-omega")
sys.path.insert(0, VGGT_DIR)  # so `from vggt_omega import ...` resolves

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

# Optional: official visualization (point cloud + cameras as .glb).
try:
    from visual_util import predictions_to_glb  # at repo root
    _HAVE_VIS = True
except Exception:
    _HAVE_VIS = False
try:
    import trimesh
    _HAVE_TRIMESH = True
except Exception:
    _HAVE_TRIMESH = False

MODEL_DIR = os.environ.get("MODEL_DIR") or os.path.join(VGGT_DIR, "..", "model", "VGGT-Omega")
INPUT_DIR = os.environ.get("INPUT_DIR") or os.path.join(VGGT_DIR, "examples")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR") or os.path.join(VGGT_DIR, "output")
VARIANT = os.environ.get("VARIANT", "1b_512")
RESOLUTION = int(os.environ.get("RESOLUTION", "512"))
MODE = os.environ.get("MODE", "balanced")
CONF_THRES = float(os.environ.get("CONF_THRES", "20"))
MAX_POINTS = int(os.environ.get("MAX_POINTS", "2000000"))
MASK_SKY = os.environ.get("MASK_SKY", "0") == "1"
MASK_BLACK_BG = os.environ.get("MASK_BLACK_BG", "0") == "1"
MASK_WHITE_BG = os.environ.get("MASK_WHITE_BG", "0") == "1"
VIDEO_FPS = float(os.environ.get("VIDEO_FPS", "1"))
DEVICE = os.environ.get("DEVICE", "cuda")

IMG_EXTS = {".webp", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
VID_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


def ckpt_path():
    if VARIANT == "1b_512":
        f, align = "vggt_omega_1b_512.pt", False
    elif VARIANT in ("1b_256_text", "text"):
        f, align = "vggt_omega_1b_256_text.pt", True
    else:
        sys.exit(f"ERROR: unknown VARIANT '{VARIANT}' (use 1b_512 | 1b_256_text)")
    p = os.path.join(MODEL_DIR, f)
    if not os.path.isfile(p):
        sys.exit(f"ERROR: checkpoint not found: {p}\nRun 01_download_models.sh VARIANT={VARIANT} first.")
    return p, align


def extract_frames(video_path, out_dir, fps):
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    step = max(int(round(src_fps / max(fps, 0.1))), 1)
    idx, saved = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            cv2.imwrite(os.path.join(out_dir, f"{saved:06}.png"), frame)
            saved += 1
        idx += 1
    cap.release()
    return saved


def unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic):
    """Port of demo_gradio.unproject_depth_map_to_point_map (numpy)."""
    depth = depth_map[..., 0]
    num_frames, height, width = depth.shape
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = np.broadcast_to(x[None], (num_frames, height, width))
    y = np.broadcast_to(y[None], (num_frames, height, width))
    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]
    camera_points = np.stack([
        (x - cx) / fx * depth,
        (y - cy) / fy * depth,
        depth,
    ], axis=-1)
    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    return np.einsum(
        "sij,shwj->shwi",
        np.transpose(rotation, (0, 2, 1)),
        camera_points - translation[:, None, None, :],
    )


def images_to_rgb(images):
    if images.ndim == 4 and images.shape[1] == 3:
        return np.transpose(images, (0, 2, 3, 1))
    return images


def build_predictions_np(predictions, extrinsic, intrinsic, world_points):
    """Assemble the dict the official visual_util.predictions_to_glb expects.
    Mirrors demo_gradio.run_model: add extrinsic/intrinsic, squeeze batch dim,
    then append world_points_from_depth."""
    d = dict(predictions)
    d["extrinsic"] = extrinsic
    d["intrinsic"] = intrinsic
    out = {}
    for key, value in d.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
            if value.ndim >= 1 and value.shape[0] == 1:
                value = value[0]
        out[key] = value
    out["world_points_from_depth"] = world_points
    return out


def save_ply(path, vertices, colors):
    """Write a plain RGB point cloud .ply (xyz + uchar rgb), binary little-endian."""
    n = len(vertices)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n".encode())
        f.write(b"property float x\nproperty float y\nproperty float z\n")
        f.write(b"property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(b"end_header\n")
        rec = np.empty(n, dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)])
        rec["xyz"] = np.ascontiguousarray(vertices, dtype="<f4")
        rec["rgb"] = np.ascontiguousarray(colors, dtype="u1")
        f.write(rec.tobytes())


def filter_points(points, conf, images_arr, depth):
    """Confidence + finite + depth-edge + optional bg filter (mirrors visual_util)."""
    vertices = points.reshape(-1, 3)
    rgb = (images_to_rgb(images_arr).reshape(-1, 3) * 255).clip(0, 255).astype(np.uint8)
    conf_flat = conf.reshape(-1).copy()
    mask = np.isfinite(vertices).all(axis=1) & np.isfinite(conf_flat)

    # depth-edge filter (drops depth-boundary artifacts); best-effort.
    try:
        from visual_util import depth_edge
        edge = depth_edge(depth[..., 0]).reshape(-1)
        conf_flat[edge] = 0.0
    except Exception:
        pass

    conf_thres = max(2.0, CONF_THRES)
    if np.any(mask):
        thr = np.percentile(conf_flat[mask], conf_thres)
        mask &= conf_flat >= thr
    mask &= conf_flat > 1e-5
    if MASK_BLACK_BG:
        mask &= rgb.sum(axis=1) >= 16
    if MASK_WHITE_BG:
        mask &= ~((rgb[:, 0] > 240) & (rgb[:, 1] > 240) & (rgb[:, 2] > 240))

    vertices = vertices[mask]
    rgb = rgb[mask]
    if MAX_POINTS > 0 and len(vertices) > MAX_POINTS:
        idx = np.linspace(0, len(vertices) - 1, MAX_POINTS).astype(np.int64)
        vertices = vertices[idx]
        rgb = rgb[idx]
    return vertices, rgb


def discover_scenes(input_dir):
    """Return [(scene_name, path), ...]. path is a folder (images) or a video."""
    if os.path.isfile(input_dir) and input_dir.lower().endswith(tuple(VID_EXTS)):
        return [(Path(input_dir).stem, input_dir)]
    if not os.path.isdir(input_dir):
        sys.exit(f"ERROR: INPUT_DIR not a file or folder: {input_dir}")
    direct_imgs = [f for f in os.listdir(input_dir)
                   if os.path.splitext(f)[1].lower() in IMG_EXTS]
    if direct_imgs:
        return [(Path(input_dir).name or "scene", input_dir)]
    vids = [f for f in os.listdir(input_dir)
            if os.path.splitext(f)[1].lower() in VID_EXTS]
    if vids:
        return [(Path(v).stem, os.path.join(input_dir, v)) for v in sorted(vids)]
    scenes = []
    for name in sorted(os.listdir(input_dir)):
        sub = os.path.join(input_dir, name)
        if not os.path.isdir(sub):
            continue
        has_img = any(os.path.splitext(f)[1].lower() in IMG_EXTS
                      for f in os.listdir(sub)) or os.path.isdir(os.path.join(sub, "images"))
        if has_img:
            scenes.append((name, sub))
    return scenes


def gather_images(scene_path):
    """Image paths for a scene folder (directly, or under images/)."""
    if os.path.isfile(scene_path):
        return None
    direct = [p for p in glob.glob(os.path.join(scene_path, "*"))
              if os.path.splitext(p)[1].lower() in IMG_EXTS]
    if direct:
        return sorted(direct)
    imgs_dir = os.path.join(scene_path, "images")
    if os.path.isdir(imgs_dir):
        g = [p for p in glob.glob(os.path.join(imgs_dir, "*"))
             if os.path.splitext(p)[1].lower() in IMG_EXTS]
        return sorted(g)
    return []


def run_scene(model, align, scene_name, scene_path, out_root):
    out_dir = os.path.join(out_root, scene_name)
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Gather images: copy/extract into frames/ so predictions.npz is self-contained.
    if os.path.isfile(scene_path) and scene_path.lower().endswith(tuple(VID_EXTS)):
        n = extract_frames(scene_path, frames_dir, VIDEO_FPS)
        image_names = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
        src_label = f"video ({n} frames @ {VIDEO_FPS}fps)"
    else:
        imgs = gather_images(scene_path)
        if not imgs:
            print(f"    ! no images found for scene '{scene_name}'; skipping", file=sys.stderr)
            return False
        for p in imgs:
            shutil.copy(p, os.path.join(frames_dir, os.path.basename(p)))
        image_names = [p for p in sorted(glob.glob(os.path.join(frames_dir, "*")))
                       if os.path.splitext(p)[1].lower() in IMG_EXTS]
        src_label = f"{len(image_names)} images"

    if not image_names:
        print(f"    ! no images for scene '{scene_name}'; skipping", file=sys.stderr)
        return False

    print(f"  scene '{scene_name}': {src_label}, res={RESOLUTION} ({MODE})")
    t0 = time.time()
    images = load_and_preprocess_images(image_names, mode=MODE, image_resolution=RESOLUTION).to(DEVICE)
    with torch.inference_mode():
        predictions = model(images)
    extrinsic, intrinsic = encoding_to_camera(predictions["pose_enc"], predictions["images"].shape[-2:])
    depth_np = predictions["depth"][0].float().cpu().numpy()
    extrinsic_np = extrinsic.squeeze(0).float().cpu().numpy()  # (N,3,4)
    intrinsic_np = intrinsic.squeeze(0).float().cpu().numpy()  # (N,3,3)
    world_points = unproject_depth_map_to_point_map(depth_np, extrinsic_np, intrinsic_np)
    pred_np = build_predictions_np(predictions, extrinsic, intrinsic, world_points)
    t_inf = time.time() - t0

    np.savez(os.path.join(out_dir, "predictions.npz"), **pred_np)

    verts, cols = filter_points(
        pred_np["world_points_from_depth"], pred_np["depth_conf"],
        pred_np["images"], pred_np["depth"],
    )
    if len(verts) == 0:
        verts = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        cols = np.array([[255, 255, 255]], dtype=np.uint8)
    save_ply(os.path.join(out_dir, "scene.ply"), verts.astype(np.float32), cols)
    extent = verts.max(0) - verts.min(0)
    print(f"    {len(verts):,} pts -> scene.ply  extent={[round(v, 2) for v in extent.tolist()]}  (inf {t_inf:.1f}s)")

    if _HAVE_VIS and _HAVE_TRIMESH:
        try:
            t1 = time.time()
            glb_max = min(MAX_POINTS, 1000000) if MAX_POINTS > 0 else 1000000
            scene = predictions_to_glb(
                pred_np, conf_thres=CONF_THRES,
                mask_black_bg=MASK_BLACK_BG, mask_white_bg=MASK_WHITE_BG,
                show_cam=True, mask_sky=MASK_SKY, max_points=glb_max,
            )
            scene.export(os.path.join(out_dir, "scene.glb"))
            print(f"    scene.glb  (vis {time.time() - t1:.1f}s)")
        except Exception as e:
            print(f"    ! scene.glb export failed: {e}", file=sys.stderr)
    else:
        print("    (scene.glb skipped: trimesh/scipy not installed — pip install trimesh scipy matplotlib)")
    return True


def main():
    ckpt, align = ckpt_path()
    print(f"[*] loading VGGT-Omega ({VARIANT}) from {ckpt}  device={DEVICE}  align={align}")
    model = VGGTOmega(enable_alignment=align).to(DEVICE).eval()
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    if align:
        print("[*] text-alignment head enabled (use RESOLUTION=256)")
    print("[*] model ready")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    scenes = discover_scenes(INPUT_DIR)
    if not scenes:
        sys.exit(f"ERROR: no scenes found under {INPUT_DIR}")
    print(f"[*] {len(scenes)} scene(s) under {INPUT_DIR} -> {OUTPUT_DIR}")

    ok = 0
    for i, (name, path) in enumerate(scenes, 1):
        print(f"[{i}/{len(scenes)}] {name}")
        try:
            if run_scene(model, align, name, path, OUTPUT_DIR):
                ok += 1
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            print(f"    ! GPU OOM — reduce RESOLUTION ({RESOLUTION}) or fewer frames; {e}", file=sys.stderr)
        except Exception as e:
            print(f"    ! failed: {e}", file=sys.stderr)
    print(f"[*] done. {ok}/{len(scenes)} scene(s) reconstructed. outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
