#!/usr/bin/env python3
"""Render TRELLIS.2 .latent.npz caches to mp4 WITHOUT re-running generation.

For each .latent.npz in LATENT_INPUT (a file or a folder), decode the latents
via the pipeline (pipeline.decode_latent) and render a turntable video using
TRELLIS.2's PBR renderer (render_utils.render_video). This mirrors the official
example.py flow (decode -> simplify -> render_video -> mimsave) but operates on
cached latents produced by 02_run_inference.sh, so the 3-stage generation isn't
repeated — only the (fast) decode + render.

RENDER_MODE:
  shaded — a clean turntable of the PBR-shaded image (result['shaded']).
  pbr    — the official composite grid (shaded + normal + base_color + metallic
           + roughness + alpha), as produced by example.py's make_pbr_vis_frames.

Env vars (set by 03_render_video.sh):
  TRELLIS_DIR, MODEL_DIR, LATENT_INPUT, VIDEOS_DIR, INPUT_NAME,
  RENDER_MODE, RENDER_RES, NUM_FRAMES, FPS, R, FOV, ENVMAP, LOW_VRAM
"""
import os
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import sys
import time

import numpy as np
import torch
import cv2
import imageio

TRELLIS_DIR = os.environ.get("TRELLIS_DIR", "../TRELLIS.2")
sys.path.insert(0, TRELLIS_DIR)

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
from trellis2.modules.sparse import SparseTensor

MODEL_DIR = os.environ.get("MODEL_DIR", "../../model/TRELLIS.2-4B")
LATENT_INPUT = os.environ.get("LATENT_INPUT") or "output"
VIDEOS_DIR = os.environ.get("VIDEOS_DIR") or "videos"
INPUT_NAME = os.environ.get("INPUT_NAME") or os.path.basename(os.path.normpath(LATENT_INPUT)) or "input"
VIDEO_OUT_DIR = os.path.join(VIDEOS_DIR, INPUT_NAME)

RENDER_MODE = os.environ.get("RENDER_MODE", "shaded").lower()
RENDER_RES = int(os.environ.get("RENDER_RES", "1024"))
NUM_FRAMES = int(os.environ.get("NUM_FRAMES", "120"))
FPS = int(os.environ.get("FPS", "15"))
R = float(os.environ.get("R", "2"))
FOV = float(os.environ.get("FOV", "40"))
ENVMAP = os.environ.get("ENVMAP", "forest")
LOW_VRAM = os.environ.get("LOW_VRAM", "1") == "1"


def load_envmap(name):
    if name.lower() == "none":
        return None
    path = os.path.join(TRELLIS_DIR, "assets", "hdri", f"{name}.exr")
    img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    return EnvMap(torch.tensor(img, dtype=torch.float32, device='cuda'))


def unpack_state(npz_path):
    """Reconstruct (shape_slat, tex_slat, res) from a .latent.npz (app.py unpack_state)."""
    data = np.load(npz_path)
    shape_slat = SparseTensor(
        feats=torch.from_numpy(data['shape_slat_feats']).cuda(),
        coords=torch.from_numpy(data['coords']).cuda(),
    )
    tex_slat = shape_slat.replace(torch.from_numpy(data['tex_slat_feats']).cuda())
    return shape_slat, tex_slat, int(data['res'])


def main():
    p = LATENT_INPUT
    if os.path.isdir(p):
        npzs = sorted(f for f in os.listdir(p) if f.endswith(".latent.npz"))
        npzs = [os.path.join(p, f) for f in npzs]
    elif os.path.isfile(p) and p.endswith(".latent.npz"):
        npzs = [p]
    else:
        sys.exit(f"ERROR: LATENT_INPUT not a .latent.npz file or folder: {p}")
    if not npzs:
        sys.exit(f"ERROR: no .latent.npz in {p}")

    envmap = load_envmap(ENVMAP)
    print(f"[*] loading pipeline from {MODEL_DIR} (device=cuda, low_vram={LOW_VRAM}) ...")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(MODEL_DIR)
    pipeline.low_vram = LOW_VRAM
    pipeline.cuda()
    print("[*] pipeline ready")

    print(f"[*] {len(npzs)} latent(s) -> {VIDEO_OUT_DIR}/  "
          f"(mode={RENDER_MODE}, {NUM_FRAMES}f@{FPS}fps, r={R}, fov={FOV}, res={RENDER_RES}, envmap={ENVMAP})")
    os.makedirs(VIDEO_OUT_DIR, exist_ok=True)
    times = []
    for i, npz in enumerate(npzs, 1):
        # stem = filename with both ".latent.npz" stripped.
        base = os.path.basename(npz)
        stem = base[:-len(".latent.npz")] if base.endswith(".latent.npz") else os.path.splitext(base)[0]
        out_mp4 = os.path.join(VIDEO_OUT_DIR, f"{stem}.mp4")
        print(f"[{i}/{len(npzs)}] {base}  ->  {os.path.relpath(out_mp4)}")
        try:
            t0 = time.time()
            shape_slat, tex_slat, res = unpack_state(npz)
            mesh = pipeline.decode_latent(shape_slat, tex_slat, res)[0]
            mesh.simplify(16777216)  # nvdiffrast limit (example.py)
            torch.cuda.empty_cache()
            result = render_utils.render_video(
                mesh, resolution=RENDER_RES, num_frames=NUM_FRAMES, r=R, fov=FOV, envmap=envmap
            )
            if RENDER_MODE == "pbr":
                frames = render_utils.make_pbr_vis_frames(result, resolution=RENDER_RES)
            else:
                frames = result['shaded']
            imageio.mimsave(out_mp4, frames, fps=FPS)
            imageio.imwrite(out_mp4.replace(".mp4", ".png"), frames[0])
            dt = time.time() - t0
            times.append(dt)
            print(f"    rendered {len(frames)}f in {dt:.1f}s")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"    ! failed: {e}", file=sys.stderr)
    print(f"[*] done. videos in {VIDEO_OUT_DIR}")
    if times:
        print(f"[*] render time: avg {sum(times)/len(times):.1f}s, "
              f"min {min(times):.1f}s, max {max(times):.1f}s over {len(times)} latent(s)")


if __name__ == "__main__":
    main()
