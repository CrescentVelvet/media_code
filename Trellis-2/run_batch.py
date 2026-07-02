#!/usr/bin/env python3
"""Batch TRELLIS.2 image-to-3D inference over a folder of images.

For each image in INPUT_DIR, run the image-to-3D pipeline and export:
  <stem>.glb           — PBR 3D asset (O-Voxel -> GLB via o_voxel.postprocess)
  <stem>.latent.npz     — cached latents (shape/tex slat + coords + res) so
                          03_render_video.sh can re-render WITHOUT re-running
                          generation (mirrors the official app.py pack/unpack).
  <stem>.mp4            — (only if RENDER_VIDEO=1) a quick shaded turntable.

The pipeline is loaded ONCE and reused across images (much faster than
relaunching python per image). Outputs are nested under
OUTPUT_DIR/<input_folder_name>/ so different INPUT_DIR runs don't clobber.

Env vars (set by 02_run_inference.sh):
  TRELLIS_DIR, MODEL_DIR, INPUT_DIR, OUTPUT_DIR, SEED, RESOLUTION,
  DECIMATION_TARGET, TEXTURE_SIZE, RENDER_VIDEO, ENVMAP,
  NUM_FRAMES, FPS, R, FOV, LOW_VRAM
"""
import os
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import sys
import time
from pathlib import Path

import numpy as np
import torch
import cv2
import imageio
from PIL import Image

TRELLIS_DIR = os.environ.get("TRELLIS_DIR", "../TRELLIS.2")
sys.path.insert(0, TRELLIS_DIR)  # so `from trellis2 import ...` resolves

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel

MODEL_DIR = os.environ.get("MODEL_DIR", "../../model/TRELLIS.2-4B")
INPUT_DIR = os.environ.get("INPUT_DIR") or os.path.join(TRELLIS_DIR, "assets", "example_image")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR") or os.path.join(TRELLIS_DIR, "output")
INPUT_NAME = os.path.basename(os.path.normpath(INPUT_DIR)) or "input"
OUTPUT_DIR = os.path.join(OUTPUT_DIR, INPUT_NAME)

SEED = int(os.environ.get("SEED", "0"))
RESOLUTION = os.environ.get("RESOLUTION", "1024")  # 512 / 1024 / 1536
DECIMATION_TARGET = int(os.environ.get("DECIMATION_TARGET", "1000000"))
TEXTURE_SIZE = int(os.environ.get("TEXTURE_SIZE", "4096"))
RENDER_VIDEO = os.environ.get("RENDER_VIDEO", "0") == "1"
ENVMAP = os.environ.get("ENVMAP", "forest")  # forest / sunset / courtyard / none
NUM_FRAMES = int(os.environ.get("NUM_FRAMES", "80"))
FPS = int(os.environ.get("FPS", "15"))
R = float(os.environ.get("R", "2"))
FOV = float(os.environ.get("FOV", "40"))
LOW_VRAM = os.environ.get("LOW_VRAM", "1") == "1"

# Pipeline type per resolution (matches app.py's mapping).
PIPELINE_TYPE = {"512": "512", "1024": "1024_cascade", "1536": "1536_cascade"}[RESOLUTION]

# Sampler params (app.py recommended defaults).
SS_PARAMS = {"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.7, "rescale_t": 5.0}
SHAPE_PARAMS = {"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.5, "rescale_t": 3.0}
TEX_PARAMS = {"steps": 12, "guidance_strength": 1.0, "guidance_rescale": 0.0, "rescale_t": 3.0}

os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR)
         if os.path.splitext(f)[1].lower() in IMG_EXTS]
files = sorted(set(files))
if not files:
    sys.exit(f"ERROR: no images in {INPUT_DIR}")


def load_envmap(name):
    if name.lower() == "none":
        return None
    path = os.path.join(TRELLIS_DIR, "assets", "hdri", f"{name}.exr")
    img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    return EnvMap(torch.tensor(img, dtype=torch.float32, device='cuda'))


def save_latent(npz_path, latents):
    """Pack the (shape_slat, tex_slat, res) tuple to a .npz (app.py pack_state)."""
    shape_slat, tex_slat, res = latents
    np.savez(
        npz_path,
        shape_slat_feats=shape_slat.feats.cpu().numpy(),
        tex_slat_feats=tex_slat.feats.cpu().numpy(),
        coords=shape_slat.coords.cpu().numpy(),
        res=np.array(res),
    )


envmap = load_envmap(ENVMAP) if RENDER_VIDEO else None

print(f"[*] loading pipeline from {MODEL_DIR} (device=cuda, low_vram={LOW_VRAM}) ...")
pipeline = Trellis2ImageTo3DPipeline.from_pretrained(MODEL_DIR)
pipeline.low_vram = LOW_VRAM
pipeline.cuda()
print("[*] pipeline ready")

print(f"[*] {len(files)} image(s): {INPUT_DIR} -> {OUTPUT_DIR}  "
      f"(resolution={RESOLUTION}, pipeline_type={PIPELINE_TYPE}, seed={SEED})")
ok = 0
gen_times = []
for i, f in enumerate(files, 1):
    stem = Path(f).stem
    glb_path = os.path.join(OUTPUT_DIR, f"{stem}.glb")
    lat_path = os.path.join(OUTPUT_DIR, f"{stem}.latent.npz")
    print(f"[{i}/{len(files)}] {os.path.basename(f)}  ->  {stem}.glb")
    try:
        image = Image.open(f)
        t0 = time.time()
        outputs, latents = pipeline.run(
            image,
            seed=SEED,
            sparse_structure_sampler_params=SS_PARAMS,
            shape_slat_sampler_params=SHAPE_PARAMS,
            tex_slat_sampler_params=TEX_PARAMS,
            pipeline_type=PIPELINE_TYPE,
            return_latent=True,
        )
        t_gen = time.time() - t0
        mesh = outputs[0]
        mesh.simplify(16777216)  # nvdiffrast limit (example.py)
        save_latent(lat_path, latents)
        torch.cuda.empty_cache()

        # Export GLB (O-Voxel -> textured GLB), same call as example.py.
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=DECIMATION_TARGET,
            texture_size=TEXTURE_SIZE,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            verbose=True,
        )
        glb.export(glb_path, extension_webp=True)
        ok += 1
        gen_times.append(t_gen)
        print(f"    generated in {t_gen:.2f}s  ->  {stem}.glb + {stem}.latent.npz")

        if RENDER_VIDEO:
            mp4_path = os.path.join(OUTPUT_DIR, f"{stem}.mp4")
            result = render_utils.render_video(
                mesh, resolution=1024, num_frames=NUM_FRAMES, r=R, fov=FOV, envmap=envmap
            )
            frames = result['shaded']
            imageio.mimsave(mp4_path, frames, fps=FPS)
            print(f"    rendered {len(frames)}f -> {stem}.mp4")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"    ! failed: {e}", file=sys.stderr)

print(f"[*] done. {ok}/{len(files)} succeeded. outputs in {OUTPUT_DIR}")
if gen_times:
    print(f"[*] generation time: avg {sum(gen_times)/len(gen_times):.2f}s, "
          f"min {min(gen_times):.2f}s, max {max(gen_times):.2f}s over {len(gen_times)} image(s)")
