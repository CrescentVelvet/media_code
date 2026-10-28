#!/usr/bin/env python3
"""Hunyuan3D-2.1 image-to-3D inference: single image or multi-view -> PBR GLB.

Two stages, loaded SEQUENTIALLY so peak VRAM stays at the texture stage's
~21GB instead of ~29GB combined (fits one 24GB card):
  1. shape  — Hunyuan3DDiTFlowMatchingPipeline (hy3dshape)
       MODE=single: subfolder hunyuan3d-dit-v2-1 (3.3B, 2.1 weights)
       MODE=mv:     subfolder hunyuan3d-dit-v2-mv (2.0-gen multi-view weights;
                    2.1 has no MV variant, but the 2.1 codebase runs it via
                    image=dict{front,left,back,right})
  2. paint  — Hunyuan3DPaintPipeline (hy3dpaint): multiview PBR diffusion +
              RealESRGAN enhance + bake -> textured obj + glb.

Env vars (set by 02_run_inference.sh):
  HY3D_DIR, SHAPE_DIR, MV_DIR, DINO_DIR, MODE, IMAGE, FRONT/BACK/LEFT/RIGHT,
  OUTPUT_DIR, OUTPUT_NAME, SEED, NUM_INFERENCE_STEPS, GUIDANCE_SCALE,
  OCTREE_RESOLUTION, NUM_CHUNKS, TEX_MAX_NUM_VIEW, TEX_RESOLUTION, SKIP_TEXTURE
"""
import gc
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from PIL import Image

HY3D_DIR = os.environ["HY3D_DIR"]
SHAPE_DIR = os.environ["SHAPE_DIR"]            # local snapshot of tencent/Hunyuan3D-2.1
MV_DIR = os.environ.get("MV_DIR", "")          # local snapshot of tencent/Hunyuan3D-2mv
DINO_DIR = os.environ.get("DINO_DIR", "")      # local snapshot of facebook/dinov2-giant

MODE = os.environ.get("MODE", "single")        # single | mv
IMAGE = os.environ.get("IMAGE", "")
VIEWS = {k: os.environ.get(k, "") for k in ("FRONT", "BACK", "LEFT", "RIGHT")}
VIEWS = {k.lower(): v for k, v in VIEWS.items() if v}

OUTPUT_DIR = os.environ["OUTPUT_DIR"]
OUTPUT_NAME = os.environ.get("OUTPUT_NAME", "")
SEED = int(os.environ.get("SEED", "0"))
NUM_INFERENCE_STEPS = int(os.environ.get("NUM_INFERENCE_STEPS", "50"))
GUIDANCE_SCALE = float(os.environ.get("GUIDANCE_SCALE", "5.0"))
OCTREE_RESOLUTION = int(os.environ.get("OCTREE_RESOLUTION", "384"))
NUM_CHUNKS = int(os.environ.get("NUM_CHUNKS", "8000"))
TEX_MAX_NUM_VIEW = int(os.environ.get("TEX_MAX_NUM_VIEW", "6"))
TEX_RESOLUTION = int(os.environ.get("TEX_RESOLUTION", "512"))
SKIP_TEXTURE = os.environ.get("SKIP_TEXTURE", "0") == "1"

# The official code imports its own packages by bare name and resolves some
# config/ckpt paths relative to the repo root — so cwd must be the repo root.
sys.path.insert(0, os.path.join(HY3D_DIR, "hy3dshape"))
sys.path.insert(0, os.path.join(HY3D_DIR, "hy3dpaint"))
os.chdir(HY3D_DIR)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_image_input():
    """Return (pipeline_image_arg, ref_image_paths, stem)."""
    if MODE == "mv":
        if not VIEWS:
            sys.exit("❌ MODE=mv needs at least one of FRONT/BACK/LEFT/RIGHT")
        arg = {k: v for k, v in VIEWS.items()}
        stem = OUTPUT_NAME or "mv_model"
        return arg, list(VIEWS.values()), stem
    if not IMAGE:
        sys.exit("❌ MODE=single needs IMAGE=<path>")
    stem = OUTPUT_NAME or os.path.splitext(os.path.basename(IMAGE))[0]
    return IMAGE, [IMAGE], stem


def has_safetensors(model_dir, subfolder):
    sub = os.path.join(model_dir, subfolder)
    return os.path.isdir(sub) and any(f.endswith(".safetensors") for f in os.listdir(sub))


def run_shape(image_arg, stem):
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    if MODE == "mv":
        if not MV_DIR:
            sys.exit("❌ MODE=mv needs MV_DIR (local tencent/Hunyuan3D-2mv snapshot)")
        model_dir, subfolder = MV_DIR, "hunyuan3d-dit-v2-mv"
    else:
        model_dir, subfolder = SHAPE_DIR, "hunyuan3d-dit-v2-1"
    use_st = has_safetensors(model_dir, subfolder)

    print(f"🏋️ [shape] {model_dir} (subfolder={subfolder}, safetensors={use_st})")
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        model_dir, subfolder=subfolder, use_safetensors=use_st, device="cuda",
    )
    t0 = time.time()
    mesh = pipeline(
        image=image_arg,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        octree_resolution=OCTREE_RESOLUTION,
        num_chunks=NUM_CHUNKS,
        generator=torch.manual_seed(SEED),
        output_type="trimesh",
    )[0]
    print(f"⏱️ [shape] {time.time() - t0:.1f}s  (verts={len(mesh.vertices)}, faces={len(mesh.faces)})")

    mesh_path = os.path.join(OUTPUT_DIR, stem + "_untextured.glb")
    mesh.export(mesh_path)
    print(f"✅ [shape] saved: {mesh_path}")

    # Free shape weights BEFORE loading the paint stage (~10GB) — this is what
    # makes a 24GB card survive the full pipeline.
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    return mesh_path


def run_paint(mesh_path, ref_image_paths, stem):
    from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

    config = Hunyuan3DPaintConfig(max_num_view=TEX_MAX_NUM_VIEW, resolution=TEX_RESOLUTION)
    # Point every path at local files — defaults are HF repo ids / repo-relative
    # paths, which re-download or break outside the official layout.
    config.multiview_pretrained_path = SHAPE_DIR  # snapshot contains hunyuan3d-paint-v2-1
    if DINO_DIR:
        config.dino_ckpt_path = DINO_DIR
    config.multiview_cfg_path = os.path.join(HY3D_DIR, "hy3dpaint", "cfgs", "hunyuan-paint-pbr.yaml")
    config.realesrgan_ckpt_path = os.path.join(HY3D_DIR, "hy3dpaint", "ckpt", "RealESRGAN_x4plus.pth")

    print(f"🏋️ [paint] max_num_view={TEX_MAX_NUM_VIEW}, resolution={TEX_RESOLUTION}")
    paint = Hunyuan3DPaintPipeline(config)

    ref_images = [Image.open(p).convert("RGB") for p in ref_image_paths]
    out_obj = os.path.join(OUTPUT_DIR, stem + "_textured.obj")
    t0 = time.time()
    paint(mesh_path=mesh_path, image_path=ref_images, output_mesh_path=out_obj, save_glb=True)
    print(f"⏱️ [paint] {time.time() - t0:.1f}s")
    out_glb = out_obj.replace(".obj", ".glb")
    print(f"✅ [paint] saved: {out_glb}")


def main():
    image_arg, ref_image_paths, stem = build_image_input()
    print(f"🚀 [inference] MODE={MODE}  views={len(ref_image_paths)}  seed={SEED}")
    print(f"  📁 repo:   {HY3D_DIR}")
    print(f"  💾 output: {OUTPUT_DIR}")

    mesh_path = run_shape(image_arg, stem)
    if SKIP_TEXTURE:
        print("⏭️ SKIP_TEXTURE=1 — untextured GLB only.")
        return
    run_paint(mesh_path, ref_image_paths, stem)
    print("🎉 Done.")


if __name__ == "__main__":
    main()
