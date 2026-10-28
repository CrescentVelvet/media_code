#!/usr/bin/env python3
"""Batch Hunyuan3D-2.1 inference over preprocessed object folders.

Reads PREP_DIR/<object>/views.json (written by assign_views.py):
  mode=mv     -> shape from dict of canonical-tagged RGBA views (2mv weights)
  mode=single -> shape from one RGBA view (2.1 weights)

Two phases so each model family loads once:
  1. all shapes (mv group then single group), free VRAM
  2. all paints (reference images = that object's rembg'd views)

Env vars: HY3D_DIR, SHAPE_DIR, MV_DIR, DINO_DIR, PREP_DIR, OUTPUT_DIR,
          SEED, NUM_INFERENCE_STEPS, GUIDANCE_SCALE, OCTREE_RESOLUTION,
          NUM_CHUNKS, TEX_MAX_NUM_VIEW, TEX_RESOLUTION, SKIP_TEXTURE,
          FLIP_LR (swap left/right tags), FORCE (redo finished objects)
"""
import gc
import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from PIL import Image

# bpy (Blender-as-module, ~330MB) is unavailable on China pypi mirrors; stub it
# so hy3dpaint's `import bpy` succeeds — the only bpy user is the GLB exporter,
# which we replace with the trimesh converter (_obj2glb) below.
try:
    import bpy  # noqa: F401
except ImportError:
    import types
    sys.modules["bpy"] = types.ModuleType("bpy")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _obj2glb import patch_hy3dpaint

HY3D_DIR = os.environ["HY3D_DIR"]
SHAPE_DIR = os.environ["SHAPE_DIR"]
MV_DIR = os.environ["MV_DIR"]
DINO_DIR = os.environ.get("DINO_DIR", "")
PREP_DIR = os.environ["PREP_DIR"]
OUTPUT_DIR = os.environ["OUTPUT_DIR"]

SEED = int(os.environ.get("SEED", "0"))
NUM_INFERENCE_STEPS = int(os.environ.get("NUM_INFERENCE_STEPS", "50"))
GUIDANCE_SCALE = float(os.environ.get("GUIDANCE_SCALE", "5.0"))
OCTREE_RESOLUTION = int(os.environ.get("OCTREE_RESOLUTION", "384"))
NUM_CHUNKS = int(os.environ.get("NUM_CHUNKS", "8000"))
TEX_MAX_NUM_VIEW = int(os.environ.get("TEX_MAX_NUM_VIEW", "6"))
TEX_RESOLUTION = int(os.environ.get("TEX_RESOLUTION", "512"))
SKIP_TEXTURE = os.environ.get("SKIP_TEXTURE", "0") == "1"
FLIP_LR = os.environ.get("FLIP_LR", "0") == "1"
FORCE = os.environ.get("FORCE", "0") == "1"

# The official code resolves cfg/ckpt paths relative to the repo root.
sys.path.insert(0, os.path.join(HY3D_DIR, "hy3dshape"))
sys.path.insert(0, os.path.join(HY3D_DIR, "hy3dpaint"))
os.chdir(HY3D_DIR)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_objects():
    objects = []
    for obj in sorted(os.listdir(PREP_DIR)):
        obj_dir = os.path.join(PREP_DIR, obj)
        views_json = os.path.join(obj_dir, "views.json")
        if not os.path.isfile(views_json):
            continue
        with open(views_json, encoding="utf-8") as fp:
            spec = json.load(fp)
        if FLIP_LR and spec.get("mode") == "mv":
            v = spec["views"]
            if "left" in v or "right" in v:
                spec["views"] = {
                    {"left": "right", "right": "left"}.get(k, k): p for k, p in v.items()
                }
        final_glb = os.path.join(OUTPUT_DIR, obj, obj + "_textured.glb")
        if os.path.exists(final_glb) and not FORCE:
            print(f"⏭️ {obj}: textured GLB exists, skip (FORCE=1 to redo)")
            continue
        spec["_dir"] = obj_dir
        spec["_name"] = obj
        objects.append(spec)
    return objects


def has_safetensors(model_dir, subfolder):
    sub = os.path.join(model_dir, subfolder)
    return os.path.isdir(sub) and any(f.endswith(".safetensors") for f in os.listdir(sub))


def load_shape_pipeline(mode):
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    if mode == "mv":
        model_dir, subfolder = MV_DIR, "hunyuan3d-dit-v2-mv"
    else:
        model_dir, subfolder = SHAPE_DIR, "hunyuan3d-dit-v2-1"
    use_st = has_safetensors(model_dir, subfolder)
    print(f"🏋️ [shape:{mode}] {model_dir} (subfolder={subfolder}, safetensors={use_st})")
    return Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        model_dir, subfolder=subfolder, use_safetensors=use_st, device="cuda",
    )


def gen_shape(pipeline, spec):
    """Return (mesh, ref_rgba_paths). image arg: tagged dict (mv) or path (single)."""
    obj_dir = spec["_dir"]
    if spec["mode"] == "mv":
        image_arg = {
            tag: os.path.join(obj_dir, rel) for tag, rel in spec["views"].items()
        }
        ref_paths = list(image_arg.values())
    else:
        image_arg = os.path.join(obj_dir, spec["image"])
        ref_paths = [image_arg]
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
    print(
        f"  ✅ {spec['_name']}: {time.time() - t0:.1f}s  "
        f"(verts={len(mesh.vertices)}, faces={len(mesh.faces)})"
    )
    return mesh, ref_paths


def run_shapes(objects):
    """Phase 1: generate + save all untextured meshes. Returns {name: (mesh_path, ref_paths)}."""
    results = {}
    for mode in ("mv", "single"):
        group = [s for s in objects if s["mode"] == mode]
        if not group:
            continue
        print(f"🚀 [shape] {len(group)} object(s) in mode={mode}")
        pipeline = load_shape_pipeline(mode)
        for spec in group:
            out_dir = os.path.join(OUTPUT_DIR, spec["_name"])
            os.makedirs(out_dir, exist_ok=True)
            mesh_path = os.path.join(out_dir, spec["_name"] + "_untextured.glb")
            try:
                mesh, ref_paths = gen_shape(pipeline, spec)
                mesh.export(mesh_path)
                results[spec["_name"]] = (mesh_path, ref_paths)
            except Exception as e:
                print(f"  ⚠️ {spec['_name']}: shape failed: {e}")
                continue
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()
    return results


def run_paints(results):
    """Phase 2: texture every mesh from phase 1 (paint model loads once)."""
    import textureGenPipeline
    patch_hy3dpaint(textureGenPipeline)
    Hunyuan3DPaintConfig = textureGenPipeline.Hunyuan3DPaintConfig
    Hunyuan3DPaintPipeline = textureGenPipeline.Hunyuan3DPaintPipeline

    config = Hunyuan3DPaintConfig(max_num_view=TEX_MAX_NUM_VIEW, resolution=TEX_RESOLUTION)
    config.multiview_pretrained_path = SHAPE_DIR
    if DINO_DIR:
        config.dino_ckpt_path = DINO_DIR
    config.multiview_cfg_path = os.path.join(HY3D_DIR, "hy3dpaint", "cfgs", "hunyuan-paint-pbr.yaml")
    config.realesrgan_ckpt_path = os.path.join(
        HY3D_DIR, "hy3dpaint", "ckpt", "RealESRGAN_x4plus.pth"
    )
    print(f"🏋️ [paint] max_num_view={TEX_MAX_NUM_VIEW}, resolution={TEX_RESOLUTION}")
    paint = Hunyuan3DPaintPipeline(config)

    for name, (mesh_path, ref_paths) in results.items():
        out_dir = os.path.dirname(mesh_path)
        out_obj = os.path.join(out_dir, name + "_textured.obj")
        try:
            refs = [Image.open(p).convert("RGBA") for p in ref_paths]
            t0 = time.time()
            paint(mesh_path=mesh_path, image_path=refs, output_mesh_path=out_obj, save_glb=True)
            print(f"  ✅ {name}: textured in {time.time() - t0:.1f}s -> {out_obj[:-4]}.glb")
        except Exception as e:
            print(f"  ⚠️ {name}: paint failed: {e}")
            continue


def main():
    objects = load_objects()
    if not objects:
        sys.exit("❌ no objects with views.json in " + PREP_DIR)
    print(f"🚀 [batch] {len(objects)} object(s), seed={SEED}, output={OUTPUT_DIR}")

    results = run_shapes(objects)
    if SKIP_TEXTURE:
        print("⏭️ SKIP_TEXTURE=1 — untextured GLBs only.")
        return
    run_paints(results)
    print("🎉 Done.")


if __name__ == "__main__":
    main()
