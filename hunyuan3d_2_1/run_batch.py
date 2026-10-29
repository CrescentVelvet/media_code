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
          FLIP_LR (swap left/right tags), FORCE (redo finished objects),
          REUSE_SHAPE (reuse existing *_untextured.glb),
          PAINT_TARGET_FACES (paint remesh face target, 0 = no decimation)
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
# 2mv 权重的 config.yaml target 指向 hy3dgen.shapegen.*（2.0 代仓库布局），
# 2.1 仓的 hy3dshape 没有对应模块——mv 管线必须从 Hunyuan3D-2 仓加载。
HY3D2_DIR = os.environ["HY3D2_DIR"]
SHAPE_DIR = os.environ["SHAPE_DIR"]
MV_DIR = os.environ["MV_DIR"]
DINO_DIR = os.environ.get("DINO_DIR", "")
PREP_DIR = os.environ["PREP_DIR"]
OUTPUT_DIR = os.environ["OUTPUT_DIR"]

# torchvision>=0.20 移除了 functional_tensor，basicsr/realesrgan import 会炸；
# 官方仓根目录自带补丁（demo.py 同款 apply_fix），必须在 realesrgan 导入前应用。
sys.path.insert(0, HY3D_DIR)
from torchvision_fix import apply_fix  # noqa: E402

apply_fix()

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
# 已有 *_untextured.glb 的对象直接复用进 paint 阶段（同种子 shape 可复现，
# 调 paint 参数时不用每次重算 13 分钟形状）
REUSE_SHAPE = os.environ.get("REUSE_SHAPE", "0") == "1"
# paint 阶段 remesh_mesh 硬编码减面到 4 万面（simplify_mesh_utils.py 默认
# target_count=40000），是成品"糙"的直接原因；用 env 覆盖目标面数。
# 16 万面 ≈ 保留 shape 输出的近一半细节，UV 展开/光栅化时间仍可控。
PAINT_TARGET_FACES = int(os.environ.get("PAINT_TARGET_FACES", "40000"))

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
    if mode == "mv":
        # hy3dgen（2.0 仓）放最前，避免被 hy3dshape 同名模块遮蔽
        sys.path.insert(0, HY3D2_DIR)
        from hy3dgen.shapegen.pipelines import Hunyuan3DDiTFlowMatchingPipeline

        model_dir, subfolder = MV_DIR, "hunyuan3d-dit-v2-mv"
    else:
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

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
    if REUSE_SHAPE:
        todo = []
        for spec in objects:
            out_dir = os.path.join(OUTPUT_DIR, spec["_name"])
            mesh_path = os.path.join(out_dir, spec["_name"] + "_untextured.glb")
            if os.path.exists(mesh_path):
                if spec["mode"] == "mv":
                    ref_paths = [os.path.join(spec["_dir"], rel) for rel in spec["views"].values()]
                else:
                    ref_paths = [os.path.join(spec["_dir"], spec["image"])]
                results[spec["_name"]] = (mesh_path, ref_paths)
                print(f"⏭️ {spec['_name']}: reuse existing shape")
            else:
                todo.append(spec)
        objects = todo
        if not objects:
            return results
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
    import huggingface_hub
    import textureGenPipeline
    patch_hy3dpaint(textureGenPipeline)

    # multiview_utils.py 对 multiview_pretrained_path 调 snapshot_download(repo_id=...)
    # 要的是 HF repo id；我们给的是本地目录，HF CDN 又不通——本地路径直接放行。
    _orig_snapshot = huggingface_hub.snapshot_download

    def _snapshot_or_local(repo_id, **kw):
        if os.path.isdir(repo_id):
            return repo_id
        return _orig_snapshot(repo_id, **kw)

    huggingface_hub.snapshot_download = _snapshot_or_local

    # trimesh 4.4.7 的 simplify_quadric_decimation 只走 open3d 后端；open3d
    # wheel ~440MB 国内镜像下载易断。改用已装的 pymeshlab 实现同等
    # quadric edge collapse，monkeypatch 掉 trimesh 方法。
    import tempfile

    import pymeshlab
    import trimesh

    def _simplify_quadric_pymeshlab(self, face_count=None, **kw):
        with tempfile.TemporaryDirectory() as td:
            in_ply = os.path.join(td, "in.ply")
            out_obj = os.path.join(td, "out.obj")
            self.export(in_ply)
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(in_ply)
            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=int(face_count))
            ms.save_current_mesh(out_obj, save_textures=False)
            return trimesh.load(out_obj, force="mesh")

    trimesh.Trimesh.simplify_quadric_decimation = _simplify_quadric_pymeshlab

    # paint 的 remesh 目标面数硬编码 40000（见文件头 env 说明）：包一层注入
    # PAINT_TARGET_FACES，传 0 表示完全不减面（直接用 shape 原始网格）。
    if PAINT_TARGET_FACES != 40000:
        from utils import simplify_mesh_utils as _smu

        _orig_simplify = _smu.mesh_simplify_trimesh

        def _simplify_override(inputpath, outputpath, target_count=40000):
            return _orig_simplify(
                inputpath, outputpath,
                target_count=PAINT_TARGET_FACES if PAINT_TARGET_FACES > 0 else 10**9,
            )

        _smu.mesh_simplify_trimesh = _simplify_override

    # 上游 bug（textureGenPipeline.__call__ ~L95）：image_path 传 list 时
    # image_prompt 两个分支都不命中 → UnboundLocalError。运行时给源码打补丁：
    # list 分支直接 image_prompt = image_path。官方仓不动，重 clone 也安全。
    # 注意 __call__ 被 @torch.no_grad() 包装，要 unwrap 拿到原函数的 globals。
    import inspect
    import textwrap

    _cls = textureGenPipeline.Hunyuan3DPaintPipeline
    _orig_fn = inspect.unwrap(_cls.__call__)
    _src = inspect.getsource(_orig_fn)
    if "isinstance(image_path, list)" not in _src:
        _src = _src.replace(
            "if isinstance(image_path, str):",
            "if isinstance(image_path, list):\n            image_prompt = image_path\n        elif isinstance(image_path, str):",
            1,
        )
        _ns = {}
        exec(textwrap.dedent(_src), _orig_fn.__globals__, _ns)
        _cls.__call__ = torch.no_grad()(_ns["__call__"])
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
        except Exception:
            import traceback
            print(f"  ⚠️ {name}: paint failed:")
            traceback.print_exc()
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
