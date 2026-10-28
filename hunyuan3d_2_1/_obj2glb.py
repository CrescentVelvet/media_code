#!/usr/bin/env python3
"""trimesh-based OBJ -> GLB converter replacing hy3dpaint's bpy-based one.

The official convert_obj_to_glb uses Blender's bpy module (a ~330MB wheel that
China pypi mirrors don't carry). The OBJ written by mesh_utils.save_obj_mesh
follows a fixed layout:
  <base>.obj + <base>.mtl
  <base>.png|jpg           diffuse/albedo
  <base>_metallic.png      grayscale
  <base>_roughness.png     grayscale
  <base>_normal.png        (optional)

glTF wants metallic+roughness packed into ONE texture (G=roughness, B=metallic),
so we merge them here.
"""
import os

import numpy as np
import trimesh
from PIL import Image


def _find_texture(base, suffix=""):
    for ext in (".png", ".jpg", ".jpeg"):
        p = base + suffix + ext
        if os.path.exists(p):
            return p
    return None


def convert_obj_to_glb_trimesh(obj_path, glb_path):
    base = obj_path[:-4] if obj_path.lower().endswith(".obj") else obj_path
    mesh = trimesh.load(obj_path, process=False)

    albedo_path = _find_texture(base)
    metallic_path = _find_texture(base, "_metallic")
    roughness_path = _find_texture(base, "_roughness")
    normal_path = _find_texture(base, "_normal")

    from trimesh.visual.material import PBRMaterial

    albedo = Image.open(albedo_path).convert("RGB") if albedo_path else None

    mr_img = None
    if metallic_path and roughness_path:
        m = Image.open(metallic_path).convert("L")
        r = Image.open(roughness_path).convert("L")
        if m.size != r.size:
            r = r.resize(m.size)
        zero = Image.fromarray(np.zeros_like(np.asarray(m)))
        mr_img = Image.merge("RGB", (zero, r, m))  # G=roughness, B=metallic (glTF)

    normal_img = Image.open(normal_path).convert("RGB") if normal_path else None

    material = PBRMaterial(
        baseColorTexture=albedo,
        metallicRoughnessTexture=mr_img,
        normalTexture=normal_img,
        metallicFactor=1.0 if mr_img is not None else 0.0,
        roughnessFactor=1.0,
    )

    # trimesh loads the obj's uv into TextureVisuals; swap in our PBR material.
    visual = getattr(mesh, "visual", None)
    if visual is not None and hasattr(visual, "material"):
        visual.material = material
    mesh.export(glb_path)
    return glb_path


def patch_hy3dpaint(texture_gen_module):
    """Redirect textureGenPipeline.convert_obj_to_glb to the trimesh version."""
    texture_gen_module.convert_obj_to_glb = convert_obj_to_glb_trimesh
