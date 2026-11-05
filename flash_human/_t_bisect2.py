#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""最小化二分渲染:只渲第 4 帧,开关定位"渲染变紫"根因。

环境变量开关(置 1 生效):
  SKIP_CS     不做贴图 colorspace 赋值
  SKIP_DITHER 不把含 opacity 贴图的材质改 DITHERED
  SKIP_COMP   不建 compositor(不落盘 depth/normal)
  SKIP_JITTER 不做灯光/曝光随机化
  FILM_OPAQUE 胶片不透明(便于对照)

用法同 01_render_gt.py(--fbx --texdir --out --mode --n_views --res --seed)。
"""
import bpy
import math
import os
import random
import sys

import mathutils

argv = sys.argv[sys.argv.index("--") + 1:]


def _arg(name, default):
    key = "--" + name
    if key in argv:
        i = argv.index(key)
        if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            return argv[i + 1]
    return default


FBX_PATH = _arg("fbx", "")
TEXDIR = _arg("texdir", "")
OUT_DIR = _arg("out", "")
MODE = _arg("mode", "full")
RES = int(_arg("res", "512"))
SEED = int(_arg("seed", "0"))
LIGHT_JITTER = 0.3
TARGET_ZFRAC = 0.55 if MODE == "full" else 0.85
Focal_MM = 50.0

random.seed(SEED)
os.makedirs(os.path.join(OUT_DIR, "rgb"), exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
try:
    bpy.ops.preferences.addon_disable(module="cycles")
except Exception:
    pass
bpy.ops.import_scene.fbx(filepath=FBX_PATH, use_anim=False,
                         ignore_leaf_bones=True,
                         automatic_bone_orientation=False)
imported = [o for o in bpy.context.scene.objects if o.type == "MESH"]
for o in list(bpy.context.scene.objects):
    if o.type in ("LIGHT", "CAMERA"):
        bpy.data.objects.remove(o, do_unlink=True)

# 贴图重连 + colorspace(开关)
TEX_KEYWORDS = {
    "sRGB": ("basecolor", "base_color", "diffuse", "albedo", "color"),
    "Non-Color": ("normal", "roughness", "specular", "metallic",
                  "opacity", "ao", "height", "rough"),
}


def _colorspace_for(fname):
    f = fname.lower()
    for cs, keys in TEX_KEYWORDS.items():
        if any(k in f for k in keys):
            return cs
    return "sRGB"


tex_index = {}
if TEXDIR and os.path.isdir(TEXDIR):
    for root, _, files in os.walk(TEXDIR):
        for f in files:
            tex_index.setdefault(f.lower(), os.path.join(root, f))

for mat in bpy.data.materials:
    if not mat.use_nodes or not mat.node_tree:
        continue
    for node in mat.node_tree.nodes:
        if node.type != "TEX_IMAGE" or node.image is None:
            continue
        img = node.image
        if img.source == "FILE":
            if not os.path.exists(bpy.path.abspath(img.filepath)):
                cand = tex_index.get(os.path.basename(img.filepath).lower())
                if cand:
                    img.filepath = cand
        if not os.environ.get("SKIP_CS"):
            img.colorspace_settings.name = _colorspace_for(img.name)
    try:
        has_alpha_tex = (not os.environ.get("SKIP_DITHER")) and any(
            n.type == "TEX_IMAGE" and n.image and "opacity" in n.image.name.lower()
            for n in mat.node_tree.nodes)
        if has_alpha_tex:
            mat.surface_render_method = "DITHERED"
    except (AttributeError, TypeError):
        pass

# 归一化(同 01)
bpy.context.view_layer.update()
bb_min = mathutils.Vector((1e9, 1e9, 1e9))
bb_max = mathutils.Vector((-1e9, -1e9, -1e9))
for o in imported:
    for c in o.bound_box:
        w = o.matrix_world @ mathutils.Vector(c)
        bb_min = mathutils.Vector(map(min, bb_min, w))
        bb_max = mathutils.Vector(map(max, bb_max, w))
height = bb_max.z - bb_min.z
scale = 1.70 / height
center_xy = (bb_min + bb_max) / 2.0
for o in imported:
    o.matrix_world = (
        mathutils.Matrix.Translation((-center_xy.x * scale,
                                      -center_xy.y * scale,
                                      -bb_min.z * scale))
        @ mathutils.Matrix.Scale(scale, 4)
        @ o.matrix_world)
bpy.context.view_layer.update()
bb_min = mathutils.Vector((1e9, 1e9, 1e9))
bb_max = mathutils.Vector((-1e9, -1e9, -1e9))
for o in imported:
    for c in o.bound_box:
        w = o.matrix_world @ mathutils.Vector(c)
        bb_min = mathutils.Vector(map(min, bb_min, w))
        bb_max = mathutils.Vector(map(max, bb_max, w))
bbox_h = bb_max.z - bb_min.z
target = mathutils.Vector(((bb_min.x + bb_max.x) / 2,
                           (bb_min.y + bb_max.y) / 2,
                           bb_min.z + bbox_h * TARGET_ZFRAC))

# 灯光/世界(同 01)
world = bpy.data.worlds.new("W")
bpy.context.scene.world = world
world.use_nodes = True
wn = world.node_tree.nodes
wl = world.node_tree.links
wn.clear()
w_out = wn.new("ShaderNodeOutputWorld")
w_bg = wn.new("ShaderNodeBackground")
w_env = wn.new("ShaderNodeTexEnvironment")
w_bg.inputs["Strength"].default_value = 0.8
wl.new(w_env.outputs["Color"], w_bg.inputs["Color"])
wl.new(w_bg.outputs["Background"], w_out.inputs["Surface"])
key = bpy.data.lights.new("Key", "SUN"); key.energy = 4.0
key_obj = bpy.data.objects.new("Key", key)
bpy.context.scene.collection.objects.link(key_obj)
key_obj.rotation_euler = (math.radians(50), 0, math.radians(30))
fill = bpy.data.lights.new("Fill", "SUN"); fill.energy = 2.0
fill_obj = bpy.data.objects.new("Fill", fill)
bpy.context.scene.collection.objects.link(fill_obj)
fill_obj.rotation_euler = (math.radians(60), 0, math.radians(-140))
rim = bpy.data.lights.new("Rim", "SUN"); rim.energy = 3.0
rim_obj = bpy.data.objects.new("Rim", rim)
bpy.context.scene.collection.objects.link(rim_obj)
rim_obj.rotation_euler = (math.radians(65), 0, math.radians(180))


def jitter_lights():
    key_obj.rotation_euler = (
        math.radians(50 + random.uniform(-1, 1) * 25 * LIGHT_JITTER), 0,
        math.radians(30 + random.uniform(-1, 1) * 120 * LIGHT_JITTER))
    key.energy = 4.0 * (1 + random.uniform(-1, 1) * 0.5 * LIGHT_JITTER)
    fill_obj.rotation_euler = (
        math.radians(60 + random.uniform(-1, 1) * 20 * LIGHT_JITTER), 0,
        math.radians(-140 + random.uniform(-1, 1) * 120 * LIGHT_JITTER))
    fill.energy = 2.0 * (1 + random.uniform(-1, 1) * 0.5 * LIGHT_JITTER)
    rim_obj.rotation_euler = (
        math.radians(65 + random.uniform(-1, 1) * 20 * LIGHT_JITTER), 0,
        math.radians(180 + random.uniform(-1, 1) * 60 * LIGHT_JITTER))
    rim.energy = 3.0 * (1 + random.uniform(-1, 1) * 0.5 * LIGHT_JITTER)
    w_bg.inputs["Strength"].default_value = 0.8 * (
        1 + random.uniform(-1, 1) * 0.6 * LIGHT_JITTER)
    bpy.context.scene.view_settings.exposure = random.uniform(
        -1, 1) * 0.5 * LIGHT_JITTER


scene = bpy.context.scene
engine_ids = [e.identifier for e in
              bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]
scene.render.engine = ("BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engine_ids
                       else "BLENDER_EEVEE")
scene.render.resolution_x = scene.render.resolution_y = RES
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.render.film_transparent = not bool(os.environ.get("FILM_OPAQUE"))
scene.view_settings.look = "AgX - Medium High Contrast"

vl = scene.view_layers[0]
if not os.environ.get("SKIP_PASSES"):
    vl.use_pass_z = True
    vl.use_pass_normal = True

# compositor(开关)
if not os.environ.get("SKIP_COMP"):
    _comp = bpy.data.node_groups.new("CompGT", "CompositorNodeTree")
    scene.compositing_node_group = _comp
    _rl = _comp.nodes.new("CompositorNodeRLayers")

    def _mk_file_output(subdir, socket_type, item_name, link_output):
        node = _comp.nodes.new("CompositorNodeOutputFile")
        node.directory = os.path.join(OUT_DIR, subdir)
        node.format.file_format = "OPEN_EXR_MULTILAYER"
        node.format.color_depth = "32"
        node.file_name = item_name + "_####"
        item = node.file_output_items.new(socket_type, item_name)
        _comp.links.new(_rl.outputs[link_output], node.inputs[item_name])

    os.makedirs(os.path.join(OUT_DIR, "depth"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "normal"), exist_ok=True)
    _mk_file_output("depth", "FLOAT", "depth", "Depth")
    _mk_file_output("normal", "VECTOR", "normal", "Normal")

# 相机 + 第 4 帧(az=270 el=10,与 6 视角分布一致)
cam_data = bpy.data.cameras.new("Cam")
cam_data.lens = Focal_MM
cam_data.sensor_width = 36.0
cam = bpy.data.objects.new("Cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam
tan_half_vfov = 36.0 / (2.0 * Focal_MM)
dist = bbox_h * 0.72 / tan_half_vfov * 1.15
azim = math.radians(270.0)
elev = math.radians(10.0)
r_xy = dist * math.cos(elev)
loc = mathutils.Vector((target.x + r_xy * math.cos(azim),
                        target.y + r_xy * math.sin(azim),
                        target.z + dist * math.sin(elev)))

if not os.environ.get("SKIP_JITTER"):
    jitter_lights()
cam.location = loc
cam.rotation_euler = (target - loc).to_track_quat("-Z", "Y").to_euler()
bpy.context.view_layer.update()
scene.frame_set(4)
bpy.ops.render.render(write_still=False)
rgb_path = os.path.join(OUT_DIR, "rgb", "000004.png")
bpy.data.images["Render Result"].save_render(rgb_path, scene=scene)
print(f"[done] -> {rgb_path}")
