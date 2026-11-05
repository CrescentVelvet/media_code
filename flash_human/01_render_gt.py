#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""01_render_gt.py — flash_human GT 数据工厂(Blender 后台渲染)

用法(在本机 Git Bash 或 cmd 均可):
  "<BLENDER>/blender.exe" -b --python 01_render_gt.py -- \
      --fbx "D:/dataset/高质量人物FBX模型/Rosemary/Rosemary winters 4.0.fbx" \
      --texdir "D:/dataset/高质量人物FBX模型/Rosemary/textures" \
      --out "D:/dataset/human_fbx_gt/rosemary_v1" \
      --mode full --n_views 72 --res 1024 --seed 0

输出(out 目录):
  rgb/000001.png ...        RGBA png(alpha 即人物 mask,背景透明)
  depth/000001.exr          float32 EXR,相机系线性深度(视图空间 +Z,米)
  normal/000001.exr         float32 EXR,世界系法线(-1~1)
  cameras.json              OpenCV 约定 cam2world 4x4 + 内参 K
  meta.json                 轨迹/引擎/种子等单位与配置

坐标约定(重要,下游 3DGS/VGGT 通用):
  cameras.json 里 transform_cam2world 是 OpenCV 相机系(x右 y下 z前);
  depth.exr 的正值 = OpenCV 相机系 z,可直接用 K 反投影:
      point_cam = K^-1 @ [u, v, 1] * z;  point_world = c2w @ point_cam
"""
import bpy
import json
import math
import os
import sys
import random

import mathutils

# ---------------------------------------------------------------- 参数解析
argv = sys.argv[sys.argv.index("--") + 1:]


def _arg(name, default):
    key = "--" + name
    if key in argv:
        i = argv.index(key)
        if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            return argv[i + 1]
    return default


def _arg_int(name, default):
    return int(_arg(name, str(default)))


def _arg_float(name, default):
    return float(_arg(name, str(default)))


FBX_PATH = _arg("fbx", "")
TEXDIR = _arg("texdir", "")
OUT_DIR = _arg("out", "")
MODE = _arg("mode", "full")               # full | face
N_VIEWS = _arg_int("n_views", 72 if MODE == "full" else 48)
RES = _arg_int("res", 1024)
SEED = _arg_int("seed", 0)
Focal_MM = _arg_float("focal_mm", 50.0)
DIST_MULT = _arg_float("distance_mult", 1.0)
ELEV_MIN = _arg_float("elev_min", -25.0 if MODE == "full" else -15.0)
ELEV_MAX = _arg_float("elev_max", 35.0 if MODE == "full" else 25.0)
YAW_SPAN = _arg_float("yaw_span", 360.0 if MODE == "full" else 150.0)  # face 档 ±75°
LIGHT_JITTER = _arg_float("light_jitter", 0.3)   # 光照域随机化强度 0~1
TARGET_ZFRAC = _arg_float("target_zfrac", 0.55 if MODE == "full" else 0.85)
                                     # look-at 目标高度 = bbox 高 * zfrac(胸口/面部)

assert FBX_PATH and OUT_DIR, "--fbx 与 --out 必填"
random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)
for sub in ("rgb", "depth", "normal"):
    os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)

# ---------------------------------------------------------------- 场景清理
bpy.ops.wm.read_factory_settings(use_empty=True)

# Blender 5.2 兼容:FBX 导入器对内置灯光对象访问 lamp.cycles.cast_shadow,
# 该属性在 5.2 已移除(hasattr 检查挡不住),触发 AttributeError 使导入中止。
# 禁用 cycles addon 后 hasattr(lamp,"cycles") 为 False,导入器自动跳过该赋值。
try:
    bpy.ops.preferences.addon_disable(module="cycles")
except Exception as e:  # noqa: BLE001
    print(f"[warn] 禁用 cycles addon 失败(依赖 EEVEE 渲染,理论上无碍): {e}")

bpy.ops.import_scene.fbx(
    filepath=FBX_PATH,
    use_anim=False,
    ignore_leaf_bones=True,
    automatic_bone_orientation=False,
)

imported = [o for o in bpy.context.scene.objects if o.type == "MESH"]
assert imported, "FBX 导入后未找到网格对象"
# FBX 自带的灯光/相机不进入渲染场景(光照由本脚本完全控制)
for o in list(bpy.context.scene.objects):
    if o.type in ("LIGHT", "CAMERA"):
        bpy.data.objects.remove(o, do_unlink=True)


def _has_texture(obj):
    for slot in obj.material_slots:
        mat = slot.material
        if mat and mat.use_nodes and mat.node_tree and any(
                n.type == "TEX_IMAGE" and n.image for n in mat.node_tree.nodes):
            return True
    return False


# 防御:FBX 可能夹带默认名辅助几何(地台/展示台),会污染 alpha mask、深度与
# bbox 归一化,按「默认名 + 无贴图」剔除。注意 Rosemary FBX 实测无夹带(上面的
# read_factory_settings 已挡掉启动场景物体;探测脚本里见过的 'Cube' 来自未清场
# 的 Blender 默认启动场景,不是 FBX 内容)——此过滤留给其它资产防身。
import re as _re
_junk_re = _re.compile(r"^(Cube|Plane|Sphere|Circle|Grid)([._]\d+)?$")
removed = []
for o in list(imported):
    if _junk_re.match(o.name) and not _has_texture(o):
        removed.append(o.name)
        bpy.data.objects.remove(o, do_unlink=True)
        imported.remove(o)
if removed:
    print(f"[warn] 剔除无贴图辅助几何: {removed}")

# ---------------------------------------------------------------- 贴图重连
# FBX 内嵌的图片路径常失效:按 basename 在 texdir 递归匹配,重设 filepath。
TEX_KEYWORDS = {  # 色彩空间判定
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


relinked = missing = 0
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
        if img.source != "FILE":
            continue
        if not os.path.exists(bpy.path.abspath(img.filepath)):
            cand = tex_index.get(os.path.basename(img.filepath).lower()) \
                or tex_index.get(os.path.basename(img.name).lower())
            if cand:
                img.filepath = cand
                relinked += 1
            else:
                missing += 1
        img.colorspace_settings.name = _colorspace_for(img.name)
    # EEVEE Next(Blender 4.2+):毛发/睫毛类含 alpha 贴图的材质需开启 DITHERED
    try:
        # DITHERED 条件:贴图名含 opacity,或 Principled 的 Alpha 输入有连接
        # (Rosemary 的头发是后一种:Alpha 有连线但贴图名无 opacity,不透明渲染会
        # 把发片的黑色镂空区域渲成实体黑块)
        has_alpha_tex = any(
            n.type == "TEX_IMAGE" and n.image and "opacity" in n.image.name.lower()
            for n in mat.node_tree.nodes) or any(
            l.to_node.type == "BSDF_PRINCIPLED" and l.to_socket.name == "Alpha"
            for l in mat.node_tree.links)
        if has_alpha_tex:
            mat.surface_render_method = "DITHERED"
    except (AttributeError, TypeError):
        pass
print(f"[tex] relinked={relinked} missing={missing}")

# ---------------------------------------------------------------- 归一化
# 缩放到人物高 1.70m,脚底 z=0,xy 居中(按全部网格并集 bbox)。
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
        @ o.matrix_world
    )
bpy.context.view_layer.update()

# 重新计算归一化后的 bbox
bb_min = mathutils.Vector((1e9, 1e9, 1e9))
bb_max = mathutils.Vector((-1e9, -1e9, -1e9))
for o in imported:
    for c in o.bound_box:
        w = o.matrix_world @ mathutils.Vector(c)
        bb_min = mathutils.Vector(map(min, bb_min, w))
        bb_max = mathutils.Vector(map(max, bb_max, w))
bbox_h = bb_max.z - bb_min.z          # ≈1.70
print(f"[norm] bbox_min={list(bb_min)} bbox_max={list(bb_max)} "
      f"height={bbox_h:.3f} scale={scale:.4f}")
target = mathutils.Vector(
    ((bb_min.x + bb_max.x) / 2, (bb_min.y + bb_max.y) / 2,
     bb_min.z + bbox_h * TARGET_ZFRAC))

# ---------------------------------------------------------------- 灯光与世界
world = bpy.data.worlds.new("W")
bpy.context.scene.world = world
world.use_nodes = True
wn = world.node_tree.nodes
wl = world.node_tree.links
wn.clear()
w_out = wn.new("ShaderNodeOutputWorld")
w_bg = wn.new("ShaderNodeBackground")
# ⚠️ 不要挂无图的 ShaderNodeTexEnvironment:EEVEE 里它会输出占位品红色,
# 世界强度 0.8 等于给全场打品红环境光(2026-11-05 实测:皮肤/头发/金属全体偏紫)。
# 无 HDRI 期间用中性灰环境色兜底,强度由 jitter_lights 随机化。
w_bg.inputs["Color"].default_value = (0.05, 0.05, 0.05, 1.0)
w_bg.inputs["Strength"].default_value = 0.8
wl.new(w_bg.outputs["Background"], w_out.inputs["Surface"])

key = bpy.data.lights.new("Key", "SUN")
key.energy = 4.0
key_obj = bpy.data.objects.new("Key", key)
bpy.context.scene.collection.objects.link(key_obj)
key_obj.rotation_euler = (math.radians(50), 0, math.radians(30))
fill = bpy.data.lights.new("Fill", "SUN")
fill.energy = 2.0
fill_obj = bpy.data.objects.new("Fill", fill)
bpy.context.scene.collection.objects.link(fill_obj)
fill_obj.rotation_euler = (math.radians(60), 0, math.radians(-140))
rim = bpy.data.lights.new("Rim", "SUN")       # 背光,避免背面视角死黑丢信息
rim.energy = 3.0
rim_obj = bpy.data.objects.new("Rim", rim)
bpy.context.scene.collection.objects.link(rim_obj)
rim_obj.rotation_euler = (math.radians(65), 0, math.radians(180))


def jitter_lights():
    """光照域随机化:三个太阳灯方位/能量/世界强度抖动。"""
    if LIGHT_JITTER <= 0:
        return
    key_obj.rotation_euler = (
        math.radians(50 + random.uniform(-1, 1) * 25 * LIGHT_JITTER),
        0,
        math.radians(30 + random.uniform(-1, 1) * 120 * LIGHT_JITTER))
    key.energy = 4.0 * (1 + random.uniform(-1, 1) * 0.5 * LIGHT_JITTER)
    fill_obj.rotation_euler = (
        math.radians(60 + random.uniform(-1, 1) * 20 * LIGHT_JITTER),
        0,
        math.radians(-140 + random.uniform(-1, 1) * 120 * LIGHT_JITTER))
    fill.energy = 2.0 * (1 + random.uniform(-1, 1) * 0.5 * LIGHT_JITTER)
    rim_obj.rotation_euler = (
        math.radians(65 + random.uniform(-1, 1) * 20 * LIGHT_JITTER),
        0,
        math.radians(180 + random.uniform(-1, 1) * 60 * LIGHT_JITTER))
    rim.energy = 3.0 * (1 + random.uniform(-1, 1) * 0.5 * LIGHT_JITTER)
    w_bg.inputs["Strength"].default_value = 0.8 * (
        1 + random.uniform(-1, 1) * 0.6 * LIGHT_JITTER)
    bpy.context.scene.view_settings.exposure = random.uniform(
        -1, 1) * 0.5 * LIGHT_JITTER


# ---------------------------------------------------------------- 渲染设置
scene = bpy.context.scene
engine_ids = [e.identifier for e in
              bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]
scene.render.engine = ("BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engine_ids
                       else "BLENDER_EEVEE")
scene.render.resolution_x = RES
scene.render.resolution_y = RES
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.render.film_transparent = True          # alpha = 人物 mask
scene.render.use_file_extension = True
# 色彩管理:Raw 保证 EXR 深度/法线不被 view transform 污染
scene.view_settings.look = "AgX - Medium High Contrast"

vl = scene.view_layers[0]
vl.use_pass_z = True
vl.use_pass_normal = True


# ---------------------------------------------------------------- 合成输出(Blender 5.2 新 API)
# 5.x 变更:compositor 改为节点组;File Output 的 base_path→directory、
# file_slots→file_output_items.new(socket_type, name);EXR 仅支持 OPEN_EXR_MULTILAYER。
# 深度/法线各用一个 File Output 节点,分目录落盘,保持"每 pass 一文件"。
_comp = bpy.data.node_groups.new("CompGT", "CompositorNodeTree")
scene.compositing_node_group = _comp
_rl = _comp.nodes.new("CompositorNodeRLayers")


def _mk_file_output(subdir, socket_type, item_name, link_output):
    node = _comp.nodes.new("CompositorNodeOutputFile")
    node.directory = os.path.join(OUT_DIR, subdir)
    node.format.file_format = "OPEN_EXR_MULTILAYER"
    node.format.color_depth = "32"
    node.file_name = item_name + "_####"   # #### 为帧号占位符
    item = node.file_output_items.new(socket_type, item_name)
    _comp.links.new(_rl.outputs[link_output], node.inputs[item_name])
    return node, item


_fout_depth, _item_depth = _mk_file_output(
    "depth", "FLOAT", "depth", "Depth")
_fout_normal, _item_normal = _mk_file_output(
    "normal", "VECTOR", "normal", "Normal")
# 注:5.2 的 File Output 仅支持 OPEN_EXR_MULTILAYER,质检走 WSL 读 EXR 数值

# ---------------------------------------------------------------- 相机
cam_data = bpy.data.cameras.new("Cam")
cam_data.lens = Focal_MM
cam_data.sensor_width = 36.0
cam = bpy.data.objects.new("Cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam

# 距离:全身档取能容纳全身(留边距),face 档更近
# 垂直半视场正切 = sensor/(2·f)(方形像素,无需 atan/tan 往返)
tan_half_vfov = 36.0 / (2.0 * Focal_MM)
if MODE == "full":
    dist = bbox_h * 0.72 / tan_half_vfov * 1.15
else:
    dist = bbox_h * 0.55
dist *= DIST_MULT


def fibonacci_sphere(n, elev_min, elev_max, yaw_span):
    """n 个视角:仰俯在 [elev_min, elev_max] 内均匀,方位按 yaw_span 分布。
    full 档 yaw_span=360 覆盖整圈;face 档 150° 居中于模型正前方(+Y)。"""
    pts = []
    e0, e1 = math.radians(elev_min), math.radians(elev_max)
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n):
        frac = (i + 0.5) / n
        elev = e0 + (e1 - e0) * frac
        if yaw_span >= 360.0:
            azim = math.radians((i * 360.0 / n + 90.0) % 360.0)
        else:
            azim = math.radians(90.0 - yaw_span / 2
                                + yaw_span * (i + 0.5) / n)
        r_xy = dist * math.cos(elev)
        loc = mathutils.Vector((
            target.x + r_xy * math.cos(azim),
            target.y + r_xy * math.sin(azim),
            target.z + dist * math.sin(elev)))
        pts.append((loc, azim, elev))
    return pts


poses = fibonacci_sphere(N_VIEWS, ELEV_MIN, ELEV_MAX, YAW_SPAN)

# Blender 相机系 → OpenCV 相机系 的固定旋转(x 相同,y/z 取反)
C_CONV = mathutils.Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))
fx = Focal_MM / 36.0 * RES
K = [[fx, 0, (RES - 1) / 2.0], [0, fx, (RES - 1) / 2.0], [0, 0, 1.0]]

cameras = []
for i, (loc, azim, elev) in enumerate(poses):
    jitter_lights()
    cam.location = loc
    direction = target - loc
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.view_layer.update()

    scene.frame_set(i + 1)                    # File Output 用帧号区分文件名
    bpy.ops.render.render(write_still=False)

    rgb_path = os.path.join(OUT_DIR, "rgb", f"{i + 1:06d}.png")
    bpy.data.images["Render Result"].save_render(rgb_path, scene=scene)

    R_b = cam.matrix_world.to_3x3()           # world→blender-cam 的逆,即 cam2world
    R_cv = C_CONV @ R_b                       # 转 OpenCV 相机系
    t = cam.matrix_world.to_translation()
    c2w = [list(R_cv.row[0]) + [t.x],
           list(R_cv.row[1]) + [t.y],
           list(R_cv.row[2]) + [t.z],
           [0, 0, 0, 1]]
    cameras.append({
        "id": i,
        "file_rgb": f"rgb/{i + 1:06d}.png",
        "file_depth": f"depth/{i + 1:06d}.exr",
        "file_normal": f"normal/{i + 1:06d}.exr",
        "transform_cam2world": c2w,
        "azimuth_deg": math.degrees(azim),
        "elevation_deg": math.degrees(elev),
        "distance_m": dist,
    })
    print(f"[render] {i + 1}/{N_VIEWS} az={math.degrees(azim):7.2f} "
          f"el={math.degrees(elev):6.2f}")

# File Output 命名 depth_0001.exr → 统一改成 000001.exr(匹配 cameras.json)
import re
for sub in ("depth", "normal"):
    d = os.path.join(OUT_DIR, sub)
    for f in os.listdir(d):
        m = re.fullmatch(r"[a-z]+_(\d+)\.(exr|png)", f)
        if m:
            os.replace(os.path.join(d, f),
                       os.path.join(d, f"{int(m.group(1)):06d}.{m.group(2)}"))

with open(os.path.join(OUT_DIR, "cameras.json"), "w", encoding="utf-8") as f:
    json.dump({
        "convention": "OpenCV(x右 y下 z前); depth.exr=相机系线性 z(米); "
                      "normal.exr=世界系(-1~1)",
        "intrinsics": {"K": K, "width": RES, "height": RES,
                       "focal_mm": Focal_MM, "sensor_mm": 36.0,
                       "distortion": [0, 0, 0, 0, 0]},
        "cameras": cameras,
    }, f, ensure_ascii=False, indent=1)

meta = {
    "fbx": FBX_PATH, "mode": MODE, "n_views": N_VIEWS, "res": RES,
    "seed": SEED, "engine": scene.render.engine,
    "focal_mm": Focal_MM, "distance_m": dist,
    "elev_range": [ELEV_MIN, ELEV_MAX], "yaw_span": YAW_SPAN,
    "target": list(target), "bbox_min": list(bb_min), "bbox_max": list(bb_max),
    "height_normalized_m": bbox_h,
    "light_jitter": LIGHT_JITTER,
    "unit": "meter",
}
with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=1)

print(f"[done] {N_VIEWS} views -> {OUT_DIR}")
