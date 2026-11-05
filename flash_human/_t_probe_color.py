# 诊断渲染变紫:导出材质节点链 + UV 层 + 单帧正面渲染
# 用法: blender.exe -b --python _t_probe_color.py -- --fbx <path> --out <png>
import bpy
import math
import mathutils
import os
import sys

argv = sys.argv[sys.argv.index("--") + 1:]


def _arg(name, default):
    key = "--" + name
    if key in argv:
        i = argv.index(key)
        if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            return argv[i + 1]
    return default


FBX_PATH = _arg("fbx", "")
OUT_PNG = _arg("out", "")

try:
    bpy.ops.preferences.addon_disable(module="cycles")
except Exception:
    pass

bpy.ops.import_scene.fbx(filepath=FBX_PATH, use_anim=False,
                         ignore_leaf_bones=True,
                         automatic_bone_orientation=False)
for o in list(bpy.context.scene.objects):
    if o.type in ("LIGHT", "CAMERA"):
        bpy.data.objects.remove(o, do_unlink=True)

print("==== mesh UV / color attributes ====")
for o in bpy.context.scene.objects:
    if o.type != "MESH":
        continue
    uvs = [(uv.name, uv.active_render) for uv in o.data.uv_layers]
    cols = [a.name for a in o.data.color_attributes]
    print(f"  mesh={o.name!r} uv_layers={uvs} color_attrs={cols}")

print("==== material base color chain ====")
for mat in bpy.data.materials:
    if not mat.use_nodes or not mat.node_tree:
        print(f"  [mat] {mat.name}: no nodes")
        continue
    for node in mat.node_tree.nodes:
        if node.type != "BSDF_PRINCIPLED":
            continue
        inp = node.inputs.get("Base Color")
        links = inp.links if inp else []
        if not links:
            print(f"  [mat] {mat.name}: BaseColor 无连接 "
                  f"default={tuple(round(v,3) for v in inp.default_value) if inp else None}")
            continue
        src = links[0].from_node
        if src.type == "TEX_IMAGE":
            img = src.image
            print(f"  [mat] {mat.name}: TEX {img.name!r} "
                  f"uv={src.inputs['Vector'].links[0].from_node.type if src.inputs['Vector'].links else 'DEFAULT'} "
                  f"exists={os.path.exists(bpy.path.abspath(img.filepath)) if img else '?'} "
                  f"cs={img.colorspace_settings.name if img else '?'}")
        else:
            print(f"  [mat] {mat.name}: 经 {src.type} 节点连接 BaseColor")

# ---- 单帧正面渲染(与 01 脚本同灯光配置) ----
world = bpy.data.worlds.new("W")
bpy.context.scene.world = world
world.use_nodes = True
wn = world.node_tree.nodes
wn.clear()
w_out = wn.new("ShaderNodeOutputWorld")
w_bg = wn.new("ShaderNodeBackground")
w_bg.inputs["Strength"].default_value = 0.8
wl = world.node_tree.links
wl.new(w_bg.outputs["Background"], w_out.inputs["Surface"])
for name, en, rot in (("Key", 4.0, (50, 0, 30)),
                      ("Fill", 2.0, (60, 0, -140)),
                      ("Rim", 3.0, (65, 0, 180))):
    ld = bpy.data.lights.new(name, "SUN")
    ld.energy = en
    lo = bpy.data.objects.new(name, ld)
    bpy.context.scene.collection.objects.link(lo)
    lo.rotation_euler = tuple(map(math.radians, rot))

scene = bpy.context.scene
ids = [e.identifier for e in
       bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]
scene.render.engine = ("BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in ids
                       else "BLENDER_EEVEE")
scene.render.resolution_x = scene.render.resolution_y = 512
scene.render.resolution_percentage = 100
scene.render.film_transparent = False  # 灰底便于看颜色
scene.view_settings.look = "AgX - Medium High Contrast"

cam_data = bpy.data.cameras.new("Cam")
cam_data.lens = 50.0
cam_data.sensor_width = 36.0
cam = bpy.data.objects.new("Cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam
cam.location = (0, -3.5, 1.2)
cam.rotation_euler = ((mathutils.Vector((0, 0, 1.1)) - cam.location)
                      .to_track_quat("-Z", "Y").to_euler())

bpy.ops.render.render(write_still=False)
bpy.data.images["Render Result"].save_render(OUT_PNG, scene=scene)
print(f"[done] -> {OUT_PNG}")
