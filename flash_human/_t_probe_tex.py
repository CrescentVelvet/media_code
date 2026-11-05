# Blender 后台探测:诊断 FBX 贴图连接状态(不落盘渲染)
# 用法: blender.exe -b --python _t_probe_tex.py -- --fbx <path> --texdir <dir>
import bpy
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
TEXDIR = _arg("texdir", "")

try:
    bpy.ops.preferences.addon_disable(module="cycles")
except Exception as e:
    print(f"[warn] disable cycles failed: {e}")

bpy.ops.import_scene.fbx(filepath=FBX_PATH, use_anim=False,
                         ignore_leaf_bones=True,
                         automatic_bone_orientation=False)

tex_index = {}
if TEXDIR and os.path.isdir(TEXDIR):
    for root, _, files in os.walk(TEXDIR):
        for f in files:
            tex_index.setdefault(f.lower(), os.path.join(root, f))

print("==== materials ====")
for mat in bpy.data.materials:
    print(f"[mat] {mat.name}")
    if not mat.use_nodes or not mat.node_tree:
        print("    (no node tree!)")
        continue
    for node in mat.node_tree.nodes:
        if node.type == "TEX_IMAGE":
            img = node.image
            if img is None:
                print(f"    TEX_IMAGE node '{node.name}': image=None")
                continue
            ap = bpy.path.abspath(img.filepath) if img.filepath else ""
            exists = os.path.exists(ap) if ap else False
            base = os.path.basename(ap or img.name)
            cand = tex_index.get(base.lower()) or \
                tex_index.get(os.path.basename(img.name).lower())
            status = "OK" if exists else \
                ("RELINKABLE->" + cand if cand else "MISSING")
            print(f"    img={img.name!r} src={img.source} "
                  f"filepath={ap!r} -> {status}")

print("==== orphan images (not in any material) ====")
used = set()
for mat in bpy.data.materials:
    if mat.use_nodes and mat.node_tree:
        for node in mat.node_tree.nodes:
            if node.type == "TEX_IMAGE" and node.image:
                used.add(node.image.name)
for img in bpy.data.images:
    if img.name not in used:
        print(f"    {img.name!r} filepath={bpy.path.abspath(img.filepath)!r}")

print("==== meshes & material slots ====")
for o in bpy.context.scene.objects:
    if o.type == "MESH":
        slots = [s.material.name if s.material else None
                 for s in o.material_slots]
        print(f"    mesh={o.name!r} mats={slots}")
print("[done]")
