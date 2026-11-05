# 导出指定材质的完整节点图(节点 + 连线)
import bpy
import sys

argv = sys.argv[sys.argv.index("--") + 1:]
FBX_PATH = argv[argv.index("--fbx") + 1]

try:
    bpy.ops.preferences.addon_disable(module="cycles")
except Exception:
    pass
bpy.ops.import_scene.fbx(filepath=FBX_PATH, use_anim=False,
                         ignore_leaf_bones=True,
                         automatic_bone_orientation=False)

for mat in bpy.data.materials:
    if mat.name not in ("hair", "eyes", "body", "Material.001", "eyes_lashes", "Material", "Dots Stroke"):
        continue
    print(f"==== {mat.name} ====")
    nt = mat.node_tree
    for n in nt.nodes:
        extra = ""
        if n.type == "TEX_IMAGE":
            extra = f" img={n.image.name if n.image else None}"
        print(f"  node {n.name} type={n.type}{extra}")
    for l in nt.links:
        print(f"  link {l.from_node.name}.{l.from_socket.name} "
              f"-> {l.to_node.name}.{l.to_socket.name}")
print("[done]")
