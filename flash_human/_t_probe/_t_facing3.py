import re
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
try:
    bpy.ops.preferences.addon_disable(module="cycles")
except Exception:
    pass
bpy.ops.import_scene.fbx(filepath=argv[1], use_anim=False,
                         ignore_leaf_bones=True,
                         automatic_bone_orientation=False)
junk = re.compile(r"^(Cube|Plane|Sphere|Circle|Grid)([._]\d+)?$")

def mesh_normal_xy(name_pat):
    """对名字匹配 name_pat 的网格,算面积加权法线均值(世界系),返回水平单位向量与模长。"""
    nx = ny = nz = 0.0
    for o in bpy.context.scene.objects:
        if o.type != "MESH" or junk.match(o.name) or not re.search(name_pat, o.name, re.I):
            continue
        mw = o.matrix_world
        for p in o.data.polygons:
            n = (mw.to_3x3() @ p.normal).normalized() * (mw.to_3x3() @ p.normal).length
            # 面积加权:直接用局部法线转世界后乘面积
            n = mw.to_3x3() @ p.normal
            nx += n.x * p.area; ny += n.y * p.area; nz += n.z * p.area
    return nx, ny, nz

for pat in ("eye", "lash"):
    nx, ny, nz = mesh_normal_xy(pat)
    horiz = (nx*nx + ny*ny) ** 0.5
    if horiz > 1e-6:
        print(f"[normal:{pat}] horiz=({nx/horiz:+.3f},{ny/horiz:+.3f}) |horiz|={horiz:.2f} nz={nz:.2f}")
    else:
        print(f"[normal:{pat}] 水平分量≈0 (nx={nx:.3f} ny={ny:.3f} nz={nz:.3f})")
print("[done]")
