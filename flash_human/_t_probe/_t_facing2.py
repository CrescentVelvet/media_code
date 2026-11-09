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
objs = [o for o in bpy.context.scene.objects
        if o.type == "MESH" and not junk.match(o.name)]

# 头部区域(z > 2.4)顶点 bbox + 眼睛/睫毛质心
pts = []
for o in objs:
    for v in o.data.vertices:
        w = o.matrix_world @ v.co
        if w.z > 2.4:
            pts.append(w)
xs = [p.x for p in pts]; ys = [p.y for p in pts]; zs = [p.z for p in pts]
cx, cy = (min(xs)+max(xs))/2, (min(ys)+max(ys))/2
print(f"[head-bbox] x[{min(xs):.3f},{max(xs):.3f}] y[{min(ys):.3f},{max(ys):.3f}] "
      f"z[{min(zs):.3f},{max(zs):.3f}] center=({cx:.3f},{cy:.3f})")

for o in objs:
    if o.name.lower() in ("eyes", "lashes"):
        n = len(o.data.vertices)
        ex = sum((o.matrix_world @ v.co).x for v in o.data.vertices) / n
        ey = sum((o.matrix_world @ v.co).y for v in o.data.vertices) / n
        ez = sum((o.matrix_world @ v.co).z for v in o.data.vertices) / n
        print(f"[{o.name}] centroid=({ex:.3f},{ey:.3f},{ez:.3f}) "
              f"相对头部bbox中心偏移=({ex-cx:+.3f},{ey-cy:+.3f})")
print("[done]")
