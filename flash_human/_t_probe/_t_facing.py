# -*- coding: utf-8 -*-
"""临时探针:判定 FBX 模型朝向(face 档假设正前方=+Y)。

眼睛/鼻子在头部最前缘 → 取 "eyes"/"head" 相关网格的顶点质心,
与全身 bbox 中心比较,输出水平朝向向量。
"""
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
meshes = []
for o in bpy.context.scene.objects:
    if o.type == "MESH" and not junk.match(o.name):
        meshes.append(o)

bb_min = Vector((1e9,) * 3)
bb_max = Vector((-1e9,) * 3)
for o in meshes:
    for c in o.bound_box:
        w = o.matrix_world @ Vector(c)
        bb_min = Vector(map(min, bb_min, w))
        bb_max = Vector(map(max, bb_max, w))
center = (bb_min + bb_max) / 2
print(f"[bbox] min={tuple(round(v,3) for v in bb_min)} "
      f"max={tuple(round(v,3) for v in bb_max)} center={tuple(round(v,3) for v in center)}")

# 逐网格统计:名称 / 顶点质心 / 顶点数(只打名字像头/眼/鼻/脸的)
for o in meshes:
    n = len(o.data.vertices)
    c = o.matrix_world @ Vector((sum(v.co.x for v in o.data.vertices) / n,
                                 sum(v.co.y for v in o.data.vertices) / n,
                                 sum(v.co.z for v in o.data.vertices) / n))
    tag = ""
    nm = o.name.lower()
    if any(k in nm for k in ("eye", "head", "face", "nose", "brow", "lash")):
        tag = "  <== head-part"
    print(f"[mesh] {o.name:30s} n={n:8d} centroid=({c.x:.3f},{c.y:.3f},{c.z:.3f}){tag}")

# 头部区域(z > bbox_min.z + 0.8*bbox_h)所有顶点质心 → 脸应该在朝向侧更凸出
z0, z1 = bb_min.z, bb_max.z
thr = z0 + 0.8 * (z1 - z0)
pts = []
for o in meshes:
    for v in o.data.vertices:
        w = o.matrix_world @ v.co
        if w.z > thr:
            pts.append(w)
if pts:
    hc = Vector((0, 0, 0))
    for p in pts:
        hc += p
    hc /= len(pts)
    d = hc - center
    print(f"[head] z>{thr:.3f} 顶点数={len(pts)} 质心=({hc.x:.3f},{hc.y:.3f},{hc.z:.3f})")
    print(f"[head] 相对 bbox 中心水平偏移 = ({d.x:.3f}, {d.y:.3f})  "
          f"→ 脸朝 {'+Y' if d.y > 0.02 else '-Y' if d.y < -0.02 else '?'} / "
          f"{'+X' if d.x > 0.02 else '-X' if d.x < -0.02 else ''}")
print("[done]")
