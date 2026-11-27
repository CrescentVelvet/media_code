#!/usr/bin/env python3
"""01d_verify_depth_pass.py -- 用已知网格验证 Blender 5.2 Depth pass。

Blender 后台运行，不修改 FBX 或已有 GT 包。像素射线由实际相机投影矩阵
生成（包含 shift），再用物体逆矩阵做 ray cast；不是用 Depth 反推真值。
对 cube/sphere/plane 分别比较 EXR、可见面轴向 Z 与射线距离 D。只有满足
绝对误差门才标记通过，不能用“两个误差谁更小”自动决定类型。
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector
from bpy_extras.object_utils import world_to_camera_view
import OpenImageIO as oiio

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True, help="必须是不存在的新目录")
ap.add_argument("--camera-z", type=float, default=3.0, help="环绕观察的相机半径")
ap.add_argument("--tilt-x", type=float, default=0.0, help="观察仰角（度）")
ap.add_argument("--tilt-y", type=float, default=0.0, help="观察方位角（度）")
ap.add_argument("--shift-x", type=float, default=0.08)
ap.add_argument("--shift-y", type=float, default=-0.04)
ap.add_argument("--tolerance", type=float, default=2e-5)
args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])
out = Path(args.out)
out.mkdir(parents=True, exist_ok=False)

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.samples = 1
scene.cycles.use_denoising = False
scene.render.resolution_x = 256
scene.render.resolution_y = 256
scene.render.resolution_percentage = 100
scene.render.pixel_aspect_x = 1.0
scene.render.pixel_aspect_y = 1.0
scene.render.film_transparent = True
scene.render.use_sequencer = False
scene.render.compositor_device = "CPU"
scene.view_settings.view_transform = "Standard"
scene.world = bpy.data.worlds.new("World")
scene.world.use_nodes = True
scene.world.node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
scene.view_layers[0].use_pass_z = True
scene.use_nodes = True
comp = bpy.data.node_groups.new("DepthValidation", "CompositorNodeTree")
comp.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
out_node = comp.nodes.new("NodeGroupOutput")
scene.compositing_node_group = comp
rl = comp.nodes.new("CompositorNodeRLayers")
fo = comp.nodes.new("CompositorNodeOutputFile")
fo.directory = str(out)  # Blender 5.2 实例化后会重置路径，必须后设
fo.format.file_format = "OPEN_EXR_MULTILAYER"
fo.format.color_depth = "32"
fo.file_output_items.new("FLOAT", "depth")
fo.file_name = "depth_####"  # 添加 item 会重写名称，必须最后设置
comp.links.new(rl.outputs["Depth"], fo.inputs["depth"])
comp.links.new(rl.outputs["Image"], out_node.inputs["Image"])

camdata = bpy.data.cameras.new("Camera")
camdata.lens = 50.0
camdata.sensor_width = 36.0
camdata.sensor_fit = "HORIZONTAL"
camdata.shift_x = args.shift_x
camdata.shift_y = args.shift_y
cam = bpy.data.objects.new("Camera", camdata)
scene.collection.objects.link(cam)
scene.camera = cam
cam.location = (args.camera_z * math.cos(math.radians(args.tilt_x)) *
                math.sin(math.radians(args.tilt_y)),
                -args.camera_z * math.sin(math.radians(args.tilt_x)),
                args.camera_z * math.cos(math.radians(args.tilt_x)) *
                math.cos(math.radians(args.tilt_y)))
cam.rotation_euler = (-cam.location).to_track_quat("-Z", "Y").to_euler()
bpy.context.view_layer.update()
scene.frame_set(1)
deps = bpy.context.evaluated_depsgraph_get()
res = scene.render.resolution_x
proj = cam.calc_matrix_camera(deps, x=res, y=res, scale_x=1.0, scale_y=1.0)
proj_inv = proj.inverted()
R_cv = cam.matrix_world.to_3x3() @ Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))
center = cam.matrix_world.translation.copy()
fx = camdata.lens / camdata.sensor_width * res


def pixel_ray(u, v):
    """像素中心经实际投影矩阵到世界系，返回 unit direction 和归一化坐标。"""
    ndc = Vector((2.0 * (u + 0.5) / res - 1.0,
                  1.0 - 2.0 * (v + 0.5) / res, -1.0, 1.0))
    p_cam_h = proj_inv @ ndc
    p_cam = Vector(p_cam_h[:3]) / p_cam_h.w
    direction = (cam.matrix_world.to_3x3() @ p_cam).normalized()
    return direction, p_cam


samples = []
pixels = ((0, 0), (-12, 4), (12, -8), (8, 12), (-14, -10))
for objtype, name in (("CUBE", "cube"), ("SPHERE", "sphere"), ("PLANE", "plane")):
    if objtype == "CUBE":
        bpy.ops.mesh.primitive_cube_add(size=2)
    elif objtype == "PLANE":
        bpy.ops.mesh.primitive_plane_add(size=200)
    else:
        bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=0.75)
    obj = bpy.context.object
    obj.name = name
    scene.frame_set(1)
    bpy.context.view_layer.update()
    deps = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(deps)
    inv_obj = obj_eval.matrix_world.inverted()
    for du, dv in pixels:
        center_ndc = world_to_camera_view(scene, cam, obj_eval.matrix_world.translation)
        u = int(round(center_ndc.x * res - 0.5)) + du
        v = int(round((1.0 - center_ndc.y) * res - 0.5)) + dv
        if not (0 <= u < res and 0 <= v < res):
            raise RuntimeError("Object-centered test pixel outside image")
        direction_world, p_cam = pixel_ray(u, v)
        origin_local = inv_obj @ center
        direction_local = (inv_obj.to_3x3() @ direction_world).normalized()
        hit, loc, normal, face_index = obj_eval.ray_cast(origin_local, direction_local)
        if not hit:
            raise RuntimeError("Known object ray missed: %s pixel=(%d,%d)" % (name, u, v))
        loc_world = obj_eval.matrix_world @ loc
        projected = world_to_camera_view(scene, cam, loc_world)
        proj_error = max(abs(projected.x * res - (u + 0.5)),
                         abs((1.0 - projected.y) * res - (v + 0.5)))
        if proj_error > 1e-3:
            raise RuntimeError("Projection round-trip failed: %s pixel=(%d,%d), error=%g" %
                               (name, u, v, proj_error))
        cv = R_cv.transposed() @ (loc_world - center)
        samples.append(dict(object=name, u=u, v=v, axial_z=cv.z,
                            ray_distance=(loc_world - center).length,
                            point_world=list(loc_world), point_cv=list(cv),
                            face_index=face_index, projection_error_px=proj_error,
                            camera_ray=list(direction_world), camera_local_point=list(p_cam)))
    scene.render.filepath = str(out / (name + ".png"))
    bpy.ops.render.render(write_still=True)
    actual = out / "depth_0001.exr"
    if not actual.is_file():
        raise RuntimeError("Depth pass was not written; refusing stale/missing EXR")
    inp = oiio.ImageInput.open(str(actual))
    if inp is None:
        raise RuntimeError("Cannot open depth EXR: " + str(actual))
    spec = inp.spec()
    dep = inp.read_image(format="float")[..., 0]
    inp.close()
    if tuple(dep.shape[:2]) != (res, res):
        raise RuntimeError("Unexpected depth shape: " + str(dep.shape))
    for r in samples:
        if r["object"] == name:
            r["depth_exr"] = float(dep[r["v"], r["u"]])
    obj.hide_render = True
    os.replace(actual, out / (name + ".exr"))

for r in samples:
    r["minus_axial"] = r["depth_exr"] - r["axial_z"]
    r["minus_distance"] = r["depth_exr"] - r["ray_distance"]
max_z = max(abs(r["minus_axial"]) for r in samples)
max_d = max(abs(r["minus_distance"]) for r in samples)
z_pass = max_z <= args.tolerance
d_pass = max_d <= args.tolerance
if z_pass == d_pass:
    convention = "ambiguous_or_failed"
elif z_pass:
    convention = "axial_z"
else:
    convention = "ray_distance"
result = {
    "blender_version": bpy.app.version_string,
    "method": "Cycles 1 sample; actual shifted camera projection -> object-local visible mesh ray cast",
    "pixel_convention": "u,v are zero-based integer indices; sample pixel centers at u+0.5,v+0.5",
    "camera_location": list(center),
    "camera_rotation_euler_deg": [args.tilt_x, args.tilt_y, 0],
    "camera_shift": [args.shift_x, args.shift_y],
    "camera_fx_fy": [fx, fx],
    "camera_projection_matrix": [list(row) for row in proj],
    "camera_c2w_opencv": [list(R_cv[i]) + [center[i]] for i in range(3)] + [[0, 0, 0, 1]],
    "tolerance_world_unit": args.tolerance,
    "samples": samples,
    "max_abs_error_axial_z": max_z,
    "max_abs_error_ray_distance": max_d,
    "axial_z_pass": z_pass,
    "ray_distance_pass": d_pass,
    "convention": convention,
    "validation_pass": z_pass != d_pass,
}
with open(out / "depth_pass_validation.json", "w", encoding="utf-8", newline="\n") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
print(json.dumps({k: v for k, v in result.items() if k not in ("samples", "camera_projection_matrix")},
                 indent=2, ensure_ascii=False), flush=True)
if not result["validation_pass"]:
    raise RuntimeError("Depth convention did not pass an unambiguous absolute-error gate")
