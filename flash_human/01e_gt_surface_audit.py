#!/usr/bin/env python3
"""01e_gt_surface_audit.py -- 从原 FBX 对账 GT Depth、法线与高斯表面距离。

Blender 后台运行，原 FBX 只读，不重渲、不修改 GT。必须读取被审计包的 manifest
和（修正后的）场景相机；通过 manifest 的文件名记录使用哪一帧 RGB/深度。

两种模式：
1. verify：在所有 GT 有效像素的网格子采样上，比较 Depth EXR 与实际可见面射线
   的轴向 Z，并对 Depth 反投影法线与实际网格法线求角误差。
2. gaussians：对投影在脸区的选中高斯，求真实网格最近点距离，不把投影中心的
   同像素深度差冒充可见性证明。

相机默认按 17_fbxgt_to_scene.py 的旧包规则做 C R C；新生成器包用 --no-fix。
输出目录必须不存在，避免混用旧缓存。依赖 Blender 自带 OpenImageIO。
"""
import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
import numpy as np
import OpenImageIO as oiio

ap = argparse.ArgumentParser()
ap.add_argument("--fbx", required=True)
ap.add_argument("--pkg", required=True)
ap.add_argument("--prior", required=True, help="含本视角 c2w 的 prior_camera.json")
ap.add_argument("--out", required=True)
ap.add_argument("--mode", choices=("verify", "gaussians"), required=True)
ap.add_argument("--ply", default=None)
ap.add_argument("--selected", default=None)
ap.add_argument("--face-mask", default=None, help="756 像素 face mask")
ap.add_argument("--landmarks", default=None, help="可空：审核图用原始视角地标")
ap.add_argument("--sim3", default=None, help="target_metrics.json，含 s/R/T")
ap.add_argument("--no-fix", action="store_true")
ap.add_argument("--max-ray-samples", type=int, default=30000)
args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])
out = Path(args.out)
out.mkdir(parents=True, exist_ok=False)
pkg = Path(args.pkg)
manifest = json.loads((pkg / "cameras.json").read_text(encoding="utf-8"))
meta = json.loads((pkg / "meta.json").read_text(encoding="utf-8"))
prior = json.loads(Path(args.prior).read_text(encoding="utf-8"))
entries = {c["camera_id"]: c for c in prior["extrinsics"]}
K = np.asarray(prior["intrinsics"][0]["matrix"], np.float64)

# 与 01_render_gt.py 相同的导入、去灯光、非贴图辅助物体过滤与归一化。
bpy.ops.wm.read_factory_settings(use_empty=True)
try:
    bpy.ops.preferences.addon_disable(module="cycles")
except Exception as e:
    print("[warn] cycles disable:", e, flush=True)
bpy.ops.import_scene.fbx(filepath=args.fbx, use_anim=False, ignore_leaf_bones=True,
                        automatic_bone_orientation=False)
objects = [o for o in bpy.context.scene.objects if o.type == "MESH"]
for o in list(bpy.context.scene.objects):
    if o.type in ("LIGHT", "CAMERA"):
        bpy.data.objects.remove(o, do_unlink=True)
import re
junk = re.compile(r"^(Cube|Plane|Sphere|Circle|Grid)([._]\d+)?$")
objects = [o for o in objects if not junk.match(o.name)]
bpy.context.view_layer.update()
coords = [o.matrix_world @ Vector(c) for o in objects for c in o.bound_box]
bbmin = Vector(tuple(min(p[i] for p in coords) for i in range(3)))
bbmax = Vector(tuple(max(p[i] for p in coords) for i in range(3)))
scale = 1.70 / (bbmax.z - bbmin.z)
center = (bbmin + bbmax) / 2
from mathutils import Matrix
norm = Matrix.Translation((-center.x * scale, -center.y * scale, -bbmin.z * scale)) @ \
       Matrix.Scale(scale, 4)
for o in objects:
    o.matrix_world = norm @ o.matrix_world
bpy.context.view_layer.update()
for o in objects:
    for v in o.data.vertices:
        v.co = o.matrix_world @ v.co
    o.matrix_world = Matrix.Identity(4)
    o.data.update()
verts = [v.co.copy() for o in objects for v in o.data.vertices]
polys = [tuple(p.vertices) for o in objects for p in o.data.polygons]
bvh = BVHTree.FromPolygons(verts, polys, all_triangles=False)
C = np.diag([1.0, -1.0, -1.0])


def read_depth(path):
    inp = oiio.ImageInput.open(str(path))
    if inp is None:
        raise RuntimeError("Cannot read " + str(path))
    names = inp.spec().channelnames
    index = list(names).index("depth.V")
    arr = inp.read_image(format="float")[..., index].astype(np.float64)
    inp.close()
    return arr


def load_png(path):
    inp = oiio.ImageInput.open(str(path))
    if inp is None:
        raise RuntimeError("Cannot read " + str(path))
    arr = inp.read_image(format="float")
    inp.close()
    return np.asarray(arr)


def cameras():
    for stem, c in entries.items():
        numeric = int(stem.rsplit("_", 1)[1])
        m = entries[stem]
        c2w = np.asarray(c["matrix"], np.float64)
        c2w[:3, :3] = C @ c2w[:3, :3] @ C if not args.no_fix else c2w[:3, :3]
        mc = next((x for x in manifest["cameras"] if int(x["id"]) == numeric), None)
        if mc is None:
            raise RuntimeError("Missing manifest camera " + str(numeric))
        yield stem, numeric, c2w, mc


def ray_for_pixel(c2w, K_native, u, v, width, height):
    # 原点/方向都在 Blender 世界；K 是 OpenCV 像素中心坐标。
    pc = np.linalg.inv(K_native) @ np.array([u, v, 1.0])
    direction = c2w[:3, :3] @ pc
    return Vector(c2w[:3, 3]), Vector(direction).normalized()


def scaled_K(K_native, width, height):
    native_w = int(manifest["intrinsics"]["width"])
    native_h = int(manifest["intrinsics"]["height"])
    Kn = np.asarray(K_native, np.float64).copy()
    Kn[0] *= width / native_w
    Kn[1] *= height / native_h
    Kn[0, 2] = (K_native[0, 2] + 0.5) * width / native_w - 0.5
    Kn[1, 2] = (K_native[1, 2] + 0.5) * height / native_h - 0.5
    return Kn


if args.mode == "verify":
    report = {"fbx": args.fbx, "camera_fix": "C R C" if not args.no_fix else "none",
              "mesh_vertices": len(verts), "mesh_polygons": len(polys), "views": {}}
    for stem, numeric, c2w, mc in cameras():
        dep = read_depth(pkg / mc["file_depth"])
        rgb = load_png(pkg / mc["file_rgb"])
        h, w = dep.shape
        valid = np.isfinite(dep) & (dep > 1e-4) & (dep < 1e8)
        er = np.asarray(rgb[..., 3] > 0.9999) & valid if rgb.shape[-1] == 4 else valid
        # 离开 alpha/深度边界一格，避免锯齿与透明发丝的插值/离散差异混入类型验收。
        er[0, :] = er[-1, :] = False
        er[:, 0] = er[:, -1] = False
        ys, xs = np.where(er)
        if len(xs) > args.max_ray_samples:
            take = np.linspace(0, len(xs) - 1, args.max_ray_samples).astype(np.int64)
            ys, xs = ys[take], xs[take]
        dz, dn, miss = [], [], 0
        point_samples, normal_samples, pixel_samples = [], [], []
        for y, x in zip(ys, xs):
            origin, direction = ray_for_pixel(c2w, K, x, y, w, h)
            loc, normal, face, distance = bvh.ray_cast(origin, direction, 1e5)
            if loc is None:
                miss += 1
                continue
            p_cam = c2w[:3, :3].T @ (np.asarray(loc) - c2w[:3, 3])
            dz.append(float(dep[y, x] - p_cam[2]))
            # 反投影深度对应 GT 网格法线（理想时接近单位视线）。
            pc = np.linalg.inv(K) @ np.array([x, y, 1.0])
            nd = c2w[:3, :3] @ (pc / np.linalg.norm(pc))
            dn.append(float(np.dot(np.asarray(normal), nd)))
            point_samples.append(list(loc))
            normal_samples.append(list(normal))
            pixel_samples.append([int(x), int(y)])
        dz = np.asarray(dz)
        dn = np.asarray(dn)
        report["views"][stem] = {
            "id": numeric, "file_rgb": mc["file_rgb"], "file_depth": mc["file_depth"],
            "sample_count": len(dz), "miss_count": miss,
            "max_abs_axial_z_error_m": float(np.max(np.abs(dz))),
            "p99_abs_axial_z_error_m": float(np.percentile(np.abs(dz), 99)),
            "median_abs_axial_z_error_m": float(np.median(np.abs(dz))),
            "ray_normal_dot_p50": float(np.percentile(dn, 50)),
            "ray_normal_dot_p01": float(np.percentile(dn, 1)),
        }
        np.savez_compressed(out / (stem + "_gt_surface_samples.npz"),
                            points=np.asarray(point_samples, np.float32),
                            normals=np.asarray(normal_samples, np.float32),
                            pixels=np.asarray(pixel_samples, np.int32),
                            depth_errors_m=dz.astype(np.float32),
                            ray_normal_dot=dn.astype(np.float32))
        print(stem, report["views"][stem], flush=True)
    vals = [v["max_abs_axial_z_error_m"] for v in report["views"].values()]
    report["validation_pass"] = bool(vals and max(vals) < 2e-4)
    with open(out / "gt_surface_validation.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    if not report["validation_pass"]:
        raise RuntimeError("GT mesh/depth validation failed; do not issue geometry metrics")
else:
    if not args.ply or not args.selected or not args.face_mask or not args.sim3:
        raise SystemExit("gaussians mode requires --ply --selected --face-mask --sim3")
    from plyfile import PlyData
    vertex = PlyData.read(args.ply)["vertex"]
    means = np.stack([vertex[n] for n in ("x", "y", "z")], axis=1).astype(np.float64)
    sel = np.load(args.selected)
    sim = json.loads(Path(args.sim3).read_text(encoding="utf-8"))["sim3"]
    s = float(sim["s"])
    R = np.asarray(sim["R"], np.float64)
    T = np.asarray(sim["T"], np.float64)
    mask = np.asarray(load_png(args.face_mask)[..., 0]) > 0.5
    h, w = mask.shape
    Kn = scaled_K(K, w, h)
    report = {"mode": "gaussians_to_actual_gt_mesh", "face_mask": args.face_mask,
              "camera_fix": "C R C" if not args.no_fix else "none", "views": {}}
    for stem, numeric, c2w_pkg, mc in cameras():
        Rm = R @ c2w_pkg[:3, :3]
        Cm = s * (R @ c2w_pkg[:3, 3]) + T
        pc = (means[sel] - Cm) @ Rm
        z = pc[:, 2]
        u = Kn[0, 0] * pc[:, 0] / np.maximum(z, 1e-12) + Kn[0, 2]
        v = Kn[1, 1] * pc[:, 1] / np.maximum(z, 1e-12) + Kn[1, 2]
        ui, vi = np.floor(u + 0.5).astype(int), np.floor(v + 0.5).astype(int)
        in_face = (z > 0) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        good = np.where(in_face)[0]
        good = good[mask[vi[good], ui[good]]]
        distances, normal_alignment = [], []
        for j in good:
            world = ((means[sel[j]] - T) / s) @ R.T
            loc, normal, face, distance = bvh.find_nearest(Vector(world))
            distances.append(distance)
            # 仅记录中心到最近面方向是否与表面法线相容，不作为朝向学习质量结论。
            direction = (np.asarray(loc) - world)
            if np.linalg.norm(direction) > 1e-12:
                normal_alignment.append(abs(float(np.dot(direction / np.linalg.norm(direction),
                                                         np.asarray(normal)))))
        distances = np.asarray(distances)
        report["views"][stem] = {
            "id": numeric, "file_rgb": mc["file_rgb"], "file_depth": mc["file_depth"],
            "n": int(len(distances)),
            "nearest_mesh_distance_m": {
                "mean": float(distances.mean()) if len(distances) else None,
                "median": float(np.median(distances)) if len(distances) else None,
                "p90": float(np.percentile(distances, 90)) if len(distances) else None,
                "max": float(distances.max()) if len(distances) else None,
            },
        }
        np.savez_compressed(out / (stem + "_gaussian_mesh_distance.npz"),
                            selected_index=sel[good].astype(np.int64),
                            distance_m=distances.astype(np.float32),
                            pixels=np.stack([ui[good], vi[good]], axis=1).astype(np.int32))
        print(stem, report["views"][stem], flush=True)
    with open(out / "gaussian_mesh_audit.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
print("DONE", out, flush=True)
