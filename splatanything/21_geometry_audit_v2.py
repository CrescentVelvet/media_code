"""N0 五视角几何测量：先验证坐标契约，再报告 source/target 分离的误差。

官方 inference.py / rasterization.py / geometry.py 的坐标契约：
  GSN 对教师深度与相机平移施加独立的 pose-align/spacing 倍率，旋转保持正交。
  depth/*.npy 是浅拷贝保留的 GSN 前教师深度，不能与 PLY/相机共用尺度。
由 source 相机对估计的 Sim(3) 为 X_model=s R X_pkg+T，故：
  R_model=R R_pkg；C_model=s R C_pkg+T；X_pkg=R.T@(X_model-T)/s。
  PLY/ED 在 model 系，轴向深度除以 s 后才与米制 GT 比较。
ED 是 gsplat 的轴向 Z 期望深度，不是射线距离。
不再采用 scaled c2w：[sR|T]@prior 不是合法刚体相机。

本脚本不训练、不拟合 target、不覆盖历史 F3 或 target_metrics.json。
深度派生表面不是独立扫描网格；原始 FBX 缺失时不冒充网格真值。
"""
import argparse
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent


def load_mod(name, filename):
    spec = importlib.util.spec_from_file_location(name, str(ROOT / filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M03 = load_mod("render_eval03", "03_render_eval.py")
M13 = load_mod("face_select13", "13_face_select.py")
C_FLIP = np.diag([1.0, -1.0, -1.0])
MP = {"nose_tip": 4, "nose_bridge": 6, "eye_l_out": 263,
      "eye_l_in": 362, "eye_r_in": 133, "eye_r_out": 33,
      "brow_l": 105, "brow_r": 334, "mouth_l": 61, "mouth_r": 291,
      "chin": 152, "forehead": 10}
REGIONS = {
    "nose_tip": [4, 1, 2, 168, 197],
    "nose_ala": [48, 64, 98, 97, 278, 294, 327, 326],
    "eye_l": [263, 362, 386, 387, 373, 380, 374, 398, 466, 388],
    "eye_r": [33, 133, 159, 160, 158, 144, 145, 153, 246, 161],
    "brow_l": [105, 107, 66, 52, 65, 70, 63],
    "brow_r": [334, 336, 296, 282, 295, 300, 293],
}


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_depth(path):
    import OpenEXR
    exr = OpenEXR.File(str(path))
    if "depth.V" not in exr.channels():
        raise RuntimeError("Missing depth.V: " + str(path))
    z = exr.channels()["depth.V"].pixels.astype(np.float32)
    return z, np.isfinite(z) & (z > 1e-4) & (z < 1e8)


def load_rgb(path, size):
    rgba = np.asarray(Image.open(path), np.float32) / 255.0
    import cv2
    alpha = rgba[..., 3] if rgba.shape[-1] == 4 else np.ones(rgba.shape[:2], np.float32)
    rgb = np.asarray(Image.fromarray(np.uint8(np.clip(rgba[..., :3] * 255, 0, 255))).resize(
        size, Image.Resampling.LANCZOS), np.float32) / 255.0
    a = cv2.resize(alpha, size, interpolation=cv2.INTER_NEAREST)
    return rgb, a


def resize_mask(mask, size, erode=True):
    import cv2
    m = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    if erode:
        m = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return m


def model_camera(c2w_pkg, s, R, T):
    """将米制 pkg c2w 转成合法刚体 model c2w；缩放只施加于相机中心。"""
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R @ c2w_pkg[:3, :3]
    out[:3, 3] = s * (R @ c2w_pkg[:3, 3]) + T
    return out


def world_from_model(points, s, R, T):
    """行向量 model→pkg：X_pkg=R.T@(X_model-T)/s，对应 points@R/s。"""
    return ((points - T) @ R) / s


def official_preprocess_geometry(K, src_size, size=(756, 756)):
    """CPU 复现 scene_io.preprocess_view 的像素映射，返回 K 和坐标映射参数。

    本包是正方形、光心居中，但官方仍会 round 光心、round bbox 并施加
    0.5 pixel-center offset。K 检查用 float32 的相同运算链；几何投影使用
    精确输出分辨率与 crop offset，不把 float32 中间舍入累积到真实像素映射。
    """
    K0 = np.asarray(K, np.float32).copy()
    W, H = src_size
    cx, cy = K0[:2, 2].round().astype(int)
    mx, my = min(cx, W - cx), min(cy, H - cy)
    first_box = (int(cx - mx), int(cy - my), int(cx + mx), int(cy + my))
    crop_size = (int(2 * mx), int(2 * my))
    K1 = K0.copy()
    K1[:2, 2] -= np.array(first_box[:2], np.float32)
    requested = np.asarray(size, np.int64)
    scale_final = float(np.max(requested / np.asarray(crop_size)) + 1e-8)
    resized = tuple(np.floor(np.asarray(crop_size) * scale_final).astype(int))
    K2 = official_K_reference(K1, crop_size, requested)
    # bbox = round(K_resize.principal - K_target.principal)，第二遍以未缩放的 K 传入。
    target_K = camera_matrix_of_crop_reference(K2, resized, requested, 1.0)
    bbox_offset = np.int32(np.round(K2[:2, 2] - target_K[:2, 2]))
    return {"K_reference": K2, "first_box": first_box,
            "crop_size": crop_size, "resized_size": resized,
            "bbox_offset": bbox_offset.tolist(), "size": list(size),
            "scale_final": scale_final,
            "native_principal_rounding": K0[:2, 2].tolist()}

def camera_matrix_of_crop_reference(K, input_size, output_size, scale):
    """cropping.camera_matrix_of_crop 的 float32 CPU 参考实现。"""
    from src.utils.cropping import camera_matrix_of_crop
    return camera_matrix_of_crop(np.asarray(K, np.float32), input_size,
                                 output_size, scaling=scale, offset_factor=0.5)


def official_K_reference(K, input_size, output_size):
    """scene_io 实际调用的两步 K，不修改图片；作为 export K 的严格门禁。"""
    from src.utils.cropping import camera_matrix_of_crop, bbox_from_intrinsics_in_out
    K1 = np.asarray(K, np.float32).copy()
    resized = np.floor(np.asarray(input_size) *
                       (max(np.asarray(output_size) / input_size) + 1e-8)).astype(int)
    K1 = camera_matrix_of_crop(K1, input_size, resized,
                               scaling=max(np.asarray(output_size) / input_size) + 1e-8)
    K2 = camera_matrix_of_crop(K1, resized, output_size, scaling=1.0,
                               offset_factor=0.5)
    bbox = bbox_from_intrinsics_in_out(K1, K2, output_size)
    K2[:2, 2] -= bbox[:2]
    return K2


def remap_nearest(values, geometry):
    """用官方 cv2.resize 的同一缩放参数做 nearest，再按整数 bbox 裁切。"""
    x0, y0, x1, y1 = geometry["first_box"]
    crop = values[y0:y1, x0:x1]
    rw, rh = geometry["resized_size"]
    scale = geometry["scale_final"]
    is_bool = crop.dtype == np.bool_
    resized = cv2.resize(crop.astype(np.uint8 if is_bool else np.float32),
                         (rw, rh), fx=scale, fy=scale,
                         interpolation=cv2.INTER_NEAREST)
    ox, oy = geometry["bbox_offset"]
    ow, oh = geometry["size"]
    out = resized[oy:oy + oh, ox:ox + ow]
    if out.shape[:2] != (oh, ow):
        raise RuntimeError(f"Official crop reference shape mismatch: {out.shape} != {(oh, ow)}")
    return np.ascontiguousarray(out.astype(bool) if is_bool else out)


def official_preprocess_rgb(path, geometry):
    """按 scene_io 的整数 crop → Lanczos resize → integer crop 复现模型输入。"""
    x0, y0, x1, y1 = geometry["first_box"]
    im = Image.open(path).convert("RGB").crop((x0, y0, x1, y1))
    im = im.resize(tuple(geometry["resized_size"]), Image.Resampling.LANCZOS)
    ox, oy = geometry["bbox_offset"]
    ow, oh = geometry["size"]
    return np.asarray(im.crop((ox, oy, ox + ow, oy + oh)), np.float32) / 255.0


def detect_landmarks(rgb):
    import cv2
    import mediapipe as mp
    h, w = rgb.shape[:2]
    best = None
    for factor in (1.0, 0.75, 0.5, 0.35, 0.25):
        small = rgb if factor == 1 else cv2.resize(
            rgb, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_AREA)
        fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                             min_detection_confidence=0.3)
        out = fm.process(small)
        fm.close()
        if not out.multi_face_landmarks:
            continue
        points = np.array([[p.x * w, p.y * h]
                           for p in out.multi_face_landmarks[0].landmark], np.float64)
        oval = sorted({i for e in mp.solutions.face_mesh.FACEMESH_FACE_OVAL for i in e})
        box = points[oval].ptp(axis=0)
        if best is None or max(box) > best[0]:
            best = (max(box), factor, points)
    if best is None:
        raise RuntimeError("No FaceMesh for held-out target; refusing approximate ROI")
    return best[2], best[1]


def region_masks(landmarks, shape):
    import cv2
    h, w = shape
    masks = {}
    # MediaPipe FACEMESH_FACE_OVAL 连接表；不把检测库依赖带进渲染环境。
    oval_edges = ((10, 338), (338, 297), (297, 332), (332, 284), (284, 251),
                  (251, 389), (389, 356), (356, 454), (454, 323), (323, 361),
                  (361, 288), (288, 397), (397, 365), (365, 379), (379, 378),
                  (378, 400), (400, 377), (377, 152), (152, 148), (148, 176),
                  (176, 149), (149, 150), (150, 136), (136, 172), (172, 58),
                  (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
                  (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10))
    oval = sorted({i for e in oval_edges for i in e})
    for name, indices in REGIONS.items():
        points = landmarks[indices]
        center = points.mean(axis=0)
        radii = np.maximum(points.ptp(axis=0) * 0.65, [5, 5])
        if name.startswith("nose"):
            radii = np.maximum(radii, [12, 12])
        if name == "nose_ala":
            # 长轴覆盖左右两侧，侧视角会把鼻翼 ROI 拉到嘴角/头发；保守收窄。
            radii *= 0.55
        masks[name] = np.zeros((h, w), np.uint8)
        cv2.ellipse(masks[name], tuple(np.round(center).astype(int)),
                    tuple(np.ceil(radii).astype(int)), 0, 0, 360, 1, -1)
    tight = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(tight, cv2.convexHull(np.asarray(landmarks[oval], np.int32)), 1)
    masks["face_tight_landmark"] = tight.astype(bool)
    return {k: v.astype(bool) for k, v in masks.items()}


def load_normal(path):
    import OpenEXR
    channels = OpenEXR.File(str(path)).channels()
    return np.stack([channels["normal." + c].pixels for c in "XYZ"], axis=-1).astype(np.float32)


def quaternion_rotation(q):
    """PLY 的 [x,y,z,w] 四元数 -> 行主序旋转矩阵 [N,3,3]。"""
    q = np.asarray(q, np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    qx, qy, qz, qw = q.T
    return np.stack([
        np.stack([1 - 2 * (qy * qy + qz * qz),
                  2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)], axis=1),
        np.stack([2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz),
                  2 * (qy * qz - qx * qw)], axis=1),
        np.stack([2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw),
                  1 - 2 * (qx * qx + qy * qy)], axis=1)], axis=1)


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def stats(err, mask):
    err = np.asarray(err)[mask]
    err = err[np.isfinite(err)]
    if not len(err):
        return None
    return {"n": int(len(err)),
            "mae_mm": float(np.mean(np.abs(err)) * 1000),
            "rmse_mm": float(np.sqrt(np.mean(err ** 2)) * 1000),
            "bias_mm": float(np.mean(err) * 1000),
            "p90_abs_mm": float(np.percentile(np.abs(err), 90) * 1000)}


def point_to_depth_surface(points, gt_z, gt_valid, K, c2w):
    ys, xs = np.where(gt_valid)
    if len(xs) > 50000:
        take = np.linspace(0, len(xs) - 1, 50000).astype(np.int64)
        ys, xs = ys[take], xs[take]
    pc = (np.stack([xs, ys, np.ones_like(xs)], axis=1) *
          gt_z[ys, xs, None]) @ np.linalg.inv(K).T
    world = pc @ c2w[:3, :3].T + c2w[:3, 3]
    from scipy.spatial import cKDTree
    dist, _ = cKDTree(world).query(points)
    return dist


def depth_surface_normals(depth, valid, K, c2w):
    h, w = depth.shape
    yy, xx = np.indices((h, w))
    pc = (np.stack([xx, yy, np.ones_like(xx)], -1) * depth[..., None]) @ np.linalg.inv(K).T
    world = pc @ c2w[:3, :3].T + c2w[:3, 3]
    du = world[:, 2:] - world[:, :-2]
    dv = world[2:, :] - world[:-2, :]
    n = np.cross(du[1:-1], dv[:, 1:-1])
    norm = np.linalg.norm(n, axis=-1)
    normals = np.zeros_like(world)
    normals[1:-1, 1:-1] = n / np.maximum(norm[..., None], 1e-12)
    view = -pc[1:-1, 1:-1]
    view /= np.maximum(np.linalg.norm(view, axis=-1, keepdims=True), 1e-12)
    dots = np.abs(np.sum(normals[1:-1, 1:-1] * view, axis=-1))
    good = (norm > 1e-12) & valid[1:-1, 2:] & valid[1:-1, :-2] & \
           valid[2:, 1:-1] & valid[:-2, 1:-1]
    valid_n = np.zeros((h, w), bool)
    valid_n[1:-1, 1:-1] = good & (dots > 0.35) & valid[1:-1, 1:-1]
    return normals, valid_n


def normal_visibility_mask(depth, valid, K, c2w, rois, excluded):
    """诊断用可见性代理：几何法线朝向相机、无 2cm 邻域断层，排除已审误检。

    这不是材质分割，也不能区分头发与皮肤；因此与紧脸原始指标并列报告。
    """
    normals, valid_n = depth_surface_normals(depth, valid, K, c2w)
    yy, xx = np.indices(depth.shape)
    pc = (np.stack([xx, yy, np.ones_like(xx)], -1) * depth[..., None]) @ np.linalg.inv(K).T
    view = -pc / np.maximum(np.linalg.norm(pc, axis=-1, keepdims=True), 1e-12)
    facing = np.sum(normals * view, axis=-1) > .35
    mask = valid_n & facing
    for name in excluded:
        mask &= ~rois[name]
    return mask


def draw_error(rgb, err, mask, vmax=0.05):
    a = np.clip(np.abs(err) / vmax, 0, 1)
    a[~mask] = 0
    heat = np.stack([a, np.zeros_like(a), 1 - a], axis=-1)
    return Image.fromarray(np.uint8(np.clip(rgb * 0.65 + heat * 0.35, 0, 1) * 255))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--gt-pkg", required=True)
    ap.add_argument("--source-scene", required=True)
    ap.add_argument("--target-scene", required=True)
    ap.add_argument("--landmarks", required=True)
    ap.add_argument("--depth-validation", required=True)
    ap.add_argument("--roi-qa", required=True,
                    help="经目检的 ROI QA JSON，明确记录须排除的视角/区域")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt-camera-fixed", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--official-repo", required=True,
                    help="SplatAnything 官方仓，用于只读调用 preprocessing/cropping 参考实现")
    ap.add_argument("--sources-only", action="store_true")
    ap.add_argument("--gsn-probe", required=True,
                    help="已通过官方前向对账的 n0_gsn_forward_probe_v1 JSON")
    args = ap.parse_args()
    import sys
    sys.path.insert(0, str(Path(args.official_repo).resolve()))
    validation = read_json(args.depth_validation)
    gsn_probe = read_json(args.gsn_probe)
    required = {"export_depth_equals_raw_teacher_max", "export_camera_equals_norm_camera_max",
                "official_lifting_vs_export_unmerged_points_max"}
    if (gsn_probe.get("schema") != "n0_gsn_forward_probe_v1" or not required <= set(gsn_probe["checks"])
            or any(gsn_probe["checks"][k] > 1e-6 for k in required)):
        raise SystemExit("Official GSN forward/lifting gate has not passed")
    if len(gsn_probe["total_depth_factor"]) != 1:
        raise SystemExit("Expected one scene scale in GSN probe")
    gsn_depth_factor = float(gsn_probe["total_depth_factor"][0])
    gsn_pose_factor = float(gsn_probe["spacing_factor"][0])
    if not validation.get("validation_pass") or validation.get("convention") != "axial_z":
        raise SystemExit("Depth pass validation has not passed axial_z gate")
    roi_qa = read_json(args.roi_qa)
    roi_exclusions = {stem: list(q.get("excluded_rois", []))
                      for stem, q in roi_qa["views"].items()}
    res, pkg = Path(args.result), Path(args.gt_pkg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    with open(out / "audit_command.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"argv": sys.argv, "official_repo": str(Path(args.official_repo).resolve())},
                  f, ensure_ascii=False, indent=2)
    manifest = read_json(pkg / "cameras.json")
    K_gt = np.asarray(manifest["intrinsics"]["K"], np.float64)
    gt_size = (int(manifest["intrinsics"]["width"]), int(manifest["intrinsics"]["height"]))
    preprocess = official_preprocess_geometry(K_gt, gt_size)
    # 先验 K 对账：官方 preprocess_view 与审计参考必须逐项相等。
    from src.utils.scene_io import preprocess_view
    _, K_official = preprocess_view(Image.new("RGB", gt_size), np.asarray(K_gt, np.float32),
                                   (756, 756))
    if np.max(np.abs(K_official.numpy() - preprocess["K_reference"])) > 1e-5:
        raise RuntimeError("Internal official K replication mismatch")
    result_c2w, result_K, stems = M03.load_cameras(res / "cameras.json")
    source_prior = read_json(Path(args.source_scene) / "prior_camera.json")
    target_prior = read_json(Path(args.target_scene) / "prior_camera.json")
    target_ids = [] if args.sources_only else [c["camera_id"] for c in target_prior["extrinsics"]]
    stems += target_ids
    priors = {c["camera_id"]: c for c in source_prior["extrinsics"] + target_prior["extrinsics"]}
    if not args.gt_camera_fixed:
        for c in priors.values():
            c["matrix"] = np.asarray(c["matrix"], np.float64)
            c["matrix"][:3, :3] = C_FLIP @ c["matrix"][:3, :3] @ C_FLIP
    M15 = load_mod("face_target_eval15", "15_face_target_eval.py")
    s, R, T, sim3_fit = M15.estimate_sim3(res, Path(args.source_scene))
    test_pkg = np.asarray(priors[stems[0]]["matrix"], np.float64)
    test_center = model_camera(test_pkg, s, R, T)[:3, 3]
    inverse_delta = np.max(np.abs(world_from_model(test_center[None], s, R, T)[0]
                                  - test_pkg[:3, 3]))
    if inverse_delta > 2e-6:
        raise RuntimeError(f"model/pkg inverse camera-center check failed: {inverse_delta}")
    result_cameras = {stem: (c2w, K) for stem, c2w, K in zip(stems[:3], result_c2w, result_K)}
    views = []
    for i, stem in enumerate(stems):
        numeric = int(stem.rsplit("_", 1)[1])
        mc = next((c for c in manifest["cameras"] if int(c["id"]) == numeric), None)
        if mc is None:
            raise RuntimeError("No manifest camera " + stem)
        c2w_pkg = np.asarray(priors[stem]["matrix"], np.float64)
        c2w_model = model_camera(c2w_pkg, s, R, T)
        camera_error = None
        if stem in result_cameras:
            c2w_export, K_export = result_cameras[stem]
            camera_error = float(np.max(np.abs(c2w_export - c2w_model)))
            k_error = float(np.max(np.abs(K_export - preprocess["K_reference"])))
            if camera_error > 2e-6 or k_error > 1e-5:
                raise RuntimeError(f"Export camera/intrinsics differ from rigid Sim3: {stem}; "
                                   f"camera={camera_error}, K={k_error}")
        views.append(dict(stem=stem, numeric=numeric, mc=mc, c2w_pkg=c2w_pkg,
                          c2w_model=c2w_model, is_source=i < 3, camera_error=camera_error,
                          pose_factor=gsn_pose_factor if i < 3 else s))
    if abs(gsn_pose_factor - s) > 2e-6:
        raise RuntimeError(f"Source Sim3 scale={s} differs from GSN camera factor={gsn_pose_factor}")
    manifest_by_id = {int(c["id"]): c for c in manifest["cameras"]}
    lookat_target = np.asarray(read_json(pkg / "meta.json")["target"], np.float64)
    lookat = {}
    for v in views:
        versions = {"prior_CRC": v["c2w_pkg"],
                    "manifest_raw": np.asarray(v["mc"]["transform_cam2world"], np.float64)}
        lookat[v["stem"]] = {}
        for name, c2w in versions.items():
            pc = (lookat_target - c2w[:3, 3]) @ c2w[:3, :3]
            uv = K_gt[:2, :2] @ pc[:2] / pc[2] + K_gt[:2, 2]
            lookat[v["stem"]][name] = {
                "camera_xyz": pc.tolist(), "pixel": uv.tolist(),
                "offset_from_principal_point_px": float(np.linalg.norm(uv - K_gt[:2, 2])),
                "in_front": bool(pc[2] > 0)}
        if (lookat[v["stem"]]["prior_CRC"]["offset_from_principal_point_px"] > .01
                or not lookat[v["stem"]]["prior_CRC"]["in_front"]):
            raise RuntimeError("GT camera fails recorded look-at projection: " + v["stem"])
    gt_camera_validation = {}
    for stem, ce in zip(stems[:3], source_prior["extrinsics"]):
        prior_R = np.asarray(ce["matrix"], np.float64)[:3, :3]
        manifest_R = np.asarray(manifest_by_id[int(stem.rsplit("_", 1)[1])]["transform_cam2world"],
                                np.float64)[:3, :3]
        expected = C_FLIP @ manifest_R @ C_FLIP
        gt_camera_validation[stem] = {
            "prior_vs_C_manifest_C_max": float(np.abs(prior_R - expected).max()),
            "prior_vs_manifest_max": float(np.abs(prior_R - manifest_R).max()),
            "translation_unchanged": bool(np.allclose(np.asarray(ce["matrix"], np.float64)[:3, 3],
                    np.asarray(manifest_by_id[int(stem.rsplit("_", 1)[1])]["transform_cam2world"],
                               np.float64)[:3, 3], atol=1e-8))}
    saved_landmarks = read_json(args.landmarks)["views"]
    gt_camera_fixed = bool(args.gt_camera_fixed)
    for v in views[:3]:
        dep, valid_native = load_depth(pkg / v["mc"]["file_depth"])
        rgb_native, alpha_native = load_rgb(pkg / v["mc"]["file_rgb"], gt_size)
        valid_native &= alpha_native > .999
        teacher = np.load(res / "depth" / (v["stem"] + ".npy")).astype(np.float64)
        landmarks = np.asarray(saved_landmarks[v["stem"]]["points"], np.float64)
        masks = region_masks(landmarks, dep.shape)
        v["normal_visibility_native"] = normal_visibility_mask(
            dep, valid_native, K_gt, v["c2w_pkg"], masks, roi_exclusions.get(v["stem"], []))
        diagnostic = masks["face_tight_landmark"] & v["normal_visibility_native"]
        diagnostic = resize_mask(remap_nearest(diagnostic, preprocess), (756, 756))
        valid = resize_mask(remap_nearest(valid_native, preprocess), (756, 756))
        dep756 = remap_nearest(np.where(valid_native, dep, 0), preprocess)
        mask = diagnostic & valid & np.isfinite(teacher) & (teacher > 1e-3)
        ratio = dep756[mask] / teacher[mask]
        v["teacher"] = teacher
        v["teacher_scale_oracle"] = float(np.median(ratio))
        v["teacher_ratio_p10_p90"] = np.percentile(ratio, [10, 90]).tolist()
    source_scale = float(np.exp(np.mean(np.log([v["teacher_scale_oracle"] for v in views[:3]]))))
    g = M03.load_ply(res / "gaussians.ply")
    selected = np.load(res / "face_opt/select/selected_ids.npy")
    selected_pkg = world_from_model(g["means"][selected].astype(np.float64), s, R, T)
    report = {
        "schema": "geometry_audit_v2", "status": "measurement audit, not a branch verdict",
        "gt_camera_semantics": ("prior_camera.json stores C@R_manifest@C; numeric identity is checked "
                                "below; no second correction is applied") if gt_camera_fixed else
                               "apply legacy C@R_manifest@C once and record the transform",
        "gt_camera_validation": gt_camera_validation,
        "gt_physical_camera_version": "prior CRC passes the recorded look-at target projection in all selected views; camera-depth-normal consistency remains diagnostic, not independent mesh GT",
        "gt_lookat_projection": {"target": lookat_target.tolist(),
                                  "target_semantics": "render look-at target, not an anatomical facial landmark",
                                  "per_view": lookat},
        "preprocess_geometry": jsonable(preprocess),
        "depth_pass_validation": {"path": args.depth_validation, "convention": validation["convention"],
                                  "max_abs_error_axial_z": validation["max_abs_error_axial_z"]},
        "sim3": {"s": s, "R": R.tolist(), "T": T.tolist(), "source_only_fit": sim3_fit,
                 "source_is_reestimated_not_legacy_target_metrics": True,
                 "official_chain": "rasterization.normalize_gaussian_spacing: depth*=s; camera translation*=s; rotation unchanged",
                 "camera_rule": "R_model=R R_pkg; C_model=s R C_pkg+T; matrix must match exported c2w",
                 "depth_rule": "PLY/ED are in model units; divide by s for metric pkg. depth/*.npy is pre-GSN teacher depth and never uses this s",
                 "center_rule": "PLY means -> metric pkg: R.T@(X-T)/s; depths follow the same inverse-scale direction",
                 "rejected_probe": "[sR|T]@prior is non-rigid; apparent render agreement from using ED without s cannot validate it"},
        "roi_qa": {"path": args.roi_qa, "sha256": sha256(args.roi_qa), "content": roi_qa},
        "teacher_source_scale": source_scale,
        "teacher_scale_rule": "geometric mean of source tight-face robust ratios; targets never enter fitting",
        "gt_depth_resize": "official preprocess crop -> cv2 nearest with scale_final -> integer crop; erode once",
        "gsn_forward_probe": {"path": args.gsn_probe, "sha256": sha256(args.gsn_probe),
                              "pose_align_factor": gsn_probe["pose_align_factor"],
                              "spacing_factor": gsn_probe["spacing_factor"],
                              "total_depth_factor": gsn_probe["total_depth_factor"]},
        "roi_rule": "native FaceMesh tight oval and conservative semantic ellipses; face_visibility_filtered adds camera-facing depth-gradient visibility and removes explicitly reviewed bad ROIs; this is not semantic hair/skin segmentation",
        "gsn_input_hashes_verified": False,
        "limitations": ["expected depth is alpha-mixed, not a single surface",
                        "depth-backprojected surface and gradients are not independent scanned mesh",
                        "ROI masks require visual inspection"], "input_files": {}, "views": {}}
    for v in views:
        stem = v["stem"]
        dep, valid = load_depth(pkg / v["mc"]["file_depth"])
        rgb_native, alpha_native = load_rgb(pkg / v["mc"]["file_rgb"], gt_size)
        valid &= alpha_native > 0.999
        mapped = np.asarray(saved_landmarks[stem]["points"], np.float64)
        factor = saved_landmarks[stem]["detection_scale"]
        lm_delta = None
        # 地标由 21b 在 sam3/mediapipe 环境生成，本脚本不再依赖第二个检测环境。
        if stem in roi_qa["views"] and roi_qa["views"][stem].get("status") not in ("accepted", "partial"):
            raise RuntimeError("ROI QA not accepted: " + stem)
        excluded_rois = roi_exclusions.get(stem, [])
        if set(excluded_rois) - set(REGIONS):
            raise RuntimeError("Unknown ROI exclusion: " + stem)
        roi_native = region_masks(mapped, dep.shape)
        roi = {k: remap_nearest(m, preprocess) for k, m in roi_native.items()}
        face_tight = roi["face_tight_landmark"]
        # 深度图 nearest 采样可改变轮廓归属。平滑表面指标排除相邻像素
        # 差超过 2cm 的断层；另保留未过滤 face，不把两者混成一个数。
        grad_max = np.max(np.abs(np.stack(np.gradient(dep), axis=-1)), axis=-1)
        continuous_native = valid & (grad_max < 0.02)
        valid_native = valid.copy()
        valid = remap_nearest(valid, preprocess)
        alpha = remap_nearest(alpha_native, preprocess) > 0.999
        valid &= alpha
        smooth = resize_mask(remap_nearest(continuous_native, preprocess), (756, 756)) & alpha
        face_dir = res / "face_opt" / ("masks" if v["is_source"] else "masks_target")
        face = np.asarray(Image.open(face_dir / (stem + "_face.png"))) > 127
        person = np.asarray(Image.open(face_dir / (stem + "_person.png"))) > 127
        if face.shape != valid.shape:
            face = cv2.resize(face.astype(np.uint8), (756, 756), interpolation=cv2.INTER_NEAREST).astype(bool)
            person = cv2.resize(person.astype(np.uint8), (756, 756), interpolation=cv2.INTER_NEAREST).astype(bool)
        roi["face"] = face_tight
        roi["face_visibility_filtered"] = remap_nearest(
            roi_native["face_tight_landmark"] & normal_visibility_mask(
                dep, valid_native, K_gt, v["c2w_pkg"], roi_native, excluded_rois), preprocess)
        roi["legacy_face_mask"] = face
        roi["person_without_face"] = person & ~face_tight
        rgb = official_preprocess_rgb(pkg / v["mc"]["file_rgb"], preprocess)
        vm = {}
        if v["is_source"]:
            gt_input = np.asarray(Image.open(res / "rgb" / (stem + ".png")).convert("RGB"), np.float32) / 255
            input_delta = float(np.max(np.abs(rgb - gt_input)))
            if input_delta > 2.0 / 255:
                raise RuntimeError("GT preprocess differs from model input " + stem)
            c2w_export, K_export = result_cameras[stem]
            rd, al = M03.render(g, c2w_export, K_export, 756, 756,
                                background=0, device=args.device)
            rd, al = rd[0], al[0]
            vm["source_render_export_camera"] = {
                "rgb_mae": float(np.mean(np.abs(rd - gt_input))),
                "psnr": M03.psnr(rd, gt_input), "alpha_mean": float(al.mean()),
                "c2w_vs_rigid_sim3_max_error": v["camera_error"]}
            Image.fromarray(np.uint8(rd * 255)).save(out / (stem + "_render_export.png"))
            # 此处只检查官方导出相机，不再用两个坐标错误相互抵消的探针判方向。
        depth_native_gt = dep
        dep = remap_nearest(np.where(valid_native, dep, 0).astype(np.float32), preprocess)
        ed, render_alpha = M13.render_expected_depth(g, v["c2w_model"],
                                                    preprocess["K_reference"],
                                                    756, 756, device=args.device)
        # model 单位换算成米制 pkg：与 PLY 中心相同，施加逆尺度 1/s，不是乘 s。
        ed = np.asarray(ed).reshape(756, 756) / s
        lifting = None
        if v["is_source"]:
            teacher = np.load(res / "depth" / (stem + ".npy")).astype(np.float64)
            lifting = teacher * gsn_depth_factor / gsn_pose_factor
            vm["premerge_lifting_depth"] = {
                "formula": "raw_teacher * total_GSN_depth_factor / GSN_camera_factor",
                "source_camera_factor": gsn_pose_factor,
                "raw_teacher_total_GSN_depth_factor": gsn_depth_factor,
                "post_GSN_model_depth_factor_to_metric_pkg": gsn_depth_factor / gsn_pose_factor,
                "is_source_observation_not_target_fit": True,
                "raw": {}, "smooth": {}}
            for region, rm in roi.items():
                if region in excluded_rois:
                    vm["premerge_lifting_depth"]["raw"][region] = None
                    vm["premerge_lifting_depth"]["smooth"][region] = None
                    continue
                vm["premerge_lifting_depth"]["raw"][region] = stats(
                    lifting - dep, rm & valid & (lifting > 1e-4))
                vm["premerge_lifting_depth"]["smooth"][region] = stats(
                    lifting - dep, rm & smooth & (lifting > 1e-4))
            draw_error(rgb, lifting - dep, valid & face_tight, 0.05).save(
                out / (stem + "_premerge_lifting_error.png"))
        render_alpha = np.asarray(render_alpha).reshape(756, 756)
        vm.update({"id": v["numeric"], "is_source": v["is_source"],
              "files": {k: v["mc"][k] for k in ("file_rgb", "file_depth", "file_normal")},
              "export_vs_rigid_sim3_max_error": v["camera_error"],
              "render_alpha_mean": float(render_alpha.mean()),
              "render_alpha_gt_coverage": float((render_alpha[valid] > 0.5).mean()),
              "landmark_detection_scale": factor, "landmark_recheck_median_px": lm_delta,
              "excluded_rois_after_visual_qa": excluded_rois,
              "smooth_surface_gt_filter": "native max abs depth gradient < 20mm; contour values retained separately",
              "fused_expected_depth": {}, "fused_expected_depth_smooth": {}})
        for region, rm in roi.items():
            if region in excluded_rois:
                vm["fused_expected_depth"][region] = None
                continue
            mask = rm & valid & (ed > 1e-4)
            vm["fused_expected_depth"][region] = stats(ed - dep, mask)
            vm["fused_expected_depth_smooth"][region] = stats(ed - dep, rm & smooth & (ed > 1e-4))
        draw_error(rgb, ed - dep, valid & (face_tight | person), 0.05).save(out / (stem + "_fused_error.png"))
        if v["is_source"]:
            teacher = v["teacher"]
            vm["teacher_depth"] = {"source_scale": source_scale,
                                   "per_view_oracle_scale": v["teacher_scale_oracle"],
                                   "ratio_p10_p90": v["teacher_ratio_p10_p90"],
                                   "raw": {}, "source_scaled": {}, "per_view_oracle": {}}
            for key, mult in (("raw", 1), ("source_scaled", source_scale),
                              ("per_view_oracle", v["teacher_scale_oracle"])):
                for region, rm in roi.items():
                    mask = rm & valid & np.isfinite(teacher) & (teacher > 1e-3)
                    if region in excluded_rois:
                        vm["teacher_depth"][key][region] = None
                        continue
                    vm["teacher_depth"][key][region] = stats(teacher * mult - dep, mask)
            draw_error(rgb, teacher * source_scale - dep, valid & face_tight, 0.05).save(
                out / (stem + "_teacher_source_scaled_error.png"))
        vm["alpha_gt_iou"] = float(np.sum((render_alpha > 0.5) & valid) /
                                    max(np.sum((render_alpha > 0.5) | valid), 1))
        # 先按当前视角与语义 ROI 筛选中心，避免把同一批脸点重复算作不同区域。
        pc_all = (selected_pkg - v["c2w_pkg"][:3, 3]) @ v["c2w_pkg"][:3, :3]
        z_all = pc_all[:, 2]
        u_all = K_gt[0, 0] * pc_all[:, 0] / np.maximum(z_all, 1e-12) + K_gt[0, 2]
        v_all = K_gt[1, 1] * pc_all[:, 1] / np.maximum(z_all, 1e-12) + K_gt[1, 2]
        ui_all, vi_all = np.floor(u_all + 0.5).astype(int), np.floor(v_all + 0.5).astype(int)
        inside = (z_all > 0) & (ui_all >= 0) & (ui_all < gt_size[0]) & \
                 (vi_all >= 0) & (vi_all < gt_size[1])
        from scipy.spatial import cKDTree
        gt_surfaces = {}
        for surface_name, surface_valid in (("visible", valid_native), ("continuous", continuous_native)):
            ys, xs = np.where(surface_valid)
            if len(xs) > 50000:
                take = np.linspace(0, len(xs) - 1, 50000).astype(np.int64)
                ys, xs = ys[take], xs[take]
            pc_gt = (np.stack([xs, ys, np.ones_like(xs)], axis=1) *
                     depth_native_gt[ys, xs, None]) @ np.linalg.inv(K_gt).T
            world_gt = pc_gt @ v["c2w_pkg"][:3, :3].T + v["c2w_pkg"][:3, 3]
            gt_surfaces[surface_name] = cKDTree(world_gt)
        normals, valid_n = depth_surface_normals(depth_native_gt, valid_native, K_gt, v["c2w_pkg"])
        normal_gt = load_normal(pkg / v["mc"]["file_normal"])
        nn = np.linalg.norm(normal_gt, axis=-1)
        dots_pass = np.abs(np.sum(normals * normal_gt, axis=-1) / np.maximum(nn, 1e-12))
        valid_pass = valid_n & (nn > 0.5)
        if v["is_source"]:
            c2w_raw = np.asarray(manifest_by_id[v["numeric"]]["transform_cam2world"], np.float64)
            raw_normals, raw_valid = depth_surface_normals(
                depth_native_gt, valid_native, K_gt, c2w_raw)
            raw_dot = np.abs(np.sum(raw_normals * normal_gt, axis=-1) / np.maximum(nn, 1e-12))
            raw_valid &= nn > 0.5
            vm["gt_camera_pass_consistency"] = {
                "interpretation": "depth-gradient world normal vs supplied world normal; geometry self-consistency, not independent mesh GT",
                "camera_used": "C@R_manifest@C" if gt_camera_fixed else "R_manifest",
                "prior_vs_C_manifest_C_max": gt_camera_validation[stem]["prior_vs_C_manifest_C_max"],
                "prior_camera_used_median_abs_dot": float(np.median(dots_pass[valid_pass])),
                "prior_camera_used_p10_abs_dot": float(np.percentile(dots_pass[valid_pass], 10)),
                "manifest_camera_used_median_abs_dot": float(np.median(raw_dot[raw_valid])),
                "manifest_camera_used_p10_abs_dot": float(np.percentile(raw_dot[raw_valid], 10)),
                "n_prior": int(valid_pass.sum()), "n_manifest": int(raw_valid.sum()),
                "physical_version_verdict": "prior CRC is self-consistent" if
                    np.median(dots_pass[valid_pass]) > np.median(raw_dot[raw_valid]) + 0.05
                    else "unresolved; do not promote to independent GT"}
        normals, valid_n = depth_surface_normals(depth_native_gt, valid_native, K_gt, v["c2w_pkg"])
        for region, rm_native in roi_native.items():
            if region not in list(REGIONS) + ["face_tight_landmark"] or region in excluded_rois:
                continue
            sel_region = inside.copy()
            sel_region[inside] &= rm_native[vi_all[inside], ui_all[inside]]
            surface_stats = {}
            for surface_name, tree in gt_surfaces.items():
                dist, _ = tree.query(selected_pkg[sel_region])
                surface_stats[surface_name] = {
                    "n": int(len(dist)), "median": float(np.median(dist) * 1000) if len(dist) else None,
                    "p90": float(np.percentile(dist, 90) * 1000) if len(dist) else None,
                    "mean": float(np.mean(dist) * 1000) if len(dist) else None}
            vm.setdefault("selected_centers_to_backprojected_surface_mm", {})[region] = surface_stats
        face_native = roi_native["face_tight_landmark"]
        face_points = inside & face_native[np.clip(vi_all, 0, gt_size[1] - 1),
                                          np.clip(ui_all, 0, gt_size[0] - 1)]
        vm["selected_center_face_tight_n"] = int(face_points.sum())
        q = g["quats"][selected]
        # gsplat 取最大尺度主轴与视线的夹角；这不是扫描法线或 GT 法向误差。
        major = np.argmax(g["scales"][selected], axis=1)
        gaussian_normals = quaternion_rotation(q)[np.arange(len(q)), :, major]
        view_dirs = v["c2w_pkg"][:3, 3][None] - selected_pkg[face_points]
        view_dirs /= np.maximum(np.linalg.norm(view_dirs, axis=1, keepdims=True), 1e-12)
        normal_dots = np.abs(np.sum(gaussian_normals[face_points] * view_dirs, axis=1))
        vm["selected_gaussian_principal_axis_visibility_abs_dot"] = {
            "definition": "major covariance axis (largest scale) vs ray to camera; not a GT normal error",
            "n": int(len(normal_dots)),
            "p10": float(np.percentile(normal_dots, 10)) if len(normal_dots) else None,
            "median": float(np.median(normal_dots)) if len(normal_dots) else None,
            "status": "diagnostic only" if len(normal_dots) else "no valid selected centers; no normal conclusion"}
        im = Image.fromarray(np.uint8(np.clip(rgb_native, 0, 1) * 255))
        draw = ImageDraw.Draw(im)
        oval_indices = (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10)
        draw.line([tuple(mapped[i]) for i in oval_indices], fill=(0, 255, 0), width=3)
        for name, indices in REGIONS.items():
            pts = mapped[indices]
            color = (255, 40, 40) if name.startswith("nose") else (20, 180, 255)
            for x, y in pts:
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
            draw.text(tuple(pts.mean(axis=0)), name, fill=color)
        im.resize((756, 756), Image.LANCZOS).save(out / (stem + "_roi_audit.png"))
        for path in [pkg / v["mc"]["file_rgb"], pkg / v["mc"]["file_depth"],
                     pkg / v["mc"]["file_normal"], face_dir / (stem + "_face.png")]:
            report["input_files"][str(path)] = sha256(path)
        report["views"][stem] = vm
        print(stem, "fused face", vm["fused_expected_depth"]["face"],
              "render check", vm.get("source_render_export_camera"),
              "teacher oracle", vm.get("teacher_depth", {}).get("per_view_oracle_scale"), flush=True)
    report["input_files"][str(res / "config.yaml")] = sha256(res / "config.yaml")
    report["input_files"][str(Path(args.source_scene) / "prior_camera.json")] = sha256(
        Path(args.source_scene) / "prior_camera.json")
    for p in gsn_probe["input_sha256"]:
        report["input_files"][p] = gsn_probe["input_sha256"][p]
    report["gsn_input_hashes_verified"] = (
        gsn_probe["config_sha256"] == sha256(res / "config.yaml") and
        gsn_probe["camera_prior_sha256"] == sha256(Path(args.source_scene) / "prior_camera.json") and
        all(sha256(p) == h for p, h in gsn_probe["input_sha256"].items()))
    if not report["gsn_input_hashes_verified"]:
        raise RuntimeError("GSN probe input/config hashes do not match this audit")
    report["summary"] = {
        "status": "N0 measurement contract passed on this package; geometry/attribute branch remains undecided",
        "measurement_gates": {
            "manifest_file_mapping_verified": True,
            "source_only_sim3_and_gsn_hash_verified": True,
            "official_K_verified": True,
            "depth_pass_is_axial_z": True,
            "five_view_lookat_prior_crc_max_offset_px": max(
                x["prior_CRC"]["offset_from_principal_point_px"] for x in lookat.values()),
            "model_to_pkg_inverse_camera_center_max_error": float(inverse_delta),
            "max_export_rigid_sim3_camera_error": max(v["camera_error"] for v in views[:3]),
            "independent_fbx_mesh_available": False,
            "surface_metric_is_nearest_depth_backprojection_not_visibility_or_mesh": True},
        "teacher_source_scale": source_scale,
        "teacher_per_view_oracle_scales": [v["teacher_scale_oracle"] for v in views[:3]],
        "max_export_rigid_sim3_camera_error": max(v["camera_error"] for v in views[:3]),
        "fused_face_mae_mm": {v["stem"]: report["views"][v["stem"]]["fused_expected_depth"]["face"]["mae_mm"]
                               for v in views},
        "fused_visibility_filtered_face_mae_mm": {
            v["stem"]: report["views"][v["stem"]]["fused_expected_depth"]["face_visibility_filtered"]["mae_mm"]
            for v in views},
        "selected_face_center_count": {
            v["stem"]: report["views"][v["stem"]]["selected_center_face_tight_n"] for v in views},
        "nose_p90_mm": {v["stem"]: report["views"][v["stem"]]["fused_expected_depth"]["nose_tip"]["p90_abs_mm"]
                         for v in views},
        "not_a_final_acceptance_score": "raw tight-face includes hair/occlusion; excluded ROIs are explicitly null; target has no teacher/lifting because model exports source depths only"}
    with open(out / "geometry_audit_v2.json.tmp", "w", encoding="utf-8", newline="\n") as f:
        json.dump(jsonable(report), f, ensure_ascii=False, indent=2)
    (out / "geometry_audit_v2.json.tmp").replace(out / "geometry_audit_v2.json")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print("DONE", out)


if __name__ == "__main__":
    main()
