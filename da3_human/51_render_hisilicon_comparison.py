#!/usr/bin/env python3
"""海思各案例六路高斯 PLY 的内插相机渲染。

stage1 案例使用随包相机，位姿/尺度只在输入源相机上拟合；竞品保持原生完整 PLY。
补充 glTF 案例没有 stage1 相机，单独记录各自原生预测轨迹，不冒充六路同协议。

Env vars:
  ROOT, OUT_DIR, OUT_DIR_OVERRIDE, CASE_FILTER, FRAMES, FPS, RENDER_SIZE, FFMPEG, ZIP_RESULTS_DIR,
  ONLY_METHODS, SKIP_EXISTING, DECK_ASSETS, RESUME_INCOMPLETE, SMOKE,
  POSE_RESULTS_DIR, GSPLAT_CACHED_SO
"""
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation, Slerp

SCRIPT = Path(__file__).resolve().parent
FLIP_HOM = np.diag([1., -1., -1., 1.])
FLIP_CV = np.diag([1., -1., -1.])
METHODS = [
    ("competitor", "海思竞品", "#9B7C46"),
    ("da3", "DA3 原版", "#4267A9"),
    ("cameras", "DA3 · 采集位姿", "#3D807C"),
    ("sfm", "DA3 · SfM 位姿", "#76589C"),
    ("sa", "SplatAnything", "#A04E42"),
    ("zip", "ZipSplat", "#327D8D"),
]


def dump(path, data):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def module37():
    spec = importlib.util.spec_from_file_location(
        "hisilicon_compare", SCRIPT.parent / "splatanything/37_compare_hisilicon.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def npz(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def ext_to_c2w(cam):
    ext = cam["extrinsics"].astype(np.float64)
    if ext.ndim == 4:
        ext = ext[0]
    if ext.shape[-2:] == (3, 4):
        full = np.tile(np.eye(4), (len(ext), 1, 1))
        full[:, :3] = ext
        ext = full
    return np.linalg.inv(ext)


def native_k(cam, size):
    K = cam["intrinsics"].astype(np.float64)
    if K.ndim == 4:
        K = K[0]
    if "depth" in cam:
        base_w, base_h = cam["depth"].shape[-2:][::-1]
    elif "image_shape" in cam:
        base_h, base_w = np.asarray(cam["image_shape"]).reshape(-1)
    elif "intrinsics_base_size" in cam:
        base_w, base_h = cam["intrinsics_base_size"]
    else:
        raise ValueError("NPZ 无内参基准尺寸")
    K = K.copy()
    K[:, 0] *= size / float(base_w)
    K[:, 1] *= size / float(base_h)
    return K


def camera_path(case):
    if "_nopose_recentered" in case.name:
        return case / f"{case.name.removesuffix('_recentered')}_stage1_cameras.json"
    return case / f"{case.name.split('_', 1)[1]}_camera.json"


def load_capture(case, size):
    path = camera_path(case)
    data = json.loads(path.read_text(encoding="utf-8"))
    if "views" in data:
        gl = []
        K = []
        for view in data["views"]:
            T = np.eye(4)
            T[:3, :3] = np.asarray(view["rotation3x3"]).reshape(3, 3)
            T[:3, 3] = view["position"]
            gl.append(T)
            K.append([[view["fx"], 0., view["cx"]], [0., view["fy"], view["cy"]], [0., 0., 1.]])
        cv = np.asarray(gl) @ FLIP_HOM
        K = np.asarray(K)
        K[:, 0] *= size / data["imageWidth"]
        K[:, 1] *= size / data["imageHeight"]
        return cv, K, data, path
    ext = data["extrinsics"]
    gl = []
    for C, quat in zip(ext["camera_path"]["translation_keyframes"],
                       ext["camera_path"]["rotation_keyframes"]):
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(quat).as_matrix()
        # 仅这两包已用图像独立验证此解释，不能推广到其它 glTF。
        T[:3, 3] = -np.asarray(C)
        gl.append(T)
    fx = 352. / np.tan(float(ext["camera_path"]["yfov"][0][0]) / 2.) * size / 704.
    K = np.tile(np.asarray([[fx, 0., size / 2.], [0., fx, size / 2.], [0., 0., 1.]]),
                (len(gl), 1, 1))
    return np.asarray(gl) @ FLIP_HOM, K, data, path


def fit_similarity(src, dst):
    """X_dst=s R X_src+t；用所有源相机，不使用任何 target 图像。"""
    x_mean, y_mean = src.mean(0), dst.mean(0)
    u, _, vt = np.linalg.svd((dst - y_mean).T @ (src - x_mean))
    R = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
    scale = float(np.sum((dst - y_mean) * ((src - x_mean) @ R.T)) /
                  np.sum((src - x_mean) ** 2))
    t = y_mean - scale * R @ x_mean
    residual = np.linalg.norm(dst - (scale * src @ R.T + t), axis=1)
    if not np.isfinite(scale) or scale <= 0 or abs(np.linalg.det(R) - 1.) > 1e-5:
        raise ValueError("非法相机相似变换")
    return R, t, scale, {"max_center_error": float(residual.max()),
                         "mean_center_error": float(residual.mean())}


def fit_sa(capture, case):
    result = case / "splatanything/inference/splatanything"
    data = json.loads((result / "cameras.json").read_text(encoding="utf-8"))
    ids = [x["camera_id"] for x in data["extrinsics"]]
    model = np.stack([np.asarray(x["matrix"], dtype=np.float64)
                      for x in data["extrinsics"]])
    expected = [f"view_{i:02d}_source_{int(v['sourceIndex']):03d}"
                for i, v in enumerate(json.loads(camera_path(case).read_text(encoding="utf-8"))["views"])]
    if ids != expected:
        raise ValueError(f"SA 输入相机身份不匹配: {ids} != {expected}")
    M, s, t, report = fit_similarity(capture[:, :3, 3], model[:, :3, 3])
    # 严格源拟合 R_model=M R_capture；不要对旋转乘尺度。
    if np.abs(model[:, :3, :3] - M @ capture[:, :3, :3]).max() > 2e-4:
        raise ValueError(f"SA 源相机旋转映射失败: {case}")
    return M, s, t, report


def fit_source_rotation(capture, model):
    transforms = model[:, :3, :3] @ capture[:, :3, :3].transpose(0, 2, 1)
    u, _, vt = np.linalg.svd(transforms.mean(0))
    R = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
    return R, float(np.abs(transforms - R).max())


def video_probe(video, ffmpeg):
    exe = str(Path(ffmpeg).with_name("ffprobe"))
    result = subprocess.check_output([
        exe, "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=nb_read_frames,codec_name,width,height,pix_fmt",
        "-of", "json", str(video)], text=True)
    return json.loads(result)["streams"][0]


def probe_first_frame(video, ffmpeg, size):
    raw = subprocess.check_output([
        ffmpeg, "-v", "error", "-i", str(video), "-frames:v", "1",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    expected = size * size * 3
    if len(raw) != expected:
        raise ValueError(f"视频首帧尺寸/解码异常: {video}")
    return np.frombuffer(raw, dtype=np.uint8).reshape(size, size, 3).copy()


def trajectory(c2w, K, count):
    """关键机位自身参与内插，首尾相接；时间用弦长参数化，位置做 Catmull-Rom。"""
    centers = c2w[:, :3, 3]
    step = np.linalg.norm(np.diff(np.concatenate([centers, centers[:1]]), axis=0), axis=1)
    knots = np.concatenate([[0.], np.cumsum(step)])
    knots /= knots[-1]
    rot = Slerp(np.arange(len(c2w) + 1),
                Rotation.from_matrix(np.concatenate([c2w[:, :3, :3], c2w[:1, :3, :3]])))
    out, out_k = [], []
    for f in range(count):
        t = f / count
        i = min(np.searchsorted(knots, t, side="right") - 1, len(c2w) - 1)
        u = (t - knots[i]) / (knots[i + 1] - knots[i])
        im = (i - 1) % len(c2w)
        ip = (i + 1) % len(c2w)
        ipp = (i + 2) % len(c2w)
        # 局部 Catmull-Rom 只作用于相邻机位之间；端点严格经过输入相机。
        c = .5 * ((2 * centers[i]) + (-centers[im] + centers[ip]) * u +
                  (2 * centers[im] - 5 * centers[i] + 4 * centers[ip] - centers[ipp]) * u ** 2 +
                  (-centers[im] + 3 * centers[i] - 3 * centers[ip] + centers[ipp]) * u ** 3)
        T = np.eye(4)
        T[:3, :3] = rot([i + u]).as_matrix()[0]
        T[:3, 3] = c
        out.append(T)
        out_k.append(K[i] * (1 - u) + K[ip] * u)
    return np.asarray(out), np.asarray(out_k)


def native_trajectory(c2w, K, count):
    return trajectory(c2w, K, count)


def transform_path(path, R, t, scale):
    out = path.copy()
    out[:, :3, :3] = R @ path[:, :3, :3]
    out[:, :3, 3] = scale * (R @ path[:, :3, 3].T).T + t
    return out


def sa_time(case):
    path = case / "splatanything/inference.log"
    if path.is_file():
        import re
        match = re.search(r"耗时\s+([0-9.]+)\s*s", path.read_text(encoding="utf-8"))
        if match:
            return float(match.group(1)), "单例含加载及导出，日志口径"
    return None, "未找到对应记录"


def zip_result_dir(case, zip_root):
    name = case.name.split("_", 1)[1] if "_nopose_recentered" not in case.name else case.name
    path = zip_root / name / "predicted_cameras.npz"
    if not path.is_file():
        raise FileNotFoundError(f"缺少 ZipSplat 原生相机: {path}")
    return path.parent


def zip_time(case, zip_root):
    path = zip_result_dir(case, zip_root) / "metrics.json"
    if path.is_file():
        timing = json.loads(path.read_text(encoding="utf-8"))["timing_seconds"]
        return timing["total"], "单例含加载、TokenBA、回放"
    return None, "未找到对应记录"


class GaussianRenderer:
    def __init__(self, path, quat="passthrough"):
        # 本机没有 nvcc，但已有同一 torch/CUDA ABI 的编译扩展。
        # 无 GSPLAT_CACHED_SO 时保持原环境行为，不硬编码机器路径。
        cached = os.environ.get("GSPLAT_CACHED_SO", "")
        if cached:
            if not Path(cached).is_file():
                raise FileNotFoundError(f"GSPLAT_CACHED_SO 不存在: {cached}")
            spec = importlib.util.spec_from_file_location("gsplat_cuda", cached)
            extension = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(extension)
            sys.modules["gsplat_cuda"] = extension
            sys.modules["gsplat.csrc"] = extension
            import gsplat
            gsplat.csrc = extension
        from gsplat import rasterization
        if not callable(rasterization):
            raise RuntimeError("gsplat 光栅化入口不可用")
        self.rasterization = rasterization
        g = module37().load_gaussians(path, quat)
        self.meta = {"points": g["point_count"], "sh_degree": g["degree"]}
        self.degree = g["degree"]
        self.gpu = {k: torch.from_numpy(g[k]).cuda() for k in
                    ("means", "scales", "opacities", "quats", "sh")}
        self.background = torch.ones(3, device="cuda")

    def render(self, c2w, K, size):
        with torch.no_grad():
            w2c = torch.from_numpy(np.linalg.inv(c2w)[None]).float().cuda()
            intr = torch.from_numpy(K[None]).float().cuda()
            rgb, alpha, _ = self.rasterization(
                self.gpu["means"], self.gpu["quats"], self.gpu["scales"],
                self.gpu["opacities"], self.gpu["sh"], w2c, intr, size, size,
                sh_degree=self.degree, packed=False, eps2d=.01, near_plane=.0001,
                far_plane=10000., backgrounds=self.background[None], render_mode="RGB")
        return rgb[0].clamp(0, 1).mul(255).byte().cpu().numpy(), float(alpha.mean())

    def close(self):
        del self.gpu
        torch.cuda.empty_cache()


def locate(case, method, zip_root, pose_root):
    if method == "competitor":
        if "_nopose_recentered" not in case.name:
            return None
        return case / f"{case.name.removesuffix('_recentered')}_stage1_recentered.ply"
    if method == "da3":
        return case / "depthanything3.ply"
    if method == "cameras":
        path = case / "depthanything3_cameras.ply"
        return path if path.is_file() else None
    if method == "sfm":
        path = case / "depthanything3_sfm.ply"
        return path if path.is_file() else None
    if method == "sa":
        path = case / "splatanything.ply"
        return path if path.is_file() else None
    if method == "zip":
        path = case / "zipsplat.ply"
        return path if path.is_file() else None
    raise ValueError(method)


def competitor_trajectory(capture, K, metadata, count, case):
    """源相机到 recentered PLY 显示世界的可复核映射。

    源中心先加 recenterOffset，再取主体正面侧；显示方向看向 bounds
    中心，显示距离按随包 min/maxDistance 约束。保留所有源机位与内参，
    不裁点、不筛点；不把原始 position 当作重居中世界的米制平移。
    """
    capture = np.asarray(capture, dtype=np.float64)
    if metadata.get("recenterOffset") is None or len(capture) < 3:
        raise ValueError("竞品重居中相机缺少 recenterOffset 或源机位不足")
    bounds_path = case / f"{case.name.removesuffix('_recentered')}_stage1_bounds.json"
    bounds = json.loads(bounds_path.read_text(encoding="utf-8"))
    if not bounds.get("recentered") or bounds["minDistance"] <= 0:
        raise ValueError(f"竞品缺少重居中显示距离范围: {bounds_path}")
    target = np.array([bounds["centerX"], bounds["centerY"], bounds["centerZ"]])
    centers = -(capture[:, :3, 3] + np.asarray(metadata["recenterOffset"])) - target
    radius = np.linalg.norm(centers, axis=1)
    if radius.min() <= 0 or not np.isfinite(capture).all():
        raise ValueError("竞品源机位中心非法")
    scales = np.clip(radius, bounds["minDistance"], bounds["maxDistance"]) / radius
    outliers = np.flatnonzero(np.abs(scales - 1) > 1e-6).tolist()
    display_roll = 180.
    if case.name == "case14_nopose_recentered":
        # 此例 view3 的源半径约为其余机位的十二倍，直接把轨迹带到屋顶远景。
        # 仅用随包初始显示距离约束该离群机位；135° 是逐帧目检确定的图像横滚。
        outliers = np.flatnonzero(radius > 3 * np.median(radius)).tolist()
        if outliers != [3]:
            raise ValueError(f"case14 离群源机位与已核验记录不一致: {outliers}")
        scales[outliers] = bounds["initDistance"] / radius[outliers]
        display_roll = 135.
    centers = centers * scales[:, None] + target
    if case.name == "case7_nopose_recentered":
        # 该例原始机位含高俯角，沿用球面方向会在中段穿过人物头顶。
        # 保留各源机位水平方位，以水平半径中位数做展示环绕。
        horizontal = centers - target
        horizontal[:, 1] = 0.
        horizontal_radius = np.linalg.norm(horizontal, axis=1)
        centers = horizontal / horizontal_radius[:, None] * np.median(horizontal_radius) + target
    forward = target - centers
    forward /= np.linalg.norm(forward, axis=1, keepdims=True)
    up = np.tile([0., 1., 0.], (len(centers), 1))
    x = np.cross(up, -forward)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    y = np.cross(-forward, x)
    gl = np.tile(np.eye(4), (len(centers), 1, 1))
    gl[:, :3, :3] = np.stack([x, y, -forward], axis=2)
    gl[:, :3, 3] = centers
    roll = np.eye(4)
    roll[:3, :3] = Rotation.from_euler("z", display_roll, degrees=True).as_matrix()
    path, path_K = trajectory(gl @ FLIP_HOM @ roll, K, count)
    return path, path_K, {
        "bounds_file": str(bounds_path),
        "source_radius_after_recenter": radius.tolist(),
        "distance_scale": scales.tolist(),
        "display_center": target.tolist(),
        "outlier_source_indices": outliers,
        "display_roll_degrees": display_roll,
        "case14_outlier_rule": "radius>3*median(source radii): use supplied initDistance; visual roll=135deg" if case.name.startswith("case14_") else None,
        "case7_display_orbit_rule": "preserve source horizontal directions; median horizontal radius; upright look-at; not exact original camera attitudes" if case.name.startswith("case7_") else None,
        "recenter_rule": "C_display=-(C_source+recenterOffset)-center; radial display distance clipped to supplied bounds; case14 extreme source outlier uses initDistance",
        "rotation_rule": "look_at(display_center,bounds.center) with source-derived direction; case-specific upright display roll",
        "mapping_evidence": "Rendering-based display mapping, not verified against original exporter source",
    }


def setup_method(case, method, capture, K, count, size, zip_root, pose_root):
    """返回 (原生世界轨迹, 原生内参, 协议)；补充包不伪造共同世界。"""
    is_stage1 = "_nopose_recentered" in case.name
    path, path_K = trajectory(capture, K, count) if is_stage1 else (None, None)
    subdir = "depthanything3" if method == "da3" else f"depthanything3_{method}"
    summary_path = case / subdir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    if method == "competitor":
        if not is_stage1:
            return None, None, None
        native_path, native_K, audit = competitor_trajectory(
            capture, K, json.loads(camera_path(case).read_text(encoding="utf-8")), count, case)
        return native_path, native_K, {
            "protocol": "重居中竞品：源机位加recenterOffset后映射至主体正面侧，朝向bounds中心并约束显示距离；全源机位闭环内插",
            "source_camera_transform": "C_display=-(C_source+recenterOffset)-center; clamp radial distance to bounds; look_at(center) @ flip_cv @ roll_z_180",
            "visual_validation": "case1/7/14 十源机位的重居中与两侧诊断；17例多机位辅助抽检",
            "coordinate_units": "recentered PLY native units; not metric",
            "not_quality_certification": True, **audit}
    if method == "da3":
        cam = npz(case / "depthanything3/cameras.npz")
        model_c2w = ext_to_c2w(cam)
        if is_stage1:
            R, t, scale, report = fit_similarity(capture[:, :3, 3], model_c2w[:, :3, 3])
            return transform_path(path, R, t, scale), path_K, {
                "protocol": "十个输入相机中心拟合 Sim(3)；不使用留出图像", **report}
        native_path, native_K = native_trajectory(model_c2w, native_k(cam, size), count)
        return native_path, native_K, {"protocol": "补充样例：原生预测轨迹，无 stage1 共同世界"}
    if method == "cameras":
        cam = npz(case / "depthanything3_cameras/cameras.npz")
        if is_stage1 and "input_extrinsics_original" in cam:
            R = np.asarray(cam["world_rotation"] if "world_rotation" in cam
                           else cam["original_world_rotation"], dtype=np.float64)
            if "world_rotation" in summary and not np.allclose(R, summary["world_rotation"]):
                raise ValueError(f"NPZ 与摘要的显示旋转不一致: {case.name}")
            return transform_path(path, R, np.zeros(3), 1.), path_K, {
                "protocol": summary.get("gaussian_frame", "stage1世界显示旋转"),
                "n_source": 8, "not_quality_certification": True}
        if is_stage1:
            R = np.asarray(summary["gaussian_rotation"])
            t = np.asarray(summary["gaussian_translation"])
            return transform_path(path, R, t, 1.), path_K, {
                "protocol": "历史10源输入世界的显示旋转版", "n_source": 10}
        native_path, native_K = native_trajectory(ext_to_c2w(cam), native_k(cam, size), count)
        return native_path, native_K, {"protocol": "补充样例：原生位姿条件轨迹，无 stage1 共同世界"}
    if method == "sfm":
        cam = npz(case / "depthanything3_sfm/cameras.npz")
        model_c2w = ext_to_c2w(cam)
        if not is_stage1:
            return None, None, None
        # PLY 严格位于这些实际 SfM 输入相机所在的世界。SfM 估计并不保证
        # 与 stage1 采集相机构成单一刚体，不能用全轨迹强制拟合掩盖该事实。
        source = np.linalg.inv(cam["input_extrinsics_original"])
        names = list(map(str, cam["image_names"]))
        meta = json.loads(camera_path(case).read_text(encoding="utf-8"))
        source_names = [f"view_{i:02d}_source_{int(v['sourceIndex']):03d}.png"
                        for i, v in enumerate(meta["views"])]
        ids = [source_names.index(name) for name in names]
        source_K = np.asarray(cam["input_intrinsics_original"], dtype=np.float64).copy()
        source_K[:, 0] *= size / float(meta["imageWidth"])
        source_K[:, 1] *= size / float(meta["imageHeight"])
        path, path_K = trajectory(source, source_K, count)
        rel = source[:, :3, :3] @ capture[ids, :3, :3].transpose(0, 2, 1)
        R, t, scale, report = fit_similarity(capture[ids, :3, 3], source[:, :3, 3])
        rot_degrees = Rotation.from_matrix(rel @ R.T).magnitude() * 180. / np.pi
        return path, path_K, {
            "protocol": "实际 SfM 输入相机关键帧及对应 SfM 内参闭环内插",
            "coordinate_units": "SfM arbitrary units; not metric",
            "capture_keyframe_differences": {
                "rotation_max_deg_after_similarity": float(rot_degrees.max()),
                "rotation_mean_deg_after_similarity": float(rot_degrees.mean()),
                **report},
            "registration_limit": "不强制共用一个 stage1 刚体；保留 SfM 实际位姿，不作几何评分"}
    if method == "sa":
        if not is_stage1:
            return None, None, None
        R, t, scale, report = fit_sa(capture, case)
        return transform_path(path, R, t, scale), path_K, {
            "protocol": "SA 已有源相机契约重新核验", **report}
    if method == "zip":
        cam = npz(zip_result_dir(case, zip_root) / "predicted_cameras.npz")
        model_c2w = cam["c2w"].astype(np.float64)
        if not is_stage1:
            zK = np.zeros((len(model_c2w), 3, 3))
            for i, c in enumerate(cam["cameras"]):
                w, h, fx, fy, cx, cy = c
                zK[i] = [[fx * size / w, 0., cx * size / w],
                         [0., fy * size / h, cy * size / h], [0., 0., 1.]]
            native_path, native_K = native_trajectory(model_c2w, zK, count)
            return native_path, native_K, {"protocol": "补充样例：原生预测轨迹，无 stage1 共同世界"}
        R, t, scale, report = fit_similarity(capture[:, :3, 3], model_c2w[:, :3, 3])
        return transform_path(path, R, t, scale), path_K, {
            "protocol": "ZipSplat 原生预测相机与 stage1 源相机拟合 Sim(3)", **report}
    raise ValueError(method)


def font(size, bold=False):
    path = Path("/mnt/c/Windows/Fonts") / ("msyhbd.ttc" if bold else "msyh.ttc")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def poster(frames, methods, size=216):
    cell_w, gap, margin, head, foot = 368, 16, 60, 60, 42
    width = margin * 2 + cell_w * 3 + gap * 2
    height = head + (cell_w + 58) * ((len(frames) + 2) // 3) + gap + foot
    im = Image.new("RGB", (width, height), "#F5F5F5")
    draw = ImageDraw.Draw(im)
    for i, ((key, label, color), rgb) in enumerate(zip(methods, frames)):
        x = margin + (i % 3) * (cell_w + gap)
        y = head + (i // 3) * (cell_w + 58 + gap)
        draw.text((x + 8, y), label, font=font(22, True), fill=color)
        tile = Image.fromarray(rgb).resize((cell_w, cell_w), Image.Resampling.LANCZOS)
        im.paste(tile, (x + 8, y + 30))
    return im


def open_encoder(fps, out, ffmpeg, size):
    """直接编码渲染帧，避免 /mnt/d 的 9p 文件系统逐张 PNG 往返。"""
    return subprocess.Popen([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(fps),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size}x{size}", "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "19", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)


def encode(raw_paths, fps, out, ffmpeg):
    """兼容旧 PNG 审计帧目录；正式新渲染走 open_encoder。"""
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(fps),
           "-i", str(raw_paths / "frame_%04d.png"), "-an", "-c:v", "libx264", "-preset", "fast",
           "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)
    return video_probe(out, ffmpeg)


def main():
    root = Path(os.environ.get("ROOT", "/mnt/d/dataset/测试数据/海思端侧重建拍摄数据"))
    out = Path(os.environ.get("OUT_DIR", "/mnt/d/output/results/da3_human_results/hisilicon_video_comparison"))
    override = os.environ.get("OUT_DIR_OVERRIDE", "")
    if override:
        out = Path(override)
        if not out.is_absolute():
            raise ValueError("OUT_DIR_OVERRIDE 必须是未占用的绝对路径")
        if out.exists():
            raise FileExistsError(f"OUT_DIR_OVERRIDE 已存在，拒绝复用旧协议产物: {out}")
    elif (out / "protocol_version.json").is_file():
        previous = json.loads((out / "protocol_version.json").read_text(encoding="utf-8"))
        if previous.get("version") != 3:
            raise RuntimeError("OUT_DIR 属于旧展示协议；请使用全新的 OUT_DIR_OVERRIDE")
    zip_root = Path(os.environ.get("ZIP_RESULTS_DIR", "/mnt/d/output/results/zipsplat_results/hisilicon_10v"))
    pose_root = Path(os.environ.get("POSE_RESULTS_DIR", "/mnt/d/output/results/da3_human_results/hisilicon_camera_ab_20261126/stage1"))
    selected = {x for x in os.environ.get("CASE_FILTER", "").split(",") if x}
    smoke = os.environ.get("SMOKE", "0") == "1"
    frames = int(os.environ.get("FRAMES", "12" if smoke else "120"))
    fps = int(os.environ.get("FPS", "15"))
    size = int(os.environ.get("RENDER_SIZE", "704"))
    ffmpeg = os.environ.get("FFMPEG", "/home/velvet/miniconda3/envs/vggt_human/bin/ffmpeg")
    only = {x for x in os.environ.get("ONLY_METHODS", "").split(",") if x}
    if any("_nopose_recentered" in name for name in selected) and zip_root.name != "hisilicon_10v":
        raise ValueError("旧17例必须使用 hisilicon_10v 的相机记录")
    if any(name.startswith("case18") or name.startswith("case19") for name in selected):
        # glTF 两补充包的 Zip 记录沿用原 hisilicon_10v 的 model_* 目录。
        zip_root = zip_root.parent / "hisilicon_10v"
    cases = sorted((p for p in root.iterdir() if p.is_dir() and
                    p.name.split("_")[0].removeprefix("case").isdigit() and
                    (not selected or p.name in selected)),
                   key=lambda p: int(p.name.split("_")[0].removeprefix("case")))
    if not cases:
        raise FileNotFoundError("没有匹配案例")
    out.mkdir(parents=True, exist_ok=True)
    dump(out / "protocol_version.json", {
        "version": 3,
        "competitor_protocol": "recentered_v1: source centers + recenterOffset mapped to front side; bounds center/distance and upright look-at",
        "competitor_mapping_evidence": "Rendering-based display mapping; original exporter contract not verified",
        "case7_display_orbit_rule": "Preserve source horizontal directions; median horizontal radius; upright look-at; not exact original camera attitudes",
        "case14_outlier_rule": "radius>3*median(source radii): use supplied initDistance; visual roll=135deg",
        "video": "704x704, 120 frames, 15 fps, H.264/yuv420p",
        "trajectory": "对应方法原生输入相机的关键机位闭环内插",
        "stage1_methods": ["competitor", "da3", "cameras", "sa", "zip"],
        "native_sfm": "实际 SfM 输入相机及内参，不强制与 stage1 单刚体一致",
        "supplementary": "case18/19 各自原生预测轨迹，无共同相机",
        "sha256_required": True, "gaussian_pruning": False})
    if not Path(ffmpeg).is_file() or not Path(ffmpeg).with_name("ffprobe").is_file():
        raise FileNotFoundError("ffmpeg/ffprobe 不完整")
    for case in cases:
        started = time.time()
        case_out = out / case.name
        case_out.mkdir(exist_ok=True)
        capture, K, metadata, camera_file = load_capture(case, size)
        metadata_path = case_out / ("smoke_summary.json" if smoke else "summary.json")
        old = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {"methods": []}
        methods_by_key = {m["key"]: m for m in old.get("methods", [])}
        if len(methods_by_key) != len(old.get("methods", [])):
            raise ValueError(f"旧整例摘要存在重复方法，不能续跑: {case.name}")
        retained = set(methods_by_key) - only if only else set()
        missing = only - set(methods_by_key)
        if missing:
            raise ValueError(f"指定方法缺少旧登记，不能安全续跑: {case.name}/{sorted(missing)}")
        methods = []
        first_frames = []
        poster_frames = {}
        for key, label, color in METHODS:
            if only and key not in only:
                if key in retained:
                    methods.append(methods_by_key[key])
                    frame_file = case_out / f"{key}_poster.png"
                    if frame_file.is_file():
                        with Image.open(frame_file) as frame:
                            poster_frames[key] = np.asarray(frame.convert("RGB")).copy()
                continue
            ply = locate(case, key, zip_root, pose_root)
            if ply is None or not ply.is_file():
                methods.append({"key": key, "label": label, "available": False, "reason": "此案例无对应 PLY"})
                print(f"⚠️ {case.name}/{key}: 无对应 PLY", flush=True)
                continue
            renderer = None
            encoder = None
            path, render_K, protocol = setup_method(case, key, capture, K, frames, size, zip_root, pose_root)
            if path is None:
                methods.append({"key": key, "label": label, "available": False, "reason": "补充包没有对应重建"})
                continue
            if len(render_K) != len(path):
                raise ValueError("轨迹与内参数量不一致")
            raw_dir = case_out / "frames" / key
            video = case_out / f"{key}.mp4"
            if smoke:
                video = case_out / f"{key}_smoke.mp4"
            skip = os.environ.get("SKIP_EXISTING", "1") == "1" and video.is_file() and not smoke
            if skip:
                old_method = None
                per_method = case_out / f"{key}_summary.json"
                if per_method.is_file():
                    old_method = json.loads(per_method.read_text(encoding="utf-8"))["method"]
                else:
                    old_method = next((x for x in old["methods"] if x["key"] == key), None)
                if skip:
                    if old_method is None:
                        raise ValueError(f"已有视频没有登记，不能复用: {video}")
                    if old_method.get("camera_protocol") != protocol:
                        raise ValueError(f"已有视频协议不匹配，不能复用: {video}")
                    if old_method.get("render_complete") is not True:
                        # 缺字段的旧登记不是完整正式结果，必须重渲染。
                        if "render_complete" not in old_method:
                            print(f"⚠️ 重渲染旧版缺字段登记: {video}", flush=True)
                            skip = False
                        elif old_method.get("video_sha256") and sha256(video) == old_method["video_sha256"]:
                            raise ValueError(f"中断登记与视频哈希一致，需明确重渲染: {video}")
                        else:
                            print(f"⚠️ 重渲染中断视频: {video}", flush=True)
                            skip = False
                    elif not old_method.get("video_sha256") or sha256(video) != old_method["video_sha256"]:
                        raise ValueError(f"完整视频哈希不匹配，不能复用: {video}")
            if skip:
                print(f"⏭️ {case.name}/{key}: 已有视频", flush=True)
                probe = video_probe(video, ffmpeg)
                if (probe.get("codec_name") != "h264" or
                        int(probe["nb_read_frames"]) != frames or
                        [probe["width"], probe["height"]] != [size, size]):
                    raise ValueError(f"已有视频规格不符，不能跳过: {video}")
                first = probe_first_frame(video, ffmpeg, size)
                alpha_mean = None
                old_method = (json.loads(per_method.read_text(encoding="utf-8"))["method"]
                              if per_method.is_file() else
                              next(x for x in old["methods"] if x["key"] == key))
                meta = {"points": old_method["points"], "sh_degree": old_method["sh_degree"]}
                alpha_mean = old_method.get("mean_alpha")
                video_probe_data = probe
            else:
                # 旧 PNG 恢复不用于正式渲染：它无法证明本协议、完整高斯及完整帧数。
                if raw_dir.exists() and not raw_dir.is_symlink():
                    raise FileExistsError(f"残留帧目录需先审计并移出，不能静默恢复或删除: {raw_dir}")
                if raw_dir.is_symlink():
                    raise ValueError(f"临时帧目录为链接，拒绝处理: {raw_dir}")
                print(f"🚀 {case.name}/{key}: {ply.name} / {frames} 帧", flush=True)
                renderer = GaussianRenderer(ply, "passthrough")
                alpha_values = []
                partial_video = video.with_name(video.stem + ".partial.mp4")
                encoder = open_encoder(fps, partial_video, ffmpeg, size)
                try:
                    for i, (T, intr) in enumerate(zip(path, render_K)):
                        if not np.isfinite(T).all() or not np.isfinite(intr).all():
                            raise ValueError(f"非有限相机: {case.name}/{key}/{i}")
                        rgb, alpha = renderer.render(T, intr, size)
                        if i == 0:
                            first = rgb.copy()
                        encoder.stdin.write(rgb.tobytes())
                        alpha_values.append(alpha)
                        if i % 30 == 0 or i + 1 == len(path):
                            print(f"  {key}: {i + 1}/{len(path)}", flush=True)
                finally:
                    if encoder.stdin:
                        encoder.stdin.close()
                        encoder.stdin = None
                if encoder.wait() != 0:
                    raise RuntimeError(f"编码失败: {case.name}/{key}")
                renderer.close()
                probe = video_probe(partial_video, ffmpeg)
                if (int(probe["nb_read_frames"]) != frames
                        or probe.get("codec_name") != "h264"
                        or probe.get("pix_fmt") != "yuv420p"
                        or [probe["width"], probe["height"]] != [size, size]):
                    raise RuntimeError(f"视频规格不符: {partial_video}")
                # 新文件通过验收后才替换旧中断产物；失败时不覆盖上一份视频。
                partial_video.replace(video)
                video_probe_data = probe
                alpha_mean = float(np.mean(alpha_values))
                meta = renderer.meta
            if not smoke:
                Image.fromarray(first).save(case_out / f"{key}_poster.png")
            poster_frames[key] = first
            if "_nopose_recentered" in case.name:
                seconds, note = (None, "无对应耗时记录") if key == "competitor" else (
                    summary_time(case, key) if key in ("da3", "cameras", "sfm") else
                    sa_time(case) if key == "sa" else zip_time(case, zip_root))
            else:
                seconds, note = (summary_time(case, key) if key in ("da3", "cameras") else
                                 zip_time(case, zip_root) if key == "zip" else (None, "此案例无对应结果"))
            methods.append({"key": key, "label": label, "color": color, "available": True,
                            "ply": str(ply), "points": meta["points"],
                            "sh_degree": meta["sh_degree"], "video": str(video),
                            "video_sha256": sha256(video), "timing_seconds": seconds,
                            "timing_note": note, "mean_alpha": alpha_mean,
                            "camera_protocol": protocol, "video_probe": video_probe_data,
                            "render_complete": True})
            method = methods[-1]
            # 正式流式编码不再生成 PNG 临时目录；只保留视频、首帧和审计表。
            print(f"✅ {case.name}/{key}: {video}", flush=True)
            dump(case_out / f"{key}_summary.json", {
                "case": case.name, "key": key, "frames": frames, "fps": fps,
                "render_size": size, "smoke": smoke,
                "method": methods[-1]})
        if len(poster_frames) >= 3:
            ordered = [m for m in methods if m.get("available")]
            ordered.sort(key=lambda m: next(i for i, x in enumerate(METHODS) if x[0] == m["key"]))
            sheet = poster([poster_frames[m["key"]] for m in ordered],
                           [(m["key"], m["label"], next(x[2] for x in METHODS if x[0] == m["key"]))
                            for m in ordered])
            sheet.save(case_out / "poster.jpg", quality=95)
            assets = os.environ.get("DECK_ASSETS")
            if assets and not smoke:
                Path(assets).mkdir(parents=True, exist_ok=True)
                shutil.copyfile(case_out / "poster.jpg", Path(assets) / f"{case.name}.jpg")
        report = {"case": case.name, "camera_file": str(camera_file), "frames": frames, "fps": fps,
                  "render_size": size, "smoke": smoke, "seconds": time.time() - started,
                  "trajectory": "10机位闭环：旋转 SLERP、位置 Catmull-Rom、内参线性内插；无360外推",
                  "fairness": "各 PLY 原生完整高斯；各世界内等价轨迹，不宣称所有方法共用绝对物理相机或评分",
                  "methods": methods}
        assert [m["key"] for m in methods] == [key for key, _, _ in METHODS]
        assert len({m["key"] for m in methods}) == len(METHODS)
        dump(case_out / ("smoke_summary.json" if smoke else "summary.json"), report)
        print(f"🎉 {case.name}: {time.time() - started:.1f}s", flush=True)
    print("🎉 全部指定案例完成", flush=True)


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def summary_time(case, key):
    subdir = "depthanything3" if key == "da3" else f"depthanything3_{key}"
    path = case / subdir / "summary.json"
    if not path.is_file():
        return None, "未找到对应记录"
    data = json.loads(path.read_text(encoding="utf-8"))
    if "seconds" in data:
        return float(data["seconds"]), "前向与导出；不含共享模型加载"
    # 新8源入口摘要由原始前向覆盖安装信息，若原实验未记秒数则不虚构。
    return None, "该 PLY 对应记录未保存耗时"


if __name__ == "__main__":
    main()
