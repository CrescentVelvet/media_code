"""海思 v2 全10图六路高斯 PLY 的输入相机闭环渲染。

复用51号的完整高斯、原生SH与轨迹插值实现；DA3新结果为采集世界直接对齐，
不再把原生预测相机重新拟合到采集相机。缺失结果保留空位，不补造视频。
"""
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

_MODULE_PATH = Path(__file__).resolve().with_name("51_render_hisilicon_comparison.py")
_spec = importlib.util.spec_from_file_location("hisilicon_comparison_v1", _MODULE_PATH)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
_original_setup_method = base.setup_method
_original_competitor_trajectory = base.competitor_trajectory
_original_zip_time = base.zip_time
ZIP_ROOT = None


def competitor_trajectory(capture, K, metadata, count, case):
    if case.name == "case14_nopose_recentered":
        # v2独立处理：不能让旧135°断言阻止新数据的相机诊断。
        bounds_path = case / f"{case.name.removesuffix('_recentered')}_stage1_bounds.json"
        bounds = json.loads(bounds_path.read_text(encoding="utf-8"))
        if not bounds.get("recentered") or bounds["minDistance"] <= 0:
            raise ValueError("v2 case14竞品缺少有效显示bounds")
        target = np.array([bounds["centerX"], bounds["centerY"], bounds["centerZ"]])
        centers = -(capture[:, :3, 3] + np.asarray(metadata["recenterOffset"])) - target
        radius = np.linalg.norm(centers, axis=1)
        if np.flatnonzero(radius > 3 * np.median(radius)).size:
            raise ValueError("v2 case14出现了未经本包抽检的离群机位")
        scales = np.clip(radius, bounds["minDistance"], bounds["maxDistance"]) / radius
        centers = centers * scales[:, None] + target
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
        roll[:3, :3] = base.Rotation.from_euler("z", 180., degrees=True).as_matrix()
        path, path_K = base.trajectory(gl @ base.FLIP_HOM @ roll, K, count)
        return path, path_K, {
            "protocol": "重居中竞品：v2源机位加recenterOffset并映射至主体正面侧，显示距离按bounds约束，全源机位闭环内插",
            "source_camera_transform": "C_display=-(C_source+recenterOffset)-center；look_at(bounds.center) @ flip_cv @ roll_z_180",
            "coordinate_units": "recentered PLY native units; not metric",
            "bounds_file": str(bounds_path),
            "source_radius_after_recenter": radius.tolist(),
            "distance_scale": scales.tolist(), "display_center": target.tolist(),
            "outlier_source_indices": [], "display_roll_degrees": 180.,
            "case14_outlier_rule": "v2无旧包view3离群机位，不使用initDistance；显示横滚180deg",
            "mapping_evidence": "Rendering-based display mapping; original exporter contract not verified",
            "visual_validation": "v2 case14四个轨迹时刻抽检及180/225deg显示横滚对照"}
    return _original_competitor_trajectory(capture, K, metadata, count, case)


base.competitor_trajectory = competitor_trajectory


def setup_method(case, method, capture, K, count, size, zip_root, pose_root):
    if method == "zip":
        # Zip与stage1均使用OpenCV c2w；capture[:3,3]是w2c平移而非中心。
        camera_file = base.zip_result_dir(case, ZIP_ROOT) / "predicted_cameras.npz"
        cam = base.npz(camera_file)
        model_c2w = cam["c2w"].astype(np.float64)
        if len(model_c2w) != 10:
            raise ValueError(f"Zip不是10个输入相机: {case.name}")
        expected_names = [f"view_{i:02d}_source_{int(v['sourceIndex']):03d}.png"
                          for i, v in enumerate(json.loads(base.camera_path(case).read_text(
                              encoding="utf-8"))["views"])]
        if list(map(str, cam["names"])) != expected_names:
            raise ValueError(f"Zip输入相机身份不匹配: {case.name}: {list(map(str, cam['names']))} != {expected_names}")
        model_K = np.zeros((len(model_c2w), 3, 3))
        for i, values in enumerate(cam["cameras"]):
            w, h, fx, fy, cx, cy = values
            model_K[i] = [[fx * size / w, 0., cx * size / w],
                          [0., fy * size / h, cy * size / h], [0., 0., 1.]]
        # 仅源相机参与显示变换；优先验证完整Sim(3)能否复现源相机朝向。
        R, t, scale, report = base.fit_similarity(capture[:, :3, 3], model_c2w[:, :3, 3])
        capture_path, capture_K = base.trajectory(capture, K, count)
        rot, rot_error = base.fit_source_rotation(capture, model_c2w)
        if rot_error <= .02:
            return base.transform_path(capture_path, R, t, scale), capture_K, {
                "protocol": "ZipSplat原生OpenCV c2w与stage1源相机中心拟合Sim(3)；源相机旋转一致；相机位置不取负",
                "coordinate_units": "ZipSplat arbitrary units; not metric",
                "display_rotation_max_error": rot_error, **report}
        # 个别样例的预测相机不满足完整Sim(3)：使用它自己的输入轨迹，不伪造共同世界。
        path, path_K = base.trajectory(model_c2w, model_K, count)
        return path, path_K, {
            "protocol": "ZipSplat原生OpenCV c2w输入轨迹；源相机不满足单一Sim(3)旋转映射，不强制配准",
            "coordinate_units": "ZipSplat arbitrary units; not metric",
            "display_rotation_max_error": rot_error, **report}
    if method == "competitor" and case.name == "case14_nopose_recentered":
        metadata = json.loads(base.camera_path(case).read_text(encoding="utf-8"))
        return base.competitor_trajectory(capture, K, metadata, count, case)
    if method not in ("da3", "cameras"):
        return _original_setup_method(case, method, capture, K, count, size, zip_root, pose_root)
    subdir = "depthanything3" if method == "da3" else "depthanything3_cameras"
    camera_file = case / subdir / "cameras.npz"
    if not camera_file.is_file():
        raise FileNotFoundError(f"已有 PLY 缺少配套对齐相机: {camera_file}")
    cam = base.npz(camera_file)
    if str(cam["camera_frame"]) != "stage1输入OpenCV世界；无显示旋转；PLY为各自原生结果事后对齐":
        raise ValueError(f"不是v2全10图对齐相机: {camera_file}")
    if not np.allclose(cam["world_rotation"], np.eye(3)):
        raise ValueError(f"v2展示不应再做显示转正: {case.name}/{method}")
    model_c2w = base.ext_to_c2w(cam)
    if len(model_c2w) != 10:
        raise ValueError(f"不是全10图: {case.name}/{method}")
    # 新全10图导出已在同一输入世界；数值门槛防止把旧版相机混入本PPT。
    error = float(np.abs(model_c2w - capture).max())
    if error > 1e-5:
        raise ValueError(f"DA3相机与采集世界不一致: {case.name}/{method}: {error}")
    path, path_K = base.trajectory(capture, K, count)
    return path, path_K, {
        "protocol": "v2全10图；PLY与相机已对齐至stage1输入OpenCV世界，直接内插采集轨迹",
        "n_source": 10, "holdout": False,
        "max_camera_difference": error,
        "camera_frame": str(cam["camera_frame"])}


def main():
    global ZIP_ROOT
    import shutil
    root = os.environ.get("ROOT", "")
    zip_root = Path(os.environ.get("ZIP_RESULTS_DIR", ""))
    ZIP_ROOT = zip_root
    if not root.rstrip("/").endswith("海思端侧重建拍摄数据v2"):
        raise ValueError("54号仅用于海思v2，不覆盖旧19例")
    if not str(zip_root).rstrip("/").endswith("hisilicon_v2_10v"):
        raise ValueError("v2 Zip相机必须来自 hisilicon_v2_10v")
    # 旧主入口仅校验目录名；真实v2路径由ZIP_ROOT贯穿相机、PLY来源与耗时读取。
    os.environ["ZIP_RESULTS_DIR"] = str(zip_root.parent / "hisilicon_10v")
    if os.environ.get("PREFLIGHT_ONLY", "0") == "1":
        root_path = Path(root)
        case_list = sorted(root_path.glob("case*_nopose_recentered"),
                           key=lambda p: int(p.name.split("_")[0].removeprefix("case")))
        records = []
        for case in case_list:
            capture, K, metadata, camera_file = base.load_capture(case, 704)
            input_paths = sorted((case / "images").glob("*.png"))
            if len(input_paths) != 10:
                raise ValueError(f"输入不是10张: {case.name}")
            expected_inputs = [f"view_{i:02d}_source_{int(v['sourceIndex']):03d}.png"
                               for i, v in enumerate(metadata["views"])]
            if [p.name for p in input_paths] != expected_inputs:
                raise ValueError(f"输入与相机身份不一致: {case.name}")
            for image_path in input_paths:
                with base.Image.open(image_path) as image:
                    if image.size != (704, 704):
                        raise ValueError(f"输入尺寸错误: {image_path}")
                    image.verify()
            for key, _, _ in base.METHODS:
                ply = base.locate(case, key, zip_root, "")
                if ply is None or not ply.is_file():
                    continue
                path, path_K, protocol = setup_method(
                    case, key, capture, K, 120, 704, zip_root, "")
                if not (np.isfinite(path).all() and np.isfinite(path_K).all()):
                    raise ValueError(f"轨迹非有限值: {case.name}/{key}")
                records.append({"case": case.name, "method": key, "camera_file": str(camera_file),
                                "camera_protocol": protocol})
        base.dump(Path(os.environ["PREFLIGHT_OUTPUT"]), {
            "cases": len(case_list), "available_methods": len(records),
            "camera_and_trajectory_finite": True, "records": records})
        print(f"PREFLIGHT_OK cases=34 methods={len(records)}", flush=True)
        return
    protocol_path = Path(os.environ["OUT_DIR"]) / "protocol_version.json"
    protocol_backup = protocol_path.with_name("protocol_version.renderer.json")
    protocol_data = {
        "version": 3, "dataset": root, "scope": "v2 / 34 stage1 cases / 200 available method videos",
        "input_images": "34 cases x 10 original images; 704x704; sourceIndex identities verified",
        "video": "704x704, 120 frames, 15 fps, 8 seconds, H.264/yuv420p",
        "trajectory": "对应方法输入相机闭环：旋转SLERP、位置Catmull-Rom、内参线性插值",
        "da3": "全10图结果与相机已在stage1 OpenCV世界；不再套旧版预测相机配准",
        "sfm": "保留实际SfM输入相机和内参；不强制与stage1构成单一刚体",
        "sa": "源相机身份和源相机Sim(3)严格核验",
        "zip": "26例通过源相机中心及旋转一致性的Sim(3)映射；8例不满足单一源旋转映射，使用Zip原生输入相机与内参闭环",
        "competitor": "随包recenterOffset/bounds的展示映射；case7水平方位中位半径环绕；v2 case14无离群机位、显示横滚180deg",
        "mapping_evidence": "展示抽检不是原始导出器契约、几何精度、全轨迹或算法质量认证",
        "gaussian_pruning": False, "native_sh": True, "sha256_required": True,
        "timing": "只登记真实算法摘要/metrics；不估算，不用展示渲染耗时替代推理耗时"}
    try:
        return base.main()
    finally:
        # 通用主入口写旧协议说明；保留原文并把本次真实适配写入独立记录。
        if protocol_path.is_file():
            shutil.copyfile(protocol_path, protocol_backup)
        base.dump(protocol_path, protocol_data)


base.setup_method = setup_method
base.sa_time = lambda case: (None, "v2批推理日志未保存单例耗时")
base.zip_time = lambda case, unused_root: _original_zip_time(case, ZIP_ROOT)


if __name__ == "__main__":
    main()
