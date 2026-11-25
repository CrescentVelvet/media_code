#!/usr/bin/env python3
"""海思相机版：把已完成的180°转正结果收敛到原文件名，不再生成第三套PLY。

只搬迁现有文件、更新配套相机和记录，不重新推理。原始位姿版先整体备份到
BACKUP_ROOT，再用已验收的显示版替换 depthanything3_cameras.ply。

Env vars:
  ROOT, BACKUP_ROOT, CASE_FILTER
"""
import os
import sys
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from importlib.util import module_from_spec, spec_from_file_location  # noqa: E402

_spec = spec_from_file_location("hisilicon_rgb", SCRIPT_DIR / "34_run_hisilicon.py")
_io = module_from_spec(_spec)
_spec.loader.exec_module(_io)

ROTATE_X_180 = np.diag([1.0, -1.0, -1.0])
DISPLAY_PLY = "depthanything3_cameras_upright.ply"
DISPLAY_DIR = "depthanything3_cameras_upright"


def camera_to_display(cam):
    """旧 OpenCV w2c 乘左侧世界旋转；原始输入相机另存原值供回溯。"""
    result = {key: cam[key].copy() for key in (cam.files if hasattr(cam, "files") else cam)}
    ext = np.asarray(cam["extrinsics"], dtype=np.float64)
    if ext.shape[1:] == (3, 4):
        ext44 = np.repeat(np.eye(4, dtype=np.float64)[None], len(ext), axis=0)
        ext44[:, :3] = ext
    elif ext.shape[1:] == (4, 4):
        ext44 = ext.copy()
    else:
        raise RuntimeError(f"相机外参形状异常: {ext.shape}")
    c2w = np.linalg.inv(ext44)
    display_c2w = c2w.copy()
    display_c2w[:, :3, :3] = np.einsum("ij,vjk->vik", ROTATE_X_180, c2w[:, :3, :3])
    display_c2w[:, :3, 3] = c2w[:, :3, 3] @ ROTATE_X_180.T
    result["extrinsics"] = np.linalg.inv(display_c2w)[:, :3]
    result["world_rotation"] = ROTATE_X_180
    result["coordinate"] = np.asarray("输入世界绕X轴180度；高斯与相机同步旋转")
    return result


def transform_ply(source, target):
    """处理完整记录数组，并用官方算子抽查四元数及SH旋转。"""
    import torch
    from plyfile import PlyData, PlyElement

    ply = PlyData.read(str(source), mmap="r")
    vertex = ply["vertex"].data
    names = vertex.dtype.names
    expected = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    expected += [f"f_rest_{i}" for i in range(24)]
    expected += ["opacity", "scale_0", "scale_1", "scale_2",
                 "rot_0", "rot_1", "rot_2", "rot_3"]
    if names != tuple(expected) or len(ply.elements) != 1:
        raise RuntimeError(f"只支持原生41属性单顶点PLY: {source}, {names}")
    arr = np.array(vertex, copy=True)
    for field in ("y", "z"):
        arr[field] *= -1.0
    # 官方输入为(N,3,9)、rotations为(N,3,3,3)，f_rest按(N,3,8)展平。
    from depth_anything_3.utils.sh_helpers import rotate_sh
    sh = np.stack([vertex[f"f_dc_{i}"] for i in range(3)], axis=-1)[:, :, None]
    # 官方/PLY 的 f_rest 展平顺序是(N,3,8)，这里不可改成channel-major。
    sh = np.concatenate([
        sh,
        np.stack([vertex[f"f_rest_{i}"] for i in range(24)], axis=-1).reshape(len(arr), 3, 8),
    ], axis=2)
    R_t = torch.from_numpy(ROTATE_X_180).float().cuda()
    with torch.no_grad():
        rotated_sh = rotate_sh(
            torch.from_numpy(sh).float().cuda(),
            R_t.reshape(1, 1, 3, 3).expand(len(arr), 3, 3, 3),
        ).cpu().numpy()
    for i in range(3):
        arr[f"f_dc_{i}"] = rotated_sh[:, i, 0]
    for channel in range(3):
        for coeff in range(8):
            arr[f"f_rest_{channel * 8 + coeff}"] = rotated_sh[:, channel, coeff + 1]
    del sh, rotated_sh
    from depth_anything_3.model.utils.transform import mat_to_quat, quat_to_mat
    q = np.stack([arr[f"rot_{i}"] for i in range(4)], axis=1)
    with torch.no_grad():
        original_matrices = quat_to_mat(torch.from_numpy(q).float().cuda())
        transformed = mat_to_quat(
            torch.from_numpy(ROTATE_X_180).float().cuda() @ original_matrices,
        ).cpu().numpy().astype(np.float64)
    q_norm = np.linalg.norm(transformed, axis=1)
    if np.min(q_norm) < 1e-8 or not np.isfinite(transformed).all():
        raise RuntimeError(f"非法高斯四元数: {source}")
    # 矩阵转四元数只需float32；避免在(N,4,4)大临时数组上重复float64归一化。
    transformed = transformed.astype(np.float32)
    for i in range(4):
        arr[f"rot_{i}"] = transformed[:, i].astype(np.float32)
    output = PlyData([PlyElement.describe(arr, "vertex")], text=False, byte_order="<")
    output.write(str(target))
    for offset in (0, len(arr) // 2, len(arr) - 1):
        actual_rot = quat_to_mat(torch.from_numpy(transformed[offset]).float()[None])[0].numpy()
        original_rot = quat_to_mat(torch.from_numpy(q[offset]).float()[None])[0].numpy()
        if not np.allclose(actual_rot, ROTATE_X_180 @ original_rot, atol=2e-5, rtol=2e-5):
            raise RuntimeError(f"四元数旋转交叉核验失败: {source}, offset={offset}")
    del ply, vertex, arr
    return _io.inspect_ply(target)


def migrate_existing(root, backup_root):
    """先备份，再原子改名替换；每例前后均核验已验收的SHA-256。"""
    selected = ""
    if os.environ.get("CASE_FILTER"):
        sys.exit("❌ 文件收敛必须整批执行，不能用CASE_FILTER缩减总表")
    cases = sorted(root.glob("case*_nopose_recentered"),
                   key=lambda p: int(p.name.split("_")[0][4:]))
    if backup_root.exists():
        sys.exit(f"❌ 备份目录已存在，拒绝覆盖: {backup_root}")
    backup_root.mkdir(parents=True)
    report_path = root / "depthanything3_cameras_batch_summary.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    previous = {r["case"]: r for r in report["cases"]}
    records = []
    pending = []

    def move_to_backup(source, relative):
        if source.is_symlink() or not source.is_file():
            raise RuntimeError(f"只迁移普通非链接文件: {source}")
        target = backup_root / relative
        if target.exists():
            raise RuntimeError(f"备份目标已存在: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = _io.sha256(source)
        source.replace(target)
        if _io.sha256(target) != expected:
            raise RuntimeError(f"备份哈希不一致: {target}")
        return expected

    try:
        for case in cases:
            original = case / "depthanything3_cameras.ply"
            original_cameras = case / "depthanything3_cameras/cameras.npz"
            transformed = case / DISPLAY_PLY
            transformed_cameras = case / DISPLAY_DIR / "cameras.npz"
            if any(p.is_symlink() or not p.is_file() for p in
                   (original, original_cameras, transformed, transformed_cameras)):
                raise RuntimeError(f"原文件或已验收转正文件缺失/为链接: {case.name}")
            old_info = _io.inspect_ply(original)
            new_info = _io.inspect_ply(transformed)
            with np.load(original_cameras, allow_pickle=False) as cam:
                original_ext = cam["extrinsics"].copy()
            with np.load(transformed_cameras, allow_pickle=False) as cam:
                transformed_ext = cam["extrinsics"].copy()
                if not np.allclose(camera_to_display(cam)["extrinsics"], original_ext,
                                   atol=2e-6, rtol=0):
                    raise RuntimeError(f"配套相机180度往返不一致: {case.name}")
            pending.append((case, old_info, new_info, original_ext, transformed_ext))
        print(f"📦 预备份{len(pending)}例原PLY和相机到: {backup_root}", flush=True)
        for case, old_info, new_info, _, _ in pending:
            move_to_backup(case / "depthanything3_cameras.ply",
                           Path(case.name) / "depthanything3_cameras.ply")
            move_to_backup(case / "depthanything3_cameras/cameras.npz",
                           Path(case.name) / "depthanything3_cameras/cameras.npz")
            move_to_backup(case / "depthanything3_cameras/summary.json",
                           Path(case.name) / "depthanything3_cameras/summary.json")
            # 源与目标均在同一文件系统，replace不复制400MiB，也不会留下半写文件。
            (case / DISPLAY_PLY).replace(case / "depthanything3_cameras.ply")
            (case / DISPLAY_DIR / "cameras.npz").replace(
                case / "depthanything3_cameras/cameras.npz")
            if _io.sha256(case / "depthanything3_cameras.ply") != new_info["sha256"]:
                raise RuntimeError(f"替换后PLY哈希不一致: {case.name}")
            out = case / "depthanything3_cameras"
            info = dict(previous[case.name])
            new_info["path"] = str(case / "depthanything3_cameras.ply")
            info.update(new_info)
            info.update({"case": case.name, "status": "completed",
                         "world_rotation": ROTATE_X_180.tolist(),
                         "gaussian_rotation": (ROTATE_X_180 @ np.asarray(
                             previous[case.name]["umeyama_rotation"])).tolist(),
                         "gaussian_translation": (ROTATE_X_180 @ np.asarray(
                             previous[case.name]["umeyama_translation"])).tolist(),
                         "pre_rotation_sha256": old_info["sha256"],
                         "output_extrinsics": transformed_ext.tolist(),
                         "camera_coordinate": "OpenCV w2c；相机与高斯已同步绕输入世界X轴180度",
                         "scope": "高斯和相机已同步绕X轴180度；不裁点、不美化，不改善几何",
                         "input_camera_centers": previous[case.name]["input_camera_centers"],
                         "output_camera_centers": (np.asarray(
                             previous[case.name]["input_camera_centers"]) @ ROTATE_X_180.T).tolist(),
                         "pre_rotation_backup": str(backup_root / case.name)})
            _io.dump_json(out / "summary.json", info)
            _io.dump_json(out / "batch_summary.json", info)
            records.append(info)
            print(f"✅ {case.name}: depthanything3_cameras.ply 已替换为转正版", flush=True)
    except Exception:
        # 备份始终保留；错误留给使用者处理，不自动跨文件回滚覆盖已验收结果。
        _io.dump_json(backup_root / "interrupted.json", {
            "completed": [r["case"] for r in records],
            "pending": [p[0].name for p in pending if p[0].name not in {r["case"] for r in records}],
        })
        raise

    report.update({"cases": records,
                   "gaussian_frame": "官方Sim(3)对齐后，高斯与相机同步绕X轴180度；相机版原地转正",
                   "migration": "复用已验收的180度结果替换原相机版；原文件在BACKUP_ROOT，无upright副本",
                   "backup_root": str(backup_root),
                   "world_rotation": ROTATE_X_180.tolist(),
                   "migrated_at": datetime.now(timezone.utc).isoformat()})
    if report.get("export") is not None:
        report["export"]["world_rotation"] = ROTATE_X_180.tolist()
    _io.dump_json(report_path, report)
    for case in cases:
        for path in case.glob("*upright*"):
            if path.is_symlink():
                raise RuntimeError(f"拒绝处理链接: {path}")
            path.replace(backup_root / path.relative_to(root))
    for path in root.glob("*upright*"):
        if path.is_symlink():
            raise RuntimeError(f"拒绝处理链接: {path}")
        path.replace(backup_root / path.relative_to(root))
    _io.dump_json(backup_root / "migration_summary.json", report)
    print(f"🎉 {len(records)}例已原地转正；数据目录不再保留upright命名；原文件备份完成", flush=True)


def main():
    migrate_existing(
        Path(os.environ.get("ROOT", "/mnt/d/dataset/测试数据/海思端侧重建拍摄数据")),
        Path(os.environ.get("BACKUP_ROOT", "/mnt/d/output/_backup/da3_cameras_pre_rotation_20261126")),
    )


if __name__ == "__main__":
    main()
