#!/usr/bin/env python3
"""海思输入相机 PLY：补绕世界 X 轴 180° 的显示坐标变换。

原 depthanything3_cameras.ply 保留不动；新文件为
CASE/depthanything3_cameras_upright.ply，配套相机在
depthanything3_cameras_upright/cameras.npz。两者同为 X' = diag(1,-1,-1) X，
所以渲染不变。此操作不改善模糊、位姿或多视图几何，也不声称两版已同一原点。

Env vars:
  ROOT, CASE_FILTER, FORCE
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
    result["original_world_rotation"] = ROTATE_X_180
    result["display_coordinate"] = np.asarray("原输入世界绕X轴180度；相机同步左乘同一世界旋转")
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


def main():
    root = Path(os.environ.get("ROOT", "/mnt/d/dataset/测试数据/海思端侧重建拍摄数据"))
    selected = os.environ.get("CASE_FILTER", "")
    force = os.environ.get("FORCE", "0") == "1"
    cases = sorted(root.glob("case*_nopose_recentered"),
                   key=lambda p: int(p.name.split("_")[0][4:]))
    if selected:
        cases = [p for p in cases if p.name == selected]
    if not root.is_dir() or not cases:
        sys.exit(f"❌ 没有匹配案例: ROOT={root}, CASE_FILTER={selected}")
    report = {"root": str(root), "source_ply": "depthanything3_cameras.ply",
              "output_ply": DISPLAY_PLY, "output_directory": DISPLAY_DIR,
              "world_rotation": ROTATE_X_180.tolist(),
              "scope": "仅位姿版显示坐标转换；相机同步，渲染不变；不改善画质或对齐",
              "started_at": datetime.now(timezone.utc).isoformat(), "cases": []}
    failed = False
    report_path = root / "depthanything3_cameras_upright_batch_summary.json"
    # 单案例补跑不能把其它15/16例从根目录总表抹掉。
    if selected and report_path.is_file():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        selected_names = {p.name for p in cases}
        report["cases"] = [r for r in previous.get("cases", [])
                           if r.get("case") not in selected_names]
    for case in cases:
        source = case / "depthanything3_cameras.ply"
        source_cameras = case / "depthanything3_cameras/cameras.npz"
        target = case / DISPLAY_PLY
        out = case / DISPLAY_DIR
        out.mkdir(parents=True, exist_ok=True)
        if target.exists() and (target.is_symlink() or not target.is_file()):
            failed = True
            print(f"❌ 拒绝处理非普通目标文件: {target}", flush=True)
            continue
        if target.is_file() and not force:
            info = _io.inspect_ply(target)
            if not (out / "cameras.npz").is_file() or not (out / "summary.json").is_file():
                raise RuntimeError(f"显示版缺少相机或记录，请用 FORCE=1 重建: {case.name}")
            info.update({"case": case.name, "status": "existing"})
        else:
            try:
                if not source.is_file() or not source_cameras.is_file():
                    raise RuntimeError(f"缺少已验收的位姿版PLY或相机: {source}={source.is_file()}, {source_cameras}={source_cameras.is_file()}")
                print(f"🔄 {case.name}: 原PLY + X轴180度显示变换", flush=True)
                info = transform_ply(source, target)
                info.update({"case": case.name, "status": "completed",
                             "source_sha256": _io.sha256(source)})
                cam = np.load(source_cameras, allow_pickle=False)
                display_cam = camera_to_display(cam)
                np.savez_compressed(out / "cameras.npz", **display_cam)
                # 用原始相机反演应回到输入外参，独立于变换路径检查。
                recovered = camera_to_display(display_cam)["extrinsics"]
                if not np.allclose(recovered, cam["extrinsics"], atol=2e-6, rtol=0):
                    raise RuntimeError("相机180度往返不一致")
                _io.dump_json(out / "summary.json", info)
                print(f"✅ {case.name}: {info['n_gaussians']:,} 高斯, "
                      f"{info['bytes'] / 1024**2:.1f} MiB", flush=True)
            except Exception as exc:
                failed = True
                info = {"case": case.name, "status": "failed", "error": str(exc)}
                _io.dump_json(out / "summary.json", info)
                print(f"❌ {case.name}: {exc}", flush=True)
        report["cases"].append(info)
        _io.dump_json(out / "batch_summary.json", report)
        _io.dump_json(root / "depthanything3_cameras_upright_batch_summary.json", report)
    report["cases"].sort(key=lambda r: int(r["case"].split("_")[0][4:]))
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    _io.dump_json(root / "depthanything3_cameras_upright_batch_summary.json", report)
    print(f"🎉 本轮{len(cases)}例已处理，总表{len(report['cases'])}例", flush=True)
    if failed or len(report["cases"]) < len(cases):
        sys.exit(1)


if __name__ == "__main__":
    main()
