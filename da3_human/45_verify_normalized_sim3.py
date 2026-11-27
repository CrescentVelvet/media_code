#!/usr/bin/env python3
"""在已保存的同次前向上验证 adapter 高斯到原始 glTF 世界的变换。

不重新推理，不乘 Nested sf，也不重复施加 adapter 尺度；用已捕获的高斯生成相机。
输入世界先经官方 `_normalize_extrinsics` 等价变换，再回到原始 glTF 坐标。
"""
import json
import os
import time
from pathlib import Path
import numpy as np
from plyfile import PlyData
import torch

SCRIPT = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(SCRIPT))
from importlib.util import spec_from_file_location, module_from_spec
s = spec_from_file_location("cam", SCRIPT / "36_run_hisilicon_cameras.py")
cam = module_from_spec(s)
s.loader.exec_module(cam)


def main():
    case = Path(os.environ["ROOT"]) / os.environ.get("CASE_FILTER", "model_man")
    source = case / "depthanything3_camera_audit/native_camera_yfov"
    out = case / "depthanything3_camera_audit/normalized_sim3"
    if (out / "aligned.ply").is_symlink():
        raise RuntimeError("拒绝覆盖符号链接诊断文件")
    out.mkdir(parents=True, exist_ok=True)
    incomplete = out / "aligned.ply"
    if incomplete.is_file():
        # 保留此前同名文件，完整 PLY 仍须通过导出后的逐点回读。
        backup = out / f"aligned.previous_{time.strftime('%Y%m%d_%H%M%S')}.ply"
        if backup.exists():
            raise RuntimeError(f"旧诊断文件备份已存在，请先人工核对: {backup}")
        incomplete.rename(backup)
    files, ext, ixt, _, _ = cam.load_gltf_inputs(case)
    z = np.load(source / "pose_gs_cameras.npz")
    pred = z["extrinsics"]
    if pred.ndim == 4 and pred.shape[0] == 1:
        pred = pred[0]
    pred_c2w = np.linalg.inv(pred)
    center_dist = np.median(np.linalg.norm(pred_c2w[:, :3, 3], axis=-1))
    input_c2w = np.linalg.inv(ext)
    c0 = input_c2w[0, :3, 3].copy()
    input_norm_c2w = input_c2w.copy()
    input_norm_c2w[:, :3, 3] = (input_norm_c2w[:, :3, 3] - c0) / center_dist
    ext_norm = np.linalg.inv(input_norm_c2w)
    from depth_anything_3.utils.pose_align import align_poses_umeyama
    # adapter 高斯仍在预测相机朝向的世界，只被施加了深度/中心尺度 a。
    # 直接求 pred -> 原始输入世界，不能混入 Nested sf 或再次除以 a。
    R, t, s = align_poses_umeyama(ext, pred, ransac=False, random_state=42)
    pred_center_after = s * (pred_c2w[:, :3, 3] @ R.T) + t
    position_error = np.linalg.norm(pred_center_after - input_c2w[:, :3, 3], axis=-1)
    print("normalization", center_dist,
          "gaussian_to_input_scale", s, "R", R, "t", t,
          "position_error", position_error, flush=True)
    v = PlyData.read(str(source / "pose_native.ply"))["vertex"].data
    xyz = torch.from_numpy(np.stack([v[k] for k in ("x", "y", "z")], 1)).cuda().float()
    scales = torch.from_numpy(np.exp(np.stack([v[k] for k in ("scale_0", "scale_1", "scale_2")], 1))).cuda().float()
    q = torch.from_numpy(np.stack([v[k] for k in ("rot_0", "rot_1", "rot_2", "rot_3")], 1)).cuda().float()
    dc = torch.from_numpy(np.stack([v[k] for k in ("f_dc_0", "f_dc_1", "f_dc_2")], 1)).cuda().float()
    g = type("G", (), {})()
    g.means = xyz[None]
    g.scales = scales[None]
    g.rotations = q[None]
    # PLY 的 DC 是 (N,3)，官方 Gaussians.harmonics 才是 (N,3,K)。
    # 必须补回 K 维，否则 (N,1,3)[...,0] 只剩一个颜色通道。
    g.harmonics = dc[None, :, :, None]
    g.opacities = torch.sigmoid(torch.from_numpy(np.asarray(v["opacity"]))).cuda().float()[None]
    # 直接变换捕获的 adapter 高斯；Nested sf 只缩放输出深度/相机，不缩放这里的高斯。
    from depth_anything_3.model.utils.transform import quat_to_mat, mat_to_quat
    Rt = torch.from_numpy(R).cuda().float()
    tt = torch.from_numpy(t).cuda().float()
    g.means = (g.means @ Rt.T) * s + tt
    g.scales = g.scales * s
    g.rotations = torch.nn.functional.normalize(
        mat_to_quat(Rt @ quat_to_mat(g.rotations)), dim=-1)
    if g.rotations.shape[-1] != 4:
        raise RuntimeError(f"四元数维度异常: {g.rotations.shape}")
    depth = np.load(source / "pose_native_cameras.npz")["depth"]
    gh = __import__("depth_anything_3.utils.gsply_helpers", fromlist=["save_gaussian_ply"])
    gh.export_ply = cam._io.fast_export_ply
    gh.save_gaussian_ply(gaussians=g, save_path=str(out / "aligned.ply"),
        ctx_depth=torch.from_numpy(depth).unsqueeze(-1).cuda(),
        shift_and_scale=False, save_sh_dc_only=True, gs_views_interval=1,
        inv_opacity=True, prune_by_depth_percent=1.0, prune_border_gs=False,
        match_3dgs_mcmc_dev=False)
    np.savez_compressed(out / "cameras.npz", extrinsics=ext[:, :3], intrinsics=ixt,
                        depth=depth, image_names=np.asarray([p.name for p in files]))
    info = {"pred_camera": "captured at gs_adapter, before Nested",
            "input_normalization_center_dist": float(center_dist),
            "input_first_camera_center": c0.tolist(),
            "scale_gaussian_to_original_input": float(s),
            "rotation": R.tolist(), "translation": t.tolist(),
            "aligned_camera_max_position_error": float(position_error.max()),
            "aligned_camera_mean_position_error": float(position_error.mean()),
            "formula": "align_poses_umeyama(original_input, pred) returns pred -> original_input; apply directly to adapter gaussians; no Nested sf or extra adapter compensation",
            "quality_status": "待源视角回放；有限值和相机残差不等于画质通过"}
    cam._io.dump_json(out / "summary.json", info)
    print(info, flush=True)


if __name__ == "__main__":
    main()
