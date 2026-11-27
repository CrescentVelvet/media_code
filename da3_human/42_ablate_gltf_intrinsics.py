#!/usr/bin/env python3
"""海思 glTF 相机输入消融：不覆盖正式 PLY，逐支保存高斯与原始预测相机。

Env vars:
  ROOT, CASE_FILTER, DA3_MODEL_DIR, OUT_SUBDIR
  INTRINSICS_MODE 顶层 top_level_K 或动画 camera_path_yfov
  USE_RAY_POSE 0/1，默认 0
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from importlib.util import module_from_spec, spec_from_file_location  # noqa: E402

_spec = spec_from_file_location("camera", SCRIPT_DIR / "36_run_hisilicon_cameras.py")
_cam = module_from_spec(_spec)
_spec.loader.exec_module(_cam)


def bounds(means):
    flat = means.detach().reshape(-1, 3)
    return [flat.amin(dim=0).tolist(), flat.amax(dim=0).tolist()]


def main():
    root = Path(os.environ.get("ROOT", "/mnt/d/dataset/测试数据/海思端侧重建拍摄数据"))
    case = root / os.environ.get("CASE_FILTER", "model_man")
    out = case / os.environ.get("OUT_SUBDIR", "depthanything3_camera_audit/intrinsics")
    out.mkdir(parents=True, exist_ok=True)
    mode = os.environ.get("INTRINSICS_MODE", "camera_path_yfov")
    use_ray_pose = os.environ.get("USE_RAY_POSE", "0") == "1"
    files, ext, ixt, _, _ = _cam.load_gltf_inputs(case)
    if mode in ("camera_path_yfov", "camera_path_yfov_raw_c2w"):
        import json
        data = json.loads((case / f"{case.name}_camera.json").read_text(encoding="utf-8"))
        path = data["extrinsics"]["camera_path"]
        yfov = np.asarray(path["yfov"]).reshape(-1)
        if len(yfov) != len(files) or not np.isfinite(yfov).all() or (yfov <= 0).any():
            raise RuntimeError("动画 yfov 与图像数量不匹配或值非法")
        size = data["intrinsics"]["reference_image_size"]
        f = size["height"] / (2 * np.tan(yfov / 2))
        ixt[:] = 0
        ixt[:, 0, 0] = f
        ixt[:, 1, 1] = f * size["width"] / size["height"]
        ixt[:, 0, 2] = size["width"] / 2
        ixt[:, 1, 2] = size["height"] / 2
        ixt[:, 2, 2] = 1
    elif mode not in ("top_level_K", "camera_path_yfov_raw_c2w"):
        raise ValueError(f"未知 INTRINSICS_MODE: {mode}")
    if mode == "camera_path_yfov_raw_c2w":
        # glTF 视角方向与 OpenCV 图像不是同一约定，不能用既有 PLY 的显示旋转倒推。
        c2w = np.asarray(data["extrinsics"]["camera_path"]["c2w_keyframes"], dtype=np.float64)
        ext = np.linalg.inv(c2w)

    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.utils import gsply_helpers as gh
    gh.export_ply = _cam._io.fast_export_ply
    model = DepthAnything3.from_pretrained(
        os.environ.get("DA3_MODEL_DIR", "/mnt/d/wheel/da3_human_ms")).to("cuda").eval()
    imgs_cpu, ext_cpu, ixt_cpu = model._preprocess_inputs(
        [str(p) for p in files], extrinsics=ext, intrinsics=ixt,
        process_res=504, process_res_method="upper_bound_resize")
    imgs, ext_t, ixt_t = model._prepare_model_inputs(imgs_cpu, ext_cpu, ixt_cpu)
    norm = model._normalize_extrinsics(ext_t.clone())
    raw = model._run_model_forward(imgs, norm, ixt_t, [], True, use_ray_pose, "saddle_balanced")
    pred = model._convert_to_prediction(raw)
    raw_extrinsics = pred.extrinsics.copy()
    raw_intrinsics = pred.intrinsics.copy()
    sf = float(pred.scale_factor)
    # 后处理去除 adapter 的 gt_extrinsics 尺度，回到未作输入相机局部缩放的同一世界。
    from depth_anything_3.utils.pose_align import batch_align_poses_umeyama
    _, _, adapter_scale = batch_align_poses_umeyama(norm.detach(), raw.extrinsics.detach())
    adapter_scale = float(adapter_scale.clamp(1 / 3, 3)[0].cpu())
    pred.gaussians.means = pred.gaussians.means / adapter_scale
    pred.gaussians.scales = pred.gaussians.scales / adapter_scale
    base = out / ("ray_pose" if use_ray_pose else "camera_dec")
    base.mkdir(exist_ok=True)
    np.savez_compressed(base / "cameras.npz", extrinsics=raw_extrinsics,
                        intrinsics=raw_intrinsics, input_extrinsics=ext,
                        input_intrinsics=ixt, depth=pred.depth,
                        image_names=np.asarray([p.name for p in files]),
                        scale_factor=np.float32(sf))
    gh.save_gaussian_ply(
        gaussians=pred.gaussians, save_path=str(base / "gaussians.ply"),
        ctx_depth=torch.from_numpy(pred.depth).unsqueeze(-1).to(pred.gaussians.means),
        shift_and_scale=False, save_sh_dc_only=False, gs_views_interval=1,
        inv_opacity=True, prune_by_depth_percent=1.0, prune_border_gs=False,
        match_3dgs_mcmc_dev=False)
    summary = {"intrinsics_mode": mode, "use_ray_pose": use_ray_pose,
               "input_focal_at_704": ixt[:, 0, 0].tolist(),
               "processed_focal": ixt_cpu[:, 0, 0].tolist(),
               "pred_focal": raw_intrinsics[:, 0, 0].tolist(),
               "scale_factor": sf, "adapter_scale": adapter_scale,
               "depth_quantiles": np.quantile(pred.depth, [.1, .5, .9]).tolist(),
               "mean_bounds": bounds(pred.gaussians.means)}
    _cam._io.dump_json(base / "summary.json", summary)
    print("SUMMARY", summary, flush=True)


if __name__ == "__main__":
    main()
