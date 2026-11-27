#!/usr/bin/env python3
"""同一次 DA3 前向的原生、输入相机反投影和完整 Sim(3) 三路对照。

直接分支只在 adapter 内换用归一化输入相机，原始输入世界平移/尺度在 adapter
外统一恢复；不混用 Nested sf。只写诊断目录，不覆盖正式 PLY。

Env vars: ROOT, CASE_FILTER, DA3_MODEL_DIR, OUT_SUBDIR, PROCESS_RES, SEED
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


def main():
    root = Path(os.environ.get("ROOT", "/mnt/d/dataset/测试数据/海思端侧重建拍摄数据"))
    case = root / os.environ.get("CASE_FILTER", "model_man")
    out = case / os.environ.get("OUT_SUBDIR", "depthanything3_camera_audit/direct_700")
    if out.is_symlink() or ".." in out.relative_to(case).parts:
        sys.exit("❌ 诊断输出不得为符号链接或越出案例目录")
    out.mkdir(parents=True, exist_ok=True)
    files, ext, ixt, camera_path, _ = _cam.load_gltf_inputs(case)
    process_res = int(os.environ.get("PROCESS_RES", "704"))
    seed = int(os.environ.get("SEED", "42"))
    torch.manual_seed(seed)
    np.random.seed(seed)

    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.utils import gsply_helpers as gh
    from depth_anything_3.utils.pose_align import align_poses_umeyama
    from depth_anything_3.specs import Gaussians
    gh.export_ply = _cam._io.fast_export_ply
    model = DepthAnything3.from_pretrained(
        os.environ.get("DA3_MODEL_DIR", "/mnt/d/wheel/da3_human_ms")).to("cuda").eval()
    imgs_cpu, ext_cpu, ixt_cpu = model._preprocess_inputs(
        [str(p) for p in files], extrinsics=ext, intrinsics=ixt,
        process_res=process_res, process_res_method="upper_bound_resize")
    imgs, ext_t, ixt_t = model._prepare_model_inputs(imgs_cpu, ext_cpu, ixt_cpu)
    ext_norm = model._normalize_extrinsics(ext_t.clone())
    input_c2w = np.linalg.inv(ext)
    first_center = input_c2w[0, :3, 3].copy()
    center_dist = torch.median(torch.linalg.inv(ext_norm)[..., :3, 3].norm(dim=-1)).item()
    adapter = model.model.da3.gs_adapter
    original = adapter.forward
    holder = {}

    def capture(*args, **kwargs):
        for key in ("extrinsics", "intrinsics", "depths", "opacities", "raw_gaussians"):
            holder[key] = kwargs[key].detach().clone()
        holder["image_shape"] = kwargs["image_shape"]
        holder["official"] = original(*args, **kwargs)
        direct_kwargs = dict(kwargs)
        direct_kwargs["extrinsics"] = ext_norm
        direct_kwargs["intrinsics"] = ixt_t
        direct_kwargs["gt_extrinsics"] = None
        holder["direct"] = original(*args, **direct_kwargs)
        return holder["official"]

    try:
        adapter.forward = capture
        raw = model._run_model_forward(imgs, ext_norm, ixt_t, [], True, False, "saddle_balanced")
    finally:
        adapter.forward = original
    pred = model._convert_to_prediction(raw)
    raw_extrinsics = pred.extrinsics.copy()
    scale_factor = float(pred.scale_factor)
    pred_extrinsics = holder["extrinsics"][0].float().cpu().numpy()
    R, t, s = align_poses_umeyama(ext, pred_extrinsics, ransac=False, random_state=42)
    pred_centers = np.linalg.inv(pred_extrinsics)[:, :3, 3]
    input_c2w = np.linalg.inv(ext)
    aligned_centers = s * (pred_centers @ R.T) + t
    center_error = np.linalg.norm(aligned_centers - input_c2w[:, :3, 3], axis=-1)
    report = {"case": case.name, "process_res": process_res, "seed": seed,
              "camera_json": camera_path.name, "scale_factor_nested": scale_factor,
              "normalization_center_dist": center_dist,
              "direct_transform": "X_input = center_dist * X_direct + R_ext0 @ C0; R_direct=R_ext0.T @ R_input",
              "sim3": {"scale": float(s), "rotation": R.tolist(), "translation": t.tolist(),
                       "center_max_error": float(center_error.max()),
                       "center_mean_error": float(center_error.mean())},
              "branches": {}}

    def clone(g):
        return Gaussians(**{name: getattr(g, name).clone()
                           for name in ("means", "scales", "rotations", "harmonics", "opacities")})

    def save(name, g, cameras, intrinsics, note):
        path = out / f"{name}.ply"
        gh.save_gaussian_ply(
            gaussians=g, save_path=str(path),
            ctx_depth=torch.from_numpy(pred.depth).unsqueeze(-1).to(g.means),
            shift_and_scale=False, save_sh_dc_only=True, gs_views_interval=1,
            inv_opacity=True, prune_by_depth_percent=1.0,
            prune_border_gs=False, match_3dgs_mcmc_dev=False)
        np.savez_compressed(out / f"{name}.npz", extrinsics=cameras[:, :3], intrinsics=intrinsics,
                            depth=pred.depth, image_names=np.asarray([p.name for p in files]))
        info = _cam._io.inspect_ply(path) if False else {
            "path": path.name, "n_gaussians": int(g.means.shape[1]),
            "coordinate_note": note}
        report["branches"][name] = info
        print(name, info, flush=True)
        del g
        torch.cuda.empty_cache()

    save("official_native", clone(holder["official"]), pred_extrinsics,
         holder["intrinsics"][0].float().cpu().numpy(),
         "adapter 原生高斯 + 生成时预测相机；不施加 Nested sf")
    direct = clone(holder["direct"])
    direct.means = direct.means * center_dist + torch.as_tensor(
        first_center, dtype=direct.means.dtype, device=direct.means.device)
    direct.scales = direct.scales * center_dist
    save("direct_input", direct, ext, ixt_cpu.numpy(),
         "同前向深度/高斯属性，adapter 中直接使用归一化输入相机，恢复原始平移/尺度")
    aligned = clone(holder["official"])
    from depth_anything_3.model.utils.transform import quat_to_mat, mat_to_quat
    Rt = torch.as_tensor(R, dtype=torch.float32, device=aligned.means.device)
    tt = torch.as_tensor(t, dtype=torch.float32, device=aligned.means.device)
    aligned.means = aligned.means @ Rt.T * s + tt
    aligned.scales = aligned.scales * s
    aligned.rotations = torch.nn.functional.normalize(
        mat_to_quat(Rt @ quat_to_mat(aligned.rotations)), dim=-1)
    save("official_sim3", aligned, ext, ixt_cpu.numpy(),
         "adapter 高斯直接经 pred -> 原始输入世界单一 Sim(3)，无 Nested sf/a 补偿")
    report["quality_status"] = "三路均需源视角回放，文件生成不等于质量通过"
    _cam._io.dump_json(out / "summary.json", report)
    print(f"✅ {out}", flush=True)


if __name__ == "__main__":
    main()
