#!/usr/bin/env python3
"""海思新 glTF 输入相机：同次前向分支诊断，不改正式 PLY。

输出到 CASE/depthanything3_camera_audit，分别保存原始官方高斯、关闭
adapter 的 gt_extrinsics 部分缩放后重建的高斯，以及各分支相机。渲染由
41_render_camera_diagnostic.py 在 gsplat 环境完成，避免 DA3 渲染依赖差异。

Env vars: ROOT, CASE_FILTER (默认 model_man), DA3_MODEL_DIR, PROCESS_RES
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from importlib.util import module_from_spec, spec_from_file_location  # noqa: E402


def load_module(name, filename):
    spec = spec_from_file_location(name, SCRIPT_DIR / filename)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_io = load_module("hisilicon_rgb", "34_run_hisilicon.py")
_cam = load_module("hisilicon_cameras", "36_run_hisilicon_cameras.py")


def save_variant(name, prediction, cameras, raw_extrinsics, out, variants,
                 disable_adapter_gt=False):
    """同一 raw output 重跑 adapter，不能把二次独立推理当单变量对照。"""
    g = prediction.gaussians
    from depth_anything_3.utils import gsply_helpers as gh
    gh.export_ply = _io.fast_export_ply
    if disable_adapter_gt:
        # holder 中的 clean 与官方分支由同一 forward、同一深度和图像派生。
        prediction.gaussians = prediction._clean_gaussians
        g = prediction.gaussians
    path = out / f"{name}.ply"
    gh.save_gaussian_ply(
        gaussians=g, save_path=str(path),
        ctx_depth=torch.from_numpy(prediction.depth).unsqueeze(-1).to(g.means),
        shift_and_scale=False, save_sh_dc_only=False, gs_views_interval=1,
        inv_opacity=True, prune_by_depth_percent=1.0, prune_border_gs=False,
        match_3dgs_mcmc_dev=False)
    np.savez_compressed(out / f"{name}.npz", extrinsics=cameras,
                        intrinsics=prediction.intrinsics,
                        image_names=np.asarray([f"frame_{i:04d}.png" for i in range(1, 11)]))
    flat = g.means.detach().reshape(-1, 3)
    variants.append({"name": name, "ply": path.name, "disable_adapter_gt": disable_adapter_gt,
                     "mean_bounds": [flat.amin(dim=0).tolist(), flat.amax(dim=0).tolist()]})


def main():
    root = Path(os.environ.get("ROOT", "/mnt/d/dataset/测试数据/海思端侧重建拍摄数据"))
    name = os.environ.get("CASE_FILTER", "model_man")
    case = root / name
    out = case / "depthanything3_camera_audit"
    out.mkdir(exist_ok=True)
    files, ext, ixt, camera_path, _ = _cam.load_gltf_inputs(case)
    model_dir = os.environ.get("DA3_MODEL_DIR", "/mnt/d/wheel/da3_human_ms")
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained(model_dir).to("cuda")
    model.eval()
    print(f"LOADED {model_dir}; CASE={name}; INPUT_CONVENTION={os.environ.get('INPUT_CONVENTION', 'cv_flip')}; EXT_SHAPE={ext.shape}", flush=True)
    if os.environ.get("INPUT_CONVENTION") == "world_x180":
        files, ext, ixt, camera_path, _ = _cam.load_gltf_inputs(case)
        F = np.diag([1.0, -1.0, -1.0, 1.0])
        ext44 = ext.copy()
        ext = np.linalg.inv(F @ np.linalg.inv(ext44))
        out = out / "world_x180"
        out.mkdir(exist_ok=True)
    if os.environ.get("EXT_SHAPE") == "3x4":
        ext = ext[:, :3].copy()
    if os.environ.get("CAMERA_MODE") == "intrinsics_only":
        ext = None
    imgs_cpu, ext_cpu, ixt_cpu = model._preprocess_inputs(
        [str(p) for p in files], extrinsics=ext, intrinsics=ixt,
        process_res=int(os.environ.get("PROCESS_RES", "504")),
        process_res_method="upper_bound_resize")
    imgs, ext_t, ixt_t = model._prepare_model_inputs(imgs_cpu, ext_cpu, ixt_cpu)
    ext_norm = model._normalize_extrinsics(ext_t.clone() if ext_t is not None else None)
    print("SHAPES", {"images": list(imgs.shape), "ext_input": None if ext_t is None else list(ext_t.shape),
                      "ixt_input": list(ixt_t.shape), "ext_normalized": None if ext_norm is None else list(ext_norm.shape)}, flush=True)
    # 同一 forward 内拦截 adapter，调用两次同一层、同一深度，仅切换 gt_extrinsics。
    # 不能从 Prediction 重跑 gs_head：官方 output_processor 会释放中间 feats。
    adapter = model.model.da3.gs_adapter
    official_adapter = adapter.forward
    holder = {}
    def capture_adapter(*args, **kwargs):
        official = official_adapter(*args, **kwargs)
        kwargs["gt_extrinsics"] = None
        holder["clean"] = official_adapter(*args, **kwargs)
        return official
    try:
        adapter.forward = capture_adapter
        raw = model._run_model_forward(imgs, ext_norm, ixt_t, [], True, False, "saddle_balanced")
    finally:
        adapter.forward = official_adapter
    prediction = model._convert_to_prediction(raw)
    raw_extrinsics = prediction.extrinsics.copy()
    clean_gaussians = holder["clean"]
    prediction._clean_gaussians = clean_gaussians
    # 保存两个真正同次前向派生的分支。
    variants = []
    save_variant("official_raw_camera", prediction, raw_extrinsics, raw_extrinsics, out, variants)
    prediction.gaussians = clean_gaussians
    save_variant("no_adapter_scale_raw_camera", prediction, raw_extrinsics, raw_extrinsics, out,
                 variants, disable_adapter_gt=True)
    np.savez_compressed(out / "forward_depth.npz", depth=prediction.depth,
                        conf=prediction.conf, intrinsics=prediction.intrinsics,
                        extrinsics=raw_extrinsics)
    report = {"case": name, "raw_keys": sorted(raw.keys()),
              "camera_json": str(camera_path),
              "raw_extrinsics_shape": list(raw_extrinsics.shape),
              "normalized_input_extrinsics": None if ext_norm is None else ext_norm[0].cpu().numpy().tolist(),
              "scale_factor": float(prediction.scale_factor), "variants": variants}
    _io.dump_json(out / "forward_diagnostic.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
