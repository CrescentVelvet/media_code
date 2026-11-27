#!/usr/bin/env python3
"""同次 DA3 前向比较原生 RGB-only 与输入相机，不覆盖正式重建结果。

每支分别保存原生高斯/原生相机、Nested 米制高斯，以及 API 返回的输入相机。
仅用于定位本地导出与原生模型之间的差异。不包含自定义 Sim(3) 后处理。

Env vars: ROOT, CASE_FILTER, DA3_MODEL_DIR, OUT_SUBDIR, PROCESS_RES, SEED,
          INTRINSICS_MODE=top_level_K|camera_path_yfov, BRANCHES=rgb,pose,
          REF_STRATEGY=saddle_balanced|first|firsts|middle|saddle
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

_spec = spec_from_file_location("camera", SCRIPT_DIR / "36_run_hisilicon_cameras.py")
_cam = module_from_spec(_spec)
_spec.loader.exec_module(_cam)


def main():
    root = Path(os.environ.get("ROOT", "/mnt/d/dataset/测试数据/海思端侧重建拍摄数据"))
    case = root / os.environ.get("CASE_FILTER", "model_man")
    out = case / os.environ.get("OUT_SUBDIR", "depthanything3_camera_audit/colleague_700")
    if out.is_symlink() or ".." in out.relative_to(case).parts:
        sys.exit("❌ 诊断输出不得为符号链接或越出案例目录")
    out.mkdir(parents=True, exist_ok=True)
    process_res = int(os.environ.get("PROCESS_RES", "704"))
    seed = int(os.environ.get("SEED", "42"))
    files, ext, ixt, camera_path, _ = _cam.load_gltf_inputs(case)
    intrinsics_mode = os.environ.get("INTRINSICS_MODE", "top_level_K")
    if intrinsics_mode == "camera_path_yfov":
        data = json.loads(camera_path.read_text(encoding="utf-8"))
        yfov = np.asarray(data["extrinsics"]["camera_path"]["yfov"]).reshape(-1)
        size = data["intrinsics"]["reference_image_size"]
        focal = size["height"] / (2 * np.tan(yfov / 2))
        ixt[:] = 0
        ixt[:, 0, 0] = focal
        ixt[:, 1, 1] = focal * size["width"] / size["height"]
        ixt[:, 0, 2] = size["width"] / 2
        ixt[:, 1, 2] = size["height"] / 2
        ixt[:, 2, 2] = 1
    elif intrinsics_mode != "top_level_K":
        raise ValueError(intrinsics_mode)

    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.utils import gsply_helpers as gh
    gh.export_ply = _cam._io.fast_export_ply

    def stable_inverse_sigmoid(x):
        eps = torch.finfo(x.dtype).eps
        return torch.where(x >= 1,
                           torch.full_like(x, -torch.log(torch.tensor(
                               eps, dtype=x.dtype, device=x.device))),
                           torch.log(x) - torch.log1p(-x))

    gh.inverse_sigmoid = stable_inverse_sigmoid
    model = DepthAnything3.from_pretrained(
        os.environ.get("DA3_MODEL_DIR", "/mnt/d/wheel/da3_human_ms")).to("cuda").eval()
    report = {"process_res": process_res, "seed": seed,
              "intrinsics_mode": intrinsics_mode,
              "ref_view_strategy": os.environ.get("REF_STRATEGY", "saddle_balanced"),
              "camera_frame": "native = adapter原生高斯 + adapter预测相机；metric = 同高斯乘Nested sf + Nested后预测相机",
              "branches": {}}
    ref_strategy = report["ref_view_strategy"]
    if ref_strategy not in ("saddle_balanced", "first", "firsts", "middle", "saddle"):
        raise ValueError(f"未知 REF_STRATEGY: {ref_strategy}")
    adapter = model.model.da3.gs_adapter
    original_adapter = adapter.forward
    captured = {}

    def capture_adapter(*args, **kwargs):
        captured["extrinsics"] = kwargs["extrinsics"].detach().float().cpu().numpy()[0]
        captured["intrinsics"] = kwargs["intrinsics"].detach().float().cpu().numpy()[0]
        return original_adapter(*args, **kwargs)

    for name, use_cameras in (("rgb", False), ("pose", True)):
        if name not in os.environ.get("BRANCHES", "rgb,pose").split(","):
            continue
        captured.clear()
        adapter.forward = capture_adapter
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.reset_peak_memory_stats()
        prediction = model.inference(
            [str(p) for p in files], extrinsics=ext if use_cameras else None,
            intrinsics=ixt if use_cameras else None, process_res=process_res,
            process_res_method="upper_bound_resize", ref_view_strategy=ref_strategy,
            infer_gs=True, export_dir=None, export_format="")
        adapter.forward = original_adapter
        if not captured:
            raise RuntimeError("未捕获高斯头的预测相机")
        prediction.gs_extrinsics = captured["extrinsics"]
        prediction.gs_intrinsics = captured["intrinsics"]
        info = {"depth_shape": list(prediction.depth.shape),
                "depth_median": float(np.median(prediction.depth)),
                "gaussian_shape": list(prediction.gaussians.means.shape),
                "intrinsics": prediction.intrinsics.tolist(),
                "extrinsics": prediction.extrinsics.tolist(),
                "is_metric": int(prediction.is_metric),
                "scale_factor": float(prediction.scale_factor),
                "gs_intrinsics": prediction.gs_intrinsics.tolist(),
                "gs_extrinsics": prediction.gs_extrinsics.tolist(),
                "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024 ** 3}
        np.savez_compressed(out / f"{name}_gs_cameras.npz",
                            extrinsics=prediction.gs_extrinsics,
                            intrinsics=prediction.gs_intrinsics,
                            image_shape=np.asarray(prediction.depth.shape[-2:]),
                            image_names=np.asarray([p.name for p in files]))
        g = prediction.gaussians
        finite = torch.isfinite(torch.cat([
            g.means, g.scales, g.rotations, g.harmonics.reshape(1, len(g.means[0]), -1),
            g.opacities[..., None]], dim=-1)).all(dim=-1)
        info["nonfinite_gaussians"] = int((~finite[0]).sum())
        info["opacity_nan"] = int(torch.isnan(g.opacities).sum())
        info["opacity_inf"] = int(torch.isinf(g.opacities).sum())
        if info["nonfinite_gaussians"]:
            # 诊断导出隔离非法点并登记，不把 NaN 写进 PLY 或渲染器；不改变有限点几何。
            from depth_anything_3.specs import Gaussians
            print(f"⚠️ {name}: 隔离 {info['nonfinite_gaussians']} 个非有限高斯", flush=True)
            g = Gaussians(**{key: getattr(g, key)[:, finite[0]]
                             for key in ("means", "scales", "rotations", "harmonics", "opacities")})
            prediction.gaussians = g
        info["finite_gaussians"] = int(g.means.shape[1])
        info["native_bounds"] = [g.means.amin(dim=(0, 1)).tolist(),
                                 g.means.amax(dim=(0, 1)).tolist()]
        native_cameras = np.repeat(np.eye(4, dtype=np.float32)[None], len(files), axis=0)
        native_cameras[:, :3] = prediction.gs_extrinsics[:, :3]
        sf = float(prediction.scale_factor)
        native_cameras[:, :3, 3] /= sf
        np.savez_compressed(out / f"{name}_native_cameras.npz",
                            extrinsics=native_cameras,
                            intrinsics=prediction.gs_intrinsics,
                            depth=prediction.depth,
                            conf=prediction.conf,
                            finite_mask=finite[0].detach().cpu().numpy(),
                            scale_factor=prediction.scale_factor,
                            camera_frame="native PLY对应的adapter世界；撤销Nested相机平移尺度",
                            image_names=np.asarray([p.name for p in files]))
        np.savez_compressed(out / f"{name}_metric_cameras.npz",
                            extrinsics=prediction.gs_extrinsics,
                            intrinsics=prediction.gs_intrinsics,
                            depth=prediction.depth,
                            conf=prediction.conf,
                            finite_mask=finite[0].detach().cpu().numpy(),
                            scale_factor=prediction.scale_factor,
                            camera_frame="metric PLY对应的Nested预测世界",
                            image_names=np.asarray([p.name for p in files]))
        np.savez_compressed(out / f"{name}_input_cameras.npz",
                            extrinsics=prediction.extrinsics,
                            intrinsics=prediction.intrinsics,
                            depth=prediction.depth,
                            image_names=np.asarray([p.name for p in files]))
        gh.save_gaussian_ply(
            gaussians=g, save_path=str(out / f"{name}_native.ply"),
            ctx_depth=torch.from_numpy(prediction.depth).unsqueeze(-1).to(g.means),
            shift_and_scale=False, save_sh_dc_only=True, gs_views_interval=1,
            inv_opacity=True, prune_by_depth_percent=1.0,
            prune_border_gs=False, match_3dgs_mcmc_dev=False)
        sf = float(prediction.scale_factor)
        g = prediction.gaussians
        g.means = g.means * sf
        g.scales = g.scales * sf
        gh.save_gaussian_ply(
            gaussians=g, save_path=str(out / f"{name}_metric.ply"),
            ctx_depth=torch.from_numpy(prediction.depth).unsqueeze(-1).to(g.means),
            shift_and_scale=False, save_sh_dc_only=True, gs_views_interval=1,
            inv_opacity=True, prune_by_depth_percent=1.0,
            prune_border_gs=False, match_3dgs_mcmc_dev=False)
        info["metric_bounds"] = [g.means.amin(dim=(0, 1)).tolist(),
                                 g.means.amax(dim=(0, 1)).tolist()]
        report["branches"][name] = info
        print(name, json.dumps({k: v for k, v in info.items()
                                if k not in ("intrinsics", "extrinsics")}), flush=True)
        _cam._io.dump_json(out / "summary.json", report)
        del prediction, g
        torch.cuda.empty_cache()
    adapter.forward = original_adapter
    print(f"✅ {out}", flush=True)


if __name__ == "__main__":
    main()
