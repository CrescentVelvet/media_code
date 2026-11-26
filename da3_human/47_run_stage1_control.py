#!/usr/bin/env python3
"""旧 stage1 相机条件的原生/对齐高斯及 8 源 2 留出对照，不覆盖历史 PLY。

Env vars: ROOT, CASE_FILTER, DA3_MODEL_DIR, RESULTS_DIR (必须不存在),
          PROCESS_RES=704, BRANCHES=pose,rgb,
          SOURCE_INDICES=0,1,3,4,5,6,8,9（同一文件内全部视角的索引）,
          HOLDOUT=1 时仅 source 前向，target 相机仅 source 拟合 Sim(3)
"""
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch

SCRIPT = Path(__file__).resolve().parent


def main():
    spec = importlib.util.spec_from_file_location("cam", SCRIPT / "36_run_hisilicon_cameras.py")
    cam = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cam)
    root = Path(os.environ["ROOT"])
    case = root / os.environ["CASE_FILTER"]
    out = Path(os.environ["RESULTS_DIR"])
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    all_files, all_ext, all_K, camera_path, manifest = cam.load_stage1_inputs(case)
    indices = np.asarray([int(x) for x in os.environ.get("SOURCE_INDICES", "0,1,3,4,5,6,8,9").split(",")])
    targets = np.asarray([i for i in range(len(all_files)) if i not in indices], dtype=int)
    if len(indices) < 3 or len(np.unique(indices)) != len(indices) or indices.min() < 0 or indices.max() >= len(all_files):
        raise ValueError("SOURCE_INDICES 须至少三个不重复的有效索引")
    holdout = os.environ.get("HOLDOUT", "0") == "1"
    files = [all_files[i] for i in indices] if holdout else all_files
    ext = all_ext[indices] if holdout else all_ext
    K = all_K[indices] if holdout else all_K
    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.utils import gsply_helpers as gh
    from depth_anything_3.utils.pose_align import batch_align_poses_umeyama, align_poses_umeyama
    gh.export_ply = cam._io.fast_export_ply
    eps = torch.finfo(torch.float32).eps
    gh.inverse_sigmoid = lambda x: torch.logit(x.clamp(eps, 1 - eps))
    model_path = os.environ["DA3_MODEL_DIR"]
    model = DepthAnything3.from_pretrained(model_path).to("cuda").eval()
    adapter = model.model.da3.gs_adapter
    original = adapter.forward
    captured = {}

    def capture(*args, **kwargs):
        E = kwargs["extrinsics"].detach().float().clone()
        a = 1.
        if kwargs.get("gt_extrinsics") is not None and not torch.equal(kwargs["extrinsics"], kwargs["gt_extrinsics"]):
            _, _, scales = batch_align_poses_umeyama(kwargs["gt_extrinsics"].detach().float(), E)
            a = float(scales.clamp(1 / 3, 3)[0].cpu())
        captured.update(extrinsics=E.cpu().numpy()[0],
                        intrinsics=kwargs["intrinsics"].detach().float().cpu().numpy()[0], adapter_scale=a)
        return original(*args, **kwargs)

    report = {"case": str(case), "model": model_path, "process_res": int(os.environ.get("PROCESS_RES", "704")),
              "input_sha256": [cam._io.sha256(p) for p in files], "camera_sha256": cam._io.sha256(camera_path),
              "seed": 42, "source_indices": indices.tolist(), "target_indices": targets.tolist(),
              "holdout": holdout, "branches": {}}
    for branch in os.environ.get("BRANCHES", "pose,rgb").split(","):
        if branch not in ("pose", "rgb"):
            raise ValueError(branch)
        dest = out / branch
        dest.mkdir()
        torch.manual_seed(42)
        np.random.seed(42)
        torch.cuda.reset_peak_memory_stats()
        captured.clear()
        adapter.forward = capture
        try:
            pred = model.inference([str(p) for p in files], extrinsics=ext if branch == "pose" else None,
                                   intrinsics=K if branch == "pose" else None,
                                   process_res=report["process_res"], infer_gs=True,
                                   ref_view_strategy="saddle_balanced", export_dir=None, export_format="")
        finally:
            adapter.forward = original
        E = cam.adapter_camera_extrinsics(captured["extrinsics"], captured["adapter_scale"])
        np.savez_compressed(dest / "native_cameras.npz", extrinsics=E[:, :3], intrinsics=captured["intrinsics"],
                            image_shape=np.asarray(pred.depth.shape[-2:]),
                            image_names=np.asarray([p.name for p in files]),
                            camera_frame="adapter原生预测世界；w2c平移含adapter尺度")
        gh.save_gaussian_ply(gaussians=pred.gaussians, save_path=str(dest / "native.ply"),
                             ctx_depth=torch.from_numpy(pred.depth).unsqueeze(-1).to(pred.gaussians.means),
                             shift_and_scale=False, save_sh_dc_only=False, gs_views_interval=1,
                             inv_opacity=True, prune_by_depth_percent=1., prune_border_gs=False,
                             match_3dgs_mcmc_dev=False)
        info = {**cam._io.inspect_ply(dest / "native.ply"), "adapter_scale": captured["adapter_scale"],
                "depth_shape": list(pred.depth.shape), "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024 ** 3}
        if branch == "pose":
            # 同前向另导一份输入相机世界；原生文件和相机始终保留，不用输出回放反推原生质量。
            pred.gs_extrinsics = E
            pred.adapter_scale = captured["adapter_scale"]
            align = cam.transform_gltf_gaussians(pred, ext)
            np.savez_compressed(dest / "input_cameras.npz", extrinsics=ext[:, :3], intrinsics=K,
                                image_shape=np.asarray([704, 704]), image_names=np.asarray([p.name for p in files]),
                                camera_frame="stage1 OpenCV；无显示旋转")
            gh.save_gaussian_ply(gaussians=pred.gaussians, save_path=str(dest / "input_world.ply"),
                                 ctx_depth=torch.from_numpy(pred.depth).unsqueeze(-1).to(pred.gaussians.means),
                                 shift_and_scale=False, save_sh_dc_only=False, gs_views_interval=1,
                                 inv_opacity=True, prune_by_depth_percent=1., prune_border_gs=False,
                                 match_3dgs_mcmc_dev=False)
            info["input_alignment"] = align
        if holdout and len(targets):
            # 列向量 Sim(3)：C_native=R.T@(C_stage1-t)/scale；只使用source估计。
            R, t, scale = align_poses_umeyama(ext, E, ransac=False)
            for suffix, selected in (("common_source", indices), ("heldout", targets)):
                c2w = np.linalg.inv(all_ext[selected]).copy()
                c2w[:, :3, :3] = R.T @ c2w[:, :3, :3]
                c2w[:, :3, 3] = (c2w[:, :3, 3] - t) @ R / scale
                np.savez_compressed(dest / f"{suffix}_cameras.npz",
                                    extrinsics=np.linalg.inv(c2w)[:, :3], intrinsics=all_K[selected],
                                    image_shape=np.asarray([704, 704]),
                                    image_names=np.asarray([all_files[i].name for i in selected]),
                                    camera_frame="stage1相机仅source拟合Sim(3)映射到adapter原生世界")
            info["source_to_native_alignment"] = {"R": R.tolist(), "t": t.tolist(), "scale": float(scale)}
        report["branches"][branch] = info
        cam._io.dump_json(out / "summary.json", report)
        del pred
        torch.cuda.empty_cache()
        print(f"✅ {branch}: {info}", flush=True)


if __name__ == "__main__":
    main()
