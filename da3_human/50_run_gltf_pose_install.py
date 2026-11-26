#!/usr/bin/env python3
"""两例glTF仅位姿分支：原生/输入世界PLY，并安全安装到默认相机版入口。
Env: ROOT, RESULTS_ROOT, BACKUP_ROOT, DA3_MODEL_DIR, CASES=model_man,model_plant
"""
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

SCRIPT = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    cam = load("camera", "36_run_hisilicon_cameras.py")
    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.utils import gsply_helpers as gh
    from depth_anything_3.utils.pose_align import batch_align_poses_umeyama, align_poses_umeyama
    from depth_anything_3.model.utils.transform import mat_to_quat, quat_to_mat
    from depth_anything_3.utils.sh_helpers import rotate_sh
    gh.export_ply = cam._io.fast_export_ply
    eps = torch.finfo(torch.float32).eps
    gh.inverse_sigmoid = lambda x: torch.logit(x.clamp(eps, 1 - eps))
    root = Path(os.environ["ROOT"])
    results = Path(os.environ["RESULTS_ROOT"])
    backup = Path(os.environ["BACKUP_ROOT"])
    names = os.environ.get("CASES", "model_man,model_plant").split(",")
    if set(names) != {"model_man", "model_plant"} or results.exists() or backup.exists():
        raise ValueError("只允许这两例，且实验与备份根目录必须不存在")
    results.mkdir(parents=True)
    backup.mkdir(parents=True)
    print("🏋️ 仅加载一次DA3；两例串行，仅位姿分支", flush=True)
    model = DepthAnything3.from_pretrained(os.environ["DA3_MODEL_DIR"]).to("cuda").eval()
    adapter = model.model.da3.gs_adapter
    original = adapter.forward
    captured = {}

    def capture(*args, **kwargs):
        E = kwargs["extrinsics"].detach().float().clone()
        a = 1.0
        if kwargs.get("gt_extrinsics") is not None and not torch.equal(kwargs["extrinsics"], kwargs["gt_extrinsics"]):
            _, _, scales = batch_align_poses_umeyama(kwargs["gt_extrinsics"].detach().float(), E)
            a = float(scales.clamp(1 / 3, 3)[0].cpu())
        captured.update(extrinsics=E.cpu().numpy()[0], adapter_scale=a)
        return original(*args, **kwargs)

    for name in names:
        case = root / name
        out = results / name
        out.mkdir()
        files, _, K, camera_path, manifest = cam.load_gltf_inputs(case)
        if len(files) != 10:
            raise RuntimeError(f"不是预期的全10视角: {name}")
        data = json.loads(camera_path.read_text(encoding="utf-8"))
        c2w = np.asarray(data["extrinsics"]["camera_path"]["c2w_keyframes"]).copy()
        c2w[:, :3, 3] *= -1
        ext = np.linalg.inv(c2w)
        torch.manual_seed(42)
        np.random.seed(42)
        torch.cuda.reset_peak_memory_stats()
        captured.clear()
        adapter.forward = capture
        try:
            print(f"🚀 {name}: 全10视角、修正位姿与动画内参", flush=True)
            pred = model.inference([str(p) for p in files], extrinsics=ext, intrinsics=K,
                                   process_res=704, infer_gs=True,
                                   ref_view_strategy="saddle_balanced", export_dir=None, export_format="")
        finally:
            adapter.forward = original
        E = cam.adapter_camera_extrinsics(captured["extrinsics"], captured["adapter_scale"])
        np.savez_compressed(out / "native_cameras.npz", extrinsics=E[:, :3],
                            intrinsics=pred.intrinsics, image_shape=np.asarray(pred.depth.shape[-2:]),
                            image_names=np.asarray([p.name for p in files]))
        gh.save_gaussian_ply(gaussians=pred.gaussians, save_path=str(out / "native.ply"),
                            ctx_depth=torch.from_numpy(pred.depth).unsqueeze(-1).to(pred.gaussians.means),
                            shift_and_scale=False, save_sh_dc_only=False, gs_views_interval=1,
                            inv_opacity=True, prune_by_depth_percent=1., prune_border_gs=False,
                            match_3dgs_mcmc_dev=False)
        pred.gs_extrinsics = E
        pred.adapter_scale = captured["adapter_scale"]
        alignment = cam.transform_gltf_gaussians(pred, ext)
        gh.save_gaussian_ply(gaussians=pred.gaussians, save_path=str(out / "input_world.ply"),
                            ctx_depth=torch.from_numpy(pred.depth).unsqueeze(-1).to(pred.gaussians.means),
                            shift_and_scale=False, save_sh_dc_only=False, gs_views_interval=1,
                            inv_opacity=True, prune_by_depth_percent=1., prune_border_gs=False,
                            match_3dgs_mcmc_dev=False)
        report = {"case": str(case), "camera_json_sha256": cam._io.sha256(camera_path),
                  "input_sha256": [cam._io.sha256(p) for p in files],
                  "camera_interpretation": "保持R，相机中心取负；动画camera_path_yfov；仅适用于已审计两例",
                  "n_views": 10, "process_res": 704, "seed": 42,
                  "adapter_scale": captured["adapter_scale"], "input_alignment": alignment}
        info = cam._io.inspect_ply(out / "input_world.ply")
        if info["n_gaussians"] != 4900000 or info["n_properties"] != 41:
            raise RuntimeError(f"不是完整10视角490万高斯: {name}")
        report["input_world_ply"] = info
        cameras = {"extrinsics": ext[:, :3], "intrinsics": K,
                   "input_extrinsics_original": ext, "input_intrinsics_original": K,
                   "depth": pred.depth, "conf": pred.conf,
                   "scale_factor": np.float32(alignment["total_gaussian_scale"]),
                   "is_metric": np.int32(0), "view_indices": np.arange(10),
                   "source_indices": np.arange(10), "image_names": np.asarray([p.name for p in files]),
                   "world_rotation": np.eye(3),
                   "coordinate": np.asarray("修正glTF输入世界；OpenCV w2c；无显示旋转")}
        np.savez_compressed(out / "input_cameras.npz", **cameras)
        del pred
        torch.cuda.empty_cache()
        print(f"📦 {name}: 新输入世界PLY已生成，核验后替换默认入口", flush=True)
        target = case / "depthanything3_cameras.ply"
        destination = case / "depthanything3_cameras"
        old_files = [target, destination / "cameras.npz", destination / "summary.json"]
        if any(p.is_symlink() or not p.is_file() for p in old_files):
            raise RuntimeError(f"默认入口缺失或为链接: {name}")
        case_backup = backup / name
        case_backup.mkdir()
        old_hashes = {}
        for old in old_files:
            digest = cam._io.sha256(old)
            old.replace(case_backup / old.name)
            if cam._io.sha256(case_backup / old.name) != digest:
                raise RuntimeError(f"旧入口备份哈希不一致: {name}")
            old_hashes[old.name] = digest
        (out / "input_world.ply").replace(target)
        (out / "input_cameras.npz").replace(destination / "cameras.npz")
        if cam._io.sha256(target) != info["sha256"]:
            raise RuntimeError(f"安装后PLY哈希不一致: {name}")
        report.update(info)
        report.update({"case": name, "status": "completed",
                       "camera_conversion": "negative_centers", "intrinsics_mode": "camera_path_yfov",
                       "world_rotation": np.eye(3).tolist(), "gaussian_frame": "修正glTF输入世界；无显示旋转",
                       "output_world_note": "保持R，中心取负的已验证输入世界；不推广为通用glTF规则",
                       "output_extrinsics": ext[:, :3].tolist(),
                       "native_ply": str(out / "native.ply"), "old_entry_backup": str(case_backup),
                       "old_entry_sha256": old_hashes,
                       "installed_at": datetime.now(timezone.utc).isoformat(),
                       "scope": "用户指定的全10视角位姿DA3默认入口；非8源留出实验，不宣称质量通过"})
        cam._io.dump_json(destination / "summary.json", report)
        cam._io.dump_json(destination / "batch_summary.json", report)
        cam._io.dump_json(out / "install_summary.json", report)
        print(f"✅ {name}: 已替换 {target}；{info['n_gaussians']:,}高斯，旧文件已备份", flush=True)
    print(f"🎉 两例均完成。实验目录={results}；旧文件备份={backup}", flush=True)


if __name__ == "__main__":
    main()
