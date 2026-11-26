#!/usr/bin/env python3
"""海思数据集：使用最终 COLMAP SfM 相机运行 DA3，导出 depthanything3_sfm.ply。

仅使用 colmap_sfm_results/sparse/0 注册成功的图像与相机。OPENCV 畸变先转为
同尺寸、同内参的无畸变 RGB，DA3 不接受畸变参数，不能忽略 k1/k2/p1/p2。
高斯经完整 Sim(3) 对齐回 SFM 世界；不施加显示旋转，不假定 SFM 单位为米。

Env vars:
  ROOT, CASE_FILTER, FORCE, PROCESS_RES, DA3_MODEL_DIR
  OUT_ROOT 可选：将案例目录映射至不存在的新实验根，正式默认仍写各案例目录。
  仅外层 shell 创建 OUT_ROOT 供日志落盘；Python 不重复创建该根。
  PLY_NAME / OUT_SUBDIR 仅新实验根可覆盖；默认 depthanything3_sfm(.ply)。
  正式重跑自动将旧 PLY、相机和摘要备份至各案例 depthanything3_sfm_backup_*。
  单例 CASE_FILTER 的摘要写到该案例输出目录，不覆盖根目录批量总表。
  无最终 summary/sparse/0/txt 的案例跳过并登记；EXPECTED_CASES 可选约束批量数。
"""
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "infinisplat_v2"))
from sfm_export import read_model, rotation
_repo_root = SCRIPT_DIR.parent.parent
# 本地开发路径由 proxy.env/环境继承；服务器仓位于 media_code 同级。
for _candidate in (_repo_root / "Depth-Anything-3/src",
                   Path.home() / "repos/Depth-Anything-3/src"):
    if _candidate.is_dir():
        sys.path.insert(0, str(_candidate))
        break

_spec = importlib.util.spec_from_file_location("hisilicon_rgb", SCRIPT_DIR / "34_run_hisilicon.py")
_io = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_io)
_spec = importlib.util.spec_from_file_location(
    "hisilicon_cameras", SCRIPT_DIR / "36_run_hisilicon_cameras.py")
_camera_pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_camera_pipeline)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_sfm_inputs(case, out):
    """按最终 SfM 模型匹配图像；不为未注册图猜位姿。"""
    sfm = case / "colmap_sfm_results"
    model_dir = sfm / "sparse/0/txt"
    cameras, records, _ = read_model(model_dir)
    if not records:
        raise RuntimeError(f"{case.name}: 最终 SfM 没有注册图像")
    images_dir, undistorted_dir = case / "images", out / "images"
    undistorted_dir.mkdir(parents=True, exist_ok=True)
    files, w2cs, intrinsics, image_hashes = [], [], [], []
    for record in sorted(records, key=lambda r: r["name"]):
        cam = cameras[record["camera_id"]]
        if cam["model"] != "OPENCV":
            raise RuntimeError(f"需要 OPENCV 畸变模型，实际为 {cam['model']}: {case.name}")
        fx, fy, cx, cy, k1, k2, p1, p2 = cam["params"]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        # COLMAP 四元数顺序为 wxyz，转成 OpenCV world-to-camera 旋转。
        R = rotation(record["q"])
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3], w2c[:3, 3] = R, record["t"]
        if np.abs(R @ R.T - np.eye(3)).max() > 1e-8 or abs(np.linalg.det(R) - 1) > 1e-8:
            raise RuntimeError(f"非法 SfM 外参: {case.name}/{record['name']}")
        source = images_dir / record["name"]
        target = undistorted_dir / record["name"]
        source_hash = sha256(source)
        source_pixels = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if source_pixels is None:
            raise RuntimeError(f"图像读取失败: {source}")
        with Image.open(source) as im:
            if im.size != (cam["width"], cam["height"]):
                raise RuntimeError(f"图像尺寸与 SfM 相机不一致: {source}")
        # R=I 表示不做立体校正，只移除镜头畸变；K 与外参保持原 SfM 口径。
        rectified = cv2.undistort(source_pixels, K,
                                  np.array([k1, k2, p1, p2], dtype=np.float64), None, K)
        if not cv2.imwrite(str(target), rectified):
            raise RuntimeError(f"无畸变图写入失败: {target}")
        files.append(target)
        w2cs.append(w2c)
        intrinsics.append(K)
        image_hashes.append(source_hash)
    auxiliary = [sfm / "summary.json", sfm / "cameras.npz",
                 model_dir / "cameras.txt", model_dir / "images.txt"]
    provenance = {str(p.relative_to(case)): sha256(p) for p in auxiliary}
    return {"files": files, "w2cs": np.asarray(w2cs), "intrinsics": np.asarray(intrinsics),
            "names": [p.name for p in files], "source_sha256": image_hashes,
            "sfm_provenance": provenance, "n_input": len(list(images_dir.glob("*.png"))),
            "sfm_dir": sfm}


def main():
    root = Path(os.environ.get("ROOT", "/mnt/d/dataset/测试数据/海思端侧重建拍摄数据"))
    selected = os.environ.get("CASE_FILTER", "")
    force = os.environ.get("FORCE", "0") == "1"
    process_res = int(os.environ.get("PROCESS_RES", "504"))
    model_dir = os.environ.get("DA3_MODEL_DIR", "") or os.environ.get("DA3_HUMAN_MODEL_DIR", "")
    if not root.is_dir() or not (Path(model_dir) / "model.safetensors").is_file():
        sys.exit(f"❌ ROOT/权重无效: ROOT={root}, DA3_MODEL_DIR={model_dir}")
    experiment_root = os.environ.get("OUT_ROOT", "")
    experiment = Path(experiment_root) if experiment_root else None
    if experiment is not None:
        if not experiment.is_absolute():
            sys.exit("❌ OUT_ROOT 必须是绝对路径")
    ply_name = os.environ.get("PLY_NAME", "depthanything3_sfm.ply")
    out_subdir = os.environ.get("OUT_SUBDIR", "depthanything3_sfm")
    if (Path(ply_name).is_absolute() or ".." in Path(ply_name).parts
            or Path(out_subdir).is_absolute() or ".." in Path(out_subdir).parts
            or (experiment is None and (ply_name != "depthanything3_sfm.ply"
                                        or out_subdir != "depthanything3_sfm"))):
        sys.exit("❌ 案例内路径必须相对且无..；仅新 OUT_ROOT 可自定义文件名/子目录")
    sys.path.insert(0, str(SCRIPT_DIR.parent / "infinisplat_v2"))
    cases = sorted((p for p in root.glob("case*_nopose_recentered")
                    if (p / "colmap_sfm_results/summary.json").is_file()
                    and (p / "colmap_sfm_results/sparse/0/txt/images.txt").is_file()),
                   key=lambda p: int(p.name.split("_")[0][4:]))
    skipped_incomplete = [p.name for p in root.glob("case*_nopose_recentered")
                          if not (p / "colmap_sfm_results/summary.json").is_file()
                          or not (p / "colmap_sfm_results/sparse/0/txt/images.txt").is_file()]
    if skipped_incomplete:
        print(f"⚠️ 跳过无最终 SFM 导出案例: {skipped_incomplete}", flush=True)
    if selected:
        cases = [p for p in cases if p.name == selected]
    if not cases:
        sys.exit("❌ 没有匹配案例")
    pending = []
    expected_cases = os.environ.get("EXPECTED_CASES", "")
    if expected_cases and len(cases) != int(expected_cases):
        sys.exit(f"❌ 最终案例数与预期不符: {len(cases)} != {expected_cases}")
    if experiment:
        report_path = experiment / "depthanything3_sfm_batch_summary.json"
    elif selected:
        report_path = cases[0] / out_subdir / "depthanything3_sfm_single_summary.json"
    else:
        report_path = root / "depthanything3_sfm_batch_summary.json"
    for case in cases:
        dest = experiment / case.name if experiment else case
        out = dest / out_subdir
        target = dest / ply_name
        if out.is_symlink() or target.is_symlink():
            sys.exit(f"❌ 输出为符号链接，拒绝写入: {case.name}")
        if experiment is not None and (target.exists() or out.exists()):
            sys.exit(f"❌ 目标已存在，拒绝覆盖或复用未登记实验: {target}, {out}")
        out.mkdir(parents=True, exist_ok=True)
        inputs = load_sfm_inputs(case, out)
        if target.is_file() and (out / "cameras.npz").is_file() and (out / "summary.json").is_file() and not force:
            previous = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            if previous.get("alignment_version") != "adapter_scaled_camera_single_sim3_v2":
                raise RuntimeError(f"{case.name}: 已有旧尺度链结果；正式更新须 FORCE=1，或将 OUT_ROOT 指向新目录")
            if previous.get("sfm_provenance") != inputs["sfm_provenance"]:
                raise RuntimeError(f"SfM 输入已改变，不能复用旧结果: {case.name}; 请用 FORCE=1")
            previous.update(_io.inspect_ply(target))
            pending.append((case, inputs, previous))
        elif target.exists() and not target.is_file():
            sys.exit(f"❌ 目标不是普通文件: {target}")
        else:
            pending.append((case, inputs, None))

    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.specs import Gaussians
    from depth_anything_3.utils import gsply_helpers as gh
    gh.export_ply = _io.fast_export_ply
    gh.inverse_sigmoid = lambda x: torch.where(
        x >= 1, torch.full_like(x, -torch.log(torch.tensor(torch.finfo(x.dtype).eps,
                                dtype=x.dtype, device=x.device))),
        torch.log(x) - torch.log1p(-x))
    print(f"🏋️ 加载原生 DA3: {model_dir}", flush=True)
    model = DepthAnything3.from_pretrained(model_dir).to("cuda")
    report = {"root": str(root), "model": model_dir, "process_res": process_res,
              "camera_input": "colmap_sfm_results/sparse/0；仅注册成功图像；OPENCV 去畸变后 K",
              "gaussian_frame": "完整 Umeyama Sim(3) 对齐至 SfM 原生坐标；无显示旋转；SfM 任意尺度非米制",
              "export": {"prune_border_gs": False, "prune_by_depth_percent": 1.0,
                         "save_sh_dc_only": False, "scale": "sfm_arbitrary_units"},
              "alignment_version": "adapter_scaled_camera_single_sim3_v2",
              "scale_formula": "Umeyama(input, adapter_scaled_pred) -> adapter原生高斯；不含Nested sf或再次除a",
              "started_at": datetime.now(timezone.utc).isoformat(),
              "skipped_incomplete_sfm": skipped_incomplete, "cases": []}
    failed = False
    for case, inputs, previous in pending:
        dest = experiment / case.name if experiment else case
        out, target = dest / out_subdir, dest / ply_name
        if previous is not None:
            previous.update({"case": case.name, "status": "existing"})
            report["cases"].append(previous)
            _io.dump_json(out / "summary.json", previous)
            _io.dump_json(report_path, report)
            print(f"⏭️ {case.name}: 有效旧结果，跳过", flush=True)
            continue
        backup = None
        if target.exists():
            # 保留旧 PLY、相机、摘要和回放；其余图像/深度可再生。
            backup_dir = case / ("depthanything3_sfm_backup_" + time.strftime("%Y%m%d_%H%M%S"))
            backup_dir.mkdir()
            for path in [target, out / "cameras.npz", out / "summary.json",
                         out / "render_check.jpg", out / "render_check.json"]:
                if path.exists():
                    shutil.move(str(path), str(backup_dir / path.name))
            backup = str(backup_dir)
            print(f"📦 原结果备份: {backup}", flush=True)
        try:
            print(f"🚀 {case.name}: {len(inputs['files'])}/{inputs['n_input']} 个 SfM 注册视角", flush=True)
            t0 = time.time()
            torch.manual_seed(42)
            np.random.seed(42)
            torch.cuda.reset_peak_memory_stats()
            prediction, raw_extrinsics = _camera_pipeline.inference_preserving_raw_cameras(
                model, inputs["files"], inputs["w2cs"], inputs["intrinsics"], process_res)
            alignment = _camera_pipeline.transform_gaussians(
                prediction, raw_extrinsics, inputs["w2cs"], world_rotation=np.eye(3),
                use_adapter_camera=True)
            np.savez_compressed(out / "raw_cameras.npz", raw_extrinsics=raw_extrinsics,
                adapter_extrinsics_before_scale=prediction.gs_extrinsics_before_scale,
                adapter_extrinsics=prediction.gs_extrinsics,
                adapter_intrinsics=prediction.gs_intrinsics)
            native = Gaussians(**{key: getattr(prediction.gaussians, key).clone()
                for key in ("means", "scales", "rotations", "harmonics", "opacities")})
            np.savez_compressed(out / "native_cameras.npz",
                extrinsics=prediction.gs_extrinsics[:, :3],
                intrinsics=prediction.gs_intrinsics, depth=prediction.depth,
                image_shape=np.asarray(prediction.depth.shape[-2:]),
                image_names=np.asarray(inputs["names"]),
                camera_frame="adapter原生预测世界；与输入SFM坐标不同，勿与input world PLY混用")
            gh.save_gaussian_ply(gaussians=native, save_path=str(out / "native.ply"),
                ctx_depth=torch.from_numpy(prediction.depth).unsqueeze(-1).to(native.means),
                shift_and_scale=False, save_sh_dc_only=False, gs_views_interval=1,
                inv_opacity=True, prune_by_depth_percent=1.0, prune_border_gs=False,
                match_3dgs_mcmc_dev=False)
            del native
            np.savez_compressed(out / "cameras.npz",
                extrinsics=alignment["output_extrinsics"], intrinsics=prediction.intrinsics,
                input_extrinsics_original=inputs["w2cs"], input_intrinsics_original=inputs["intrinsics"],
                depth=prediction.depth, conf=prediction.conf,
                scale_factor=np.float32(alignment["total_gaussian_scale"]), is_metric=np.int32(0),
                image_names=np.asarray(inputs["names"]),
                source_indices=np.asarray([int(p.stem.split("_source_")[1]) for p in inputs["files"]]),
                world_rotation=np.eye(3), coordinate_units="SfM arbitrary units, not meters",
                image_shape=np.asarray(prediction.depth.shape[-2:]),
                alignment_version="adapter_scaled_camera_single_sim3_v2")
            ctx_depth = torch.from_numpy(prediction.depth).unsqueeze(-1).to(prediction.gaussians.means)
            gh.save_gaussian_ply(gaussians=prediction.gaussians, save_path=str(target), ctx_depth=ctx_depth,
                shift_and_scale=False, save_sh_dc_only=False, gs_views_interval=1, inv_opacity=True,
                prune_by_depth_percent=1.0, prune_border_gs=False, match_3dgs_mcmc_dev=False)
            info = _io.inspect_ply(target)
            info.update({"case": case.name, "status": "completed", "seconds": round(time.time() - t0, 1),
                "n_views": len(inputs["files"]), "n_input": inputs["n_input"],
                "image_names": inputs["names"], "input_sha256": inputs["source_sha256"],
                "sfm_provenance": inputs["sfm_provenance"],
                "sfm_registered_names": json.loads((inputs["sfm_dir"] / "summary.json").read_text(
                    encoding="utf-8"))["registered_names"],
                "depth_median_sfm_units": float(np.median(prediction.depth)),
                "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
                "backup": backup, "world_rotation": np.eye(3).tolist(),
                "alignment_version": "adapter_scaled_camera_single_sim3_v2", **alignment})
            _io.dump_json(out / "summary.json", info)
            report["cases"].append(info)
            print(f"✅ {case.name}: {info['n_gaussians']:,} 高斯, "
                  f"{info['bytes'] / 1024 ** 2:.1f} MiB, {info['seconds']:.1f}s", flush=True)
        except Exception as exc:
            failed = True
            report["cases"].append({"case": case.name, "status": "failed", "error": str(exc)})
            print(f"❌ {case.name}: {type(exc).__name__}: {exc}", flush=True)
            import traceback
            traceback.print_exc()
        finally:
            if "prediction" in locals():
                del prediction
            torch.cuda.empty_cache()
            _io.dump_json(report_path, report)
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    _io.dump_json(report_path, report)
    print(f"🎉 批量结束: {len(report['cases'])}/{len(cases)} 有结果", flush=True)
    if failed or len(report["cases"]) != len(cases):
        sys.exit(1)


if __name__ == "__main__":
    main()
