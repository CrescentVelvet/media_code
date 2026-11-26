#!/usr/bin/env python3
"""海思十视角：输入 stage1 相机运行原生 DA3，导出 depthanything3_cameras.ply。

JSON 的 OpenGL c2w 转 OpenCV w2c；DA3 的高斯仍在自估世界，必须用与输出
相机相同的 Umeyama Sim(3) 把 means/scales/rotations/SH 对齐到输入世界。
最终再与相机同步施加绕 X 轴 180° 的显示旋转，直接写 depthanything3_cameras.ply。
米制倍率与相机对齐尺度只组合一次。保留所有高斯和完整 SH，不读取竞品几何。

Env vars:
  ROOT, CASE_FILTER, FORCE, PROCESS_RES, DA3_MODEL_DIR
"""
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from importlib.util import module_from_spec, spec_from_file_location  # noqa: E402

_spec = spec_from_file_location("hisilicon_rgb", SCRIPT_DIR / "34_run_hisilicon.py")
_io = module_from_spec(_spec)
_spec.loader.exec_module(_io)

FLIP_CV = np.diag([1.0, -1.0, -1.0, 1.0])
WORLD_ROTATION = np.diag([1.0, -1.0, -1.0])


def load_inputs(case):
    """按 sourceIndex 一一配对 RGB 和原始 JSON 相机，拒绝猜测文件顺序。"""
    images_dir = case / "splatanything/images"
    manifest_path = case / "splatanything/extraction.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    camera_path = case / manifest["camera_file"]
    if camera_path.parent != case or not camera_path.is_file():
        raise RuntimeError(f"原始相机不存在或不在案例目录: {camera_path}")
    data = json.loads(camera_path.read_text(encoding="utf-8"))
    views, frames = data["views"], manifest["frames"]
    if not (len(views) == len(frames) == data["viewCount"] == 10):
        raise RuntimeError(f"{case.name}: 相机与图像不是十视角")
    if [v["sourceIndex"] for v in views] != [r["sourceIndex"] for r in frames]:
        raise RuntimeError(f"{case.name}: sourceIndex 顺序不一致")
    if [v["sourceIndex"] for v in views] != data["sourceIndices"]:
        raise RuntimeError(f"{case.name}: sourceIndices 清单不一致")
    if manifest["input_camera_convention"] != "OpenGL camera-to-world (-Z forward, +Y up)":
        raise RuntimeError(f"{case.name}: 输入相机坐标约定未登记")
    files, w2cs, intrinsics = [], [], []
    for view, frame in zip(views, frames):
        path = images_dir / Path(frame["image"]).name
        if _io.sha256(path) != frame["sha256"]:
            raise RuntimeError(f"输入图与抽帧清单不一致: {path}")
        from PIL import Image
        with Image.open(path) as im:
            if im.size != (data["imageWidth"], data["imageHeight"]):
                raise RuntimeError(f"图像尺寸与相机 JSON 不一致: {path}")
        R = np.asarray(view["rotation3x3"], dtype=np.float64).reshape(3, 3)
        if (np.abs(R @ R.T - np.eye(3)).max() > 1e-4
                or abs(np.linalg.det(R) - 1) > 1e-4):
            raise RuntimeError(f"旋转矩阵不是有效 SO(3): {camera_path}")
        c2w_gl = np.eye(4, dtype=np.float64)
        c2w_gl[:3, :3] = R
        c2w_gl[:3, 3] = view["position"]
        c2w_cv = c2w_gl @ FLIP_CV
        K = np.array([[view["fx"], 0, view["cx"]],
                      [0, view["fy"], view["cy"]], [0, 0, 1]], dtype=np.float64)
        if not (0 < K[0, 0] and 0 < K[1, 1]
                and 0 <= K[0, 2] < data["imageWidth"]
                and 0 <= K[1, 2] < data["imageHeight"]):
            raise RuntimeError(f"内参异常: {camera_path}")
        files.append(path)
        w2cs.append(np.linalg.inv(c2w_cv))
        intrinsics.append(K)
    return files, np.asarray(w2cs), np.asarray(intrinsics), camera_path, manifest


def inference_preserving_raw_cameras(model, files, ext, ixt, process_res):
    """复刻官方 inference 至对齐阶段，保留覆盖前的自估外参。"""
    imgs_cpu, ext_t, ixt_t = model._preprocess_inputs(
        [str(p) for p in files], extrinsics=ext, intrinsics=ixt,
        process_res=process_res, process_res_method="upper_bound_resize")
    ext_cpu, ixt_cpu = ext_t, ixt_t
    imgs, ext_t, ixt_t = model._prepare_model_inputs(imgs_cpu, ext_t, ixt_t)
    ext_norm = model._normalize_extrinsics(ext_t.clone())
    raw_output = model._run_model_forward(
        imgs, ext_norm, ixt_t, [], True, False, "saddle_balanced")
    prediction = model._convert_to_prediction(raw_output)
    raw_extrinsics = np.asarray(prediction.extrinsics).copy()
    prediction = model._align_to_input_extrinsics_intrinsics(
        ext_cpu, ixt_cpu, prediction, True)
    prediction = model._add_processed_images(prediction, imgs_cpu)
    return prediction, raw_extrinsics


def transform_gaussians(prediction, raw_extrinsics, input_extrinsics, world_rotation=None):
    """用官方同一次 Umeyama 参数把高斯从自估世界送到输入世界。

    官方 API 只覆盖 prediction.extrinsics/depth，未变换 gaussians。这里不能
    改喂 self.extrinsics：depth 仍定义在自估相机局部坐标，直接换相机会错。
    官方 gs_adapter 的 gt_extrinsics 分支又只缩放了 c2w 平移/深度；本路径以
    捕获的预测相机 + 独立 Sim(3) 后处理覆盖 R、t、s，不直接采用该分支结果。
    world_rotation 仅用于最终显示转正；None 保持海思历史口径，单位阵保留
    SFM 原生世界坐标。坐标/高斯/SH 必须使用同一个旋转。
    """
    from depth_anything_3.model.utils.transform import mat_to_quat, quat_to_mat
    from depth_anything_3.utils.pose_align import align_poses_umeyama
    from depth_anything_3.utils.sh_helpers import rotate_sh

    raw_ext = np.asarray(raw_extrinsics, dtype=np.float64)
    if raw_ext.shape[1:] == (3, 4):
        raw_ext44 = np.repeat(np.eye(4, dtype=np.float64)[None], len(raw_ext), axis=0)
        raw_ext44[:, :3] = raw_ext
    else:
        raw_ext44 = raw_ext.copy()
    # 当前 DA3 align_poses_umeyama(pred, input) 内部把两路输入反过来传给 evo，
    # 所以返回 input -> pred，而非名字暗示的 pred -> input；这里显式求逆。
    R_in_to_pred, t_in_to_pred, s_in_to_pred = align_poses_umeyama(
        raw_ext44, input_extrinsics, ransac=len(raw_ext44) >= 10, random_state=42)
    if not (np.isfinite(R_in_to_pred).all() and np.isfinite(t_in_to_pred).all()
            and np.isfinite(s_in_to_pred) and s_in_to_pred > 0):
        # 已输入的相机仍应保留在输出，不能因数值对齐失败写一帧/一批坏几何。
        raise RuntimeError("相机 Umeyama 返回非有限参数或非正尺度；拒绝导出，不静默退化为错位结果")
    if (np.abs(R_in_to_pred @ R_in_to_pred.T - np.eye(3)).max() > 1e-5
            or abs(np.linalg.det(R_in_to_pred) - 1) > 1e-5):
        raise RuntimeError("相机 Umeyama 旋转不是 SO(3)")
    R = R_in_to_pred.T
    pose_scale = 1.0 / float(s_in_to_pred)
    t = -(R @ t_in_to_pred) * pose_scale
    if not np.isfinite(R).all() or not np.isfinite(t).all() or not np.isfinite(pose_scale) or pose_scale <= 0:
        raise RuntimeError(f"非法相机对齐: pose_scale={pose_scale}")
    if abs(np.linalg.det(R) - 1) > 1e-5:
        raise RuntimeError("Umeyama 旋转不是 SO(3)")
    sf, _ = _io.gs_scale_factor(prediction)
    if sf is None or not np.isfinite(sf) or sf <= 0:
        raise RuntimeError(f"模型未提供有效米制倍率: {sf}")
    total_scale = float(sf * pose_scale)
    g = prediction.gaussians
    if g is None:
        raise RuntimeError("原生权重未返回 gaussians")
    device = g.means.device
    R_t = torch.as_tensor(R, dtype=torch.float32, device=device)
    t_t = torch.as_tensor(t, dtype=torch.float32, device=device)
    # 对角F与Umeyama旋转交换会引入符号变化，不能合写成F@R再乘t。
    # 单独施加F可保证与已验收的“Sim(3)后180度转正”逐点一致。
    world_rotation = WORLD_ROTATION if world_rotation is None else np.asarray(
        world_rotation, dtype=np.float64)
    if (world_rotation.shape != (3, 3)
            or np.abs(world_rotation @ world_rotation.T - np.eye(3)).max() > 1e-6
            or abs(np.linalg.det(world_rotation) - 1) > 1e-6):
        raise RuntimeError("显示世界旋转不是 SO(3)")
    F_t = torch.as_tensor(world_rotation, dtype=torch.float32, device=device)
    g.means = (g.means.float() @ R_t.T) * total_scale + t_t
    g.scales = g.scales * total_scale
    g.means = g.means @ F_t.T
    rot = F_t @ (R_t @ quat_to_mat(g.rotations.float()))
    g.rotations = torch.nn.functional.normalize(mat_to_quat(rot), dim=-1)
    # sh_helpers 的 einsum 需要与 SH 相同的点数维，不能用单矩阵广播。
    R_sh = F_t @ R_t
    rotations = R_sh.reshape(1, 1, 3, 3).expand(len(g.harmonics), 3, 3, 3)
    g.harmonics = rotate_sh(g.harmonics, rotations)

    c2w_pred = np.linalg.inv(raw_ext44)
    c2w_aligned = c2w_pred.copy()
    c2w_aligned[:, :3, :3] = world_rotation @ (R @ c2w_pred[:, :3, :3])
    c2w_aligned[:, :3, 3] = (pose_scale * (c2w_pred[:, :3, 3] @ R.T) + t) @ world_rotation.T
    c2w_output = np.linalg.inv(input_extrinsics)
    c2w_output[:, :3, :3] = world_rotation @ c2w_output[:, :3, :3]
    c2w_output[:, :3, 3] = c2w_output[:, :3, 3] @ world_rotation.T
    camera_error = np.abs(c2w_aligned - c2w_output).max()
    position_error = np.linalg.norm(c2w_aligned[:, :3, 3] - c2w_output[:, :3, 3], axis=-1)
    if not np.isfinite(camera_error) or not np.isfinite(position_error).all():
        raise RuntimeError("相机 Sim(3) 往返出现非有限值")
    return {"metric_scale_factor": float(sf), "pose_scale": float(pose_scale),
            "total_gaussian_scale": total_scale, "umeyama_rotation": R.tolist(),
            "umeyama_translation": t.tolist(), "gaussian_rotation": (world_rotation @ R).tolist(),
            "gaussian_translation": (world_rotation @ t).tolist(), "aligned_camera_max_abs_error": float(camera_error),
            "aligned_camera_max_position_error": float(position_error.max()),
            "aligned_camera_mean_position_error": float(position_error.mean()),
            "raw_camera_centers": c2w_pred[:, :3, 3].tolist(),
            "aligned_camera_centers": c2w_aligned[:, :3, 3].tolist(),
            "output_camera_centers": c2w_output[:, :3, 3].tolist(),
            "world_rotation": world_rotation.tolist(),
            "output_extrinsics": np.linalg.inv(c2w_output)[:, :3].tolist()}


def main():
    root = Path(os.environ.get("ROOT", "/mnt/d/dataset/测试数据/海思端侧重建拍摄数据"))
    selected = os.environ.get("CASE_FILTER", "")
    force = os.environ.get("FORCE", "0") == "1"
    process_res = int(os.environ.get("PROCESS_RES", "504"))
    model_dir = os.environ.get("DA3_MODEL_DIR", "") or os.environ.get("DA3_HUMAN_MODEL_DIR", "")
    if not root.is_dir() or not (Path(model_dir) / "model.safetensors").is_file():
        sys.exit(f"❌ ROOT/权重无效: ROOT={root}, DA3_MODEL_DIR={model_dir}")
    cases = sorted(root.glob("case*_nopose_recentered"),
                   key=lambda p: int(p.name.split("_")[0][4:]))
    if selected:
        cases = [p for p in cases if p.name == selected]
    if not cases:
        sys.exit("❌ 没有匹配案例")

    pending = []
    for case in cases:
        files, ext, ixt, camera_path, manifest = load_inputs(case)
        target = case / "depthanything3_cameras.ply"
        out = case / "depthanything3_cameras"
        if (target.is_file() and (out / "cameras.npz").is_file()
                and (out / "summary.json").is_file() and not force):
            previous = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            if not np.array_equal(previous.get("world_rotation"), WORLD_ROTATION):
                raise RuntimeError(f"旧相机版未登记180度转正，不能直接跳过: {case.name}；请用 FORCE=1 重建")
            previous.update(_io.inspect_ply(target))
            pending.append((case, files, ext, ixt, camera_path, manifest, target, out,
                            previous))
        elif target.exists() and not target.is_file():
            sys.exit(f"❌ 目标不是普通文件，拒绝覆盖: {target}")
        else:
            pending.append((case, files, ext, ixt, camera_path, manifest, target, out, None))

    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.utils import gsply_helpers as gh
    gh.export_ply = _io.fast_export_ply

    def stable_inverse_sigmoid(x):
        # 官方 log(x/(1-x)) 在 opacity=1 时产生 Inf；饱和点不删、不改几何。
        eps = torch.finfo(x.dtype).eps
        logit = torch.where(x >= 1,
                            torch.full_like(x, -torch.log(torch.tensor(
                                eps, dtype=x.dtype, device=x.device))),
                            torch.log(x) - torch.log1p(-x))
        if not torch.isfinite(logit).all():
            raise RuntimeError("DA3 opacity 含非有限 logit")
        return logit

    gh.inverse_sigmoid = stable_inverse_sigmoid
    print(f"🏋️ 加载原生 DA3: {model_dir}", flush=True)
    model = DepthAnything3.from_pretrained(model_dir).to("cuda")
    report = {"root": str(root), "model": model_dir, "process_res": process_res,
              "camera_input": "各案例 stage1_cameras.json；OpenGL c2w -> OpenCV w2c",
              "gaussian_frame": "官方 Sim(3) 对齐后与相机同步绕X轴180度显示旋转；直接写相机版PLY",
              "export": {"prune_border_gs": False, "prune_by_depth_percent": 1.0,
                         "save_sh_dc_only": False, "scale": "input_world",
                         "world_rotation": WORLD_ROTATION.tolist()},
              "started_at": datetime.now(timezone.utc).isoformat(), "cases": []}
    failed = False
    for case, files, ext, ixt, camera_path, manifest, target, out, existing in pending:
        out.mkdir(parents=True, exist_ok=True)
        if existing is not None:
            info = dict(existing)
            info.update({"case": case.name, "status": "existing"})
            report["cases"].append(info)
            print(f"⏭️ {case.name}: 已有有效 PLY，跳过", flush=True)
            _io.dump_json(out / "summary.json", info)
            _io.dump_json(root / "depthanything3_cameras_batch_summary.json", report)
            continue
        backup = None
        if target.exists():
            backup = target.with_name(f"depthanything3_cameras.backup_{time.strftime('%Y%m%d_%H%M%S')}.ply")
            target.rename(backup)
            print(f"📦 已有结果备份: {backup}", flush=True)
        try:
            print(f"🚀 {case.name}: 十视角输入相机前向", flush=True)
            t0 = time.time()
            torch.cuda.reset_peak_memory_stats()
            prediction, raw_extrinsics = inference_preserving_raw_cameras(
                model, files, ext, ixt, process_res)
            np.savez_compressed(
                out / "raw_cameras.npz", raw_extrinsics=raw_extrinsics)
            alignment = transform_gaussians(prediction, raw_extrinsics, ext)
            np.savez_compressed(
                out / "cameras.npz", extrinsics=alignment["output_extrinsics"],
                intrinsics=prediction.intrinsics,
                input_extrinsics_original=ext, input_intrinsics_original=ixt,
                depth=prediction.depth, conf=prediction.conf,
                scale_factor=np.float32(alignment["total_gaussian_scale"]),
                is_metric=np.int32(1), view_indices=np.arange(len(files)),
                image_names=np.asarray([p.name for p in files]),
                world_rotation=WORLD_ROTATION,
                source_indices=np.asarray([v["sourceIndex"] for v in
                                           json.loads(camera_path.read_text(encoding="utf-8"))["views"]]))
            ctx_depth = torch.from_numpy(prediction.depth).unsqueeze(-1).to(
                prediction.gaussians.means)
            gh.save_gaussian_ply(
                gaussians=prediction.gaussians, save_path=str(target), ctx_depth=ctx_depth,
                shift_and_scale=False, save_sh_dc_only=False, gs_views_interval=1,
                inv_opacity=True, prune_by_depth_percent=1.0,
                prune_border_gs=False, match_3dgs_mcmc_dev=False)
            info = _io.inspect_ply(target)
            info.update({"case": case.name, "status": "completed", "seconds": round(time.time() - t0, 1),
                         "n_views": len(files), "image_names": [p.name for p in files],
                         "input_sha256": [_io.sha256(p) for p in files],
                         "camera_json": camera_path.name, "camera_json_sha256": _io.sha256(camera_path),
                         "camera_coordinate": "OpenCV w2c；相机与高斯已同步绕输入世界X轴180度",
                         "depth_median_m": float(np.median(prediction.depth)),
                         "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
                         "backup": str(backup) if backup else None, **alignment})
            _io.dump_json(out / "summary.json", info)
            report["cases"].append(info)
            print(f"✅ {case.name}: {info['n_gaussians']:,} 高斯, "
                  f"{info['bytes'] / 1024 ** 2:.1f} MiB, sf={info['metric_scale_factor']:.4f}, "
                  f"pose_scale={info['pose_scale']:.4f}, {info['seconds']:.1f}s", flush=True)
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
            _io.dump_json(root / "depthanything3_cameras_batch_summary.json", report)
    print(f"🎉 批量结束: {len(report['cases'])}/{len(cases)} 有结果", flush=True)
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    _io.dump_json(root / "depthanything3_cameras_batch_summary.json", report)
    if failed or len(report["cases"]) != len(cases):
        sys.exit(1)


if __name__ == "__main__":
    main()
