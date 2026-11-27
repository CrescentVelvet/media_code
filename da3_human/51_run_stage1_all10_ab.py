#!/usr/bin/env python3
"""海思 stage1 全10图 RGB-only / 输入相机 DA3 双分支批量重建。

每例所有 RGB 均进入前向，不设留出、不筛选案例或视角。相机只进入 pose 分支；
input_world 是各自原生高斯事后对齐到输入世界的派生物，不是额外优化。
默认直接把最终高斯写到样例目录：depthanything3.ply / depthanything3_cameras.ply，
与 SplatAnything 相同，不再为安装复制一份 input_world.ply。原生PLY默认不另存。
同视角渲染仅供回放诊断，不支持泛化或新视角质量结论。不覆盖数据集输入。

Env: ROOT, DA3_MODEL_DIR, RESULTS_ROOT, PROCESS_RES=704, CASE_FILTER（逗号分隔可选）。
     BRANCH_FILTER=rgb,pose（默认）；仅诊断补跑才可用 rgb 或 pose 单分支。
     DIRECT_INSTALL=1（默认）/0；SAVE_NATIVE=0（默认）/1；SAVE_INPUT_WORLD=0（默认）/1。
     DIRECT_INSTALL=0时至少归档一种PLY；单分支诊断须另用输出根并明确BRANCH_FILTER。
输出根目录可含已经完整核验的案例以续跑；不完整案例拒绝静默覆盖。
CASE_FILTER 的汇总写入 batch_summary_subset.json，不覆盖全量 batch_summary.json。
"""
import importlib.util
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

SCRIPT = Path(__file__).resolve().parent


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def main():
    cam = load_module("hisilicon_cameras", "36_run_hisilicon_cameras.py")
    root = Path(os.environ["ROOT"])
    selected = {x for x in os.environ.get("CASE_FILTER", "").split(",") if x}
    branches = tuple(x for x in os.environ.get("BRANCH_FILTER", "rgb,pose").split(",") if x)
    if branches not in (("rgb", "pose"), ("rgb",), ("pose",)):
        raise ValueError(f"BRANCH_FILTER只允许rgb,pose或诊断单分支: {branches}")
    cases = sorted(root.glob("case*_nopose_recentered"),
                   key=lambda p: int(p.name.split("_")[0].removeprefix("case")))
    cases = [case for case in cases if not selected or case.name in selected]
    if not cases or (selected and {p.name for p in cases} != selected):
        raise FileNotFoundError(f"未找到完整指定案例: ROOT={root}, CASE_FILTER={sorted(selected)}")
    results = Path(os.environ["RESULTS_ROOT"])
    if results.is_symlink():
        raise ValueError(f"输出根目录不能是符号链接: {results}")
    direct_install = os.environ.get("DIRECT_INSTALL", "1") == "1"
    save_native = os.environ.get("SAVE_NATIVE", "0") == "1"
    save_input_world = os.environ.get("SAVE_INPUT_WORLD", "0") == "1"
    if direct_install and root.is_symlink():
        raise ValueError(f"直接安装的数据集根目录不能是符号链接: {root}")
    if direct_install and branches != ("rgb", "pose"):
        raise ValueError("单分支诊断必须DIRECT_INSTALL=0并使用独立RESULTS_ROOT；拒绝覆盖完整双分支")
    if not direct_install and not (save_native or save_input_world):
        raise ValueError("DIRECT_INSTALL=0须至少SAVE_NATIVE=1或SAVE_INPUT_WORLD=1")
    ply_names = {"rgb": "depthanything3.ply", "pose": "depthanything3_cameras.ply"}
    camera_dirs = {"rgb": "depthanything3", "pose": "depthanything3_cameras"}
    expected = {"native_cameras.npz", "input_cameras.npz", "common_source_cameras.npz"}
    expected.update(name for name, enabled in (("native.ply", save_native),
                                                ("input_world.ply", save_input_world)) if enabled)
    for case in cases:
        for branch in branches:
            for path in (case / ply_names[branch], case / camera_dirs[branch]):
                if path.is_symlink():
                    raise ValueError(f"拒绝直接安装到符号链接: {path}")
            if direct_install:
                for name in (ply_names[branch], f"{camera_dirs[branch]}/cameras.npz",
                             f"{camera_dirs[branch]}/summary.json"):
                    path = case / name
                    if path.exists() and not path.is_file():
                        raise FileExistsError(f"直接安装目标存在且不是普通文件: {path}")
                    partial = (case / ply_names[branch]).with_suffix(".partial.ply")
                    if partial.exists() or partial.is_symlink():
                        raise FileExistsError(f"样例目录有未完成PLY，需先核验: {partial}")
    if results.exists():
        if not results.is_dir():
            raise FileExistsError(f"输出根存在且不是普通目录: {results}")
    else:
        results.mkdir(parents=True)
    pending = []
    batch_path = results / "batch_summary.json"
    report = {"protocol": "all10_rgb_vs_input_pose", "root": str(root),
              "model": os.environ["DA3_MODEL_DIR"],
              "process_res": int(os.environ.get("PROCESS_RES", "704")), "seed": 42,
              "ref_view_strategy": "saddle_balanced", "holdout": False,
              "requested_branches": list(branches),
              "source_indices": list(range(10)), "target_indices": [],
              "export": {"n_gaussians_per_branch": 4900000, "n_properties": 41,
                         "prune": False, "sh_degree": 2, "display_rotation": False},
              "input_installation": direct_install,
              "save_native": save_native, "save_input_world": save_input_world,
              "final_ply_names": ply_names, "status": "running", "started_at": utc_now(),
              "limits": "十图均前向；input_world仅事后Sim(3)对齐。源回放不是留出泛化证据。",
              "cases": []}
    if batch_path.is_file():
        previous = json.loads(batch_path.read_text(encoding="utf-8"))
        if previous.get("protocol") != "all10_rgb_vs_input_pose" or previous.get("holdout") is not False:
            raise ValueError("已有批处理不是全10图双分支协议")
        previous.update(status="running", last_resume_at=utc_now(),
                        requested_branches=list(branches))
        if previous.get("scope") == "CASE_FILTER 子集运行；不覆盖全量批处理总表":
            previous = {k: v for k, v in previous.items()
                        if k not in ("status", "last_resume_at", "requested_branches")}
        report.update(previous)
    report["cases"] = []
    report["completed_cases"] = 0
    for case in cases:
        files, ext, K, camera_path, manifest = cam.load_stage1_inputs(case)
        if len(files) != 10:
            raise ValueError(f"{case.name}: 必须恰好有10个输入视角")
        out = results / case.name
        summary_path = out / "summary.json"
        if not out.exists():
            pending.append((case, out, files, ext, K, camera_path, manifest))
            continue
        if out.is_symlink() or not summary_path.is_file():
            raise FileExistsError(f"不完整或链接案例目录，拒绝覆盖: {out}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "completed":
            raise FileExistsError(f"失败/不完整案例需保留原目录并使用独立续跑根: {out}")
        if (summary.get("source_indices") != list(range(10))
                or summary.get("target_indices") != [] or summary.get("holdout") is not False
                or summary.get("status") != "completed"
                or bool(summary.get("input_installation", True)) != direct_install
                or bool(summary.get("save_native", True)) != save_native
                or bool(summary.get("save_input_world", True)) != save_input_world):
            raise ValueError(f"已有案例的全10图或存储配置不同，拒绝复用/覆盖: {out}")
        if summary.get("camera_sha256") != cam._io.sha256(camera_path):
            raise ValueError(f"已有案例相机输入身份变化: {out}")
        if summary.get("input_sha256") != [cam._io.sha256(p) for p in files]:
            raise ValueError(f"已有案例RGB输入身份变化: {out}")
        if set(summary.get("branches", {})) != set(branches) or tuple(summary.get(
                "requested_branches", ["rgb", "pose"])) != branches:
            raise ValueError(f"已有案例请求分支不完整或口径不同: {out}")
        for branch, info in summary["branches"].items():
            for name in expected:
                path = out / branch / name
                if path.is_symlink() or not path.is_file():
                    raise FileNotFoundError(f"已完成案例缺少文件: {path}")
            ply_key = "native_ply" if save_native else "input_world_ply"
            path = out / branch / ("native.ply" if save_native else "input_world.ply")
            if (save_native or save_input_world) and info.get(
                    ply_key, {}).get("sha256") != cam._io.sha256(path):
                raise ValueError(f"已有归档PLY哈希不一致: {out}/{branch}")
            if direct_install:
                for path, key in ((case / ply_names[branch], "input_world_ply"),
                                  (case / camera_dirs[branch] / "cameras.npz", "installed_cameras"),
                                  (case / camera_dirs[branch] / "summary.json", "installed_summary")):
                    if path.is_symlink() or not path.is_file():
                        raise FileNotFoundError(f"直接安装案例缺少配套文件: {path}")
                expected_sha = info.get("input_world_ply", {}).get("sha256")
                if expected_sha != cam._io.sha256(case / ply_names[branch]):
                    raise ValueError(f"样例目录已有不同PLY，不覆盖: {case}/{branch}")
                installed_summary = json.loads((case / camera_dirs[branch] / "summary.json").read_text(encoding="utf-8"))
                if (installed_summary.get("case") != case.name
                        or installed_summary.get("branch") != branch
                        or installed_summary.get("protocol") != summary.get("protocol")
                        or installed_summary.get("status") != "completed"
                        or installed_summary.get("sha256") != expected_sha):
                    raise ValueError(f"样例目录安装摘要身份不符: {case}/{branch}")
                with np.load(case / camera_dirs[branch] / "cameras.npz", allow_pickle=False) as cameras:
                    if [str(x) for x in cameras["image_names"]] != summary["image_names"]:
                        raise ValueError(f"样例目录相机图像顺序不符: {case}/{branch}")
        report["cases"] = [r for r in report["cases"] if Path(r.get("case", "")).name != case.name]
        report["cases"].append(summary)
        report["completed_cases"] = len(report["cases"])
        print(f"⏭️ {case.name}: 完整全10图双分支已核验，跳过", flush=True)

    def write_batch():
        if selected:
            scoped = dict(report)
            scoped.update(scope="CASE_FILTER 子集运行；不覆盖全量批处理总表",
                          selected_cases=sorted(selected), expected_cases=len(cases),
                          completed_cases=len(report["cases"]), status="completed_subset")
            cam._io.dump_json(results / "batch_summary_subset.json", scoped)
        else:
            cam._io.dump_json(batch_path, report)

    if not pending:
        if selected:
            report.update(finished_at=utc_now())
        else:
            report.update(status="completed" if len(report["cases"]) == len(cases) else "partial",
                          completed_cases=len(report["cases"]), finished_at=utc_now())
        write_batch()
        print(f"⏭️ {len(cases)}例均已核验，无需加载模型；{results}", flush=True)
        return
    if direct_install:
        for case, _, _, _, _, _, _ in pending:
            for branch in branches:
                paths = [case / ply_names[branch], case / camera_dirs[branch] / "cameras.npz",
                         case / camera_dirs[branch] / "summary.json"]
                if any(path.exists() or path.is_symlink() for path in paths):
                    raise FileExistsError(f"已有样例目录结果，不覆盖: {case}/{branch}")
    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.utils import gsply_helpers as gh
    from depth_anything_3.utils.pose_align import batch_align_poses_umeyama, align_poses_umeyama
    gh.export_ply = cam._io.fast_export_ply
    eps = torch.finfo(torch.float32).eps
    gh.inverse_sigmoid = lambda x: torch.logit(x.clamp(eps, 1 - eps))
    model_path = os.environ["DA3_MODEL_DIR"]
    if not (Path(model_path) / "model.safetensors").is_file():
        raise FileNotFoundError(f"DA3权重不存在: {model_path}")
    print(f"🏋️ 加载一次DA3；{len(pending)}例待跑，每例全10图；请求分支={branches}", flush=True)
    model = DepthAnything3.from_pretrained(model_path).to("cuda").eval()
    adapter = model.model.da3.gs_adapter
    original = adapter.forward
    captured = {}

    def capture(*args, **kwargs):
        E = kwargs["extrinsics"].detach().float().clone()
        scale = 1.0
        if kwargs.get("gt_extrinsics") is not None and not torch.equal(kwargs["extrinsics"], kwargs["gt_extrinsics"]):
            _, _, scales = batch_align_poses_umeyama(
                kwargs["gt_extrinsics"].detach().float(), E)
            scale = float(scales.clamp(1 / 3, 3)[0].cpu())
        captured.update(extrinsics=E.cpu().numpy()[0],
                        intrinsics=kwargs["intrinsics"].detach().float().cpu().numpy()[0],
                        adapter_scale=scale)
        return original(*args, **kwargs)

    def save_ply(pred, path):
        for key in ("means", "scales", "rotations", "harmonics", "opacities"):
            tensor = getattr(pred.gaussians, key).detach().float()
            if not torch.isfinite(tensor).all():
                raise RuntimeError(f"原生{key}含NaN/Inf；保持失败，不做静默修补")
            if key == "scales" and (tensor <= 0).any():
                bad = int((tensor <= 0).sum())
                raise RuntimeError(f"原生scales含{bad}个非正分量，log导出无定义；保持失败，不取绝对值/夹紧")
        partial = path.with_suffix(".partial.ply")
        if path.exists() or partial.exists() or path.is_symlink() or partial.is_symlink():
            raise FileExistsError(f"拒绝覆盖PLY: {path}")
        gh.save_gaussian_ply(
            gaussians=pred.gaussians, save_path=str(partial),
            ctx_depth=torch.from_numpy(pred.depth).unsqueeze(-1).to(pred.gaussians.means),
            shift_and_scale=False, save_sh_dc_only=False, gs_views_interval=1,
            inv_opacity=True, prune_by_depth_percent=1., prune_border_gs=False,
            match_3dgs_mcmc_dev=False)
        info = cam._io.inspect_ply(partial)
        if info["n_gaussians"] != 4900000 or info["n_properties"] != 41:
            raise RuntimeError(f"全10图原生SH2高斯数量/字段异常: {info}")
        info["path"] = str(path)
        info["sha256"] = cam._io.sha256(partial)
        os.replace(partial, path)
        if cam._io.sha256(path) != info["sha256"]:
            raise RuntimeError(f"PLY原子改名后哈希变化: {path}")
        return info

    def save_cameras(path, **arrays):
        temporary = path.with_name(f".{path.stem}.{os.getpid()}.installing.npz")
        if path.is_symlink() or path.exists() or temporary.exists() or temporary.is_symlink():
            raise FileExistsError(f"相机输出或临时文件已存在: {path}")
        try:
            np.savez_compressed(temporary, **arrays)
            with np.load(temporary, allow_pickle=False) as check:
                if any(not np.array_equal(check[key], value) for key, value in arrays.items()):
                    raise RuntimeError(f"相机临时文件回读不一致: {path}")
            os.replace(temporary, path)
        finally:
            if temporary.is_file() and not temporary.is_symlink():
                temporary.unlink()

    def ensure_direct_targets_free(case, branches):
        # 在耗时前向之前核验；已有同名但不属于本次摘要的结果绝不静默覆盖。
        for branch in branches:
            targets = [case / ply_names[branch], case / camera_dirs[branch] / "cameras.npz",
                       case / camera_dirs[branch] / "summary.json"]
            for path in targets:
                if path.exists() or path.is_symlink():
                    raise FileExistsError(f"已有样例目录结果，不覆盖: {path}")
            partial = (case / ply_names[branch]).with_suffix(".partial.ply")
            if partial.exists() or partial.is_symlink():
                raise FileExistsError(f"样例目录有未完成PLY，需先核验: {partial}")

    for case, out, files, ext, K, camera_path, manifest in pending:
        out.mkdir()
        indices = list(range(len(files)))
        summary = {"case": str(case), "protocol": "all10_rgb_vs_input_pose",
                   "model": model_path, "process_res": report["process_res"], "seed": 42,
                   "source_indices": indices, "target_indices": [], "holdout": False,
                   "n_input_views": len(files), "image_names": [p.name for p in files],
                   "input_sha256": [cam._io.sha256(p) for p in files],
                   "camera_json": str(camera_path), "camera_sha256": cam._io.sha256(camera_path),
                   "camera_conversion": "stage1 OpenGL c2w -> OpenCV w2c（轴翻转；不使用glTF中心取负）",
                   "requested_branches": list(branches),
                   "input_installation": direct_install, "save_native": save_native,
                   "save_input_world": save_input_world,
                   "status": "running", "started_at": utc_now(), "branches": {}}
        if branches != ("rgb", "pose"):
            summary["scope"] = "单分支诊断；completed只表示请求分支成功，不是完整A/B"
        cam._io.dump_json(out / "summary.json", summary)
        if direct_install:
            ensure_direct_targets_free(case, branches)
        for branch in branches:
            if summary["branches"].get(branch, {}).get("status") == "failed":
                break
            pred = None
            try:
                print(f"🚀 {case.name}/{branch}: 全10图前向", flush=True)
                t0 = time.time()
                dest = out / branch
                dest.mkdir()
                torch.manual_seed(42)
                np.random.seed(42)
                torch.cuda.reset_peak_memory_stats()
                captured.clear()
                adapter.forward = capture
                try:
                    pred = model.inference(
                        [str(p) for p in files], extrinsics=ext if branch == "pose" else None,
                        intrinsics=K if branch == "pose" else None,
                        process_res=report["process_res"], infer_gs=True,
                        ref_view_strategy="saddle_balanced", export_dir=None, export_format="")
                finally:
                    adapter.forward = original
                E = cam.adapter_camera_extrinsics(captured["extrinsics"], captured["adapter_scale"])
                save_cameras(
                    dest / "native_cameras.npz", extrinsics=E[:, :3],
                    intrinsics=captured["intrinsics"], image_shape=np.asarray(pred.depth.shape[-2:]),
                    image_names=np.asarray([p.name for p in files]),
                    source_indices=np.arange(len(files)),
                    camera_frame=np.asarray("adapter原生预测世界；w2c平移含adapter尺度"))
                native_info = save_ply(pred, dest / "native.ply") if save_native else None
                pred.gs_extrinsics = E
                pred.adapter_scale = captured["adapter_scale"]
                alignment = cam.transform_gltf_gaussians(pred, ext)
                world_path = (case / ply_names[branch] if direct_install
                              else dest / "input_world.ply")
                world_info = save_ply(pred, world_path)
                save_cameras(
                    dest / "input_cameras.npz", extrinsics=ext[:, :3], intrinsics=K,
                    image_shape=np.asarray([704, 704]), image_names=np.asarray([p.name for p in files]),
                    source_indices=np.arange(len(files)), world_rotation=np.eye(3),
                    camera_frame=np.asarray("stage1输入OpenCV世界；无显示旋转；PLY为各自原生结果事后对齐"))
                # 仅诊断回放：同一物理输入相机映射到该分支原生世界。
                R, t, scale = align_poses_umeyama(ext, E, ransac=False)
                c2w = np.linalg.inv(ext).copy()
                c2w[:, :3, :3] = R.T @ c2w[:, :3, :3]
                c2w[:, :3, 3] = (c2w[:, :3, 3] - t) @ R / scale
                save_cameras(
                    dest / "common_source_cameras.npz", extrinsics=np.linalg.inv(c2w)[:, :3],
                    intrinsics=K, image_shape=np.asarray([704, 704]),
                    image_names=np.asarray([p.name for p in files]), source_indices=np.arange(len(files)),
                    camera_frame=np.asarray("stage1相机经同一Sim(3)映射到adapter原生世界；全10图均已前向"))
                if direct_install:
                    camera_dir = case / camera_dirs[branch]
                    camera_dir.mkdir(exist_ok=True)
                    installed_summary = camera_dir / "summary.json"
                    if installed_summary.exists() or installed_summary.is_symlink():
                        raise FileExistsError(f"配套摘要已存在，拒绝覆盖: {installed_summary}")
                    save_cameras(camera_dir / "cameras.npz", extrinsics=ext[:, :3], intrinsics=K,
                                 image_shape=np.asarray([704, 704]),
                                 image_names=np.asarray([p.name for p in files]),
                                 source_indices=np.arange(len(files)), world_rotation=np.eye(3),
                                 camera_frame=np.asarray("stage1输入OpenCV世界；与样例目录PLY同坐标，无显示旋转"))
                    install_info = {"case": case.name, "branch": branch,
                                    "protocol": "all10_rgb_vs_input_pose", "status": "completed",
                                    "path": str(world_path), "sha256": world_info["sha256"],
                                    "n_views": len(files), "image_names": [p.name for p in files],
                                    "input_sha256": summary["input_sha256"],
                                    "camera_json": str(camera_path),
                                    "camera_sha256": summary["camera_sha256"],
                                    "experimental_summary": str(out / "summary.json"),
                                    "scope": "最终input_world直接原子写入样例目录；不额外复制；不表示画质验收通过"}
                    cam._io.dump_json(camera_dir / "summary.json", install_info)
                    world_info["installed_cameras"] = str(camera_dir / "cameras.npz")
                    world_info["installed_summary"] = str(camera_dir / "summary.json")
                info = {"n_views": len(files), "seconds": round(time.time() - t0, 2),
                        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024 ** 3,
                        "depth_shape": list(pred.depth.shape),
                        "adapter_scale": captured["adapter_scale"],
                        "input_world_ply": world_info,
                        "input_alignment": alignment,
                        "diagnostic_source_alignment": {"R": R.tolist(), "t": t.tolist(), "scale": float(scale)},
                        "limits": "input_world为事后Sim(3)派生；无显示旋转；十图均参与前向"}
                if native_info is not None:
                    info["native_ply"] = native_info
                summary["branches"][branch] = info
                cam._io.dump_json(out / "summary.json", summary)
                print(f"✅ {case.name}/{branch}: {world_info['n_gaussians']:,}高斯, "
                      f"最终PLY={world_info['path']}, {world_info['bytes']:,}B, "
                      f"{info['seconds']:.1f}s", flush=True)
            except Exception as exc:
                summary.update(status="failed", failed_branch=branch, error=str(exc), failed_at=utc_now())
                summary["branches"][branch] = {"status": "failed", "error": str(exc)}
                report.setdefault("failed_cases", []).append({"case": str(case), "branch": branch,
                                                               "error": str(exc), "time": utc_now()})
                cam._io.dump_json(out / "summary.json", summary)
                write_batch()
                print(f"❌ {case.name}/{branch}: {type(exc).__name__}: {exc}；保留现场并继续", flush=True)
                break
            finally:
                del pred
                torch.cuda.empty_cache()
        if summary["status"] == "failed":
            continue
        summary.update(status="completed", finished_at=utc_now())
        cam._io.dump_json(out / "summary.json", summary)
        report["cases"] = [r for r in report["cases"] if Path(r.get("case", "")).name != case.name]
        report["cases"].append(summary)
        report["completed_cases"] = len(report["cases"])
        write_batch()
    report.update(status="completed" if len(report["cases"]) == len(cases) else "partial",
                  completed_cases=len(report["cases"]), finished_at=utc_now())
    write_batch()
    print(f"🎉 全10图批处理结束: {len(report['cases'])}/{len(cases)}例；"
          f"分支={branches}；{results}", flush=True)
    if len(report["cases"]) != len(cases):
        raise RuntimeError("批处理未覆盖全部指定案例")


if __name__ == "__main__":
    main()
