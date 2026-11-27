#!/usr/bin/env python3
"""串行渲染摘要指定的原生 PLY 与共同物理相机，不做二次前向。

Env vars: RESULTS_ROOT（单案例或 per-case 根）, DATASET_ROOT, GSPLAT_CACHED_SO,
          CASE_FILTER（可选，逗号分隔案例名）, BRANCH_FILTER, RENDER_SIZE=504（保留已有渲染）
支持固定8源2留出，或 protocol=all10_rgb_vs_input_pose 的全10图源视角回放。
全10图渲染要求显式 SAVE_NATIVE=1 或 SAVE_INPUT_WORLD=1 的归档高斯；
SAVE_INPUT_WORLD=1时原生诊断相机必须配套保存，不能把input_world当原生PLY。
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SCRIPT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_case(summary_path, dataset_root):
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    case = Path(summary["case"])
    if not case.is_dir():
        candidate = dataset_root / case.name
        if not candidate.is_dir():
            raise FileNotFoundError(f"摘要是 {case}，数据集候选 {candidate} 不存在；拒绝猜测输入")
        case = candidate
    expected = dataset_root / case.name
    if case.resolve() != expected.resolve():
        raise ValueError(f"案例 {case} 不属于指定数据集 {dataset_root}")
    is_all10 = summary.get("protocol") == "all10_rgb_vs_input_pose" and summary.get("holdout") is False
    if is_all10 and summary.get("requested_branches") == ["pose"]:
        # 单分支诊断允许RGB缺失，但不能把case级completed误当成完整A/B。
        if summary.get("scope") != "单分支诊断；completed只表示请求分支成功，不是完整A/B":
            raise ValueError(f"{summary_path} 缺少明确的单分支范围声明")
        if set(summary.get("branches", {})) != {"pose"}:
            raise ValueError(f"{summary_path} 位姿单分支诊断缺少pose或混有其他分支")
    elif len(summary["branches"]) != 2:
        raise ValueError(f"{summary_path} 需恰好含 RGB 与一个相机分支")
    if not is_all10 and not summary.get("holdout", summary.get("target_indices")):
        raise ValueError(f"{summary_path} 不是明确的留出前向或全10图摘要")
    indices = summary.get("source_indices")
    if is_all10:
        if indices != list(range(10)) or summary.get("target_indices") != []:
            raise ValueError(f"{summary_path} 不符合全10图协议")
    elif indices != [0, 1, 3, 4, 5, 6, 8, 9] or summary.get("target_indices") != [2, 7]:
        raise ValueError(f"{summary_path} 不符合固定8源2留出协议")
    if summary.get("process_res") != 704 or summary.get("seed") != 42:
        raise ValueError(f"{summary_path} 处理分辨率或seed不符")
    return summary_path.parent, summary, case


def verify_input(summary, case, images_dir, names, cameras):
    manifest = summary.get("loader_manifest")
    if manifest is None:
        frames = summary_path_frames(case)
    else:
        frames = [str(item["image"]).split("/", 1)[-1] for item in manifest["frames"]]
    source_names = [frames[i] for i in summary["source_indices"]]
    for index, name in enumerate(source_names):
        if sha256(images_dir / name) != summary["input_sha256"][index]:
            raise ValueError(f"输入hash不匹配: {case.name}/{name}")
    camera_path = case / f"{case.name.removesuffix('_nopose_recentered')}_stage1_cameras.json"
    if manifest is not None:
        camera_path = case / manifest["camera_file"]
    if not camera_path.is_file():
        candidates = list(case.glob("*_stage1_cameras.json"))
        if len(candidates) != 1:
            raise FileNotFoundError(f"无法唯一确定相机: {case}")
        camera_path = candidates[0]
    if sha256(camera_path) != summary.get("camera_sha256", summary.get("camera_json_sha256")):
        raise ValueError(f"相机hash不匹配: {camera_path}")
    for key in ("extrinsics", "intrinsics"):
        if key not in cameras or not np.isfinite(cameras[key]).all():
            raise ValueError(f"相机数组缺失或非有限: {key}")
    if "image_names" not in cameras or not list(map(str, names)):
        raise ValueError("相机缺少图像名")
    if len(names) != len(set(map(str, names))) or len(names) != cameras["extrinsics"].shape[0]:
        raise ValueError("相机图像名重复或与外参数量不一致")
    if len(set(frames)) != len(frames):
        raise ValueError("加载清单含重复图像名")
    if not set(map(str, names)).issubset(frames):
        raise ValueError(f"相机图像不属于加载清单: {case.name}/{list(map(str, names))}")


def summary_path_frames(case):
    spec = importlib.util.spec_from_file_location("camera_inputs", SCRIPT / "36_run_hisilicon_cameras.py")
    camera_inputs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(camera_inputs)
    files, _, _, _, _ = camera_inputs.load_stage1_inputs(case)
    return [p.name for p in files]


def validate_existing_render(out, cameras, images_dir, role, native, branch, size):
    """旧渲染也必须由磁盘图像复核；身份与量化PSNR复核另存，不改原始指标。"""
    metrics_path = out / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if len(metrics["views"]) != len(cameras["image_names"]):
        raise ValueError(f"渲染视角数与相机不一致: {out}")
    target_hashes = {}
    errors = []
    max_bound_excess = 0.0
    for index, view in enumerate(metrics["views"]):
        name = str(cameras["image_names"][index])
        if view["image_name"] != name:
            raise ValueError(f"渲染图名错位: {out}/{name}")
        # 与渲染器一致，采用PIL默认bicubic；Pillow版本导致的小量化差单列，不解释成模型变化。
        source = Image.open(images_dir / name).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
        target_hashes[name] = sha256(images_dir / name)
        saved_gt = out / f"{index:02d}_gt.png"
        if saved_gt.is_file():
            if not np.array_equal(np.asarray(Image.open(saved_gt).convert("RGB")), np.asarray(source)):
                raise ValueError(f"保存GT与当前RGB不一致: {saved_gt}")
        rendered = np.asarray(Image.open(out / f"{index:02d}.png").convert("RGB"), dtype=np.float64) / 255.
        if rendered.shape != (size, size, 3):
            raise ValueError(f"渲染分辨率不符: {out}/{index:02d}.png")
        psnr = float(-10 * np.log10(np.mean((rendered - np.asarray(source) / 255.) ** 2).clip(1e-12)))
        # 原始PSNR在float32渲染上计算，保存PNG时两图均截断到8bit。逐通道误差差最多
        # 1/255，因此可由原始MSE给出严格的量化后MSE上界，不能用固定dB阈值误拒真图。
        metric_mse = 10 ** (-float(view["psnr_db"]) / 10)
        quantized_mse = 10 ** (-psnr / 10)
        quantization_bound = (np.sqrt(metric_mse) + 1 / 255.) ** 2 + 1e-12
        if quantized_mse > quantization_bound:
            raise ValueError(f"磁盘渲染超出8bit量化误差上界: {out}/{index:02d}.png: "
                             f"MSE {quantized_mse} > {quantization_bound}")
        errors.append(abs(psnr - float(view["psnr_db"])))
        max_bound_excess = max(max_bound_excess, quantized_mse - quantization_bound)
    validation = {"role": role, "branch": branch, "ply": str(native), "ply_sha256": sha256(native),
                  "input_sha256": target_hashes, "metrics_sha256": sha256(metrics_path),
                  "max_png_quantization_psnr_delta": max(errors, default=0.),
                  "max_png_mse_above_quantization_bound": max_bound_excess,
                  "limits": "磁盘PNG按原始MSE及8bit误差上界复核；未重新计算alpha，原始指标不改写。"}
    with open(out / "render_validation.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(validation, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return target_hashes


def main():
    cached = os.environ["GSPLAT_CACHED_SO"]
    spec = importlib.util.spec_from_file_location("gsplat_cuda", cached)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import sys
    sys.modules["gsplat_cuda"] = mod
    sys.modules["gsplat.csrc"] = mod
    import gsplat
    gsplat.csrc = mod
    from gsplat import rasterization
    spec = importlib.util.spec_from_file_location("render", SCRIPT / "35_render_hisilicon.py")
    render = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(render)
    out_root = Path(os.environ["RESULTS_ROOT"])
    root = Path(os.environ["DATASET_ROOT"])
    selected = {x for x in os.environ.get("CASE_FILTER", "").split(",") if x}
    summaries = [out_root / "summary.json"] if (out_root / "summary.json").is_file() else sorted(out_root.glob("*/summary.json"))
    if selected:
        cases = [locate_case(p, root) for p in summaries if Path(json.loads(p.read_text(encoding="utf-8"))["case"]).name in selected]
    else:
        cases = [locate_case(p, root) for p in summaries]
    if not cases:
        raise FileNotFoundError(f"没有匹配的完整对照摘要: {out_root}")
    size = int(os.environ.get("RENDER_SIZE", "504"))
    report_path = out_root / "render_summary.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else []
    for case_out, summary, case in cases:
        name = case.name
        images_dir = case / "images" if (case / "images").is_dir() else case / "splatanything/images"
        if not images_dir.is_dir():
            raise FileNotFoundError(images_dir)
        for branch in ("rgb", "negative_centers", "pose"):
            if branch not in summary["branches"]:
                continue
            if os.environ.get("BRANCH_FILTER") and branch != os.environ["BRANCH_FILTER"]:
                continue
            branch_info = summary["branches"][branch]
            native_info = branch_info.get("native_ply", branch_info)
            native = Path(native_info["path"])
            is_all10 = summary.get("protocol") == "all10_rgb_vs_input_pose"
            world_archive = is_all10 and not summary.get("save_native", True)
            if is_all10 and not summary.get("save_native", True) and not summary.get("save_input_world", False):
                raise ValueError(f"直接安装默认不复制PLY到实验目录；仅做源回放须显式SAVE_NATIVE=1")
            if world_archive:
                if not summary.get("save_input_world"):
                    raise ValueError(f"实验目录没有可回放PLY: {case_out}/{branch}")
                native_info = branch_info["input_world_ply"]
                native = Path(native_info["path"])
            if (native.name == "native.partial.ply" and is_all10):
                # 原子导出摘要记录的是校验时的临时名，完成后才改名；只允许这一精确路径。
                native = native.with_name("native.ply")
            if native.parent.resolve() != (case_out / branch).resolve() or native.name != (
                    "input_world.ply" if world_archive else "native.ply") or not native.is_file():
                raise ValueError(f"PLY不归属摘要目录: {native}")
            if sha256(native) != native_info["sha256"]:
                raise ValueError(f"PLY hash与摘要不同: {native}")
            is_all10 = summary.get("protocol") == "all10_rgb_vs_input_pose"
            roles = (("source", "common_source"),) if is_all10 else (("heldout", "heldout"), ("source", "common_source"))
            if world_archive and summary.get("holdout", True):
                raise ValueError(f"世界归档回放只支持全10图输入相机，不能冒充原生世界留出协议")
            for role, suffix in roles:
                cameras = case_out / branch / f"{suffix}_cameras.npz"
                if world_archive and role == "source":
                    cameras = case_out / branch / "input_cameras.npz"
                if not cameras.is_file():
                    raise FileNotFoundError(cameras)
                out = case_out / branch / f"{role}_render"
                if out.exists():
                    metrics_path = out / "metrics.json"
                    if not metrics_path.is_file():
                        raise FileExistsError(f"不完整输出，请备份后再运行: {out}")
                    z = np.load(cameras, allow_pickle=False)
                    cams = {k: z[k] for k in z.files}
                    verify_input(summary, case, images_dir, cams["image_names"], cams)
                    target_hashes = validate_existing_render(out, cams, images_dir, role, native, branch, size)
                    identity = json.loads(metrics_path.read_text(encoding="utf-8")).get("input_sha256")
                    if identity is None:
                        identity_path = case_out / branch / f"{role}_input_identity.json"
                        if role == "heldout":
                            # 早期一次性核验只记录留出身份，不能用于验证source集合。
                            identity_path = case_out / branch / "input_identity.json"
                        if identity_path.is_file():
                            identity = json.loads(identity_path.read_text(encoding="utf-8"))
                    if identity not in (None, target_hashes):
                        raise ValueError(f"历史渲染身份记录与当前RGB不匹配: {out}")
                    print(f"⏭️ 保留并核验 {name}/{branch}/{role}", flush=True)
                    continue
                out.mkdir()
                means, scales, quats, _, sh, opacity = render.load_ply(native)
                z = np.load(cameras, allow_pickle=False)
                cams = {k: z[k] for k in z.files}
                verify_input(summary, case, images_dir, cams["image_names"], cams)
                target_hashes = {}
                for image_name in cams["image_names"]:
                    target_hashes[str(image_name)] = sha256(images_dir / str(image_name))
                expected_names = [str(cams["image_names"][i]) for i in range(len(cams["image_names"]))]
                available = {p.name for p in images_dir.iterdir() if p.is_file()}
                if not set(expected_names).issubset(available):
                    raise ValueError(f"RGB缺少相机对应文件: {name}/{branch}/{role}")
                K, base = render.load_intrinsics(cams, size, size)
                records = []
                for i, image_name in enumerate(cams["image_names"]):
                    w2c = torch.eye(4, device="cuda")
                    w2c[:3] = torch.tensor(cams["extrinsics"][i, :3], device="cuda", dtype=torch.float32)
                    color, alpha, _ = rasterization(means=means, scales=scales.exp(),
                        quats=torch.nn.functional.normalize(quats, dim=-1), opacities=opacity.sigmoid(), colors=sh,
                        viewmats=w2c[None], Ks=torch.tensor(K[i:i + 1], device="cuda"),
                        width=size, height=size, packed=False, sh_degree=int(sh.shape[1] ** .5) - 1,
                        eps2d=.01, backgrounds=torch.ones(1, 3, device="cuda"), render_mode="RGB")
                    rgb = color[0].clamp(0, 1).cpu().numpy()
                    gt = np.asarray(Image.open(images_dir / str(image_name)).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)) / 255.
                    psnr = float(-10 * np.log10(np.mean((rgb - gt) ** 2).clip(1e-12)))
                    Image.fromarray((rgb * 255).astype("uint8")).save(out / f"{i:02d}.png")
                    Image.fromarray((np.asarray(gt) * 255).astype("uint8")).save(out / f"{i:02d}_gt.png")
                    records.append({"local_view": i, "image_name": str(image_name), "psnr_db": psnr,
                                    "alpha_mean": float(alpha.mean())})
                info = {"case": str(case), "branch": branch, "role": role, "ply": str(native),
                        "ply_sha256": native_info["sha256"], "cameras": str(cameras),
                        "render_size": size, "intrinsics_base_size": list(base), "input_sha256": target_hashes,
                        "mean_psnr_db": float(np.mean([r["psnr_db"] for r in records])),
                        "mean_alpha": float(np.mean([r["alpha_mean"] for r in records])), "views": records,
                        "limits": "源回放仅诊断" if role == "source" else "留出RGB未前向；仅source拟合Sim(3)，非独立几何真值"}
                with open(out / "metrics.json", "w", encoding="utf-8", newline="\n") as f:
                    json.dump(info, f, ensure_ascii=False, indent=2)
                    f.write("\n")
                report = [r for r in report if not (r["case"] == info["case"] and r["branch"] == branch and r["role"] == role)]
                report.append(info)
                with open(report_path, "w", encoding="utf-8", newline="\n") as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
                    f.write("\n")
                print(f"✅ {name}/{branch}/{role}: {info['mean_psnr_db']:.4f} dB", flush=True)
                del means, scales, quats, sh, opacity
                torch.cuda.empty_cache()
    with open(report_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
