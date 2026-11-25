#!/usr/bin/env python3
"""批量运行原生 DA3 GSDPT，将海思十视角案例导出为 depthanything3.ply。

复用已验收的 splatanything/images 抽帧结果；只读取 RGB，不读其相机或 PLY。
模型只加载一次；高斯与相机统一为米制，不裁边/深度、保留全部 SH。

Env vars:
  ROOT            海思案例根目录
  OUT_SUBDIR      每个案例的中间产物目录，默认 depthanything3
  CASE_FILTER     可选，只跑一个完整目录名（冒烟用）
  FORCE           1=备份已有 PLY 后重建；默认完整输出直接跳过
  PROCESS_RES     DA3 输入分辨率，默认 504
  DA3_MODEL_DIR   原生 DA3NESTED-GIANT-LARGE-1.1 权重目录
"""
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "compare_4methods"))
from da3_frame import gaussians_to_metric, gs_scale_factor  # noqa: E402


def dump_json(path, data):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fast_export_ply(means, scales, rotations, harmonics, opacities, path,
                    shift_and_scale=False, save_sh_dc_only=True,
                    match_3dgs_mcmc_dev=False):
    """与官方 export_ply 字段和数值一致，只将 Python tuple 写盘改为连续数组。"""
    if shift_and_scale or match_3dgs_mcmc_dev:
        raise ValueError("本批处理只支持原生坐标与标准 3DGS 字段")

    def np32(t):
        return t.detach().cpu().numpy().astype(np.float32)

    n = means.shape[0]
    k = 0 if save_sh_dc_only else harmonics.shape[-1] - 1
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    names += [f"f_rest_{i}" for i in range(3 * k)]
    names += ["opacity", "scale_0", "scale_1", "scale_2",
              "rot_0", "rot_1", "rot_2", "rot_3"]
    cols = [np32(means), np.zeros((n, 3), np.float32), np32(harmonics[..., 0])]
    if not save_sh_dc_only:
        cols.append(np32(harmonics[..., 1:].reshape(n, -1)))
    cols += [np32(opacities[..., None]), np32(scales.log()), np32(rotations)]
    arr = np.ascontiguousarray(np.concatenate(cols, axis=1), dtype="<f4")
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    header += [f"property float {name}" for name in names] + ["end_header"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        arr.tofile(f)


def inspect_ply(path):
    from plyfile import PlyData

    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    names = vertex.data.dtype.names
    expected = 17 + 3 * (9 - 1)  # 标准 3DGS + 原生 SH degree 2 的 f_rest
    if len(vertex) == 0 or len(names) != expected:
        raise RuntimeError(f"PLY 字段/高斯数异常: {path}, {len(vertex)}, {len(names)}")
    values = np.asarray(vertex.data).view("<f4").reshape(len(vertex), -1)
    if not np.isfinite(values).all():
        raise RuntimeError(f"PLY 含 NaN/Inf: {path}")
    xyz = np.stack([vertex.data[k] for k in ("x", "y", "z")], axis=1)
    return {"path": str(path), "bytes": path.stat().st_size,
            "n_gaussians": len(vertex), "n_properties": len(names),
            "sh_degree": 2, "all_finite": True, "sha256": sha256(path),
            "bounds_min": xyz.min(axis=0).tolist(),
            "bounds_max": xyz.max(axis=0).tolist()}


def main():
    root = Path(os.environ.get("ROOT", "/mnt/d/dataset/测试数据/海思端侧重建拍摄数据"))
    out_subdir = os.environ.get("OUT_SUBDIR", "depthanything3")
    selected = os.environ.get("CASE_FILTER", "")
    force = os.environ.get("FORCE", "0") == "1"
    process_res = int(os.environ.get("PROCESS_RES", "504"))
    model_dir = os.environ.get("DA3_MODEL_DIR", "")
    if not model_dir:
        model_dir = os.environ.get("DA3_HUMAN_MODEL_DIR", "")
    if not root.is_dir() or not model_dir:
        sys.exit("❌ ROOT 或 DA3_MODEL_DIR 未设置")
    if not (Path(model_dir) / "model.safetensors").is_file():
        sys.exit(f"❌ 权重缺失: {model_dir}")
    if Path(out_subdir).is_absolute() or ".." in Path(out_subdir).parts:
        sys.exit("❌ OUT_SUBDIR 必须是案例目录内的相对路径")
    cases = sorted((p for p in root.iterdir() if p.is_dir() and p.name.startswith("case")),
                   key=lambda p: int(p.name.split("_")[0][4:]))
    if selected:
        cases = [p for p in cases if p.name == selected]
    if not cases:
        sys.exit("❌ 没有匹配案例")

    pending = []
    for case in cases:
        images_dir = case / "splatanything" / "images"
        manifest_path = case / "splatanything" / "extraction.json"
        if not manifest_path.is_file():
            sys.exit(f"❌ 缺少已验收抽帧清单: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = [images_dir / Path(r["image"]).name for r in manifest["frames"]]
        if len(files) != 10 or any(not p.is_file() for p in files):
            sys.exit(f"❌ {case.name} 应有 10 张已登记图像: {images_dir}")
        for path, record in zip(files, manifest["frames"]):
            if sha256(path) != record["sha256"]:
                sys.exit(f"❌ 输入图与抽帧清单不一致: {path}")
        target = case / "depthanything3.ply"
        if target.is_file() and not force:
            print(f"⏭️ {case.name}: 已有 PLY，核验后跳过", flush=True)
            info = inspect_ply(target)
            pending.append((case, files, target, info))
            continue
        if target.exists() and not target.is_file():
            sys.exit(f"❌ 目标不是普通文件，拒绝覆盖: {target}")
        pending.append((case, files, target, None))

    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.utils import gsply_helpers as gh
    gh.export_ply = fast_export_ply

    def stable_inverse_sigmoid(x):
        # 官方 log(x/(1-x)) 在 pdf 饱和到 1 时产生 Inf；数学等价的 logit 更稳定。
        # 不改透明度或几何，只把饱和点的可计算 logit 表示为 float32 最大有限值。
        eps = torch.finfo(x.dtype).eps
        logit = torch.where(
            x >= 1,
            torch.full_like(x, -torch.log(torch.tensor(eps, dtype=x.dtype, device=x.device))),
            torch.log(x) - torch.log1p(-x),
        )
        if not torch.isfinite(logit).all():
            raise RuntimeError("DA3 opacity 含非有限 logit")
        return logit

    gh.inverse_sigmoid = stable_inverse_sigmoid

    t0 = time.time()
    print(f"🏋️ 加载原生 DA3: {model_dir}", flush=True)
    model = DepthAnything3.from_pretrained(model_dir).to("cuda")
    print(f"⏱️ 模型加载 {time.time() - t0:.1f}s", flush=True)
    report = {"root": str(root), "model": model_dir, "process_res": process_res,
              "input_policy": "已验收 splatanything/images 的 RGB；不使用海思/SA 相机或几何",
              "export": {"prune_border_gs": False, "prune_by_depth_percent": 1.0,
                         "save_sh_dc_only": False, "scale": "metric"},
              "started_at": datetime.now(timezone.utc).isoformat(), "cases": []}
    failed = False
    for case, files, target, existing in pending:
        out = case / out_subdir
        out.mkdir(parents=True, exist_ok=True)
        summary_path = out / "summary.json"
        if existing is not None:
            report["cases"].append({"case": case.name, "status": "existing", **existing})
            dump_json(out / "batch_summary.json", report)
            continue
        print(f"🚀 {case.name}: {len(files)} 张 RGB", flush=True)
        backup = None
        if target.exists():
            backup = target.with_name(f"depthanything3.backup_{time.strftime('%Y%m%d_%H%M%S')}.ply")
            target.rename(backup)
            print(f"📦 已有结果备份: {backup}", flush=True)
        try:
            t0 = time.time()
            torch.cuda.reset_peak_memory_stats()
            prediction = model.inference(
                [str(p) for p in files], process_res=process_res,
                ref_view_strategy="saddle_balanced", infer_gs=True,
                export_dir=None, export_format="")
            if prediction.gaussians is None:
                raise RuntimeError("原生权重未返回 gaussians")
            sf, _ = gs_scale_factor(prediction)
            gaussians_to_metric(prediction, mode="metric")
            np.savez_compressed(
                out / "cameras.npz", extrinsics=np.asarray(prediction.extrinsics),
                intrinsics=np.asarray(prediction.intrinsics), depth=np.asarray(prediction.depth),
                conf=np.asarray(prediction.conf), scale_factor=np.float32(sf),
                is_metric=np.int32(prediction.is_metric), view_indices=np.arange(len(files)),
                image_names=np.asarray([p.name for p in files]))
            ctx_depth = torch.from_numpy(prediction.depth).unsqueeze(-1).to(
                prediction.gaussians.means)
            gh.save_gaussian_ply(
                gaussians=prediction.gaussians, save_path=str(target), ctx_depth=ctx_depth,
                shift_and_scale=False, save_sh_dc_only=False, gs_views_interval=1,
                inv_opacity=True, prune_by_depth_percent=1.0,
                prune_border_gs=False, match_3dgs_mcmc_dev=False)
            info = inspect_ply(target)
            info.update({
                "case": case.name, "status": "completed", "seconds": round(time.time() - t0, 1),
                "n_views": len(files), "image_names": [p.name for p in files],
                "input_sha256": [sha256(p) for p in files],
                "scale_factor": sf, "is_metric": int(prediction.is_metric),
                "depth_median_m": float(np.median(prediction.depth)),
                "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
                "backup": str(backup) if backup else None})
            dump_json(summary_path, info)
            report["cases"].append(info)
            print(f"✅ {case.name}: {info['n_gaussians']:,} 高斯, "
                  f"{info['bytes'] / 1024 ** 2:.1f} MiB, {info['seconds']:.1f}s", flush=True)
        except Exception as exc:
            failed = True
            report["cases"].append({"case": case.name, "status": "failed", "error": str(exc)})
            print(f"❌ {case.name}: {exc}", flush=True)
        finally:
            if "prediction" in locals():
                del prediction
            torch.cuda.empty_cache()
            dump_json(out / "batch_summary.json", report)
    print(f"🎉 批量结束: {len(report['cases'])}/{len(cases)} 有结果", flush=True)
    if failed or len(report["cases"]) != len(cases):
        sys.exit(1)


if __name__ == "__main__":
    main()
