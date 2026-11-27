#!/usr/bin/env python3
"""对每个 case/images 独立运行完整 COLMAP SfM，输出位姿和质量报告。

Env: DATASET_ROOT, CASE_REGEX, EXPECTED_IMAGES, OUTPUT_DIRNAME, FORCE, RESUME,
     SIFT_RATIO, MAX_CASES, USE_GPU, COLMAP_ENV, ANALYSIS_PYTHON.
窄基线诊断可显式设 INIT_MIN_TRI_ANGLE / INIT_IMAGE_ID1 / INIT_IMAGE_ID2；
这些参数只影响初始化，最终仍做几何验证和全局 BA。
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from sfm_export import export_result, read_model

ROOT = Path(os.environ["DATASET_ROOT"])
PATTERN = re.compile(os.environ.get("CASE_REGEX", r"^case[0-9]+.*$"))
EXPECTED = int(os.environ.get("EXPECTED_IMAGES", "10"))
OUT_NAME = os.environ.get("OUTPUT_DIRNAME", "colmap_sfm_results")
FORCE = os.environ.get("FORCE", "0") == "1"
MAX_CASES = int(os.environ.get("MAX_CASES", "0"))
USE_GPU = os.environ.get("USE_GPU", "1")
COLMAP = shutil.which("colmap") or str(Path.home() / "miniconda3/envs/colmap/bin/colmap")
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}


def run_colmap(args, log):
    with log.open("w", encoding="utf-8", newline="\n") as f:
        proc = subprocess.run([COLMAP] + args, stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode:
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]
        raise RuntimeError(f"{args[0]} failed (exit {proc.returncode})\n" + "\n".join(tail))


def run_case(case):
    start = time.monotonic()
    images_dir, out = case / "images", case / OUT_NAME
    files = sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTS], key=lambda p: p.name)
    db = out / "database.db"
    if len(files) != EXPECTED:
        raise RuntimeError(f"期望 {EXPECTED} 张图，实际 {len(files)} 张")
    if out.exists() and not FORCE and (out / "sparse_raw" / "0" / "images.bin").exists() and not (out / "summary.json").exists():
        print(f"🔄 {case.name}: 复用已完成的 SIFT/匹配/mapper，补做导出", flush=True)
        candidates = []
        for model in sorted((out / "sparse_raw").iterdir()):
            if (model / "images.bin").exists():
                (model / "txt").mkdir(exist_ok=True)
                if not (model / "txt" / "images.txt").exists():
                    run_colmap(["model_converter", "--input_path", str(model), "--output_path", str(model / "txt"),
                                "--output_type", "TXT"], out / "logs/04_model_to_txt.log")
                _, ims, _ = read_model(model / "txt")
                candidates.append((len(ims), model))
        _, best = max(candidates, key=lambda x: x[0])
        finish_and_export(case, files, out, best, db, start)
        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        print(f"✅ {case.name}: {summary['registered_count']}/{len(files)} 注册, {summary['points3D_count']} 点, "
              f"median={summary['median_reprojection_error_px']:.3f}px", flush=True)
        return summary
    if out.exists() and not FORCE and (out / "summary.json").exists() and os.environ.get("RESUME", "0") == "1":
        current = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        if current["registered_count"] == len(files):
            print(f"⏭️ {case.name}: 已十帧全注册，跳过", flush=True)
            return None
        print(f"🔄 {case.name}: 从已有 {current['registered_count']} 帧模型继续注册", flush=True)
        mapper_args = ["image_registrator", "--database_path", str(db),
            "--input_path", str(out / "sparse" / "0"), "--output_path", str(out / "sparse_registered"),
            "--Mapper.tri_re_max_trials", "5", "--Mapper.tri_re_min_ratio", "0.1",
            "--Mapper.max_reg_trials", "5"]
        if os.environ.get("RELAX_POSE", "0") == "1":
            mapper_args += ["--Mapper.abs_pose_min_num_inliers", "15", "--Mapper.abs_pose_min_inlier_ratio", "0.15"]
        (out / "sparse_registered").mkdir(exist_ok=True)
        run_colmap(mapper_args, out / "logs/08_image_registration.log")
        best = out / "sparse_registered"
        if not (best / "images.bin").exists():
            raise RuntimeError("image_registrator 未生成模型")
        (best / "txt").mkdir(exist_ok=True)
        run_colmap(["model_converter", "--input_path", str(best), "--output_path", str(best / "txt"),
                    "--output_type", "TXT"], out / "logs/09_resume_txt.log")
        finish_and_export(case, files, out, best, db, start)
        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        print(f"✅ {case.name}: {summary['registered_count']}/{len(files)} 注册, {summary['points3D_count']} 点, "
              f"median={summary['median_reprojection_error_px']:.3f}px", flush=True)
        return summary
    if out.exists():
        if not FORCE:
            if (out / "summary.json").exists():
                print(f"⏭️ {case.name}: 已有 summary，跳过", flush=True)
                return None
            raise RuntimeError("已有不完整输出目录；为防覆盖已停止，可用 FORCE=1 自动备份重跑")
        backup = out.parent / f"{OUT_NAME}_backup_{datetime.now():%Y%m%d_%H%M%S}"
        shutil.move(str(out), str(backup))
        print(f"📦 原输出已备份: {backup}", flush=True)
    out.mkdir(parents=True)
    (out / "logs").mkdir()
    (out / "sparse_raw").mkdir()
    db = out / "database.db"
    common = ["--database_path", str(db), "--image_path", str(images_dir)]
    print(f"🚀 {case.name}: {len(files)} 图 SIFT / 穷举匹配 / 增量 SfM", flush=True)
    mapper_overrides = []
    if os.environ.get("INTRINSICS_FROM_STAGE1", "0") == "1":
        camera_json = next(case.glob("*_stage1_cameras.json"))
        camera_data = json.loads(camera_json.read_text(encoding="utf-8"))
        first_view = camera_data["views"][0]
        camera_params = [first_view[k] for k in ("fx", "fy", "cx", "cy")] + [0.0] * 4
        mapper_overrides += ["--ImageReader.camera_params", ",".join(str(x) for x in camera_params)]
    run_colmap(["feature_extractor"] + common + [
        "--ImageReader.camera_model", "OPENCV", "--ImageReader.single_camera", "1",
        "--ImageReader.default_focal_length_factor", "1.2", "--FeatureExtraction.use_gpu", USE_GPU,
        "--FeatureExtraction.max_image_size", "2000", "--SiftExtraction.max_num_features", "16384",
        "--SiftExtraction.first_octave", "-1", "--SiftExtraction.peak_threshold", "0.004",
        "--SiftExtraction.estimate_affine_shape", "1", "--SiftExtraction.domain_size_pooling", "1"] + mapper_overrides, out / "logs/01_features.log")
    mapper_overrides = []
    for key in ("INIT_IMAGE_ID1", "INIT_IMAGE_ID2"):
        if os.environ.get(key):
            mapper_overrides += ["--Mapper." + key.lower(), os.environ[key]]
    run_colmap(["exhaustive_matcher", "--database_path", str(db), "--FeatureMatching.use_gpu", USE_GPU,
        "--FeatureMatching.guided_matching", "1", "--SiftMatching.max_ratio", os.environ.get("SIFT_RATIO", "0.8"),
        "--TwoViewGeometry.max_error", "4"], out / "logs/02_matching.log")
    run_colmap(["mapper"] + common + ["--output_path", str(out / "sparse_raw"),
        "--Mapper.multiple_models", "1", "--Mapper.init_min_num_inliers", "80",
        "--Mapper.min_num_matches", "15", "--Mapper.abs_pose_min_num_inliers", os.environ.get("ABS_POSE_MIN_INLIERS", "30"),
        "--Mapper.init_num_trials", os.environ.get("INIT_NUM_TRIALS", "200"),
        "--Mapper.tri_re_max_trials", "3", "--Mapper.tri_re_min_ratio", "0.15",
        "--Mapper.init_min_tri_angle", os.environ.get("INIT_MIN_TRI_ANGLE", "16"),
        "--Mapper.filter_max_reproj_error", "3", "--Mapper.ba_refine_focal_length", os.environ.get("REFINE_FOCAL", "1"),
        "--Mapper.ba_refine_principal_point", os.environ.get("REFINE_PRINCIPAL", "1"),
        "--Mapper.ba_refine_extra_params", os.environ.get("REFINE_DISTORTION", "1"),
        "--Mapper.ba_global_max_refinements", "5"] + mapper_overrides, out / "logs/03_mapper.log")
    candidates = []
    for model in sorted((out / "sparse_raw").iterdir()):
        if (model / "images.bin").exists():
            (model / "txt").mkdir(exist_ok=True)
            run_colmap(["model_converter", "--input_path", str(model), "--output_path", str(model / "txt"),
                        "--output_type", "TXT"], out / "logs/04_model_to_txt.log")
            _, ims, _ = read_model(model / "txt")
            candidates.append((len(ims), model))
    if not candidates:
        raise RuntimeError("COLMAP 未生成子模型，查看 logs/03_mapper.log")
    _, best = max(candidates, key=lambda x: x[0])
    finish_and_export(case, files, out, best, db, start)
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    print(f"✅ {case.name}: {summary['registered_count']}/{len(files)} 注册, {summary['points3D_count']} 点, "
          f"median={summary['median_reprojection_error_px']:.3f}px, {summary['elapsed_seconds']:.1f}s", flush=True)
    return summary


def finish_and_export(case, files, out, best, db, start):
    (out / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    ba_overrides = []
    for flag, option in (("REFINE_FOCAL", "refine_focal_length"),
                         ("REFINE_PRINCIPAL", "refine_principal_point"),
                         ("REFINE_DISTORTION", "refine_extra_params")):
        if os.environ.get(flag) == "0":
            ba_overrides += ["--BundleAdjustment." + option, "0"]
    run_colmap(["bundle_adjuster", "--input_path", str(best), "--output_path", str(out / "sparse" / "0"),
        "--BundleAdjustment.refine_focal_length", os.environ.get("REFINE_FOCAL", "1"),
        "--BundleAdjustment.refine_principal_point", os.environ.get("REFINE_PRINCIPAL", "1"),
        "--BundleAdjustment.refine_extra_params", os.environ.get("REFINE_DISTORTION", "1"),
        "--BundleAdjustment.refine_points3D", "1",
        "--BundleAdjustmentCeres.max_num_iterations", "200", "--BundleAdjustmentCeres.function_tolerance", "1e-10",
        "--BundleAdjustmentCeres.gradient_tolerance", "1e-12", "--BundleAdjustmentCeres.parameter_tolerance", "1e-10"] + ba_overrides, out / "logs/05_global_ba.log")
    (out / "sparse" / "0" / "txt").mkdir(exist_ok=True)
    run_colmap(["model_converter", "--input_path", str(out / "sparse" / "0"), "--output_path", str(out / "sparse" / "0" / "txt"),
                "--output_type", "TXT"], out / "logs/06_final_txt.log")
    run_colmap(["model_converter", "--input_path", str(out / "sparse" / "0"), "--output_path", str(out / "sparse_points.ply"),
                "--output_type", "PLY"], out / "logs/07_final_ply.log")
    return export_result(case, files, out, out / "sparse" / "0", db, time.monotonic() - start)


def main():
    cases = sorted([p for p in ROOT.iterdir() if p.is_dir() and PATTERN.match(p.name)],
                   key=lambda p: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", p.name)])
    if MAX_CASES:
        cases = cases[:MAX_CASES]
    if not cases:
        sys.exit("❌ 未找到匹配 case 目录")
    print(f"🚀 批量 COLMAP SfM: {len(cases)} 个 case；输出 <case>/{OUT_NAME}/", flush=True)
    records, failures = [], []
    for case in cases:
        try:
            summary = run_case(case)
            if summary:
                records.append(summary)
        except Exception as exc:
            print(f"❌ {case.name}: {exc}", flush=True)
            failures.append({"case": case.name, "error": str(exc)})
    report = {"dataset_root": str(ROOT), "cases": records, "failures": failures,
              "successful": len(records), "failed": len(failures)}
    (ROOT / f"{OUT_NAME}_batch_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(f"🎉 完成: 新成功 {len(records)}，失败 {len(failures)}", flush=True)
    from collect_sfm_results import main as collect
    collect()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
