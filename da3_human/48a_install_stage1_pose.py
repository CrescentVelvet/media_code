#!/usr/bin/env python3
"""将已完成的8源相机分支转正后替换旧入口；先备份，绝不重新解释原生指标。

只处理指定且已有完整结果的案例。源实验目录和原生PLY保持不动，数据集内只替换
默认相机版PLY及其配套相机/摘要。每批最多10例，失败立即停止。
Env vars: ROOT, RESULTS_ROOT, BACKUP_ROOT, CASE_FILTER（逗号分隔）
"""
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parent


def main():
    spec = importlib.util.spec_from_file_location("display", SCRIPT / "37_flip_hisilicon_display.py")
    display = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(display)
    from depth_anything_3.utils.sh_helpers import rotate_sh  # noqa: F401
    from depth_anything_3.model.utils.transform import mat_to_quat  # noqa: F401
    root = Path(os.environ["ROOT"])
    results = Path(os.environ["RESULTS_ROOT"])
    backup = Path(os.environ["BACKUP_ROOT"])
    names = [s for s in os.environ["CASE_FILTER"].split(",") if s]
    if not names or len(names) > 10:
        raise ValueError("CASE_FILTER须明确指定1–10个案例，禁止整目录模糊替换")
    if backup.exists() or backup.is_symlink():
        raise FileExistsError(f"备份目录必须不存在: {backup}")
    backup.mkdir(parents=True)
    pending = []
    for name in names:
        case = root / name
        experiment = results / name
        summary_path = experiment / "summary.json"
        source = experiment / "pose/input_world.ply"
        native_cameras = experiment / "pose/input_cameras.npz"
        target = case / "depthanything3_cameras.ply"
        destination = case / "depthanything3_cameras"
        old_files = [target, destination / "cameras.npz", destination / "summary.json"]
        if any(p.is_symlink() or not p.is_file() for p in [source, native_cameras, summary_path] + old_files):
            raise RuntimeError(f"源产物或旧入口缺失/为符号链接: {name}")
        report = json.loads(summary_path.read_text(encoding="utf-8"))
        if report["case"] != str(case) or report["source_indices"] != [0, 1, 3, 4, 5, 6, 8, 9]:
            raise RuntimeError(f"案例或source索引不匹配: {name}")
        if report["target_indices"] != [2, 7] or set(report["branches"]) != {"rgb", "pose"}:
            raise RuntimeError(f"不是完整的8源+2留出双分支: {name}")
        print(f"🔎 校验实验与输入身份: {name}", flush=True)
        # input_world是同一前向派生，须由原生PLY信息加输入对齐记录独立确认。
        if not (experiment / "pose/native.ply").is_file():
            raise RuntimeError(f"缺少原生PLY: {name}")
        if display._io.sha256(experiment / "pose/native.ply") != report["branches"]["pose"]["sha256"]:
            raise RuntimeError(f"原生PLY哈希与摘要不一致: {name}")
        source_info = display._io.inspect_ply(source)
        if source_info["n_gaussians"] != 3920000 or source_info["n_properties"] != 41:
            raise RuntimeError(f"替换源不是完整392万高斯/41属性: {name}")
        with np.load(native_cameras, allow_pickle=False) as cam:
            if len(cam["extrinsics"]) != 8 or "depth" in cam.files:
                raise RuntimeError(f"输入相机不是预期的8源无深度文件: {name}")
            flipped = display.camera_to_display(cam)
        camera_spec = importlib.util.spec_from_file_location("camera_input", SCRIPT / "36_run_hisilicon_cameras.py")
        camera_input = importlib.util.module_from_spec(camera_spec)
        camera_spec.loader.exec_module(camera_input)
        all_files, all_ext, all_K, camera_path, manifest = camera_input.load_stage1_inputs(case)
        source = np.asarray(report["source_indices"])
        targets = np.asarray(report["target_indices"])
        full_ext = np.empty((len(all_files), 3, 4), dtype=np.float64)
        full_ext[source] = flipped["extrinsics"]
        # 留出相机的原生映射已由同一组source相机拟合；只做与PLY一致的显示旋转。
        with np.load(experiment / "pose/heldout_cameras.npz", allow_pickle=False) as heldout:
            heldout_display = display.camera_to_display(heldout)
        full_ext[targets] = heldout_display["extrinsics"]
        full = {key: flipped[key].copy() for key in flipped}
        full.update({"extrinsics": full_ext, "intrinsics": all_K,
                     "image_shape": np.asarray([700, 700]),
                     "image_names": np.asarray([p.name for p in all_files]),
                     "input_extrinsics_original": all_ext, "input_intrinsics_original": all_K,
                     "world_rotation": display.ROTATE_X_180,
                     "source_indices": source, "target_indices": targets,
                     "is_metric": np.int32(0),
                     "coordinate": np.asarray("输入世界绕X轴180度；高斯与相机同步旋转；8源前向，2留出未参与")})
        # 原生source映射回输入世界后的旋转与输入相机应数值一致，容差只覆盖浮点拟合。
        if not np.allclose(full_ext[source], flipped["extrinsics"], atol=1e-7, rtol=0):
            raise RuntimeError(f"源相机写入顺序错误: {name}")
        pending.append({"name": name, "case": case, "experiment": experiment,
                        "report": report, "source": experiment / "pose/input_world.ply", "target": target,
                        "destination": destination, "old_files": old_files,
                        "source_info": source_info, "flipped": full, "camera_path": camera_path,
                        "input_sha256": [display._io.sha256(p) for p in all_files],
                        "temporary": backup / f"{name}.upright.staged.ply"})
        print(f"📦 校验通过，待转正替换: {name}", flush=True)
    records = []
    print("将先在备份盘生成转正PLY，验证后再替换以下默认入口：", flush=True)
    for row in pending:
        print(f"  {row['target']}", flush=True)
    try:
        for row in pending:
            name = row["name"]
            temporary = row["temporary"]
            if temporary.exists() or temporary.is_symlink():
                raise RuntimeError(f"临时目标已存在/为链接: {temporary}")
            print(f"🔄 生成并验证转正PLY: {name}", flush=True)
            new_info = display.transform_ply(row["source"], temporary)
            if display._io.sha256(temporary) != new_info["sha256"]:
                raise RuntimeError(f"转正PLY暂存校验失败: {name}")
            # 新文件已成功落盘之后才进入旧入口的备份窗口。
            case_backup = backup / name
            case_backup.mkdir()
            old_hashes = {}
            for old in row["old_files"]:
                digest = display._io.sha256(old)
                saved = case_backup / old.name
                # 同盘改名保留旧失败文件，避免复制过程中产生另一份半写数据。
                old.replace(saved)
                if display._io.sha256(saved) != digest:
                    raise RuntimeError(f"旧文件备份校验失败: {saved}")
                old_hashes[old.name] = digest
            try:
                temporary.replace(row["target"])
                if display._io.sha256(row["target"]) != new_info["sha256"]:
                    raise RuntimeError(f"转正PLY写入校验失败: {name}")
                np.savez_compressed(row["destination"] / "cameras.npz", **row["flipped"])
                info = dict(row["report"])
                info.update(new_info)
                info.update({
                    "status": "completed", "n_views": 8, "source_indices": [0, 1, 3, 4, 5, 6, 8, 9],
                    "target_indices": [2, 7], "gaussian_frame": "8源原生pose分支完整Sim(3)对齐后，与相机同步绕X轴180度",
                    "camera_coordinate": "OpenCV w2c；相机与高斯已同步绕输入世界X轴180度",
                    "coordinate": "输入世界绕X轴180度；高斯与相机同步旋转",
                    "world_rotation": display.ROTATE_X_180.tolist(),
                    "output_extrinsics": row["flipped"]["extrinsics"].tolist(),
                    "experimental_summary": str(row["experiment"] / "summary.json"),
                    "input_world_ply": str(row["source"]),
                    "native_ply": str(row["experiment"] / "pose/native.ply"),
                    "source_input_world_sha256": row["source_info"]["sha256"],
                    "old_entry_backup": str(case_backup), "old_entry_sha256": old_hashes,
                    "scope": "默认入口改用本轮8源pose输入世界转正版；不是质量通过认证，源/留出指标另查实验目录",
                })
                display._io.dump_json(row["destination"] / "summary.json", info)
                for key in ("extrinsics", "intrinsics", "image_names", "image_shape"):
                    if key not in row["flipped"]:
                        raise RuntimeError(f"配套相机缺少字段 {key}: {name}")
                records.append({"case": name, "status": "completed", "ply": str(row["target"]),
                                "sha256": new_info["sha256"], "backup": str(case_backup)})
                print(f"✅ {name}: 默认相机版PLY/相机/摘要已替换，旧文件已备份", flush=True)
            finally:
                if temporary.exists():
                    if temporary.is_symlink():
                        raise RuntimeError(f"临时文件意外成为链接: {temporary}")
                    temporary.unlink()
    except Exception:
        display._io.dump_json(backup / "interrupted.json", {
            "completed": records, "unprocessed": [r["name"] for r in pending if r["name"] not in {x["case"] for x in records}],
            "note": "已替换案例不自动回滚；旧入口在对应case备份目录，当前案例的失败点须单独检查",
        })
        raise
    result = {"completed": records, "n_completed": len(records),
              "finished_at": datetime.now(timezone.utc).isoformat(),
              "note": "本轮仅替换明确列出的案例；原始RGB/相机JSON及实验输出目录不变"}
    display._io.dump_json(backup / "migration_summary.json", result)
    print(f"🎉 完成 {len(records)}/{len(pending)} 例；旧文件备份: {backup}", flush=True)


if __name__ == "__main__":
    main()
