#!/usr/bin/env python3
"""把已验收的全10图DA3双分支安装到v2各采集样例目录。

rgb -> depthanything3.ply；pose -> depthanything3_cameras.ply。
两支均使用input_world.ply及配套input_cameras.npz，保持stage1输入OpenCV世界，
不做显示转正，不改原始RGB/相机JSON，不重跑前向。默认已有目标只核验、不覆盖；
FORCE=1须先复制备份原有目标文件，且每批明确不超过10例。

Env: ROOT, RESULTS_ROOT, BACKUP_ROOT, CASE_FILTER（逗号分隔1–10例）,
     EXPECTED_CASES（本批案例数）, FORCE=0（默认）/1,
     REPORT_PATH（可选；独立批次使用独立清单，最终可只读合并）。
"""
import importlib.util
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parent


BRANCHES = {"rgb": "depthanything3", "pose": "depthanything3_cameras"}


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump_json(path, data):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.installing")
    if path.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"摘要路径或临时文件需人工检查: {path}")
    try:
        with open(temporary, "x", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.is_file() and not temporary.is_symlink():
            temporary.unlink()


def copy_verified(source, target, expected_hash):
    """先复制为同目录临时文件，全量哈希通过后才原子安装。"""
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"源不是普通文件: {source}")
    if target.parent.is_symlink() or not target.parent.is_dir():
        raise RuntimeError(f"目标父目录不是普通目录: {target.parent}")
    temporary = target.with_name(f".{target.name}.{os.getpid()}.installing")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"安装临时文件已存在，需人工检查: {temporary}")
    try:
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"最终PLY已存在，拒绝覆盖: {target}")
        source_hash = sha256(source)
        if source_hash != expected_hash:
            raise RuntimeError(f"安装源哈希不一致: {source}")
        digest = __import__("hashlib").sha256()
        with source.open("rb") as src, temporary.open("xb") as dst:
            while block := src.read(8 * 1024 * 1024):
                digest.update(block)
                dst.write(block)
            dst.flush()
            os.fsync(dst.fileno())
        if digest.hexdigest() != expected_hash:
            raise RuntimeError(f"复制流哈希不一致: {source}")
        os.replace(temporary, target)
        if target.stat().st_size != source.stat().st_size:
            raise RuntimeError(f"安装后文件大小不一致: {target}")
    finally:
        if temporary.exists():
            if temporary.is_symlink():
                raise RuntimeError(f"临时文件意外成为链接: {temporary}")
            temporary.unlink()


def load_cameras(path):
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"配套相机不是普通文件: {path}")
    with np.load(path, allow_pickle=False) as data:
        cameras = {k: data[k].copy() for k in data.files}
    if cameras["extrinsics"].shape != (10, 3, 4):
        raise ValueError(f"相机不是全10图w2c: {path}, {cameras['extrinsics'].shape}")
    if cameras["intrinsics"].shape != (10, 3, 3) or len(cameras["image_names"]) != 10:
        raise ValueError(f"相机内参或图像名不是全10图: {path}")
    if not np.isfinite(cameras["extrinsics"]).all() or not np.isfinite(cameras["intrinsics"]).all():
        raise ValueError(f"相机含NaN/Inf: {path}")
    return cameras


def camera_record(cameras, source_path):
    return {"extrinsics": cameras["extrinsics"].tolist(),
            "intrinsics": cameras["intrinsics"].tolist(),
            "image_names": [str(x) for x in cameras["image_names"]],
            "source_camera_npz": str(source_path),
            "source_camera_sha256": sha256(source_path)}


def inspect_ply(path):
    from plyfile import PlyData
    ply = PlyData.read(str(path), mmap="r")
    vertex = ply["vertex"]
    if len(vertex) != 4900000 or len(vertex.data.dtype.names) != 41:
        raise RuntimeError(f"PLY不是完整全10图高斯: {path}")
    if os.environ.get("CHECK_FINITE", "0") == "1":
        values = np.asarray(vertex.data).view("<f4").reshape(len(vertex), -1)
        if not np.isfinite(values).all():
            raise RuntimeError(f"PLY含NaN/Inf: {path}")
    # 源摘要已完成全有限检查，安装以SHA证明逐字节不变；不为复制再次扫数亿浮点。
    return {"n_gaussians": len(vertex), "n_properties": len(vertex.data.dtype.names),
            "bytes": path.stat().st_size, "sha256": sha256(path)}


def main():
    root = Path(os.environ["ROOT"])
    results = Path(os.environ["RESULTS_ROOT"])
    backup_root = Path(os.environ["BACKUP_ROOT"])
    names = [x for x in os.environ.get("CASE_FILTER", "").split(",") if x]
    if not names or len(names) > 10 or len(set(names)) != len(names):
        raise ValueError("CASE_FILTER须明确指定1–10个唯一案例")
    expected_cases = int(os.environ.get("EXPECTED_CASES", len(names)))
    if expected_cases != len(names):
        raise ValueError(f"EXPECTED_CASES={expected_cases}与实际筛选{len(names)}例不一致")
    force = os.environ.get("FORCE", "0") == "1"
    if root.is_symlink() or results.is_symlink() or not root.is_dir() or not results.is_dir():
        raise ValueError("输入根或实验根无效/为符号链接")
    if force and backup_root.exists():
        raise FileExistsError(f"FORCE备份目录必须不存在: {backup_root}")
    if force:
        backup_root.mkdir(parents=True)
    pending = []
    for name in names:
        case = root / name
        experiment = results / name
        summary_path = experiment / "summary.json"
        if case.is_symlink() or not case.is_dir() or summary_path.is_symlink() or not summary_path.is_file():
            raise FileNotFoundError(f"案例或完整实验摘要不存在: {name}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        spec = importlib.util.spec_from_file_location("hisilicon_cameras", SCRIPT / "36_run_hisilicon_cameras.py")
        camera_input = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(camera_input)
        files, _, _, camera_path, _ = camera_input.load_stage1_inputs(case)
        if camera_input._io.sha256(camera_path) != summary["camera_sha256"]:
            raise RuntimeError(f"采集相机JSON身份变化: {name}")
        if [camera_input._io.sha256(p) for p in files] != summary["input_sha256"]:
            raise RuntimeError(f"原始RGB身份变化: {name}")
        if (summary.get("status") != "completed" or Path(summary.get("case", "")).name != name
                or summary.get("protocol") != "all10_rgb_vs_input_pose"
                or summary.get("source_indices") != list(range(10))
                or summary.get("target_indices") != [] or summary.get("holdout") is not False
                or tuple(summary.get("requested_branches", ["rgb", "pose"])) != ("rgb", "pose")
                or set(summary.get("branches", {})) != set(BRANCHES)):
            raise ValueError(f"不是完整全10图双分支，拒绝安装: {name}")
        rows = []
        for branch, subdir in BRANCHES.items():
            source = experiment / branch / "input_world.ply"
            source_cameras = experiment / branch / "input_cameras.npz"
            target = case / ("depthanything3.ply" if branch == "rgb" else "depthanything3_cameras.ply")
            out = case / subdir
            info = summary["branches"][branch].get("input_world_ply", {})
            if info.get("sha256") != sha256(source):
                raise RuntimeError(f"input_world源哈希与摘要不一致: {source}")
            if info.get("n_gaussians") != 4900000 or info.get("n_properties") != 41:
                raise ValueError(f"input_world不是完整SH2高斯: {source}")
            cameras = load_cameras(source_cameras)
            if cameras["image_names"].tolist() != summary["image_names"]:
                raise ValueError(f"相机图像顺序与摘要不一致: {name}/{branch}")
            expected_output = np.asarray(summary["branches"][branch]["input_alignment"]["output_extrinsics"])
            if not np.allclose(cameras["extrinsics"], expected_output, atol=1e-7, rtol=0):
                raise ValueError(f"相机与input_world对齐摘要不一致: {name}/{branch}")
            if not np.array_equal(cameras["world_rotation"], np.eye(3)):
                raise ValueError(f"安装输入世界出现额外显示转正: {name}/{branch}")
            if out.is_symlink() or (out.exists() and not out.is_dir()):
                raise ValueError(f"配套目录不是普通目录: {out}")
            targets = [target, out / "cameras.npz", out / "summary.json"]
            if any(p.is_symlink() for p in targets):
                raise ValueError(f"拒绝安装到符号链接: {name}")
            if any(p.exists() for p in targets) and force:
                raise ValueError("本次中断续装仅核验/补齐，FORCE暂不支持部分安装目录")
            rows.append({"branch": branch, "source": source, "source_cameras": source_cameras,
                         "target": target, "out": out, "targets": targets, "cameras": cameras,
                         "expected_hash": info["sha256"], "skip": all(p.is_file() for p in targets)})
        pending.append({"name": name, "case": case, "experiment": experiment,
                        "summary": summary, "rows": rows})

    def verify_existing(row, summary):
        """中断续装只放行完整一致的文件，不因存在同名PLY就跳过。"""
        if not all(p.is_file() and not p.is_symlink() for p in row["targets"]):
            raise FileExistsError(f"已有安装不完整且含异常路径，拒绝覆盖: {row['targets']}")
        ply_info = inspect_ply(row["target"])
        if ply_info["sha256"] != row["expected_hash"]:
            raise RuntimeError(f"已有PLY不是指定实验结果: {row['target']}")
        cameras = load_cameras(row["out"] / "cameras.npz")
        for key, value in row["cameras"].items():
            if not np.array_equal(cameras[key], value):
                raise ValueError(f"已有配套相机与实验不一致: {row['out']}, {key}")
        previous = json.loads((row["out"] / "summary.json").read_text(encoding="utf-8"))
        expected_fields = {"case": Path(summary["case"]).name, "branch": row["branch"],
                           "protocol": "all10_rgb_vs_input_pose", "status": "completed",
                           "input_sha256": summary["input_sha256"],
                           "camera_sha256": summary["camera_sha256"], "sha256": row["expected_hash"]}
        if any(previous.get(key) != value for key, value in expected_fields.items()):
            raise ValueError(f"已有安装摘要身份不符: {row['out']}")
        return {"branch": row["branch"], "ply": str(row["target"]),
                "cameras": str(row["out"] / "cameras.npz"), "summary": str(row["out"] / "summary.json"),
                **ply_info, "status": "verified_existing", "backups": []}

    records = []
    for case_row in pending:
        name = case_row["name"]
        installed = []
        for row in case_row["rows"]:
            if row["skip"]:
                installed.append(verify_existing(row, case_row["summary"]))
                print(f"⏭️ {name}/{row['branch']}: 已有PLY、相机及摘要全部核验一致，跳过", flush=True)
                continue
            backups = []
            if force:
                case_backup = backup_root / name
                case_backup.mkdir(parents=True, exist_ok=True)
                for old in row["targets"]:
                    if old.exists():
                        digest = sha256(old)
                        saved = case_backup / old.parent.name / old.name
                        if saved.parent.is_symlink():
                            raise RuntimeError(f"备份目录为链接: {saved.parent}")
                        saved.parent.mkdir(parents=True, exist_ok=True)
                        if saved.exists() or saved.is_symlink():
                            raise FileExistsError(f"备份目标已存在: {saved}")
                        shutil.copy2(old, saved)
                        if sha256(saved) != digest:
                            raise RuntimeError(f"备份哈希不一致: {saved}")
                        backups.append({"original": str(old), "backup": str(saved), "sha256": digest})
            print(f"📦 {name}/{row['branch']}: 安装已验收input_world PLY", flush=True)
            copy_verified(row["source"], row["target"], row["expected_hash"])
            ply_info = inspect_ply(row["target"])
            if ply_info["sha256"] != row["expected_hash"]:
                raise RuntimeError(f"安装后PLY完整检查失败: {name}")
            row["out"].mkdir(exist_ok=True)
            camera_target = row["out"] / "cameras.npz"
            temporary = camera_target.with_name(f".cameras.{os.getpid()}.installing.npz")
            if camera_target.exists() or camera_target.is_symlink() or temporary.exists() or temporary.is_symlink():
                raise FileExistsError(f"相机目标或临时文件需人工检查: {camera_target}")
            try:
                np.savez_compressed(temporary, **row["cameras"])
                installed_cameras = load_cameras(temporary)
                if any(not np.array_equal(installed_cameras[key], value)
                       for key, value in row["cameras"].items()):
                    raise RuntimeError(f"配套相机回读不一致: {name}")
                os.replace(temporary, camera_target)
            finally:
                if temporary.is_file() and not temporary.is_symlink():
                    temporary.unlink()
            if load_cameras(camera_target)["image_names"].tolist() != case_row["summary"]["image_names"]:
                raise RuntimeError(f"配套相机回读失败: {name}")
            info = {"case": name, "status": "completed", "branch": row["branch"],
                    "protocol": "all10_rgb_vs_input_pose", "n_views": 10,
                    "source_indices": list(range(10)), "target_indices": [], "holdout": False,
                    "process_res": case_row["summary"]["process_res"],
                    "model": case_row["summary"]["model"], "seed": 42,
                    "path": str(row["target"]), **ply_info,
                    "image_names": case_row["summary"]["image_names"],
                    "input_sha256": case_row["summary"]["input_sha256"],
                    "camera_json": case_row["summary"]["camera_json"],
                    "camera_sha256": case_row["summary"]["camera_sha256"],
                    "camera_coordinate": "stage1输入OpenCV世界；高斯与相机未做额外显示转正",
                    "world_rotation": np.eye(3).tolist(),
                    "output_extrinsics": row["cameras"]["extrinsics"].tolist(),
                    "experimental_summary": str(case_row["experiment"] / "summary.json"),
                    "input_world_ply": str(row["source"]),
                    "native_ply": str(case_row["experiment"] / row["branch"] / "native.ply"),
                    "input_alignment": case_row["summary"]["branches"][row["branch"]]["input_alignment"],
                    "installed_camera": camera_record(row["cameras"], row["source_cameras"]),
                    "old_entry_backups": backups,
                    "installed_at": datetime.now(timezone.utc).isoformat(),
                    "scope": "用户指定默认入口名；全10图前向，input_world事后Sim(3)；不代表画质或几何验收通过"}
            if (row["out"] / "summary.json").exists() or (row["out"] / "summary.json").is_symlink():
                raise FileExistsError(f"安装摘要已存在，拒绝覆盖: {row['out'] / 'summary.json'}")
            dump_json(row["out"] / "summary.json", info)
            installed.append({"branch": row["branch"], "ply": str(row["target"]),
                              "cameras": str(camera_target), "summary": str(row["out"] / "summary.json"),
                              "sha256": row["expected_hash"], "n_gaussians": ply_info["n_gaussians"],
                              "n_properties": ply_info["n_properties"], "bytes": ply_info["bytes"],
                              "backups": backups})
            print(f"✅ {name}/{row['branch']}: {row['target']} / {ply_info['bytes']:,}B", flush=True)
        records.append({"case": name, "status": "completed", "branches": installed})

    report_path = Path(os.environ["REPORT_PATH"]) if os.environ.get("REPORT_PATH") else results / "dataset_installation_summary.json"
    if report_path.is_symlink():
        raise ValueError(f"安装报告不能是符号链接: {report_path}")
    previous = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    if previous and (previous.get("protocol") != "all10_rgb_vs_input_pose"
                     or any(r["case"] not in names for r in previous.get("cases", []))):
        raise ValueError(f"已有安装总表协议不同或含本批以外案例，请使用独立报告: {report_path}")
    merged = {r["case"]: r for r in previous.get("cases", [])}
    merged.update({r["case"]: r for r in records})
    report = {"protocol": "all10_rgb_vs_input_pose", "root": str(root), "results_root": str(results),
              "names": {"rgb": "depthanything3.ply", "pose": "depthanything3_cameras.ply"},
              "coordinate": "各分支input_world，stage1输入OpenCV世界，无额外显示转正",
              "n_installed_cases_this_run": len(records), "n_installed_cases_total": len(merged),
              "expected_dataset_cases": 34, "expected_complete_ab_cases": 33,
              "case14": "两支PLY导出失败，不安装，不伪造",
              "cases": [merged[k] for k in sorted(merged)],
              "last_run_cases": [r["case"] for r in records],
              "finished_at": datetime.now(timezone.utc).isoformat(),
              "limits": "安装是文件复制和相机配套，不是画质或新视角评价；原始RGB/相机JSON不改"}
    dump_json(report_path, report)
    print(f"🎉 本轮安装{len(records)}例，共{len(report['cases'])}例；清单 {report_path}", flush=True)


if __name__ == "__main__":
    main()
