#!/usr/bin/env python3
"""批量核验导出文件、NPZ 互逆、注册名单和未合并子模型。

Env: DATASET_ROOT, OUTPUT_DIRNAME.
"""
import json
import os
from pathlib import Path

import numpy as np
from sfm_export import read_model

root = Path(os.environ["DATASET_ROOT"])
name = os.environ.get("OUTPUT_DIRNAME", "colmap_sfm_results")
required = ["cameras.npz", "camera_poses.csv", "index.html", "overview.png", "quality_report.txt",
            "sparse/0/cameras.bin", "sparse/0/images.bin", "sparse/0/points3D.bin",
            "sparse/0/txt/cameras.txt", "sparse/0/txt/images.txt", "sparse_with_cameras.ply"]
failed = []
for case in sorted(root.glob("case*")):
    out = case / name
    if not (out / "summary.json").is_file():
        continue
    missing = [p for p in required if not (out / p).is_file()]
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    z = np.load(str(out / "cameras.npz"))
    assert len(z["names"]) == summary["registered_count"]
    assert list(z["names"]) == summary["registered_names"]
    assert len(summary["registered_names"]) + len(summary["unregistered_names"]) == summary["input_count"]
    inv_error = float(np.max(np.abs(z["c2w"] @ z["w2c"] - np.eye(4))))
    assert inv_error < 1e-5, (case.name, inv_error)
    models = []
    for model in sorted((out / "sparse_raw").iterdir()):
        if (model / "txt" / "images.txt").exists():
            _, images, _ = read_model(model / "txt")
            models.append({"model": model.name, "registered": len(images), "names": [x["name"] for x in images]})
    summary["submodels"] = models
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(f"{case.name}: {summary['registered_count']}/{summary['input_count']} | {summary['points3D_count']} 点 | "
          f"互逆误差 {inv_error:.2e} | {len(models)} 个原始子模型")
    if len(models) > 1:
        for model in models:
            print(f"  {model['model']}: {model['registered']} 帧, {model['names']}")
    if missing:
        failed.append({"case": case.name, "missing": missing})
if failed:
    raise SystemExit(f"❌ 缺失文件: {failed}")
print("✅ 所有已完成 case 的必需文件、名单与 NPZ 互逆校验通过")
