#!/usr/bin/env python3
"""只重绘已完成 COLMAP 模型的报告，不重新估计位姿。"""
import json
import os
from pathlib import Path

from sfm_export import export_result

root = Path(os.environ["DATASET_ROOT"])
name = os.environ.get("OUTPUT_DIRNAME", "colmap_sfm_results")
exts = {".jpg", ".jpeg", ".png", ".webp"}
for case in sorted(root.glob("case*")):
    out = case / name
    if not (out / "summary.json").exists():
        continue
    files = sorted([p for p in (case / "images").iterdir() if p.suffix.lower() in exts], key=lambda p: p.name)
    old = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    print(f"🖼️ {case.name}: 重绘报告（模型不变）", flush=True)
    export_result(case, files, out, out / "sparse" / "0", out / "database.db", old["elapsed_seconds"])
from collect_sfm_results import main as collect
collect()
