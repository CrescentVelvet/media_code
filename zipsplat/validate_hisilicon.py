#!/usr/bin/env python3
"""独立核验批量 ZipSplat PLY：格式、点属性、有限值、输入十图与 SHA-256。

默认只读，不改写 PLY 或输入；报告保存到 RESULTS_DIR/final_validation.json。
WSL: python /mnt/c/code/media_code/zipsplat/validate_hisilicon.py
"""
import hashlib
import json
import os
from pathlib import Path
import struct
import sys

import numpy as np

ROOT = Path(os.environ.get("DATASET_ROOT", "/mnt/d/dataset/测试数据/海思端侧重建拍摄数据"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "/mnt/d/output/results/zipsplat_results/hisilicon_10v"))
SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
SIZES = {"char": 1, "uchar": 1, "int8": 1, "uint8": 1, "short": 2,
         "ushort": 2, "int16": 2, "uint16": 2, "int": 4, "uint": 4,
         "int32": 4, "uint32": 4, "float": 4, "float32": 4,
         "double": 8, "float64": 8}
DTYPES = {"char": "<i1", "uchar": "<u1", "int8": "<i1", "uint8": "<u1",
          "short": "<i2", "ushort": "<u2", "int16": "<i2", "uint16": "<u2",
          "int": "<i4", "uint": "<u4", "int32": "<i4", "uint32": "<u4",
          "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8"}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_ply(path):
    properties = []
    vertex_count = None
    fmt = None
    with path.open("rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError("missing ply magic")
        in_vertex = False
        while True:
            line = f.readline()
            if not line:
                raise ValueError("unterminated PLY header")
            text = line.decode("ascii").strip()
            if text.startswith("format "):
                fmt = text.split()[1]
            elif text.startswith("element "):
                fields = text.split()
                in_vertex = fields[1] == "vertex"
                if in_vertex:
                    vertex_count = int(fields[2])
            elif text.startswith("property ") and in_vertex:
                fields = text.split()
                if fields[1] == "list":
                    raise ValueError("unexpected vertex list property")
                properties.append((fields[2], fields[1]))
            elif text == "end_header":
                break
        if fmt != "binary_little_endian" or not vertex_count:
            raise ValueError(f"invalid format/count: {fmt}/{vertex_count}")
        row_bytes = sum(SIZES[t] for _, t in properties)
        expected = f.tell() + row_bytes * vertex_count
        if expected != path.stat().st_size:
            raise ValueError(f"truncated data: expected {expected}, got {path.stat().st_size}")
        dtype = np.dtype([(name, DTYPES[t]) for name, t in properties])
        points = np.fromfile(f, dtype=dtype, count=vertex_count)
    return points, fmt, row_bytes


def main():
    cases = sorted(p for p in ROOT.iterdir() if p.is_dir() and (p / "images").is_dir())
    if not cases:
        sys.exit(f"❌ 没有 images 子目录: {ROOT}")
    results = []
    for scene in cases:
        files = sorted(p for p in (scene / "images").iterdir()
                       if p.is_file() and p.suffix.lower() in SUFFIXES)
        record = {"scene": scene.name, "image_count": len(files),
                  "ply": str(scene / "zipsplat.ply")}
        print(f"🔍 {scene.name}: {len(files)} images", flush=True)
        try:
            if len(files) != 10:
                raise ValueError(f"expected 10 images, got {len(files)}")
            points, fmt, row_bytes = read_ply(scene / "zipsplat.ply")
            required = ["x", "y", "z", "scale_0", "scale_1", "scale_2", "opacity",
                        "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2"]
            missing = [x for x in required if x not in points.dtype.names]
            if missing:
                raise ValueError(f"missing Gaussian properties: {missing}")
            nonfinite = {name: int((~np.isfinite(points[name])).sum())
                         for name in points.dtype.names
                         if np.issubdtype(points.dtype[name], np.floating)}
            if any(nonfinite.values()):
                raise ValueError(f"nonfinite Gaussian values: {nonfinite}")
            expected_count = 10 * 54 * 24 * 16
            if len(points) != expected_count:
                raise ValueError(f"unexpected 504px square ten-view count {len(points)} != {expected_count}")
            intermediate = RESULTS_DIR / scene.name / "metrics.json"
            if not intermediate.is_file():
                raise ValueError(f"missing inference metrics: {intermediate}")
            metrics = json.loads(intermediate.read_text(encoding="utf-8"))
            if metrics["num_views"] != 10 or metrics["image_files"] != [p.name for p in files]:
                raise ValueError("image list differs from inference metrics")
            if metrics["num_gaussians"] != len(points) or not metrics["token_ba"]:
                raise ValueError("not the expected ten-view TokenBA result")
            if metrics["model_input_size"] != [504, 504]:
                raise ValueError(f"unexpected model input size: {metrics['model_input_size']}")
            record.update(status="passed", vertex_count=len(points), property_count=len(points.dtype.names),
                          row_bytes=row_bytes, format=fmt, bytes=(scene / "zipsplat.ply").stat().st_size,
                          all_finite=True, sha256=sha256(scene / "zipsplat.ply"),
                          mean_input_alpha_gt_05=float(np.mean([
                              x["alpha_gt_05_fraction"] for x in metrics["render_alpha_coverage"]])),
                          model_input_size=metrics["model_input_size"], token_ba=True,
                          source_camera_inputs_used=False,
                          scale_range=[float(points["scale_0"].min()), float(points["scale_0"].max())])
        except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
            record.update(status="failed", error=str(exc))
            print(f"❌ {scene.name}: {exc}", flush=True)
        results.append(record)
        (RESULTS_DIR / "final_validation.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    passed = sum(x["status"] == "passed" for x in results)
    print(f"🎉 验收: {passed}/{len(results)} passed；报告 {RESULTS_DIR / 'final_validation.json'}", flush=True)
    if passed != len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
