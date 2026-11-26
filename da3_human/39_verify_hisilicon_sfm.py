#!/usr/bin/env python3
"""核验海思 depthanything3_sfm.ply 的字段、数值和 SfM 相机配对。

Env vars: ROOT
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from plyfile import PlyData


def main():
    root = Path(os.environ.get("ROOT", "/mnt/d/dataset/测试数据/海思端侧重建拍摄数据"))
    cases = sorted(root.glob("case*_nopose_recentered"),
                   key=lambda p: int(p.name.split("_")[0][4:]))
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "cases": [], "failures": []}
    for case in cases:
        try:
            ply, out = case / "depthanything3_sfm.ply", case / "depthanything3_sfm"
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            saved = np.load(out / "cameras.npz", allow_pickle=False)
            sfm = np.load(case / "colmap_sfm_results/cameras.npz", allow_pickle=False)
            vertex = PlyData.read(str(ply))["vertex"].data
            order = [list(sfm["names"]).index(n) for n in saved["image_names"]]
            assert saved["image_names"].tolist() == summary["image_names"]
            assert len(vertex) == summary["n_gaussians"] == len(order) * 504 * 504
            assert len(vertex.dtype.names) == 41
            values = np.asarray(vertex).view("<f4").reshape(len(vertex), -1)
            assert np.isfinite(values).all(), "PLY 含 NaN/Inf"
            assert np.array_equal(saved["world_rotation"], np.eye(3))
            assert int(saved["is_metric"]) == 0
            w2c_error = float(np.abs(saved["extrinsics"] - sfm["w2c"][order, :3]).max())
            k_error = float(np.abs(saved["input_intrinsics_original"] - sfm["intrinsics"][order]).max())
            # 输入是 float32 NPZ，4x4 求逆再截回 3x4 有浮点误差，不要求逐 bit 相同。
            assert w2c_error < 2e-6, f"外参误差 {w2c_error}"
            assert k_error < 2e-4, f"内参误差 {k_error}"
            roundtrip = float(np.abs(saved["extrinsics"] - saved["input_extrinsics_original"][:, :3]).max())
            assert roundtrip < 2e-6, f"外参往返误差 {roundtrip}"
            assert saved["depth"].shape == (len(order), 504, 504)
            assert (out / "render_check.jpg").is_file(), "缺少源视角回放"
            record = {"case": case.name, "status": "passed", "n_views": len(order),
                      "n_gaussians": len(vertex), "n_properties": len(vertex.dtype.names),
                      "bytes": ply.stat().st_size, "sha256": summary["sha256"],
                      "camera_max_error": w2c_error, "intrinsics_max_error": k_error,
                      "camera_roundtrip_max_error": roundtrip, "all_finite": True}
            report["cases"].append(record)
            print(f"✅ {case.name}: {len(order)} 视角, {len(vertex):,} 高斯, 41 属性, 全有限", flush=True)
        except Exception as exc:
            report["failures"].append({"case": case.name, "error": repr(exc)})
            print(f"❌ {case.name}: {exc!r}", flush=True)
    report.update({"total_bytes": sum(x["bytes"] for x in report["cases"]),
                   "total_views": sum(x["n_views"] for x in report["cases"]),
                   "successful": len(report["cases"]), "failed": len(report["failures"])})
    path = root / "depthanything3_sfm_validation.json"
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"🎉 {report['successful']}/{len(cases)} 通过；{report['total_views']} 视角；"
          f"{report['total_bytes']:,} bytes；报告: {path}", flush=True)
    if report["failures"] or len(cases) != 17:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
