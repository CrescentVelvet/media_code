#!/usr/bin/env python3
"""Export independently inferred depth maps as per-frame RGB point clouds.

No model inference, cross-frame fusion, registration or sampling. The valid-point
mask, original cameras and scene scale match render_single_frame_splat_depth.py.
Environment: SCENE_DIR (the human or router result directory).
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData

from render_geometry import depth_to_world, save_ply, write_json


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    scene_dir = Path(os.environ["SCENE_DIR"])
    single = scene_dir / "single_frame"
    inference_path = single / "inference.json"
    inference = json.loads(inference_path.read_text(encoding="utf-8"))
    manifest = json.loads((scene_dir / "manifest.json").read_text(encoding="utf-8"))
    assert inference["joint_frames"] is False and inference["window"] == 1
    assert len(inference["frames"]) == len(manifest["frames"]) == 3
    assert sha256(scene_dir / "manifest.json") == inference["input_manifest_sha256"]
    scale_path = scene_dir / "sfm_scale.json"
    scale = float(json.loads(scale_path.read_text(encoding="utf-8"))["metric_to_sfm_scale"]) if scale_path.is_file() else 1.0
    geometry = json.loads((scene_dir / "geometry.json").read_text(encoding="utf-8"))
    far = geometry["depth_display_range"][1]
    output = single / "pcd"
    assert not output.is_symlink()
    output.mkdir(exist_ok=True)
    report = {
        "scene": manifest["scene"],
        "inference": "three independent model.infer(T=1) calls; no cross-frame attention",
        "source_single_frame_inference_sha256": sha256(inference_path),
        "coordinate": "OpenCV world coordinates from the original manifest c2w; axial Z lifting",
        "metric_to_camera_depth_scale": scale,
        "registration": "none; no ICP, no per-frame scale fitting, no cross-frame fusion",
        "sampling": "none; the same valid-point mask as own-camera PCD splat/depth",
        "color": "uint8 RGB from each original input image",
        "frames": [],
    }
    for index, frame in enumerate(manifest["frames"]):
        i = frame["index"]
        depth_path = single / "depth" / f"{i:02d}.npz"
        rgb_path = scene_dir / "rgb" / f"{i:02d}.png"
        assert sha256(depth_path) == inference["frames"][index]["depth_npz_sha256"]
        assert sha256(rgb_path) == inference["frames"][index]["input_rgb_sha256"]
        depth = np.load(depth_path)["depth"] * scale
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        world, _ = depth_to_world(depth, np.asarray(frame["K"]), np.asarray(frame["c2w"]))
        valid = (depth.ravel() > 0.05) & np.isfinite(depth.ravel()) & (depth.ravel() < far * 1.2)
        points = world[valid].astype(np.float32)
        colors = rgb.reshape(-1, 3)[valid]
        path = output / f"{i:02d}.ply"
        save_ply(path, points, colors, np.full(len(points), i, np.uint8))

        # Read back the delivered PLY, not merely the in-memory arrays.
        vertices = PlyData.read(str(path))["vertex"].data
        assert len(vertices) == len(points)
        saved = np.column_stack([vertices[k] for k in ("x", "y", "z")])
        error = float(np.max(np.abs(saved - points)))
        assert error <= 2e-6, error
        assert np.array_equal(np.column_stack([vertices[k] for k in ("red", "green", "blue")]), colors)
        assert np.all(vertices["source_index"] == i)
        record = {
            "index": i, "file": f"single_frame/pcd/{i:02d}.ply",
            "point_count": len(points), "ply_sha256": sha256(path),
            "source_depth_sha256": sha256(depth_path), "source_rgb_sha256": sha256(rgb_path),
            "K": frame["K"], "c2w": frame["c2w"],
            "max_ply_float32_roundtrip_error": error,
        }
        report["frames"].append(record)
        print(f"✅ {manifest['scene']} single-frame {i}: {len(points):,} points -> {path}", flush=True)
    report["total_points_across_separate_files"] = sum(f["point_count"] for f in report["frames"])
    write_json(output / "pcd_export_audit.json", report)
    print(f"🎉 Scene complete: {output}", flush=True)


if __name__ == "__main__":
    main()
