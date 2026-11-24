#!/usr/bin/env python3
"""Render independently inferred frames in their own original cameras.

All valid points are kept; no cross-frame fusion, registration, scale fitting or
RNG. The router inherits the one pooled SfM scale from the joint experiment.
Environment: SCENE_DIR (required), VIEW_POINT_RADIUS (default 0.8).
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from render_geometry import depth_color, depth_to_world, render_splat, write_json


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    scene_dir = Path(os.environ["SCENE_DIR"])
    output = scene_dir / "single_frame"
    assert (output / "inference.json").is_file()
    inference = json.loads((output / "inference.json").read_text(encoding="utf-8"))
    assert inference["joint_frames"] is False and inference["window"] == 1
    assert sha256(scene_dir / "manifest.json") == inference["input_manifest_sha256"]
    manifest = json.loads((scene_dir / "manifest.json").read_text(encoding="utf-8"))
    scale_path = scene_dir / "sfm_scale.json"
    scale = float(json.loads(scale_path.read_text(encoding="utf-8"))["metric_to_sfm_scale"]) if scale_path.is_file() else 1.0
    geometry = json.loads((scene_dir / "geometry.json").read_text(encoding="utf-8"))
    near_far = geometry["depth_display_range"]
    radius = float(os.environ.get("VIEW_POINT_RADIUS", "0.8"))
    output = output / "own_camera_splat_depth"
    output.mkdir(exist_ok=True)
    report = {
        "scene": manifest["scene"], "inference": "three independent model.infer(T=1) calls; no cross-frame attention",
        "source_single_frame_inference_sha256": sha256(scene_dir / "single_frame" / "inference.json"),
        "sampling": "none; all valid points from each independently predicted depth",
        "registration": "none; original cameras only, no fusion, no ICP, no per-frame scale fitting",
        "metric_to_camera_depth_scale": scale,
        "depth_display_range": near_far,
        "depth_color_note": "Display range inherited from joint scene for qualitative comparison; independently inferred metric depth may drift",
        "renderer": f"finite RGB discs with far-to-near axial-Z ordering; point_radius={radius}; not 3DGS",
        "boundary": "Own-camera projection is not cross-view consistency or independent ground truth",
        "frames": [],
    }
    for fr in manifest["frames"]:
        index = fr["index"]
        depth_path = scene_dir / "single_frame" / "depth" / f"{index:02d}.npz"
        assert sha256(depth_path) == inference["frames"][index]["depth_npz_sha256"]
        depth = np.load(depth_path)["depth"] * scale
        rgb = np.asarray(Image.open(scene_dir / "rgb" / f"{index:02d}.png").convert("RGB"))
        camera = {"K": np.asarray(fr["K"]), "c2w": np.asarray(fr["c2w"]), "w2c": np.asarray(fr["w2c"])}
        world, _ = depth_to_world(depth, camera["K"], camera["c2w"])
        valid = (depth.ravel() > 0.05) & np.isfinite(depth.ravel()) & (depth.ravel() < near_far[1] * 1.2)
        splat, pcd_depth, coverage = render_splat(world[valid], rgb.reshape(-1, 3)[valid], camera, depth.shape, radius)
        for kind, image in [("splat", splat), ("pcd_depth", depth_color(pcd_depth, *near_far))]:
            Image.fromarray(image).save(output / f"{index:02d}_{kind}.png")
        record = {
            "index": index, "point_count": int(valid.sum()), "disc_coverage": coverage,
            "source_depth_sha256": sha256(depth_path), "source_rgb_sha256": sha256(scene_dir / "rgb" / f"{index:02d}.png"),
            "K": camera["K"].tolist(), "c2w": camera["c2w"].tolist(),
            "files": [f"{index:02d}_splat.png", f"{index:02d}_pcd_depth.png"],
        }
        report["frames"].append(record)
        write_json(output / f"{index:02d}_camera.json", record)
        print(f"✅ {manifest['scene']} single-frame {index}: {valid.sum():,} points, coverage={coverage:.2%}", flush=True)
    write_json(output / "view_render_audit.json", report)
    print(f"🎉 Own-camera single-frame renders: {output}", flush=True)


if __name__ == "__main__":
    main()
