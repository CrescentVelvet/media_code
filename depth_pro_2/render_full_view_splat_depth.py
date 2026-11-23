#!/usr/bin/env python3
"""Render all valid pooled PCD points in three input and six virtual cameras.

No sampling, inference, registration or per-frame scale fitting is performed.
PCD splat is a finite RGB-disc projection, not a trained Gaussian model.
Environment: SCENE_DIR (required), VIEW_POINT_RADIUS (default 0.8).
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from render_geometry import depth_color, depth_to_world, render_splat, write_json
try:
    from render_view_splat_depth import virtual_camera
except ModuleNotFoundError as error:
    if error.name != "cv2":
        raise
    # PPT/camera validation needs rotations but not the OpenCV image renderer.
    from scipy.spatial.transform import Rotation

    def virtual_camera(first, second, fraction):
        r0 = np.asarray(first["c2w"], dtype=np.float64)[:3, :3]
        r1 = np.asarray(second["c2w"], dtype=np.float64)[:3, :3]
        c0 = np.asarray(first["c2w"], dtype=np.float64)[:3, 3]
        c1 = np.asarray(second["c2w"], dtype=np.float64)[:3, 3]
        relative = Rotation.from_matrix(r1 @ r0.T)
        rotation = Rotation.from_rotvec(relative.as_rotvec() * fraction).as_matrix() @ r0
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = rotation
        c2w[:3, 3] = (1.0 - fraction) * c0 + fraction * c1
        return c2w


def camera_specs(frames):
    specs = []
    for index, frame in enumerate(frames):
        specs.append({
            "view": f"input_{index:02d}", "kind": "input",
            "description": f"Real input camera {index:02d}",
            "c2w": np.asarray(frame["c2w"], dtype=np.float64),
            "K": np.asarray(frame["K"], dtype=np.float64),
            "reference_frame": index, "fraction": None, "pair": None,
        })
    for label, first, second, fraction in [
        ("interpolated_00_01", 0, 1, 0.5),
        ("interpolated_01_02", 1, 2, 0.5),
        ("interpolated_00_02", 0, 2, 0.5),
    ]:
        specs.append({
            "view": label, "kind": "interpolated",
            "description": f"Virtual midpoint of cameras {first:02d}->{second:02d}; t=0.5",
            "c2w": virtual_camera(frames[first], frames[second], fraction),
            "K": np.asarray(frames[second]["K"], dtype=np.float64),
            "reference_frame": second, "fraction": fraction, "pair": [first, second],
        })
    for label, first, second, fraction in [
        ("extrapolated_before_00", 0, 1, -0.5),
        ("extrapolated_after_02", 1, 2, 1.5),
        ("extrapolated_full_arc", 0, 2, -0.25),
    ]:
        specs.append({
            "view": label, "kind": "extrapolated",
            "description": f"Virtual extrapolation along cameras {first:02d}->{second:02d}; t={fraction:g}",
            "c2w": virtual_camera(frames[first], frames[second], fraction),
            "K": np.asarray(frames[second]["K"], dtype=np.float64),
            "reference_frame": second, "fraction": fraction, "pair": [first, second],
        })
    return specs


def main():
    scene_dir = Path(os.environ["SCENE_DIR"])
    radius = float(os.environ.get("VIEW_POINT_RADIUS", "0.8"))
    output = scene_dir / "view_splat_depth_full_9views"
    output.mkdir(exist_ok=True)
    manifest = json.loads((scene_dir / "manifest.json").read_text(encoding="utf-8"))
    frames = manifest["frames"]
    assert len(frames) == 3
    scale_path = scene_dir / "sfm_scale.json"
    scale = float(json.loads(scale_path.read_text(encoding="utf-8"))["metric_to_sfm_scale"]) if scale_path.is_file() else 1.0
    near_far = json.loads((scene_dir / "geometry.json").read_text(encoding="utf-8"))["depth_display_range"]
    worlds, colors, source_counts = [], [], []
    for frame in frames:
        index = frame["index"]
        rgb = np.asarray(Image.open(scene_dir / "rgb" / f"{index:02d}.png").convert("RGB"))
        depth = np.load(scene_dir / "depth" / f"{index:02d}.npz")["depth"] * scale
        world, _ = depth_to_world(depth, np.asarray(frame["K"]), np.asarray(frame["c2w"]))
        valid = (depth.ravel() > 0.05) & np.isfinite(depth.ravel()) & (depth.ravel() < near_far[1] * 1.2)
        worlds.append(world[valid].astype(np.float32))
        colors.append(rgb.reshape(-1, 3)[valid].astype(np.uint8))
        source_counts.append(int(valid.sum()))
    # Do not write another large PLY: geometry.json's source counts are checked
    # against the exact valid-point contract shared with render_geometry.py.
    geometry = json.loads((scene_dir / "geometry.json").read_text(encoding="utf-8"))
    assert geometry["point_count"] == source_counts
    full_world = np.concatenate(worlds)
    full_colors = np.concatenate(colors)
    source_indices = np.concatenate([np.full(count, i, dtype=np.uint8) for i, count in enumerate(source_counts)])
    # Reuse the already validated complete PLY; this is every valid point, no RNG.
    assert len(full_world) == sum(source_counts)
    write_json(output / "full_valid_point_provenance.json", {
        "source_counts": source_counts,
        "point_count": len(full_world),
        "source_fused_ply": "fused_no_registration.ply",
        "source_fused_ply_sha256": hashlib.sha256((scene_dir / "fused_no_registration.ply").read_bytes()).hexdigest(),
        "source_indices_sha256": hashlib.sha256(source_indices.tobytes()).hexdigest(),
    })
    records = []
    for spec in camera_specs(frames):
        c2w, K = spec["c2w"], spec["K"]
        assert np.allclose(c2w[:3, :3].T @ c2w[:3, :3], np.eye(3), atol=1e-6)
        assert np.isclose(np.linalg.det(c2w[:3, :3]), 1, atol=1e-6)
        camera = {"K": K, "w2c": np.linalg.inv(c2w)}
        reference = frames[spec["reference_frame"]]
        h, w = np.load(scene_dir / "depth" / f"{reference['index']:02d}.npz")["depth"].shape
        splat, projected_depth, coverage = render_splat(full_world, full_colors, camera, (h, w), radius)
        name = spec["view"]
        Image.fromarray(splat).save(output / f"{name}_splat.png")
        Image.fromarray(depth_color(projected_depth, *near_far)).save(output / f"{name}_pcd_depth.png")
        record = {key: value for key, value in spec.items() if key not in ("c2w", "K")}
        record.update({
            "is_real_camera": spec["kind"] == "input", "c2w": c2w.tolist(), "K": K.tolist(),
            "image_width": w, "image_height": h, "disc_coverage": coverage,
            "point_count": len(full_world), "source_counts": source_counts,
            "files": [f"{name}_splat.png", f"{name}_pcd_depth.png"],
        })
        write_json(output / f"{name}_camera.json", record)
        records.append(record)
        print(f"FULL {scene_dir.name}/{name}: {len(full_world):,} points, coverage={coverage:.2%}", flush=True)
    report = {
        "scene": manifest["scene"], "point_count": len(full_world), "source_counts": source_counts,
        "sampling": "none; all valid points from all three source frames; identical point set for all nine cameras",
        "source_depth_npz_sha256": {f"depth_{f['index']:02d}": hashlib.sha256(
            (scene_dir / "depth" / f"{f['index']:02d}.npz").read_bytes()).hexdigest() for f in frames},
        "manifest_sha256": hashlib.sha256((scene_dir / "manifest.json").read_bytes()).hexdigest(),
        "camera_interpolation": "relative-rotation SLERP-equivalent rotvec interpolation; linear camera-center interpolation; K from second endpoint",
        "renderer": f"finite RGB discs with far-to-near axial-Z ordering; point_radius={radius}; not 3DGS",
        "depth_colormap": "OpenCV TURBO RGB; near red / far blue; formal per-scene 2%-98% display range",
        "depth_display_range": near_far, "registration": "none; no ICP; no per-frame scale fitting",
        "novel_view_boundary": "Virtual-camera PCD projections, not captured images or independent ground truth. Holes may reflect disocclusion, geometry/pose mismatch or motion; no sparsity from point sampling.",
        "views": records,
    }
    write_json(output / "view_render_audit.json", report)
    print(f"DONE full nine views: {output}", flush=True)


if __name__ == "__main__":
    main()
