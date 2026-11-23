#!/usr/bin/env python3
"""Render one fixed 100k-point PCD sample in input/interpolated/extrapolated cameras.

The two virtual cameras use SLERP rotations and linear translations between frame
00 and frame 01. No model inference, ICP, per-frame scaling or new geometry is
introduced. PCD splat means finite RGB point discs, not a trained Gaussian model.

Env vars:
  SCENE_DIR: One formal scene directory containing manifest/rgb/depth.
  VIEW_POINT_COUNT: Points shared by all three views (default: 100000).
  VIEW_SEED: Reproducible sampling seed (default: 20261123).
  VIEW_EXTRAPOLATION: Fraction beyond frame 00 (default: -0.5).
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from render_geometry import depth_color, depth_to_world, render_splat, save_ply, write_json


def virtual_camera(first, second, fraction):
    """Relative rotation SLERP and translation lerp, preserving OpenCV convention."""
    r0 = np.asarray(first["c2w"], dtype=np.float64)[:3, :3]
    r1 = np.asarray(second["c2w"], dtype=np.float64)[:3, :3]
    c0 = np.asarray(first["c2w"], dtype=np.float64)[:3, 3]
    c1 = np.asarray(second["c2w"], dtype=np.float64)[:3, 3]
    relative = Rotation.from_matrix(r1 @ r0.T)
    # Rotvec interpolation is the same constant-angular-velocity geodesic as
    # SLERP, but unlike scipy.Slerp it also supports negative extrapolation.
    rotation = Rotation.from_rotvec(relative.as_rotvec() * fraction).as_matrix() @ r0
    center = (1.0 - fraction) * c0 + fraction * c1
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = rotation
    c2w[:3, 3] = center
    return c2w


def main():
    scene_dir = Path(os.environ["SCENE_DIR"])
    point_count = int(os.environ.get("VIEW_POINT_COUNT", "100000"))
    seed = int(os.environ.get("VIEW_SEED", "20261123"))
    extrapolation = float(os.environ.get("VIEW_EXTRAPOLATION", "-0.5"))
    if point_count <= 0 or not -1.0 <= extrapolation < 0.0:
        raise ValueError("Require positive point count and -1 <= extrapolation < 0")
    manifest = json.loads((scene_dir / "manifest.json").read_text(encoding="utf-8"))
    frames = manifest["frames"]
    if len(frames) != 3:
        raise ValueError("The defined view contract requires three input frames")
    scale_path = scene_dir / "sfm_scale.json"
    scale = float(json.loads(scale_path.read_text(encoding="utf-8"))["metric_to_sfm_scale"]) if scale_path.is_file() else 1.0
    near_far = json.loads((scene_dir / "geometry.json").read_text(encoding="utf-8"))["depth_display_range"]
    worlds, colors, depths = [], [], []
    for frame in frames:
        index = frame["index"]
        rgb = np.asarray(Image.open(scene_dir / "rgb" / f"{index:02d}.png").convert("RGB"))
        depth = np.load(scene_dir / "depth" / f"{index:02d}.npz")["depth"] * scale
        world, _ = depth_to_world(depth, np.asarray(frame["K"]), np.asarray(frame["c2w"]))
        far_cutoff = near_far[1] * 1.2
        valid = (depth.ravel() > 0.05) & np.isfinite(depth.ravel()) & (depth.ravel() < far_cutoff)
        worlds.append(world[valid])
        colors.append(rgb.reshape(-1, 3)[valid])
        depths.append(depth)
    full_world = np.concatenate(worlds)
    full_colors = np.concatenate(colors)
    if len(full_world) < point_count:
        raise ValueError(f"Only {len(full_world)} valid points, requested {point_count}")
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(full_world), point_count, replace=False))
    sampled_world = full_world[selected].astype(np.float32)
    sampled_colors = full_colors[selected].astype(np.uint8)
    sampled_sources = np.concatenate([
        np.full(len(points), index, dtype=np.uint8) for index, points in enumerate(worlds)
    ])[selected]
    output = scene_dir / "view_splat_depth_100k"
    output.mkdir(exist_ok=True)
    save_ply(output / "sampled_100k.ply", sampled_world, sampled_colors, sampled_sources)
    selected_bytes = np.asarray(selected, dtype=np.int64).tobytes()
    source_hashes = {
        f"depth_{frame['index']:02d}": hashlib.sha256(
            (scene_dir / "depth" / f"{frame['index']:02d}.npz").read_bytes()
        ).hexdigest() for frame in frames
    }
    cameras = [
        ("input", "Input camera 01", 1, None),
        ("interpolated", "SLERP t=0.5: camera 00 to 01", 0, 0.5),
        ("extrapolated", f"SLERP t={extrapolation:g}: beyond camera 00", 0, extrapolation),
    ]
    records = []
    for name, camera_description, reference_index, fraction in cameras:
        reference = frames[reference_index]
        c2w = np.asarray(reference["c2w"], dtype=np.float64) if fraction is None else virtual_camera(frames[0], frames[1], fraction)
        # Use the common input K in both virtual cameras, matching camera 01.
        camera = {
            "K": np.asarray(frames[1]["K"], dtype=np.float64),
            "w2c": np.linalg.inv(c2w),
        }
        h, w = depths[reference_index].shape
        splat, projected_depth, coverage = render_splat(sampled_world, sampled_colors, camera, (h, w), 0.8)
        Image.fromarray(splat).save(output / f"{name}_splat.png")
        Image.fromarray(depth_color(projected_depth, *near_far)).save(output / f"{name}_pcd_depth.png")
        np.save(output / f"{name}_axial_z.npy", projected_depth)
        record = {
            "view": name,
            "camera_description": camera_description,
        "interpolation": "SLERP-equivalent relative-rotation rotvec interpolation, including extrapolation",
            "is_real_camera": fraction is None,
            "reference_frame": reference_index,
            "interpolation_fraction_00_to_01": fraction,
            "c2w": c2w.tolist(),
            "K": camera["K"].tolist(),
            "image_width": w,
            "image_height": h,
            "disc_coverage": coverage,
            "source_counts": [int(np.sum(sampled_sources == i)) for i in range(3)],
            "files": [f"{name}_splat.png", f"{name}_pcd_depth.png", f"{name}_axial_z.npy"],
        }
        (output / f"{name}_camera.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
        )
        records.append(record)
        print(f"✅ {scene_dir.name}/{name}: coverage={coverage:.2%}")
    report = {
        "scene": manifest["scene"],
        "source": "formal depth NPZ and manifest cameras; fixed source-only geometry",
        "point_count": point_count,
        "seed": seed,
        "sampling": "uniform without replacement over the pooled valid three-frame PCD; same points for all views",
        "sampled_point_indices_sha256": hashlib.sha256(selected_bytes).hexdigest(),
        "sampled_point_ply": "sampled_100k.ply",
        "source_depth_npz_sha256": source_hashes,
        "manifest_sha256": hashlib.sha256((scene_dir / "manifest.json").read_bytes()).hexdigest(),
        "full_point_count_before_sampling": len(full_world),
        "camera_interpolation": "relative-rotation SLERP; linear camera-center interpolation; K from reference camera 01",
        "extrapolation_fraction_00_to_01": extrapolation,
        "renderer": "finite RGB discs with far-to-near axial-Z ordering; point_radius=0.8; not 3DGS",
        "depth_colormap": "OpenCV TURBO RGB; near red / far blue; formal per-scene 2%-98% display range",
        "depth_display_range": near_far,
        "registration": "none; no ICP; no per-frame scale fitting",
        "novel_view_boundary": "Virtual-camera PCD projections, not captured images or independent ground truth. Holes may reflect sparse sampling, disocclusion, geometry/pose mismatch or motion.",
        "views": records,
    }
    write_json(output / "view_render_audit.json", report)
    print(f"🎉 Saved {point_count:,} shared sampled points and three cameras: {output}")


if __name__ == "__main__":
    main()
