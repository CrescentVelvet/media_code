#!/usr/bin/env python3
"""Render world-registered continuous-camera DP2 depths and export full PCDs.

Env vars: SCENE_DIR, PCD_PIXEL_STRIDE (2), VIDEO_FPS (12),
NEIGHBORHOOD_RADIUS (8), PCD_POINT_RADIUS (0.65), FULL_PLY (1),
METRIC_STRIDE (8). No ICP, per-frame scale fit, voxel averaging or optical flow.
PCD splat is RGB-disc projection, not trained 3DGS. Camera order supplies the
sequence; supplied c2w supplies registration only after depth inference.
"""
import json
import os
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData

from io_utils import sha256, write_json
from render_geometry import depth_color, depth_to_world, render_splat, save_ply


def font(size):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def encode(raw, destination, fps, count):
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(raw), "-an",
                    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(destination)], check=True)
    probe = cv2.VideoCapture(str(destination))
    assert probe.isOpened() and int(probe.get(cv2.CAP_PROP_FRAME_COUNT)) == count
    assert abs(probe.get(cv2.CAP_PROP_FPS) - fps) < .01
    probe.release()


def z_stats(values):
    if not len(values):
        return {"n": 0}
    return {"n": int(len(values)), "median": float(np.median(values)),
            "p90": float(np.quantile(values, .9)),
            "fraction_below_4pct": float(np.mean(values < .04))}


def main():
    out = Path(os.environ["SCENE_DIR"])
    stride = int(os.environ.get("PCD_PIXEL_STRIDE", "2"))
    fps = float(os.environ.get("VIDEO_FPS", "12"))
    neighborhood = int(os.environ.get("NEIGHBORHOOD_RADIUS", "8"))
    radius = float(os.environ.get("PCD_POINT_RADIUS", "0.65"))
    metric_stride = int(os.environ.get("METRIC_STRIDE", "8"))
    assert stride > 0 and neighborhood >= 0 and metric_stride > 0
    manifest = json.loads((out / "input/manifest.json").read_text(encoding="utf-8"))
    inference = json.loads((out / "inference.json").read_text(encoding="utf-8"))
    count = manifest["frame_count"]
    assert inference["processed_frames"] == count == len(inference["frames"])
    assert "inference_seconds" in inference
    for name in ("rgb_depth_frames", "pcd_frames", "previews", "pcd", "metrics"):
        assert not (out / name).is_symlink()
        (out / name).mkdir(exist_ok=True)
    h, w = inference["output_height"], inference["output_width"]
    rgbs = [np.asarray(Image.open(out / "input/frames" / f"{i:06d}.jpg").convert("RGB")) for i in range(count)]
    depths, worlds, colors, points, point_colors = [], [], [], [], []
    pooled = []
    sampled_depths = []
    for i, frame in enumerate(manifest["frames"]):
        depth = np.load(out / "depth" / f"{i:06d}.npz")["depth"]
        depths.append(depth)
        sampled_depths.append(depth[::stride, ::stride].copy())
    for i, frame in enumerate(manifest["frames"]):
        depth = depths[i]
        world, local = depth_to_world(depth, np.asarray(frame["K"]), np.asarray(frame["c2w"]))
        del local
        worlds.append(world[::stride].copy())
        colors.append(rgbs[i].reshape(-1, 3)[::stride].copy())
        valid = np.isfinite(depth.ravel()) & (depth.ravel() > .05)
        xyz = world[valid].astype(np.float32)
        color = rgbs[i].reshape(-1, 3)[valid]
        points.append(xyz)
        point_colors.append(color)
        pooled.append(depth[::8, ::8].ravel())
        ply_path = out / "pcd" / f"{i:06d}.ply"
        save_ply(ply_path, xyz, color, np.full(len(xyz), i, np.uint8))
        vertices = PlyData.read(str(ply_path))["vertex"].data
        assert len(vertices) == len(xyz)
        assert np.array_equal(np.column_stack([vertices[k] for k in ("x", "y", "z")]), xyz)
        assert np.array_equal(np.column_stack([vertices[k] for k in ("red", "green", "blue")]), color)
        print(f"✅ full per-frame PCD {i + 1}/{count}: {len(xyz):,}", flush=True)
    pooled = np.concatenate(pooled)
    near, far = np.quantile(pooled[np.isfinite(pooled) & (pooled > 0)], [.02, .98])
    if os.environ.get("FULL_PLY", "1") == "1":
        fused = out / "fused_all_82_no_registration.ply"
        if fused.is_file() and fused.stat().st_size > 1000000:
            print(f"⏭️ Keep existing full PLY; validate source and bytes: {fused}", flush=True)
        else:
            save_ply(fused, np.concatenate(points), np.concatenate(point_colors),
                     np.concatenate([np.full(len(p), i, np.uint8) for i, p in enumerate(points)]))
        fused_sha = sha256(fused)
    else:
        fused_sha = None
    all_world = np.concatenate(worlds)
    all_colors = np.concatenate(colors)
    all_sources = np.concatenate([np.full(len(p), i, np.uint8) for i, p in enumerate(worlds)])
    report = {
        "frames": count, "image_width": w, "image_height": h, "fps": fps,
        "fps_note": "Ordered still-camera playback; capture timestamps unavailable",
        "coordinate": "OpenCV world using supplied c2w after OpenGL conversion",
        "pose_conditioning": False, "registration_optimization": "none; no ICP, per-frame scale or depth correction",
        "per_frame_pcd": {"sampling": "none; every finite Z > 0.05 m", "counts": [len(p) for p in points]},
        "fused_ply": {"file": "fused_all_82_no_registration.ply", "sha256": fused_sha,
                      "points": int(sum(map(len, points))), "voxel_merge": False},
        "display_pcd_pixel_stride": stride,
        "source_pixel_centers": "Original u,v = 0,2,...; mapped to sampled grid by (u+1)/2-1",
        "neighborhood_radius_frames": neighborhood,
        "pcd_point_radius_px_at_unit_depth": radius,
        "depth_display_range_m": [float(near), float(far)],
        "depth_color": "OpenCV TURBO RGB; near red / far blue; shared sequence range",
        "splat": "Finite RGB discs sorted far-to-near by axial Z; not trained 3DGS",
        "videos": [], "views": [], "near_pairs": [],
        "boundary": "Qualitative registered depth visualization. Occlusion, disocclusion, motion and depth errors remain; disc coverage is not geometric accuracy.",
    }
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    rgb_raw = out / "rgb_depth_preview.avi"
    pcd_raw = out / "pcd_preview.avi"
    rgb_writer = cv2.VideoWriter(str(rgb_raw), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w * 2, h + 38))
    pcd_writer = cv2.VideoWriter(str(pcd_raw), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w * 2, h + 38))
    assert rgb_writer.isOpened() and pcd_writer.isOpened()
    title_font = font(17)
    try:
        for i, frame in enumerate(manifest["frames"]):
            depth_rgb = depth_color(depths[i], near, far)
            banner = Image.new("RGB", (2 * w, 38), "white")
            draw = ImageDraw.Draw(banner)
            draw.text((10, 8), f"Input RGB | ordered camera {i:02d}/{count - 1}", fill=(25, 30, 38), font=title_font)
            draw.text((w + 10, 8), "Depth Pro 2 | near red / far blue", fill=(25, 30, 38), font=title_font)
            rgb_frame = np.concatenate([np.asarray(banner), np.concatenate([rgbs[i], depth_rgb], axis=1)], axis=0)
            rgb_writer.write(cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
            selected = np.abs(all_sources.astype(np.int16) - i) <= neighborhood
            splat, projected, coverage = render_splat(all_world[selected], all_colors[selected],
                                                     frame, (h, w), radius)
            pcd_rgb = depth_color(projected, near, far)
            banner = Image.new("RGB", (2 * w, 38), "white")
            draw = ImageDraw.Draw(banner)
            used = int(selected.sum())
            draw.text((10, 8), f"PCD splat | neighborhood {max(0, i-neighborhood)}..{min(count-1, i+neighborhood)}",
                      fill=(25, 30, 38), font=title_font)
            draw.text((w + 10, 8), f"PCD depth | target camera {i:02d}", fill=(25, 30, 38), font=title_font)
            pcd_frame = np.concatenate([np.asarray(banner), np.concatenate([splat, pcd_rgb], axis=1)], axis=0)
            pcd_writer.write(cv2.cvtColor(pcd_frame, cv2.COLOR_RGB2BGR))
            if i in (0, 10, 20, 41, 55, 70, 81):
                Image.fromarray(splat).save(out / "previews" / f"{i:06d}_splat.png")
                Image.fromarray(pcd_rgb).save(out / "previews" / f"{i:06d}_pcd_depth.png")
                Image.fromarray(depth_rgb).save(out / "previews" / f"{i:06d}_depth_rgb.png")
                Image.fromarray(rgbs[i]).save(out / "previews" / f"{i:06d}_rgb.png")
                Image.fromarray(rgb_frame).save(out / "previews" / f"{i:06d}_rgb_depth.jpg", quality=94)
                Image.fromarray(pcd_frame).save(out / "previews" / f"{i:06d}_pcd.jpg", quality=94)
            report["views"].append({"index": i, "neighborhood_frames": int(selected.sum() // (h * w // stride)),
                                    "subsampled_points": used, "disc_coverage": coverage})
            if i < count - 1:
                # Ordered source-to-target diagnostics, without NCC/visibility filtering.
                target = i + 1
                f = manifest["frames"][target]
                camera = worlds[i] @ np.asarray(f["w2c"])[:3, :3].T + np.asarray(f["w2c"])[:3, 3]
                z = camera[:, 2]
                K = np.asarray(f["K"])
                u = np.round((K[0, 0] * camera[:, 0] / z + K[0, 2] + .5) / stride - .5).astype(np.int32)
                v = np.round((K[1, 1] * camera[:, 1] / z + K[1, 2] + .5) / stride - .5).astype(np.int32)
                inside = (z > .02) & (u >= 0) & (u < sampled_depths[target].shape[1]) & (v >= 0) & (v < sampled_depths[target].shape[0])
                ids = np.flatnonzero(inside)[::metric_stride]
                target_z = sampled_depths[target][v[ids], u[ids]]
                finite = np.isfinite(target_z) & (target_z > 0)
                relative = np.abs(z[ids][finite] - target_z[finite]) / target_z[finite]
                report["near_pairs"].append({"source": i, "target": target, "stride": metric_stride,
                                             "projected_samples": int(inside.sum()), "all_projected": z_stats(relative),
                                             "occlusion_aware": False,
                                             "caution": "Sampled-grid source-to-target projection; includes disocclusion, motion and depth-edge outliers; not GT accuracy."})
            print(f"✅ render {i + 1}/{count}: nearby frames={report['views'][-1]['neighborhood_frames']}, coverage={coverage:.1%}", flush=True)
    finally:
        rgb_writer.release()
        pcd_writer.release()
    rgb_video, pcd_video = out / "rgb_depth_comparison.mp4", out / "pcd_splat_depth_comparison.mp4"
    encode(rgb_raw, rgb_video, fps, count)
    encode(pcd_raw, pcd_video, fps, count)
    report["videos"] = [{"file": p.name, "sha256": sha256(p), "frames": count, "fps": fps,
                         "width": w * 2, "height": h + 38} for p in (rgb_video, pcd_video)]
    # Explicit full-arc versus old three-camera lifting, on identical sequence predictions.
    report["three_vs_82"] = []
    comparison_dir = out / "three_vs_82"
    comparison_dir.mkdir(exist_ok=True)
    sparse_indices = [0, 41, 81]
    sparse_mask = np.isin(all_sources, sparse_indices)
    target = manifest["frames"][41]
    target_h, target_w = depths[41].shape
    small_w, small_h = 576, 768
    small_camera = {"K": np.asarray(target["K"]) * (small_w / target_w), "c2w": target["c2w"], "w2c": target["w2c"]}
    small_camera["K"][2, :] = [0, 0, 1]
    panels = []
    for index, (label, mask) in enumerate((("Same sequence depth | 3 cameras (00,41,81)", sparse_mask),
                        ("Same sequence depth | all 82 cameras", np.ones(len(all_sources), bool)))):
        splat, projected, coverage = render_splat(all_world[mask], all_colors[mask], target, (target_h, target_w), radius)
        small, small_depth, _ = render_splat(all_world[mask], all_colors[mask], small_camera, (small_h, small_w), radius)
        Image.fromarray(small).save(comparison_dir / f"{'three' if len(panels) == 0 else 'dense'}_splat.png")
        Image.fromarray(depth_color(small_depth, near, far)).save(comparison_dir / f"{'three' if len(panels) == 0 else 'dense'}_pcd_depth.png")
        bar = Image.new("RGB", (small_w, 62), (248, 249, 250))
        draw = ImageDraw.Draw(bar)
        draw.text((10, 6), label, fill=(25, 30, 38), font=font(17))
        draw.text((10, 36), f"target camera 41 | disc coverage {coverage:.1%} | no ICP / scale fit",
                  fill=(75, 83, 92), font=font(13))
        panels.append(np.concatenate([np.asarray(bar), small], axis=0))
        report["three_vs_82"].append({"cameras": 3 if index == 0 else 82,
                                      "target": 41, "disc_coverage": coverage,
                                      "boundary": "Same buffered depth, only fusion membership changes; not a reproduction of old T=3 inference."})
        print(f"🔍 {label}: coverage={coverage:.2%}", flush=True)
    Image.fromarray(np.concatenate(panels, axis=1)).save(comparison_dir / "same_depth_three_vs_82.jpg", quality=95)
    write_json(out / "render_audit.json", report)
    print(f"🎉 Sequence videos, PCDs and comparison: {out}", flush=True)


if __name__ == "__main__":
    main()
