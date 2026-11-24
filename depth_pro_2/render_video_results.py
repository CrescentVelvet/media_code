#!/usr/bin/env python3
"""Render video depth, per-frame local-camera PCDs, and temporal diagnostics.

Env vars: SCENE_DIR, PREVIEW_COUNT (3), VIDEO_FPS (inference output_fps),
PCD_FRAME_STRIDE (30), RGB_FROM_SOURCE (0: lossless PNG, 1: decode source),
VIDEO_ONLY (0; 1 renders only the RGB/full-scene-depth video). PCD_FRAME_STRIDE
is optional and defaults to one PCD per 30th frame. PCDs have
unknown intrinsics and no known camera poses:
x/y are a pinhole back-projection using the official estimated focal; they are
never combined into a world-space cloud.
"""
import hashlib
import json
import os
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData

from render_geometry import depth_color, depth_to_world, save_ply, write_json


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = Path(os.environ["SCENE_DIR"])
    inference = json.loads((out / "inference.json").read_text(encoding="utf-8"))
    source = json.loads((out / "video_source.json").read_text(encoding="utf-8"))
    count = inference["processed_frames"]
    assert count == len(inference["frames"]) > 0
    assert "inference_seconds" in inference, "Inference must complete before rendering"
    fps = float(os.environ.get("VIDEO_FPS", inference["output_fps"]))
    preview_count = int(os.environ.get("PREVIEW_COUNT", "3"))
    preview_indices = [0, count // 2, count - 1] if preview_count == 3 else np.linspace(0, count - 1, preview_count).round().astype(int).tolist()
    assert len(set(preview_indices)) == preview_count
    (out / "pcd").mkdir(exist_ok=True)
    (out / "previews").mkdir(exist_ok=True)
    pcd_stride = int(os.environ.get("PCD_FRAME_STRIDE", "30"))
    video_only = os.environ.get("VIDEO_ONLY", "0") == "1"
    assert pcd_stride > 0
    # Include every requested comparison frame even if it misses the PCD stride.
    pcd_indices = [] if video_only else sorted(set(list(range(0, count, pcd_stride)) + preview_indices + [count - 1]))
    records = []
    for i in pcd_indices:
        rgb = np.asarray(Image.open(out / "frames" / f"{i:06d}.jpg").convert("RGB"))
        depth = np.load(out / "depth" / f"{i:06d}.npz")["depth"]
        focal = float(inference["frames"][i]["focal_px"])
        h, w = depth.shape
        # An estimated FOV is not calibrated K: cx/cy are explicitly assumed.
        K = np.array([[focal, 0, (w - 1) / 2], [0, focal, (h - 1) / 2], [0, 0, 1]], dtype=np.float64)
        c2w = np.eye(4)
        world, _ = depth_to_world(depth, K, c2w)
        valid = np.isfinite(depth.ravel()) & (depth.ravel() > 0.05)
        points = world[valid].astype(np.float32)
        colors = rgb.reshape(-1, 3)[valid]
        ply_path = out / "pcd" / f"{i:06d}.ply"
        save_ply(ply_path, points, colors, np.full(len(points), i % 256, np.uint8))
        vertices = PlyData.read(str(ply_path))["vertex"].data
        assert len(vertices) == len(points)
        assert np.array_equal(np.column_stack([vertices[k] for k in ("x", "y", "z")]), points)
        assert np.array_equal(np.column_stack([vertices[k] for k in ("red", "green", "blue")]), colors)
        assert np.all(vertices["source_index"] == i % 256)
        camera = {"K": K.tolist(), "w2c": np.eye(4).tolist(), "c2w": c2w.tolist()}
        splat, pcd_depth, coverage = rgb.copy(), depth.copy(), float(valid.mean())
        splat[~valid.reshape(depth.shape)] = 244
        near, far = np.quantile(depth[np.isfinite(depth) & (depth > 0)], [0.02, 0.98])
        Image.fromarray(depth_color(pcd_depth, near, far)).save(out / "previews" / f"{i:06d}_pcd_depth.png")
        Image.fromarray(splat).save(out / "previews" / f"{i:06d}_splat.png")
        Image.fromarray(depth_color(depth, near, far)).save(out / "previews" / f"{i:06d}_depth_rgb.png")
        error = np.abs(pcd_depth[np.isfinite(pcd_depth)] - depth[np.isfinite(pcd_depth)])
        assert float(np.max(error)) < 1e-4
        records.append({"index": i, "source_index": inference["frames"][i]["source_index"],
                        "point_count": len(points), "pcd_file": f"pcd/{i:06d}.ply", "ply_sha256": sha256(ply_path),
                        "camera": "local OpenCV; fx=fy=official FOV estimate; assumed principal point at image center",
                        "focal_px": focal, "near_far": [float(near), float(far)],
                        "coverage_zero_radius": coverage,
                        "pcd_world_registration": "none; do not concatenate as a registered world point cloud"})
        print(f"✅ local-camera PCD {i}: {len(points):,} points", flush=True)

    preview_frames = {}
    if os.environ.get("RGB_FROM_SOURCE", "0") == "1":
        cap = cv2.VideoCapture(source["path"])
        assert cap.isOpened()
        for i in preview_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, inference["frames"][i]["source_index"])
            ok, bgr = cap.read()
            assert ok
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rgb.shape[:2] != (inference["output_height"], inference["output_width"]):
                rgb = np.asarray(Image.fromarray(rgb).resize((inference["output_width"], inference["output_height"]), Image.Resampling.LANCZOS))
            preview_frames[i] = rgb
        cap.release()
    for i in preview_indices:
        rgb = preview_frames.get(i)
        if rgb is None:
            rgb = np.asarray(Image.open(out / "frames" / f"{i:06d}.jpg").convert("RGB"))
        depth = np.load(out / "depth" / f"{i:06d}.npz")["depth"]
        focal = float(inference["frames"][i]["focal_px"])
        near, far = np.quantile(depth[np.isfinite(depth) & (depth > 0)], [0.02, 0.98])
        Image.fromarray(depth_color(depth, near, far)).save(out / "previews" / f"{i:06d}_depth_rgb.png")
        panels = [rgb, depth_color(depth, near, far)]
        labels = ["Input RGB", "Depth Pro 2 | near red / far blue"]
        if (out / "pcd" / f"{i:06d}.ply").is_file():
            panels.extend([np.asarray(Image.open(out / "previews" / f"{i:06d}_splat.png")),
                           np.asarray(Image.open(out / "previews" / f"{i:06d}_pcd_depth.png"))])
            labels.extend(["Local-camera PCD splat", "PCD depth | unregistered"])
        h, w = depth.shape
        combined = Image.new("RGB", (w * len(panels), h + 42), "white")
        draw = ImageDraw.Draw(combined)
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
        for j, (panel, label) in enumerate(zip(panels, labels)):
            combined.paste(Image.fromarray(panel), (j * w, 42))
            draw.text((j * w + 12, 12), f"{label} | source frame {inference['frames'][i]['source_index']}", fill=(25, 30, 38), font=font)
        combined.save(out / "previews" / f"{i:06d}_comparison.png")
    quantile_depths = []
    for record in inference["frames"]:
        d = np.load(out / "depth" / f"{record['index']:06d}.npz")["depth"]
        quantile_depths.append(d[::8, ::8])
    pooled = np.concatenate([d.ravel() for d in quantile_depths])
    pooled = pooled[np.isfinite(pooled) & (pooled > 0)]
    near, far = np.quantile(pooled, [0.02, 0.995])
    near_far_near = [1.0, float(os.environ.get("NEAR_PREVIEW_FAR", "15"))]
    columns = 2 if video_only else 3
    raw_path = out / "depth_preview.avi"
    h, w = inference["output_height"], inference["output_width"]
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w * columns, h + 38))
    assert writer.isOpened()
    font = ImageFont.truetype("DejaVuSans.ttf", 17)
    try:
        for i, record in enumerate(inference["frames"]):
            rgb = np.asarray(Image.open(out / "frames" / f"{i:06d}.jpg").convert("RGB"))
            depth = np.load(out / "depth" / f"{i:06d}.npz")["depth"]
            colored = depth_color(depth, near, far)
            panels = [rgb, colored]
            if not video_only:
                near_colored = depth_color(depth, *near_far_near)
                panels.append(near_colored)
            combined = np.concatenate(panels, axis=1)
            banner = Image.new("RGB", (w * columns, 38), "white")
            draw = ImageDraw.Draw(banner)
            draw.text((10, 8), "Input RGB", fill=(25, 30, 38), font=font)
            draw.text((w + 10, 8), f"Full-scene depth | frame {record['source_index']} | near red / far blue", fill=(25, 30, 38), font=font)
            if not video_only:
                draw.text((2 * w + 10, 8), f"Near-depth view | clipped at {near_far_near[1]:g} m", fill=(25, 30, 38), font=font)
            writer.write(cv2.cvtColor(np.concatenate([np.asarray(banner), combined], axis=0), cv2.COLOR_RGB2BGR))
            if i % 60 == 0:
                print(f"✅ preview video {i}/{count}", flush=True)
    finally:
        writer.release()
    video_path = out / "rgb_depth_comparison.mp4"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(raw_path), "-an", "-c:v", "libx264",
                    "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video_path)], check=True)
    probe = cv2.VideoCapture(str(video_path))
    assert probe.isOpened() and int(probe.get(cv2.CAP_PROP_FRAME_COUNT)) == count
    assert abs(probe.get(cv2.CAP_PROP_FPS) - fps) < 0.01
    probe.release()
    write_json(out / "render_audit.json", {
        "video": {"file": "rgb_depth_comparison.mp4", "sha256": sha256(video_path), "frames": count,
                  "fps": fps, "near_far_shared": [float(near), float(far)], "columns": columns,
                  "near_preview_range_m": None if video_only else near_far_near, "near_red_far_blue": True,
                  "note": "VIDEO_ONLY has RGB and full-scene depth only. Otherwise the third panel clips display at the near-preview limit; raw NPZ depth is unchanged."},
        "previews": records,
        "boundary": "Qualitative temporal depth only. No GT, no calibrated K, no camera poses, no world-space registration; not a cross-view accuracy metric.",
    })
    print(f"🎉 Video and local PCD previews: {out}", flush=True)


if __name__ == "__main__":
    main()
