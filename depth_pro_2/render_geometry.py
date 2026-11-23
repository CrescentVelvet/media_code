#!/usr/bin/env python3
"""Lift native axial depths to world-space PCDs and render splat/depth images.

No ICP or per-frame scale fitting is used: all frames share the manifest's poses.
Splat radii are world-space-derived, not a Gaussian model or a trained 3DGS.
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData, PlyElement


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")


def depth_to_world(depth, K, c2w):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    xyz = np.stack(((u - K[0, 2]) / K[0, 0] * depth,
                    (v - K[1, 2]) / K[1, 1] * depth, depth), axis=-1)
    world = xyz.reshape(-1, 3) @ c2w[:3, :3].T + c2w[:3, 3]
    return world, xyz.reshape(-1, 3)


def render_splat(world, colors, fr, shape, point_radius=0.8):
    h, w = shape
    K, w2c = np.asarray(fr["K"]), np.asarray(fr["w2c"])
    camera = world @ w2c[:3, :3].T + w2c[:3, 3]
    z = camera[:, 2]
    u = K[0, 0] * camera[:, 0] / z + K[0, 2]
    v = K[1, 1] * camera[:, 1] / z + K[1, 2]
    radius = np.clip(point_radius * np.sqrt((K[0, 0] * K[1, 1]) / 2) / z, 0.6, 2.5)
    keep = (z > 0.02) & (u >= -radius) & (u < w + radius) & (v >= -radius) & (v < h + radius)
    camera, colors, u, v, radius, z = camera[keep], colors[keep], u[keep], v[keep], radius[keep], z[keep]
    order = np.argsort(z)[::-1]  # far to near painter's ordering
    depth_img = np.full((h, w), np.nan, np.float32)
    color_img = np.full((h, w, 3), 244, np.uint8)
    ix, iy = np.round(u).astype(np.int32), np.round(v).astype(np.int32)
    camera, colors, ix, iy, radius, z = camera[order], colors[order], ix[order], iy[order], radius[order], z[order]
    # Vectorized discs avoid millions of Python draw calls at native resolution.
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            disc_keep = (ix + dx >= 0) & (ix + dx < w) & (iy + dy >= 0) & (iy + dy < h) & (dx*dx + dy*dy <= radius*radius)
            flat = (iy[disc_keep] + dy) * w + ix[disc_keep] + dx
            depth_img.ravel()[flat] = z[disc_keep]
            color_img.reshape(-1, 3)[flat] = colors[disc_keep]
    return color_img, depth_img, float(np.isfinite(depth_img).mean())


def depth_color(depth, near, far, valid=None):
    """Official red-near / blue-far display; depth data itself is unchanged."""
    if valid is None:
        valid = np.isfinite(depth) & (depth > 0)
    x = np.clip((depth - near) / max(far - near, 1e-8), 0, 1)
    x[~valid] = 0.5
    color = cv2.applyColorMap(np.uint8(x * 255), cv2.COLORMAP_TURBO)
    color[~valid] = 244
    return color


def label(image, title, footer=""):
    image = np.asarray(image).copy()
    bar = np.full((62, image.shape[1], 3), 250, np.uint8)
    pil = Image.fromarray(bar)
    draw = ImageDraw.Draw(pil)
    font = ImageFont.truetype("DejaVuSans.ttf", 18)
    small = ImageFont.truetype("DejaVuSans.ttf", 12)
    draw.text((12, 7), title, fill=(25, 30, 38), font=font)
    draw.text((12, 37), footer, fill=(80, 86, 94), font=small)
    return np.concatenate((np.asarray(pil), image), axis=0)


def save_ply(path, points, colors, source_index):
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"),
             ("green", "u1"), ("blue", "u1"), ("source_index", "u1")]
    vertices = np.empty(len(points), dtype=dtype)
    for j, key in enumerate(("x", "y", "z")):
        vertices[key] = points[:, j]
    for j, key in enumerate(("red", "green", "blue")):
        vertices[key] = colors[:, j]
    vertices["source_index"] = source_index
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(str(path))


def main():
    scene_dir = Path(os.environ["SCENE_DIR"])
    out = scene_dir
    manifest = json.loads((scene_dir / "manifest.json").read_text(encoding="utf-8"))
    (out / "pcd").mkdir(exist_ok=True)
    (out / "splat").mkdir(exist_ok=True)
    (out / "pcd_depth").mkdir(exist_ok=True)
    (out / "depth_rgb").mkdir(exist_ok=True)
    (out / "pcd_reprojection").mkdir(exist_ok=True)
    (out / "video_frames").mkdir(exist_ok=True)
    frames = manifest["frames"]
    scale_path = scene_dir / "sfm_scale.json"
    depth_scale = 1.0
    if scale_path.is_file():
        depth_scale = float(json.loads(scale_path.read_text(encoding="utf-8"))["metric_to_sfm_scale"])
    print(f"📐 Geometry depth scale: {depth_scale:.8f}; poses unchanged")
    rgbs, depths, worlds, colors = [], [], [], []
    for fr in frames:
        i = fr["index"]
        rgb = np.asarray(Image.open(scene_dir / "rgb" / f"{i:02d}.png").convert("RGB"))
        depth = np.load(scene_dir / "depth" / f"{i:02d}.npz")["depth"] * depth_scale
        world, _ = depth_to_world(depth, np.asarray(fr["K"]), np.asarray(fr["c2w"]))
        rgbs.append(rgb)
        depths.append(depth)
        worlds.append(world.reshape(-1, 3))
        colors.append(rgb.reshape(-1, 3))
    all_depth = np.concatenate([d.ravel()[::8] for d in depths])
    near, far = np.quantile(all_depth[np.isfinite(all_depth) & (all_depth > 0)], [0.02, 0.98])
    full_world, full_colors = [], []
    for i, (fr, rgb, depth, world, color) in enumerate(zip(frames, rgbs, depths, worlds, colors)):
        valid = (depth.ravel() > 0.05) & np.isfinite(depth.ravel()) & (depth.ravel() < far * 1.2)
        pts = world[valid].astype(np.float32)
        cols = color[valid].astype(np.uint8)
        save_ply(out / "pcd" / f"{i:02d}.ply", pts, cols, np.full(len(pts), i, np.uint8))
        full_world.append(pts)
        full_colors.append(cols)
        h, w = depth.shape
        splat, pcd_depth, coverage = render_splat(world, color, fr, (h, w))
        Image.fromarray(splat).save(out / "splat" / f"{i:02d}.png")
        Image.fromarray(depth_color(depth, near, far)).save(out / "depth_rgb" / f"{i:02d}.png")
        Image.fromarray(depth_color(pcd_depth, near, far)).save(out / "pcd_depth" / f"{i:02d}.png")
        rendered, check_depth, reproj_cov = render_splat(world, color, fr, (h, w), 0)
        own_valid = np.isfinite(check_depth)
        if own_valid.sum() == 0 or not np.isfinite(depth[own_valid]).all():
            raise ValueError("Own-camera PCD round-trip has invalid samples")
        max_z_error = float(np.max(np.abs(check_depth[own_valid] - depth[own_valid])))
        if max_z_error > 1e-4:
            raise ValueError(f"Own-camera PCD round-trip failed: max axial Z error {max_z_error}")
        other_world = np.concatenate([worlds[j] for j in range(3) if j != i])
        other_colors = np.concatenate([colors[j] for j in range(3) if j != i])
        rendered, _, _ = render_splat(other_world, other_colors, fr, (h, w), 1.2)
        Image.fromarray(rendered).save(out / "pcd_reprojection" / f"{i:02d}.png")
        combined = np.concatenate([
            label(rgb, f"{manifest['scene']} frame {i} | RGB"),
            label(depth_color(depth, near, far), "Depth Pro 2 H+ | axial Z", f"near red / far blue; {near:.3f}..{far:.3f}"),
            label(splat, "PCD splat | original camera", f"point coverage {coverage:.1%}"),
            label(depth_color(pcd_depth, near, far), "PCD depth | original camera", "near red / far blue; same pose"),
        ], axis=1)
        Image.fromarray(combined).save(out / "video_frames" / f"{i:02d}.jpg", quality=93)
        print(f"✅ frame {i}: {len(pts):,} PCD points, splat coverage={coverage:.1%}")
    save_ply(out / "fused_no_registration.ply", np.concatenate(full_world),
             np.concatenate(full_colors), np.concatenate([np.full(len(x), i, np.uint8)
                                                          for i, x in enumerate(full_world)]))
    write_json(out / "geometry.json", {
        "coordinate": "OpenCV world from manifest; c2w=inv(w2c)",
        "depth": "native axial Z converted into manifest camera units",
        "depth_metric_over_sfm_z": 1.0 / depth_scale if scale_path.is_file() else None,
        "metric_to_camera_depth_scale": depth_scale,
        "registration": "none; no ICP, no per-frame scale fitting",
        "scale_source": "one pooled sparse-SfM depth scale" if scale_path.is_file() else "original camera units",
        "depth_display_range": [float(near), float(far)],
        "depth_colormap": "OpenCV TURBO RGB; red near, blue far; shared 2%-98% scene range",
        "splat": "finite world-radius point splats with far-to-near ordering; not 3D Gaussian Splatting",
        "point_count": [len(p) for p in full_world],
        "pcd_reprojection": "other two frames rendered into each original camera; not novel-view ground truth",
    })
    print(f"🎉 Geometry and video frames: {out}")


if __name__ == "__main__":
    main()
