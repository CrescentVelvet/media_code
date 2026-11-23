#!/usr/bin/env python3
"""Audit native COLMAP tracks, pose conversion, and sparse-point depth scales.

Env vars: SCENE_DIR, POSE_DIR. This diagnostic never changes cameras or depths.
Track reprojection is independent of Depth Pro 2. Native observations are
converted into the resized RGB image. points3D.txt TRACK[] may contain zero
placeholders; valid images.txt POINT3D_ID links are used instead.
"""
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from prepare_data import qvec2rotmat


def stats(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "median": float(np.median(x)),
            "p90": float(np.quantile(x, 0.9)), "mean": float(np.mean(x))}


def read_sfm(pose_dir):
    cameras = {}
    for line in (pose_dir / "sparse/0/cameras.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.split()
        if f[1] != "PINHOLE":
            raise ValueError(f"Expected PINHOLE, got {f[1]}")
        cameras[int(f[0])] = {"size": [int(f[2]), int(f[3])],
                              "K": [[float(f[4]), 0, float(f[6])],
                                    [0, float(f[5]), float(f[7])], [0, 0, 1]]}
    lines = [line for line in (pose_dir / "sparse/0/images.txt").read_text().splitlines()
             if line and not line.startswith("#")]
    images = {}
    i = 0
    while i < len(lines):
        f = lines[i].split()
        if len(f) != 10:
            raise ValueError(f"Expected camera row at logical line {i}, got {len(f)} fields")
        obs = lines[i + 1].split() if i + 1 < len(lines) else []
        if len(obs) % 3:
            raise ValueError(f"Malformed POINTS2D row after image {f[0]}")
        images[int(f[0])] = {
            "R": qvec2rotmat(np.array(list(map(float, f[1:5])))),
            "t": np.array(list(map(float, f[5:8]))), "camera_id": int(f[8]),
            "name": f[9],
            "xy": np.asarray(list(map(float, obs)), dtype=np.float64).reshape(-1, 3)[:, :2]
            if obs else np.empty((0, 2)),
            "point3d_ids": np.asarray(obs[2::3], dtype=np.int64) if obs else np.empty(0, dtype=np.int64),
        }
        i += 2
    points = {}
    for line in (pose_dir / "sparse/0/points3D.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.split()
        points[int(f[0])] = {"xyz": np.array(list(map(float, f[1:4]))),
                             "error": float(f[7]),
                             "track": np.array(list(map(int, f[8:]))).reshape(-1, 2)}
    return cameras, images, points


def main():
    root = Path(os.environ["SCENE_DIR"])
    pose_dir = Path(os.environ["POSE_DIR"])
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    cameras, images, points = read_sfm(pose_dir)
    name_to_id = {v["name"]: k for k, v in images.items()}
    selected = {name_to_id[fr["name"]]: fr for fr in manifest["frames"]}
    track_index = defaultdict(list)
    depth_ratios = []
    for fr in manifest["frames"]:
        im = images[name_to_id[fr["name"]]]
        depth = np.load(root / "depth" / f"{fr['index']:02d}.npz")["depth"]
        roi = fr["undistort_roi"]
        for point2d_idx, pid in enumerate(im["point3d_ids"]):
            if pid < 0 or pid not in points:
                continue
            camera = im["R"] @ points[pid]["xyz"] + im["t"]
            uv = im["xy"][point2d_idx]
            pixel = np.round((uv - roi[:2]) * fr["resize_scale"]).astype(int)
            if camera[2] <= 0 or not (0 <= pixel[0] < fr["width"] and 0 <= pixel[1] < fr["height"]):
                continue
            ratio = float(depth[pixel[1], pixel[0]] / camera[2])
            if np.isfinite(ratio) and ratio > 0:
                depth_ratios.append({"frame": fr["index"], "point3d_id": int(pid),
                                     "observed_uv_native": uv.tolist(), "ratio": ratio})
    ratio_values = np.array([x["ratio"] for x in depth_ratios])
    group_ratio = float(np.median(ratio_values))
    group_scale = 1.0 / group_ratio
    residual = np.abs(ratio_values * group_scale - 1.0)
    (root / "sfm_scale.json").write_text(json.dumps({
        "method": "one pooled median ratio for all three frames; no ICP or per-frame scale fitting",
        "definition": "pcd_sfm_units = metric_camera_xyz / (depth_metric/sfm_z); cameras stay unchanged",
        "depth_metric_over_sfm_z": group_ratio, "metric_to_sfm_scale": group_scale,
        "observations": len(ratio_values),
        "relative_z_residual_after_group_scale": stats(residual),
        "caution": "sparse SfM is diagnostic, not independent ground truth; metric depth and arbitrary SfM camera units cannot be mixed without one shared scale",
    }, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(f"📐 One global SfM scale: {group_scale:.6f}; pooled relative Z median={np.median(residual):.4f}")
    for fr in manifest["frames"]:
        frame_ratios = np.array([x["ratio"] for x in depth_ratios if x["frame"] == fr["index"]])
        print(f"  frame {fr['index']}: n={len(frame_ratios)}, ratio median={np.median(frame_ratios):.6f}")
    track_index = defaultdict(list)
    all_errors = []
    rows = {i: {"name": im["name"], "selected": i in selected,
                "observations": len(im["xy"]), "errors": []} for i, im in images.items()}
    for image_id, im in images.items():
        for point2d_idx, pid in enumerate(im["point3d_ids"]):
            if pid < 0:
                continue
            if pid not in points:
                raise ValueError(f"POINTS2D links missing point3D {pid}")
            camera = im["R"] @ points[pid]["xyz"] + im["t"]
            K = np.asarray(cameras[im["camera_id"]]["K"])
            uv = camera[:2] / camera[2] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
            error = uv - im["xy"][point2d_idx]
            norm = float(np.linalg.norm(error))
            rows[image_id]["errors"].append(norm)
            all_errors.append(norm)
            if image_id in selected:
                track_index[image_id].append((pid, point2d_idx, error))
    out = {"source": str(pose_dir), "images": len(images), "points3d": len(points),
           "track_source": "images.txt POINTS2D POINT3D_ID; points3D TRACK[] (0,0) placeholders skipped",
           "all_images_track_reprojection_px": stats(all_errors),
           "image_summaries": [{"image_id": i, "name": row["name"], "selected": row["selected"],
                                "observations": row["observations"],
                                "track_reprojection_px": stats(row["errors"])}
                               for i, row in rows.items()],
           "selected_track_reprojection_px": {}, "frames": [],
           "caution": "COLMAP tracks validate camera-file self-consistency, not true geometry. Sparse SfM is not independent ground truth. One pooled scale is exported for PCD conversion; no per-frame fitting."}
    (root / "pose_check").mkdir(exist_ok=True)
    for image_id, fr in selected.items():
        im = images[image_id]
        records = track_index[image_id]
        errors = np.asarray([r[2] for r in records])
        if not len(records):
            print(f"⚠️ No linked observations: {im['name']}")
            continue
        rgb = np.asarray(Image.open(pose_dir / "image" / im["name"]).convert("RGB"))
        depth = np.load(root / "depth" / f"{fr['index']:02d}.npz")["depth"]
        ratio, z_sfm, z_dp, obs_scaled = [], [], [], []
        for pid, point2d_idx, error in records:
            point = points[pid]
            camera = im["R"] @ point["xyz"] + im["t"]
            uv = im["xy"][point2d_idx]
            roi = fr["undistort_roi"]
            scaled = (uv - roi[:2]) * fr["resize_scale"]
            pixel = np.round(scaled).astype(int)
            if camera[2] <= 0 or not (0 <= pixel[0] < fr["width"] and 0 <= pixel[1] < fr["height"]):
                continue
            dp = float(depth[pixel[1], pixel[0]])
            ratio.append(dp / camera[2])
            z_sfm.append(camera[2])
            z_dp.append(dp)
            obs_scaled.append(scaled)
        ratio = np.asarray(ratio)
        positive = ratio[np.isfinite(ratio) & (ratio > 0)]
        scale = float(np.median(positive)) if len(positive) else float("nan")
        rel = np.abs(np.asarray(z_dp) - scale * np.asarray(z_sfm)) / np.asarray(z_sfm)
        out["selected_track_reprojection_px"][im["name"]] = stats(np.linalg.norm(errors, axis=1))
        out["frames"].append({"name": im["name"], "depth_over_sfm_z": stats(ratio),
                              "single_scale_for_diagnostic": scale,
                              "relative_z_after_scale": stats(rel),
                              "manifest_matches_colmap_RT": bool(
                                  np.allclose(fr["w2c"][:3], np.column_stack((im["R"], im["t"])), atol=1e-6))})
        vis = Image.fromarray(rgb).resize((fr["width"], fr["height"]), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(vis)
        for pid, point2d_idx, error in records:
            uv = (im["xy"][point2d_idx] - fr["undistort_roi"][:2]) * fr["resize_scale"]
            x, y = uv
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=(30, 230, 70), width=2)
            end = uv + np.clip(error * fr["resize_scale"][0], -120, 120)
            draw.line((x, y, float(end[0]), float(end[1])), fill=(240, 30, 30), width=2)
        bar = Image.new("RGB", (fr["width"], 64), (250, 250, 250))
        d = ImageDraw.Draw(bar)
        d.text((10, 5), f"COLMAP observations / reprojection | {im['name']}",
               fill=(25, 30, 35), font=ImageFont.truetype("DejaVuSans.ttf", 17))
        d.text((10, 34), "green: observation; red: native SfM reprojection residual (clipped)",
               fill=(50, 55, 65), font=ImageFont.truetype("DejaVuSans.ttf", 13))
        vis = Image.fromarray(np.concatenate((np.asarray(bar), np.asarray(vis)), axis=0))
        vis.save(root / "pose_check" / f"colmap_track_{fr['index']:02d}.jpg", quality=94)
        print(f"🔍 {im['name']}: {len(records)} tracked observations, "
              f"reprojection median={out['selected_track_reprojection_px'][im['name']].get('median', float('nan')):.4f}px, "
              f"depth/SfM-Z median={scale:.4f}")
    (root / "colmap_audit.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                            encoding="utf-8", newline="\n")
    print(f"🎉 Native COLMAP audit: {root / 'colmap_audit.json'}")


if __name__ == "__main__":
    main()
