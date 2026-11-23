#!/usr/bin/env python3
"""Measure raw cross-frame geometry consistency without optimizing any transform.

Env var: SCENE_DIR. Reprojection samples are sparse but deterministic (stride=4).
Photometric gating only defines a subset; it does not establish pixel-wise rigidity.
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData
from scipy.spatial import cKDTree

from render_geometry import depth_to_world


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0}
    return {"n": len(values), "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.9)),
            "mean": float(values.mean()), "fraction_below_3pct": float((values < 0.03).mean())}


def ncc_patch(gray0, gray1, y0, x0, y1, x1):
    # Point correspondences are integer-rounded; no optical flow or learned tracker.
    a = gray0[y0[:, None, None] + np.arange(-2, 3)[None, :, None],
              x0[:, None, None] + np.arange(-2, 3)[None, None, :]].astype(np.float32)
    b = gray1[y1[:, None, None] + np.arange(-2, 3)[None, :, None],
              x1[:, None, None] + np.arange(-2, 3)[None, None, :]].astype(np.float32)
    a -= a.mean(axis=(1, 2), keepdims=True)
    b -= b.mean(axis=(1, 2), keepdims=True)
    denom = np.sqrt((a*a).sum(axis=(1, 2)) * (b*b).sum(axis=(1, 2)))
    return (a*b).sum(axis=(1, 2)) / np.maximum(denom, 1e-8), denom > 30


def smooth_mask(depth):
    med = cv2.medianBlur(depth.astype(np.float32), 3)
    return np.abs(depth - med) / np.maximum(med, 1e-8) < 0.02


def main():
    root = Path(os.environ["SCENE_DIR"])
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    frames = manifest["frames"]
    rgb = [np.asarray(Image.open(root / "rgb" / f"{i:02d}.png").convert("RGB")) for i in range(3)]
    depth = [np.load(root / "depth" / f"{i:02d}.npz")["depth"] for i in range(3)]
    scale_path = root / "sfm_scale.json"
    if scale_path.is_file():
        scale = float(json.loads(scale_path.read_text(encoding="utf-8"))["metric_to_sfm_scale"])
        depth = [d * scale for d in depth]
    gray = [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).astype(np.float32) for x in rgb]
    world = [depth_to_world(depth[i], np.asarray(frames[i]["K"]), np.asarray(frames[i]["c2w"]))[0]
             for i in range(3)]
    smooth = [smooth_mask(d) for d in depth]
    metrics = {"registration": "none", "geometry_depth_scale": (
        json.loads(scale_path.read_text(encoding="utf-8"))["metric_to_sfm_scale"]
        if scale_path.is_file() else 1.0), "reprojection": {
        "sample_stride_px": 4, "depth_agreement_relative_threshold": 0.04,
        "photometric_ncc_threshold": 0.8, "smooth_depth_relative_threshold": 0.02,
        "definition": "abs(source_z-target_z)/target_z for projected samples inside the target image",
        "error_gate_policy": "Primary error statistics are not truncated by the 4% depth-agreement threshold",
        "occlusion": "No occlusion or z-buffer visibility test; finite target depth is not a visibility proof",
        "caution": "appearance-stable subset is not proven rigid; moving objects, occlusion and reflections can pass NCC",
    }, "pairs": []}
    (root / "consistency").mkdir(exist_ok=True)
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            h, w = depth[i].shape
            y, x = np.mgrid[2:h-2:4, 2:w-2:4].astype(np.int32)
            y, x = y.ravel(), x.ravel()
            points = world[i][y * w + x]
            w2c = np.asarray(frames[j]["w2c"])
            K = np.asarray(frames[j]["K"])
            camera = points @ w2c[:3, :3].T + w2c[:3, 3]
            z = camera[:, 2]
            u = np.round(K[0, 0] * camera[:, 0] / z + K[0, 2]).astype(np.int32)
            v = np.round(K[1, 1] * camera[:, 1] / z + K[1, 2]).astype(np.int32)
            valid = (z > 0.02) & (u >= 2) & (u < w-2) & (v >= 2) & (v < h-2)
            ids = np.flatnonzero(valid)
            u, v, z, ids = u[valid], v[valid], z[valid], ids
            y, x = y[ids], x[ids]
            target_z = depth[j][v, u]
            finite_target = np.isfinite(target_z) & (target_z > 0)
            relative = np.abs(z-target_z) / np.maximum(target_z, 1e-8)
            visible = finite_target & (relative < 0.04)  # depth agreement, not occlusion visibility
            ncc, textured = ncc_patch(gray[i], gray[j], y, x, v, u)
            shared_smooth = smooth[i][y, x] & smooth[j][v, u]
            row = {"source": i, "target": j, "projected_samples": int(len(z)),
                   "finite_target_n": int(finite_target.sum()),
                   "depth_consistent_n": int(visible.sum()),
                   "depth_consistent_fraction": float(visible.mean()),
                   "all_projected_relative_depth": stats(relative[finite_target]),
                   "smooth_projected_relative_depth": stats(relative[finite_target & shared_smooth]),
                   "appearance_projected_relative_depth": stats(
                       relative[finite_target & shared_smooth & textured & (ncc > 0.8)]),
                   "appearance_stable_n": int((finite_target & shared_smooth & textured & (ncc > 0.8)).sum())}
            metrics["pairs"].append(row)
            error = np.full((h, w), np.nan, np.float32)
            use = finite_target & shared_smooth & textured & (ncc > 0.8)
            error[y[use], x[use]] = relative[use]
            heat = np.full((h, w, 3), 245, np.uint8)
            valid_error = np.isfinite(error)
            if valid_error.any():
                heat[valid_error] = cv2.applyColorMap(
                    np.uint8(np.clip(error[valid_error] / 0.10, 0, 1) * 255),
                    cv2.COLORMAP_TURBO)[:, 0, ::-1]
            bar = Image.new("RGB", (w, 55), (250, 250, 250))
            d = ImageDraw.Draw(bar)
            font = ImageFont.truetype("DejaVuSans.ttf", 16)
            d.text((10, 6), f"Reprojection {i} -> {j}: relative Z error", fill=(20, 25, 30), font=font)
            d.text((10, 31), "smooth + textured NCC>0.8; unfiltered Z error; blank=no sample; clip 10%",
                   fill=(70, 75, 85), font=ImageFont.truetype("DejaVuSans.ttf", 12))
            Image.fromarray(np.concatenate((np.asarray(bar), heat), axis=0)).save(
                root / "consistency" / f"{i:02d}_to_{j:02d}.png")
            print(f"🔍 {i}->{j}: projected={len(z)}, consistent_fraction={visible.mean():.3f}, "
                  f"appearance_stable_n={row['appearance_stable_n']}, "
                  f"untruncated_median={row['appearance_projected_relative_depth'].get('median', float('nan')):.4f}")
    # Unfiltered, asymmetrically sampled symmetric nearest-neighbor distances.
    combined = np.concatenate(world)
    sample = combined[::8]
    nn = []
    for i in range(3):
        points = world[i][::8]
        # Build one KD-tree per other frame to exclude self matches explicitly.
        distances = []
        for j in range(3):
            if i == j:
                continue
            other_tree = cKDTree(world[j][::8])
            distances.append(other_tree.query(points, k=1)[0])
        nearest = np.minimum(*distances)
        nn.append(nearest)
    for i, distances in enumerate(nn):
        median_depth = float(np.median(depth[i]))
        metrics.setdefault("nearest_neighbor", {})[str(i)] = {
            "sampling_stride": 8, "unfiltered": True,
            "distance_median": float(np.median(distances)),
            "distance_p90": float(np.quantile(distances, 0.9)),
            "median_depth_scale": median_depth,
            "median_over_depth_scale": float(np.median(distances) / median_depth),
            "p90_over_depth_scale": float(np.quantile(distances, 0.9) / median_depth),
            "caution": "no common-visibility filtering; sensitive to point density and depth-edge outliers",
        }
    reference = manifest.get("reference_pcd")
    if reference and Path(reference).is_file():
        v = PlyData.read(reference)["vertex"]
        pts = np.stack([v["x"], v["y"], v["z"]], axis=1)
        visible = np.zeros(len(pts), bool)
        for fr in frames:
            camera = pts @ np.asarray(fr["w2c"])[:3, :3].T + np.asarray(fr["w2c"])[:3, 3]
            K = np.asarray(fr["K"])
            u = K[0, 0] * camera[:, 0] / camera[:, 2] + K[0, 2]
            v = K[1, 1] * camera[:, 1] / camera[:, 2] + K[1, 2]
            visible |= ((camera[:, 2] > 0) & (u >= 0) & (u < fr["width"])
                        & (v >= 0) & (v < fr["height"]))
        distances = cKDTree(sample).query(pts[visible], k=1)[0]
        metrics["capture_reference_diagnostic"] = {
            "source": reference, "visible_in_any_selected_frame": int(visible.sum()),
            "nn_distance_median_m": float(np.median(distances)),
            "nn_distance_p90_m": float(np.quantile(distances, 0.9)),
            "caution": "capture PCD is not independent ground truth; no occlusion or static-region filtering",
        }
    (root / "consistency.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                                          encoding="utf-8", newline="\n")
    print(f"🎉 Consistency metrics: {root / 'consistency.json'}")


if __name__ == "__main__":
    main()
