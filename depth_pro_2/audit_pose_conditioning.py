#!/usr/bin/env python3
"""Audit supplied Remy poses and compare local-only versus world-space lifting.

This does not feed extrinsics into Depth Pro 2 or rerun inference. Its A/B uses
exactly the same cached metric depth: local clouds are a deliberately misregistered
counterfactual, not a historical baseline. The existing PLYs must be reproduced.

Env vars: SCENE_DIR, RAW_DIR, OUT_DIR.
"""
import hashlib
import inspect
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from render_geometry import depth_to_world, render_splat, save_ply, write_json


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_ply(path):
    data = PlyData.read(path)["vertex"]
    return (np.stack([data[key] for key in ("x", "y", "z")], axis=1),
            np.stack([data[key] for key in ("red", "green", "blue")], axis=1))


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0}
    return {"n": int(len(values)), "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.9)),
            "fraction_below_4pct": float(np.mean(values < 0.04))}


def print_metric_summary(identity, posed):
    print("📊 Same cached joint depth: ignoring poses -> supplied poses")
    for source in range(3):
        local_nn = identity["crossframe_nearest_neighbor"][source]
        world_nn = posed["crossframe_nearest_neighbor"][source]
        print(f"  frame {source} cross-frame NN median: {local_nn['median_m']:.4f} -> "
              f"{world_nn['median_m']:.4f} m (unfiltered; not geometric error)")
    for local, world in zip(identity["ordered_projected_relative_z"], posed["ordered_projected_relative_z"]):
        pair = f"{local['source']}->{local['target']}"
        if not local["projected_n"]:
            print(f"  {pair}: no projected samples without poses -> "
                  f"{world['projected_n']:,} with poses; Z median={world['all_projected']['median']:.2%}")
        else:
            print(f"  {pair}: Z median={local['all_projected']['median']:.2%} -> "
                  f"{world['all_projected']['median']:.2%}; "
                  f"<4%={local['all_projected']['fraction_below_4pct']:.2%} -> "
                  f"{world['all_projected']['fraction_below_4pct']:.2%}")


def project(points, frame):
    w2c, K = np.asarray(frame["w2c"]), np.asarray(frame["K"])
    camera = points @ w2c[:3, :3].T + w2c[:3, 3]
    z = camera[:, 2]
    u = np.round(K[0, 0] * camera[:, 0] / z + K[0, 2]).astype(np.int32)
    v = np.round(K[1, 1] * camera[:, 1] / z + K[1, 2]).astype(np.int32)
    return camera, u, v


def main():
    scene_dir = Path(os.environ["SCENE_DIR"])
    raw_dir = Path(os.environ["RAW_DIR"])
    out = Path(os.environ["OUT_DIR"])
    if out.resolve() == scene_dir.resolve():
        raise ValueError("Pose audit must not overwrite the original scene")
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((scene_dir / "manifest.json").read_text(encoding="utf-8"))
    transforms_path = raw_dir / "transforms.json"
    metadata = json.loads(transforms_path.read_text(encoding="utf-8"))
    source_hashes = {}
    for source_frame in metadata["frames"]:
        name = Path(source_frame["file_path"]).name
        source_hashes[name] = sha256(raw_dir / source_frame["file_path"])
    records = {Path(frame["file_path"]).name: frame for frame in metadata["frames"]}
    if len(records) != 3:
        raise ValueError(f"Expected three supplied poses, got {len(records)}")
    conversion = np.diag([1.0, -1.0, -1.0, 1.0])
    frames, depths, rgbs, local_clouds, world_clouds = [], [], [], [], []
    audit = {
        "model_pose_conditioning": {
            "supported": False,
            "official_revision": json.loads((scene_dir / "inference.json").read_text())["official_revision"],
            "infer_signature": "infer(x, f_px=None)",
            "reason": "Only RGB and a clip focal length enter inference. No extrinsic, camera ray, or relative-pose argument is accepted.",
            "experiment_scope": "Cached joint three-frame metric depth plus post-inference camera transforms; no model weight or architecture changes.",
        },
        "input": {"scene_dir": str(scene_dir), "raw_dir": str(raw_dir),
                  "transforms_json": str(transforms_path), "transforms_sha256": sha256(transforms_path),
                  "source_rgb_sha256": source_hashes},
        "coordinate_contract": "Supplied transform_matrix is Remy c2w OpenGL; c2w_cv = transform_matrix @ diag(1,-1,-1,1). Lift axial Z using K, then X_world = R_c2w_cv @ X_camera + t_c2w_cv.",
        "important": "The original joint and independent-single-frame PLY exports already apply these c2w poses. They are not camera-local-only PLYs.",
        "counterfactual": "Local-only A means ignoring all supplied extrinsics. It is not the existing export and is not recommended as a fusion result.",
        "conclusion": "Correct extrinsics are necessary for a shared world, but this exact post-inference transform is already in the formal PLYs. Reapplying it cannot further reduce the observed depth/motion/occlusion mismatch. DP2 does not condition its prediction on these poses.",
        "frames": [], "existing_ply_reproduction": [],
    }
    for index, old in enumerate(manifest["frames"]):
        name = old["name"]
        if name not in records:
            raise ValueError(f"Missing supplied pose for {name}")
        if source_hashes[name] != old["source_sha256"]:
            raise ValueError(f"Source RGB identity differs: {name}")
        c2w = np.asarray(records[name]["transform_matrix"], dtype=np.float64) @ conversion
        saved = np.asarray(old["c2w"], dtype=np.float64)
        if not np.allclose(c2w, saved, rtol=0, atol=1e-12):
            raise ValueError(f"Supplied pose differs from saved manifest for frame {index}")
        R, center = c2w[:3, :3], c2w[:3, 3]
        orthonormal_error = float(np.max(np.abs(R.T @ R - np.eye(3))))
        if orthonormal_error > 1e-5 or not np.isclose(np.linalg.det(R), 1, atol=1e-5):
            raise ValueError(f"Invalid camera rotation for frame {index}")
        if not np.allclose(np.linalg.inv(c2w), np.asarray(old["w2c"]), rtol=0, atol=1e-10):
            raise ValueError(f"Camera inverse mismatch for frame {index}")
        frame = {"K": np.asarray(old["K"], dtype=np.float64), "c2w": c2w,
                 "w2c": np.linalg.inv(c2w), "width": old["width"], "height": old["height"]}
        frames.append(frame)
        depth = np.load(scene_dir / "depth" / f"{index:02d}.npz")["depth"].astype(np.float32)
        rgb = np.asarray(Image.open(scene_dir / "rgb" / f"{index:02d}.png").convert("RGB"))
        world, local = depth_to_world(depth, frame["K"], c2w)
        depths.append(depth)
        rgbs.append(rgb)
        local_clouds.append(local)
        world_clouds.append(world)
        old_points, old_colors = load_ply(scene_dir / "pcd" / f"{index:02d}.ply")
        expected_colors = rgb.reshape(-1, 3)
        valid = np.isfinite(depth.ravel()) & (depth.ravel() > 0.05)
        far_cutoff = json.loads((scene_dir / "geometry.json").read_text())["depth_display_range"][1] * 1.2
        valid &= depth.ravel() < far_cutoff
        if len(old_points) != int(valid.sum()):
            raise ValueError(f"Unexpected PCD export mask for frame {index}")
        reproduction = {
            "frame": index, "point_count": int(valid.sum()),
            "max_abs_xyz_difference_from_existing_ply": float(np.max(np.abs(world[valid] - old_points))),
            "rgb_identical": bool(np.array_equal(expected_colors[valid], old_colors)),
            "validity": "All finite depths >0.05m and below the original 2%-98% display upper bound times 1.2.",
        }
        if reproduction["max_abs_xyz_difference_from_existing_ply"] > 3e-5 or not reproduction["rgb_identical"]:
            raise ValueError(f"Existing PLY is not reproduced by supplied pose: {reproduction}")
        audit["existing_ply_reproduction"].append(reproduction)
        audit["frames"].append({
            "frame": index, "name": name, "source_sha256": old["source_sha256"],
            "camera_center_world_m": center.tolist(), "K": frame["K"].tolist(),
            "c2w_opencv": c2w.tolist(), "rotation_orthonormal_max_error": orthonormal_error,
            "determinant": float(np.linalg.det(R)),
            "roundtrip_max_error": float(np.max(np.abs(project(world, frame)[0] - local))),
            "depth_quantiles_m": {str(q): float(np.quantile(depth, q)) for q in (0.01, 0.5, 0.99)},
        })
        print(f"✅ frame {index}: supplied pose exactly matches manifest; PCD reproduced, {valid.sum():,} points", flush=True)

    audit["relative_camera_motions"] = []
    for i in range(3):
        for j in range(i + 1, 3):
            relative = np.linalg.inv(frames[i]["c2w"]) @ frames[j]["c2w"]
            angle = float(np.degrees(np.linalg.norm(Rotation.from_matrix(relative[:3, :3]).as_rotvec())))
            audit["relative_camera_motions"].append({
                "from": i, "to": j, "baseline_m": float(np.linalg.norm(frames[j]["c2w"][:3, 3] - frames[i]["c2w"][:3, 3])),
                "rotation_degrees": angle,
            })
    metric_reports = {}
    for variant, clouds in (("identity_extrinsics_counterfactual", local_clouds),
                            ("supplied_poses_world", world_clouds)):
        report = {"coordinate": "All points remain in frame 00 local OpenCV coordinates" if variant.startswith("identity")
                  else "Shared Remy world after OpenGL-to-OpenCV c2w conversion",
                  "depth_source": "Identical cached joint three-frame metric depth as the existing formal exports",
                  "registration_optimization": "none; no ICP and no per-frame scale fitting",
                  "crossframe_nearest_neighbor": [], "ordered_projected_relative_z": []}
        for i in range(3):
            source = clouds[i].reshape(-1, 3)[::8]
            nearest = np.minimum(*[cKDTree(clouds[j].reshape(-1, 3)[::8]).query(source, k=1)[0]
                                   for j in range(3) if j != i])
            report["crossframe_nearest_neighbor"].append({
                "source": i, "stride": 8, "no_visibility_filter": True,
                "median_m": float(np.median(nearest)), "p90_m": float(np.quantile(nearest, 0.9)),
                "caution": "Includes occlusion, non-common views, motion and depth-edge outliers; not a geometric error metric.",
            })
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                h, w = depths[i].shape
                y, x = np.mgrid[:h:4, :w:4]
                y, x = y.astype(np.int32).ravel(), x.astype(np.int32).ravel()
                ids = y * w + x
                camera, u, v = project(clouds[i][ids], frames[j])
                z = camera[:, 2]
                inside = (z > 0.02) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
                target = depths[j][v[inside], u[inside]]
                finite = np.isfinite(target) & (target > 0)
                error = np.abs(z[inside][finite] - target[finite]) / target[finite]
                row = {"source": i, "target": j, "projected_n": int(inside.sum()),
                       "all_projected": stats(error), "occlusion_aware": False,
                       "note": "Ordered source-to-target projection; unlike evaluate_consistency.py, no NCC or smoothness subset selection."}
                if variant == "supplied_poses_world":
                    old = json.loads((scene_dir / "consistency.json").read_text())
                    previous = next(p for p in old["pairs"] if p["source"] == i and p["target"] == j)
                    row["original_appearance_stable_subset"] = previous["appearance_projected_relative_depth"]
                report["ordered_projected_relative_z"].append(row)
        write_json(out / f"{variant}_metrics.json", report)
        metric_reports[variant] = report
        if variant.startswith("identity"):
            continue
        save_ply(out / "fused_supplied_poses_world.ply", np.concatenate([c[::8] for c in clouds]),
                 np.concatenate([r.reshape(-1, 3)[::8] for r in rgbs]),
                 np.concatenate([np.full(len(c.reshape(-1, 3)[::8]), i, np.uint8) for i, c in enumerate(clouds)]))

    # Native-resolution, per-pixel single-source comparisons; all three inputs are full-coverage.
    side = out / "side_by_side"
    side.mkdir(exist_ok=True)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 22)
    small = ImageFont.truetype(font_path, 16)
    for i in range(3):
        panels = [rgbs[i]]
        for clouds, footer in ((local_clouds, "A | ignore extrinsics (counterfactual)"),
                               (world_clouds, "B | supplied c2w (existing export)")):
            others = np.concatenate([clouds[j].reshape(-1, 3) for j in range(3) if j != i])
            colors = np.concatenate([rgbs[j].reshape(-1, 3) for j in range(3) if j != i])
            rendered, _, coverage = render_splat(others, colors, frames[i], depths[i].shape, 0.8)
            panels.append(rendered)
            print(f"🔍 {footer}, target={i}: disc coverage={coverage:.2%}")
        canvas = Image.new("RGB", (sum(p.shape[1] for p in panels), depths[i].shape[0] + 64), (248, 248, 248))
        draw = ImageDraw.Draw(canvas)
        x = 0
        for panel, title, note in zip(panels, ("RGB target", "Other frames | no pose", "Other frames | supplied pose"),
                                     ("Reference only", "Same depth; no registration optimization", "OpenGL c2w -> OpenCV; world-space PCD")):
            draw.text((x + 12, 8), title, fill=(22, 30, 40), font=font)
            draw.text((x + 12, 37), note, fill=(75, 83, 92), font=small)
            canvas.paste(Image.fromarray(panel), (x, 64))
            x += panel.shape[1]
        canvas.save(side / f"target_{i:02d}.jpg", quality=94)

    # An intermediate real camera keeps all three frames visible for the same-points comparison.
    c2w = np.asarray(frames[0]["c2w"]).copy()
    c2w[:3, :3] = Rotation.from_rotvec(0.5 * Rotation.from_matrix(
        frames[1]["c2w"][:3, :3] @ frames[0]["c2w"][:3, :3].T).as_rotvec()).as_matrix() @ frames[0]["c2w"][:3, :3]
    c2w[:3, 3] = 0.5 * (frames[0]["c2w"][:3, 3] + frames[1]["c2w"][:3, 3])
    camera = {"K": frames[0]["K"], "c2w": c2w, "w2c": np.linalg.inv(c2w)}
    panels = []
    for clouds, label in ((local_clouds, "A | identity extrinsics (counterfactual)"),
                          (world_clouds, "B | transforms.json c2w (existing export)")):
        image, _, coverage = render_splat(np.concatenate([c[::8] for c in clouds]),
                                         np.concatenate([r.reshape(-1, 3)[::8] for r in rgbs]),
                                         camera, depths[0].shape, 0.8)
        bar = Image.new("RGB", (image.shape[1], 64), (248, 248, 248))
        d = ImageDraw.Draw(bar)
        d.text((12, 8), label, fill=(22, 30, 40), font=font)
        d.text((12, 37), f"Identical cached depth; midpoint real camera; 1/8 pixels; coverage={coverage:.1%}",
               fill=(75, 83, 92), font=small)
        panels.append(np.concatenate([np.asarray(bar), image], axis=0))
        print(f"✅ midpoint fused rendering: {label}")
    Image.fromarray(np.concatenate(panels, axis=1)).save(out / "fused_pose_comparison.jpg", quality=95)
    print_metric_summary(metric_reports["identity_extrinsics_counterfactual"],
                         metric_reports["supplied_poses_world"])
    official_repo = Path(os.environ.get("DEPTH_PRO_2_DIR", str(Path.home() / "repos/ml-depth-pro-2")))
    sys.path.insert(0, str(official_repo / "src"))
    # Class-signature introspection constructs no model or GPU tensors.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="xFormers is available*", category=UserWarning)
        from depth_pro_2 import DepthPro2
    audit["model_pose_conditioning"]["infer_signature"] = str(inspect.signature(DepthPro2.infer))
    print(f"🔍 Official DepthPro2.infer signature: {audit['model_pose_conditioning']['infer_signature']}")
    print("🔍 No camera extrinsic/pose argument is accepted; transforms apply after inference only")
    write_json(out / "pose_audit.json", audit)
    print(f"🎉 Pose audit and same-depth A/B: {out}")


if __name__ == "__main__":
    main()
