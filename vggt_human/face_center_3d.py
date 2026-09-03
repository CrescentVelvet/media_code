#!/usr/bin/env python3
"""face_center_3d.py — Compute 3D face center from 3DGS point cloud + SAM2 masks.

Strategy:
  1. Load 3DGS point cloud (PLY) — millions of 3D gaussians.
  2. Load SAM2 face masks (79 frames, pixel-level).
  3. For each 3D point, project to all 79 camera views using COLMAP poses.
  4. A point is a "face point" if it falls inside the SAM2 mask in >= N frames
     (multi-view consistency, default N=3).
  5. Face 3D center = centroid of all face points.

Output: face_center.json with {center: [x,y,z], n_face_points, n_total_points,
       per_view_coverage: {frame: coverage%}}

Env vars:
  PLY_PATH     : 3DGS point cloud PLY
  SOURCE_DIR   : COLMAP source (sparse/0/ has cameras.txt, images.txt)
  MASKS_DIR    : SAM2 face masks folder (has <stem>.mask.png)
  OUTPUT_JSON  : output JSON path
  MIN_HITS     : min frames a point must hit to count as face (default 3)
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def load_ply_points(ply_path):
    """Load PLY as (N, 3) xyz array. Minimal PLY parser for gaussian-splatting output."""
    print(f"📥 loading PLY: {ply_path}")
    t0 = time.time()
    with open(ply_path, "rb") as f:
        header = b""
        while True:
            line = f.readline()
            header += line
            if b"element vertex" in line:
                n_vertices = int(line.split()[-1])
            if b"end_header" in line:
                break
        # Read binary data
        # gaussian-splatting PLY: x,y,z, ... (float32) + opacity + SH ...
        # We only need x,y,z — first 3 float32 per vertex
        # But need to skip the rest. Parse property list from header.
        props = []
        for line in header.split(b"\n"):
            line = line.strip()
            if line.startswith(b"property") and b"float" in line:
                props.append(("float", 4))
            elif line.startswith(b"property") and b"uchar" in line:
                props.append(("uchar", 1))

        stride = sum(s for _, s in props)
        n_floats_before_xyz = 0  # x,y,z are first 3 in GS PLY
        xyz_offset = 0
        # Actually, parse properly: find x, y, z property positions
        prop_names = []
        for line in header.split(b"\n"):
            line = line.strip().decode("ascii", errors="ignore")
            if line.startswith("property"):
                parts = line.split()
                if len(parts) >= 3:
                    prop_names.append(parts[2])

        xyz_idx = [prop_names.index("x"), prop_names.index("y"), prop_names.index("z")]
        # Build dtype
        dtype_fields = []
        offset = 0
        for i, (pname, (ptype, psize)) in enumerate(zip(prop_names, props)):
            if ptype == "float":
                dtype_fields.append((pname, "<f4"))
            else:
                dtype_fields.append((pname, "u1"))
        dt = np.dtype(dtype_fields)

        data = np.frombuffer(f.read(n_vertices * stride), dtype=dt)
        xyz = np.stack([data["x"], data["y"], data["z"]], axis=-1).astype(np.float64)
    print(f"  {len(xyz)} points, {time.time()-t0:.1f}s")
    return xyz


def parse_colmap_cameras(source_dir):
    """Parse COLMAP cameras.txt + images.txt → list of (stem, qvec, tvec, fx, fy, cx, cy, W, H)."""
    cam_file = Path(source_dir) / "sparse" / "0" / "cameras.txt"
    img_file = Path(source_dir) / "sparse" / "0" / "images.txt"
    if not cam_file.exists():
        cam_file = Path(source_dir) / "cameras.txt"
        img_file = Path(source_dir) / "images.txt"

    # cameras.txt
    cameras = {}
    with open(cam_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cid = int(parts[0])
            model = parts[1]
            W, H = int(parts[2]), int(parts[3])
            params = list(map(float, parts[4:]))
            if model == "PINHOLE":
                fx, fy, cx, cy = params[:4]
            elif model == "SIMPLE_PINHOLE":
                f, cx, cy = params[:3]
                fx = fy = f
            else:
                continue
            cameras[cid] = dict(W=W, H=H, fx=fx, fy=fy, cx=cx, cy=cy)

    # images.txt
    views = []
    with open(img_file) as f:
        lines = f.readlines()
    for i in range(0, len(lines), 2):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        img_id = int(parts[0])
        qvec = np.array(list(map(float, parts[1:5])), dtype=np.float64)  # qw,qx,qy,qz
        tvec = np.array(list(map(float, parts[5:8])), dtype=np.float64)
        cid = int(parts[8])
        name = parts[9]
        if cid not in cameras:
            continue
        cam = cameras[cid]
        stem = os.path.splitext(name)[0]
        views.append(dict(stem=stem, qvec=qvec, tvec=tvec, **cam))

    print(f"  parsed {len(views)} camera views")
    return views


def load_masks(masks_dir, stems):
    """Load SAM2 binary masks. Returns dict stem -> bool array (H, W)."""
    masks = {}
    for stem in stems:
        mask_path = os.path.join(masks_dir, f"{stem}.mask.png")
        if os.path.exists(mask_path):
            m = np.array(Image.open(mask_path).convert("L"))
            masks[stem] = m > 127
    return masks


def project_points(xyz, view):
    """Project 3D points (N,3) to 2D pixel coords. Returns (N,2) and validity mask."""
    q = view["qvec"]
    t = view["tvec"]
    # Rotation from quaternion (COLMAP: world->camera)
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])
    # World->camera
    cam_pts = (R @ xyz.T).T + t  # (N, 3)
    # Only points in front of camera (z > 0)
    valid = cam_pts[:, 2] > 0
    # Project
    z = cam_pts[:, 2:3].copy()
    z[z == 0] = 1e-6
    uv = np.zeros((len(xyz), 2))
    uv[:, 0] = (view["fx"] * cam_pts[:, 0] / z[:, 0] + view["cx"])
    uv[:, 1] = (view["fy"] * cam_pts[:, 1] / z[:, 0] + view["cy"])
    return uv, valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--source_dir", required=True)
    ap.add_argument("--masks_dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--min_hits", type=int, default=3)
    args = ap.parse_args()

    print(f"🧑 face 3D center computation")
    print(f"  📂 PLY: {args.ply}")
    print(f"  📂 source: {args.source_dir}")
    print(f"  📂 masks: {args.masks_dir}")
    print(f"  📐 min_hits: {args.min_hits}")
    print("")

    # 1. Load points
    xyz = load_ply_points(args.ply)

    # 2. Parse cameras
    print("\n📷 parsing COLMAP cameras...")
    views = parse_colmap_cameras(args.source_dir)

    # 3. Load masks
    print("\n🎭 loading SAM2 masks...")
    masks = load_masks(args.masks_dir, [v["stem"] for v in views])
    print(f"  {len(masks)} masks found (of {len(views)} views)")

    if not masks:
        sys.exit("no masks found")

    # 4. Project + count hits
    print(f"\n🎯 projecting {len(xyz)} points to {len(masks)} views...")
    t0 = time.time()
    hit_counts = np.zeros(len(xyz), dtype=np.int32)
    per_view_cov = {}

    for vi, view in enumerate(views):
        if view["stem"] not in masks:
            continue
        mask = masks[view["stem"]]
        uv, valid = project_points(xyz, view)
        # Check bounds + mask
        in_bounds = (
            valid &
            (uv[:, 0] >= 0) & (uv[:, 0] < view["W"]) &
            (uv[:, 1] >= 0) & (uv[:, 1] < view["H"])
        )
        # Sample mask at projected coords
        mask_hits = np.zeros(len(xyz), dtype=bool)
        valid_idx = np.where(in_bounds)[0]
        if len(valid_idx) > 0:
            px = uv[valid_idx, 0].astype(int)
            py = uv[valid_idx, 1].astype(int)
            px = np.clip(px, 0, mask.shape[1] - 1)
            py = np.clip(py, 0, mask.shape[0] - 1)
            mask_hits[valid_idx] = mask[py, px]
        hit_counts += mask_hits.astype(np.int32)
        cov = mask.sum() / mask.size * 100
        per_view_cov[view["stem"]] = round(cov, 2)
        if (vi + 1) % 20 == 0 or vi == 0:
            n_hit = mask_hits.sum()
            print(f"  [{vi+1}/{len(views)}] {view['stem']}: cov={cov:.1f}%, pts_in_mask={n_hit}")

    print(f"  projection done in {time.time()-t0:.1f}s")

    # 5. Face points = hit >= min_hits
    face_pts_mask = hit_counts >= args.min_hits
    n_face = face_pts_mask.sum()
    print(f"\n  face points (hits>={args.min_hits}): {n_face} / {len(xyz)} ({n_face/len(xyz)*100:.2f}%)")

    if n_face == 0:
        print("  ⚠️ no face points found, trying min_hits=1")
        face_pts_mask = hit_counts >= 1
        n_face = face_pts_mask.sum()
        print(f"  face points (hits>=1): {n_face}")

    if n_face == 0:
        sys.exit("no face points even with min_hits=1")

    # 6. Centroid = face 3D center
    center = xyz[face_pts_mask].mean(axis=0)
    spread = xyz[face_pts_mask].std(axis=0)
    print(f"\n  face 3D center: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
    print(f"  spread (std):   [{spread[0]:.4f}, {spread[1]:.4f}, {spread[2]:.4f}]")

    # 7. Save
    result = {
        "center": [float(center[0]), float(center[1]), float(center[2])],
        "spread": [float(spread[0]), float(spread[1]), float(spread[2])],
        "n_face_points": int(n_face),
        "n_total_points": int(len(xyz)),
        "min_hits": args.min_hits,
        "per_view_coverage": per_view_cov,
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n💾 saved: {args.output}")
    print(f"\n✅ face 3D center: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")


if __name__ == "__main__":
    main()
