#!/usr/bin/env python3
"""render_closeup.py — Generate closeup + pitch-extrapolated views for face enhancement.

Strategy (from grill-me design):
  1. Parse COLMAP cameras, compute azimuth for each (relative to face center).
  2. Bin cameras into 10 azimuth bins (36° each).
  3. Per bin, pick the frame with highest SAM2 mask coverage = 10 base cameras.
  4. For each base camera: move along optical axis toward face center until
     SAM2 mask coverage >= 40% (min distance = 30% of original).
  5. At closeup position, generate 2 extrapolated views: pitch +10°, pitch -10°
     (rotation around face center's horizontal axis).
  6. 3DGS render all 20 extrapolated views → save RGB + alpha + poses.

Output:
  closeup_renders/closeup_0001.png ... closeup_0020.png
  closeup_alpha/closeup_0001.png ...
  closeup_poses.json  (camera params for HYPIR enhance + finetune)

Env vars:
  GAUSSIAN_DIR  : 3DGS model dir (has point_cloud/iteration_*/point_cloud.ply)
  SOURCE_DIR    : COLMAP source (sparse/0/)
  RESULTS_DIR   : output root
  FACE_CENTER   : JSON with face 3D center (from face_center_3d.py)
  MASKS_DIR     : SAM2 face masks folder
  N_BINS        : azimuth bins (default 10)
  TARGET_COV    : target SAM2 coverage % to stop (default 40)
  MIN_DIST_RATIO: min distance ratio (default 0.3)
  PITCH_DEG     : pitch extrapolation degrees (default 10)
  ITERATION     : 3DGS iteration to load (default 30000)
  GS_DIR        : gaussian-splatting repo path
  DEVICE        : cuda
"""
import os
import sys
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Config from env
GAUSSIAN_DIR = os.environ.get("GAUSSIAN_DIR", "")
SOURCE_DIR = os.environ.get("SOURCE_DIR", "")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "")
FACE_CENTER_JSON = os.environ.get("FACE_CENTER", "")
MASKS_DIR = os.environ.get("MASKS_DIR", "")
N_BINS = int(os.environ.get("N_BINS", "10"))
TARGET_COV = float(os.environ.get("TARGET_COV", "40"))
MIN_DIST_RATIO = float(os.environ.get("MIN_DIST_RATIO", "0.3"))
PITCH_DEG = float(os.environ.get("PITCH_DEG", "10"))
ITERATION = int(os.environ.get("ITERATION", "30000"))
GS_DIR = os.environ.get("GS_DIR", os.path.expanduser("~/repos/gaussian-splatting"))
DEVICE = os.environ.get("DEVICE", "cuda")


def quat_to_rotmat(qw, qx, qy, qz):
    """Quaternion to 3x3 rotation matrix (COLMAP convention: world->camera)."""
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)],
    ])


def parse_colmap(source_dir):
    """Parse COLMAP cameras.txt + images.txt → list of view dicts.

    Handles layouts: sparse/0/cameras.txt, sparse/0_text/cameras.txt,
    or source_dir/cameras.txt.
    """
    sparse = Path(source_dir) / "sparse" / "0"
    cam_file = sparse / "cameras.txt"
    img_file = sparse / "images.txt"
    if not cam_file.exists():
        # Try sparse/0_text/ (BA output text format)
        sparse_text = Path(source_dir) / "sparse" / "0_text"
        cam_file = sparse_text / "cameras.txt"
        img_file = sparse_text / "images.txt"
    if not cam_file.exists():
        cam_file = Path(source_dir) / "cameras.txt"
        img_file = Path(source_dir) / "images.txt"

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
                fx = fy = params[0]; cx, cy = params[1:3]
            else:
                continue
            cameras[cid] = dict(W=W, H=H, fx=fx, fy=fy, cx=cx, cy=cy)

    views = []
    with open(img_file) as f:
        lines = f.readlines()
    for i in range(0, len(lines), 2):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        qvec = np.array(list(map(float, parts[1:5])))
        tvec = np.array(list(map(float, parts[5:8])))
        cid = int(parts[8])
        name = parts[9]
        if cid not in cameras:
            continue
        cam = cameras[cid]
        R = quat_to_rotmat(*qvec)
        C = -R.T @ tvec  # camera position in world
        stem = os.path.splitext(name)[0]
        views.append(dict(
            stem=stem, name=name, R=R, T=tvec, C=C,
            **cam
        ))
    return views


def load_sam2_coverage(masks_dir, stems):
    """Load SAM2 mask coverage (%) per stem."""
    cov = {}
    for stem in stems:
        p = os.path.join(masks_dir, f"{stem}.mask.png")
        if os.path.exists(p):
            m = np.array(Image.open(p).convert("L"))
            cov[stem] = m.sum() / m.size * 100 / 255  # fraction * 100
    return cov


def select_base_views(views, coverage, n_bins):
    """Bin by azimuth, pick highest-coverage per bin. Returns list of view dicts."""
    if not coverage:
        print("  ⚠️ no SAM2 coverage data, using first N_BINS views")
        return views[:n_bins]

    # Azimuth relative to scene center (mean of camera positions)
    center = np.mean([v["C"] for v in views], axis=0)
    for v in views:
        d = v["C"] - center
        v["azimuth"] = math.atan2(d[2], d[0])

    # Bin by azimuth [0, 2pi) → n_bins
    bin_size = 2 * math.pi / n_bins
    bins = [[] for _ in range(n_bins)]
    for v in views:
        az = v["azimuth"] % (2 * math.pi)
        bi = min(int(az / bin_size), n_bins - 1)
        bins[bi].append(v)

    selected = []
    for bi, bv in enumerate(bins):
        if not bv:
            continue
        # Pick highest coverage in bin
        scored = [(coverage.get(v["stem"], 0), v) for v in bv]
        scored.sort(reverse=True, key=lambda x: x[0])
        best_cov, best_v = scored[0]
        best_v["bin"] = bi
        best_v["coverage"] = best_cov
        selected.append(best_v)
        az_deg = math.degrees(best_v["azimuth"] % (2 * math.pi))
        print(f"  bin {bi}: az={az_deg:.0f}° {best_v['stem']} cov={best_cov:.1f}%")

    print(f"  selected {len(selected)} base views from {n_bins} bins")
    return selected


def compute_closeup_pose(view, face_center, target_cov, masks_dir, min_ratio):
    """Move camera along optical axis toward face center until coverage >= target.

    Returns new (R, T, C) — rotation unchanged, only position moves.
    """
    # Optical axis direction (camera forward in world) = -R[2,:] (third row of R, negated)
    # Camera looks along -Z in camera frame, so forward in world = R^T @ [0,0,-1] = -R[2,:]
    forward = -view["R"][2, :]  # (3,)
    forward = forward / (np.linalg.norm(forward) + 1e-8)

    orig_C = view["C"].copy()
    orig_dist = np.linalg.norm(face_center - orig_C)
    min_dist = orig_dist * min_ratio

    # Binary search for target coverage
    # Start from original, move toward face_center
    # Coverage increases as we get closer
    best_C = orig_C.copy()
    best_cov = view.get("coverage", 0)

    # Step toward face center
    direction = (face_center - orig_C)
    direction = direction / (np.linalg.norm(direction) + 1e-8)

    # Try distances from orig_dist down to min_dist
    for ratio in np.arange(0.95, min_ratio - 0.05, -0.05):
        new_dist = orig_dist * ratio
        new_C = face_center - direction * new_dist
        # New T: T = -R @ C
        new_T = -view["R"] @ new_C

        # Estimate coverage: linear approximation
        # (actual rendering needed for precise coverage, but we can approximate
        #  by scaling: coverage ~ (orig_dist/new_dist)^2 * orig_cov, capped at 100)
        est_cov = min(view.get("coverage", 0) * (orig_dist / new_dist) ** 2, 100)

        if est_cov >= target_cov:
            best_C = new_C
            best_T = new_T
            best_cov = est_cov
            break
        best_C = new_C
        best_T = new_T
        best_cov = est_cov

    new_T = -view["R"] @ best_C
    return view["R"], new_T, best_C, best_cov


def apply_pitch(view, face_center, pitch_deg):
    """Rotate camera around face center by pitch angle (X-axis rotation).

    Returns new (R, T, C).
    """
    theta = math.radians(pitch_deg)
    # Rotation around X-axis (pitch)
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(theta), -math.sin(theta)],
        [0, math.sin(theta), math.cos(theta)],
    ])

    R = view["R"]
    C = view["C"]

    # Rotate camera position around face center
    # C_new = face_center + Rx @ (C - face_center)
    C_new = face_center + Rx @ (C - face_center)

    # Rotate camera orientation: R_new = Rx @ R
    R_new = Rx @ R

    # New T = -R_new @ C_new
    T_new = -R_new @ C_new

    return R_new, T_new, C_new


def render_views(closeup_views, out_dir, alpha_dir):
    """3DGS render closeup views. closeup_views: list of {R, T, W, H, fx, fy, name}."""
    import torch
    sys.path.insert(0, GS_DIR)
    from scene import GaussianModel
    from scene.cameras import Camera
    from gaussian_renderer import render
    from argparse import Namespace

    ply_path = os.path.join(GAUSSIAN_DIR, "point_cloud", f"iteration_{ITERATION}", "point_cloud.ply")
    if not os.path.isfile(ply_path):
        sys.exit(f"PLY not found: {ply_path}")

    sh_degree = 3
    cfg_path = os.path.join(GAUSSIAN_DIR, "cfg_args")
    if os.path.isfile(cfg_path):
        import re
        with open(cfg_path) as f:
            m = re.search(r"sh_degree\s*=\s*(\d+)", f.read())
            if m:
                sh_degree = int(m.group(1))

    print(f"  loading 3DGS: {ply_path} (sh={sh_degree})")
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(ply_path)
    gaussians.active_sh_degree = sh_degree

    pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False,
                     antialiasing=False, debug=False)
    bg_black = torch.zeros(3, device=DEVICE)
    bg_white = torch.ones(3, device=DEVICE)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(alpha_dir).mkdir(parents=True, exist_ok=True)

    poses = []
    for i, cv in enumerate(closeup_views):
        R = cv["R"]; T = cv["T"]; W = cv["W"]; H = cv["H"]
        fx = cv["fx"]; fy = cv["fy"]
        FoVx = 2 * math.atan(W / (2 * fx))
        FoVy = 2 * math.atan(H / (2 * fy))

        dummy = Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8))
        cam = Camera(resolution=(W, H), colmap_id=0, R=R, T=T,
                     FoVx=FoVx, FoVy=FoVy, depth_params=None,
                     image=dummy, invdepthmap=None,
                     image_name=cv["name"], uid=i, data_device=DEVICE)

        with torch.no_grad():
            pkg_b = render(cam, gaussians, pipe, bg_black)
            pkg_w = render(cam, gaussians, pipe, bg_white)

        rgb = pkg_b["render"].clamp(0, 1)
        white = pkg_w["render"].clamp(0, 1)
        alpha = 1 - (white - rgb).mean(dim=0, keepdim=True).clamp(0, 1)

        rgb_path = os.path.join(out_dir, cv["name"])
        alpha_path = os.path.join(alpha_dir, cv["name"])
        Image.fromarray((rgb.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(rgb_path)
        Image.fromarray((alpha[0].cpu().numpy() * 255).astype(np.uint8)).save(alpha_path)

        avg_a = float(alpha.mean())
        print(f"  [{i+1}/{len(closeup_views)}] {cv['name']} alpha={avg_a:.3f} "
              f"{'⚠️' if avg_a < 0.3 else '✅'}")

        poses.append(dict(
            name=cv["name"],
            R=cv["R"].tolist(),
            T=cv["T"].tolist(),
            C=cv["C"].tolist(),
            W=W, H=H, fx=fx, fy=fy,
            cx=cv["cx"], cy=cv["cy"],
            base_view=cv.get("base_stem", ""),
            pitch=cv.get("pitch", 0),
            est_coverage=cv.get("est_cov", 0),
        ))

    del gaussians
    torch.cuda.empty_cache()
    return poses


def main():
    print("🧑 [render_closeup] face closeup + pitch extrapolation")
    print(f"  📂 3DGS: {GAUSSIAN_DIR} (iter={ITERATION})")
    print(f"  📂 source: {SOURCE_DIR}")
    print(f"  📂 masks: {MASKS_DIR}")
    print(f"  📐 bins={N_BINS}, target_cov={TARGET_COV}%, min_dist={MIN_DIST_RATIO}, pitch=±{PITCH_DEG}°")
    print("")

    # 1. Load face center
    with open(FACE_CENTER_JSON) as f:
        fc = json.load(f)
    face_center = np.array(fc["center"])
    print(f"  face center: {face_center}")

    # 2. Parse COLMAP cameras
    print("\n📷 parsing COLMAP cameras...")
    views = parse_colmap(SOURCE_DIR)

    # 3. Load SAM2 coverage
    coverage = load_sam2_coverage(MASKS_DIR, [v["stem"] for v in views])
    print(f"  SAM2 coverage for {len(coverage)} views")

    # 4. Select base views (azimuth binning)
    print(f"\n🎯 selecting {N_BINS} base views by azimuth binning...")
    base_views = select_base_views(views, coverage, N_BINS)

    # 5. For each base view: closeup + pitch extrapolation
    print(f"\n🔭 generating closeup + pitch views...")
    closeup_views = []
    for bv in base_views:
        # Closeup: move toward face center
        R, T, C, cov = compute_closeup_pose(bv, face_center, TARGET_COV, MASKS_DIR, MIN_DIST_RATIO)
        closeup = dict(bv)
        closeup["R"] = R; closeup["T"] = T; closeup["C"] = C
        closeup["est_cov"] = cov
        closeup["base_stem"] = bv["stem"]
        closeup["pitch"] = 0

        # Pitch +10
        R_p, T_p, C_p = apply_pitch(closeup, face_center, PITCH_DEG)
        vp = dict(closeup)
        vp["R"] = R_p; vp["T"] = T_p; vp["C"] = C_p
        vp["pitch"] = +PITCH_DEG
        vp["name"] = f"closeup_{len(closeup_views)+1:04d}.png"
        closeup_views.append(vp)

        # Pitch -10
        R_m, T_m, C_m = apply_pitch(closeup, face_center, -PITCH_DEG)
        vm = dict(closeup)
        vm["R"] = R_m; vm["T"] = T_m; vm["C"] = C_m
        vm["pitch"] = -PITCH_DEG
        vm["name"] = f"closeup_{len(closeup_views)+1:04d}.png"
        closeup_views.append(vm)

    print(f"  {len(closeup_views)} closeup views generated")

    # 6. 3DGS render
    print(f"\n🖼️ rendering closeup views...")
    renders_dir = os.path.join(RESULTS_DIR, "closeup_renders")
    alpha_dir = os.path.join(RESULTS_DIR, "closeup_alpha")
    poses = render_views(closeup_views, renders_dir, alpha_dir)

    # 7. Save poses
    poses_path = os.path.join(RESULTS_DIR, "closeup_poses.json")
    with open(poses_path, "w") as f:
        json.dump(poses, f, indent=2)
    print(f"\n💾 poses: {poses_path}")
    print(f"🖼️ renders: {renders_dir}")
    print(f"✅ done: {len(poses)} closeup views rendered")


if __name__ == "__main__":
    main()
