#!/usr/bin/env python3
"""render_novel.py — Load 3DGS model, select novel viewpoints, render + save.

Stage 1 of the denoise-augment pipeline (called by 05_denoise_novel.sh):
  1. Load 3DGS gaussians from step 03's PLY checkpoint.
  2. Parse COLMAP cameras.txt + images.txt to get training camera trajectory.
  3. Find angular gaps in the camera trajectory; insert NUM_NOVEL_VIEWS
     intermediate viewpoints (linear position + SLERP rotation).
  4. Render each novel viewpoint (black + white bg) → RGB + alpha.
  5. Save: 05_novel_renders/*.png (RGB), 05_novel_alpha/*.png (alpha),
           05_novel_poses.json (camera params for stage 2).

Stage 2 (denoise_images.py) reads these outputs, denoises, AdaIN-corrects,
and writes the augmented COLMAP scene.

Env vars (set by 05_denoise_novel.sh):
  GS_DIR, GAUSSIAN_DIR, SOURCE_DIR, RESULTS_DIR, ITERATION,
  NUM_NOVEL_VIEWS, RENDER_WIDTH, RENDER_HEIGHT, DEVICE
"""
import os
import sys
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

GS_DIR = os.environ.get("GS_DIR", "../gaussian-splatting")
GAUSSIAN_DIR = os.environ.get("GAUSSIAN_DIR", "")
SOURCE_DIR = os.environ.get("SOURCE_DIR", "")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "")
ITERATION = int(os.environ.get("ITERATION", "30000"))
NUM_NOVEL_VIEWS = int(os.environ.get("NUM_NOVEL_VIEWS", "10"))
RENDER_WIDTH = int(os.environ.get("RENDER_WIDTH", "0"))   # 0 = use training image size
RENDER_HEIGHT = int(os.environ.get("RENDER_HEIGHT", "0"))
DEVICE = os.environ.get("DEVICE", "cuda")


# ---------------------------------------------------------------------------
# COLMAP text format parsers.
# ---------------------------------------------------------------------------
def parse_cameras_txt(path):
    """Returns {cam_id: (model, W, H, [fx, fy, cx, cy])}."""
    cameras = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            W, H = int(parts[2]), int(parts[3])
            params = [float(x) for x in parts[4:]]
            cameras[cam_id] = (model, W, H, params)
    return cameras


def parse_images_txt(path):
    """Returns list of (img_id, [qw,qx,qy,qz], [tx,ty,tz], cam_id, name)."""
    images = []
    with open(path) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        parts = line.split()
        img_id = int(parts[0])
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        cam_id = int(parts[8])
        name = parts[9]
        images.append((img_id, [qw, qx, qy, qz], [tx, ty, tz], cam_id, name))
        i += 2  # skip the points2D line
    return images


# ---------------------------------------------------------------------------
# Quaternion / rotation helpers.
# ---------------------------------------------------------------------------
def quat_to_rotmat(qw, qx, qy, qz):
    """COLMAP quaternion (scalar-first) -> 3x3 rotation matrix (w2c)."""
    from scipy.spatial.transform import Rotation
    # scipy uses scalar-last [qx, qy, qz, qw]
    return Rotation.from_quat([qx, qy, qz, qw]).as_matrix()


def rotmat_to_quat(R):
    """3x3 rotation matrix -> (qw, qx, qy, qz)."""
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(R).as_quat()  # [qx, qy, qz, qw]
    return q[3], q[0], q[1], q[2]  # -> (qw, qx, qy, qz)


def slerp_rotations(R1, R2, t):
    """SLERP between two 3x3 rotation matrices at parameter t in [0,1]."""
    from scipy.spatial.transform import Rotation, Slerp
    key_rots = Rotation.from_matrix(np.stack([R1, R2]))
    slerp = Slerp([0.0, 1.0], key_rots)
    return slerp([t]).as_matrix()[0]


# ---------------------------------------------------------------------------
# Novel view selection: find angular gaps in camera trajectory.
# ---------------------------------------------------------------------------
def select_novel_views(images_info, cameras_info, num_novel):
    """Insert novel viewpoints in the largest angular gaps of the camera orbit.

    Returns list of dicts: {R, T, fx, fy, cx, cy, W, H, name, nearest_idx}.
    """
    from scipy.spatial.transform import Rotation

    # Extract camera world positions: C = -R^T @ T
    cams = []
    for idx, (img_id, quat, t, cam_id, name) in enumerate(images_info):
        R = quat_to_rotmat(*quat)
        T = np.array(t)
        C = -R.T @ T
        cams.append({"idx": idx, "R": R, "T": T, "C": C, "cam_id": cam_id, "name": name})

    # Scene center = mean of camera positions
    center = np.mean([c["C"] for c in cams], axis=0)

    # Azimuth angle (xz-plane) relative to center
    for c in cams:
        d = c["C"] - center
        c["azimuth"] = math.atan2(d[2], d[0])

    # Sort by azimuth
    cams_sorted = sorted(cams, key=lambda c: c["azimuth"])

    # Gaps between adjacent cameras (handle wrap-around)
    n = len(cams_sorted)
    gaps = []
    for i in range(n):
        j = (i + 1) % n
        gap = cams_sorted[j]["azimuth"] - cams_sorted[i]["azimuth"]
        if gap < 0:
            gap += 2 * math.pi  # wrap-around
        gaps.append((gap, i, j))
    gaps.sort(reverse=True, key=lambda x: x[0])

    # Insert novel views in the largest gaps
    novel_views = []
    for rank in range(min(num_novel, len(gaps))):
        gap, i, j = gaps[rank]
        if gap < 0.1:  # too small, skip
            continue
        n_interp = max(1, int(gap / (2 * math.pi) * num_novel * 2))
        for k in range(1, n_interp + 1):
            t = k / (n_interp + 1)
            c1 = cams_sorted[i]
            c2 = cams_sorted[j]
            # Interpolate position (linear)
            C_interp = (1 - t) * c1["C"] + t * c2["C"]
            # Interpolate rotation (SLERP)
            R_interp = slerp_rotations(c1["R"], c2["R"], t)
            # Convert to w2c: T = -R @ C
            T_interp = -R_interp @ C_interp
            # Use nearest camera's intrinsics
            nearest = c1 if t < 0.5 else c2
            cam_info = cameras_info[nearest["cam_id"]]
            _, W, H, params = cam_info
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            # Override resolution if specified
            if RENDER_WIDTH > 0:
                scale = RENDER_WIDTH / W
                W, H = RENDER_WIDTH, int(H * scale)
                fx, fy, cx, cy = fx * scale, fy * scale, cx * scale, cy * scale
            name = f"novel_{len(novel_views):04d}.png"
            novel_views.append({
                "R": R_interp.tolist(),
                "T": T_interp.tolist(),
                "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                "W": W, "H": H,
                "name": name,
                "nearest_idx": nearest["idx"],
            })
            if len(novel_views) >= num_novel:
                break
        if len(novel_views) >= num_novel:
            break

    return novel_views


# ---------------------------------------------------------------------------
# 3DGS rendering.
# ---------------------------------------------------------------------------
def render_novel_views(novel_views):
    """Load 3DGS model and render novel views. Returns list of (rgb_path, alpha_path)."""
    import torch

    sys.path.insert(0, GS_DIR)
    from scene import GaussianModel  # noqa: E402
    from scene.cameras import Camera  # noqa: E402
    from gaussian_renderer import render  # noqa: E402
    from argparse import Namespace  # noqa: E402

    # Load gaussians
    ply_path = os.path.join(GAUSSIAN_DIR, "point_cloud", f"iteration_{ITERATION}", "point_cloud.ply")
    if not os.path.isfile(ply_path):
        sys.exit(f"❌ 3DGS PLY not found: {ply_path}")

    # Read SH degree from cfg_args
    sh_degree = 3
    cfg_path = os.path.join(GAUSSIAN_DIR, "cfg_args")
    if os.path.isfile(cfg_path):
        import re
        with open(cfg_path) as f:
            cfg_str = f.read()
        m = re.search(r"sh_degree\s*=\s*(\d+)", cfg_str)
        if m:
            sh_degree = int(m.group(1))

    print(f"  🏋️ loading 3DGS: {ply_path}  (sh_degree={sh_degree})")
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(ply_path)
    gaussians.active_sh_degree = sh_degree

    pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False, antialiasing=False, debug=False)
    bg_black = torch.zeros(3, device=DEVICE)
    bg_white = torch.ones(3, device=DEVICE)

    renders_dir = Path(RESULTS_DIR) / "05_novel_renders"
    alpha_dir = Path(RESULTS_DIR) / "05_novel_alpha"
    renders_dir.mkdir(parents=True, exist_ok=True)
    alpha_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image as PILImage

    results = []
    for i, nv in enumerate(novel_views):
        R = np.array(nv["R"])
        T = np.array(nv["T"])
        W, H = nv["W"], nv["H"]
        fx, fy = nv["fx"], nv["fy"]
        FoVx = 2 * math.atan(W / (2 * fx))
        FoVy = 2 * math.atan(H / (2 * fy))

        dummy = PILImage.fromarray(np.zeros((H, W, 3), dtype=np.uint8))
        # gaussian-splatting 的 getWorld2View2 内部会 R.transpose(),
        # Camera 期望 camera-to-world 的 R; 我们的 R 是 world-to-camera, 传 R.T。
        cam = Camera(resolution=(W, H), colmap_id=0, R=R.T, T=T, FoVx=FoVx, FoVy=FoVy,
                     depth_params=None, image=dummy, invdepthmap=None,
                     image_name=nv["name"], uid=i,
                     data_device=DEVICE)

        with torch.no_grad():
            pkg_black = render(cam, gaussians, pipe, bg_black)
            pkg_white = render(cam, gaussians, pipe, bg_white)

        rgb = pkg_black["render"].clamp(0, 1)  # (3, H, W)
        white = pkg_white["render"].clamp(0, 1)
        alpha = 1 - (white - rgb).mean(dim=0, keepdim=True).clamp(0, 1)  # (1, H, W)

        rgb_path = str(renders_dir / nv["name"])
        alpha_path = str(alpha_dir / nv["name"])
        PILImage.fromarray((rgb.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(rgb_path)
        PILImage.fromarray((alpha[0].cpu().numpy() * 255).astype(np.uint8)).save(alpha_path)

        avg_alpha = float(alpha.mean())
        print(f"  [{i+1}/{len(novel_views)}] {nv['name']}  "
              f"alpha_avg={avg_alpha:.3f}  {'⚠️ sparse' if avg_alpha < 0.3 else '✅'}")
        results.append((rgb_path, alpha_path, nv["name"]))

    # Free GPU memory before stage 2
    del gaussians
    torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main():
    if not GAUSSIAN_DIR:
        sys.exit("❌ GAUSSIAN_DIR not set (step 03's model dir)")
    if not SOURCE_DIR:
        sys.exit("❌ SOURCE_DIR not set (step 02's COLMAP scene)")
    if not RESULTS_DIR:
        sys.exit("❌ RESULTS_DIR not set")

    sparse_dir = os.path.join(SOURCE_DIR, "sparse", "0")
    cameras_path = os.path.join(sparse_dir, "cameras.txt")
    images_path = os.path.join(sparse_dir, "images.txt")

    if not os.path.isfile(cameras_path) or not os.path.isfile(images_path):
        sys.exit(f"❌ COLMAP scene not found at {sparse_dir}")

    t0 = time.time()

    # 1. Parse COLMAP scene
    print("🔍 [1/3] parsing COLMAP scene")
    cameras_info = parse_cameras_txt(cameras_path)
    images_info = parse_images_txt(images_path)
    print(f"  {len(cameras_info)} cameras, {len(images_info)} images")

    # (no PoseAdjuster transform: official train.py trains in the COLMAP frame,
    #  so novel views are inserted directly in the source scene's coordinates)

    # 2. Select novel viewpoints
    print(f"🎯 [2/3] selecting {NUM_NOVEL_VIEWS} novel viewpoints")
    novel_views = select_novel_views(images_info, cameras_info, NUM_NOVEL_VIEWS)
    print(f"  → {len(novel_views)} novel views")

    # Save novel poses for stage 2
    poses_path = Path(RESULTS_DIR) / "05_novel_poses.json"
    with open(poses_path, "w") as f:
        json.dump({"novel_cameras": novel_views}, f, indent=2)
    print(f"  → {poses_path}")

    # 3. Render
    print(f"🖼️ [3/3] rendering novel views (3DGS)")
    render_novel_views(novel_views)

    print(f"\n🎉 Done. {time.time() - t0:.1f}s")
    print(f"  renders: {RESULTS_DIR}/05_novel_renders/*.png")
    print(f"  alpha:   {RESULTS_DIR}/05_novel_alpha/*.png")
    print(f"  poses:   {RESULTS_DIR}/05_novel_poses.json")
    print(f"  Next: denoise_images.py (stage 2)")


if __name__ == "__main__":
    main()
