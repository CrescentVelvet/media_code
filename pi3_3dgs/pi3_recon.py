#!/usr/bin/env python3
"""pi3_recon.py — Pi3 (π³, ICLR 2026) feed-forward pose estimation + dense point
cloud, exported to COLMAP text format for 2D Gaussian Splatting (2DGS) training.

Pipeline (one forward pass over the input frames):
  1. Gather frames (extract from video @ FRAME_FPS, or copy images from a folder).
  2. Load Pi3 model from a local checkpoint (model.safetensors).
  3. Run inference -> per-view camera_poses (c2w, OpenCV), points (global cloud,
     per-pixel), local_points, conf.
  4. Save raw predictions.npz + dense_cloud.ply (confidence-filtered) for inspection.
  5. Convert to COLMAP text format that 2DGS's scene loader reads:
       source/images/<frame>.png        # the training images
       source/sparse/0/cameras.txt      # PINHOLE intrinsics (one per image)
       source/sparse/0/images.txt       # image_id, w2c (qw qx qy qz tx ty tz), name
       source/sparse/0/points3D.txt     # sampled dense cloud w/ RGB (init for 3DGS)

Pi3 outputs are affine/scale-invariant; intrinsics are NOT produced by the model.
We assume fx=fy=max(W,H), cx=W/2, cy=H/2 (typical FOV) — overridable via --fx/--fy.
Because camera_poses + points come from the SAME model, they are mutually
consistent; 3DGS training only fixes poses+intrinsics and optimizes gaussians, so
even approximate intrinsics produce a plausible reconstruction (gaussians adapt).

Env vars (set by 01_pi3_recon.sh):
  PI3_DIR, PI3_CKPT, OUTPUT_DIR, INPUT, FRAME_FPS, FRAME_MAX, CONF_THRES,
  MAX_POINTS, FX, FY, CX, CY, DEVICE
"""
import os
import sys
import shutil
import argparse
import time
import math
from pathlib import Path

import numpy as np
import torch

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")
VID_EXTS = (".mp4", ".mov", ".avi", ".mkv")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", default=os.environ.get("INPUT"),
                   help="Video file or folder of images (env: INPUT)")
    p.add_argument("--output_dir", default=os.environ.get("OUTPUT_DIR", "./output"),
                   help="Output root (env: OUTPUT_DIR)")
    p.add_argument("--ckpt", default=os.environ.get("PI3_CKPT", "../../model/Pi3/model.safetensors"),
                   help="Pi3 checkpoint path (env: PI3_CKPT)")
    p.add_argument("--pi3_dir", default=os.environ.get("PI3_DIR", "../Pi3"),
                   help="Pi3 official repo (added to sys.path)")
    p.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    p.add_argument("--frame_fps", type=float,
                   default=float(os.environ.get("FRAME_FPS", "10")),
                   help="Frame sampling fps when input is a video (default 10)")
    p.add_argument("--frame_max", type=int,
                   default=int(os.environ.get("FRAME_MAX", "60")),
                   help="Max frames to keep (avoid OOM; default 60)")
    p.add_argument("--conf_thres", type=float,
                   default=float(os.environ.get("CONF_THRES", "0.1")),
                   help="Sigmoid-conf threshold for filtering init points (default 0.1)")
    p.add_argument("--max_points", type=int,
                   default=int(os.environ.get("MAX_POINTS", "100000")),
                   help="Cap on init COLMAP points (default 100k; 3DGS densifies from there)")
    p.add_argument("--fx", type=float, default=None_if_env_unset("FX"),
                   help="Override fx (default: assume fx=max(W,H))")
    p.add_argument("--fy", type=float, default=None_if_env_unset("FY"),
                   help="Override fy (default: same as fx)")
    p.add_argument("--cx", type=float, default=None_if_env_unset("CX"),
                   help="Override cx (default: W/2)")
    p.add_argument("--cy", type=float, default=None_if_env_unset("CY"),
                   help="Override cy (default: H/2)")
    p.add_argument("--no_colmap", action="store_true",
                   help="Skip COLMAP text format export (pose-only mode; e.g. for wan22_rotate step 04)")
    args = p.parse_args()
    if not args.input:
        p.error("--input (or env INPUT) is required")
    if not os.path.isfile(args.ckpt):
        print(f"WARNING: Pi3 ckpt not found at {args.ckpt}", file=sys.stderr)
        print(f"         download from https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors", file=sys.stderr)
    return args


def None_if_env_unset(env_key):
    """For argparse default: read env var, return None if unset (so we can compute later)."""
    v = os.environ.get(env_key)
    return float(v) if v else None


# ---------------------------------------------------------------------------
# 1. Gather frames (extract from video or copy from image folder).
# ---------------------------------------------------------------------------
def gather_frames(input_path, frames_dir, fps, max_frames):
    """Returns list of saved frame filenames (basenames only)."""
    os.makedirs(frames_dir, exist_ok=True)
    if os.path.isfile(input_path) and input_path.lower().endswith(VID_EXTS):
        import cv2
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise OSError(f"Cannot open video: {input_path}")
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
        step = max(int(round(src_fps / max(fps, 0.1))), 1)
        saved = []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                name = f"frame_{len(saved):06d}.png"
                cv2.imwrite(os.path.join(frames_dir, name), frame)
                saved.append(name)
                if len(saved) >= max_frames:
                    break
            idx += 1
        cap.release()
        print(f"  extracted {len(saved)} frames @ ~{fps}fps (step={step}, src_fps={src_fps:.1f})")
        return saved
    if os.path.isdir(input_path):
        files = sorted([f for f in os.listdir(input_path) if f.lower().endswith(IMG_EXTS)])
        saved = []
        for f in files:
            shutil.copy(os.path.join(input_path, f), os.path.join(frames_dir, f))
            saved.append(f)
            if len(saved) >= max_frames:
                break
        print(f"  copied {len(saved)} images from {input_path}")
        return saved
    raise ValueError(f"Input must be a video file or folder: {input_path}")


# ---------------------------------------------------------------------------
# 2. Load Pi3 model.
# ---------------------------------------------------------------------------
def load_pi3_model(ckpt_path, device, pi3_dir):
    sys.path.insert(0, pi3_dir)
    from pi3.models.pi3 import Pi3  # noqa: E402
    print(f"[*] loading Pi3 from {ckpt_path}")
    model = Pi3().to(device).eval()
    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        weight = load_file(ckpt_path)
    else:
        weight = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(weight)
    print("[*] Pi3 model loaded")
    return model


# ---------------------------------------------------------------------------
# 3. Geometry helpers: rotation matrix -> quaternion (scalar-first, COLMAP order).
# ---------------------------------------------------------------------------
def rotmat_to_quat(R):
    """R: 3x3 numpy rotation matrix -> (qw, qx, qy, qz) normalized quaternion."""
    # Shepperd's method, branch on the largest diagonal.
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0  # s = 4 * qw
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n > 0:
        qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return qw, qx, qy, qz


def c2w_to_w2c(c2w):
    """c2w: 4x4 numpy (camera-to-world). Returns w2c 4x4 (world-to-camera).
    Uses rotation transpose for orthonormal matrices (numerically stable)."""
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    w2c = np.eye(4, dtype=np.float64)
    # R is orthonormal: inv(R) = R.T
    w2c[:3, :3] = R.T
    w2c[:3, 3] = -R.T @ t
    return w2c


# ---------------------------------------------------------------------------
# 4. COLMAP text format writers (the 2DGS scene loader reads these when .bin
#    is absent; .txt is human-readable and easy to debug).
# ---------------------------------------------------------------------------
def write_cameras_txt(path, intrinsics_list):
    """intrinsics_list: [(camera_id, model_name, width, height, params), ...]"""
    with open(path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(intrinsics_list)}\n")
        for cam_id, model, w, h, params in intrinsics_list:
            params_str = " ".join(f"{p:.6f}" for p in params)
            f.write(f"{cam_id} {model} {w} {h} {params_str}\n")


def write_images_txt(path, images_list):
    """images_list: [(image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name, points2D), ...]
    points2D: list of (x, y, point3D_id) — can be empty (3DGS doesn't need 2D points)."""
    with open(path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(images_list)}\n")
        for img in images_list:
            img_id, qw, qx, qy, qz, tx, ty, tz, cam_id, name, pts2d = img
            f.write(f"{img_id} {qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} "
                    f"{tx:.8f} {ty:.8f} {tz:.8f} {cam_id} {name}\n")
            if pts2d:
                pts_str = " ".join(f"{x:.4f} {y:.4f} {pid}" for x, y, pid in pts2d)
                f.write(pts_str + "\n")
            else:
                f.write("\n")  # empty points2D line (COLMAP expects one line per image)


def write_points3D_txt(path, points_list):
    """points_list: [(point3D_id, x, y, z, r, g, b, error, track), ...]
    track: list of (image_id, point2D_idx) — empty tracks are fine for 3DGS init."""
    with open(path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(points_list)}\n")
        for pt in points_list:
            pid, x, y, z, r, g, b, err, track = pt
            line = f"{pid} {x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)} {err:.6f}"
            if track:
                line += " " + " ".join(f"{img_id} {p2d_idx}" for img_id, p2d_idx in track)
            f.write(line + "\n")


# ---------------------------------------------------------------------------
# 5. Dense PLY writer (for inspection; uses Pi3's write_ply when available).
# ---------------------------------------------------------------------------
def save_ply_ascii(path, vertices, colors):
    """Plain ASCII PLY (xyz + uchar rgb) — no extra deps, viewable in MeshLab/SuperSplat."""
    n = len(vertices)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for v, c in zip(vertices, colors):
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available; falling back to CPU (will be slow).", file=sys.stderr)
        device = torch.device("cpu")

    output_dir = Path(args.output_dir).resolve()
    frames_dir = output_dir / "frames"
    source_dir = output_dir / "source"
    images_dir = source_dir / "images"
    sparse_dir = source_dir / "sparse" / "0"
    # Only create source/ dirs when exporting COLMAP (default). --no_colmap skips it.
    dirs_to_create = [frames_dir]
    if not args.no_colmap:
        dirs_to_create.extend([images_dir, sparse_dir])
    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)

    # ── 1. Gather frames (extract from video / copy from folder) ───────────
    print(f"[1/5] gathering frames from: {args.input}")
    frame_names = gather_frames(args.input, str(frames_dir), args.frame_fps, args.frame_max)
    if not frame_names:
        sys.exit("ERROR: no frames gathered from input")
    # Copy to source/images/ (2DGS reads training images from here) — skip if --no_colmap
    if not args.no_colmap:
        for name in frame_names:
            shutil.copy(str(frames_dir / name), str(images_dir / name))

    # ── 2. Load Pi3 model + load images as tensor ──────────────────────────
    print(f"[2/5] loading Pi3 model + images")
    model = load_pi3_model(args.ckpt, device, args.pi3_dir)
    from pi3.utils.basic import load_images_as_tensor  # noqa: E402
    from pi3.utils.geometry import depth_normal_edge   # noqa: E402

    print(f"[*] loading images as tensor (Pi3's load_images_as_tensor, interval=1)")
    imgs = load_images_as_tensor(str(frames_dir), interval=1).to(device)  # (N, 3, H, W) in [0,1]
    N, _, H, W = imgs.shape
    print(f"  imgs: {tuple(imgs.shape)}  dtype={imgs.dtype}  range=[{imgs.min():.3f}, {imgs.max():.3f}]")

    # ── 3. Run Pi3 inference ────────────────────────────────────────────────
    print(f"[3/5] running Pi3 inference (this may take ~10-60s depending on N)")
    dtype = (torch.bfloat16 if torch.cuda.is_available()
             and torch.cuda.get_device_capability()[0] >= 8 else torch.float16)
    t0 = time.time()
    # On CPU we skip autocast (no cuda). On CUDA we always autocast (Pi3 example
    # uses bf16 on Ampere+, fp16 otherwise). device.type guards the autocast ctx.
    with torch.no_grad():
        if device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=dtype):
                res = model(imgs[None])  # add batch dim -> (1, N, 3, H, W)
        else:
            res = model(imgs[None])
    t_inf = time.time() - t0
    print(f"  inference done in {t_inf:.1f}s")
    print(f"  outputs: points={tuple(res['points'].shape)}  "
          f"camera_poses={tuple(res['camera_poses'].shape)}  "
          f"conf={tuple(res['conf'].shape)}")

    # ── 4. Save raw predictions + dense PLY (for inspection) ───────────────
    print(f"[4/5] saving raw predictions + dense_cloud.ply")
    pred_np = {
        k: (v.detach().float().cpu().numpy() if isinstance(v, torch.Tensor) else v)
        for k, v in res.items()
    }
    pred_np["images"] = imgs.detach().float().cpu().numpy()  # (N, 3, H, W) in [0,1]
    pred_np["frame_names"] = np.array(frame_names)
    np.savez(output_dir / "predictions.npz", **pred_np)
    print(f"  -> {output_dir / 'predictions.npz'}")

    # Confidence + edge mask (same as Pi3's example.py).
    masks = torch.sigmoid(res["conf"][..., 0]) > args.conf_thres  # (1, N, H, W)
    try:
        non_edge = ~depth_normal_edge(res["local_points"], rtol=0.03, mask=masks)
        masks = torch.logical_and(masks, non_edge)
    except Exception as e:
        print(f"  (depth_normal_edge skipped: {e})", file=sys.stderr)
    masks_np = masks[0].cpu().numpy()  # (N, H, W)

    # Build dense cloud (per-view points + RGB, all views concatenated).
    pts_world = res["points"][0].float().cpu().numpy()  # (N, H, W, 3)
    imgs_hwc = imgs.permute(0, 2, 3, 1).cpu().numpy()    # (N, H, W, 3) in [0,1]
    dense_v, dense_c = [], []
    for i in range(N):
        m = masks_np[i]
        dense_v.append(pts_world[i][m])
        dense_c.append((imgs_hwc[i][m] * 255).clip(0, 255).astype(np.uint8))
    if dense_v:
        dense_v = np.concatenate(dense_v, axis=0)
        dense_c = np.concatenate(dense_c, axis=0)
    else:
        dense_v = np.zeros((1, 3), dtype=np.float32)
        dense_c = np.zeros((1, 3), dtype=np.uint8)
    # Cap the dense PLY (it's just for inspection; not used by 2DGS).
    ply_max = min(len(dense_v), 1_000_000)
    if len(dense_v) > ply_max:
        idx = np.linspace(0, len(dense_v) - 1, ply_max).astype(np.int64)
        dense_v, dense_c = dense_v[idx], dense_c[idx]
    save_ply_ascii(output_dir / "dense_cloud.ply", dense_v, dense_c)
    print(f"  -> {output_dir / 'dense_cloud.ply'}  ({len(dense_v):,} points)")

    # ── 4b. Save human-readable poses.json (always; small, useful for inspection)
    import json
    camera_poses_np = res["camera_poses"][0].float().cpu().numpy()  # (N, 4, 4) c2w
    poses_dict = {
        "frame_names": frame_names,
        "camera_poses_c2w": camera_poses_np.tolist(),  # (N, 4, 4)
        "num_frames": N,
        "image_size": [W, H],
        "convention": "OpenCV (z forward, y down, x right)",
        "note": ("c2w = camera-to-world 4x4 matrix. Invert for w2c "
                 "(COLMAP stores w2c: R_w2c=R_c2w.T, t_w2c=-R_c2w.T@t_c2w)."),
    }
    with open(output_dir / "poses.json", "w") as f:
        json.dump(poses_dict, f, indent=2)
    print(f"  -> {output_dir / 'poses.json'}  ({N} poses, human-readable)")

    # ── 5. Export COLMAP text format (source/sparse/0/) ─────────────────────
    #     Skipped when --no_colmap is set (pose-only mode, e.g. wan22_rotate step 04).
    if args.no_colmap:
        print(f"[5/5] exporting COLMAP text format -> SKIPPED (--no_colmap)")
        print()
        print(f"=== Done. Pi3 pose estimation complete (no COLMAP export). ===")
        print(f"    Predictions:  {output_dir / 'predictions.npz'}")
        print(f"    Dense cloud:  {output_dir / 'dense_cloud.ply'}")
        print(f"    Poses (JSON): {output_dir / 'poses.json'}")
        print(f"    Frames:       {frames_dir}")
        print(f"    (Skipped COLMAP export + 2DGS training — drop --no_colmap to enable)")
        return

    print(f"[5/5] exporting COLMAP text format -> {sparse_dir}")

    # 5a. cameras.txt — one PINHOLE per image (all share the same assumed intrinsics).
    fx = args.fx if args.fx is not None else float(max(W, H))
    fy = args.fy if args.fy is not None else fx
    cx = args.cx if args.cx is not None else W / 2.0
    cy = args.cy if args.cy is not None else H / 2.0
    print(f"  assumed intrinsics: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f} "
          f"(PINHOLE; override via --fx/--fy/--cx/--cy)")

    cameras = [(i + 1, "PINHOLE", W, H, [fx, fy, cx, cy]) for i in range(N)]
    write_cameras_txt(sparse_dir / "cameras.txt", cameras)

    # 5b. images.txt — image_id, w2c quaternion + translation, camera_id, name.
    #     c2w (OpenCV: z forward, y down) -> w2c by inverse. OpenCV convention
    #     matches COLMAP (no axis flip needed).
    camera_poses = res["camera_poses"][0].float().cpu().numpy()  # (N, 4, 4) c2w
    images = []
    for i, name in enumerate(frame_names):
        c2w = camera_poses[i]
        # Numerically stable inversion of an orthonormal c2w (R|t):
        #   w2c.R = c2w.R^T,  w2c.t = -c2w.R^T @ c2w.t
        R = c2w[:3, :3]
        t = c2w[:3, 3]
        R_w2c = R.T
        t_w2c = -R.T @ t
        qw, qx, qy, qz = rotmat_to_quat(R_w2c)
        images.append((i + 1, qw, qx, qy, qz,
                       float(t_w2c[0]), float(t_w2c[1]), float(t_w2c[2]),
                       i + 1, name, []))  # empty points2D (3DGS doesn't need them)
    write_images_txt(sparse_dir / "images.txt", images)

    # 5c. points3D.txt — sampled dense cloud w/ RGB (used as 3DGS init gaussians).
    #     Deduplicate by voxel downsampling (Open3D) if available, else random sample.
    print(f"  building init point cloud (max {args.max_points:,} pts)...")
    if len(dense_v) > args.max_points:
        try:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(dense_v.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(dense_c.astype(np.float64) / 255.0)
            # Voxel size auto-tuned so output ~= max_points (scene extent / 100).
            extent = float(np.linalg.norm(dense_v.max(0) - dense_v.min(0)))
            voxel = max(extent / 200.0, 1e-4)
            pcd = pcd.voxel_down_sample(voxel)
            if len(pcd.points) > args.max_points:
                # Still too many — random sample.
                idx = np.random.default_rng(0).choice(
                    len(pcd.points), size=args.max_points, replace=False)
                pts = np.asarray(pcd.points)[idx]
                cols = (np.asarray(pcd.colors)[idx] * 255).clip(0, 255).astype(np.uint8)
            else:
                pts = np.asarray(pcd.points)
                cols = (np.asarray(pcd.colors) * 255).clip(0, 255).astype(np.uint8)
        except ImportError:
            print("  (open3d not available; using random sampling instead of voxel downsample)")
            idx = np.random.default_rng(0).choice(
                len(dense_v), size=args.max_points, replace=False)
            pts = dense_v[idx]
            cols = dense_c[idx]
    else:
        pts = dense_v
        cols = dense_c
    points3D = [(i + 1, float(p[0]), float(p[1]), float(p[2]),
                 c[0], c[1], c[2], 0.0, [])
                for i, (p, c) in enumerate(zip(pts, cols))]
    write_points3D_txt(sparse_dir / "points3D.txt", points3D)

    extent = (pts.max(0) - pts.min(0)) if len(pts) else np.zeros(3)
    print(f"  -> {sparse_dir / 'cameras.txt'}  ({N} PINHOLE cameras)")
    print(f"  -> {sparse_dir / 'images.txt'}   ({N} images w/ c2w->w2c extrinsics)")
    print(f"  -> {sparse_dir / 'points3D.txt'} ({len(pts):,} init points; "
          f"extent=[{extent[0]:.2f}, {extent[1]:.2f}, {extent[2]:.2f}])")

    print()
    print(f"=== Done. Pi3 reconstruction exported to COLMAP format. ===")
    print(f"    Source dir (pass to 2DGS train.py -s): {source_dir}")
    print(f"    Output dir:                           {output_dir}")
    print(f"    Next: bash pi3_3dgs/02_train_2dgs.sh  (or run_all.sh will call it)")


if __name__ == "__main__":
    main()
