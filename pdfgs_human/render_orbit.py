#!/usr/bin/env python3
"""Render a turntable / orbit video of the reconstructed person from PDF-GS gaussians.

03_train_pdfgs.sh only renders the INPUT training views (reconstruction-vs-GT). This
script generates a NOVEL orbit camera path around the person and renders an mp4 —
the standard showcase output for a human 3D-Gaussian reconstruction.

Approach (no image loading — light & fast):
  1. Load the final-phase gaussians via GaussianModel.load_ply (point_cloud.ply).
     Gaussian centroid -> orbit center; PCA -> person's vertical axis (largest
     variance = height); horizontal half-extent -> orbit radius (× ORBIT_RADIUS_MULT).
  2. FoV / image size from the COLMAP cameras.txt (intrinsics are unaffected by any
     normalization, so they match the training FoV).
  3. For each angle theta in [0, 2pi*turns): build a MiniCam at
     center + radius*(cos*u + sin*v) + height*up, looking at the target, up=up.
     MiniCam mirrors scene.cameras.Camera's matrix conventions:
       world_view_transform = w2c.T,  full_proj = wvt @ proj.T.
  4. render(view, gaussians, pipe, bg) per frame -> RGB -> mp4 + frames/.

Run inside the PDF-GS repo (cd $PDFGS_DIR) so scene/ gaussian_renderer/ utils/
arguments/ relative imports resolve, same as 03_train_pdfgs.sh.

Env vars (set by 04_render_orbit.sh):
  SOURCE_DIR, MODEL_DIR, PHASE, ITER, OUTPUT_NAME,
  ORBIT_FRAMES, ORBIT_TURNS, ORBIT_RADIUS_MULT, ORBIT_HEIGHT, UP_AXIS,
  WHITE_BG, FPS, RES, SH_DEGREE, DEVICE
"""
import os
import sys
import math
import glob
from argparse import Namespace

import numpy as np
import torch
import cv2

SOURCE_DIR = os.environ.get("SOURCE_DIR", "")
MODEL_DIR = os.environ.get("MODEL_DIR", "")
PHASE = int(os.environ.get("PHASE", "4"))
ITER = os.environ.get("ITER", "")  # empty = auto-detect max iteration_* dir
OUTPUT_NAME = os.environ.get("OUTPUT_NAME", "orbit")
ORBIT_FRAMES = int(os.environ.get("ORBIT_FRAMES", "120"))
ORBIT_TURNS = float(os.environ.get("ORBIT_TURNS", "1.0"))
ORBIT_RADIUS_MULT = float(os.environ.get("ORBIT_RADIUS_MULT", "3.5"))
ORBIT_HEIGHT = float(os.environ.get("ORBIT_HEIGHT", "0.0"))
UP_AXIS = os.environ.get("UP_AXIS", "").strip().lower()  # ""(auto PCA) | x | y | z
WHITE_BG = os.environ.get("WHITE_BG", "1") == "1"
FPS = int(os.environ.get("FPS", "30"))
RES = os.environ.get("RES", "").strip()  # cap max edge, e.g. "1280"; "" = native
SH_DEGREE = int(os.environ.get("SH_DEGREE", "3"))
DEVICE = os.environ.get("DEVICE", "cuda")


def look_at(eye, target, up):
    """OpenGL-style world-to-camera matrix (camera looks down -Z), consistent with
    3DGS getWorld2View2 / getProjectionMatrix(z_sign=1). Returns 4x4 numpy."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    f = target - eye
    f = f / (np.linalg.norm(f) + 1e-9)        # forward (camera looks along -Z)
    s = np.cross(f, up)
    s = s / (np.linalg.norm(s) + 1e-9)        # right
    u = np.cross(s, f)                         # true up (orthogonal)
    M = np.eye(4, dtype=np.float64)
    M[0, :3] = s
    M[1, :3] = u
    M[2, :3] = -f
    M[:3, 3] = -(M[:3, :3] @ eye)             # translation = -R @ eye
    return M


def read_colmap_intrinsics(sparse_dir):
    """Parse COLMAP cameras.txt -> (FoVx, FoVy, width, height) of the first camera.
    FoV = 2*atan(pixels / (2*focal)). Handles PINHOLE / SIMPLE_* / RADIAL / OPENCV.
    Intrinsics are not affected by NeRF normalization -> match the training FoV."""
    cam_path = os.path.join(sparse_dir, "cameras.txt")
    if not os.path.isfile(cam_path):
        return None
    with open(cam_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            model = parts[1].upper()
            w = int(parts[2]); h = int(parts[3])
            if model == "PINHOLE" or model.startswith("OPENCV") or model == "FULL_OPENCV":
                fx, fy = float(parts[4]), float(parts[5])
            else:  # SIMPLE_PINHOLE / SIMPLE_RADIAL / RADIAL: single focal f
                fx = fy = float(parts[4])
            fovx = 2.0 * math.atan(w / (2.0 * fx))
            fovy = 2.0 * math.atan(h / (2.0 * fy))
            return fovx, fovy, w, h
    return None


def find_final_iter(phase_dir):
    """Auto-pick iteration_<N>/ with the largest N under phase_dir/point_cloud/."""
    pc_root = os.path.join(phase_dir, "point_cloud")
    best = -1
    if os.path.isdir(pc_root):
        for d in glob.glob(os.path.join(pc_root, "iteration_*")):
            try:
                n = int(os.path.basename(d).split("_")[1])
                if n > best:
                    best = n
            except (ValueError, IndexError):
                continue
    return best if best > 0 else None


def main():
    if not SOURCE_DIR:
        sys.exit("❌ SOURCE_DIR not set (COLMAP scene, must match step 03)")
    sparse0 = os.path.join(SOURCE_DIR, "sparse", "0")
    if not os.path.isfile(os.path.join(sparse0, "cameras.txt")):
        sys.exit(f"❌ COLMAP cameras.txt not found at {sparse0}/cameras.txt")

    if not MODEL_DIR:
        sys.exit("❌ MODEL_DIR not set (PDF-GS gaussians root, = step 03 --model_path)")
    phase_dir = os.path.join(MODEL_DIR, f"phase_{PHASE}")
    if not os.path.isdir(phase_dir):
        sys.exit(f"❌ phase dir not found: {phase_dir} (check PHASE / step 03 output)")

    if ITER:
        iteration = int(ITER)
    else:
        iteration = find_final_iter(phase_dir)
        if iteration is None:
            sys.exit(f"❌ no iteration_* dir under {phase_dir}/point_cloud/")
    ply_path = os.path.join(phase_dir, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    if not os.path.isfile(ply_path):
        sys.exit(f"❌ gaussians not found: {ply_path}")

    out_dir = os.path.join(MODEL_DIR, "orbit_render")
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    mp4_path = os.path.join(out_dir, f"{OUTPUT_NAME}_orbit.mp4")

    print("🚀 [orbit render] turntable video from PDF-GS gaussians")
    print(f"  🏋️ gaussians: {ply_path}")
    print(f"  📂 source:    {SOURCE_DIR}")
    print(f"  🎬 output:    {mp4_path}")
    print(f"  📐 frames:    {ORBIT_FRAMES}  turns: {ORBIT_TURNS}  fps: {FPS}")
    print(f"  🎨 white_bg:  {WHITE_BG}  radius_mult: {ORBIT_RADIUS_MULT}"
          f"  height: {ORBIT_HEIGHT}  up: {UP_AXIS or 'auto'}")

    # ── 1. Load gaussians ──────────────────────────────────────────────────
    from gaussian_renderer import GaussianModel, render
    from scene.cameras import MiniCam
    from utils.graphics_utils import getProjectionMatrix

    gaussians = GaussianModel(SH_DEGREE)
    loaded = False
    last = None
    # Match Scene's call signature: load_ply(path, train_test_exp=False). Try a
    # couple of signatures in case PDF-GS's load_ply differs slightly.
    for sig in [(ply_path, False), (ply_path,), (ply_path, -1)]:
        try:
            gaussians.load_ply(*sig)
            loaded = True
            break
        except Exception as e:
            last = e
    if not loaded:
        sys.exit(f"❌ GaussianModel.load_ply failed: {last}")
    gaussians.eval()

    xyz = gaussians.get_xyz.detach().cpu().numpy().astype(np.float64)
    if xyz.shape[0] == 0:
        sys.exit("❌ empty gaussian cloud (0 points)")
    center = xyz.mean(axis=0)

    # PCA -> vertical axis. For a standing person, height is the LARGEST-variance
    # axis -> last eigenvector of cov (eigh returns ascending eigenvalues).
    xyz_c = xyz - center
    cov = np.cov(xyz_c.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    if UP_AXIS in ("x", "y", "z"):
        idx = "xyz".index(UP_AXIS)
        up = np.zeros(3, dtype=np.float64); up[idx] = 1.0
    else:
        up = eigvecs[:, -1]
        up = up / (np.linalg.norm(up) + 1e-9)
    # Two orthonormal vectors spanning the horizontal plane (⊥ up).
    ref = np.array([1.0, 0.0, 0.0])
    if abs(float(up @ ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = np.cross(up, ref); u = u / (np.linalg.norm(u) + 1e-9)
    v = np.cross(up, u);   v = v / (np.linalg.norm(v) + 1e-9)

    # Horizontal half-extent (max |coord| along u/v) -> orbit radius.
    proj_u = xyz_c @ u
    proj_v = xyz_c @ v
    half_w = max(float(np.abs(proj_u).max()), float(np.abs(proj_v).max()))
    half_h = float(np.abs(xyz_c @ up).max())
    radius = ORBIT_RADIUS_MULT * half_w
    if radius < 1e-6:
        sys.exit("❌ estimated orbit radius ~0 (degenerate gaussian cloud?)")
    # Target = orbit center + height offset along up (frames upper body slightly).
    target = center + ORBIT_HEIGHT * half_h * up

    print(f"  📐 center:    {[round(float(c), 4) for c in center]}")
    print(f"  📐 up axis:   {[round(float(c), 4) for c in up]}")
    print(f"  📐 half_w/h:  {half_w:.4f} / {half_h:.4f}  radius: {radius:.4f}")
    print(f"  📐 target:    {[round(float(c), 4) for c in target]}")

    # ── 2. FoV + image size from COLMAP cameras.txt ────────────────────────
    intr = read_colmap_intrinsics(sparse0)
    if intr is None:
        sys.exit(f"❌ could not parse COLMAP cameras.txt at {sparse0}/cameras.txt")
    fovx, fovy, W, H = intr
    if RES:
        scale = float(RES) / max(W, H)
        W = int(round(W * scale)); H = int(round(H * scale))
    print(f"  📐 FoVx/FoVy: {math.degrees(fovx):.1f}/{math.degrees(fovy):.1f}°  size: {W}x{H}")

    # ── 3. Pipeline + background ────────────────────────────────────────────
    pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False,
                     debug=False, antialiasing=False)
    bg = torch.tensor([1.0, 1.0, 1.0] if WHITE_BG else [0.0, 0.0, 0.0],
                      dtype=torch.float32, device=DEVICE)
    if DEVICE == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA not available — render will likely fail (PDF-GS needs CUDA).")

    # 3DGS/PDF-GS store world_view_transform TRANSPOSED (.T) and projection_matrix
    # TRANSPOSED (.T); full_proj = wvt @ proj_T. MiniCam mirrors Camera exactly.
    znear, zfar = 0.01, 100.0
    proj_T = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy
                                 ).transpose(0, 1).to(DEVICE)

    # ── 4. Render orbit frames ──────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(mp4_path, fourcc, FPS, (W, H))
    if not video.isOpened():
        sys.exit(f"❌ cv2.VideoWriter failed to open {mp4_path} (codec mp4v?)")

    angles = np.linspace(0.0, 2.0 * math.pi * ORBIT_TURNS, ORBIT_FRAMES, endpoint=False)
    n = len(angles)
    with torch.no_grad():
        for i, theta in enumerate(angles, 1):
            eye = center + radius * (math.cos(theta) * u + math.sin(theta) * v)
            w2c = look_at(eye, target, up)
            wvt = torch.tensor(w2c, dtype=torch.float32, device=DEVICE).transpose(0, 1)
            fpt = (wvt.unsqueeze(0).bmm(proj_T.unsqueeze(0))).squeeze(0)
            cam = MiniCam(W, H, fovy, fovx, znear, zfar, wvt, fpt)
            out = render(cam, gaussians, pipe, bg, scaling_modifier=1.0,
                         use_trained_exp=False)
            rgb = out["render"]  # (3, H, W) on DEVICE, ~[0,1]
            frame = (rgb.clamp(0, 1).detach().cpu().permute(1, 2, 0).numpy() * 255.0
                     ).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video.write(frame_bgr)
            cv2.imwrite(os.path.join(frames_dir, f"{i:05d}.png"), frame_bgr)
            if i % 10 == 0 or i == n:
                print(f"  [{i}/{n}] theta={math.degrees(theta) % 360:.0f}°")
    video.release()
    print(f"✅ saved: {mp4_path}")
    print(f"  frames: {frames_dir}/*.png")
    print(f"\n🎉 Done.")


if __name__ == "__main__":
    main()
