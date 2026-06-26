#!/usr/bin/env python3
"""Render 3DGS .ply files to mp4 along a spiral camera trajectory (gsplat).

For each .ply in PLY_INPUT (a file or a folder), render FRAMES frames along a
helical sweep around the object and write <stem>.mp4 to VIDEOS_DIR/INPUT_NAME/
(same nesting logic as 02_run_inference.sh: output dir / input folder name).
Also writes <stem>.png (first frame) for a quick camera-convention check.

Env vars (set by 03_render_video.sh):
  PLY_INPUT, VIDEOS_DIR, INPUT_NAME, DEVICE,
  WIDTH, HEIGHT, FOV, TURNS, ELEV, FRAMES, FPS, UP_AXIS, RADIUS_SCALE, VIEWMAT_C2W

If frames come out black / sideways, try: UP_AXIS=y, or VIEWMAT_C2W=1.
"""
import math
import os
import sys
import time

import numpy as np
import torch
import gsplat
import imageio
from plyfile import PlyData

PLY_INPUT = os.environ.get("PLY_INPUT") or "output"
VIDEOS_DIR = os.environ.get("VIDEOS_DIR") or "videos"
INPUT_NAME = os.environ.get("INPUT_NAME") or os.path.basename(os.path.normpath(PLY_INPUT)) or "input"
VIDEO_OUT_DIR = os.path.join(VIDEOS_DIR, INPUT_NAME)

DEVICE = os.environ.get("DEVICE", "cuda")
WIDTH = int(os.environ.get("WIDTH", "1080"))
HEIGHT = int(os.environ.get("HEIGHT", "720"))
FOV = math.radians(float(os.environ.get("FOV", "60")))
TURNS = float(os.environ.get("TURNS", "2"))
ELEV = math.radians(float(os.environ.get("ELEV", "30")))
FRAMES = int(os.environ.get("FRAMES", "81"))
FPS = int(os.environ.get("FPS", "15"))
UP_AXIS = os.environ.get("UP_AXIS", "z").lower()
RADIUS_SCALE = float(os.environ.get("RADIUS_SCALE", "1.15"))
C2W = os.environ.get("VIEWMAT_C2W", "0") == "1"

FOCAL = WIDTH / (2 * math.tan(FOV / 2))
C0 = 0.28209479


def cross(a, b):
    return torch.stack([a[1] * b[2] - a[2] * b[1],
                        a[2] * b[0] - a[0] * b[2],
                        a[0] * b[1] - a[1] * b[0]], dim=0)


def lookat_w2c(eye, target, up):
    f = target - eye
    f = f / f.norm()
    s = cross(f, up)
    s = s / s.norm()
    u = cross(s, f)
    R = torch.stack([s, u, -f], dim=0)  # OpenGL: cam looks down -Z
    M = torch.eye(4, device=eye.device)
    M[:3, :3] = R
    M[:3, 3] = -R @ eye
    return M


def load_ply(path):
    ply = PlyData.read(path)
    v = ply["vertex"]
    xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], -1)
    scales = np.exp(np.stack([np.asarray(v["scale_0"]), np.asarray(v["scale_1"]), np.asarray(v["scale_2"])], -1))
    quats = np.stack([np.asarray(v["rot_0"]), np.asarray(v["rot_1"]), np.asarray(v["rot_2"]), np.asarray(v["rot_3"])], -1)
    qn = np.linalg.norm(quats, axis=-1, keepdims=True)
    qn[qn == 0] = 1
    quats = quats / qn
    opacity = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"])))
    fdc = np.stack([np.asarray(v["f_dc_0"]), np.asarray(v["f_dc_1"]), np.asarray(v["f_dc_2"])], -1)
    colors = np.clip(0.5 + C0 * fdc, 0, 1)
    to = lambda a: torch.from_numpy(a).float().to(DEVICE)
    return to(xyz), to(scales), to(quats), to(opacity), to(colors)


def gsplat_render(means, quats, scales, opacities, colors, viewmats, Ks):
    """Call gsplat.rasterize, trying the (Ks) signature then (focals)."""
    dev = means.device
    W = torch.tensor([WIDTH], device=dev)
    H = torch.tensor([HEIGHT], device=dev)
    try:
        return gsplat.rasterize(means, quats, scales, opacities, colors, viewmats, Ks, W, H)
    except TypeError:
        pass
    try:
        focal = torch.tensor([float(Ks[0, 0, 0])], device=dev)
        return gsplat.rasterize(means, quats, scales, opacities, colors, viewmats, focal, W, H)
    except TypeError:
        import inspect
        print("[!] gsplat.rasterize signature:", inspect.signature(gsplat.rasterize), file=sys.stderr)
        raise


def render_one(path, out_mp4):
    means, scales, quats, opacities, colors = load_ply(path)
    center = (means.amin(0) + means.amax(0)) / 2.0
    radius = float((means.amax(0) - means.amin(0)).norm()) / 2.0
    fov_v = 2 * math.atan(HEIGHT / (2 * FOCAL))
    dist = max(radius / math.tan(fov_v / 2) * RADIUS_SCALE, 1e-3)
    up = torch.tensor([0.0, 0.0, 1.0], device=DEVICE) if UP_AXIS == "z" else torch.tensor([0.0, 1.0, 0.0], device=DEVICE)
    K = torch.tensor([[FOCAL, 0.0, WIDTH / 2.0], [0.0, FOCAL, HEIGHT / 2.0], [0.0, 0.0, 1.0]], device=DEVICE)

    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
    writer = imageio.get_writer(out_mp4, fps=FPS, codec="libx264", quality=8, macro_block_size=1)
    t0 = time.time()
    for i in range(FRAMES):
        t = i / max(FRAMES - 1, 1)
        theta = t * 2 * math.pi * TURNS
        phi = -ELEV + 2 * ELEV * t
        d = torch.tensor([math.cos(theta) * math.cos(phi), math.sin(theta) * math.cos(phi), math.sin(phi)], device=DEVICE)
        eye = center + dist * d
        vm = lookat_w2c(eye, center, up)
        if C2W:
            vm = torch.inverse(vm)
        out = gsplat_render(means, quats, scales, opacities, colors, vm.unsqueeze(0), K.unsqueeze(0))
        renders = out[0] if isinstance(out, (tuple, list)) else out
        img = renders[0].clamp(0, 1).detach().cpu().numpy()
        if img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
            img = np.transpose(img, (1, 2, 0))
        img = (np.clip(img[:, :, :3], 0, 1) * 255).astype(np.uint8)
        writer.append_data(img)
        if i == 0:
            imageio.imwrite(out_mp4.replace(".mp4", ".png"), img)
    writer.close()
    dt = time.time() - t0
    print(f"    rendered {FRAMES} frames in {dt:.1f}s ({dt / FRAMES:.2f}s/frame)")
    return dt


def main():
    p = PLY_INPUT
    if os.path.isdir(p):
        plys = sorted(f for f in os.listdir(p) if f.lower().endswith(".ply"))
        plys = [os.path.join(p, f) for f in plys]
    elif os.path.isfile(p) and p.lower().endswith(".ply"):
        plys = [p]
    else:
        sys.exit(f"ERROR: PLY_INPUT not a .ply file or folder: {p}")
    if not plys:
        sys.exit(f"ERROR: no .ply in {p}")

    print(f"[*] {len(plys)} ply(s) -> {VIDEO_OUT_DIR}/  ({WIDTH}x{HEIGHT}, {FRAMES}f@{FPS}fps, "
          f"turns={TURNS}, elev={math.degrees(ELEV):.0f}°, up={UP_AXIS})")
    times = []
    for i, ply in enumerate(plys, 1):
        out = os.path.join(VIDEO_OUT_DIR, os.path.splitext(os.path.basename(ply))[0] + ".mp4")
        print(f"[{i}/{len(plys)}] {os.path.basename(ply)}  ->  {os.path.relpath(out)}")
        try:
            times.append(render_one(ply, out))
        except Exception as e:
            print(f"    ! failed: {e}", file=sys.stderr)
    print(f"[*] done. videos in {VIDEO_OUT_DIR}")
    if times:
        print(f"[*] render time: avg {sum(times)/len(times):.1f}s, min {min(times):.1f}s, max {max(times):.1f}s over {len(times)} ply(s)")


if __name__ == "__main__":
    main()
