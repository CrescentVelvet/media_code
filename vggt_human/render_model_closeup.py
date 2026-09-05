#!/usr/bin/env python3
"""用已有 closeup poses json 渲染指定模型在近景视角的图 (对比 04b 基线 vs 06e_v2)。

poses json 自带 W H fx fy cx cy, 无需读 COLMAP。

Env vars:
  GAUSSIAN_DIR  : 3DGS model dir
  ITERATION     : iteration number
  POSES_ROOT    : 含 06c_closeup_poses_p*.json 的目录
  OUT_DIR       : 输出目录
  GS_DIR        : gaussian-splatting repo
"""
import os, sys, json, math, glob
import numpy as np
from pathlib import Path
from PIL import Image

GAUSSIAN_DIR = os.environ.get("GAUSSIAN_DIR", "")
ITERATION = int(os.environ.get("ITERATION", "30000"))
POSES_ROOT = os.environ.get("POSES_ROOT", "")
OUT_DIR = os.environ.get("OUT_DIR", "")
GS_DIR = os.environ.get("GS_DIR", "")
DEVICE = "cuda"

def main():
    if not GAUSSIAN_DIR:
        sys.exit("❌ GAUSSIAN_DIR not set")
    if not POSES_ROOT:
        sys.exit("❌ POSES_ROOT not set")
    if not OUT_DIR:
        sys.exit("❌ OUT_DIR not set")
    if not GS_DIR:
        sys.exit("❌ GS_DIR not set")

    import torch
    sys.path.insert(0, GS_DIR)
    from scene import GaussianModel
    from scene.cameras import Camera
    from gaussian_renderer import render
    from argparse import Namespace

    ply_path = os.path.join(GAUSSIAN_DIR, "point_cloud", f"iteration_{ITERATION}", "point_cloud.ply")
    if not os.path.isfile(ply_path):
        sys.exit(f"❌ PLY not found: {ply_path}")

    sh_degree = 3
    cfg_path = os.path.join(GAUSSIAN_DIR, "cfg_args")
    if os.path.isfile(cfg_path):
        import re
        with open(cfg_path) as f:
            m = re.search(r"sh_degree\s*=\s*(\d+)", f.read())
            if m:
                sh_degree = int(m.group(1))

    print(f"🧑 [render_model_closeup] {ply_path} (sh={sh_degree})")
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(ply_path)
    gaussians.active_sh_degree = sh_degree

    pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False,
                     antialiasing=False, debug=False)
    bg = torch.zeros(3, device=DEVICE)

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    pose_files = sorted(glob.glob(os.path.join(POSES_ROOT, "06c_closeup_poses_p*.json")))
    if not pose_files:
        sys.exit(f"❌ 无 poses json in {POSES_ROOT}")
    print(f"  📂 {len(pose_files)} pose files")

    total = 0
    for pf in pose_files:
        tag = os.path.basename(pf).replace("06c_closeup_poses_", "").replace(".json", "")
        out_sub = os.path.join(OUT_DIR, f"closeup_renders_{tag}")
        Path(out_sub).mkdir(parents=True, exist_ok=True)
        with open(pf) as f:
            poses = json.load(f)
        print(f"\n  👤 {tag}: {len(poses)} views")
        for i, cv in enumerate(poses):
            R = np.array(cv["R"]); T = np.array(cv["T"])
            W = int(cv["W"]); H = int(cv["H"])
            fx = cv["fx"]; fy = cv["fy"]
            name = cv["name"]
            FoVx = 2 * math.atan(W / (2 * fx))
            FoVy = 2 * math.atan(H / (2 * fy))
            dummy = Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8))
            # R 是 w2c, Camera 期望 c2w, 传 R.T (gaussian-splatting 内部会 transpose)
            cam = Camera(resolution=(W, H), colmap_id=0, R=R.T, T=T,
                         FoVx=FoVx, FoVy=FoVy, depth_params=None,
                         image=dummy, invdepthmap=None,
                         image_name=name, uid=i, data_device=DEVICE)
            with torch.no_grad():
                out = render(cam, gaussians, pipe, bg)["render"]
            img = (out.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(img).save(os.path.join(out_sub, name))
            total += 1
    print(f"\n🎉 Done. {total} renders -> {OUT_DIR}")

if __name__ == "__main__":
    main()
