#!/usr/bin/env python3
"""检验 body 区域 PSNR 差距是否来自 mask 边界带：腐蚀 mask 后重算。"""
import os, sys, math
import numpy as np, torch
from PIL import Image
from pathlib import Path
import cv2

RESULTS = os.environ.get("RESULTS_DIR", "/mnt/d/output/vggt_human_ms")
PID = os.environ.get("PID", "00")
N_VIS = int(os.environ.get("N_VIS", "8"))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/repos/gaussian-splatting"))
from render_head_compare import load_gs, MiniGS, psnr_region, qvec2rotmat_np
from gaussian_renderer import render
from fit_head_3dmm import parse_colmap_cameras
from argparse import Namespace
from scene.cameras import Camera

pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False, antialiasing=False, debug=False)
bg = torch.zeros(3, device="cuda")
views = {v["stem"]: v for v in parse_colmap_cameras(f"{RESULTS}/03b_source_ba")}

g_03i = MiniGS(load_gs(f"{RESULTS}/03i_body_p{PID}/point_cloud/iteration_30000/point_cloud.ply"))
g_04b = MiniGS(load_gs(f"{RESULTS}/04b_model_3dgs_ba/point_cloud/iteration_30000/point_cloud.ply"))

mask_dir = f"{RESULTS}/03i_region_masks"
cands = sorted(p.name[:-len(f".p{PID}.body.png")] for p in Path(mask_dir).glob(f"*.p{PID}.body.png"))
step = max(1, len(cands) // N_VIS)
picked = cands[::step][:N_VIS]

def render_one(g, v, W, H, stem):
    FoVx = 2 * math.atan(W / (2 * v["fx"])); FoVy = 2 * math.atan(H / (2 * v["fy"]))
    cam = Camera(resolution=(W, H), colmap_id=0, R=qvec2rotmat_np(v["qvec"]).T, T=np.asarray(v["tvec"]),
                 FoVx=FoVx, FoVy=FoVy, depth_params=None,
                 image=Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8)),
                 invdepthmap=None, image_name=stem, uid=0, data_device="cuda")
    with torch.no_grad():
        return render(cam, g, pipe, bg)["render"].clamp(0, 1)

for erode_px in [0, 5, 12]:
    ps_03i, ps_04b = [], []
    for stem in picked:
        m = np.asarray(Image.open(f"{mask_dir}/{stem}.p{PID}.body.png").convert("L"), dtype=np.float32) / 255.0
        if m.max() < 0.1: continue
        gt_files = list(Path(f"{RESULTS}/03b_source_ba/images").glob(f"{stem}.*"))
        if not gt_files or stem not in views: continue
        gt = np.asarray(Image.open(gt_files[0]).convert("RGB"), dtype=np.float32) / 255.0
        if erode_px > 0:
            k = np.ones((erode_px*2+1, erode_px*2+1), np.uint8)
            m = cv2.erode((m > 0.5).astype(np.uint8), k).astype(np.float32)
        if m.sum() < 100: continue
        v = views[stem]; W, H = int(v["W"]), int(v["H"])
        gt_t = torch.tensor(gt, device="cuda").permute(2, 0, 1)
        m_t = torch.tensor(m, device="cuda")[None]
        img_03i = render_one(g_03i, v, W, H, stem)
        img_04b = render_one(g_04b, v, W, H, stem)
        ps_03i.append(psnr_region(img_03i, gt_t, m_t))
        ps_04b.append(psnr_region(img_04b, gt_t, m_t))
    ps_03i, ps_04b = np.array(ps_03i), np.array(ps_04b)
    print(f"erode={erode_px:2d}px: 03i_body={ps_03i.mean():.2f}  04b_base={ps_04b.mean():.2f}  gap={ps_03i.mean()-ps_04b.mean():+.2f} dB  ({len(ps_03i)}帧)")
