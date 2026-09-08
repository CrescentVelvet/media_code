#!/usr/bin/env python3
"""head 层合成诊断：head 叠加在 GT 背景 vs 合成背景。

若 head-on-GT 分数高而 head-on-composite 低 → 底层（body/scene）在 head 区泄漏；
若 head-on-GT 也低 → head 层自身 alpha/rgb 有问题。
"""
import os, sys, math, json
import numpy as np, torch
from PIL import Image
from pathlib import Path

RESULTS = os.environ.get("RESULTS_DIR", "/mnt/d/output/vggt_human_ms")
HEAD_ITERS = os.environ.get("HEAD_ITERS", "10000")
N_VIS = int(os.environ.get("N_VIS", "6"))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/repos/gaussian-splatting"))
from render_head_compare import load_gs, MiniGS, warp_gs, restore_gs, qvec2rotmat_np
from gaussian_renderer import render
from fit_head_3dmm import parse_colmap_cameras
from argparse import Namespace
from scene.cameras import Camera

fit = json.loads(open(f"{RESULTS}/03e_head_3dmm/head_fit.json").read())
pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False, antialiasing=False, debug=False)
bg = torch.zeros(3, device="cuda")
views = {v["stem"]: v for v in parse_colmap_cameras(f"{RESULTS}/03b_source_ba")}

stems_all = sorted(views.keys())
step = max(1, len(stems_all) // N_VIS)
picked = stems_all[::step][:N_VIS]

def render_rgb_alpha(g, cam):
    with torch.no_grad():
        rgb = render(cam, g, pipe, bg)["render"].clamp(0, 1)
        dc0 = g._features_dc.clone(); op0 = g._opacity.clone()
        fr0 = g._features_rest.clone() if hasattr(g, "_features_rest") else None
        deg0 = getattr(g, "active_sh_degree", 0)
        g._features_dc = torch.full_like(dc0, 1.7724539)
        if fr0 is not None: g._features_rest = torch.zeros_like(fr0)
        g.active_sh_degree = 0
        a = render(cam, g, pipe, bg)["render"].clamp(0, 1).mean(0, keepdim=True)
        g._features_dc = dc0; g._opacity = op0
        if fr0 is not None: g._features_rest = fr0
        g.active_sh_degree = deg0
    return rgb, a

for pid in ["00", "01", "02"]:
    g = MiniGS(load_gs(f"{RESULTS}/03i_head_p{pid}/point_cloud/iteration_{HEAD_ITERS}/point_cloud.ply"))
    fr = fit["persons"][pid]
    R_ref = np.asarray(fr["R_ref"]); t_ref = np.asarray(fr["t_ref"])
    for stem in picked:
        mp = f"{RESULTS}/03i_region_masks/{stem}.p{pid}.head.png"
        if not os.path.isfile(mp) or stem not in fr["per_frame"]: continue
        m = np.asarray(Image.open(mp).convert("L"), dtype=np.float32)/255.
        if m.max() < 0.1: continue
        v = views[stem]; W, H = int(v["W"]), int(v["H"])
        gt_f = list(Path(f"{RESULTS}/03b_source_ba/images").glob(stem+".*"))
        if not gt_f: continue
        gt = np.asarray(Image.open(gt_f[0]).convert("RGB"), dtype=np.float32)/255.
        gt_t = torch.tensor(gt, device="cuda").permute(2,0,1)
        m_t = torch.tensor(m, device="cuda")[None]
        FoVx = 2*math.atan(W/(2*v["fx"])); FoVy = 2*math.atan(H/(2*v["fy"]))
        cam = Camera(resolution=(W,H), colmap_id=0, R=qvec2rotmat_np(v["qvec"]).T, T=np.asarray(v["tvec"]),
                     FoVx=FoVx, FoVy=FoVy, depth_params=None,
                     image=Image.fromarray(np.zeros((H,W,3),dtype=np.uint8)), invdepthmap=None,
                     image_name=stem, uid=0, data_device="cuda")
        R_f = np.asarray(fr["per_frame"][stem]["R"]).reshape(3,3); t_f = np.asarray(fr["per_frame"][stem]["t"]).reshape(3)
        A = R_f @ R_ref.T; b = t_f - A @ t_ref
        x0, r0 = warp_gs(g, A, b)
        rgb_h, a_h = render_rgb_alpha(g, cam)
        restore_gs(g, x0, r0)

        # 三种背景下的 head 合成
        on_gt = rgb_h * a_h + gt_t * (1 - a_h)                    # head over GT
        on_black = rgb_h * a_h                                     # head over black
        def psnr(img):
            mse = (((img - gt_t)**2) * m_t).sum() / (m_t.sum() * 3 + 1e-8)
            return float(-10*np.log10(mse.item()+1e-8))
        # head 区内 alpha 统计
        a_np = a_h.mean(0).cpu().numpy()
        in_mask = m > 0.5
        a_mean = a_np[in_mask].mean() if in_mask.any() else 0
        a_low = (a_np[in_mask] < 0.5).mean()*100 if in_mask.any() else 0
        print(f"p{pid} {stem}: on_GT={psnr(on_gt):6.2f}  on_black={psnr(on_black):6.2f}  "
              f"head区alpha: mean={a_mean:.3f} <0.5比例={a_low:.1f}%")
