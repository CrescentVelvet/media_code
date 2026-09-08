#!/usr/bin/env python3
"""分解合成渲染的全帧误差来源：scene/body/head 区域内 vs 区域过渡带。

背景：三模型区域 PSNR 都 20+ dB，但合成全帧只有 15.65。
假设：误差集中在 ① body alpha 雾污染 scene（30k densify 后高斯越界）
     ② head/body 边界 ③ scene 在人走区域的空洞透出。
"""
import os, sys, math
import numpy as np, torch
from PIL import Image
from pathlib import Path

RESULTS = os.environ.get("RESULTS_DIR", "/mnt/d/output/vggt_human_ms")
HEAD_ITERS = os.environ.get("HEAD_ITERS", "4000")
BODY_ITERS = os.environ.get("BODY_ITERS", "30000")
SCENE_ITERS = os.environ.get("SCENE_ITERS", "30000")
N_VIS = int(os.environ.get("N_VIS", "6"))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/repos/gaussian-splatting"))
from render_head_compare import load_gs, MiniGS, warp_gs, restore_gs, qvec2rotmat_np
from render_composite import render_rgb_alpha
from gaussian_renderer import render
from fit_head_3dmm import parse_colmap_cameras
from argparse import Namespace
from scene.cameras import Camera

import json
fit = json.loads(open(f"{RESULTS}/03e_head_3dmm/head_fit.json").read())

pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False, antialiasing=False, debug=False)
bg = torch.zeros(3, device="cuda")
views = {v["stem"]: v for v in parse_colmap_cameras(f"{RESULTS}/03b_source_ba")}
mask_dir = f"{RESULTS}/03i_region_masks"

# 找三 mask 齐全的帧
cands = sorted(p.name[:-len(".scene.png")] for p in Path(mask_dir).glob("*.scene.png"))
step = max(1, len(cands) // N_VIS)
picked = cands[::step][:N_VIS]

# 区域内/过渡带统计
zone_names = ["scene区内", "body区内", "head区内", "head-body过渡", "body-scene过渡", "head-scene过渡", "无mask区"]
zone_se = {z: [] for z in zone_names}  # squared error 累计
zone_px = {z: [] for z in zone_names}

for stem in picked:
    gt_files = list(Path(f"{RESULTS}/03b_source_ba/images").glob(f"{stem}.*"))
    if not gt_files or stem not in views: continue
    gt = np.asarray(Image.open(gt_files[0]).convert("RGB"), dtype=np.float32) / 255.0
    v = views[stem]; W, H = int(v["W"]), int(v["H"])
    if gt.shape[0] != H: gt = np.asarray(Image.fromarray((gt*255).astype(np.uint8)).resize((W, H)), dtype=np.float32)/255.

    # 各区域 mask
    m_scene = np.asarray(Image.open(f"{mask_dir}/{stem}.scene.png").convert("L"), dtype=np.float32)/255. > 0.5
    m_body = np.zeros_like(m_scene); m_head = np.zeros_like(m_scene)
    for pid in ["00","01","02"]:
        p = Path(mask_dir)/f"{stem}.p{pid}.body.png"
        if p.exists():
            m_body |= np.asarray(Image.open(p).convert("L"), dtype=np.float32)/255. > 0.5
        p = Path(mask_dir)/f"{stem}.p{pid}.head.png"
        if p.exists():
            m_head |= np.asarray(Image.open(p).convert("L"), dtype=np.float32)/255. > 0.5

    # 渲染合成（复用 render_composite 的函数）
    FoVx = 2*math.atan(W/(2*v["fx"])); FoVy = 2*math.atan(H/(2*v["fy"]))
    cam = Camera(resolution=(W,H), colmap_id=0, R=qvec2rotmat_np(v["qvec"]).T, T=np.asarray(v["tvec"]),
                 FoVx=FoVx, FoVy=FoVy, depth_params=None,
                 image=Image.fromarray(np.zeros((H,W,3),dtype=np.uint8)), invdepthmap=None,
                 image_name=stem, uid=0, data_device="cuda")

    g_scene = MiniGS(load_gs(f"{RESULTS}/03i_scene/point_cloud/iteration_{SCENE_ITERS}/point_cloud.ply"))
    rgb_s, a_s = render_rgb_alpha(g_scene, cam, pipe, bg)
    comp = rgb_s.clone()
    acc_a = torch.zeros(1, H, W, device="cuda")
    body_alphas = []
    for pid in ["00","01","02"]:
        g_b = MiniGS(load_gs(f"{RESULTS}/03i_body_p{pid}/point_cloud/iteration_{BODY_ITERS}/point_cloud.ply"))
        rgb_b, a_b = render_rgb_alpha(g_b, cam, pipe, bg)
        body_alphas.append(a_b)
        comp = comp * (1 - a_b * (1 - acc_a)) + rgb_b * (a_b * (1 - acc_a))
        acc_a = torch.clamp(acc_a + a_b * (1 - acc_a), 0, 1)
    for pid in ["00","01","02"]:
        g_h = MiniGS(load_gs(f"{RESULTS}/03i_head_p{pid}/point_cloud/iteration_{HEAD_ITERS}/point_cloud.ply"))
        # head warp
        fr = fit["persons"][pid]
        warped = False
        if stem in fr["per_frame"]:
            R_ref = np.asarray(fr["R_ref"]); t_ref = np.asarray(fr["t_ref"])
            R_f = np.asarray(fr["per_frame"][stem]["R"]).reshape(3,3); t_f = np.asarray(fr["per_frame"][stem]["t"]).reshape(3)
            A = R_f @ R_ref.T; b = t_f - A @ t_ref
            x0, r0 = warp_gs(g_h, A, b)
            warped = True
        rgb_h, a_h = render_rgb_alpha(g_h, cam, pipe, bg)
        if warped:
            restore_gs(g_h, x0, r0)
        a_h = a_h * (1 - acc_a)
        comp = comp * (1 - a_h) + rgb_h * a_h
        acc_a = torch.clamp(acc_a + a_h, 0, 1)

    # 误差图
    gt_t = torch.tensor(gt, device="cuda").permute(2,0,1)
    se = ((comp - gt_t)**2).mean(0).cpu().numpy()  # H,W

    # 过渡带：膨胀区域 - 腐蚀区域（边界 8px）
    import cv2
    k = np.ones((17,17), np.uint8)
    def band(m):
        if m.sum() == 0: return np.zeros_like(m)
        return cv2.dilate(m.astype(np.uint8), k).astype(bool) & ~cv2.erode(m.astype(np.uint8), k).astype(bool)

    zones = {
        "scene区内": m_scene & ~m_body & ~m_head,
        "body区内": m_body & ~m_head,
        "head区内": m_head,
        "head-body过渡": band(m_head) & m_body & ~m_head,
        "body-scene过渡": band(m_body) & m_scene & ~m_body & ~m_head,
        "head-scene过渡": band(m_head) & m_scene & ~m_body & ~m_head,
        "无mask区": ~m_scene & ~m_body & ~m_head,
    }
    for z, mm in zones.items():
        n = mm.sum()
        if n > 0:
            zone_se[z].append((se*mm).sum()); zone_px[z].append(n)

    # 该帧 body alpha 在 scene 区的污染
    a_body_total = torch.clamp(sum(body_alphas), 0, 1)
    pol = (a_body_total.mean(0).cpu().numpy() > 0.1) & m_scene
    if pol.sum() > 0 and m_scene.sum() > 0:
        print(f"  {stem}: body alpha>0.1 污染 scene 区 {pol.sum()} px ({pol.sum()/m_scene.sum()*100:.1f}% of scene)")

print("\n=== 全帧误差空间分解（加权 MSE → 等效 PSNR）===")
total_se, total_px = 0, 0
for z in zone_names:
    if not zone_px[z]: continue
    mse = sum(zone_se[z]) / sum(zone_px[z])
    px = sum(zone_px[z])
    psnr = -10*np.log10(mse + 1e-8)
    print(f"  {z:12s}: {px:>9,} px  ({px/sum(sum(v) for v in zone_px.values())*100:5.1f}%)  mse={mse:.5f}  等效PSNR={psnr:6.2f} dB")
    total_se += sum(zone_se[z]); total_px += px
print(f"  {'合计':12s}: {total_px:>9,} px  mse={total_se/total_px:.5f}  全帧PSNR={-10*np.log10(total_se/total_px+1e-8):.2f} dB")
