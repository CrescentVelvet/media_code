#!/usr/bin/env python3
"""量化 body_p00 区域内被非自身内容遮挡的像素比例。

body mask = person mask - head alpha（定义时已扣 head），但仍可能被：
1. 其它人 p01/p02 的 person mask 交叉遮挡
2. head alpha 边缘残余（0 < alpha < 1 的软边）
这些像素 03i body_p00 无法渲染（前景高斯在别的模型里），合成时补回。
"""
import os, sys, json, math
import numpy as np, torch
from PIL import Image
from pathlib import Path

RESULTS = os.environ.get("RESULTS_DIR", "/mnt/d/output/vggt_human_ms")
PID = os.environ.get("PID", "00")
N_VIS = int(os.environ.get("N_VIS", "8"))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/repos/gaussian-splatting"))
from render_head_compare import load_gs, MiniGS, warp_gs, restore_gs, qvec2rotmat_np
from gaussian_renderer import render
from fit_head_3dmm import parse_colmap_cameras
from argparse import Namespace
from scene.cameras import Camera

fit = json.loads(open(f"{RESULTS}/03e_head_3dmm/head_fit.json").read())
fr = fit["persons"][PID]
R_ref = np.asarray(fr["R_ref"]); t_ref = np.asarray(fr["t_ref"])

d = load_gs(f"{RESULTS}/03e_head_3dmm/head_gs_p{PID}.ply")
g_head = MiniGS(d, recolor=1.7724539)   # 染白测 alpha
g_head._opacity = torch.full_like(g_head._opacity, 10.0)
g_head._features_rest = g_head._features_rest.cuda()  # recolor 路径下 _features_rest 可能在 CPU

pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False, antialiasing=False, debug=False)
bg = torch.zeros(3, device="cuda")
views = {v["stem"]: v for v in parse_colmap_cameras(f"{RESULTS}/03b_source_ba")}

mask_dir = f"{RESULTS}/03i_region_masks"
cands = sorted(p.name[:-len(f".p{PID}.body.png")] for p in Path(mask_dir).glob(f"*.p{PID}.body.png"))
step = max(1, len(cands) // N_VIS)
picked = cands[::step][:N_VIS]

ratios_other, ratios_head, ratios_all = [], [], []
for stem in picked:
    bm = np.asarray(Image.open(f"{mask_dir}/{stem}.p{PID}.body.png").convert("L"), dtype=np.float32) / 255.0
    if bm.max() < 0.1: continue
    area = (bm > 0.5).sum()
    if area == 0: continue
    # 其它人 person mask 遮挡
    occ_other = np.zeros_like(bm, dtype=bool)
    for pid in ["00", "01", "02"]:
        if pid == PID: continue
        p = f"{mask_dir}/{stem}.p{pid}.head.png"
        # person mask 目录在 03_sam3_person_masks
    for pid in ["00", "01", "02"]:
        if pid == PID: continue
        pdir = f"{RESULTS}/03_sam3_person_masks"
        import glob as _g
        hits = _g.glob(f"{pdir}/{stem}*p{pid}*.png") or _g.glob(f"{pdir}/*{stem}*p{pid}*")
        if hits:
            pm = np.asarray(Image.open(hits[0]).convert("L"), dtype=np.float32) / 255.0
            if pm.shape == bm.shape:
                occ_other |= pm > 0.5
    # head alpha 遮挡（>0.3 即算）
    if stem not in fr["per_frame"]:
        print(f"  {stem}: 无 head_fit（跳过 head alpha，只算他人遮挡）")
        r_head = 0.0
        a_head = np.zeros_like(bm)
    else:
        v = views[stem]
        W, H = int(v["W"]), int(v["H"])
        R_f = np.asarray(fr["per_frame"][stem]["R"]).reshape(3,3)
        t_f = np.asarray(fr["per_frame"][stem]["t"]).reshape(3)
        A = R_f @ R_ref.T; b = t_f - A @ t_ref
        FoVx = 2 * math.atan(W / (2 * v["fx"])); FoVy = 2 * math.atan(H / (2 * v["fy"]))
        cam = Camera(resolution=(W, H), colmap_id=0, R=qvec2rotmat_np(v["qvec"]).T, T=np.asarray(v["tvec"]),
                     FoVx=FoVx, FoVy=FoVy, depth_params=None,
                     image=Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8)),
                     invdepthmap=None, image_name=stem, uid=0, data_device="cuda")
        x0, r0 = warp_gs(g_head, A, b)
        with torch.no_grad():
            a_head = render(cam, g_head, pipe, bg)["render"].clamp(0, 1).mean(0).cpu().numpy()
        restore_gs(g_head, x0, r0)

    bm_hard = bm > 0.5
    r_other = (occ_other & bm_hard).sum() / area
    r_head = ((a_head > 0.3) & bm_hard).sum() / area
    ratios_other.append(r_other); ratios_head.append(r_head)
    print(f"  {stem}: body_area={area}  被他人遮挡={r_other*100:.1f}%  被head覆盖={r_head*100:.1f}%")

if ratios_other:
    print(f"\n=== body_p00 区域内遮挡比例（{len(ratios_other)} 帧）===")
    print(f"  被他人遮挡 : mean={np.mean(ratios_other)*100:.1f}%  max={np.max(ratios_other)*100:.1f}%")
    print(f"  被 head 覆盖: mean={np.mean(ratios_head)*100:.1f}%  max={np.max(ratios_head)*100:.1f}%")
