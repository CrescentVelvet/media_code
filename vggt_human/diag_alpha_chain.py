#!/usr/bin/env python3
"""测量合成链路中 head 区被 body 层 alpha 阻挡的程度。

对指定帧输出：face 区（head mask 内）各层 alpha 均值：
  acc_a(body 累积) / ah(head clamp 后) / a_h = ah*(1-acc_a)（head 实际贡献）
若 acc_a≈1 → head 层被 body 越界高斯完全阻挡 → face 露 body 垃圾。
"""
import os, sys, json, math
import numpy as np, torch
from PIL import Image
import cv2

RESULTS = os.environ.get("RESULTS_DIR", "/mnt/d/output/vggt_human_ms")
STEM = os.environ.get("STEM", "681533632532078")
CLAMP_DILATE = int(os.environ.get("CLAMP_DILATE", "10"))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/repos/gaussian-splatting"))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from render_head_compare import load_gs, MiniGS, warp_gs, restore_gs, qvec2rotmat_np
from render_composite import make_camera, render_rgb_alpha
from face_center_3d import parse_colmap_cameras
from argparse import Namespace

PIDS = ["00", "01", "02"]
pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False, antialiasing=False, debug=False)
bg = torch.zeros(3, device="cuda")
views = {v["stem"]: v for v in parse_colmap_cameras(f"{RESULTS}/03b_source_ba")}
fit = json.loads(open(f"{RESULTS}/03e_head_3dmm/head_fit.json").read())

v = views[STEM]; W, H = int(v["W"]), int(v["H"])
cam = make_camera(v, W, H)
k = np.ones((CLAMP_DILATE*2+1, CLAMP_DILATE*2+1), np.uint8)

body_masks, head_masks = {}, {}
for pid in PIDS:
    bp = f"{RESULTS}/03i_region_masks/{STEM}.p{pid}.body.png"
    hp = f"{RESULTS}/03i_region_masks/{STEM}.p{pid}.head.png"
    print(f"p{pid}: body_mask={'Y' if os.path.isfile(bp) else 'N'}  head_mask={'Y' if os.path.isfile(hp) else 'N'}")
    if os.path.isfile(bp):
        m = np.asarray(Image.open(bp).convert("L"), dtype=np.float32)/255.
        if m.max() >= 0.1: body_masks[pid] = torch.tensor(cv2.dilate((m>0.3).astype(np.uint8), k).astype(np.float32), device="cuda")[None]
    if os.path.isfile(hp):
        m = np.asarray(Image.open(hp).convert("L"), dtype=np.float32)/255.
        if m.max() >= 0.1: head_masks[pid] = torch.tensor(cv2.dilate((m>0.3).astype(np.uint8), k).astype(np.float32), device="cuda")[None]

acc_a = torch.zeros(1, H, W, device="cuda")
for pid in PIDS:
    g = MiniGS(load_gs(f"{RESULTS}/03i_body_p{pid}/point_cloud/iteration_30000/point_cloud.ply"))
    _, ab = render_rgb_alpha(g, cam, pipe, bg)
    ab_raw = ab.mean(0)
    if pid in body_masks:
        ab = ab * body_masks[pid]
    acc_a = torch.clamp(acc_a + ab*(1-acc_a), 0, 1)
    # 在每个人的 head mask（未膨胀原始 mask）内统计
    hp = f"{RESULTS}/03i_region_masks/{STEM}.p{pid}.head.png"
    if os.path.isfile(hp):
        hm = np.asarray(Image.open(hp).convert("L"), dtype=np.float32)/255.
        hmt = torch.tensor(hm > 0.5, device="cuda")
        if hmt.sum() > 0:
            print(f"  body_p{pid}: 在 head_p{pid} 区内 alpha raw mean={ab_raw[hmt].mean():.3f}  clamp后={ab.mean(0)[hmt].mean():.3f}  (区px={int(hmt.sum())})")

print(f"\nacc_a(body 累积) 完成后:")
for pid in PIDS:
    hp = f"{RESULTS}/03i_region_masks/{STEM}.p{pid}.head.png"
    if not os.path.isfile(hp): continue
    hm = np.asarray(Image.open(hp).convert("L"), dtype=np.float32)/255.
    hmt = torch.tensor(hm > 0.5, device="cuda")
    if hmt.sum() == 0: continue
    print(f"  head_p{pid} 区内: acc_a mean={acc_a.mean(0)[hmt].mean():.3f}  >0.9比例={(acc_a.mean(0)[hmt]>0.9).float().mean():.1%}")

    # head 层自身
    fr = fit["persons"][pid]
    pf = fr["per_frame"].get(STEM)
    if pf is None:
        print(f"    head_p{pid}: 该帧无 per_frame 拟合 → head 层跳过!")
        continue
    R_ref = np.asarray(fr["R_ref"]); t_ref = np.asarray(fr["t_ref"])
    R_f = np.asarray(pf["R"]).reshape(3,3); t_f = np.asarray(pf["t"]).reshape(3)
    A = R_f @ R_ref.T; b = t_f - A @ t_ref
    g = MiniGS(load_gs(f"{RESULTS}/03i_head_p{pid}/point_cloud/iteration_10000/point_cloud.ply"))
    x0, r0 = warp_gs(g, A, b)
    _, ah = render_rgb_alpha(g, cam, pipe, bg)
    restore_gs(g, x0, r0)
    if pid in head_masks:
        ah = ah * head_masks[pid]
    a_h = ah * (1 - acc_a)
    print(f"    head_p{pid}: ah mean={ah.mean(0)[hmt].mean():.3f}  实际贡献 a_h mean={a_h.mean(0)[hmt].mean():.3f}")
