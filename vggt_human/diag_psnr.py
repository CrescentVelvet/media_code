import os, sys, json, math
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as _Rot

RESULTS = "/mnt/d/output/vggt_human_ms"
PID = "00"
HEAD_DIR = f"{RESULTS}/03e_head_3dmm"
TRAIN_DIR = f"{RESULTS}/03i_head_p00"
sys.path.insert(0, "/mnt/c/code/media_code/vggt_human")
sys.path.insert(0, os.path.expanduser("~/repos/gaussian-splatting"))
os.chdir("/mnt/c/code/media_code/vggt_human")
from render_head_compare import load_gs, MiniGS, warp_gs, restore_gs, qvec2rotmat_np
from gaussian_renderer import render
from scene.cameras import Camera
from argparse import Namespace
from face_center_3d import parse_colmap_cameras
from pathlib import Path

fit = json.loads(open(f"{HEAD_DIR}/head_fit.json").read())
fr = fit["persons"][PID]
R_ref = np.asarray(fr["R_ref"]); t_ref = np.asarray(fr["t_ref"])
pf = fr["per_frame"]
views = {v["stem"]: v for v in parse_colmap_cameras(f"{RESULTS}/03b_source_ba")}

d_init = load_gs(f"{HEAD_DIR}/head_gs_p{PID}.ply")
g = MiniGS(d_init)
pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False, antialiasing=False, debug=False)
bg = torch.zeros(3, device="cuda")

def make_camera(v, W, H):
    FoVx = 2 * math.atan(W / (2 * v["fx"]))
    FoVy = 2 * math.atan(H / (2 * v["fy"]))
    return Camera(resolution=(W, H), colmap_id=0,
                  R=qvec2rotmat_np(v["qvec"]).T, T=np.asarray(v["tvec"]),
                  FoVx=FoVx, FoVy=FoVy, depth_params=None,
                  image=Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8)),
                  invdepthmap=None, image_name=v["stem"], uid=0, data_device="cuda")

stems = sorted(pf.keys())[::12]
for stem in stems:
    mpath = f"{RESULTS}/03i_region_masks/{stem}.p{PID}.head.png"
    if not os.path.isfile(mpath): continue
    mask = np.asarray(Image.open(mpath).convert("L"), dtype=np.float32)/255.0
    if mask.max() < 0.1: continue
    gt_files = list(Path(f"{RESULTS}/03b_source_ba/images").glob(f"{stem}.*"))
    if not gt_files: continue
    gt = np.asarray(Image.open(gt_files[0]).convert("RGB"), dtype=np.float32)/255.0
    H, W = gt.shape[:2]
    R_f = np.asarray(pf[stem]["R"]).reshape(3,3); t_f = np.asarray(pf[stem]["t"]).reshape(3)
    A = R_f @ R_ref.T; b = t_f - A @ t_ref
    v = views[stem]
    cam = make_camera(v, W, H)
    x0, r0 = warp_gs(g, A, b)
    with torch.no_grad():
        img = render(cam, g, pipe, bg)["render"].clamp(0,1)
    restore_gs(g, x0, r0)
    img_np = img.permute(1,2,0).cpu().numpy()
    # 渲染自身 alpha（亮度代理）：render 在 bg=0 时 RGB 即 premultiplied
    a_hat = img_np.mean(2)
    core = mask > 0.7; edge = (mask > 0.05) & ~core
    # 核心区内：render 均值 vs gt 均值（逐通道）
    rc = img_np[core]; gc = gt[core]
    # alpha 检查：核心区渲染亮度（低=半透明）
    print(f"{stem}: core px={core.sum()}  render_rgb mean=({rc[:,0].mean():.2f},{rc[:,1].mean():.2f},{rc[:,2].mean():.2f})  gt mean=({gc[:,0].mean():.2f},{gc[:,1].mean():.2f},{gc[:,2].mean():.2f})")
    print(f"   a_hat core: med={np.median(a_hat[core]):.3f} p10={np.percentile(a_hat[core],10):.3f}   edge: med={np.median(a_hat[edge]):.3f}")
    # gt 头部亮度（深色=头发）
    dark = (gc.mean(1) < 0.25).mean()
    print(f"   gt 核心区暗像素占比（头发）: {dark*100:.0f}%")
