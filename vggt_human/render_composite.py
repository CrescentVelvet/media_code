#!/usr/bin/env python3
"""三模型合成渲染验证：scene + bodies + heads(warped) 合成 vs GT。

每个模型单独渲染（各自独立 ply），head 按逐帧 warp，然后 alpha 合成：
  C = head_rgb + (1-head_a)·[body_rgb + (1-body_a)·scene_rgb]
合成前提：三区域 mask 互斥（100% 验证过），跨模型遮挡由 alpha 链自然处理。
输出：合成 mosaic + 全帧 PSNR + 头部区域内 PSNR。
"""
import os, sys, json, math
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as _Rot

os.environ.setdefault("OMP_NUM_THREADS", "8")

RESULTS = os.environ.get("RESULTS_DIR", "/mnt/d/output/vggt_human_ms")
ITERS = os.environ.get("ITERS", "4000")
N_VIS = int(os.environ.get("N_VIS", "6"))
OUT_DIR = f"{RESULTS}/03i_composite_vis"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/repos/gaussian-splatting"))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from render_head_compare import load_gs, MiniGS, warp_gs, restore_gs, qvec2rotmat_np
from gaussian_renderer import render
from scene.cameras import Camera
from argparse import Namespace
from face_center_3d import parse_colmap_cameras
from pathlib import Path

PIDS = ["00", "01", "02"]


def make_camera(v, W, H):
    FoVx = 2 * math.atan(W / (2 * v["fx"]))
    FoVy = 2 * math.atan(H / (2 * v["fy"]))
    return Camera(resolution=(W, H), colmap_id=0,
                  R=qvec2rotmat_np(v["qvec"]).T, T=np.asarray(v["tvec"]),
                  FoVx=FoVx, FoVy=FoVy, depth_params=None,
                  image=Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8)),
                  invdepthmap=None, image_name=v["stem"], uid=0,
                  data_device="cuda")


def render_rgb_alpha(g, cam, pipe, bg):
    """渲染两次：正常 RGB + 染白 alpha。
    ⚠️ 不能用 rgb.mean(0) 当 alpha——黑头发/深色衣服 RGB≈0 但 alpha=1，
    会严重低估。染白（_features_dc=1.7724539, opacity raw=10）后 RGB=alpha。
    """
    with torch.no_grad():
        rgb = render(cam, g, pipe, bg)["render"].clamp(0, 1)
        # 染白渲染取 alpha
        dc0 = g._features_dc.clone()
        op0 = g._opacity.clone()
        g._features_dc = torch.full_like(dc0, 1.7724539)
        g._opacity = torch.full_like(op0, 10.0)
        a = render(cam, g, pipe, bg)["render"].clamp(0, 1).mean(0, keepdim=True)
        g._features_dc = dc0
        g._opacity = op0
    return rgb, a


def main():
    fit = json.loads(open(f"{RESULTS}/03e_head_3dmm/head_fit.json").read())
    views = {v["stem"]: v for v in parse_colmap_cameras(f"{RESULTS}/03b_source_ba")}

    pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False,
                     antialiasing=False, debug=False)
    bg = torch.zeros(3, device="cuda")

    # 加载 7 个模型
    models = {}
    g_scene = MiniGS(load_gs(f"{RESULTS}/03i_scene/point_cloud/iteration_{ITERS}/point_cloud.ply"))
    models["scene"] = g_scene
    heads = {}
    for pid in PIDS:
        heads[pid] = MiniGS(load_gs(f"{RESULTS}/03i_head_p{pid}/point_cloud/iteration_{ITERS}/point_cloud.ply"))
        models[f"body{pid}"] = MiniGS(load_gs(f"{RESULTS}/03i_body_p{pid}/point_cloud/iteration_{ITERS}/point_cloud.ply"))
    print("models loaded:", len(models) + len(heads))

    stems_all = sorted(views.keys())
    step = max(1, len(stems_all) // N_VIS)
    picked = stems_all[::step][:N_VIS]

    os.makedirs(OUT_DIR, exist_ok=True)
    psnrs_full, psnrs_head = [], []
    rows = []
    for stem in picked:
        v = views[stem]
        W, H = int(v["W"]), int(v["H"])
        gt_files = list(Path(f"{RESULTS}/03b_source_ba/images").glob(f"{stem}.*"))
        if not gt_files: continue
        gt = np.asarray(Image.open(gt_files[0]).convert("RGB"), dtype=np.float32) / 255.0
        cam = make_camera(v, W, H)

        # scene 渲染（最底层）
        c_scene, _ = render_rgb_alpha(g_scene, cam, pipe, bg)
        comp = c_scene.clone()
        acc_a = torch.zeros(1, H, W, device="cuda")

        # body 渲染叠加（中间层）
        for pid in PIDS:
            cb, ab = render_rgb_alpha(models[f"body{pid}"], cam, pipe, bg)
            comp = comp * (1 - ab * (1 - acc_a)) + cb * (ab * (1 - acc_a))
            acc_a = torch.clamp(acc_a + ab * (1 - acc_a), 0, 1)

        # head 渲染叠加（最上层，逐帧 warp）
        for pid in PIDS:
            fr = fit["persons"].get(pid)
            if fr is None: continue
            pf = fr["per_frame"].get(stem)
            if pf is None:
                continue
            R_ref = np.asarray(fr["R_ref"]); t_ref = np.asarray(fr["t_ref"])
            R_f = np.asarray(pf["R"]).reshape(3, 3)
            t_f = np.asarray(pf["t"]).reshape(3)
            A = R_f @ R_ref.T
            b = t_f - A @ t_ref
            g = heads[pid]
            x0, r0 = warp_gs(g, A, b)
            ch, ah = render_rgb_alpha(g, cam, pipe, bg)
            restore_gs(g, x0, r0)
            a_h = ah * (1 - acc_a)
            comp = comp * (1 - a_h) + ch * a_h
            acc_a = torch.clamp(acc_a + a_h, 0, 1)

        # 全帧 PSNR
        gt_t = torch.tensor(gt, device="cuda").permute(2, 0, 1)
        mse = ((comp - gt_t) ** 2).mean()
        p_full = float(-10 * np.log10(mse.item() + 1e-8))
        psnrs_full.append(p_full)

        # head 区域内 PSNR（任一 head mask）
        best_head_psnr = None
        for pid in PIDS:
            mpath = f"{RESULTS}/03i_region_masks/{stem}.p{pid}.head.png"
            if not os.path.isfile(mpath): continue
            m = np.asarray(Image.open(mpath).convert("L"), dtype=np.float32) / 255.0
            if m.max() < 0.1: continue
            m_t = torch.tensor(m, device="cuda")[None]
            mse_h = (((comp - gt_t) ** 2) * m_t).sum() / (m_t.sum() * 3 + 1e-8)
            p_h = float(-10 * np.log10(mse_h.item() + 1e-8))
            best_head_psnr = p_h if best_head_psnr is None else max(best_head_psnr, p_h)
        if best_head_psnr is not None:
            psnrs_head.append(best_head_psnr)

        comp_np = comp.permute(1, 2, 0).cpu().numpy()
        side = np.concatenate([gt, comp_np], axis=1)
        rows.append((stem, side, p_full))

    psnrs_full = np.array(psnrs_full)
    psnrs_head = np.array(psnrs_head)
    print(f"\n=== 三模型合成渲染（{len(rows)} 帧）===")
    print(f"  全帧 PSNR : med={np.median(psnrs_full):.2f}  mean={psnrs_full.mean():.2f}  min={psnrs_full.min():.2f}")
    print(f"  head 区 PSNR: med={np.median(psnrs_head):.2f}  mean={psnrs_head.mean():.2f}" if len(psnrs_head) else "  head 区: n/a")

    if rows:
        th = min(r.shape[0] for _, r, _ in rows)
        mosaic = np.concatenate([r[:th] for _, r, _ in rows], axis=0)
        img = Image.fromarray((np.clip(mosaic, 0, 1) * 255).astype(np.uint8))
        total = Image.new("RGB", (mosaic.shape[1], mosaic.shape[0] + 34), (10, 10, 10))
        total.paste(img, (0, 34))
        dr = ImageDraw.Draw(total)
        dr.text((10, 8), f"composite [GT | render]  full-PSNR med={np.median(psnrs_full):.2f} dB", fill=(255, 255, 255))
        out = f"{OUT_DIR}/composite_mosaic.png"
        total.save(out)
        print(f"\n🖼️ {out}")
        for s, _, p in rows:
            print(f"  {s}: {p:.2f} dB")


if __name__ == "__main__":
    main()
