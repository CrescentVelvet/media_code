#!/usr/bin/env python3
"""三模型通用区域内 PSNR 评估：body/scene（无 warp）或 head（带 warp）。

用法（WSL）:
  RESULTS_DIR=/mnt/d/output/vggt_human_ms KIND=body PID=00 ITERS=30000 N_VIS=8 \
  python eval_region_psnr.py
  KIND=head PID=00 ITERS=10000  # head 自动按 head_fit.json 逐帧 warp
"""
import os, sys, json
import numpy as np
import torch
from PIL import Image
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")

RESULTS = os.environ.get("RESULTS_DIR", "/mnt/d/output/vggt_human_ms")
KIND = os.environ.get("KIND", "body")          # head | body | scene
PID = os.environ.get("PID", "00")
ITERS = os.environ.get("ITERS", "30000")
N_VIS = int(os.environ.get("N_VIS", "8"))
TRAIN_DIR = os.environ.get("TRAIN_DIR", f"{RESULTS}/03i_{KIND}_p{PID}" if KIND != "scene" else f"{RESULTS}/03i_scene")
PLY = os.environ.get("PLY", f"{TRAIN_DIR}/point_cloud/iteration_{ITERS}/point_cloud.ply")
OUT_DIR = os.environ.get("OUT_DIR", f"{TRAIN_DIR}/vis_region")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/repos/gaussian-splatting"))

from render_head_compare import load_gs, MiniGS, warp_gs, restore_gs, psnr_region, qvec2rotmat_np


def main():
    from gaussian_renderer import render
    from fit_head_3dmm import parse_colmap_cameras
    from argparse import Namespace
    from scene.cameras import Camera
    import math

    assert os.path.isfile(PLY), f"missing ply: {PLY}"
    d = load_gs(PLY)
    print(f"{KIND} p{PID} iter{ITERS}: {len(d['xyz'])} gaussians")
    g = MiniGS(d)

    pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False,
                     antialiasing=False, debug=False)
    bg = torch.zeros(3, device="cuda")

    # head warp 参数
    warp_map = None
    if KIND == "head":
        fit = json.loads(open(f"{RESULTS}/03e_head_3dmm/head_fit.json").read())
        fr = fit["persons"][PID]
        R_ref = np.asarray(fr["R_ref"]); t_ref = np.asarray(fr["t_ref"])
        warp_map = {}
        for stem, pfd in fr["per_frame"].items():
            R_f = np.asarray(pfd["R"]).reshape(3, 3); t_f = np.asarray(pfd["t"]).reshape(3)
            warp_map[stem] = (R_f @ R_ref.T, t_f - (R_f @ R_ref.T) @ t_ref)

    views = {v["stem"]: v for v in parse_colmap_cameras(f"{RESULTS}/03b_source_ba")}

    # 选有 mask 的帧
    mask_dir = f"{RESULTS}/03i_region_masks"
    if KIND == "scene":
        cands = sorted(p.name[:-len(".scene.png")] for p in Path(mask_dir).glob("*.scene.png"))
    else:
        cands = sorted(p.name[:-len(f".p{PID}.{KIND}.png")]
                       for p in Path(mask_dir).glob(f"*.p{PID}.{KIND}.png"))
    step = max(1, len(cands) // N_VIS)
    picked = cands[::step][:N_VIS]
    print(f"eval {len(picked)}/{len(cands)} frames (region masks)")

    os.makedirs(OUT_DIR, exist_ok=True)
    psnrs, rows = [], []
    for stem in picked:
        mpath = (f"{mask_dir}/{stem}.scene.png" if KIND == "scene"
                 else f"{mask_dir}/{stem}.p{PID}.{KIND}.png")
        mask = np.asarray(Image.open(mpath).convert("L"), dtype=np.float32) / 255.0
        if mask.max() < 0.1:
            continue
        gt_files = list(Path(f"{RESULTS}/03b_source_ba/images").glob(f"{stem}.*"))
        if not gt_files or stem not in views:
            continue
        gt = np.asarray(Image.open(gt_files[0]).convert("RGB"), dtype=np.float32) / 255.0
        v = views[stem]
        W, H = int(v["W"]), int(v["H"])
        FoVx = 2 * math.atan(W / (2 * v["fx"])); FoVy = 2 * math.atan(H / (2 * v["fy"]))
        cam = Camera(resolution=(W, H), colmap_id=0,
                     R=qvec2rotmat_np(v["qvec"]).T, T=np.asarray(v["tvec"]),
                     FoVx=FoVx, FoVy=FoVy, depth_params=None,
                     image=Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8)),
                     invdepthmap=None, image_name=stem, uid=0, data_device="cuda")

        w0 = warp_map.get(stem) if warp_map else None
        x0 = r0 = None
        if w0 is not None:
            x0, r0 = warp_gs(g, *w0)
        with torch.no_grad():
            img = render(cam, g, pipe, bg)["render"].clamp(0, 1)
        if w0 is not None:
            restore_gs(g, x0, r0)

        m_t = torch.tensor(mask, device="cuda")[None]
        p = psnr_region(img, torch.tensor(gt, device="cuda").permute(2, 0, 1), m_t)
        psnrs.append(p)
        # 对比行: gt | render overlay
        img_np = img.permute(1, 2, 0).cpu().numpy()
        ov = gt * 0.4 + img_np * 0.6
        rows.append((stem, np.concatenate([gt, ov], axis=1), p))

    if not rows:
        print("no evaluable frames"); return
    psnrs = np.array(psnrs)
    print(f"\n=== {KIND} p{PID} iter{ITERS} 区域内 PSNR（{len(rows)} 帧）===")
    print(f"  med={np.median(psnrs):.2f}  mean={psnrs.mean():.2f}  min={psnrs.min():.2f}  max={psnrs.max():.2f}")
    for s, _, p in rows:
        print(f"  {s}: {p:.2f} dB")

    th = min(r.shape[0] for _, r, _ in rows)
    mosaic = np.concatenate([r[:th] for _, r, _ in rows], axis=0)
    from PIL import ImageDraw
    im = Image.fromarray((mosaic * 255).astype(np.uint8))
    total = Image.new("RGB", (mosaic.shape[1], mosaic.shape[0] + 34), (10, 10, 10))
    total.paste(im, (0, 34))
    ImageDraw.Draw(total).text(
        (10, 8),
        f"{KIND} p{PID} iter{ITERS} [GT | render overlay]  region PSNR mean={psnrs.mean():.2f} med={np.median(psnrs):.2f} dB",
        fill=(255, 255, 255))
    out = f"{OUT_DIR}/region_psnr_{KIND}_p{PID}_{ITERS}.png"
    total.save(out)
    print(f"\n🖼️ {out}")


if __name__ == "__main__":
    main()
