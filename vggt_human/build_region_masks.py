#!/usr/bin/env python3
"""build_region_masks.py — 生成三模型（head / body / scene）监督 mask。

三模型训练的前提：每个模型只在**自己的区域**被监督，这样三个模型才能真正
独立（而不是退化成一个大模型）。区域划分沿用 03h 的 head/body/scene 拆分：

  head  mask : head_gs 逐帧重摆后染白渲染 → 累积 alpha（染白染黑法，零 CUDA）
  body  mask : SAM3 person mask − 该人 head alpha
  scene mask : 1 − 所有人的 person mask 并集

head mask 用「染白染黑法」（用户 09-05 拍板）：head_gs 的 SH_DC 设为
_DC_WHITE=1.7724539（color=1）、opacity=sigmoid(10)≈1，其余高斯不参与；
渲染结果 C = Σ_head αT = 头部高斯的**累积 alpha**（已含遮挡，因为 3DGS
混合本身按深度前到后进行）。

与 head_gs.visualize 的区别：这里要跑**全部训练帧**（不是抽样 6 帧），
且默认半分辨率渲染后上采样（450 次渲染，全分辨率太慢；mask 羽化后
精度反而更好）。

输出（OUT_DIR）：
  {stem}.p{pid}.head.png   软 alpha（0-255），头部累积 alpha
  {stem}.p{pid}.body.png   person mask 去掉头部后的软 mask
  {stem}.scene.png         非人区域

Env:
  RESULTS_DIR  输出根
  SOURCE_DIR   COLMAP 场景（默认 $RESULTS_DIR/03_source）
  FIT_DIR      拟合输出（默认 $RESULTS_DIR/03e_head_3dmm）
  HEAD_GS_DIR  head_gs 目录（默认 $FIT_DIR）
  PERSON_MASKS_DIR  SAM3 person mask（默认 $RESULTS_DIR/03_sam3_person_masks）
  OUT_DIR      输出目录（默认 $RESULTS_DIR/03i_region_masks）
  RENDER_SCALE 渲染分辨率缩放（默认 0.5）
  DEVICE       cuda
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from face_center_3d import parse_colmap_cameras  # noqa: E402
from fit_head_3dmm import qvec2rotmat  # noqa: E402

RESULTS_DIR = os.environ.get("RESULTS_DIR", "/mnt/d/output/vggt_human_ms")
SOURCE_DIR = os.environ.get("SOURCE_DIR", f"{RESULTS_DIR}/03_source")
FIT_DIR = os.environ.get("FIT_DIR", f"{RESULTS_DIR}/03e_head_3dmm")
HEAD_GS_DIR = os.environ.get("HEAD_GS_DIR", f"{FIT_DIR}")
PERSON_MASKS_DIR = os.environ.get(
    "PERSON_MASKS_DIR", f"{RESULTS_DIR}/03_sam3_person_masks")
OUT_DIR = os.environ.get("OUT_DIR", f"{RESULTS_DIR}/03i_region_masks")
RENDER_SCALE = float(os.environ.get("RENDER_SCALE", "1.0"))
DEVICE = os.environ.get("DEVICE", "cuda")
GS_DIR = os.environ.get("GS_DIR", os.path.expanduser("~/repos/gaussian-splatting"))
PIDS = [p.strip() for p in os.environ.get("PIDS", "00,01,02").split(",") if p.strip()]

# SH_DC 使 color = 0.5 + 0.28209479177387814 * DC = 1.0
_DC_WHITE = 1.7724539


def log(msg):
    print(msg, flush=True)


def load_person_mask(stem, pid):
    """→ (H,W) float32 0-1，缺失返回 None。"""
    p = Path(PERSON_MASKS_DIR) / f"{stem}.p{pid}.mask.png"
    if not p.is_file():
        return None
    a = np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0
    return a


def main():
    t0 = time.time()
    sys.path.insert(0, GS_DIR)
    from argparse import Namespace
    from scene import GaussianModel
    from scene.cameras import Camera
    from gaussian_renderer import render

    fit_path = Path(FIT_DIR) / "head_fit.json"
    if not fit_path.is_file():
        sys.exit(f"❌ 缺少拟合结果: {fit_path}")
    fit = json.loads(fit_path.read_text())
    views = {v["stem"]: v for v in parse_colmap_cameras(SOURCE_DIR)}
    log(f"📷 相机 {len(views)} 帧 | 输出 {OUT_DIR} | scale={RENDER_SCALE}")

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False,
                     antialiasing=False, debug=False)
    bg_black = torch.zeros(3, device=DEVICE)

    # ── 1. head mask：head_gs 染白渲染 ────────────────────────────────────
    head_alpha = {}   # {(pid, stem): (H,W) float32}
    for pid in PIDS:
        if pid not in fit["persons"]:
            log(f"  ⚠️ pid={pid} 无拟合结果，跳过")
            continue
        ply = Path(HEAD_GS_DIR) / f"head_gs_p{pid}.ply"
        if not ply.is_file():
            log(f"  ⚠️ 缺少 {ply}，跳过")
            continue
        fr = fit["persons"][pid]
        R_ref = np.asarray(fr["R_ref"], dtype=np.float64)
        t_ref = np.asarray(fr["t_ref"], dtype=np.float64)

        g = GaussianModel(3)
        g.load_ply(str(ply))
        with torch.no_grad():
            g._features_dc.data = torch.full_like(g._features_dc, _DC_WHITE)
            g._features_rest.data = torch.zeros_like(g._features_rest)
            g._opacity.data = torch.full_like(g._opacity, 10.0)
        xyz0 = g.get_xyz.detach().clone()
        rot0 = g.get_rotation.detach().clone()
        log(f"  🎨 pid={pid} head_gs {int(xyz0.shape[0])} 点 | "
            f"{len(fr['per_frame'])} 帧")

        from scipy.spatial.transform import Rotation as _Rot
        for stem, pf in fr["per_frame"].items():
            if stem not in views:
                continue
            v = views[stem]
            W, H = v["W"], v["H"]
            rw, rh = max(int(round(W * RENDER_SCALE)), 8), max(int(round(H * RENDER_SCALE)), 8)

            # 逐帧刚性重摆（head_gs 建在中位姿态）
            R_f = np.asarray(pf["R"], dtype=np.float64).reshape(3, 3)
            t_f = np.asarray(pf["t"], dtype=np.float64).reshape(3)
            A = R_f @ R_ref.T
            b = t_f - A @ t_ref
            xyz_new = (A @ xyz0.cpu().numpy().T).T + b
            qA = _Rot.from_matrix(A)
            q_old = _Rot.from_quat(rot0.cpu().numpy()[:, [1, 2, 3, 0]])
            q_new = (qA * q_old).as_quat()          # x,y,z,w
            rot_new = np.stack([q_new[:, 3], q_new[:, 0], q_new[:, 1], q_new[:, 2]], axis=1)
            g.get_xyz.data = torch.tensor(xyz_new, dtype=xyz0.dtype, device=xyz0.device)
            g.get_rotation.data = torch.tensor(rot_new, dtype=rot0.dtype,
                                               device=xyz0.device)

            FoVx = 2 * math.atan(W / (2 * v["fx"]))
            FoVy = 2 * math.atan(H / (2 * v["fy"]))
            cam = Camera(resolution=(rw, rh), colmap_id=0,
                         R=qvec2rotmat(v["qvec"]).T, T=np.asarray(v["tvec"]),
                         FoVx=FoVx, FoVy=FoVy, depth_params=None,
                         image=Image.fromarray(np.zeros((rh, rw, 3), dtype=np.uint8)),
                         invdepthmap=None, image_name=stem, uid=0,
                         data_device=DEVICE)
            with torch.no_grad():
                rgb = render(cam, g, pipe, bg_black)["render"].clamp(0, 1)
            a = rgb.mean(0).cpu().numpy()            # 累积 alpha (rh,rw)
            # 上采样回原尺寸
            if (rw, rh) != (W, H):
                a = np.asarray(
                    Image.fromarray((a * 255).astype(np.uint8)).resize((W, H),
                                                                       Image.BILINEAR),
                    dtype=np.float32) / 255.0
            head_alpha[(pid, stem)] = a
        del g
        torch.cuda.empty_cache()

    # ── 2. body / scene mask ──────────────────────────────────────────────
    stems_all = sorted(views.keys())
    n_head = n_body = n_scene = 0
    for stem in stems_all:
        pmasks = {}
        for pid in PIDS:
            m = load_person_mask(stem, pid)
            if m is not None:
                pmasks[pid] = m
        if not pmasks:
            continue
        H, W = next(iter(pmasks.values())).shape
        union = np.zeros((H, W), dtype=np.float32)
        for pid, m in pmasks.items():
            union = np.maximum(union, m)
            ha = head_alpha.get((pid, stem))
            if ha is not None and ha.shape == (H, W):
                # head mask
                Image.fromarray((np.clip(ha, 0, 1) * 255).astype(np.uint8)).save(
                    out_dir / f"{stem}.p{pid}.head.png")
                n_head += 1
                # body = person − head
                body = np.clip(m - ha, 0, 1)
            else:
                # 该帧没检出这个人的脸（head_gs 无位姿）→ body 退化为完整
                # person mask。不能跳过：否则这块区域在 head/body/scene
                # 三个模型里都无监督（scene 已排除 person），训练出黑洞。
                body = m
            if body.max() > 0:
                Image.fromarray((body * 255).astype(np.uint8)).save(
                    out_dir / f"{stem}.p{pid}.body.png")
                n_body += 1
        # scene = 1 − 所有人并集
        scene = np.clip(1.0 - union, 0, 1)
        Image.fromarray((scene * 255).astype(np.uint8)).save(
            out_dir / f"{stem}.scene.png")
        n_scene += 1

    log(f"💾 head {n_head} | body {n_body} | scene {n_scene} → {out_dir}")
    log(f"✅ 完成 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
