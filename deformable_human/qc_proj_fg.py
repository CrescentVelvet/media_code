#!/usr/bin/env python3
"""qc_proj_fg.py — bg 变形软正则的投影采样 QC（先验证再训练）

把已训练模型的 canonical 高斯中心投影到某个训练相机，
用该帧 SAM3 前景 mask 采样得到逐点 fg 概率，按概率染色叠加回原图。
y 轴方向两种约定各输出一张，目视选对的那张（人身上的点应染红、背景点染蓝）。

用法（WSL，deformable_human env，cwd 无所谓）：
  python deformable_human/qc_proj_fg.py \
      --model_path ~/output/deformable_human_results/hand_motion/model_mask \
      --frame_idx 60
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DG_DIR = os.environ.get(
    "DG_DIR",
    os.path.expanduser("~/repos/Deformable-3D-Gaussians"))
sys.path.insert(0, DG_DIR)

from arguments import ModelParams, PipelineParams  # noqa: E402
from scene import Scene, GaussianModel  # noqa: E402
from argparse import Namespace  # noqa: E402


def project_sample(xyz, cam, mask_t, flip_y):
    """xyz:(N,3) detach → per-point fg 概率（越界/相机背后 = 0）"""
    N = xyz.shape[0]
    ones = torch.ones(N, 1, device=xyz.device, dtype=xyz.dtype)
    pts_h = torch.cat([xyz, ones], dim=1)                       # N,4
    clip = pts_h @ cam.full_proj_transform.cuda()               # N,4（矩阵已转置，右乘约定）
    w = clip[:, 3:4].clamp_min(1e-6)
    ndc = clip[:, :3] / w
    gx = ndc[:, 0]
    gy = -ndc[:, 1] if flip_y else ndc[:, 1]
    valid = (clip[:, 3] > 0) & (gx.abs() <= 1) & (gy.abs() <= 1)
    grid = torch.stack([gx, gy], dim=1).view(1, 1, N, 2)
    p = F.grid_sample(mask_t, grid, mode="bilinear",
                      padding_mode="zeros", align_corners=False)
    p = p.view(N)
    return torch.where(valid, p, torch.zeros_like(p)), valid, ndc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True,
                    help="含 cfg_args 与 point_cloud/iteration_*/point_cloud.ply 的模型目录")
    ap.add_argument("--frame_idx", type=int, default=60, help="训练相机序号（0-based）")
    ap.add_argument("--out_dir", default=None, help="默认 <model_path>/qc_proj")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.model_path, "qc_proj")
    os.makedirs(out_dir, exist_ok=True)

    # 复用训练产物里的 cfg_args（与训练时完全同口径的场景解析）
    cfg_path = os.path.join(args.model_path, "cfg_args")
    with open(cfg_path) as f:
        cfg_str = f.read().strip()
    dataset = eval(cfg_str.replace("Namespace", "Namespace"))
    assert isinstance(dataset, Namespace)
    if not hasattr(dataset, "mask_dir"):
        dataset.mask_dir = os.path.join(dataset.source_path, "masks")

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)
    # 加载训练后的点云（取最大 iteration）
    pc_dir = os.path.join(args.model_path, "point_cloud")
    iters = sorted(int(d.split("_")[1]) for d in os.listdir(pc_dir))
    ply = os.path.join(pc_dir, f"iteration_{iters[-1]}", "point_cloud.ply")
    gaussians.load_ply(ply)
    print(f"[QC] ply: {ply} ({gaussians.get_xyz.shape[0]} pts)")

    cams = scene.getTrainCameras()
    cam = cams[args.frame_idx]
    print(f"[QC] camera: {cam.image_name} ({cam.image_width}x{cam.image_height})")

    mask_path = os.path.join(dataset.mask_dir, cam.image_name + ".png")
    m = np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
    mask_t = torch.from_numpy(m)[None, None].cuda()  # 1,1,Hm,Wm

    xyz = gaussians.get_xyz.detach().cuda()
    gt = (cam.original_image.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    for flip_y, tag in [(True, "flipY"), (False, "noFlipY")]:
        p, valid, ndc = project_sample(xyz, cam, mask_t, flip_y)
        # 只画相机前方、画面内的点，随机抽 4000 个防糊
        idx = torch.nonzero(valid).squeeze(1)
        if idx.numel() > 4000:
            idx = idx[torch.randperm(idx.numel(), device=idx.device)[:4000]]
        h, w = gt.shape[:2]
        overlay = gt.copy()
        pp = p[idx].cpu().numpy()
        # 绘制与采样用同一坐标系：grid_sample 的 gy=-1 对应图像顶行
        gy = -ndc[:, 1] if flip_y else ndc[:, 1]
        u = ((ndc[idx, 0] + 1) * 0.5 * (w - 1)).long().clamp(0, w - 1).cpu().numpy()
        v = ((gy[idx] + 1) * 0.5 * (h - 1)).long().clamp(0, h - 1).cpu().numpy()
        fg_ratio = float((pp > 0.5).mean())
        for ui, vi, pi in zip(u, v, pp):
            color = (255, 48, 48) if pi > 0.5 else (48, 96, 255)
            overlay[max(0, vi - 2):vi + 3, max(0, ui - 2):ui + 3] = color
        Image.fromarray(overlay).save(os.path.join(out_dir, f"qc_{tag}_{cam.image_name}.jpg"), quality=90)
        print(f"[QC] {tag}: 有效点 {idx.numel()}, fg命中率 {fg_ratio*100:.1f}% "
              f"→ {out_dir}/qc_{tag}_{cam.image_name}.jpg")


if __name__ == "__main__":
    main()
