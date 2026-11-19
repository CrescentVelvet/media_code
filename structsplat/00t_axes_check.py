#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
structsplat/00t_axes_check.py — 验收 axes 模式：为什么朝向残差还是 90°？

背景（2026-11-20）：
  把 align_gt_to_model 从中心 Kabsch 换成朝向 Kabsch 后，端到端 PSNR 从 7.30 → 10.94 dB，
  与 00o 离线扫描的「A2_axes transp=1 kswap=0 → 10.94dB」完全吻合，说明策略切换生效。
  但新加的朝向自检仍报 90.00°，而且中心残差反而从 0.033 涨到 0.858 —— 这不合逻辑：
  如果 R 真错了 90°，绝对渲染不该有 10.94；如果 R 对了，残差该接近 0。

本脚本把每一层量都摊开，判断究竟是「指标算错」还是「对齐真的只对到 up-to-gauge」：
  ① axes Kabsch 的 fit_resid（方向残差，无量纲）
  ② 三种残差口径：
     - abs    : Rm @ R_pred^T       —— 含全局规范（gauge）
     - rel    : 两两相对旋转 (R_i R_0^T vs R'_i R'_0^T) —— gauge 无关
     - axis   : 逐轴 |Rm·e - R_pred·e| —— 分轴看是哪一轴错
  ③ 中心残差（axes 模式下中心本来就不该闭合，因为 t 只用质心对上）
  ④ 把 Rm 与 R_pred 的 9 个元素并排打出来，肉眼判断是否差一个轴置换

用法：
  python 00t_axes_check.py --cache ~/output/structsplat_results/_cache/hand01_gt4_asp.pt \
      --transforms "/mnt/d/dataset/.../transforms.json"
"""

import argparse
import os
import sys

import torch

SD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SD)

import infer as I  # noqa: E402


def rot_angle_deg(A, B):
    """逐视角 A_i @ B_i^T 的旋转角（度），A/B 均为 (S,3,3)。"""
    M = A @ B.transpose(-1, -2)
    cos = ((M.diagonal(dim1=1, dim2=2).sum(-1) - 1.0) / 2.0).clamp(-1, 1)
    return torch.rad2deg(torch.acos(cos))


def rel_rot_err(A, B):
    """
    gauge 无关的相对旋转误差：
      参考视角 0，比 A_i A_0^T 与 B_i B_0^T 的角度。
    全局再叠一个任意旋转都不影响这个量 → 用来区分「全局规范错」和「形状错」。
    """
    out = []
    refA = A[0]
    refB = B[0]
    for i in range(1, A.shape[0]):
        RA = A[i] @ refA.transpose(-1, -2)
        RB = B[i] @ refB.transpose(-1, -2)
        M = RA @ RB.transpose(-1, -2)
        cos = ((M.diagonal().sum() - 1.0) / 2.0).clamp(-1, 1)
        out.append(float(torch.rad2deg(torch.acos(cos))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--transforms", required=True)
    ap.add_argument("--no-undistort", action="store_true")
    args = ap.parse_args()

    d = torch.load(args.cache, map_location="cpu", weights_only=False)
    pred_w2c = d["pred_w2c"].float()
    pred_K = d["pred_K"].float()
    paths = d["paths"]
    new_h, new_w = int(d["new_h"]), int(d["new_w"])
    print(f"缓存: {len(paths)} 图  输入 {new_h}×{new_w}  pred_w2c {tuple(pred_w2c.shape)}")

    undist = None if args.no_undistort else I.probe_distortion(args.transforms)
    gt_raw, Ks, matched = I.load_gt_cameras(args.transforms, paths, new_h, new_w,
                                            undist=undist)
    print(f"GT 相机匹配 {len(matched)}/{len(paths)}")

    print("\n" + "=" * 78)
    print("pred_w2c det:", [round(float(x), 4) for x in torch.linalg.det(pred_w2c[:, :3, :3])])
    print("gt_raw  det:", [round(float(x), 4) for x in torch.linalg.det(gt_raw[:, :3, :3])])

    for mode in ("center", "axes"):
        al, info = I.align_gt_to_model(pred_w2c, gt_raw, Ks, pred_K, mode=mode)
        Rm = al[:, :3, :3]
        Rp = pred_w2c[:, :3, :3]
        print("\n" + "=" * 78)
        print(f"mode = {mode}")
        print(f"  尺度 s = {info['scale_gt_to_model']:.5f}")
        print(f"  fit_resid（方向/中心拟合残差，无量纲）= {info['fit_resid']:.6f}")
        print(f"  中心残差 RMS/max = {info['center_rms']:.4f} / {info['center_max']:.4f}")
        print(f"  abs 旋转差 = {[round(float(x), 3) for x in rot_angle_deg(Rm, Rp)]}")
        print(f"  rel 旋转差(对 v0) = {[round(x, 3) for x in rel_rot_err(Rm, Rp)]}")
        # 分轴：看 Rm 的每一行（相机基在模型系）与 Rp 对应行的夹角
        for ax, name in enumerate(("右 x", "下 y", "前 z")):
            a = Rm[:, ax, :]
            b = Rp[:, ax, :]
            a = a / a.norm(dim=1, keepdim=True).clamp_min(1e-9)
            b = b / b.norm(dim=1, keepdim=True).clamp_min(1e-9)
            cos = (a * b).sum(-1).clamp(-1, 1)
            print(f"  {name} 轴夹角 = {[round(float(x), 2) for x in torch.rad2deg(torch.acos(cos))]}")
        print(f"  Rm[0] =\n{Rm[0]}")
        print(f"  Rp[0] =\n{Rp[0]}")

    # 直接看 raw GT 与 pred 的朝向差（未对齐），确认 90° 是「本来就差」还是「对齐引入」
    print("\n" + "=" * 78)
    print("参考：raw GT（未对齐）与 pred 的朝向")
    print(f"  abs 旋转差 = {[round(float(x), 3) for x in rot_angle_deg(gt_raw[:, :3, :3], pred_w2c[:, :3, :3])]}")
    print(f"  rel 旋转差(对 v0) = {[round(x, 3) for x in rel_rot_err(gt_raw[:, :3, :3], pred_w2c[:, :3, :3])]}")


if __name__ == "__main__":
    main()
