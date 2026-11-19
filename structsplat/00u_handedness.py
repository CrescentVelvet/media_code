#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
structsplat/00u_handedness.py — 手性判据：GT det=-1 是「恒定 90.001°」残差的真因？

背景（2026-11-20，本轮决定性发现）：
  00t 摊开三层残差后看到一组自相矛盾的数：
    · rel 旋转差（gauge 无关，v0 为参考）= [2.88, 0.353, 1.373]°   ← 相对几何很好
    · abs 旋转差（Rm @ Rpᵀ）         = [90.001, 90.033, 90.003, 90.015]°  ← 全局固定 90°
    · center / axes 两 mode 的 rel **完全相同** → 对齐只改了全局规范，没改形状
    · pred_w2c det = +1.000×4，gt_raw det = **-1.000×4**  ← 手性不一致
  一个反射无法被任何 proper rotation 抵消：Kabsch 只能在「最接近的合法旋转」停住，
  残差正好 90°（trace(M)=1 的旋转恰对应 90°），且两个 mode 得到同一个 90°。
  这与 00s（D4 八变换 + 平移搜索全救不回，最佳仍是 k=0）互相印证：
  误差不是「错位」而是「镜像」，二面体变换和平移都不改变这个性质。

根因位置（infer.py load_gt_cameras）：
    c2w[:3,:3] = c2w_raw @ FLIP     # FLIP = diag(1,-1,-1)，det=+1 ✓
    c2w[:3,:3] = c2w[:3,:3] @ M     # M = 交换 x<->y，det=**-1** ✗
  为什么 M 一定 det=-1（不是笔误）：**图像 transpose 本身就是 2D 镜像**。
  原图 (u,v) → 转置图 (u'=v, v'=u)，要让 u'=fx'·x'/z+cx' 成立必须 x'=y、y'=x，
  基变换天然 det=-1。即「转置竖图」与「右手相机系」数学上不可兼容。

修法：把「transpose」换成「真旋转 90°」(torch.rot90 k=1)：
  (u,v) → (u'=v, v'=W0-1-u)  ⇒ x'=y, y'=-x  ⇒ A = [[0,1,0],[-1,0,0],[0,0,1]]，det=+1 ✓
  注意 c2w 右乘的是 **Aᵀ**（基变换作用在列向量上 x'=A x 时，R' = R Aᵀ）。
  这里 transpose 的 A 恰好对称所以 Aᵀ=A 蒙对了；rot90 的 A 非对称，必须转置。

本脚本纯数值（不渲染、不跑前馈，秒级），对比三种约定：
  T0  现状 transpose                （det=-1，预期 abs 残差 90°）
  T1  rot90 但错误地仍乘 A（不转置）（det=-1，另一种镜像，预期也是 90° 级）
  T2  rot90 并正确乘 Aᵀ             （det=+1，预期 abs 残差 → 数度）
判据：T2 的 abs 旋转残差与 fit_resid 若显著下降，手性根因即确认。

用法：
  python 00u_handedness.py --cache .../hand01_gt4_asp.pt --transforms .../transforms.json
"""

import argparse
import json
import os
import sys

import torch

SD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SD)
import infer as I  # noqa: E402

FLIP = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=torch.float32))

# 「竖图转横图」的像素变换对应的相机系基变换 A（x' = A x，作用于列向量）
#   transpose : (u,v) -> (u'=v, v'=  u)        => x'= y, y'= x   det = -1  ← 镜像
#   rot90_ccw : (u,v) -> (u'=v, v'=W0-1-u)     => x'= y, y'=-x   det = +1  ← 真旋转
A_TR = torch.tensor([[0.0, 1.0, 0.0],
                     [1.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0]], dtype=torch.float32)
A_R90 = torch.tensor([[0.0, 1.0, 0.0],
                      [-1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0]], dtype=torch.float32)


def rot_angle_deg(A, B):
    """逐视角 A_i B_iᵀ 的旋转角（度）。"""
    M = A @ B.transpose(-1, -2)
    cos = ((M.diagonal(dim1=1, dim2=2).sum(-1) - 1.0) / 2.0).clamp(-1, 1)
    return torch.rad2deg(torch.acos(cos))


def build_gt_rows(transforms, image_paths, new_h, new_w, A, use_At=True):
    """
    用给定的基变换 A 组装 GT w2c。

    c2w_cv = R_raw @ FLIP @ Aᵀ
      · FLIP：GL(相机 y 上/z 后) → CV(y 下/z 前)，det=+1
      · Aᵀ  ：转置/旋转带来的相机系基变换。**必须是右乘的转置**
              （x' = A x  ⇒  R_new = R Aᵀ），transpose 因 A 对称而不易察觉。
    """
    with open(transforms, "r", encoding="utf-8") as f:
        js = json.load(f)
    by_base = {}
    for fr in js.get("frames", []):
        fp = fr.get("file_path") or fr.get("filePath")
        if fp:
            by_base[os.path.basename(fp)] = fr

    c2ws, names = [], []
    for p in image_paths:
        fr = by_base.get(os.path.basename(p))
        if fr is None:
            continue
        R = torch.tensor(fr["transform_matrix"], dtype=torch.float32)[:3, :3].clone()
        R = R @ FLIP[:3, :3]
        R = R @ (A.transpose(0, 1) if use_At else A)
        c2w = torch.eye(4, dtype=torch.float32)
        c2w[:3, :3] = R
        c2w[:3, 3] = torch.tensor(fr["transform_matrix"], dtype=torch.float32)[:3, 3]
        c2ws.append(c2w)
        names.append(os.path.basename(p))
    return torch.stack(c2ws), names


def report(tag, Rm, Rp, s, t, P, G, note=""):
    """打印一组判据。Rm/Rp 为 (S,3,3)。"""
    det = [round(float(x), 4) for x in torch.linalg.det(Rm)]
    ang = rot_angle_deg(Rm, Rp)
    print(f"  det(R_gt)   = {det}")
    print(f"  abs 旋转差  = {[round(float(x), 3) for x in ang]}  均 {float(ang.mean()):.3f}°")
    # 中心残差（对齐后）
    cal = -torch.einsum("sji,sj->si", Rm, t)
    cp = -torch.einsum("sji,sj->si", Rp, torch.zeros_like(t))  # 占位，下面覆盖
    print(f"  {note}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--transforms", required=True)
    args = ap.parse_args()

    d = torch.load(args.cache, map_location="cpu", weights_only=False)
    pred_w2c = d["pred_w2c"].float()
    paths = d["paths"]
    new_h, new_w = int(d["new_h"]), int(d["new_w"])
    print(f"缓存 {len(paths)} 图  输入 {new_h}×{new_w}")
    print(f"pred det = {[round(float(x), 4) for x in torch.linalg.det(pred_w2c[:, :3, :3])]}")

    cases = [
        ("T0 现状 transpose (A det=-1)", A_TR, True),
        ("T1 rot90 但乘 A^T? 误乘 A (det=-1)", A_R90, False),
        ("T2 rot90 正确乘 Aᵀ (det=+1)", A_R90, True),
    ]

    for tag, A, use_At in cases:
        c2ws, names = build_gt_rows(args.transforms, paths, new_h, new_w, A, use_At)
        gt_raw = torch.linalg.inv(c2ws)
        print("\n" + "=" * 74)
        print(tag)
        print(f"  det(R_gt_raw) = {[round(float(x), 4) for x in torch.linalg.det(gt_raw[:, :3, :3])]}")
        for mode in ("axes", "center"):
            al, info = I.align_gt_to_model(pred_w2c, gt_raw, None, None, mode=mode)
            ang = rot_angle_deg(al[:, :3, :3], pred_w2c[:, :3, :3])
            print(f"  [{mode:6s}] s={info['scale_gt_to_model']:.5f} "
                  f"fit_resid={info['fit_resid']:.6f} "
                  f"中心RMS={info['center_rms']:.4f} "
                  f"abs旋转={[round(float(x), 2) for x in ang]} 均 {float(ang.mean()):.3f}°")


if __name__ == "__main__":
    main()
