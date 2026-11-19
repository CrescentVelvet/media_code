#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
structsplat/00v_reflect_align.py — 决定性验证：对齐必须允许反射（det=-1）

── 之前的四层判据把现象锁死了，但没锁到根因 ──
  00t：rel 旋转差 2.88/0.35/1.37°（相对几何很好），abs 旋转差恒 90.000°
  00u：pred det=+1，gt_raw det=-1；换 rot90 约定也只到 38°
  00s：D4 八变换 + 平移搜索全救不回，最佳仍 k=0
矛盾点：相对几何对、绝对差一个固定 90°、且**两种 Kabsch 都停在同一个 90°**。
这是「约束边界」的特征——不是拟合不够好，而是最优解被 det=+1 的约束挡住了。

── 根因链（本轮定论）──
1. 输入是竖拍 1440×1920，模型只吃 H<=W 的横图 → load_image_tensor 做了
   `t.transpose(1,2)`，即**图像转置**。
2. 图像 transpose 在几何上就是**关于对角线的镜像**，它把右手像素系变成左手。
   要让 u'=fx'·x'/z+cx' 成立必须 x'=y、y'=x ⇒ 基变换 det=-1。
3. 因此配对上转置图的 GT 相机必然是**非正常正交阵**（det=-1），代码里乘 M 正是
   这个事实的体现，`gt_raw det = -1.000×4` 是**正确的**，不是 bug。
4. 但 pred 相机 det=+1（模型只会输出正常旋转）。于是「GT 相机」与「pred 相机」
   之间差一个**整体反射**——反射无法表示为任何旋转，Kabsch 只能在 det=+1 的
   流形边界上停住，残差恰好 90°（trace=1 的旋转恰为 90°）。
5. 更本质：模型从**镜像图**重建出的场景，相对真实场景就是镜像的。
   所以模型系与 GT 系之间真实的变换是「相似变换 + 一次反射」，不是纯 Sim(3)。

── 修法 ──
不要强制 det=+1，让 SVD 自己决定（即 R = V·Uᵀ，去掉 D 修正）。
若数据真的是反射关系，R 自然给出 det=-1，此时
    Rm = R_g · Rᵀ  （det = -1 × -1 = +1）→ **GT 相机在模型系里变回正常旋转**
这正是我们要的：让 GT 相机合法，同时承认模型把场景镜像了。

本脚本用两种 mode × 三种约束做 3×2 网格，判据是 abs 旋转残差与 fit_resid：
  proper   : 强制 det=+1（现状，预期 90°）
  reflect  : 让 SVD 自选（预期 det(R)=-1 且残差骤降）

用法： python 00v_reflect_align.py --cache ... --transforms ...
"""
import argparse
import os
import sys

import torch

SD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SD)
import infer as I  # noqa: E402


def centers(w2c):
    return -torch.einsum("sji,sj->si", w2c[:, :3, :3], w2c[:, :3, 3])


def axes(w2c):
    Rw = w2c[:, :3, :3]
    return Rw / Rw.norm(dim=2, keepdim=True).clamp_min(1e-9)


def solve(pred_w2c, gt_w2c, mode, allow_reflect):
    """返回 (aligned_w2c, s, R, t, fit_resid, detR)"""
    P = centers(pred_w2c).double()
    G = centers(gt_w2c).double()
    Pc, Gc = P.mean(0, keepdim=True), G.mean(0, keepdim=True)
    dP, dG = (P - Pc).norm(dim=1).mean(), (G - Gc).norm(dim=1).mean()
    s = float(dP / dG) if float(dG) > 1e-9 else 1.0

    if mode == "center":
        A = (G - Gc) * s
        B = P - Pc
    else:  # axes
        A = axes(gt_w2c).reshape(-1, 3).double()
        B = axes(pred_w2c).reshape(-1, 3).double()

    H = A.transpose(0, 1) @ B
    U, _, Vt = torch.linalg.svd(H)
    if allow_reflect:
        R = Vt.transpose(0, 1) @ U.transpose(0, 1)           # 自选 det
    else:
        d = torch.sign(torch.linalg.det(Vt.transpose(0, 1) @ U.transpose(0, 1)))
        D = torch.diag(torch.tensor([1.0, 1.0, float(d)],
                                    dtype=torch.float64, device=H.device))
        R = Vt.transpose(0, 1) @ D @ U.transpose(0, 1)        # 强制 det=+1

    t = Pc.squeeze(0) - s * (R @ Gc.squeeze(0))
    fit = float((A @ R.transpose(0, 1) - B).norm(dim=1).mean())

    Rf = R.to(pred_w2c.dtype)
    tf = t.to(pred_w2c.dtype)
    Rm = gt_w2c[:, :3, :3] @ Rf.transpose(0, 1)
    tm = s * gt_w2c[:, :3, 3] - torch.einsum("sij,j->si", Rm, tf)
    al = gt_w2c.clone()
    al[:, :3, :3] = Rm
    al[:, :3, 3] = tm
    return al, s, Rf, tf, fit, float(torch.linalg.det(Rf))


def rot_deg(A, B):
    M = A @ B.transpose(-1, -2)
    cos = ((M.diagonal(dim1=1, dim2=2).sum(-1) - 1.0) / 2.0).clamp(-1, 1)
    return torch.rad2deg(torch.acos(cos))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--transforms", required=True)
    args = ap.parse_args()

    d = torch.load(args.cache, map_location="cpu", weights_only=False)
    pred = d["pred_w2c"].float()
    paths = d["paths"]
    nh, nw = int(d["new_h"]), int(d["new_w"])
    gt, Ks, mt = I.load_gt_cameras(args.transforms, paths, nh, nw, undist=None)
    print(f"图 {len(mt)}  pred det={[round(float(x),3) for x in torch.linalg.det(pred[:,:3,:3])]}")
    print(f"        gt   det={[round(float(x),3) for x in torch.linalg.det(gt[:,:3,:3])]}")

    for mode in ("axes", "center"):
        for ar in (False, True):
            al, s, R, t, fit, detR = solve(pred, gt, mode, ar)
            ang = rot_deg(al[:, :3, :3], pred[:, :3, :3])
            det_al = torch.linalg.det(al[:, :3, :3])
            cen = (centers(al) - centers(pred)).norm(dim=1)
            tag = "reflect(自选)" if ar else "proper (强制+1)"
            print(f"\n[{mode:6s}] {tag}")
            print(f"   det(R_est)={detR:+.4f}   fit_resid={fit:.6f}   s={s:.5f}")
            print(f"   det(R_aligned)={[round(float(x),3) for x in det_al]}")
            print(f"   abs旋转差 = {[round(float(x),2) for x in ang]}  均 {float(ang.mean()):.3f}°")
            print(f"   中心RMS = {float(cen.mean()):.4f}")


if __name__ == "__main__":
    main()
