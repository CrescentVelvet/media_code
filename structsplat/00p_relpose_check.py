# -*- coding: utf-8 -*-
"""
00p_relpose_check.py — 决定性判据：GT(对齐后) 与 pred 的**相对位姿**是否一致

为什么这是决定性判据（2026-11-20）：
  自检 PSNR 把两件事混在一起：①相机对不对 ②高斯的几何对不对。
  pose-free 自洽 20.5dB 是**构造使然**（高斯就是用它自己的相机+深度反投影出来的），
  不代表几何正确；GT 分支 11dB 也可能是几何本身不一致，不一定是相机错。

  相对位姿是**相机是否传对**的纯净判据，与高斯完全无关：
      对每对视角 (i,j)，比较  pred 的 R_ij = R_i·R_jᵀ  与 GT(对齐后) 的 R'_ij
      以及 相机中心间距比。
  若相对旋转都差 <2°、间距比都 ≈1 → 相机传递正确，PSNR 差距归因于几何。

用法： python 00p_relpose_check.py <cache.pt> <transforms.json>
"""
import os, sys, json, itertools
import numpy as np
import torch

SD = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SD)
import infer as I
import importlib.util as ilu
sp = ilu.spec_from_file_location("k00", os.path.join(SD, "00k_offline_align.py"))
K = ilu.module_from_spec(sp); sp.loader.exec_module(K)


def centers(w):
    return -torch.einsum("sji,sj->si", w[:, :3, :3], w[:, :3, 3])


def rel_rot(Ra, Rb):
    R = Ra @ Rb.transpose(1, 2)
    tr = R.diagonal(dim1=1, dim2=2).sum(-1)
    return torch.rad2deg(torch.arccos(((tr - 1) / 2).clamp(-1, 1)))


def main():
    cache, tpath = sys.argv[1], sys.argv[2]
    d = torch.load(cache, map_location="cpu", weights_only=False)
    P = d["pred_w2c"].float()
    paths = d["paths"]
    new_h, new_w = int(d["new_h"]), int(d["new_w"])
    pred_K = d["pred_K"].float()

    undist = I.probe_distortion(tpath)
    import importlib
    o = importlib.util.spec_from_file_location("o", os.path.join(SD, "00o_joint_scan.py"))
    om = importlib.util.module_from_spec(o); o.loader.exec_module(om)

    # 决定性对比：多种约定 + A2_axes，看相对位姿误差
    print("=" * 74)
    print("相对位姿对齐误差（相机传递正确性的纯净判据，与高斯无关）")
    print("=" * 74)
    for gl2cv, transp, kswap in itertools.product([0, 1], repeat=3):
        G_raw, Kk = om.build_gt(tpath, paths, new_h, new_w, undist,
                                bool(gl2cv), bool(transp), bool(kswap))
        G_raw = G_raw.float(); Kk = Kk.float()
        s_, R_, t_, _ = K.solve_rot_from_axes(P, G_raw)
        R_ = R_.to(torch.float32); t_ = t_.to(torch.float32)
        Ga = K.apply_sim3(G_raw, s_, R_, t_)

        cp, cg = centers(P), centers(Ga)
        rot_err, dist_err = [], []
        for i, j in itertools.combinations(range(P.shape[0]), 2):
            Ri = P[i, :3, :3] @ P[j, :3, :3].transpose(0, 1)
            Rg = Ga[i, :3, :3] @ Ga[j, :3, :3].transpose(0, 1)
            tr = (Ri @ Rg.transpose(0, 1)).diagonal().sum()
            rot_err.append(float(torch.rad2deg(torch.arccos(((tr - 1) / 2).clamp(-1, 1)))))
            dp, dg = (cp[i] - cp[j]).norm(), (cg[i] - cg[j]).norm()
            dist_err.append(float((dg / dp.clamp_min(1e-9))))
        print(f"  gl2cv={gl2cv} transp={transp} kswap={kswap}"
              f" | 相对旋转误差 {np.max(rot_err):7.3f}° (均 {np.mean(rot_err):.3f})"
              f" | 间距比 {np.mean(dist_err):.4f}"
              f" (min {np.min(dist_err):.3f} max {np.max(dist_err):.3f})")


if __name__ == "__main__":
    main()
