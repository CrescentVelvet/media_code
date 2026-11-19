# -*- coding: utf-8 -*-
"""
00m_diag_gt_align.py — GT 相机与预测相机的逐视角差异诊断（秒级，用前向缓存）

为什么需要：端到端 PSNR 只给一个标量，7.3dB 到底是「旋转错」「平移错」
还是「K 错」分不出来。本脚本把差异拆成可读的量：

  1. 相机中心距离（对齐后应当很小；大 = Sim(3) 或约定错）
  2. 相对旋转角（每个视角的 R_gt vs R_pred，逐视角报，避免均值掩盖单视角错）
  3. 高斯在 GT 相机下的正深度比例 / 投影画面内比例（最能定位"整体朝向错"）
  4. 用 GT 相机渲染的 PSNR（与 infer.py 自检同口径，便于对照）
"""
import os, sys, math, json
import numpy as np
import torch

SD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SD)

import infer as I


def centers(w2c):
    R, t = w2c[:, :3, :3], w2c[:, :3, 3]
    return -torch.einsum("sji,sj->si", R, t)


def rot_angle_deg(Ra, Rb):
    """Ra, Rb 均为 w2c 的旋转块；返回两者相对旋转角（度）。"""
    R = Ra @ Rb.transpose(1, 2)
    tr = R.diagonal(dim1=1, dim2=2).sum(-1)
    cos = ((tr - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos))


def main():
    cache = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.expanduser("~/output/structsplat_results/_cache/hand01_gt4_asp.pt")
    tpath = sys.argv[2] if len(sys.argv) > 2 else None
    d = torch.load(cache, map_location="cpu", weights_only=False)
    pred_w2c = d["pred_w2c"].double()
    pred_K = d["pred_K"].double()
    means = d["means"].float()
    paths = d["paths"]
    new_h, new_w = int(d["new_h"]), int(d["new_w"])
    print(f"缓存: {cache}")
    print(f"  高斯 {means.shape[0]}  输入 {new_h}×{new_w}  视角 {len(paths)}")
    print(f"  渲染口径 {d.get('render_h')}×{d.get('render_w')}")

    undist = I.probe_distortion(tpath) if tpath else None
    gt_raw, Ks, matched = I.load_gt_cameras(tpath, paths, new_h, new_w, undist=undist)
    gt_raw = gt_raw.double()
    Ks = Ks.double()
    print(f"  GT 相机匹配 {len(matched)}/{len(paths)}")

    # 逐视角：GT 原始（未对齐）下的正深度/画面内
    def cover(w2c, K, tag):
        Rw, tw = w2c[:, :3, :3], w2c[:, :3, 3]
        Xc = torch.einsum("sij,nj->sni", Rw, means.double()) + tw[:, None, :]
        z = Xc[..., 2]
        pos = (z > 1e-6)
        uvw = torch.einsum("sij,snj->sni", K, Xc)
        uv = uvw[..., :2] / uvw[..., 2:3].clamp(min=1e-9)
        inside = (uv[..., 0] >= 0) & (uv[..., 0] < new_w) & \
                 (uv[..., 1] >= 0) & (uv[..., 1] < new_h)
        print(f"  [{tag}] 每视角 正深度% / 画面内%:")
        for i in range(w2c.shape[0]):
            p = 100.0 * pos[i].float().mean().item()
            q = 100.0 * (inside[i] & pos[i]).float().mean().item()
            print(f"      v{i}: 正深度 {p:6.2f}%   画面内 {q:6.2f}%   "
                  f"z中位 {z[i].median().item():+.4f}")
        return

    cover(gt_raw, Ks, "GT 原始")
    aligned, info = I.align_gt_to_model(pred_w2c.float(), gt_raw.float(),
                                        Ks.float(), pred_K.float())
    aligned = aligned.double()
    print(f"  Sim(3): 尺度 {info['scale_gt_to_model']:.4f}  "
          f"中心残差 RMS {info['center_rms']:.4f} max {info['center_max']:.4f}")
    cover(aligned, Ks, "GT 对齐后")

    # 逐视角旋转角与中心距离
    cg, cp = centers(aligned), centers(pred_w2c)
    ang = rot_angle_deg(aligned[:, :3, :3], pred_w2c[:, :3, :3])
    dist = (cg - cp).norm(dim=1)
    print("\n  逐视角 GT(对齐后) vs pred:")
    for i in range(aligned.shape[0]):
        print(f"      v{i}: 旋转差 {ang[i].item():7.3f}°   中心距离 {dist[i].item():.4f}")

    # K 对比
    print("\n  K 对比（fx/fy/cx/cy）：")
    for i in range(min(4, Ks.shape[0])):
        a, b = Ks[i], pred_K[i]
        print(f"      v{i}: GT fx={a[0,0]:.2f} fy={a[1,1]:.2f} cx={a[0,2]:.2f} cy={a[1,2]:.2f}"
              f"  | pred fx={b[0,0]:.2f} fy={b[1,1]:.2f} cx={b[0,2]:.2f} cy={b[1,2]:.2f}"
              f"  fx/fy GT={a[0,0]/a[1,1]:.4f} pred={b[0,0]/b[1,1]:.4f}")


if __name__ == "__main__":
    main()
