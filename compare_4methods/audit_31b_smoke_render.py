#!/usr/bin/env python3
"""audit_31 渲染活锁的二分冒烟测试：只渲 DA3-GS 3v 的 view 0，带计时。

环境变量：
  MODE   orig（原样） | san（丢非有限 + 距中位 >10m 离群） | halfres（原样但半分辨率）
  VIEW   默认 0
出口：成功渲完打印 RENDER_OK + 耗时；超时被外层 GNU timeout 杀。

设计意图：在 audit_31 全量之前定位 5.5h 活锁的充要条件（离群高斯？全分辨率？）。
"""
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_24_five_way_video as a24

O = os.path.expanduser("~/output")
PLY = O + "/da3_gs_results/佳佳抱枕_3v/gs_ply_full/0000.ply"
CAMS = O + "/da3_gs_results/佳佳抱枕_3v_ff/cameras.npz"
GT_CAMS = O + "/da3_results/_captures/佳佳抱枕/cameras.npz"
MODE = os.environ.get("MODE", "orig")
VIEW = int(os.environ.get("VIEW", "0"))

t0 = time.time()
z = np.load(GT_CAMS)
c2w = z["c2w"].astype(np.float64)
K = z["intrinsics"].astype(np.float64)[0]
Cref, dref, upref = a24.read_cams("c2w", GT_CAMS)
W2C = np.linalg.inv(c2w)

g = a24.read_ply(PLY)
n0 = g["n"]
if MODE == "san":
    means = g["means"]
    med = np.median(means, axis=0)
    keep = (np.isfinite(means).all(1) & np.isfinite(g["scales"]).all(1)
            & np.isfinite(g["quats"]).all(1) & np.isfinite(g["opac"])
            & (np.linalg.norm(means - med, axis=1) <= 10.0))
    for k in ("means", "quats", "scales", "opac", "sh"):
        g[k] = g[k][keep]
    g["n"] = int(keep.sum())
    print(f"清洗: {n0:,} -> {g['n']:,} (丢 {n0 - g['n']:,})", flush=True)

Cx, dx, upx = a24.read_cams("w2c34", CAMS)
idx3 = [0, 7, 31]  # 与 audit_31 日志一致的最近邻（先对齐再单渲，判据已在主脚本验过）
s, R, t = a24.umeyama(Cx, Cref[idx3])
g = a24.warp(g, s, R, t)
dev = "cuda"
G = {k: torch.from_numpy(g[k]).to(dev) for k in ("means", "quats", "scales", "opac", "sh")}

W, H = 1439, 1919
if MODE == "halfres":
    W, H = 720, 960
    K = K.copy()
    K[:2, :] *= 0.5

from gsplat import rasterization
t1 = time.time()
print(f"准备 {t1-t0:.1f}s，开渲 view {VIEW} ({W}x{H}, MODE={MODE}) …", flush=True)
with torch.no_grad():
    img, alpha, info = rasterization(
        G["means"], G["quats"], G["scales"], G["opac"], G["sh"],
        torch.from_numpy(W2C[VIEW]).float().to(dev)[None],
        torch.from_numpy(K).float().to(dev)[None],
        W, H, sh_degree=g["deg"], packed=False,
        backgrounds=torch.ones(1, 3, device=dev), render_mode="RGB")
t2 = time.time()
print(f"RENDER_OK 渲完 {t2-t1:.1f}s  α中位 {float(alpha.median()):.3f}  "
      f"可见高斯 {int((info['radii'] > 0).sum()):,}/{g['n']:,}", flush=True)
