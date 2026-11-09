#!/usr/bin/env python3
"""audit_31 渲染活锁的逐级二分：从纯 PyTorch 到完整管线，定位卡死层级。

STAGE 环境变量：
  0  纯 PyTorch matmul 30s（GPU  sanity，与 19_orbit 共存验证）
  1  只触发 gsplat JIT 编译（import _backend）
  2  最小 rasterization：1000 颗随机高斯
  3  真实 PLY 截前 5 万颗（不 warp）
  4  真实 PLY 全量（不 warp，真实 W2C[0]）
  5  完整管线（read_cams + umeyama + warp + W2C[0]）= 复现 audit_31 现场
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_24_five_way_video as a24

O = os.path.expanduser("~/output")
PLY = O + "/da3_gs_results/佳佳抱枕_3v/gs_ply_full/0000.ply"
CAMS = O + "/da3_gs_results/佳佳抱枕_3v_ff/cameras.npz"
GT_CAMS = O + "/da3_results/_captures/佳佳抱枕/cameras.npz"
STAGE = int(os.environ.get("STAGE", "5"))
VIEW = int(os.environ.get("VIEW", "0"))

t0 = time.time()
def mark(msg):
    print(f"[{time.time()-t0:8.1f}s] {msg}", flush=True)

z = np.load(GT_CAMS)
c2w = z["c2w"].astype(np.float64)
K = z["intrinsics"].astype(np.float64)[0]
W, H = 1439, 1919
W2C = np.linalg.inv(c2w)
dev = "cuda"

if STAGE == 0:
    mark("STAGE 0: 纯 PyTorch matmul 30s")
    a = torch.randn(4096, 4096, device=dev)
    for i in range(30):
        a = a @ a * 1e-4
        torch.cuda.synchronize()
        mark(f"  iter {i} ok, |a|={float(a.abs().mean()):.3f}")
    mark("STAGE0_DONE")
    sys.exit(0)

mark("STAGE 1: 触发 gsplat JIT 编译")
from gsplat.cuda import _backend
_C = _backend._C
mark("gsplat 后端加载完成")
if STAGE == 1:
    mark("STAGE1_DONE")
    sys.exit(0)

from gsplat import rasterization
Kt = torch.from_numpy(K).float().to(dev)[None]
bg = torch.ones(1, 3, device=dev)

if STAGE == 2:
    mark("STAGE 2: 最小 rasterization（1000 颗随机高斯）")
    n = 1000
    means = torch.randn(n, 3, device=dev) * 0.3
    quats = torch.nn.functional.normalize(torch.randn(n, 4, device=dev), dim=1)
    scales = torch.rand(n, 3, device=dev) * 0.05 + 0.001
    opac = torch.rand(n, device=dev)
    sh = torch.rand(n, 3, (2 + 1) ** 2, device=dev)
    V = torch.from_numpy(W2C[VIEW]).float().to(dev)[None]
    with torch.no_grad():
        img, alpha, info = rasterization(means, quats, scales, opac, sh, V, Kt, W, H,
                                         sh_degree=2, packed=False, backgrounds=bg)
    torch.cuda.synchronize()
    mark(f"STAGE2_DONE α中位 {float(alpha.median()):.3f}")
    sys.exit(0)

g = a24.read_ply(PLY)
mark(f"PLY 读完 {g['n']:,} 点")

def to_dev(gg):
    return {k: torch.from_numpy(gg[k]).to(dev) for k in ("means", "quats", "scales", "opac", "sh")}

if STAGE == 3:
    mark("STAGE 3: 真实 PLY 截前 5 万颗（不 warp）")
    sub = {k: g[k][:50000] for k in ("means", "quats", "scales", "opac", "sh")}
    V = torch.from_numpy(W2C[VIEW]).float().to(dev)[None]
    G = to_dev(sub)
    with torch.no_grad():
        img, alpha, info = rasterization(G["means"], G["quats"], G["scales"], G["opac"], G["sh"],
                                         V, Kt, W, H, sh_degree=g["deg"], packed=False, backgrounds=bg)
    torch.cuda.synchronize()
    mark(f"STAGE3_DONE α中位 {float(alpha.median()):.3f}")
    sys.exit(0)

if STAGE == 4:
    mark("STAGE 4: 真实 PLY 全量（不 warp，真实 W2C[0]）")
    V = torch.from_numpy(W2C[VIEW]).float().to(dev)[None]
    G = to_dev(g)
    with torch.no_grad():
        img, alpha, info = rasterization(G["means"], G["quats"], G["scales"], G["opac"], G["sh"],
                                         V, Kt, W, H, sh_degree=g["deg"], packed=False, backgrounds=bg)
    torch.cuda.synchronize()
    mark(f"STAGE4_DONE α中位 {float(alpha.median()):.3f} 可见 {int((info['radii']>0).sum()):,}")
    sys.exit(0)

if STAGE == 5:
    mark("STAGE 5: 完整管线（read_cams + umeyama + warp）")
    Cref, dref, upref = a24.read_cams("c2w", GT_CAMS)
    Cx, dx, upx = a24.read_cams("w2c34", CAMS)
    idx3 = [0, 7, 31]
    s, R, t = a24.umeyama(Cx, Cref[idx3])
    mark(f"umeyama: s={s:.4f}")
    g = a24.warp(g, s, R, t)
    G = to_dev(g)
    V = torch.from_numpy(W2C[VIEW]).float().to(dev)[None]
    with torch.no_grad():
        img, alpha, info = rasterization(G["means"], G["quats"], G["scales"], G["opac"], G["sh"],
                                         V, Kt, W, H, sh_degree=g["deg"], packed=False, backgrounds=bg)
    torch.cuda.synchronize()
    mark(f"STAGE5_DONE α中位 {float(alpha.median()):.3f} 可见 {int((info['radii']>0).sum()):,}")
    sys.exit(0)
