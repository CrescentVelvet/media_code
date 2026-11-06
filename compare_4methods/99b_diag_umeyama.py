#!/usr/bin/env python3
"""99b · 判定 DA3 导出高斯与「深度图抬起点云」是**口径错位**还是**真几何不一致**

对应关系是确定的：PLY 的第 k 个顶点 = 第 v 个视角、第 p 个保留像素的高斯（view-major），
而 lift(depth_v, pose_v) 给出同一个像素的 3D 位置。于是可以做**带对应点**的 Umeyama 拟合：

  * 若每个视角拟合出的相似变换残差 ≈ 0（毫米级）→ 两套几何只差一个 s/R/t，
    即"口径错位"，一个全局变换就能修好；
  * 若残差仍很大 → 是真的几何不一致（不同视角的壳互相打架），变换救不回来。

同时输出 scale 的三轴分位数（判断高斯是不是"针状"导致拉丝）。

用法：NPZ=... PLY=... OUT=...
"""
import json
import os
from pathlib import Path

import numpy as np
from plyfile import PlyData

NPZ, PLY, OUT = os.environ["NPZ"], os.environ["PLY"], os.environ["OUT"]

d = np.load(NPZ)
ext = d["extrinsics"].astype(np.float64)
K = d["intrinsics"].astype(np.float64)
depth = d["depth"].astype(np.float64)
V, H, W = depth.shape

trim_h, trim_w = int(8 / 256 * H), int(8 / 256 * W)
mask = np.zeros((V, H, W), dtype=bool)
mask[:, trim_h:H - trim_h, trim_w:W - trim_w] = True
mask &= depth <= np.quantile(depth.reshape(V, -1), 0.90, axis=1).reshape(V, 1, 1)
keep = [np.flatnonzero(mask[v].reshape(-1)) for v in range(V)]
counts = np.array([len(k) for k in keep])
bounds = np.concatenate([[0], np.cumsum(counts)])

ply = PlyData.read(PLY)["vertex"]
assert ply.count == counts.sum()
means = np.stack([ply["x"], ply["y"], ply["z"]], 1).astype(np.float64)
scale_log = np.stack([ply[f"scale_{i}"] for i in range(3)], 1).astype(np.float64)
op_logit = np.asarray(ply["opacity"], dtype=np.float64)

# 深度图抬起点云（与 means 一一对应）
us, vs = np.meshgrid(np.arange(W), np.arange(H))
us = us.ravel().astype(np.float64)
vs = vs.ravel().astype(np.float64)
fx, fy = K[0, 0, 0], K[0, 1, 1]
cx, cy = K[0, 0, 2], K[0, 1, 2]
lift_all = np.zeros_like(means)
for v in range(V):
    p = keep[v]
    z = depth[v].reshape(-1)[p]
    x = (us[p] - cx) / fx * z
    y = (vs[p] - cy) / fy * z
    Xc = np.stack([x, y, z, np.ones_like(z)], 1)
    E = np.eye(4)
    E[:3] = ext[v]
    Xw = Xc @ np.linalg.inv(E).T
    lift_all[bounds[v]:bounds[v + 1]] = Xw[:, :3]


def umeyama(src, dst):
    """求相似变换 dst ≈ s R src + t"""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S = src - mu_s
    D = dst - mu_d
    cov = D.T @ S / len(src)
    U, sig, Vt = np.linalg.svd(cov)
    S_fix = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S_fix[2, 2] = -1
    R = U @ S_fix @ Vt
    var_s = (S ** 2).sum() / len(src)
    s = np.trace(np.diag(sig) @ S_fix) / var_s
    t = mu_d - s * R @ mu_s
    return s, R, t


def rot_angle(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


rows = []
for v in range(V):
    s, e = bounds[v], bounds[v + 1]
    src = lift_all[s:e]
    dst = means[s:e]
    sc, R, t = umeyama(src, dst)
    pred = sc * src @ R.T + t
    resid = np.linalg.norm(pred - dst, axis=1)
    rows.append(dict(view=v, n=int(e - s),
                     umeyama_scale=float(sc), umeyama_rot_deg=rot_angle(R),
                     umeyama_trans_m=float(np.linalg.norm(t)),
                     resid_med_m=float(np.median(resid)),
                     resid_p90_m=float(np.percentile(resid, 90)),
                     raw_med_m=float(np.median(np.linalg.norm(dst - src, axis=1)))))

# 全局拟合（全部视角一起）
sc, R, t = umeyama(lift_all, means)
pred = sc * lift_all @ R.T + t
gresid = np.linalg.norm(pred - means, axis=1)

sl = np.exp(scale_log)
summary = dict(
    ply_vertices=int(ply.count),
    scale_px_equiv_median=[float(np.median(sl[:, i] / depth.reshape(-1)[
        np.concatenate([keep[v] for v in range(V)])] * fx)) for i in range(3)],
    scale_m_median=[float(np.median(sl[:, i])) for i in range(3)],
    scale_m_p1=[float(np.percentile(sl[:, i], 1)) for i in range(3)],
    scale_m_p99=[float(np.percentile(sl[:, i], 99)) for i in range(3)],
    opacity_med=float(1 / (1 + np.exp(-np.median(op_logit)))),
    global_fit=dict(scale=float(sc), rot_deg=rot_angle(R),
                    trans_m=float(np.linalg.norm(t)),
                    resid_med_m=float(np.median(gresid)),
                    resid_p90_m=float(np.percentile(gresid, 90))),
    per_view=rows,
)
Path(OUT).parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    json.dump(summary, f, ensure_ascii=False, indent=1)

print("scale(m) p1/med/p99 per axis:")
for i in range(3):
    print(f"  axis{i}: {summary['scale_m_p1'][i]:.3e} / {summary['scale_m_median'][i]:.3e} / {summary['scale_m_p99'][i]:.3e}")
print("scale 折算到像素（用各自视角深度）:", [f"{x:.3f}" for x in summary['scale_px_equiv_median']])
print(f"\n全局 Umeyama: s={sc:.4f} rot={rot_angle(R):.2f}° t={np.linalg.norm(t):.4f} m "
      f"resid_med={np.median(gresid)*1000:.1f} mm p90={np.percentile(gresid,90)*1000:.1f} mm")
print("\nview   raw_med_mm  s      rot_deg  t_mm   resid_med_mm  resid_p90_mm")
for r in rows:
    print(f"{r['view']:4d} {r['raw_med_m']*1000:10.2f} {r['umeyama_scale']:7.4f} "
          f"{r['umeyama_rot_deg']:7.2f} {r['umeyama_trans_m']*1000:7.1f} "
          f"{r['resid_med_m']*1000:12.2f} {r['resid_p90_m']*1000:12.2f}")
