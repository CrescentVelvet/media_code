#!/usr/bin/env python3
"""99 · 诊断 DA3 导出高斯的**逐视角**属性（只读，不训练）

回答 fig10 里的三个疑问：
  Q1 「DA3 直出正面偏白」── 是不是 alpha 欠饱和？看逐视角 opacity 中位数。
  Q2 「侧面完全看不到人物」── 是位姿错，还是几何/尺度错？看逐视角 reprojection 误差
      （means 投影回自己的相机应落在原像素上）与像素足迹。
  Q3 「前馈+5k 重影比 COLMAP+5k 重」── 拓扑来自 init，看每个视角贡献多少"壳层"。

做法：
  1. 完全复刻 DA3 `save_gaussian_ply` 的屏蔽规则（border 8/256 + depth ≤ q90）还原每个
     视角保留了多少点 —— 顺序是 view-major，所以可以把 PLY 顶点按视角切块。
  2. 逐视角统计 opacity / scale / 像素足迹 / 自投影误差。

用法：
  NPZ=.../cameras.npz PLY=.../gs_ply/0000.ply OUT=.../diag_perview.json \
    python compare_4methods/99_diag_gs_ply_perview.py
"""
import json
import os
from pathlib import Path

import numpy as np
from plyfile import PlyData

NPZ = os.environ["NPZ"]
PLY = os.environ["PLY"]
OUT = os.environ["OUT"]

d = np.load(NPZ)
ext = d["extrinsics"].astype(np.float64)   # (V,3,4) world->cam
K = d["intrinsics"].astype(np.float64)     # (V,3,3) 处理分辨率口径
depth = d["depth"].astype(np.float64)      # (V,H,W)
V, H, W = depth.shape
print(f"V={V} H={H} W={W}  fx={K[0,0,0]:.2f} cx={K[0,0,2]:.2f} cy={K[0,1,2]:.2f}")

# ---- 1. 复刻导出屏蔽 ----
trim_h = int(8 / 256 * H)
trim_w = int(8 / 256 * W)
mask = np.zeros((V, H, W), dtype=bool)
mask[:, trim_h:H - trim_h, trim_w:W - trim_w] = True
q90 = np.quantile(depth.reshape(V, -1), 0.90, axis=1).reshape(V, 1, 1)
mask &= (depth <= q90)

counts = mask.reshape(V, -1).sum(1)
print("per-view kept:", counts.min(), counts.max(), counts.sum())

# ---- 2. 读 PLY ----
ply = PlyData.read(PLY)["vertex"]
N = ply.count
print(f"ply vertices={N}  sum(mask)={counts.sum()}  diff={N - counts.sum()}")
if N != counts.sum():
    raise SystemExit("顶点数与复刻的屏蔽不一致，顺序假设不成立，需换定位方式")

xyz = np.stack([ply["x"], ply["y"], ply["z"]], 1).astype(np.float64)
op_logit = np.asarray(ply["opacity"], dtype=np.float64)
scale_log = np.stack([ply[f"scale_{i}"] for i in range(3)], 1).astype(np.float64)
f_dc = np.stack([ply[f"f_dc_{i}"] for i in range(3)], 1).astype(np.float64)

op = 1.0 / (1.0 + np.exp(-op_logit))            # sigma
scale_lin = np.exp(scale_log)                    # 米
scale_geo = np.exp(scale_log.mean(1))            # 各向同性代表值

view_id = np.repeat(np.arange(V), counts)
bounds = np.concatenate([[0], np.cumsum(counts)])

# 相机中心 C = -R^T t
R = ext[:, :, :3]
t = ext[:, :, 3]
C = -np.einsum("vji,vj->vi", R, t)

rows = []
for v in range(V):
    s, e = bounds[v], bounds[v + 1]
    P = xyz[s:e]
    z = P @ R[v].T + t[v]                      # 到自己相机
    zc = z[:, 2]
    ok = zc > 1e-6
    u = K[v][0, 0] * z[:, 0] / np.maximum(zc, 1e-9) + K[v][0, 2]
    vv = K[v][1, 1] * z[:, 1] / np.maximum(zc, 1e-9) + K[v][1, 2]
    # 该点在自己图像里的"应落位置"：由屏蔽反推 pixel 索引
    idx = np.flatnonzero(mask[v].reshape(-1))
    px = (idx % W).astype(np.float64)
    py = (idx // W).astype(np.float64)
    err = np.hypot(u[ok] - px[ok], vv[ok] - py[ok]) if ok.any() else np.zeros(1)

    rows.append(dict(
        view=v,
        n=int(e - s),
        opacity_med=float(np.median(op[s:e])),
        opacity_p90=float(np.percentile(op[s:e], 90)),
        scale_mm_med=float(np.median(scale_geo[s:e]) * 1000),
        footprint_px_med=float(np.median(scale_geo[s:e] / np.maximum(zc, 1e-6) * K[v][0, 0])),
        depth_med=float(np.median(zc)),
        reproj_px_med=float(np.median(err)),
        reproj_px_p90=float(np.percentile(err, 90)),
        dc_mean=float(f_dc[s:e].mean()),
        dist_to_ref=float(np.linalg.norm(C[v] - C[0])),
    ))

# ---- 3. 汇总 ----
dist = np.array([r["dist_to_ref"] for r in rows])
opm = np.array([r["opacity_med"] for r in rows])
fp = np.array([r["footprint_px_med"] for r in rows])
rp = np.array([r["reproj_px_med"] for r in rows])
sc = np.array([r["scale_mm_med"] for r in rows])

def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])

summary = dict(
    n_views=V,
    ply_vertices=int(N),
    camera_radius_m=dict(mean=float(np.linalg.norm(C - C.mean(0), axis=1).mean()),
                         min=float(np.linalg.norm(C - C.mean(0), axis=1).min()),
                         max=float(np.linalg.norm(C - C.mean(0), axis=1).max())),
    opacity_med=float(np.median(op)),
    opacity_p90=float(np.percentile(op, 90)),
    scale_mm_med=float(np.median(scale_geo) * 1000),
    footprint_px_med=float(np.median(scale_geo / np.maximum(
        np.abs(xyz @ R[0].T + t[0])[:, 2], 1e-6) * K[0, 0, 0])),
    corr_dist_opacity=corr(dist, opm),
    corr_dist_footprint=corr(dist, fp),
    corr_dist_scale=corr(dist, sc),
    corr_dist_reproj=corr(dist, rp),
    per_view=rows,
)
Path(OUT).parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    json.dump(summary, f, ensure_ascii=False, indent=1)

print("\nview  dist_m  op_med  op_p90  scale_mm  fp_px  depth_m  reproj_px")
for r in rows:
    print(f"{r['view']:4d} {r['dist_to_ref']:7.3f} {r['opacity_med']:7.4f} "
          f"{r['opacity_p90']:7.4f} {r['scale_mm_med']:8.2f} {r['footprint_px_med']:6.2f} "
          f"{r['depth_med']:7.3f} {r['reproj_px_med']:9.2f}")
print("\ncorr(dist, opacity_med) =", f"{summary['corr_dist_opacity']:+.3f}")
print("corr(dist, footprint_px)=", f"{summary['corr_dist_footprint']:+.3f}")
print("corr(dist, scale_mm)    =", f"{summary['corr_dist_scale']:+.3f}")
print("corr(dist, reproj_px)   =", f"{summary['corr_dist_reproj']:+.3f}")
print("global opacity med =", f"{summary['opacity_med']:.4f}",
      " scale med =", f"{summary['scale_mm_med']:.2f} mm")
