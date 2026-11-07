#!/usr/bin/env python3
"""AUDIT-7：训练前后参数漂移 —— 「训练真的在学什么」的体检

B1 闸门（99c）只验证 xyz 动过。但重影/模糊的关键量是另外三个：
  scaling   训练能不能把 σ 从"1.6 px 的圆盘"缩到"贴合像素"的量级（决定锐度）
  opacity   训练能不能把 0.036 的薄 alpha 提到接近 1（决定对比度/透明度叠影）
  rotation  能不能把薄盘摆正

对照：同一个 init ckpt（98b 产出，含剪枝后的点集与顺序）vs 训练后的 PLY。
两点集相同、顺序相同（densify=0 不增删点），所以可以逐点比，不必配准。

用法：
  INIT=.../ffinit_ckpt/point_cloud/iteration_0/point_cloud.ply \
  TRAINED=.../point_cloud/iteration_2000/point_cloud.ply \
  NPZ=.../cameras.npz OUT=... \
  ~/miniconda3/envs/querysplat/bin/python compare_4methods/audit_7_param_drift.py
"""
import json
import os
import re
from pathlib import Path

import numpy as np

INIT = Path(os.path.expanduser(os.environ["INIT"]))
TRAINED = Path(os.path.expanduser(os.environ["TRAINED"]))
NPZ = Path(os.path.expanduser(os.environ["NPZ"]))
OUT = Path(os.path.expanduser(os.environ["OUT"]))
RES = int(os.environ.get("RES", "1"))
ORIG_W = int(os.environ.get("ORIG_W", "1439"))


def log(s):
    print(s, flush=True)


def read_ply(path):
    with open(path, "rb") as f:
        raw = f.read(65536)
        end = raw.index(b"end_header") + len(b"end_header") + 1
        head = raw[:end].decode("ascii", "replace")
        n = int(re.search(r"element vertex (\d+)", head).group(1))
        props = [l.split()[2] for l in head.splitlines() if l.startswith("property")]
        f.seek(end)
        arr = np.fromfile(f, dtype=np.float32).reshape(n, len(props))
    return {k: arr[:, i] for i, k in enumerate(props)}, n


def stats(d, tag):
    sc = np.exp(np.stack([d[f"scale_{i}"] for i in range(3)], 1).astype(np.float64))
    op = 1 / (1 + np.exp(-d["opacity"].astype(np.float64)))
    log(f"  [{tag}] σ(m) 中位 {np.median(sc, 0)}  | σ 最大轴 p90 {np.percentile(sc.max(1), 90):.5f}")
    log(f"  [{tag}] opacity 中位 {np.median(op):.4f}  p10 {np.percentile(op,10):.4f}  "
        f"p90 {np.percentile(op,90):.4f}  >0.5 占比 {(op>0.5).mean()*100:.1f}%")
    return sc, op


def prune_98b(d, npz_path, min_opacity=0.005, max_dist=5.0):
    """复刻 98b 的剪枝：opacity>min 且 距最近相机<max_dist（与 ckpt 点集/顺序一致）。"""
    e = np.load(npz_path)["extrinsics"].astype(np.float64)
    C = -np.einsum("nij,nj->ni", e[:, :, :3].transpose(0, 2, 1), e[:, :, 3])
    op = 1 / (1 + np.exp(-d["opacity"].astype(np.float64)))
    xyz = np.stack([d[c] for c in "xyz"], 1).astype(np.float64)
    dd = np.sqrt(((xyz[:, None, :] - C[None]) ** 2).sum(-1)).min(1)
    keep = (op > min_opacity) & (dd < max_dist)
    return {k: v[keep] for k, v in d.items()}, int(keep.sum()), int(len(op))


d0 = np.load(NPZ)
fx0 = float(d0["intrinsics"][0, 0, 0])
dep = d0["depth"]
z_med = float(np.median(dep[dep > 0]))
npx = ORIG_W / (dep.shape[2] * RES) * 0  # unused

a, na = read_ply(INIT) if INIT.exists() else (None, 0)
if a is None:
    src = Path(os.path.expanduser(os.environ["INIT_SRC_PLY"]))
    a_raw, na_raw = read_ply(src)
    a, na, n_before = prune_98b(a_raw, NPZ)
    log(f"init 无独立 PLY，从 {src.name}（{na_raw:,}）复刻 98b 剪枝 → {na:,} 点")
b, nb = read_ply(TRAINED)
log(f"init {na:,} 点 / trained {nb:,} 点")
if na != nb:
    raise SystemExit(f"❌ 点集不同（{na:,} vs {nb:,}），densify 应关闭；无法逐点比")

s0, o0 = stats(a, "init")
s1, o1 = stats(b, "trained")

xyz0 = np.stack([a[c] for c in "xyz"], 1).astype(np.float64)
xyz1 = np.stack([b[c] for c in "xyz"], 1).astype(np.float64)
disp = np.linalg.norm(xyz1 - xyz0, axis=1)

q0 = np.stack([a[f"rot_{i}"] for i in range(4)], 1).astype(np.float64)
q1 = np.stack([b[f"rot_{i}"] for i in range(4)], 1).astype(np.float64)
q0 /= np.linalg.norm(q0, axis=1, keepdims=True)
q1 /= np.linalg.norm(q1, axis=1, keepdims=True)
ang = 2 * np.degrees(np.arccos(np.clip(np.abs((q0 * q1).sum(1)), -1, 1)))

dc0 = np.stack([a[f"f_dc_{i}"] for i in range(3)], 1).astype(np.float64)
dc1 = np.stack([b[f"f_dc_{i}"] for i in range(3)], 1).astype(np.float64)

log("\n逐点变化")
log(f"  xyz 位移: 中位 {np.median(disp)*1000:.2f} mm  p90 {np.percentile(disp,90)*1000:.2f} mm  "
    f"max {disp.max()*1000:.1f} mm")
log(f"  rotation 变化角: 中位 {np.median(ang):.2f}°  p90 {np.percentile(ang,90):.2f}°")
log(f"  f_dc 变化: 中位 {np.median(np.abs(dc1-dc0)):.5f}  max {np.abs(dc1-dc0).max():.4f}")

r = s1 / np.maximum(s0, 1e-12)
log(f"  σ 变化倍率: 中位 {np.median(r, 0)}（<1 = 变小变锐）")
log(f"  opacity 变化: 中位 {np.median(o1/o0):.3f}×  "
    f"中位绝对值 {np.median(o0):.4f} → {np.median(o1):.4f}")

# σ 折算到"渲染分辨率下每像素当量"
# ⚠️ 渲染焦距 = 处理分辨率焦距 × (原图宽/处理宽) / RES，不能写成 fx0*RES（差 3.8 倍）
proc_w = dep.shape[2]
fx_render = fx0 * (ORIG_W / proc_w) / RES
sig0 = np.median(s0 / z_med * fx_render, axis=0)
sig1 = np.median(s1 / z_med * fx_render, axis=0)
log(f"\n  σ 像素当量（深度取全图中位 {z_med:.2f} m；渲染 {ORIG_W//RES}px 宽, "
    f"fx={fx_render:.1f}）:")
log(f"    init    {np.round(sig0, 3)}")
log(f"    trained {np.round(sig1, 3)}")
# 另给 378 口径（交付/eval 尺寸）下的同一组数，便于与 fig16/20 对齐
fx378 = fx0
log(f"    同一组在 378 口径下（fx={fx378:.1f}）: init {np.round(np.median(s0/z_med*fx378,0),3)} "
    f"→ trained {np.round(np.median(s1/z_med*fx378,0),3)}")

out = dict(init=str(INIT), trained=str(TRAINED), n=int(na), res=RES,
           sigma_m_init=np.median(s0, 0).tolist(), sigma_m_trained=np.median(s1, 0).tolist(),
           sigma_ratio_median=np.median(r, 0).tolist(),
           sigma_px_render_init=sig0.tolist(), sigma_px_render_trained=sig1.tolist(),
           fx_render=float(fx_render), fx_378=float(fx378),
           sigma_px_378_init=np.median(s0 / z_med * fx378, 0).tolist(),
           sigma_px_378_trained=np.median(s1 / z_med * fx378, 0).tolist(),
           opacity_median_init=float(np.median(o0)), opacity_median_trained=float(np.median(o1)),
           opacity_gt_half_init=float((o0 > 0.5).mean()),
           opacity_gt_half_trained=float((o1 > 0.5).mean()),
           xyz_disp_med_m=float(np.median(disp)), xyz_disp_p90_m=float(np.percentile(disp, 90)),
           rot_delta_med_deg=float(np.median(ang)), f_dc_abs_med=float(np.median(np.abs(dc1 - dc0))))
Path(OUT).parent.mkdir(parents=True, exist_ok=True)
with open(Path(OUT) / "audit_param_drift.json", "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
log(f"\n💾 {Path(OUT) / 'audit_param_drift.json'}")
