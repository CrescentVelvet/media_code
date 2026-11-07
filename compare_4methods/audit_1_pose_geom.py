#!/usr/bin/env python3
"""AUDIT-1：位姿/内参/几何口径的端到端体检（只读）

四项，每一项都可能是「重影/模糊」的代码级根因：

  A. npz 的 extrinsics/intrinsics 与训练真正读的那份（colmap_scene/sparse/0）是否一致
     —— 高斯按 npz 摆、训练按 cameras.bin 摆；不一致 = 初值系统性错位。
  B. 内参形状：fx/fy 比值（正方形像素假设）、cx/cy 是否落在图心。
  C. 薄盘朝向 vs 视线（四元数组件序判据：wxyz 对 → |cos| 峰值高；错 → 接近 0.5 均匀）。
  D. DA3 自身的跨视角几何一致性：view i 的深度抬成点云投到 view j，与 j 的深度比。
     这一项与渲染/训练代码无关，是模型上限；它多大，重影就至少有多大。

用法：
  SRCDIR=~/output/da3_gs_results/佳佳抱枕_p1m \
  ~/miniconda3/envs/querysplat/bin/python compare_4methods/audit_1_pose_geom.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

SRC = Path(os.path.expanduser(os.environ["SRCDIR"]))
NPZ = SRC / "cameras.npz"
PLY = Path(os.path.expanduser(os.environ.get("PLY", str(SRC / "gs_ply" / "0000.ply"))))
SCENE = Path(os.path.expanduser(os.environ.get("SCENE", str(SRC / "colmap_scene"))))
OUT = Path(os.path.expanduser(os.environ.get("OUT", str(SRC / "audit_pose_geom.json"))))
NSUB = int(os.environ.get("NSUB", "4"))
DA3_SRC = Path(os.path.expanduser(os.environ.get(
    "DA3_SRC", "~/repos/Depth-Anything-3/src")))

rep = {}
print("=" * 78)
print("AUDIT-1  位姿/内参/几何口径体检")
print("=" * 78)

d = np.load(NPZ)
E = d["extrinsics"].astype(np.float64)
K = d["intrinsics"].astype(np.float64)
DEP = d["depth"].astype(np.float64)
CONF = d["conf"].astype(np.float64) if "conf" in d.files else None
V, H, W = DEP.shape
SF = float(d["scale_factor"]) if "scale_factor" in d.files else float("nan")
print(f"npz: V={V} H={H} W={W} scale_factor={SF:.4f} "
      f"gs_scale_applied={int(d['gs_scale_applied']) if 'gs_scale_applied' in d.files else '?'}")
Rc = E[:, :, :3]
C = -np.einsum("vji,vj->vi", Rc, E[:, :, 3])

# ── B ─────────────────────────────────────────────────────────────────────
fx, fy, cx, cy = K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2]
print("\n[B] 内参形状")
print(f"  fx mean {fx.mean():.2f} std {fx.std():.3f} | fy mean {fy.mean():.2f} std {fy.std():.3f}")
print(f"  fx/fy: mean {np.mean(fx/fy):.5f}  min {np.min(fx/fy):.5f}  max {np.max(fx/fy):.5f}")
print(f"  cx {cx.min():.1f}..{cx.max():.1f}（图心 {W/2:.1f}）  cy {cy.min():.1f}..{cy.max():.1f}（图心 {H/2:.1f}）")
rep["intrinsics"] = dict(fx_mean=float(fx.mean()), fy_mean=float(fy.mean()),
                         fx_over_fy=float(np.mean(fx / fy)),
                         cx_mean=float(cx.mean()), cy_mean=float(cy.mean()))

# ── A ─────────────────────────────────────────────────────────────────────
print("\n[A] npz ↔ colmap_scene（训练实际读的那份）")
try:
    sys.path.insert(0, str(DA3_SRC))
    from depth_anything_3.utils import read_write_model as rwm
    cams = rwm.read_cameras_binary(str(SCENE / "sparse" / "0" / "cameras.bin"))
    imgs = rwm.read_images_binary(str(SCENE / "sparse" / "0" / "images.bin"))
    names = sorted(imgs.keys(), key=lambda k: imgs[k].name)
    print(f"  colmap: {len(names)} 图 / {len(cams)} 相机")
    dR, dt, dK = [], [], []
    for i, k in enumerate(names):
        im = imgs[k]
        Rm = rwm.qvec2rotmat(im.qvec)          # world->cam
        dR.append(np.abs(Rm - E[i, :, :3]).max())
        dt.append(np.abs(im.tvec - E[i, :, 3]).max())
        cam = cams[im.camera_id]
        p = np.asarray(cam.params, dtype=np.float64)
        sx, sy = cam.width / W, cam.height / H
        Ksc = np.array([K[i, 0, 0] * sx, K[i, 1, 1] * sy, K[i, 0, 2] * sx, K[i, 1, 2] * sy])
        dK.append(np.abs(p[:4] - Ksc).max())
    dR, dt, dK = map(np.asarray, (dR, dt, dK))
    print(f"  |ΔR|max {dR.max():.2e}  |Δt|max {dt.max():.2e} m  |ΔK|max {dK.max():.4f} px")
    ok = bool(dR.max() < 1e-5 and dt.max() < 1e-5 and dK.max() < 1.0)
    print(f"  ⇒ {'✅ 位姿链路一致（训练与高斯同一世界系/同尺度）' if ok else '❌ 位姿链路不一致'}")
    rep["colmap_vs_npz"] = dict(n=len(names), dR_max=float(dR.max()), dt_max=float(dt.max()),
                                dK_max_px=float(dK.max()), ok=ok,
                                cam_wh=[int(cam.width), int(cam.height)],
                                cam_model=getattr(cam, "model", "?"))
except Exception as e:  # noqa: BLE001
    print(f"  ❌ {type(e).__name__}: {e}")
    rep["colmap_vs_npz"] = dict(error=str(e))

# ── C ─────────────────────────────────────────────────────────────────────
print("\n[C] 薄盘朝向 vs 视线（四元数组件序判据）")


def q_to_R(q):
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


try:
    from plyfile import PlyData
    pv = PlyData.read(str(PLY))["vertex"]
    th, tw = int(8 / 256 * H), int(8 / 256 * W)
    m = np.zeros((V, H, W), bool)
    m[:, th:H - th, tw:W - tw] = True
    m &= DEP <= np.quantile(DEP.reshape(V, -1), 0.90, axis=1).reshape(V, 1, 1)
    cnt = m.reshape(V, -1).sum(1)
    print(f"  PLY {pv.count:,} 点；复刻屏蔽 {cnt.sum():,} 点  diff={pv.count - cnt.sum()}")
    q = np.stack([np.asarray(pv[f"rot_{i}"]) for i in range(4)], 1).astype(np.float64)
    sc = np.exp(np.stack([np.asarray(pv[f"scale_{i}"]) for i in range(3)], 1).astype(np.float64))
    xyz = np.stack([np.asarray(pv[c]) for c in "xyz"], 1).astype(np.float64)
    imin = np.argmin(sc, axis=1)
    vid = np.repeat(np.arange(V), cnt)
    ray = xyz - C[vid]
    ray /= np.maximum(np.linalg.norm(ray, axis=1, keepdims=True), 1e-12)
    pick = lambda R, idx: np.take_along_axis(R, idx[:, None, None].repeat(3, 2), axis=2)[:, :, 0]
    c1 = np.abs((pick(q_to_R(q), imin) * ray).sum(1))
    alt = np.stack([q[:, 3], q[:, 0], q[:, 1], q[:, 2]], 1)
    c2 = np.abs((pick(q_to_R(alt), imin) * ray).sum(1))
    print(f"  |cos(最短轴, 视线)|  wxyz序 mean {c1.mean():.3f} 中位 {np.median(c1):.3f}"
          f"（均匀=0.5，完全对齐=1）")
    print(f"  |cos(最短轴, 视线)|  另一序 mean {c2.mean():.3f} 中位 {np.median(c2):.3f}")
    # σ 折算到像素
    z = DEP.reshape(-1)[np.concatenate([np.flatnonzero(m[v].ravel()) for v in range(V)])]
    sig_px = np.median(sc / np.maximum(z, 1e-9)[:, None] * fx.mean(), axis=0)
    print(f"  σ 三轴中位（像素当量）: {np.round(sig_px, 3)}")
    print(f"  opacity 中位/sigmoid 后: {1/(1+np.exp(-np.asarray(pv['opacity'], dtype=np.float64)))[:5].mean():.3f}"
          f" (前5点) 中位 {np.median(1/(1+np.exp(-np.asarray(pv['opacity'], dtype=np.float64)))):.4f}")
    rep["orientation"] = dict(cos_mean_wxyz=float(c1.mean()), cos_med_wxyz=float(np.median(c1)),
                              cos_mean_alt=float(c2.mean()), sigma_px_median=sig_px.tolist())
except Exception as e:  # noqa: BLE001
    print(f"  ❌ {type(e).__name__}: {e}")
    rep["orientation"] = dict(error=str(e))

# ── D ─────────────────────────────────────────────────────────────────────
print("\n[D] 跨视角几何一致性：view i 的深度点云投到 view j，与 j 的深度比")
us, vs = np.meshgrid(np.arange(W), np.arange(H))
us = us.astype(np.float64)
vs = vs.astype(np.float64)
conf_thr = np.percentile(CONF, 60) if CONF is not None else -np.inf
rows = []
for i in range(V):
    sel = (slice(None, None, NSUB), slice(None, None, NSUB))
    z = DEP[i][sel]
    ok = (CONF[i][sel] > conf_thr) & (z > 1e-6) if CONF is not None else z > 1e-6
    x = (us[sel] - cx[i]) / fx[i] * z
    y = (vs[sel] - cy[i]) / fy[i] * z
    Pc = np.stack([x, y, z], -1).reshape(-1, 3)[ok.ravel()]
    if len(Pc) < 500:
        continue
    Pw = (Pc - E[i, :, 3]) @ Rc[i]          # Xw = R^T (Xc - t)
    for j in range(V):
        if j == i:
            continue
        Xc = Pw @ Rc[j].T + E[j, :, 3]
        zz = Xc[:, 2]
        fr = zz > 1e-6
        uu = fx[j] * Xc[:, 0] / np.maximum(zz, 1e-9) + cx[j]
        vv = fy[j] * Xc[:, 1] / np.maximum(zz, 1e-9) + cy[j]
        ui = np.round(uu).astype(int)
        vi = np.round(vv).astype(int)
        ins = fr & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        if ins.sum() < 500:
            continue
        zj = DEP[j][vi[ins], ui[ins]]
        zi = zz[ins]
        m2 = zj > 1e-6
        if CONF is not None:
            m2 &= CONF[j][vi[ins], ui[ins]] > conf_thr
        if m2.sum() < 500:
            continue
        rel = np.abs(zi[m2] - zj[m2]) / np.maximum(zj[m2], 1e-9)
        rows.append(dict(i=i, j=j, n=int(m2.sum()), rel_med=float(np.median(rel)),
                         within1pct=float((rel < 0.01).mean())))
if rows:
    relmed = np.array([r["rel_med"] for r in rows])
    w1 = np.array([r["within1pct"] for r in rows])
    print(f"  {len(rows)} 对视角：相对深度误差 中位 {np.median(relmed)*100:.2f}% "
          f"（p90 对 {np.percentile(relmed,90)*100:.2f}%）")
    print(f"  <1% 的像素占比（逐对中位）：{np.median(w1)*100:.1f}%")
    # 换算成像素位移：误差 δz 在深度 z 处的横向位移 ≈ 0（纯深度方向），
    # 但表面倾斜时 δz 会变成切向滑动；用 fx 与中位深度给一个量级
    zmid = float(np.median(DEP[DEP > 0]))
    print(f"  量级参照：中位深度 {zmid:.2f} m，1% 深度误差 ≈ {1.0*zmid*fx.mean()/zmid:.1f} px 的切向滑动上限")
    rep["cross_view_geometry"] = dict(n_pairs=len(rows),
                                      rel_med_global=float(np.median(relmed)),
                                      rel_p90_of_pairs=float(np.percentile(relmed, 90)),
                                      within1pct_med=float(np.median(w1)),
                                      depth_median=zmid)

with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    json.dump(rep, f, ensure_ascii=False, indent=1, default=float)
print(f"\n💾 {OUT}")
