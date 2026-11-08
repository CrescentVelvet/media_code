#!/usr/bin/env python3
"""AUDIT-15b · I-03 主判据的可判性诊断（只读，不渲染）

背景：IMPROVEMENTS.md §3.5 把 I-03 主判据写成「掠射区（slant>60°）跨视角深度
一致性中位从 0.95% 降到 <0.85%」。但 C2 同时写着 0.95% 来自 DA3 的位姿/深度、
2DGS 改不了。这两句放在一起会自相矛盾：**若 0.95% 是输入深度自己的误差，
那任何渲染器都不可能把它压到 0.85%**。

所以先把这个数摸清楚再定判据（不是改判据，是搞清楚判据测量的是什么）：

  D1 DA3 自己输出的深度图，跨视角一致性是多少？按 slant 分档。
     → 这是**渲染深度的下界**：渲染器只能逼近输入深度，不能超过它。
  D2 视角对几何：v→v+1 的相机基线距离与旋转角，确认"相邻"够近、基线可比。
  D3 一致性随 slant 怎么变？若掠射区本身就已经 20%，那 0.85% 这个阈值
     从来就不该安在掠射区上，要改成"相对 3DGS 渲染的相对改善"。

口径（跑前定死）：
  · slant = ∠(GT 深度图法向, 视线)，用世界坐标 u/v 前向差分叉积（同 audit_12）
  · 深度用相机系 z（沿视线），相对误差 = |z_proj − z_gt| / z_proj
  · 视角对固定为 v → (v+1) mod V（32 视角环，间隔 11.25°）
  · 只统计：投射点落在邻视角画面内、且邻视角 GT 深度 > 0
  · 前置闸门（本次实测发现，不是预设）：采集序列**首尾不相邻**。31→0 的基线
    210.75 cm（其余 0.64~30 cm 量级）、有效比 41.8%、中位 3.19%。环形口径会把
    这一对混进中位数。故剔除「基线 > 3× 中位基线」或「有效比 < 50%」的视角对。
"""
import json
import os

import numpy as np

NPZ = os.environ["NPZ"]
OUT = os.environ.get("OUT", "/tmp/audit_15b.json")


def log(*a):
    print(*a, flush=True)


d = np.load(NPZ)
W2C = d["extrinsics"].astype(np.float64)
if W2C.shape[-2] == 4:
    W2C = W2C[:, :3, :]
K = d["intrinsics"].astype(np.float64)
DEP = d["depth"].astype(np.float64)
V, H, W = DEP.shape
fx = K[:, 0, 0]
fy = K[:, 1, 1]
cx = K[:, 0, 2]
cy = K[:, 1, 2]
log(f"📷 {V} 视角 {W}x{H}  fx={fx.mean():.2f}  scale_factor="
    f"{float(d['scale_factor']):.4f}")

uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)


def unproject(v, z):
    """GT 深度图 → 世界点（只取 z>0）"""
    R, t = W2C[v, :, :3], W2C[v, :, 3]
    xc = np.stack([(uu - cx[v]) / fx[v] * z, (vv - cy[v]) / fy[v] * z, z], -1)
    return (xc - t) @ R


def project(v, Pw):
    R, t = W2C[v, :, :3], W2C[v, :, 3]
    p = Pw @ R.T + t
    z = p[..., 2]
    zs = np.where(np.abs(z) < 1e-9, np.nan, z)
    return fx[v] * p[..., 0] / zs + cx[v], fy[v] * p[..., 1] / zs + cy[v], z


def normal_map(Pw, C):
    du = np.zeros_like(Pw)
    dv = np.zeros_like(Pw)
    du[:, :-1] = Pw[:, 1:] - Pw[:, :-1]
    dv[:-1, :] = Pw[1:, :] - Pw[:-1, :]
    n = np.cross(du, dv)
    nn = np.linalg.norm(n, axis=-1, keepdims=True)
    n = np.divide(n, np.maximum(nn, 1e-12))
    view = Pw - C[None, None, :]
    view /= np.maximum(np.linalg.norm(view, axis=-1, keepdims=True), 1e-12)
    return n * np.where(np.sum(n * view, -1, keepdims=True) > 0, -1.0, 1.0), nn[..., 0] > 1e-12


# ── D2 视角对几何 ───────────────────────────────────────────────────────────
cent = np.stack([-W2C[v, :, :3].T @ W2C[v, :, 3] for v in range(V)])
step = np.linalg.norm(np.roll(cent, -1, 0) - cent, axis=1)
ang = np.degrees(np.arccos(np.clip(
    (np.trace(np.roll(W2C[:, :, :3], -1, 0) @ np.transpose(W2C[:, :, :3], (0, 2, 1)),
              axis1=1, axis2=2) - 1) / 2, -1, 1)))
log(f"\n[D2] 相邻视角对 v→v+1（mod {V}）：基线 {step.mean()*100:.2f}±"
    f"{step.std()*100:.2f} cm（{step.min()*100:.2f}~{step.max()*100:.2f}），"
    f"旋转 {ang.mean():.2f}±{ang.std():.2f}°")

# ── D1/D3 GT 深度跨视角一致性 ───────────────────────────────────────────────
TIERS = ((0, 30), (30, 60), (60, 90), (0, 90))
log("\n[D1] DA3 自己输出的深度图，跨视角一致性（v 的 GT 深度投到 v+1）")
log(f"  {'视角对':>9} {'有效比':>8} {'基线cm':>8} {'全 slant 中位':>13} {'p90':>8} "
    f"{'0-30°':>8} {'30-60°':>8} {'60-90°':>8}")

pairs = []          # 每个视角对的 (rel, slant, ok_any, validfrac, step)
for v in range(V):
    u2 = (v + 1) % V
    z = DEP[v]
    valid = z > 0
    if valid.sum() < 100:
        continue
    Pw = unproject(v, z)
    C = cent[v]
    nL, okN = normal_map(Pw, C)
    view = Pw - C[None, None, :]
    view /= np.maximum(np.linalg.norm(view, axis=-1, keepdims=True), 1e-12)
    slant = np.degrees(np.arccos(np.clip(np.abs(np.sum(nL * view, -1)), -1, 1)))

    uu2, vv2, z2 = project(u2, Pw)
    ok = valid & okN & np.isfinite(z2) & (z2 > 0)
    ui = np.round(uu2).astype(np.int64)
    vi = np.round(vv2).astype(np.int64)
    ok &= (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    zb = np.zeros_like(z2)
    zb[ok] = DEP[u2][vi[ok], ui[ok]]
    ok &= zb > 0
    rel = np.abs(z2 - zb) / np.maximum(z2, 1e-9) * 100.0
    pairs.append(dict(v=v, u=u2, rel=rel, slant=slant, ok=ok,
                      vf=float(ok.sum() / max(valid.sum(), 1)), step=float(step[v])))

base_med = np.median([p["step"] for p in pairs])
keep = [p for p in pairs if p["step"] <= 3 * base_med and p["vf"] >= 0.50]
drop = [p for p in pairs if p not in keep]
drop_names = [f"{p['v']}→{p['u']}" for p in drop]
log(f"  ⛳ 闸门（基线 ≤3×{base_med*100:.2f}cm 中位 且 有效比 ≥50%）："
    f"保留 {len(keep)} 对，剔除 {len(drop)} 对"
    + (f" → {drop_names}" if drop else ""))

acc = {t: [] for t in TIERS}
acc_all = []
for p in keep:
    m0 = p["ok"]
    rel, slant = p["rel"], p["slant"]
    cells = []
    for lo, hi in TIERS:
        m = m0 & (slant >= lo) & (slant < hi)
        cells.append(np.median(rel[m]) if m.sum() > 500 else np.nan)
        if m.sum() > 500:
            acc[(lo, hi)].append(rel[m])
    acc_all.append(rel[m0])
    log(f"  {p['v']:>3}→{p['u']:<5} {p['vf']:>7.1%} {p['step']*100:>8.2f} "
        f"{np.median(rel[m0]):>12.2f}% {np.percentile(rel[m0],90):>7.2f}% "
        f"{cells[0]:>7.2f}% {cells[1]:>7.2f}% {cells[2]:>7.2f}%")

log("\n[D3] 分档汇总（保留视角对的中位，%）")
summary = {}
for (lo, hi) in TIERS:
    if not acc[(lo, hi)]:
        continue
    a = np.concatenate(acc[(lo, hi)])
    summary[f"{lo}-{hi}"] = dict(
        med_pct=float(np.median(a)), p90_pct=float(np.percentile(a, 90)),
        n=int(a.size), views=int(len(acc[(lo, hi)])))
    log(f"  slant {lo:>2}-{hi:<2}°  中位 {np.median(a):6.2f}%  "
        f"p90 {np.percentile(a,90):6.2f}%  ({len(acc[(lo,hi)])} 视角对, "
        f"{a.size:,} 像素)")

allc = np.concatenate(acc_all)
vf_mean = float(np.mean([p["vf"] for p in keep]))
log(f"  全 slant     中位 {np.median(allc):6.2f}%  p90 "
    f"{np.percentile(allc,90):6.2f}%  （有效比均值 {vf_mean:.1%}）")

summary["all"] = dict(med_pct=float(np.median(allc)),
                      p90_pct=float(np.percentile(allc, 90)),
                      n=int(allc.size), valid_frac=vf_mean)
summary["pair_geom"] = dict(baseline_cm=float(step.mean() * 100),
                            rot_deg=float(ang.mean()),
                            kept=int(len(keep)), dropped=drop_names)

log("\n【判读】")
log(f"  · 掠射区 GT 深度自身的一致性 = {summary['60-90']['med_pct']:.2f}%，"
    f"全图 = {summary['all']['med_pct']:.2f}%")
log("  · 这两个数是**渲染深度的下界**：任何光栅器只能逼近输入深度，不能超过它。")
if summary["60-90"]["med_pct"] < 0.85:
    log("  · 掠射区 GT 下界已低于 0.85% → 原判据（掠射区 <0.85%）可判，直接比即可。")
else:
    log(f"  · 掠射区 GT 下界 {summary['60-90']['med_pct']:.2f}% 已高于 0.85%，"
        "\n    原阈值安错了位置。I-03 主判据须改为「2DGS 相对 3DGS 的相对改善」，"
        "\n    阈值以本次实测的 GT 下界为锚（见台账订正）。")

with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    json.dump(dict(npz=NPZ, wh=[W, H], tiers=summary), f,
              indent=1, ensure_ascii=False, default=float)
log(f"\n💾 {OUT}")
