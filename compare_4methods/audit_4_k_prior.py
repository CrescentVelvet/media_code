#!/usr/bin/env python3
"""AUDIT-4：内参先验扫掠 —— 找「系统性的相机口径错」，判据是跨视角一致性

## 为什么怀疑内参

AUDIT-1 测到 DA3 预测的 fx/fy = 1.0212（各视角 1.011~1.046），而真实相机是方像素。
fx 与 fy 不能靠深度图吸收（深度只吸收尺度，不吸收"横向拉伸"），所以各向异性会变成
**每个视角各自的一次 y 向拉伸**，视角之间对不上 → 叠出重影。

判据不看 PSNR（PSNR 会奖励"糊"），看**跨视角一致性**：
把 view i 的深度抬成点云、按相机 K_j 投到 view j，与 j 的深度比；
再用同一套几何把 i 的颜色 warp 到 j，与 j 的图像比。两者都应与 K 的真值同时最优。

扫三个量：
  A  fy/fx 比值（1.00 = 方像素先验；1.021 = DA3 原始预测）
  B  全局焦距系数 β（同时乘 fx,fy；探"焦距整体偏大/偏小"）
  C  cx,cy 偏移（AUDIT-1 测到恒为图心，这里只做确认性扫掠）

用法：
  SRCDIR=~/output/da3_gs_results/佳佳抱枕_p1m \
  ~/miniconda3/envs/querysplat/bin/python compare_4methods/audit_4_k_prior.py
"""
import json
import os
from pathlib import Path

import numpy as np

SRC = Path(os.path.expanduser(os.environ["SRCDIR"]))
NPZ = SRC / "cameras.npz"
OUT = Path(os.path.expanduser(os.environ.get("OUT", str(SRC / "audit_k_prior.json"))))
PAIRS = [1, 2, 4, 8, 16]
NSUB = int(os.environ.get("NSUB", "3"))
ALPHAS = [float(x) for x in os.environ.get("ALPHAS", "0.97,0.98,0.99,1.0,1.01,1.02,1.03").split(",")]
BETAS = [float(x) for x in os.environ.get("BETAS", "0.97,0.98,0.99,1.0,1.01,1.02").split(",")]
REF_DIR = Path(os.path.expanduser(os.environ.get(
    "REF", "~/output/da3_results/_captures/佳佳抱枕/images")))


def log(s):
    print(s, flush=True)


d = np.load(NPZ)
E = d["extrinsics"].astype(np.float64)
K0 = d["intrinsics"].astype(np.float64)
DEP = d["depth"].astype(np.float64)
CONF = d["conf"].astype(np.float64)
V, H, W = DEP.shape
Rc = E[:, :, :3]
conf_thr = np.percentile(CONF, 70)
log(f"npz V={V} {H}x{W}  fx/fy={np.mean(K0[:,0,0]/K0[:,1,1]):.5f}")

# 参考图像（photometric 用）
import cv2
refs = sorted([p for p in REF_DIR.iterdir()
               if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
IMGS = np.stack([cv2.cvtColor(cv2.imread(str(refs[v])), cv2.COLOR_BGR2RGB) for v in range(V)])
IMGS = np.stack([cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA) for im in IMGS]).astype(np.float32)

# 预计算抽样像素
us, vs = np.meshgrid(np.arange(0, W, NSUB), np.arange(0, H, NSUB))
us = us.astype(np.float64) + 0.5 * (NSUB - 1)     # 抽样点仍在像素中心附近
vs = vs.astype(np.float64) + 0.5 * (NSUB - 1)


def lift(i, K):
    """view i 的抽样深度抬成世界点，返回 (Pw, color, z)"""
    fx, fy, cx, cy = K[i, 0, 0], K[i, 1, 1], K[i, 0, 2], K[i, 1, 2]
    z = DEP[i][vs.astype(int), us.astype(int)]
    c = CONF[i][vs.astype(int), us.astype(int)]
    ok = (c > conf_thr) & (z > 1e-6)
    x = (us - cx) / fx * z
    y = (vs - cy) / fy * z
    Pc = np.stack([x, y, z], -1)[ok]
    col = IMGS[i][vs.astype(int), us.astype(int)][ok]
    Pw = (Pc - E[i, :, 3]) @ Rc[i]
    return Pw, col, Pc[:, 2]


def project(j, Pf, K):
    fx, fy, cx, cy = K[j, 0, 0], K[j, 1, 1], K[j, 0, 2], K[j, 1, 2]
    Xc = Pf @ Rc[j].T + E[j, :, 3]
    z = Xc[:, 2]
    fr = z > 1e-6
    u = fx * Xc[:, 0] / np.maximum(z, 1e-9) + cx
    v = fy * Xc[:, 1] / np.maximum(z, 1e-9) + cy
    return u, v, z, fr


def metrics(K):
    """返回 (深度相对误差中位, <1%占比, 光度 warp 误差)"""
    rels, w1, pho = [], [], []
    for i in range(V):
        Pw, col, zi = lift(i, K)
        for k in PAIRS:
            j = (i + k) % V
            u, v, z, fr = project(j, Pw, K)
            ui = np.round(u).astype(int)
            vi = np.round(v).astype(int)
            ins = fr & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            if ins.sum() < 800:
                continue
            zj = DEP[j][vi[ins], ui[ins]]
            cj = CONF[j][vi[ins], ui[ins]]
            ok = (cj > conf_thr) & (zj > 1e-6)
            if ok.sum() < 800:
                continue
            rel = np.abs(z[ins][ok] - zj[ok]) / zj[ok]
            rels.append(np.median(rel))
            w1.append(float((rel < 0.01).mean()))
            # 光度：只比"深度也大致一致"的点，避免把遮挡物算进来
            good = rel < np.percentile(rel, 60)
            if good.sum() > 200:
                a = col[ins][ok][good]
                b = IMGS[j][vi[ins][ok][good], ui[ins][ok][good]]
                pho.append(float(np.mean(np.abs(a - b))))
    return (float(np.median(rels)), float(np.median(w1)), float(np.median(pho)))


base_alpha = float(np.mean(K0[:, 0, 0] / K0[:, 1, 1]))
rows = []

log("\n[A] fy/fx 扫掠（其余保持 DA3 原始预测）")
for a in ALPHAS:
    K = K0.copy()
    K[:, 1, 1] = K[:, 0, 0] / a            # 让 fx/fy = a
    rm, w1, ph = metrics(K)
    rows.append(dict(kind="alpha", value=a, rel_med=rm, within1pct=w1, photo=ph))
    log(f"  fx/fy={a:.4f}  深度相对误差中位 {rm*100:5.2f}%  <1% {w1*100:5.1f}%  "
        f"光度 |Δ| {ph:6.2f}")

log("\n[B] 全局焦距系数扫掠（fx,fy 同乘 β；fy/fx 保持 {:.4f}）".format(base_alpha))
for b in BETAS:
    K = K0.copy()
    K[:, 0, 0] *= b
    K[:, 1, 1] *= b
    rm, w1, ph = metrics(K)
    rows.append(dict(kind="beta", value=b, rel_med=rm, within1pct=w1, photo=ph))
    log(f"  β={b:.3f}  深度相对误差中位 {rm*100:5.2f}%  <1% {w1*100:5.1f}%  光度 |Δ| {ph:6.2f}")

best_a = min((r for r in rows if r["kind"] == "alpha"), key=lambda r: r["photo"])
best_b = min((r for r in rows if r["kind"] == "beta"), key=lambda r: r["photo"])
a0 = [r for r in rows if r["kind"] == "alpha" and abs(r["value"] - 1.0) < 1e-9][0]
b0 = [r for r in rows if r["kind"] == "beta" and abs(r["value"] - 1.0) < 1e-9][0]
log("\n" + "=" * 66)
log(f"方像素先验（α=1.000）光度误差 {a0['photo']:.2f}  vs  DA3 原始（α={base_alpha:.4f}）"
    f" 光度误差 见上表；最优 α={best_a['value']:.4f}（{best_a['photo']:.2f}）")
log(f"焦距最优 β={best_b['value']:.3f}（{best_b['photo']:.2f}） vs β=1（{b0['photo']:.2f}）")
log(f"⇒ 各向异性是否可修：{'✅ α=1 更好，方像素先验可用' if best_a['value'] == 1.0 or a0['photo'] <= best_a['photo'] + 0.05 else '⚠️ 最优不在 α=1'}")
log(f"⇒ 焦距整体是否偏：{'✅ 峰在 β=1' if abs(best_b['value'] - 1.0) < 1e-9 else '⚠️ 峰在 β=' + format(best_b['value'], '.3f')}")

with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    json.dump(dict(src=str(SRC), alpha_da3=base_alpha, rows=rows), f,
              ensure_ascii=False, indent=1, default=float)
log(f"\n💾 {OUT}")
