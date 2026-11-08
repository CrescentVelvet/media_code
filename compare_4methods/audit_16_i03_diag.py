#!/usr/bin/env python3
"""AUDIT-16 · I-03 解释性诊断：2DGS 的损失集中在哪？

I-03 的判定是 B 档（median 深度大幅改善，但 RGB 更糊）。要解释"为什么更糊"，
需要回答一个二分问题：

  H1（透视问题）：损失集中在掠射区（slant>60°）→ 是 ray-splat 求交在高倾斜下的病态
  H2（法向问题）：损失在所有 slant 档上都差不多 → 是 DA3 的盘法向本身随机
      （audit_12：法向 vs 真实表面 56.9~66.4°，随机基线 57.3°），
      2DGS 拿这个法向去做求交平面，等于把每个高斯投到一个随机朝向的平面上，
      屏幕足迹被随机拉伸 → 层间弥散变宽 → RGB 混合更糊、expected 深度更散。
      这与 slant 无关，所以损失应当**均匀分布**。

判别很干净：如果 ΔMAE 在 0-30° 档也明显为负，H2 成立、H1 不成立。
这直接决定 I-04 的必要性——法向必须先有意义（靠正则学出来），2DGS 才有意义。

只读，不重渲（用 I-03 已存的 PNG + depth npy）。
"""
import json
import os

import numpy as np
from PIL import Image

NPZ = os.environ["NPZ"]
OUT = os.environ["OUT"]
VIEWS = [int(x) for x in os.environ.get("VIEWS", "0,8,16").split(",")]


def log(*a):
    print(*a, flush=True)


d = np.load(NPZ)
W2C = d["extrinsics"].astype(np.float64)
if W2C.shape[-2] == 4:
    W2C = W2C[:, :3, :]
K = d["intrinsics"].astype(np.float64)
DEP = d["depth"].astype(np.float64)
GT = d["processed_images"]
V, H, W = DEP.shape
fx, fy = K[:, 0, 0], K[:, 1, 1]
cx, cy = K[:, 0, 2], K[:, 1, 2]
uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)

FACE = (0.35, 0.105, 0.565, 0.335)
fbox = (int(FACE[0] * W), int(FACE[1] * H), int(FACE[2] * W), int(FACE[3] * H))


def slant_of(v):
    R, t = W2C[v, :, :3], W2C[v, :, 3]
    z = DEP[v]
    xc = np.stack([(uu - cx[v]) / fx[v] * z, (vv - cy[v]) / fy[v] * z, z], -1)
    Pw = (xc - t) @ R
    C = -R.T @ t
    du = np.zeros_like(Pw)
    dv = np.zeros_like(Pw)
    du[:, :-1] = Pw[:, 1:] - Pw[:, :-1]
    dv[:-1, :] = Pw[1:, :] - Pw[:-1, :]
    n = np.cross(du, dv)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
    view = Pw - C[None, None, :]
    view /= np.maximum(np.linalg.norm(view, axis=-1, keepdims=True), 1e-12)
    return np.degrees(np.arccos(np.clip(np.abs(np.sum(n * view, -1)), -1, 1))), z > 0


TIERS = (("0-30", 0, 30), ("30-60", 30, 60), ("60-90", 60, 90))
log(f"{'view':>5} " + " ".join(f"{t[0]+'°':>16}" for t in TIERS)
    + f" {'全图':>14}")
log(f"{'':>5} " + " ".join(f"{'3DGS / 2DGS / Δ':>16}" for _ in TIERS) + f" {'ΔMAE':>14}")

rep = {"views": {}, "per_tier": {}}
agg = {t[0]: {"g": [], "b": []} for t in TIERS}
agg_all = {"g": [], "b": []}

for v in VIEWS:
    sl, gtv = slant_of(v)
    g3 = np.asarray(Image.open(os.path.join(OUT, f"3dgs_view{v:03d}.png"))).astype(np.float64)
    g2 = np.asarray(Image.open(os.path.join(OUT, f"2dgs_view{v:03d}.png"))).astype(np.float64)
    gt = GT[v].astype(np.float64)
    e3 = np.abs(g3 - gt).mean(-1)
    e2 = np.abs(g2 - gt).mean(-1)
    row = []
    for tag, lo, hi in TIERS:
        m = gtv & (sl >= lo) & (sl < hi)
        a, b = e3[m].mean(), e2[m].mean()
        agg[tag]["g"].append(a)
        agg[tag]["b"].append(b)
        row.append(f"{a:6.2f} / {b:6.2f} / {b-a:+5.2f}")
    m0 = gtv
    agg_all["g"].append(e3[m0].mean())
    agg_all["b"].append(e2[m0].mean())
    log(f"{v:>5} " + " ".join(f"{x:>16}" for x in row)
        + f" {e2[m0].mean()-e3[m0].mean():>+13.3f}")

    # 人脸窗（与 I-03 同一框）
    fm = np.zeros((H, W), bool)
    fm[fbox[1]:fbox[3], fbox[0]:fbox[2]] = True
    face3, face2 = e3[fm].mean(), e2[fm].mean()
    # 局部锐度（Laplacian var）按 slant 分档
    import cv2
    sh = {}
    for tag, lo, hi in TIERS:
        m = gtv & (sl >= lo) & (sl < hi)
        if m.sum() < 500:
            continue
        for nm, im in (("g3", g3), ("g2", g2)):
            gr = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)
            lap = np.abs(cv2.Laplacian(gr, cv2.CV_64F))
            sh.setdefault(tag, {})[nm] = float(lap[m].mean())
    rep["views"][str(v)] = dict(face_mae_3dgs=float(face3), face_mae_2dgs=float(face2),
                                face_delta=float(face2 - face3), sharp_by_tier=sh)
    log(f"      人脸窗 MAE 3DGS {face3:.2f} → 2DGS {face2:.2f}（Δ {face2-face3:+.2f}）")
    log(f"      局部梯度均值（|Laplacian|）  " + "  ".join(
        f"{t}: {sh[t]['g3']:.2f}→{sh[t]['g2']:.2f}({sh[t]['g2']-sh[t]['g3']:+.2f})"
        for t in sh))

log("\n" + "=" * 74)
log("汇总（3 视角均值，MAE 单位 0-255）")
log("=" * 74)
log(f"{'slant':>8} {'3DGS':>9} {'2DGS':>9} {'Δ':>9} {'相对':>9}")
for tag, lo, hi in TIERS:
    a, b = np.mean(agg[tag]["g"]), np.mean(agg[tag]["b"])
    rep["per_tier"][tag] = dict(mae_3dgs=float(a), mae_2dgs=float(b),
                                delta=float(b - a), rel=float((b - a) / a))
    log(f"{tag+'°':>8} {a:>9.2f} {b:>9.2f} {b-a:>+9.2f} {(b-a)/a:>+8.2%}")
a, b = np.mean(agg_all["g"]), np.mean(agg_all["b"])
rep["per_tier"]["all"] = dict(mae_3dgs=float(a), mae_2dgs=float(b),
                              delta=float(b - a), rel=float((b - a) / a))
log(f"{'全图':>8} {a:>9.2f} {b:>9.2f} {b-a:>+9.2f} {(b-a)/a:>+8.2%}")

# ── 判别 H1 vs H2 ────────────────────────────────────────────────────────────
log("\n【H1 vs H2 判别】")
t0 = rep["per_tier"]["0-30"]
t2 = rep["per_tier"]["60-90"]
r0, r2 = t0["rel"], t2["rel"]
log(f"  H1（透视/掠射病态）：掠射档损失应**显著大于**正对档（r2 > 1.5·r0）"
    f"  → 实测 r0 {r0:+.2%} / r2 {r2:+.2%}")
log(f"  H2（法向随机）      ：各档损失接近（|r2−r0| ≤ 0.5·max）"
    f"  → 实测差 {abs(r2-r0):.2%} vs 阈值 {0.5*max(abs(r0),abs(r2)):.2%}")
if r2 > 1.5 * r0:
    verdict = "H1 透视病态（掠射档显著更差）"
elif abs(r2 - r0) <= 0.5 * max(abs(r0), abs(r2)):
    verdict = "H2 法向随机（各档均匀变差，正对档甚至更大）"
else:
    verdict = "H1+H2 叠加"
log(f"  ⇒ **{verdict}**")
if verdict.startswith("H2"):
    log("     正对档（slant<30°）本该是 ray-splat 求交最稳的区域，它却变得更差 ——")
    log("     说明损失来源不是透视，而是**每个盘的朝向本身没有几何意义**")
    log("     （audit_12：DA3 盘法向 vs 真实表面 56.9~66.4°，随机基线 57.3°）。")
    log("     2DGS 拿这个法向去定求交平面 = 把每个高斯投到一个随机朝向的平面上，")
    log("     屏幕足迹被随机拉伸；而 3DGS 的 EWA 走 `add_blur(eps2d, covar2d)` 兜底，")
    log("     对第三轴退化不敏感，反而稳。→ I-04（法向正则）是让 2DGS 有意义的前提。")
rep["verdict"] = verdict

with open(os.path.join(OUT, "i03_diag.json"), "w", encoding="utf-8", newline="\n") as f:
    json.dump(rep, f, indent=1, ensure_ascii=False, default=float)
log(f"\n💾 {os.path.join(OUT, 'i03_diag.json')}")
