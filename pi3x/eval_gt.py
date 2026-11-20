#!/usr/bin/env python3
"""Pi3X 预测 vs 采集包 GT（transforms.json 经 02 转成 conditions.npz）。

指标（2026-11-20 修订）：
  - Sim(3) Umeyama（相机中心）→ scale / ATE RMSE (m)。注意：近平面轨迹下中心对齐在
    面外方向退化（本包第三主成分 0.03m），此时 ATE 与「中心对齐旋转误差」不可靠，
    应以后两项为准。
  - 旋转误差：Wahba 全局旋转对齐（与中心对齐解耦）后的逐帧角度差（deg）。
  - RPE（相邻帧，对齐无关）：相对位姿的平移差 (m) 与旋转差 (deg)。
  - 内参：恢复 K（resize 像素）换算回去畸变全幅后与 GT newK 的相对误差。

Env vars:
  PRED_DIR    03 的输出目录（含 predictions.npz）
  SCENE_DIR   02 的输出目录（含 conditions.npz / gt_meta.json）
  OUT_JSON    可选，默认 PRED_DIR/eval_gt.json
"""
import json
import os
import sys

import numpy as np

PRED_DIR = os.environ.get("PRED_DIR", "")
SCENE_DIR = os.environ.get("SCENE_DIR", "")
OUT_JSON = os.environ.get("OUT_JSON", "")


def umeyama(src, dst):
    """src,dst (N,3) → s,R,t 使 dst ≈ s*(src@R.T)+t。带反射修正的标准实现。"""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    cov = D.T @ S / len(src)
    U, sig, Vt = np.linalg.svd(cov)
    corr = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ corr @ U.T
    var_s = float((S ** 2).sum() / len(src))
    s = float((sig * np.diag(corr)).sum() / var_s)
    t = mu_d - s * R @ mu_s
    return s, R, t


def wahba(Rp, Rg):
    """找全局旋转 Rw 最小化 sum ||Rw @ Rp_i - Rg_i||_F。"""
    M = np.einsum("nij,nkj->ik", Rp, Rg)  # sum_i Rp_i @ Rg_i^T
    U, _, Vt = np.linalg.svd(M)
    corr = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ corr @ U.T


def rot_deg(dR):
    c = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def main():
    if not PRED_DIR or not SCENE_DIR:
        sys.exit("❌ PRED_DIR / SCENE_DIR 未设")
    pred = np.load(os.path.join(PRED_DIR, "predictions.npz"), allow_pickle=True)
    gt = np.load(os.path.join(SCENE_DIR, "conditions.npz"))
    with open(os.path.join(SCENE_DIR, "gt_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)

    P = pred["camera_poses"][:, :3, 3].astype(np.float64)   # (N,3) pred 中心
    G = gt["poses"][:, :3, 3].astype(np.float64)            # (N,3) gt 中心 (m)
    Rp = pred["camera_poses"][:, :3, :3].astype(np.float64)
    Rg = gt["poses"][:, :3, :3].astype(np.float64)
    n = len(P)
    assert n == len(G), "帧数不匹配: pred %d vs gt %d" % (n, len(G))
    print("🚀 [eval] %d 帧  pred=%s" % (n, PRED_DIR))

    # --- 1. Sim(3) 中心对齐（平面退化仅作参考）---
    s, R, t = umeyama(P, G)
    A = s * (P @ R.T) + t
    terr = np.linalg.norm(A - G, axis=1)
    Gc = G - G.mean(0)
    pca_sv = np.linalg.svd(Gc, compute_uv=False) / np.sqrt(n)

    # --- 2. Wahba 旋转对齐（与中心解耦）---
    Rw = wahba(Rp, Rg)
    rerr = np.array([rot_deg(Rg[i] @ (Rw @ Rp[i]).T) for i in range(n)])

    # --- 3. RPE 相邻帧（对齐无关；pred 平移乘 sim3 scale 换成米）---
    tp = np.concatenate([Rp, (s * P)[..., None]], axis=2)
    tp = np.concatenate([tp, np.tile(np.array([0, 0, 0, 1.0]), (n, 1, 1))], axis=1)
    tg = gt["poses"].astype(np.float64)
    rpe_t, rpe_r = [], []
    for i in range(n - 1):
        dTp = np.linalg.inv(tp[i]) @ tp[i + 1]
        dTg = np.linalg.inv(tg[i]) @ tg[i + 1]
        d = np.linalg.inv(dTg) @ dTp
        rpe_t.append(float(np.linalg.norm(d[:3, 3])))
        rpe_r.append(rot_deg(d[:3, :3]))
    rpe_t, rpe_r = np.array(rpe_t), np.array(rpe_r)

    # --- 4. 内参：resize 像素 → 去畸变全幅像素 ---
    tw, th = [int(v) for v in pred["target_wh"]]
    rw, rh = meta["crop_xywh"][2], meta["crop_xywh"][3]
    K_rec = pred["K_recovered"].astype(np.float64)
    K_gt = gt["intrinsics"].astype(np.float64)
    fx_pred = K_rec[:, 0, 0] * rw / tw
    fy_pred = K_rec[:, 1, 1] * rh / th
    fx_gt, fy_gt = K_gt[:, 0, 0], K_gt[:, 1, 1]
    fx_rel = np.abs(fx_pred - fx_gt) / fx_gt
    fy_rel = np.abs(fy_pred - fy_gt) / fy_gt

    summary = {
        "n_frames": n,
        "sim3_scale": s,
        "ate_rmse_m": float(np.sqrt((terr ** 2).mean())),
        "trans_err_m": {"mean": float(terr.mean()), "median": float(np.median(terr)),
                        "max": float(terr.max())},
        "centers_pca_std_m": [float(v) for v in pca_sv],
        "planar_degenerate": bool(pca_sv[2] < 0.1 * pca_sv[0]),
        "rot_err_wahba_deg": {"mean": float(rerr.mean()), "median": float(np.median(rerr)),
                              "max": float(rerr.max())},
        "rpe_adjacent": {"trans_mean_m": float(rpe_t.mean()), "trans_max_m": float(rpe_t.max()),
                         "rot_mean_deg": float(rpe_r.mean()), "rot_max_deg": float(rpe_r.max())},
        "fx_rel_err": {"mean": float(fx_rel.mean()), "median": float(np.median(fx_rel))},
        "fy_rel_err": {"mean": float(fy_rel.mean()), "median": float(np.median(fy_rel))},
        "gt_track_len_m": float(np.linalg.norm(np.diff(G, axis=0), axis=1).sum()),
    }
    out = OUT_JSON or os.path.join(PRED_DIR, "eval_gt.json")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("  📏 Sim(3) scale = %.4f（pred 单位 → m）" % s)
    print("  📍 ATE RMSE = %.4f m | trans mean/med/max = %.4f/%.4f/%.4f m%s"
          % (summary["ate_rmse_m"], terr.mean(), np.median(terr), terr.max(),
             "  ⚠️ 近平面轨迹，中心对齐面外退化，仅供参考" if summary["planar_degenerate"] else ""))
    print("  🔄 rot err (Wahba) mean/med/max = %.2f/%.2f/%.2f deg"
          % (rerr.mean(), np.median(rerr), rerr.max()))
    print("  🧭 RPE 相邻帧 trans mean/max = %.4f/%.4f m | rot mean/max = %.2f/%.2f deg"
          % (rpe_t.mean(), rpe_t.max(), rpe_r.mean(), rpe_r.max()))
    print("  🔍 fx rel err mean = %.2f%% | fy rel err mean = %.2f%%"
          % (100 * fx_rel.mean(), 100 * fy_rel.mean()))
    print("✅ -> %s" % out)
    print("🎉 Done.")


if __name__ == "__main__":
    main()
