#!/usr/bin/env python3
"""诊断 v2：头模型「前后拉长」根因定位（定量）。

重点回答三个问题：
  Q1 三角化 468 点云是否被运动/噪声沿某方向拖长？（对比 MP canonical 形状）
  Q2 拟合出的头网格 shape 是否被 identity 拉长？（对比模板 PCA 轴长）
  Q3 重投影误差在「脸部前突点」上是否系统性偏前？

用法（WSL, conda vggt_human）：
  MODEL_3DMM_DIR=~/model/ICT-FaceKit RESULTS_DIR=/mnt/d/output/vggt_human_ms \
      python diag_head_v2.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_head_3dmm import (  # noqa: E402
    N_LM,
    build_proj_matrices,
    parse_colmap_cameras,
    qvec2rotmat,
    triangulate_all,
)

# MediaPipe canonical 语义点（prepare_3dmm_template.py 里的常量）
NOSE_TIP = 1
LM_LEFT_EAR = 234
LM_RIGHT_EAR = 454


def umeyama(src, dst, with_scale=True):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    X, Y = src - mu_s, dst - mu_d
    C = Y.T @ X / len(src)
    U, S, Vt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1.0, 1.0, d]) @ Vt
    s = (S * np.asarray([1.0, 1.0, d])).sum() / (X**2).sum() if with_scale else 1.0
    return s, R, mu_d - s * (R @ mu_s)


def pca_axes(P):
    """返回 (轴长_std, 主轴行向量)，按方差降序。"""
    X = P - P.mean(0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return S / np.sqrt(len(P)), Vt


def main():
    model_dir = Path(os.environ.get("MODEL_3DMM_DIR", "~/model/ICT-FaceKit")).expanduser()
    R = Path(os.environ.get("RESULTS_DIR", "/mnt/d/output/vggt_human_ms"))
    tmpl = np.load(str(model_dir / "3dmm_template.npz"), allow_pickle=True)
    mc = tmpl["mp_canonical"]
    lm = json.loads((R / "03e_head_3dmm" / "face_landmarks.json").read_text())
    views = parse_colmap_cameras(str(R / "03_source"))
    projs = build_proj_matrices(views)
    fit = json.loads((R / "03e_head_3dmm" / "head_fit.json").read_text())

    # 相机中心（用 R|t 直接算，避免 SVD 量纲问题）
    cam_pos = {}
    for v in views:
        Rm = qvec2rotmat(v["qvec"])
        cam_pos[v["stem"]] = -Rm.T @ np.asarray(v["tvec"]).reshape(3)

    print("参考形状 MP canonical 468 点云:")
    L, A = pca_axes(mc)
    print(f"  PCA 轴长(std): {np.round(L,3)}  比值 {np.round(L/L[0],3)}")
    bb = mc.max(0) - mc.min(0)
    print(f"  bbox: {np.round(bb,2)}  深(z)/宽(x)={bb[2]/bb[0]:.2f}")

    for pid in sorted(fit["persons"]):
        recs = []
        for stem, rec in lm["frames"].items():
            if stem not in projs:
                continue
            pr = rec.get("persons", {}).get(pid)
            if not pr or not pr.get("ok") or not pr.get("lm"):
                continue
            a = np.asarray(pr["lm"], dtype=np.float64)
            if a.shape == (N_LM, 2):
                recs.append((stem, a))
        if len(recs) < 4:
            continue
        stems = [s for s, _ in recs]
        Ps = [projs[s][0] for s in stems]
        obs = np.stack([a for _, a in recs])
        Xw = triangulate_all(Ps, obs)
        cp = np.asarray([cam_pos[s] for s in stems])

        print(f"\n{'='*72}\n👤 pid={pid}  观测 {len(recs)} 帧")

        # --- Q1 三角化点云形状 ---
        Lt, At = pca_axes(Xw)
        print(f"  [Q1] 三角化点云 PCA 轴长: {np.round(Lt,4)} 比值 {np.round(Lt/Lt[0],3)}")
        s, Rm, t = umeyama(mc, Xw)
        res = (s * (Rm @ mc.T)).T + t - Xw
        print(f"       vs canonical 各向同性配准 rms={np.sqrt((res**2).sum(1).mean()):.4f} "
              f"(头尺寸≈{np.linalg.norm(Lt)*2:.4f} → 相对 {np.sqrt((res**2).sum(1).mean())/np.linalg.norm(Lt):.1%})")
        # 三角化点云主轴 vs 视线方向
        vdir = Xw.mean(0) - cp.mean(0)
        vdir /= np.linalg.norm(vdir)
        for k in range(3):
            print(f"       axis{k} len={Lt[k]:.4f}  |cos(视线)|={abs(At[k]@vdir):.2f}")

        # 射线夹角（正确算法）
        ang = []
        for j in range(0, N_LM, 7):
            v = cp - Xw[j]
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            G = v @ v.T
            np.fill_diagonal(G, -2)
            ang.append(np.degrees(np.arccos(np.clip(G.max(), -1, 1))))
        ang = np.asarray(ang)
        print(f"  [Q1b] 射线夹角 med={np.median(ang):.1f}° p10={np.percentile(ang,10):.1f}° "
              f"max={ang.max():.1f}°")

        # --- 运动量：每帧 landmark 2D bbox 中心漂移（世界尺度换算） ---
        cen2 = obs.mean(1)
        K = np.array([[views[0]["fx"], 0, views[0]["cx"]],
                      [0, views[0]["fy"], views[0]["cy"]], [0, 0, 1.0]])
        dep = np.median(np.linalg.norm(Xw.mean(0) - cp, axis=1))
        drift_px = np.linalg.norm(cen2 - cen2.mean(0), axis=1)
        f_px = (K[0, 0] + K[1, 1]) / 2
        print(f"  [Q1c] 2D landmark 中心漂移: med={np.median(drift_px):.1f}px "
              f"max={drift_px.max():.1f}px → 世界 ≈{np.median(drift_px)*dep/f_px*100:.1f}cm "
              f"(头尺寸≈{np.linalg.norm(Lt)*2*100:.1f}cm, 深度≈{dep*100:.1f}cm)")

        # --- Q2 拟合网格形状 ---
        fr = fit["persons"][pid]
        sc = fr["scale"]
        beta = np.asarray(fr.get("beta") or [])
        Rref = np.asarray(fr["R_ref"])
        tref = np.asarray(fr["t_ref"])
        base = tmpl["id_mean"]
        V = base + np.einsum("b,bij->ij", beta, tmpl["id_basis"]) if len(beta) else base
        L0, A0 = pca_axes(base)
        L1, A1 = pca_axes(V)
        print(f"  [Q2] 模板 PCA 轴长: {np.round(L0,4)} → 拟合后 {np.round(L1,4)} "
              f"变化 {np.round(L1/L0,3)}")
        print(f"       |beta|={np.linalg.norm(beta):.2f}  scale={sc:.5g}  "
              f"头半径p90={fr['head_radius_p90']:.4f}")

        # 拟合网格到头中心的「前突」：鼻尖 landmark 沿视线方向的深度
        lm_idx, bary = tmpl["lm468_idx"], tmpl["lm468_bary"]
        lmV = (V[lm_idx] * bary[:, :, None]).sum(1)
        lmW = sc * (Rref @ lmV.T).T + tref
        # 各帧重投影误差（用拟合的 per-frame 位姿）
        pf = fr["per_frame"]
        errs = []
        for stem, a in recs:
            if stem not in pf:
                continue
            Rf = np.asarray(pf[stem]["R"])
            tf = np.asarray(pf[stem]["t"])
            Xf = sc * (Rf @ lmV.T).T + tf
            Xh = np.hstack([Xf, np.ones((len(Xf), 1))])
            q = (projs[stem][0] @ Xh.T).T
            q = q[:, :2] / q[:, 2:3]
            errs.append(np.linalg.norm(q - a, axis=1))
        E = np.asarray(errs)
        med_e = np.median(E, axis=0)
        print(f"  [Q3] 重投影 med={np.median(med_e):.2f}px p90={np.percentile(med_e,90):.2f}px")
        # 前突点（鼻尖 1、鼻梁 4/5、下巴 152/199）vs 侧边（耳 234/454）
        front = [1, 4, 5, 6, 195, 197, 152, 199, 2, 98, 327]
        side = [234, 454, 127, 356, 93, 323]
        print(f"       前突点 med_err={np.median(med_e[front]):.2f}px  "
              f"侧边点 med_err={np.median(med_e[side]):.2f}px")

        # --- Q4 头网格世界尺寸 ---
        M = np.load(R / "03e_head_3dmm" / f"head_mesh_p{pid}.npz")
        Vw = M["verts_world"]
        Lw, Aw = pca_axes(Vw)
        print(f"  [Q4] 拟合头网格 PCA 轴长(世界): {np.round(Lw*100,2)}cm  "
              f"比值 {np.round(Lw/Lw[0],3)}")
        Lb, _ = pca_axes(base)
        print(f"       模板(局部) PCA 轴长: {np.round(Lb,3)} 比值 {np.round(Lb/Lb[0],3)} "
              f"→ 世界应为 {np.round(Lb*sc*100,2)}cm")
        print(f"       ⚖️ 实际/应然 轴长比: {np.round(Lw/(Lb*sc),3)}")


if __name__ == "__main__":
    main()
