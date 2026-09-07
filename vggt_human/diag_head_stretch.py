#!/usr/bin/env python3
"""诊断：头模型「前后拉长」的根因定位。

对比链：
  ① 三角化得到的 468 landmark 点云（世界坐标）
  ② MediaPipe canonical 468（真实人脸形状，各向同性基准）
  ③ 拟合后的模板 landmark（3DMM 输出）

报告：
  A. 三角化点云 vs canonical 的各向同性残差 / 各向异性三轴缩放
  B. 三角化几何条件（多视角射线夹角 → 深度不确定性放大倍数）
  C. identity 系数引起的形变各向异性
  D. 拟合残差沿「深度方向」的分量占比

用法（WSL）：
  MODEL_3DMM_DIR=~/model/ICT-FaceKit RESULTS_DIR=~/output/vggt_human_ms \
    python3 diag_head_stretch.py [--pid 2]
"""
import argparse
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


def umeyama(src, dst, with_scale=True):
    """src→dst 相似变换，返回 (s, R, t)。"""
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    X = src - mu_s
    Y = dst - mu_d
    C = Y.T @ X / len(src)
    U, S, Vt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, 1.0, d])
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / (X**2).sum() if with_scale else 1.0
    t = mu_d - s * (R @ mu_s)
    return s, R, t


def aniso_scales(src, dst, n_iter=60):
    """各向异性配准：求对角 D 使 s*R*D*src + t ≈ dst，返回三轴缩放（已归一）。
    在 src 的 PCA 主轴坐标系里求 D。"""
    mu = src.mean(0)
    Xc = src - mu
    # PCA 主轴
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    A = Vt  # (3,3) 行=主轴
    sp = Xc @ A.T  # src 在主轴坐标
    dp = dst - dst.mean(0)
    d = np.ones(3)
    for _ in range(n_iter):
        cur = sp * d
        s, R, t = umeyama(cur, dp)
        cur_p = (s * (R @ cur.T)).T + t
        # 目标在主轴坐标
        dp_local = (R.T @ (dp - t).T).T / s
        for k in range(3):
            num = (sp[:, k] * dp_local[:, k]).sum()
            den = (sp[:, k] ** 2).sum() + 1e-12
            d[k] = num / den
    d = d / np.cbrt(np.prod(d))  # 归一化，去掉整体尺度
    return d, A, sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", default="2")
    args = ap.parse_args()

    model_dir = Path(os.environ.get("MODEL_3DMM_DIR", "~/model/ICT-FaceKit")).expanduser()
    results_dir = Path(os.environ.get("RESULTS_DIR", "~/output/vggt_human_ms")).expanduser()
    tmpl = np.load(str(model_dir / "3dmm_template.npz"), allow_pickle=True)
    lm = json.loads((results_dir / "03e_head_3dmm" / "face_landmarks.json").read_text())
    views = parse_colmap_cameras(str(results_dir / "03_source"))
    projs = build_proj_matrices(views)
    fit = json.loads((results_dir / "03e_head_3dmm" / "head_fit.json").read_text())

    mp_canonical = tmpl["mp_canonical"]  # (468,3) MP canonical，真实人脸形状

    for pid in ([args.pid] if args.pid != "all" else sorted(fit["persons"])):
        recs = []
        for stem, rec in lm["frames"].items():
            if stem not in projs:
                continue
            pr = rec.get("persons", {}).get(pid)
            if not pr or not pr.get("ok") or not pr.get("lm"):
                continue
            arr = np.asarray(pr["lm"], dtype=np.float64)
            if arr.shape == (N_LM, 2):
                recs.append((stem, arr))
        if len(recs) < 4:
            print(f"pid={pid} 观测不足 {len(recs)}")
            continue

        stems = [s for s, _ in recs]
        Ps = [projs[s][0] for s in stems]
        obs = np.stack([a for _, a in recs])
        Xw = triangulate_all(Ps, obs)

        print(f"\n{'='*70}\n👤 pid={pid}  观测帧 {len(recs)}")

        # ---------- B. 三角化几何条件 ----------
        cam_pos = []
        for P in Ps:
            # P = K[R|t]，相机中心 C = -R^T t
            K = np.eye(3)
            M = np.linalg.inv(P[:, :3])
            K_ = P[:, :3]
            # 用 P 分解：先估 K 的尺度不准，直接用 SVD 求零空间更快
            _, _, Vt = np.linalg.svd(P)
            C = Vt[-1]
            cam_pos.append(C[:3] / C[3])
        cam_pos = np.asarray(cam_pos)
        ang = []
        for j in range(N_LM):
            if j % 7:  # 抽样
                continue
            v = cam_pos - Xw[j]
            v = v / np.linalg.norm(v, axis=1, keepdims=True)
            # 最大两两夹角
            G = v @ v.T
            np.fill_diagonal(G, -2)
            ang.append(np.degrees(np.arccos(np.clip(G.max(), -1, 1))))
        ang = np.asarray(ang)
        print(f"  📐 三角化射线夹角: med={np.median(ang):.2f}°  p10={np.percentile(ang,10):.2f}°  "
              f"→ 深度/横向不确定比 ≈ {1/np.sin(np.radians(max(np.median(ang),1e-3))):.1f}x")

        # ---------- A. 三角化点云 vs canonical ----------
        s, R, t = umeyama(mp_canonical, Xw)
        res = (s * (R @ mp_canonical.T)).T + t - Xw
        rms = np.sqrt((res**2).sum(1).mean())
        print(f"  📏 各向同性配准: s={s:.4g}  rms={rms:.3f} (世界单位)  "
              f"≈ {rms/s*100:.2f}% of canonical 尺寸")

        d, axes, _ = aniso_scales(mp_canonical, Xw)
        # 把主轴方向换算到世界/相机语义
        cam_mean = cam_pos.mean(0)
        view_dir = Xw.mean(0) - cam_mean
        view_dir /= np.linalg.norm(view_dir)
        print("  🧊 各向异性三轴缩放（dev=相对各向同性，>1=被拉长）:")
        for k in range(3):
            axw = axes[k] @ R.T if False else R @ axes[k]
            axw = axw / np.linalg.norm(axw)
            cosv = abs(float(axw @ view_dir))
            tag = "深度(视线)" if cosv > 0.6 else ("横向" if cosv < 0.35 else "斜向")
            print(f"      axis{k}: {d[k]:.3f}  |cos(视线)|={cosv:.2f}  {tag}")

        # ---------- A2. 沿视线方向的残差占比 ----------
        proj_v = (res @ view_dir)[:, None] * view_dir
        res_lat = res - proj_v
        print(f"  📊 残差分解: 沿视线 RMS={np.sqrt((proj_v**2).sum(1).mean()):.4f} | "
              f"横向 RMS={np.sqrt((res_lat**2).sum(1).mean()):.4f} | "
              f"比值={np.sqrt((proj_v**2).sum(1).mean())/max(np.sqrt((res_lat**2).sum(1).mean()),1e-9):.2f}")

        # ---------- C. identity 形变各向异性 ----------
        beta = np.asarray(fit["persons"][pid].get("beta") or [])
        if len(beta):
            base = tmpl["id_mean"]
            basis = tmpl["id_basis"]
            V = base + np.einsum("b,bij->ij", beta, basis)
            dmean = V.mean(0) - base.mean(0)
            deform = V - base
            nrm = np.linalg.norm(deform, axis=1)
            print(f"  🧬 identity: |beta|={np.linalg.norm(beta):.3f}  "
                  f"顶点形变 med={np.median(nrm):.4f} max={nrm.max():.4f} "
                  f"(模板尺度≈{np.linalg.norm(base.std(0)):.3f})")
            # 形变沿视线方向的偏置
            Rf = np.asarray(fit["persons"][pid]["R_ref"])
            axis_w = Rf.T @ view_dir  # 视线在模板局部坐标
            comp = deform @ (axis_w / np.linalg.norm(axis_w))
            print(f"     形变沿视线分量: mean={comp.mean():+.4f}  "
                  f"(正=整体前移, 非拉长)  前后不对称={np.percentile(comp,90)+np.percentile(comp,10):+.4f}")

        # ---------- D. 尺寸检查 ----------
        bb = Xw.max(0) - Xw.min(0)
        print(f"  📦 三角化点云包围盒: {np.round(bb,4).tolist()}  长宽比(x/z,y/z)={bb[0]/bb[2]:.2f},{bb[1]/bb[2]:.2f}")
        bb_c = mp_canonical.max(0) - mp_canonical.min(0)
        print(f"  📦 canonical 包围盒: {np.round(bb_c,4).tolist()}  长宽比(x/z,y/z)={bb_c[0]/bb_c[2]:.2f},{bb_c[1]/bb_c[2]:.2f}")

        # ---------- E. 头网格实际尺寸（世界） ----------
        m = np.load(results_dir / "03e_head_3dmm" / f"head_mesh_p{pid}.npz")
        Vw = m["verts_world"]
        bbw = Vw.max(0) - Vw.min(0)
        print(f"  🗿 拟合头网格包围盒(世界): {np.round(bbw*100,2).tolist()} cm  "
              f"对角线={np.linalg.norm(bbw)*100:.1f}cm")
        scale = fit["persons"][pid]["scale"]
        Vl = (Vw - np.asarray(fit["persons"][pid]["t_ref"])) / scale
        Rl = np.asarray(fit["persons"][pid]["R_ref"])
        Vloc = (Rl.T @ Vl.T).T
        bbl = Vloc.max(0) - Vloc.min(0)
        bbl0 = tmpl["id_mean"].max(0) - tmpl["id_mean"].min(0)
        print(f"  🗿 局部坐标包围盒: {np.round(bbl,3).tolist()} vs 模板均值 {np.round(bbl0,3).tolist()} "
              f"→ 比 {np.round(bbl/bbl0,3).tolist()}")


if __name__ == "__main__":
    main()
