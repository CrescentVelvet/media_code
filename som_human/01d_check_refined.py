#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""01d_check_refined.py — 01c COLMAP 精化模型的质量校验（01b 的精化版对照）

三项证据：
  1. 一致性：精化模型每帧 inlier（有效 2D-3D 观测）数量分布 —— 对比 01b 在
     VIO 位姿上测出的 436~112750（250 倍摆幅），精化后应收敛到窄分布。
  2. 内禀精度：points3D.txt 的重投影误差统计（中位/均值/max）。
  3. 漂移量化：VIO 相机中心 vs COLMAP 相机中心，Umeyama 相似变换对齐后
     逐帧位置残差 + 旋转角残差 —— 直接量出 App VIO 漂移幅度。
另外把精化点云投到若干帧生成叠加图（点带 COLMAP 三角化颜色），肉眼复核
「点坐在正确结构上」。

用法（deformable_human env）：
  python 01d_check_refined.py <refined_scene> <vio_scene> [n_frames] [out_dir]
  refined_scene : 01c 产物（colmap_refined，含 sparse/0/*.txt）
  vio_scene     : 01 产物（colmap_scene，VIO 位姿，用于漂移对比）
输出：
  stdout 统计表 + <out_dir>/overlay_<img>.png 叠加图
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("DG_DIR", os.path.expanduser(
    "~/repos/Deformable-3D-Gaussians")))
from scene.colmap_loader import (read_extrinsics_text, read_intrinsics_text,
                                 read_points3D_text, qvec2rotmat)


def umeyama(src, dst):
    """src→dst 相似变换 (s, R, t)，使 s*R@src + t ≈ dst。Horn/Umeyama 闭式解。"""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    A = src - mu_s
    B = dst - mu_d
    C = (B.T @ A) / len(src)
    U, S, Vt = np.linalg.svd(C)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / np.maximum((A ** 2).sum() / len(src), 1e-12)
    t = mu_d - s * R @ mu_s
    return s, R, t


def centers(ext):
    """COLMAP w2c → 相机中心（世界系）。返回 {name: (C, R_w2c)}"""
    out = {}
    for e in ext.values():
        R = qvec2rotmat(e.qvec)
        out[e.name] = (-R.T @ e.tvec, R)
    return out


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    ref_dir, vio_dir = sys.argv[1], sys.argv[2]
    n_check = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    out_dir = sys.argv[4] if len(sys.argv) > 4 else os.path.join(
        ref_dir, "reproj_check")
    os.makedirs(out_dir, exist_ok=True)

    ext_ref = read_extrinsics_text(os.path.join(ref_dir, "sparse/0/images.txt"))
    intr_ref = read_intrinsics_text(os.path.join(ref_dir, "sparse/0/cameras.txt"))
    pts_xyz, pts_rgb, pts_err = read_points3D_text(
        os.path.join(ref_dir, "sparse/0/points3D.txt"))
    ext_vio = read_extrinsics_text(os.path.join(vio_dir, "sparse/0/images.txt"))

    # ── 1. 每帧 inlier 数（points2D 中 point3D_id != -1 的观测）────────────
    n_in = {e.name: int((e.point3D_ids >= 0).sum()) for e in ext_ref.values()}
    v = np.array(sorted(n_in.values()))
    print("[01d] inlier/frame: min=%d p25=%d med=%d p75=%d max=%d (max/min=%.1fx)"
          % (v.min(), np.percentile(v, 25), np.median(v),
             np.percentile(v, 75), v.max(), v.max() / max(v.min(), 1)))

    # ── 2. 重投影误差 ─────────────────────────────────────────────────────
    errs = np.asarray(pts_err)
    print("[01d] reproj err(px): med=%.3f mean=%.3f p95=%.3f max=%.3f  (N=%d)"
          % (np.median(errs), errs.mean(), np.percentile(errs, 95),
             errs.max(), len(errs)))

    # ── 3. VIO vs COLMAP 轨迹对齐残差 ─────────────────────────────────────
    c_ref, c_vio = centers(ext_ref), centers(ext_vio)
    names = sorted(set(c_ref) & set(c_vio))
    if not names:
        print("[01d] ⚠️ 两侧无同名帧，跳过轨迹对比")
    else:
        A = np.array([c_vio[n][0] for n in names])
        B = np.array([c_ref[n][0] for n in names])
        s, R, t = umeyama(A, B)
        A_al = s * (R @ A.T).T + t
        pos_err = np.linalg.norm(A_al - B, axis=1)
        # 旋转残差：R_vio 对齐到世界系后与 R_ref 的夹角
        rot_deg = []
        for i, n in enumerate(names):
            Rv = c_vio[n][1] @ R.T        # 对齐：世界系 p'=sR p+t ⇒ R_w2c' = R_w2c R⁻¹
            Rr = c_ref[n][1]
            dR = Rr @ Rv.T
            c = np.clip((np.trace(dR) - 1) / 2, -1, 1)
            rot_deg.append(np.degrees(np.arccos(c)))
        rot_deg = np.array(rot_deg)
        extent = np.linalg.norm(B.max(0) - B.min(0))
        print("[01d] VIO→COLMAP 对齐: scale=%.4f, 轨迹包围盒对角=%.2f（COLMAP 单位）"
              % (s, extent))
        print("[01d] 位置残差(m): med=%.3f mean=%.3f max=%.3f"
              % (np.median(pos_err), pos_err.mean(), pos_err.max()))
        print("[01d] 旋转残差(°): med=%.3f mean=%.3f max=%.3f"
              % (np.median(rot_deg), rot_deg.mean(), rot_deg.max()))
        # 轨迹平滑度对比：相邻帧步长变异系数
        for tag, cc in (("VIO", A), ("COLMAP", B)):
            step = np.linalg.norm(np.diff(cc, axis=0), axis=1)
            print("[01d] %s 帧间步长: mean=%.4f std/mean=%.2f"
                  % (tag, step.mean(), step.std() / max(step.mean(), 1e-12)))

    # ── 4. 叠加图（精化点云投影，带三角化颜色）─────────────────────────────
    from PIL import Image
    P = np.asarray(pts_xyz)
    C = np.asarray(pts_rgb)
    cam = list(intr_ref.values())[0]
    K = np.array([[cam.params[0], 0, cam.params[2]],
                  [0, cam.params[1], cam.params[3]], [0, 0, 1]])
    W, H = int(cam.width), int(cam.height)
    ids = sorted(ext_ref.keys())
    pick = [ids[int(i * (len(ids) - 1) / max(1, n_check - 1))] for i in range(n_check)]
    sub = np.random.RandomState(0).choice(len(P), min(60000, len(P)), replace=False)
    Ps, Cs = P[sub], C[sub]
    print("[01d] overlay %d frames -> %s" % (len(pick), out_dir))
    for img_id in pick:
        e = ext_ref[img_id]
        R = qvec2rotmat(e.qvec)
        Pc = (R @ Ps.T).T + e.tvec
        z = Pc[:, 2]
        ok = z > 0.01
        uv = np.empty((len(Ps), 2))
        uv[:, 0] = K[0, 0] * Pc[:, 0] / np.maximum(z, 1e-9) + K[0, 2]
        uv[:, 1] = K[1, 1] * Pc[:, 1] / np.maximum(z, 1e-9) + K[1, 2]
        m = ok & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        img = np.array(Image.open(os.path.join(
            ref_dir, "images", e.name)).convert("RGB"))
        if img.shape[1] != W or img.shape[0] != H:
            img = np.array(Image.fromarray(img).resize((W, H)))
        xs = uv[m, 0].astype(int)
        ys = uv[m, 1].astype(int)
        img[ys, xs] = (0.35 * img[ys, xs] + 0.65 * Cs[m]).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(out_dir, "overlay_" + e.name))
        print("  %s: in-image %.1f%%" % (e.name, 100 * m.mean()))


if __name__ == "__main__":
    main()
