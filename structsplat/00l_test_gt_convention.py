#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
structsplat/00l_test_gt_convention.py — Remy/Huawei 包 GT 相机约定的离线判据

为什么需要这个脚本（2026-11-20，GT 分支长期只有 6~7dB 的第三个根因）：
  修好 Sim(3) 对齐与宽高比之后，GT 分支仍然只有 7dB，而 pose-free 自洽有 17.5dB。
  诊断发现 GT 相机下有视角「正深度 0%」—— 高斯全跑到相机背后了。
  根因是 transform_matrix 的**相机系约定**搞错了：

    Remy/华为包给的是 **OpenGL 相机系** c2w（y 上、z 朝后）。
    转 OpenCV（y 下、z 朝前）必须在**相机系右乘** diag(1,-1,-1)：
        c2w_cv = c2w_gl @ diag(1,-1,-1,1)
    左乘 / 转置 / 不转 都是错的。

  判据不靠直觉，直接用 pcd.ply 的点投到图上：
      正深度率（点应在相机前方）+ 画面内率（应落在图像范围）。
  实测（8000 点，本仓 2026-11-20）：
      a) 原样当 CV（Rcᵀ）              正深度   0.0%  画面内  0.0%   ✗
      b) Rc @ diag(1,-1,-1)            正深度 100.0%  画面内 86.1%   ✅
      c) diag(1,-1,-1) @ Rc            正深度  19.6%  画面内  0.0%   ✗
      d) Rcᵀ 再取负 z                   正深度  13.0%  画面内  0.0%   ✗
      e) 把 Rc 直接当 w2c                正深度  87.1%  画面内  4.9%   ✗

这个脚本把它固化成自动化判据，防止以后「顺手改一下」又踩回去。

用法：
  python 00l_test_gt_convention.py --pkg /path/to/remy_pkg
"""

import argparse
import json
import os
import sys

import numpy as np


def load_ply_points(ply_path, n_sample=8000, seed=0):
    from plyfile import PlyData
    ply = PlyData.read(ply_path)
    v = ply["vertex"]
    pts = np.stack([np.asarray(v[k], dtype=np.float64) for k in ("x", "y", "z")], 1)
    if pts.shape[0] > n_sample:
        rng = np.random.default_rng(seed)
        pts = pts[rng.choice(pts.shape[0], n_sample, replace=False)]
    return pts


def project_stats(pts, C, R_cv, K, w, h):
    """R_cv: 3x3，把世界方向转到 CV 相机系。返回 (正深度率, 画面内率, 中位深度)。"""
    Xc = (pts - C) @ R_cv.T
    z = Xc[:, 2]
    pos = z > 1e-6
    if pos.sum() == 0:
        return 0.0, 0.0, float("nan")
    fx, fy, cx, cy = K
    u = fx * Xc[:, 0] / np.where(pos, z, 1.0) + cx
    v = fy * Xc[:, 1] / np.where(pos, z, 1.0) + cy
    inside = pos & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return float(pos.mean()), float(inside.mean()), float(np.median(z[pos]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True, help="Remy 采集包目录（含 transforms.json / pcd.ply）")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    pkg = args.pkg
    tf = os.path.join(pkg, "transforms.json")
    ply = os.path.join(pkg, "pcd.ply")
    for p in (tf, ply):
        if not os.path.exists(p):
            raise SystemExit(f"❌ 缺文件 {p}")

    meta = json.load(open(tf, "r", encoding="utf-8"))
    f0 = meta["frames"][0]
    w, h = int(f0["w"]), int(f0["h"])
    K = (f0["fl_x"], f0["fl_y"], f0["cx"], f0["cy"])
    M = np.array(f0["transform_matrix"], dtype=np.float64)   # c2w
    C, Rc = M[:3, 3], M[:3, :3]

    pts = load_ply_points(ply)
    print("=" * 70)
    print(" Remy 包 GT 相机约定判据")
    print("=" * 70)
    print(f"  图像 {w}×{h}   fx={K[0]:.1f} fy={K[1]:.1f} cx={K[2]:.1f} cy={K[3]:.1f}")
    print(f"  相机中心 {C}")
    print(f"  相机 +z(原样) {Rc[:,2]}")
    print(f"  点云 {pts.shape[0]} 点  质心 {pts.mean(0)}")

    FLIP = np.diag([1.0, -1.0, -1.0])
    cands = [
        ("a) 原样当 CV（Rcᵀ）",        Rc.T,                    False),
        ("b) Rc @ diag(1,-1,-1)",      (Rc @ FLIP).T,           True),
        ("c) diag(1,-1,-1) @ Rc",      (FLIP @ Rc).T,           False),
        ("d) Rcᵀ 再取负 z",            Rc.T @ np.diag([1., 1., -1.]), False),
        ("e) 把 Rc 直接当 w2c",         Rc,                      False),
    ]

    print("\n  候选变换（正深度率 / 画面内率 / 中位深度）")
    ok = []
    for name, R, expect in cands:
        posr, insr, medz = project_stats(pts, C, R, K, w, h)
        verdict = "✅" if (posr > 0.9 and insr > 0.7) else "✗"
        print(f"    {name:28s} 正深度 {posr*100:5.1f}%  "
              f"画面内 {insr*100:5.1f}%  中位z {medz:+7.3f}  {verdict}"
              + ("   ← 期望" if expect else ""))
        if verdict == "✅":
            ok.append(name)

    print()
    if len(ok) == 1:
        print(f"🎉 唯一正确约定：{ok[0]}")
        print("   → load_gt_cameras 必须做 c2w[:3,:3] @ diag(1,-1,-1)，"
              "且与竖图转置的 M 是**连乘**关系")
    elif not ok:
        print("❌ 没有一个候选通过 —— 该包可能不是 OpenGL c2w 约定，需重新判定")
        return 1
    else:
        print(f"⚠️  多个候选通过（{ok}），判据不够强，需加亮度一致性再验")
        return 1

    # 附加：确认 det 关系。GL→CV 的 FLIP 是 det=+1（两次轴反转），
    # 所以 c2w_cv 应仍是右手系 det=+1；而**转置的 M 是 det=-1**。
    det_cv = float(np.linalg.det(Rc @ FLIP))
    det_t = float(np.linalg.det(FLIP))
    det_m = float(np.linalg.det(np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 1.]])))
    print(f"\n  det(c2w_gl·FLIP) = {det_cv:+.3f}   det(FLIP)={det_t:+.3f}   det(M)={det_m:+.3f}")
    print("  → 竖图转置后 w2c 的 det 会是 -1，这是**正常**的（转置本身就是镜像像素），"
          "不要试图\"修正\"它")
    return 0


if __name__ == "__main__":
    sys.exit(main())
