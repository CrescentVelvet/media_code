#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
structsplat/00x_cam_geometry.py — 输入包的相机几何体检（判断 Sim(3) 是否可观测）

为什么需要（2026-11-20）：
  00w 的 target 评测用「82 张里均匀取 2 张」当 src，结果 11.26dB。可疑点：若这批图是
  绕物体拍一圈，linspace(0,81,2) 取到的**首尾两张很可能相邻**（同一位置附近），
  基线极短 → 尺度与旋转在数学上不可观测（残差会显得很小，但解族很大，是假象）。
  → 报数之前必须先量相机分布，确认选中的 src 是否真有基线。

输出：
  ① 逐帧相机中心（相机系半径归一化后）与视线方位角/仰角
  ② 目标点（视线交汇最小二乘）
  ③ 指定 src 子集的基线长度、夹角跨度、共线度
  ④ 全部帧两两基线矩阵的极值 → 选出「基线最大」的 src 组合

用法：
  python 00x_cam_geometry.py --transforms ".../transforms.json"
  python 00x_cam_geometry.py --transforms "..." --src-names a.heic,b.heic
"""
import argparse
import json
import os
import sys

import numpy as np

SD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SD)


def load_raw_c2w(transforms):
    """读 transforms.json 的原始 c2w（OpenGL 约定，未做任何转换），保持帧顺序。"""
    with open(transforms, "r", encoding="utf-8") as f:
        j = json.load(f)
    c2ws, names = [], []
    for fr in j["frames"]:
        c2ws.append(np.array(fr["transform_matrix"], dtype=np.float64))
        names.append(os.path.basename(fr.get("file_path", fr.get("image", ""))))
    return np.stack(c2ws), names


def look_at_target(c2ws):
    """视线交汇最小二乘：对每台相机 c 与视线 d，最小化 Σ|(I-ddᵀ)(X-c)|²。
    返回世界系目标点 X。"""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for M in c2ws:
        # OpenGL c2w：相机朝 -z
        z = M[:3, 2]
        d = -z / (np.linalg.norm(z) + 1e-12)
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ M[:3, 3]
    return np.linalg.solve(A, b)


def az_el(c2ws, target):
    """逐帧方位角/仰角（以 target 为原点，世界系 +z 为上）。"""
    out = []
    for M in c2ws:
        c = M[:3, 3]
        v = c - target
        r = np.linalg.norm(v)
        az = np.degrees(np.arctan2(v[1], v[0]))
        el = np.degrees(np.arcsin(np.clip(v[2] / (r + 1e-12), -1, 1)))
        out.append((r, az, el))
    return np.array(out)


def subset_stats(c2ws, idx, target):
    """给定子集：基线、方位角跨度、共线度。"""
    Cs = np.stack([c2ws[i][:3, 3] for i in idx])
    if len(idx) < 2:
        return {}
    # 两两基线
    D = np.linalg.norm(Cs[:, None, :] - Cs[None, :, :], axis=-1)
    iu = np.triu_indices(len(idx), 1)
    base = D[iu]
    # 相对目标的方位角跨度（处理环绕）
    v = Cs - target
    az = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
    az_sorted = np.sort(az)
    gaps = np.diff(np.concatenate([az_sorted, [az_sorted[0] + 360.0]]))
    span = 360.0 - gaps.max()          # 去掉最大空档后剩下的跨度
    # 共线度：中心化后 PCA，最小奇异值/最大奇异值
    Cc = Cs - Cs.mean(0)
    sv = np.linalg.svd(Cc, compute_uv=False)
    collinear = float(sv[-1] / (sv[0] + 1e-12)) if sv[0] > 1e-12 else 0.0
    # 尺度参照：相机到目标的平均半径
    rad = np.linalg.norm(v, axis=1).mean()
    return {
        "baseline_min": float(base.min()), "baseline_max": float(base.max()),
        "baseline_mean": float(base.mean()),
        "baseline_over_radius": float(base.max() / (rad + 1e-12)),
        "az_span_deg": float(span), "el_span_deg": float(np.ptp(np.degrees(
            np.arcsin(np.clip(v[:, 2] / (np.linalg.norm(v, axis=1) + 1e-12), -1, 1))))),
        "collinear_ratio": collinear,
        "radius_mean": float(rad),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transforms", required=True)
    ap.add_argument("--src-names", default=None,
                    help="逗号分隔的 src 文件名，检查这个子集的基线")
    ap.add_argument("--top-pairs", type=int, default=6,
                    help="打印基线最大的 N 个组合（恰 2 台相机）")
    args = ap.parse_args()

    c2ws, names = load_raw_c2w(args.transforms)
    N = len(c2ws)
    print(f"📦 {N} 帧  {os.path.basename(os.path.dirname(args.transforms))}")

    Cs = c2ws[:, :3, 3]
    # 输入包原点不一定是物体中心，先看原始中心散布
    origin_r = np.linalg.norm(Cs, axis=1)
    print(f"  相机系原点半径（相对世界原点）: min {origin_r.min():.3f}  "
          f"med {np.median(origin_r):.3f}  max {origin_r.max():.3f}  "
          f"比值 {origin_r.max()/(origin_r.min()+1e-12):.2f}x")

    tgt = look_at_target(c2ws)
    print(f"🎯 视线交汇目标点（世界系）: [{tgt[0]:+.3f} {tgt[1]:+.3f} {tgt[2]:+.3f}]")

    R = az_el(c2ws, tgt)
    print(f"  距目标半径: min {R[:,0].min():.4f}  med {np.median(R[:,0]):.4f}  "
          f"max {R[:,0].max():.4f}  比值 {R[:,0].max()/(R[:,0].min()+1e-12):.2f}x")
    print(f"  方位角 az: {R[:,1].min():+.1f}° … {R[:,1].max():+.1f}°")
    print(f"  仰角 el:   {R[:,2].min():+.1f}° … {R[:,2].max():+.1f}°")

    # 逐帧打印（首尾 + 抽样）
    show = sorted(set([0, 1, N - 2, N - 1] + list(np.linspace(0, N - 1, 10).astype(int))))
    print(f"\n{'idx':>4s} {'name':<26s} {'r':>7s} {'az':>8s} {'el':>7s}")
    for i in show:
        print(f"{i:>4d} {names[i][:26]:<26s} {R[i,0]:>7.4f} {R[i,1]:>8.2f} {R[i,2]:>7.2f}")

    # 均匀取 2 张（linspace 口径）
    idx2 = np.linspace(0, N - 1, 2).round().astype(int).tolist()
    print(f"\n── linspace(0,{N-1},2) 选中的 src = {idx2} "
          f"{[names[i] for i in idx2]} ──")
    st = subset_stats(c2ws, idx2, tgt)
    for k, v in st.items():
        print(f"    {k:<22s} {v:.5f}")
    d01 = np.linalg.norm(Cs[idx2[0]] - Cs[idx2[1]])
    azgap = abs(((R[idx2[0],1] - R[idx2[1],1] + 180) % 360) - 180)
    print(f"    → 基线 {d01:.5f} = 半径的 {d01/(st['radius_mean']+1e-12)*100:.1f}%  "
          f"方位角差 {azgap:.1f}°")
    if d01 / (st['radius_mean'] + 1e-12) < 0.25 or azgap < 20:
        print("    ⚠️ 基线过短 / 视角过近 → 尺度与旋转基本不可观测（Sim(3) 退化）")

    # 全库最短/最长基线
    D = np.linalg.norm(Cs[:, None, :] - Cs[None, :, :], axis=-1)
    np.fill_diagonal(D, np.inf)
    fmin = np.unravel_index(np.argmin(D), D.shape)
    fmax = np.unravel_index(np.argmax(D), D.shape)
    print(f"\n全库最短基线: {names[fmin[0]]} ↔ {names[fmin[1]]}  {D[fmin]:.5f}")
    print(f"全库最长基线: {names[fmax[0]]} ↔ {names[fmax[1]]}  {D[fmax]:.5f}")

    if args.src_names:
        want = [x.strip() for x in args.src_names.split(",") if x.strip()]
        idx = [names.index(w) for w in want if w in names]
        miss = [w for w in want if w not in names]
        if miss:
            print(f"⚠️ 未找到: {miss}")
        print(f"\n── 指定 src {[names[i] for i in idx]} ──")
        for k, v in subset_stats(c2ws, idx, tgt).items():
            print(f"    {k:<22s} {v:.5f}")


if __name__ == "__main__":
    main()
