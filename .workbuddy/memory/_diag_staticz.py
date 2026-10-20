#!/usr/bin/env python3
"""诊断写字/弹琴样例：手部运动类型（整体位移 vs 局部形变）+ 相机视角覆盖。

输出：
- 相机方位角/仰角覆盖（决定"静态化"能拿到多少新视角）
- 逐帧运动能量与运动块质心轨迹（判断是持续位移还是有静止段）
- 运动区 vs 静止区的锐度（判断输入是否糊）
"""
import os
import sys
import glob
import json

import numpy as np
import cv2

SCENES = {
    "hand06_xiezi": "/home/velvet/output/som_human_results/hand06_nvsheng_xiezi/colmap_refined",
    "hand08_tanqin": "/home/velvet/output/som_human_results/hand08_nvsheng_tanqin/colmap_refined",
}
OUT = "/mnt/d/output/som_human_hand/diag"
os.makedirs(OUT, exist_ok=True)


def read_poses_txt(path):
    """解析 COLMAP images.txt -> {name: (qvec, tvec)}。只处理两行一图的完整格式。"""
    poses = {}
    with open(path, "r") as f:
        lines = [l for l in f if not l.startswith("#")]
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) < 9:
            i += 1
            continue
        try:
            qw, qx, qy, qz = [float(x) for x in parts[1:5]]
            tx, ty, tz = [float(x) for x in parts[5:8]]
            name = parts[9]
        except (ValueError, IndexError):
            i += 1
            continue
        poses[name] = (np.array([qw, qx, qy, qz]), np.array([tx, ty, tz]))
        i += 2
    return poses


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ])


def cam_centers(poses, order):
    cs = []
    for n in order:
        if n not in poses:
            continue
        q, t = poses[n]
        R = qvec2rotmat(q)
        cs.append(-R.T @ t)
    return np.array(cs)


def coverage(centers):
    """相对质心的方位角/仰角覆盖；同时给出绕行角度跨度。"""
    c = centers.mean(0)
    v = centers - c
    # 用主平面（丢弃方差最小轴）算方位角
    u, s, vt = np.linalg.svd(v, full_matrices=False)
    proj = v @ vt[:2].T          # [N,2]
    az = np.degrees(np.arctan2(proj[:, 1], proj[:, 0]))
    el = np.degrees(np.arctan2(v[:, 2], np.linalg.norm(proj, axis=1)))
    # 方位角跨度：排序后取最大间隙，补集即覆盖
    a = np.sort(np.mod(az, 360))
    if len(a) < 2:
        span = 0.0
    else:
        gaps = np.diff(np.concatenate([a, [a[0] + 360]]))
        span = 360.0 - gaps.max()
    return dict(az_min=float(np.min(az)), az_max=float(np.max(az)),
                az_span=float(span),
                el_min=float(np.min(el)), el_max=float(np.max(el)),
                radius=float(np.median(np.linalg.norm(v, axis=1))))


def motion_analysis(img_paths, max_frames=60):
    """相邻帧光流：网格运动能量、主运动块质心轨迹、运动/静止区锐度。"""
    n = len(img_paths)
    idx = np.linspace(0, n - 1, min(max_frames, n)).astype(int)
    idx = np.unique(idx)
    prev_gray = None
    rows = []
    for i in idx:
        im = cv2.imread(img_paths[i], cv2.IMREAD_COLOR)
        if im is None:
            continue
        h, w = im.shape[:2]
        sc = 720.0 / max(h, w)
        if sc < 1.0:
            im = cv2.resize(im, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        if prev_gray is None or prev_gray.shape != gray.shape:
            prev_gray = gray
            continue
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            0.5, 4, 15, 3, 5, 1.2, 0)
        gh, gw = gray.shape
        gx, gy = 6, 8
        cells = []
        for r in range(gy):
            for c in range(gx):
                y0, y1 = r * gh // gy, (r + 1) * gh // gy
                x0, x1 = c * gw // gx, (c + 1) * gw // gx
                sub = flow[y0:y1, x0:x1]
                mag = np.linalg.norm(sub, axis=2)
                cells.append((float(mag.mean()), r, c))
        cells.sort(reverse=True)
        top = cells[:3]
        # 主运动块质心（像素坐标，归一化）
        ys, xs = [], []
        for m, r, c in top:
            ys.append((r + 0.5) / gy)
            xs.append((c + 0.5) / gx)
        mag_all = np.linalg.norm(flow, axis=2)
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        # 运动块 vs 静止块锐度
        r0 = top[0][1]
        c0 = top[0][2]
        y0, y1 = r0 * gh // gy, (r0 + 1) * gh // gy
        x0, x1 = c0 * gw // gx, (c0 + 1) * gw // gx
        sharp_mov = float(np.log10(np.var(lap[y0:y1, x0:x1]) + 1e-6))
        # 静止：取最冷方块
        rq, cq = cells[-1][1], cells[-1][2]
        y0b, y1b = rq * gh // gy, (rq + 1) * gh // gy
        x0b, x1b = cq * gw // gx, (cq + 1) * gw // gx
        sharp_sta = float(np.log10(np.var(lap[y0b:y1b, x0b:x1b]) + 1e-6))
        rows.append(dict(frame=int(i), p75=float(np.percentile(mag_all, 75)),
                         mean=float(mag_all.mean()),
                         cx=float(np.mean(xs)), cy=float(np.mean(ys)),
                         sharp_mov=sharp_mov, sharp_sta=sharp_sta))
        prev_gray = gray
    return rows


def summarize(name, rows):
    if not rows:
        print(name, "无数据")
        return
    p75 = np.array([r["p75"] for r in rows])
    cx = np.array([r["cx"] for r in rows])
    cy = np.array([r["cy"] for r in rows])
    sm = np.array([r["sharp_mov"] for r in rows])
    ss = np.array([r["sharp_sta"] for r in rows])
    # 质心位移：相邻帧增量 + 全序列跨度
    jit_x = np.abs(np.diff(cx)).mean()
    drift = max(cx.max() - cx.min(), cy.max() - cy.min())
    print("  flow_p75  中位 {:.2f}px  p90 {:.2f}".format(np.median(p75), np.percentile(p75, 90)))
    print("  运动块质心 x/y 抖动(逐帧) {:.4f}/{:.4f} (归一化)".format(jit_x, np.abs(np.diff(cy)).mean()))
    print("  运动块质心 跨度 max {:.4f}".format(drift))
    print("  运动区锐度 {:.2f}  静止区锐度 {:.2f}  Δ={:+.2f}".format(np.median(sm), np.median(ss), np.median(sm) - np.median(ss)))
    # 运动能量的时序分布：前中后三段
    k = max(1, len(p75) // 3)
    print("  p75 分段 前{:.1f} 中{:.1f} 后{:.1f}".format(
        np.median(p75[:k]), np.median(p75[k:2 * k]), np.median(p75[2 * k:])))
    # 静止帧占比（低于全局中位 30%）
    thr = np.median(p75) * 0.5
    print("  低运动帧占比 {:.0%} (p75<{:.2f})".format((p75 < thr).mean(), thr))


def main():
    for name, root in SCENES.items():
        print("=" * 60)
        print(name, root)
        txt = os.path.join(root, "sparse", "0", "images.txt")
        imgs = sorted(glob.glob(os.path.join(root, "images", "*")))
        print("  图像数", len(imgs))
        poses = read_poses_txt(txt) if os.path.exists(txt) else {}
        print("  位姿数", len(poses))
        if poses:
            order = [os.path.basename(p) for p in imgs]
            order = [n for n in order if n in poses]
            cs = cam_centers(poses, order)
            cov = coverage(cs)
            print("  视角: az[{:.1f},{:.1f}] 跨度 {:.1f}° | el[{:.1f},{:.1f}] | 半径 {:.2f}m".format(
                cov["az_min"], cov["az_max"], cov["az_span"],
                cov["el_min"], cov["el_max"], cov["radius"]))
        rows = motion_analysis(imgs)
        summarize(name, rows)
        # 出图：运动能量时序 + 质心轨迹
        if rows:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 3, figsize=(15, 3.2), dpi=130)
            fr = [r["frame"] for r in rows]
            ax[0].plot(fr, [r["p75"] for r in rows], lw=0.9)
            ax[0].set_title("flow p75 (px)")
            ax[1].plot(fr, [r["cx"] for r in rows], lw=0.9, label="cx")
            ax[1].plot(fr, [r["cy"] for r in rows], lw=0.9, label="cy")
            ax[1].legend(fontsize=7)
            ax[1].set_title("motion blob centroid")
            ax[2].plot(fr, [r["sharp_mov"] for r in rows], lw=0.9, label="moving")
            ax[2].plot(fr, [r["sharp_sta"] for r in rows], lw=0.9, label="static")
            ax[2].legend(fontsize=7)
            ax[2].set_title("sharpness log10 LapVar")
            for a in ax:
                a.set_xlabel("frame")
            fig.tight_layout()
            fig.savefig(os.path.join(OUT, "motion_{}.png".format(name)))
            plt.close(fig)


if __name__ == "__main__":
    main()
