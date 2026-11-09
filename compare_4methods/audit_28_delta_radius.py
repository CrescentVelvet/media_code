#!/usr/bin/env python3
"""为什么 DA3-GS 不重影、InfiniSplat v2 重影：同一误差、不同尺度。

已成立的台账（infinisplat_v2/EXPERIMENTS.md）
--------------------------------------------
E-19 双峰率 24.8%（底噪 0.5%、人工鬼影 73%）→ 轻度边缘分裂
E-20 体素合并倍数 1.4→0.5 全指标落在噪声内 → 不是合并造成
E-21 重影随观测重叠度下降、32 视角饱和；自洽渲染（位姿同源）也有 → **不是位姿，是深度噪声**

本脚本补最后一环
----------------
两个方法同源（InfiniSplat `backend: da3`，即同一个 DA3 前向 + 体素合并），
跨视角确定性误差量级理应相近。差别在**高斯尺度**：
  DA3-GS      尺度中位 11.6 mm，在 1.8 m / f=521 下 ≈ 3.2 px 半径
  InfiniSplat 尺度中位  5.2 mm，≈ 1.45 px 半径
若表面位置不确定 δ ≈ 1~2 px：
  δ < r  → 不一致被高斯自身的空间延展吸收，视觉上是「糊」
  δ > r  → 无法吸收，两条平行边缘各自成像，视觉上是「重影」
换句话说，糊和重影是**同一个误差除以不同高斯半径**的两种表现。

测量
----
D1 跨视角表面位置差 δ：每视角先用 **z-buffer 最小值**取最近表面深度（不吃期望深度平均），
   反投影到 3D 再投到相邻视角，与对方 z-buffer 比较，把深度差换算成该处的切向像素位移。
D2 有效高斯半径 r（像素）：median(scale_max) / z_med × f。
D3 δ/r 与残差热图：热区若落在头部，就同时解释了「只有脸糊」。
"""
import importlib.util
import os

import numpy as np
import torch
from PIL import Image

SPEC = importlib.util.spec_from_file_location(
    "m24", "/mnt/c/code/media_code/compare_4methods/audit_24_five_way_video.py")
m24 = importlib.util.module_from_spec(SPEC)
os.environ["NFRAMES"] = "1"
SPEC.loader.exec_module(m24)
from gsplat import rasterization            # noqa: E402

CELLW, CELLH = 372, 496
OUT = os.environ.get("OUT", m24.A + "/orbit_video/_ghost")
STEP = 3.0            # 视角对步长（32 视角 / 3 ≈ 11 对，避免全量 31 对太慢）
DEPTH_TOL = 0.02      # 深度差超过 2% 视为「不在同一表面」，剔除（遮挡边界）


def log(*a):
    print(*a, flush=True)


def zbuffer(g, w2c, K):
    """每像素最近高斯的 z（不吃 alpha 期望深度）；无点=0。"""
    P = g["means"].astype(np.float64)
    R, t = w2c[:3, :3], w2c[:3, 3]
    Pc = P @ R.T + t
    z = Pc[:, 2]
    m = z > 1e-3
    Pc, z = Pc[m], z[m]
    u = K[0, 0] * Pc[:, 0] / z + K[0, 2]
    v = K[1, 1] * Pc[:, 1] / z + K[1, 2]
    inb = (u >= 0) & (u < CELLW) & (v >= 0) & (v < CELLH)
    ui = u[inb].astype(np.int32)
    vi = v[inb].astype(np.int32)
    zz = z[inb]
    key = vi.astype(np.int64) * CELLW + ui
    order = np.lexsort((zz, key))
    key, zz = key[order], zz[order]
    new_seg = np.concatenate([[True], key[1:] != key[:-1]])
    seg_start = np.flatnonzero(new_seg)
    zmin = np.minimum.reduceat(zz, seg_start)
    out = np.zeros(CELLW * CELLH)
    out[key[seg_start]] = zmin
    return out.reshape(CELLH, CELLW)


def main():
    os.makedirs(OUT, exist_ok=True)
    C, d, up = m24.read_cams(*m24.CASES[0]["cams"])
    orb = m24.fit_orbit(C, d, up)
    K0 = np.load(m24.A + "/cameras.npz")["intrinsics"].astype(np.float64)[0]
    K = K0.copy()
    K[0, 0] *= CELLW / 378.0
    K[1, 1] *= CELLH / 504.0
    K[0, 2], K[1, 2] = (CELLW - 1) / 2.0, (CELLH - 1) / 2.0

    # 视 0 的头部粗框（372×496，由放大图目视定位；只用于分区统计，不参与判据）
    HEAD = (slice(int(0.13 * CELLH), int(0.34 * CELLH)),
            slice(int(0.38 * CELLW), int(0.60 * CELLW)))
    log("头部粗框 行 %d~%d 列 %d~%d" % (HEAD[0].start, HEAD[0].stop,
                                      HEAD[1].start, HEAD[1].stop))

    rows = []
    for tag in ("DA3-GS", "InfiniSplat v2"):
        c = [x for x in m24.CASES if x["name"] == tag][0]
        g = m24.read_ply(c["ply"])
        Cx, _, _ = m24.read_cams(*c["cams"])
        s, R, t = (1.0, np.eye(3), np.zeros(3)) if tag == "DA3-GS" else m24.umeyama(Cx, C)
        g = m24.warp(g, s, R, t)
        log("\n" + "─" * 70)
        log("▶ %s  %d 点 SH%d" % (tag, g["n"], g["deg"]))

        zb = [zbuffer(g, m24.look_at(C[i], C[i] + d[i], up[i]), K) for i in range(len(C))]

        # D2 有效像素半径
        zmed = float(np.median(zb[0][zb[0] > 0]))
        r_m = float(np.median(g["scales"].max(1)))
        r_px = r_m / zmed * float(K[0, 0])
        log("  D2 高斯尺度 中位 %.2f mm ｜ 深度中位 %.2f m ｜ 半径 ≈ %.2f px"
            % (1e3 * r_m, zmed, r_px))

        # D1 跨视角表面位置差
        vv, uu = np.mgrid[0:CELLH:2, 0:CELLW:2]
        res_px, acc = [], np.zeros((CELLH, CELLW))
        acc_n = np.zeros((CELLH, CELLW))
        pairs = list(range(0, len(C) - 1, int(STEP)))
        for i in pairs:
            j = i + 1
            wi, wj = m24.look_at(C[i], C[i] + d[i], up[i]), m24.look_at(C[j], C[j] + d[j], up[j])
            z = zb[i][vv, uu]
            ok = z > 0.05
            vv_i, uu_i, z_ok = vv[ok], uu[ok], z[ok]      # 三者同序，后面反查用它
            xc = (uu_i - K[0, 2]) / K[0, 0] * z_ok
            yc = (vv_i - K[1, 2]) / K[1, 1] * z_ok
            Pw = np.stack([xc, yc, z_ok], 1) @ wi[:3, :3] + (-wi[:3, :3].T @ wi[:3, 3])
            Pj = Pw @ wj[:3, :3].T + wj[:3, 3]
            zj = Pj[:, 2]
            uj = K[0, 0] * Pj[:, 0] / np.maximum(zj, 1e-9) + K[0, 2]
            vj = K[1, 1] * Pj[:, 1] / np.maximum(zj, 1e-9) + K[1, 2]
            inm = (zj > 0.05) & (uj >= 0) & (uj < CELLW - 1) & (vj >= 0) & (vj < CELLH - 1)
            idx = np.flatnonzero(inm)
            zg = zb[j][vj[idx].astype(np.int32), uj[idx].astype(np.int32)]
            good = zg > 0.05
            rel = np.abs(zj[idx][good] - zg[good]) / np.maximum(zg[good], 1e-6)
            keep = rel < DEPTH_TOL                       # 剔遮挡边界（跨表面误配）
            sel = idx[np.flatnonzero(good)[keep]]
            dz_m = np.abs(zj[sel] - zb[j][vj[sel].astype(np.int32), uj[sel].astype(np.int32)])
            px = dz_m * K[1, 1] / np.maximum(zb[j][vj[sel].astype(np.int32),
                                                  uj[sel].astype(np.int32)], 1e-6)
            res_px.append(px)
            if i == 0:                                   # 热图只用同一视角的坐标，别跨视角累加
                acc[vv_i[sel], uu_i[sel]] += dz_m
                acc_n[vv_i[sel], uu_i[sel]] += 1

        px_all = np.concatenate(res_px)
        log("  D1 跨视角表面位置差：中位 %.2f px ｜ p90 %.2f px ｜ p99 %.2f px（%d 对视角，%d 采样）"
            % (np.median(px_all), np.percentile(px_all, 90), np.percentile(px_all, 99),
               len(pairs), len(px_all)))
        hz = acc[HEAD] / np.maximum(acc_n[HEAD], 1)
        hz = hz[acc_n[HEAD] > 0] * K[1, 1] / zmed
        log("     头部区（视 0）中位 %.2f px ｜ 覆盖 %.1f%% 的头框像素 ｜ 全图 δ/r = %.2f"
            % (np.median(hz), 100 * float(np.mean(acc_n[HEAD] > 0)),
               np.median(px_all) / r_px))
        ha = acc[acc_n > 0] / acc_n[acc_n > 0] * K[1, 1] / zmed
        log("     全图（视 0）中位 %.2f px ｜ 头部/全图 = %.2f"
            % (np.median(ha), np.median(hz) / max(np.median(ha), 1e-6)))

        hm = acc / np.maximum(acc_n, 1)
        hm = np.clip(hm / 0.05, 0, 1)                        # 5 cm 封顶
        Image.fromarray((255 * (1 - hm)).astype(np.uint8)).save(
            os.path.join(OUT, "resid_%s.png" % tag.replace(" ", "_").replace("+", "p")))
        np.save(os.path.join(OUT, "resid_%s.npy" % tag.replace(" ", "_")), acc / np.maximum(acc_n, 1))

        rows.append((tag, r_px, float(np.median(px_all)), float(np.percentile(px_all, 90)),
                     float(np.median(px_all) / r_px)))
        del g
        torch.cuda.empty_cache()

    log("\n" + "=" * 78)
    log("%-15s %10s %12s %12s %10s" % ("方法", "半径r(px)", "位置差δ(px)", "δ p90", "δ/r"))
    for t, r, m1, m2, ratio in rows:
        log("%-15s %10.2f %12.2f %12.2f %10.2f" % (t, r, m1, m2, ratio))
    log("\n读法：δ 是两个方法共有的深度噪声量级；r 是各自的「吸收半径」。")
    log("      δ/r < 1 → 不一致融进高斯自身，看起来是糊；δ/r ≳ 1 → 分得开，看起来是重影。")


if __name__ == "__main__":
    main()
