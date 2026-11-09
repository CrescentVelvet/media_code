#!/usr/bin/env python3
"""InfiniSplat v2「静图正常 / 视频重影」的定量归因。

背景
----
`eval/render_000..031.png` 是**输入视角**的重渲染；视频走的是**插值视角**。
前馈方法在输入视角天然对齐（高斯就是从那个像素预测出来的），
重影只在视点离开输入视角时暴露 —— 所以「图像没问题」不构成反证。

四个独立判据（分别对应一种机制）
--------------------------------
G1 局部 PCA 法向厚度  t = sqrt(λ3)/(尺度几何均值)
   同一表面存了多层薄片 → 邻居散在自己尺度之外，t 抬升。对照组=DA3-GS。
G4 每像素投影冗余度   输入视角 0 下，把全部高斯投影计数：
   每像素平均叠了多少颗。前馈未融合 → 每像素 ≥1 颗、大片区域 2~4 颗。
G2 半照度实验         全部 opacity ×0.5 后重渲：
   单层闭合 → 几乎不变；靠多层叠加 → 显著变淡。
   分别在「输入视角」和「插值视角」各做一次 —— 差异只出现在后者才叫机制解释。
G3 深度双峰           插值视角的期望深度图，出现同物体的两个深度台阶。

环境：/home/velvet/miniconda3/envs/querysplat/bin/python
环境变量：N_SUB（PCA 采样数，默认 120 万）
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

OUT = os.environ.get("OUT", m24.A + "/orbit_video/_ghost")
N_SUB = int(os.environ.get("N_SUB", "1200000"))
KNN = 16
CELLW = 372
CELLH = 496


def log(*a):
    print(*a, flush=True)


def same(tag):
    return [x for x in m24.CASES if x["name"] == tag][0]


def load(tag, Cref):
    c = same(tag)
    g = m24.read_ply(c["ply"])
    Cx, dx, upx = m24.read_cams(*c["cams"])
    if tag == "DA3-GS":
        s, R, t = 1.0, np.eye(3), np.zeros(3)
    else:
        s, R, t = m24.umeyama(Cx, Cref)
    ang = float(np.median(np.degrees(np.arccos(np.clip(
        np.sum((R @ dx.T).T * m24.read_cams(*same("DA3-GS")["cams"])[1], 1), -1, 1)))))
    g = m24.warp(g, s, R, t)
    log("  %-14s %8d 点 SH%d ｜ s=%.4f ｜ 对齐后视线差 %.2f°" % (tag, g["n"], g["deg"], s, ang))
    return g


def proj_redundancy(g, C, d, up):
    """把全部高斯投影到该相机，统计每像素叠了几颗（不做遮挡排序）。"""
    w2c = m24.look_at(C, C + d, up)
    Pm = w2c[:3, :3] @ g["means"].astype(np.float64).T + w2c[:3, 3:4]
    z = Pm[2]
    m = z > 1e-6
    Pm = Pm[:, m]
    # 用 DA3 视角 0 的像素内参
    K = K0_SCALED
    u = K[0, 0] * Pm[0] / Pm[2] + K[0, 2]
    v = K[1, 1] * Pm[1] / Pm[2] + K[1, 2]
    inb = (u >= 0) & (u < CELLW) & (v >= 0) & (v < CELLH)
    ui = np.clip(u[inb].astype(np.int32), 0, CELLW - 1)
    vi = np.clip(v[inb].astype(np.int32), 0, CELLH - 1)
    cnt = np.zeros(CELLW * CELLH, np.int64)
    np.add.at(cnt, vi * CELLW + ui, 1)
    hit = cnt > 0
    return dict(n_gauss=int(inb.sum()), cover=float(hit.mean()),
                per_px=float(cnt[hit].mean()) if hit.any() else 0.0,
                p90=float(np.percentile(cnt[hit], 90)) if hit.any() else 0.0,
                ge2=float(np.mean(cnt[hit] >= 2)) if hit.any() else 0.0,
                ge3=float(np.mean(cnt[hit] >= 3)) if hit.any() else 0.0)


def pca_thickness(g, rng):
    from scipy.spatial import cKDTree
    n = len(g["means"])
    idx = rng.choice(n, size=min(N_SUB, n), replace=False)
    tree = cKDTree(g["means"].astype(np.float64))
    _, nb = tree.query(g["means"][idx].astype(np.float64), k=KNN, workers=-1)
    Q = g["means"][nb].astype(np.float64)
    Q = Q - Q.mean(1, keepdims=True)
    cov = np.einsum("mki,mkj->mij", Q, Q) / KNN
    ev = np.linalg.eigvalsh(cov)
    s_gm = np.exp(np.log(g["scales"][idx]).mean(1))
    t = np.sqrt(np.clip(ev[:, 0], 0, None)) / np.maximum(s_gm, 1e-12)
    return t, ev[:, 0] / np.maximum(ev.sum(1), 1e-30)


def main():
    rng = np.random.default_rng(0)
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda"

    Cref, dref, upref = m24.read_cams(*same("DA3-GS")["cams"])
    orb = m24.fit_orbit(Cref, dref, upref)
    log("📷 基准 DA3-GS｜注视点 (%.3f, %.3f, %.3f)｜%d 相机"
        % (tuple(orb["target"]) + (len(Cref),)))

    # 插值视角：在观测弧内挑一对相邻相机，取正中（避开缺口边界，那里的相机是斜视的）
    ths = orb["th"]                                     # 已按方位角排序（fit_orbit 内 argsort）
    dif = np.diff(np.concatenate([ths, ths[:1] + 2 * np.pi]))
    k = int(np.argmax(dif))                             # 最大缺口的起点
    pick = (k + 1 + len(ths) // 3) % len(ths)           # 缺口之后走 1/3 圈，安全区
    a0 = float(ths[pick])
    a1 = float(ths[(pick + 1) % len(ths)]) + (2 * np.pi if pick + 1 >= len(ths) else 0.0)
    th_mid = 0.5 * (a0 + a1)
    mid_eye = m24.ring_interp(orb, np.array([th_mid]))[0]
    log("🎥 插值视角：相机 %.1f° / %.1f° 之间 → %.1f°（间隔 %.1f°）"
        % (np.degrees(a0), np.degrees(a1), np.degrees(th_mid), np.degrees(a1 - a0)))

    global K0_SCALED
    K0 = np.load(m24.A + "/cameras.npz")["intrinsics"].astype(np.float64)[0]
    K = K0.copy()
    K[0, 0] *= CELLW / 378.0
    K[1, 1] *= CELLH / 504.0
    K[0, 2], K[1, 2] = (CELLW - 1) / 2.0, (CELLH - 1) / 2.0
    K0_SCALED = K
    Kt = torch.from_numpy(K).float().to(dev)[None]

    views = {"V_in 输入视角 0": (Cref[0], dref[0], upref[0]),
             "V_mid 插值视角": (mid_eye, orb["target"] - mid_eye, orb["axis"])}

    stats = {}
    for tag in ("DA3-GS", "InfiniSplat v2"):
        log("\n" + "─" * 74)
        log("▶ %s" % tag)
        g = load(tag, Cref)
        op, sc = g["opac"], g["scales"]
        smax = sc.max(1)
        band = [100.0 * float(np.mean((op >= a) & (op < b)))
                for a, b in ((0.0, 0.1), (0.1, 0.7), (0.7, 0.95), (0.95, 1.01))]
        log("  T0 opacity   : 中位 %.3f ｜ 分档 <0.1 %.1f%% / 0.1~0.7 %.1f%% / 0.7~0.95 %.1f%% / ≥0.95 %.1f%%"
            % (np.median(op), band[0], band[1], band[2], band[3]))
        log("     尺度      : 中位 %.4f mm ｜ p99 %.4f mm"
            % (1e3 * np.median(smax), 1e3 * np.percentile(smax, 99)))

        red = proj_redundancy(g, *views["V_in 输入视角 0"])
        log("  G4 投影冗余  : 投出 %d 颗 ｜ 覆盖 %.1f%% 画面 ｜ 每像素 %.2f 颗（p90 %.0f）"
            % (red["n_gauss"], 100 * red["cover"], red["per_px"], red["p90"]))
        log("     叠≥2 颗 %.1f%% ｜ 叠≥3 颗 %.1f%% 的已覆盖像素"
            % (100 * red["ge2"], 100 * red["ge3"]))

        t, flat = pca_thickness(g, rng)
        log("  G1 局部厚度  : t 中位 %.3f ｜ p90 %.3f ｜ 平面性 λ3/Σλ 中位 %.4f"
            % (np.median(t), np.percentile(t, 90), np.median(flat)))

        G = {k2: torch.from_numpy(g[k2]).to(dev) for k2 in ("means", "quats", "scales", "opac", "sh")}
        bg = torch.ones(3, device=dev)[None]
        rgb, half = {}, {}
        for vn, (Ce, de, ue) in views.items():
            w2c = m24.look_at(Ce, Ce + de, ue)
            w2ct = torch.from_numpy(w2c).float().to(dev)[None]
            with torch.no_grad():
                i1, a1, _ = rasterization(G["means"], G["quats"], G["scales"], G["opac"], G["sh"],
                                          w2ct, Kt, CELLW, CELLH, sh_degree=g["deg"], packed=False,
                                          backgrounds=bg, render_mode="RGB+ED")
                h = (G["opac"] * 0.5).contiguous()
                i2, a2, _ = rasterization(G["means"], G["quats"], G["scales"], h, G["sh"],
                                          w2ct, Kt, CELLW, CELLH, sh_degree=g["deg"], packed=False,
                                          backgrounds=bg, render_mode="RGB+ED")
            d1 = i1[0, :, :3].clamp(0, 1).cpu().numpy()
            d2 = i2[0, :, :3].clamp(0, 1).cpu().numpy()
            mse = float(((d1 - d2) ** 2).mean())
            psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
            log("  G2 %-12s α %.4f→%.4f ｜ ×0.5 后差异 %.2f dB ｜ 深度 p50 %.3f m"
                % (vn, float(a1[0, :, :, 0].median()), float(a2[0, :, :, 0].median()), psnr,
                   float(i1[0, :, :, 3][i1[0, :, :, 3] > 0].median())))
            rgb[vn], half[vn] = d1, d2
            dep = i1[0, :, :, 3].cpu().numpy()
            np.save(os.path.join(OUT, "%s_%s_depth.npy"
                                 % (tag.replace(" ", "_").replace("+", "p"), vn.split()[0])), dep)
        stats[tag] = dict(red=red, t=t, band=band, rgb=rgb, half=half, dep=dep)

    # 并列图：上排原图 / 下排半照度，左=DA3-GS 右=InfiniSplat
    for vn in views:
        cols = []
        for tag in ("DA3-GS", "InfiniSplat v2"):
            cols.append(stats[tag]["rgb"][vn])
            cols.append(stats[tag]["half"][vn])
        sheet = Image.new("RGB", (CELLW * 4 + 30, CELLH), "white")
        for i, c in enumerate(cols):
            sheet.paste(Image.fromarray((c * 255).astype(np.uint8)), (i * (CELLW + 10), 0))
        fn = os.path.join(OUT, "half_exposure_%s.png" % vn.split()[0])
        sheet.save(fn)
        log("🖼  %s（左→右：DA3 原 / DA3 ×0.5 / Infini 原 / Infini ×0.5）" % fn)

    log("\n" + "=" * 74)
    log("%-14s %10s %10s %10s %10s %10s" % ("方法", "每像素颗", "叠≥2占比", "厚度p90", "半照度in", "半照度mid"))
    for tag, o in stats.items():
        log("%-14s %10.2f %9.1f%% %10.3f %9.2fdB %9.2fdB"
            % (tag, o["red"]["per_px"], 100 * o["red"]["ge2"], np.percentile(o["t"], 90),
               _p(o, "V_in 输入视角 0"), _p(o, "V_mid 插值视角")))


def _p(o, vn):
    a, b = o["rgb"][vn], o["half"][vn]
    mse = float(((a - b) ** 2).mean())
    return 10 * np.log10(1.0 / max(mse, 1e-12))


if __name__ == "__main__":
    main()
