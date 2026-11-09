#!/usr/bin/env python3
"""InfiniSplat v2 重影的**层结构**测量（G5 幽灵能见度 + G6 跨视角深度一致性）。

现象定位
--------
放大抽帧后，重影**只在头/脸/头发**：柜子、地板、牛仔裤都干净。这排除了
「渲染参数」「尺度错了」「对齐错了」这类全局解释 —— 那些会让整幅一起糊。

机制假设
--------
每帧/每 chunk 各自出一套高斯；静止部分各帧落在同一位置，叠加后仍是单层；
头部有帧间微动 + 深度误差，各帧在 3D 里错开 1~3 cm。输入视角靠遮挡盖住，
视点一离开输入视角，第二层就从缝隙里透出来 → 双轮廓。

G5 幽灵能见度（主判据）
----------------------
对插值视角的每个像素，把落在它上面的高斯按深度排序，累积透射率
T_k = Π_{j<k}(1-α_j)，该高斯对颜色的贡献权重 w_k = α_k·T_k。
每个像素因此得到一条「权重-深度」曲线：
  · 单层闭合 → 权重集中在主峰附近，主峰之外几乎为 0
  · 双层透出 → 主峰之外还有一个深度上明显分离、权重却不可忽略的次峰
ghost = Σ{w_k : |z_k − z_peak| > 3 cm} / Σ w_k
  ghost ≈ 0 单层；> 0.15 就是肉眼可见的第二层。

G6 跨视角深度一致性
-------------------
用同一组 32 个输入位姿渲深度图，把视角 i 的深度反投影到 3D 再投到视角 i+1，
比较深度残差。这是「为什么只有它重影」的直接回答：DA3 一次前向全局注意力，
跨视角严格一致；流式/chunk 方法逐段预测，段间对齐了相机、没对齐表面。
（渲染深度是期望深度，双层会被平均 → 残差是低估，只用于比好坏。）

环境：/home/velvet/miniconda3/envs/querysplat/bin/python
"""
import importlib.util
import os

import numpy as np
import torch

SPEC = importlib.util.spec_from_file_location(
    "m24", "/mnt/c/code/media_code/compare_4methods/audit_24_five_way_video.py")
m24 = importlib.util.module_from_spec(SPEC)
os.environ["NFRAMES"] = "1"
SPEC.loader.exec_module(m24)
from gsplat import rasterization            # noqa: E402

CELLW, CELLH = 372, 496
GHOST_GAP = 0.03            # 距主峰 3 cm 以外算「另一层」
OUT = os.environ.get("OUT", m24.A + "/orbit_video/_ghost")


def log(*a):
    print(*a, flush=True)


def load_aligned(tag, Cref):
    c = [x for x in m24.CASES if x["name"] == tag][0]
    g = m24.read_ply(c["ply"])
    Cx, _, _ = m24.read_cams(*c["cams"])
    s, R, t = (1.0, np.eye(3), np.zeros(3)) if tag == "DA3-GS" else m24.umeyama(Cx, Cref)
    return m24.warp(g, s, R, t), s


def ghost_metric(g, w2c, K):
    P = g["means"].astype(np.float64)
    R, t = w2c[:3, :3], w2c[:3, 3]
    Pc = P @ R.T + t
    z = Pc[:, 2]
    m = z > 1e-3
    Pc, z = Pc[m], z[m]
    al = g["opac"][m].astype(np.float64)
    u = K[0, 0] * Pc[:, 0] / z + K[0, 2]
    v = K[1, 1] * Pc[:, 1] / z + K[1, 2]
    inb = (u >= 0) & (u < CELLW) & (v >= 0) & (v < CELLH)
    ui = np.clip(u[inb], 0, CELLW - 1e-3).astype(np.int32)
    vi = np.clip(v[inb], 0, CELLH - 1e-3).astype(np.int32)
    zz, aa = z[inb], al[inb]

    key = vi.astype(np.int64) * CELLW + ui
    order = np.lexsort((zz, key))
    key, zz, aa = key[order], zz[order], aa[order]

    # 段 = 同一像素；用 log 前缀和做段内累积，段首重置
    lg = np.log(np.clip(1.0 - aa, 1e-12, 1.0))
    cs = np.concatenate([[0.0], np.cumsum(lg)])            # cs[k] = Σ_{<k} lg
    new_seg = np.concatenate([[True], key[1:] != key[:-1]])
    seg_start = np.flatnonzero(new_seg)
    seg_id = np.cumsum(new_seg) - 1
    seg_base = cs[seg_start][seg_id]
    w = aa * np.exp(cs[:len(lg)] - seg_base)               # 该高斯对颜色的权重

    seg_end = np.concatenate([seg_start[1:], [len(w)]])
    seglen = seg_end - seg_start
    seg_w = np.add.reduceat(w, seg_start)
    wmax = np.maximum.reduceat(w, seg_start)
    # 每段内 w == wmax 的那颗的深度 → 主峰深度
    zpk = np.where(w >= np.repeat(wmax, seglen) - 1e-15, zz, 0.0)[:len(w)]
    zpk = np.maximum.reduceat(zpk, seg_start)
    zpk_full = np.repeat(zpk, seglen)

    ghost_w = np.where(np.abs(zz - zpk_full) > GHOST_GAP, w, 0.0)
    gw = np.add.reduceat(ghost_w, seg_start)
    ratio = gw / np.maximum(seg_w, 1e-12)

    ok = (seg_w > 0.8) & (seglen >= 3)
    return dict(ratio=ratio[ok], seglen=seglen[ok], n_px=int(ok.sum()),
                n_gauss=int(len(w)))


def depth_consistency(g, C, d, up, K):
    dev = "cuda"
    G = {k: torch.from_numpy(g[k]).to(dev) for k in ("means", "quats", "scales", "opac", "sh")}
    Kt = torch.from_numpy(K).float().to(dev)[None]
    bg = torch.ones(3, device=dev)[None]
    V = len(C)
    deps = []
    for i in range(V):
        w2c = m24.look_at(C[i], C[i] + d[i], up[i])
        with torch.no_grad():
            img, _, _ = rasterization(G["means"], G["quats"], G["scales"], G["opac"], G["sh"],
                                      torch.from_numpy(w2c).float().to(dev)[None], Kt,
                                      CELLW, CELLH, sh_degree=g["deg"], packed=False,
                                      backgrounds=bg, render_mode="RGB+ED")
        deps.append(img[0, :, :, 3].cpu().numpy())
    del G
    torch.cuda.empty_cache()

    res, base = [], []
    step = 4
    vv, uu = np.mgrid[0:CELLH:step, 0:CELLW:step]
    for i in range(V - 1):                        # 不闭环：采集序列首尾不相邻
        j = i + 1
        wi = m24.look_at(C[i], C[i] + d[i], up[i])
        wj = m24.look_at(C[j], C[j] + d[j], up[j])
        z = deps[i][vv, uu]
        ok = z > 0.05
        xc = (uu[ok] - K[0, 2]) / K[0, 0] * z[ok]
        yc = (vv[ok] - K[1, 2]) / K[1, 1] * z[ok]
        Pw = np.stack([xc, yc, z[ok]], 1) @ wi[:3, :3] + (-wi[:3, :3].T @ wi[:3, 3])
        Pj = Pw @ wj[:3, :3].T + wj[:3, 3]
        zj = Pj[:, 2]
        uj = K[0, 0] * Pj[:, 0] / np.maximum(zj, 1e-9) + K[0, 2]
        vj = K[1, 1] * Pj[:, 1] / np.maximum(zj, 1e-9) + K[1, 2]
        inm = (zj > 0.05) & (uj >= 0) & (uj < CELLW - 1) & (vj >= 0) & (vj < CELLH - 1)
        uj = uj[inm].astype(np.int32)
        vj = vj[inm].astype(np.int32)
        zg = deps[j][vj, uj]
        good = zg > 0.05
        rel = np.abs(zj[inm][good] - zg[good]) / np.maximum(zg[good], 1e-6)
        res.append(rel)
        base.append(float(np.linalg.norm(C[j] - C[i])))
    allr = np.concatenate(res)
    return dict(med=100 * float(np.median(allr)), p90=100 * float(np.percentile(allr, 90)),
                base_med=float(np.median(base)), base_span=(float(min(base)), float(max(base))))


def main():
    os.makedirs(OUT, exist_ok=True)
    Cref, dref, upref = m24.read_cams(*m24.CASES[0]["cams"])
    orb = m24.fit_orbit(Cref, dref, upref)
    K0 = np.load(m24.A + "/cameras.npz")["intrinsics"].astype(np.float64)[0]
    K = K0.copy()
    K[0, 0] *= CELLW / 378.0
    K[1, 1] *= CELLH / 504.0
    K[0, 2], K[1, 2] = (CELLW - 1) / 2.0, (CELLH - 1) / 2.0

    ths = orb["th"]
    dif = np.diff(np.concatenate([ths, ths[:1] + 2 * np.pi]))
    k = int(np.argmax(dif))
    pick = (k + 1 + len(ths) // 3) % len(ths)
    a0 = float(ths[pick])
    a1 = float(ths[(pick + 1) % len(ths)]) + (2 * np.pi if pick + 1 >= len(ths) else 0.0)
    mid_eye = m24.ring_interp(orb, np.array([0.5 * (a0 + a1)]))[0]
    w2c_mid = m24.look_at(mid_eye, orb["target"], orb["axis"])
    log("🎥 插值视角 %.1f°（相机 %.1f° / %.1f°）"
        % (np.degrees(0.5 * (a0 + a1)), np.degrees(a0), np.degrees(a1)))

    rows = []
    for tag in ("DA3-GS", "InfiniSplat v2"):
        log("\n" + "─" * 70)
        g, s = load_aligned(tag, Cref)
        log("▶ %s  %d 点 SH%d  s=%.4f" % (tag, g["n"], g["deg"], s))
        gm = ghost_metric(g, w2c_mid, K)
        r = gm["ratio"]
        log("  G5 幽灵能见度：有效像素 %d ｜ 主峰外权重占比 中位 %.3f ｜ p75 %.3f ｜ p90 %.3f"
            % (gm["n_px"], np.median(r), np.percentile(r, 75), np.percentile(r, 90)))
        log("     >0.15 的像素 %.1f%% ｜ >0.30 %.1f%% ｜ >0.50 %.1f%%"
            % (100 * np.mean(r > 0.15), 100 * np.mean(r > 0.30), 100 * np.mean(r > 0.50)))
        dv = gm["seglen"]
        log("     每像素高斯数 中位 %d p90 %d" % (np.median(dv), np.percentile(dv, 90)))
        dc = depth_consistency(g, Cref, dref, upref, K)
        log("  G6 跨视角深度：残差 中位 %.2f%% ｜ p90 %.2f%% ｜ 基线中位 %.2f m（%.2f~%.2f）"
            % (dc["med"], dc["p90"], dc["base_med"], *dc["base_span"]))
        rows.append((tag, float(np.median(r)), 100 * float(np.mean(r > 0.15)),
                     dc["med"], dc["p90"]))
        del g
        torch.cuda.empty_cache()

    log("\n" + "=" * 74)
    log("%-15s %13s %12s %14s %10s" % ("方法", "幽灵占比中位", ">0.15像素", "深度残差中位", "p90"))
    for t, a, b, c, dd in rows:
        log("%-15s %13.3f %11.1f%% %13.2f%% %9.2f%%" % (t, a, b, c, dd))


if __name__ == "__main__":
    main()
