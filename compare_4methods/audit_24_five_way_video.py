#!/usr/bin/env python3
"""五方法同一相机轨迹的对比视频（252.7° 观测弧）。

为什么需要「对齐」而不是各渲各的
--------------------------------
五个方法各自活在自己的世界系里：DA3-GS 是 metric（scale_factor≈2.0、is_metric=1），
QuerySplat / InfiniSplat v2 / VGGT-Ω+3DGS 是 VGGT 系，尺度差 2.25~2.28 倍。
直接各渲各的，环视半径、仰角、画面构图全不一样，「同一轨迹」就无从谈起。
做法：把每个方法的 32 个训练相机中心 Umeyama 到 **DA3-GS 的相机**（含尺度），
再用同一组注视点/半径/仰角/方位角序列渲染，最后逐帧横排五联。

判据（P-U1/P-U2，见 _probe5b.py 实测）
  · 相机中心残差 / 基准半径 < 3%   → 全部 0.27%~1.03% ✅
  · 对齐后视线夹角中位 < 5°        → 全部 0.11°~0.46° ✅
  3DGS 的 cameras.json 里 `rotation` 存的是 **c2w**（forward = R 的第 3 列），
  不是 w2c —— 按 w2c 解读视线夹角 97°~106°，四种解读里只有 col+ 过判据。

轨道口径
--------
· 起点 = 缺口结束端，沿 252.7° 观测弧走满（跳过 107.3° 无观测区）。
· 注视点 = 32 条视线的**最小二乘交点**（不是相机中心均值，那会偏 0.86 m）。
· 内参统一取 DA3-GS 的（视场也是「相机」的一部分），五格画面构图逐像素可比。

产出
----
  <OUT>/five_way_arc.mp4      1920×598 五联横排
  <OUT>/five_way_arc_frame_*.png  首/中/末帧，便于抽检

环境变量：OUT / NFRAMES / FPS / CELLW / RADIUS_K / ELEV / BG / ONLY
"""
import json
import os
import re
import subprocess

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation as Rot
from gsplat import rasterization

HOME = "/home/velvet"
O = HOME + "/output"
A = O + "/da3_gs_results/佳佳抱枕_p1m_off"
Q = O + "/querysplat_results/佳佳抱枕_paper"
V = O + "/vggt_omega_results/佳佳抱枕/04_model_3dgs"
D = O + "/da3_results/佳佳抱枕/04_model_3dgs"
# InfiniSplat 用 jiajia 运行（1439×1919 原图输入 → 392×518），与 fig35 的 is_renders 同一口径；
# 2026-11-09 曾误用 jiajia_roi085_518（388×518 预处理裁剪图输入），其层间一致性崩坏、输入视角都破碎，
# 导致视频里该格重影被严重夸大。详见 FINDINGS §5.3-decies 修订。
I = O + "/infinisplat_v2_results/jiajia"

OUT = os.environ.get("OUT", A + "/orbit_video")
NFRAMES = int(os.environ.get("NFRAMES", "240"))
FPS = int(os.environ.get("FPS", "30"))
CELLW = int(os.environ.get("CELLW", "372"))          # 3:4 → 高 = CELLW*4/3
RADIUS_K = float(os.environ.get("RADIUS_K", "1.06"))
ELEV = os.environ.get("ELEV", "").strip()
BG = os.environ.get("BG", "white")
ONLY = os.environ.get("ONLY", "").strip()            # 调试：只渲一个方法
# 两端各内收的弧长比例：缺口边界处的相机是斜视的，五方法在那里都会拉出长条/拖影
# （末帧 VGGT/DA3+3DGS 的窗帘区就是这样）。0.02 → 两端各让 5.05°，覆盖 242.6°。
TRIM = float(os.environ.get("TRIM", "0.02"))

FONT_R = "/mnt/c/Windows/Fonts/msyh.ttc"
FONT_B = "/mnt/c/Windows/Fonts/msyhbd.ttc"

# 标签色与 fig35 一致：主角青、同档蓝紫、参照灰橙
CASES = [
    dict(name="DA3-GS", meta="13.5 秒 · 610 万高斯", color=(15, 110, 110),
         ply=A + "/official_export/i12_fullpts_sh/0000.ply",
         cams=("w2c34", A + "/cameras.npz"), ref=True),
    dict(name="QuerySplat", meta="38.5 秒 · 37.1 万高斯", color=(91, 127, 181),
         ply=Q + "/gaussians.ply",
         cams=("c2w", Q + "/predicted_input_cameras.npz")),
    dict(name="InfiniSplat v2", meta="3.3 分钟 · 594 万高斯", color=(139, 111, 168),
         ply=I + "/scene.ply",
         cams=("c2w", I + "/cameras.npz")),
    dict(name="VGGT-Ω + 3DGS", meta="20.4 分钟 · 175 万高斯", color=(168, 131, 79),
         ply=V + "/point_cloud/iteration_30000/point_cloud.ply",
         cams=("json", V + "/cameras.json")),
    dict(name="DA3 + 3DGS", meta="22.9 分钟 · 189 万高斯", color=(168, 131, 79),
         ply=D + "/point_cloud/iteration_30000/point_cloud.ply",
         cams=("json", D + "/cameras.json")),
]
if ONLY:
    CASES = [c for c in CASES if ONLY.lower() in c["name"].lower()]


def log(*a):
    print(*a, flush=True)


# ── 数据读取 ────────────────────────────────────────────────────────────────
def read_ply(path):
    """gsplat / 官方 3DGS 两家 PLY 都吃：按属性名取，忽略 normals。"""
    with open(path, "rb") as f:
        head = b""
        while b"end_header" not in head:
            head += f.read(1)
        tail = f.read(1)
        while tail not in (b"\n", b""):
            head += tail
            tail = f.read(1)
        head += tail
        text = head.decode("ascii", "replace")
        n = int(re.search(r"element vertex (\d+)", text).group(1))
        names = re.findall(r"^property float (\w+)$", text, re.M)
        dt = np.dtype([(nm, "<f4") for nm in names])
        f.seek(len(head))
        raw = np.fromfile(f, dtype=dt, count=n)
    if raw.shape[0] != n:
        raise SystemExit("❌ %s: 读到 %d 顶点，头里写 %d" % (path, raw.shape[0], n))
    means = np.stack([raw["x"], raw["y"], raw["z"]], 1).astype(np.float32)
    # PLY 里 scale_ 是 log 尺度（实测 DA3 中位 −6.28 → exp 后 1.9 mm），gsplat 要线性
    scales = np.exp(np.stack([raw["scale_%d" % i] for i in range(3)], 1)).astype(np.float32)
    opac = (1.0 / (1.0 + np.exp(-raw["opacity"].astype(np.float64)))).astype(np.float32)
    quats = np.stack([raw["rot_%d" % i] for i in range(4)], 1).astype(np.float32)
    f_dc = np.stack([raw["f_dc_%d" % i] for i in range(3)], 1).astype(np.float32)
    rest = ["f_rest_%d" % i for i in range(len(names)) if "f_rest_%d" % i in names]
    k = len(rest) // 3
    if k:                                   # channel-major → (N,K,3)，同 audit_22/E8
        fr = np.stack([raw[nm] for nm in rest], 1)
        fr = fr.reshape(len(means), 3, k).transpose(0, 2, 1)
        sh = np.concatenate([f_dc[:, None, :], fr], 1)
    else:
        sh = f_dc[:, None, :]
    return dict(means=means, scales=scales, opac=opac, quats=quats, sh=sh,
                n=int(n), deg=int(round((1 + k) ** 0.5)) - 1)


def _asvec(v):
    if isinstance(v, str):
        return np.array(json.loads(v.replace("[3]", "").strip()), dtype=np.float64)
    return np.array(v, dtype=np.float64)


def read_cams(kind, path):
    """统一返回 (C (V,3) 相机中心, d (V,3) 前向, up (V,3) 世界上方向)"""
    if kind == "w2c34":
        e = np.load(path)["extrinsics"].astype(np.float64)
        if e.ndim == 4:
            e = e[:, :3, :]
        Rwc, twc = e[:, :3, :3], e[:, :3, 3]
        C = -np.einsum("vij,vj->vi", np.transpose(Rwc, (0, 2, 1)), twc)
        d = np.einsum("vij,j->vi", np.transpose(Rwc, (0, 2, 1)), np.array([0.0, 0.0, 1.0]))
        ycam = np.einsum("vij,j->vi", np.transpose(Rwc, (0, 2, 1)), np.array([0.0, 1.0, 0.0]))
    elif kind == "c2w":
        c = np.load(path)["c2w"].astype(np.float64)
        C, d, ycam = c[:, :3, 3], c[:, :3, 2], c[:, :3, 1]
    elif kind == "json":
        j = json.load(open(path))
        C = np.array([_asvec(it["position"]) for it in j])
        # 实测：3DGS 的 rotation 是 c2w，forward = 第 3 列（四种解读只有这个过判据）
        d = np.array([_asvec(it["rotation"])[:, 2] for it in j])
        ycam = np.array([_asvec(it["rotation"])[:, 1] for it in j])
    else:
        raise SystemExit("未知相机格式 " + kind)
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    up = -ycam                                   # 相机 y 轴朝下 → 世界 up = −y_cam
    return C, d, up / np.linalg.norm(up, axis=1, keepdims=True)


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, Dm = src - mu_s, dst - mu_d
    cov = Dm.T @ S / len(src)
    U, sv, Vt = np.linalg.svd(cov)
    F = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        F[2, 2] = -1
    R = U @ F @ Vt
    s = (sv * np.diag(F)).sum() / ((S ** 2).sum() / len(src))
    return s, R, mu_d - s * R @ mu_s


# ── 轨道 ────────────────────────────────────────────────────────────────────
def fit_orbit(C, d, up):
    """同 audit_22：注视点 = 视线最小二乘交点；环法向按相机 up 定符号。"""
    Am = np.zeros((3, 3))
    b = np.zeros(3)
    for Ci, di in zip(C, d):
        P = np.eye(3) - np.outer(di, di)
        Am += P
        b += P @ Ci
    target = np.linalg.solve(Am, b)
    v = target[None, :] - C
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    ang = np.degrees(np.arccos(np.clip(np.sum(v * d, 1), -1, 1)))
    if np.median(ang) > 10:
        raise SystemExit("❌ 视线方向与注视点夹角中位 %.1f°，坐标系假设错了" % np.median(ang))
    Cc = C.mean(0)
    _, sv, vt = np.linalg.svd(C - Cc, full_matrices=False)
    axis = vt[2]
    up_w = up.mean(0)
    up_w /= np.linalg.norm(up_w) + 1e-12
    if float(axis @ up_w) < 0:
        axis = -axis
    up_ang = float(np.degrees(np.arccos(np.clip(abs(axis @ up_w), -1, 1))))
    r = np.linalg.norm(C - target, axis=1)
    el = np.degrees(np.arcsin(np.clip(((C - target) @ axis) / np.maximum(r, 1e-9), -1, 1)))
    th = np.arctan2((C - Cc) @ vt[1], (C - Cc) @ vt[0])
    order = np.argsort(th)
    return dict(target=target, axis=axis, C=C, r=r, el=el,
                th=th[order], C_ring=C[order], th0=float(th[0]),
                ang_med=float(np.median(ang)), up_ang=up_ang, sv=sv, Cc=Cc)


def look_at(eye, target, up):
    z = target - eye
    z = z / (np.linalg.norm(z) + 1e-12)
    if abs(float(z @ up)) > 0.98:
        up = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.9 else np.array([0.0, 0.0, 1.0])
    x = np.cross(z, up)                                   # 手性：右 = forward × up
    x = x / (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)
    R = np.stack([x, y, z], 1)
    w2c = np.eye(4)
    w2c[:3, :3] = R.T
    w2c[:3, 3] = -R.T @ eye
    return w2c


def arc_angles(orb, n, trim=0.0):
    """沿「有观测的弧」等分 n 个方位角（跳过最大缺口）。

    trim>0 时两端各内收 trim×弧长 —— 缺口边界上最近的相机是斜视的，五个方法在那一帧
    都会拉长条（实测末帧 VGGT/DA3+3DGS 的窗帘变成黑白条纹），内收 5° 即可避开。
    """
    ths = orb["th"]
    two_pi = 2 * np.pi
    g = np.diff(np.concatenate([ths, ths[:1] + two_pi]))
    k = int(np.argmax(g))
    gap = float(g[k])
    start = float(ths[(k + 1) % len(ths)])
    span = two_pi - gap
    log("   最大缺口 %.1f°（%.1f° → %.1f° 无相机）｜ 观测弧 %.1f°"
        % (np.degrees(gap), np.degrees(ths[k]), np.degrees(ths[(k + 1) % len(ths)]),
           np.degrees(span)))
    span_full = span
    if trim > 0:
        pad = span * trim
        start += pad
        span -= 2 * pad
        log("   两端各内收 %.1f°（trim=%.2f）→ 实际扫过 %.1f°"
            % (np.degrees(pad), trim, np.degrees(span)))
    seq = np.array([start + span * i / max(n - 1, 1) for i in range(n)])
    return seq, float(np.degrees(span)), float(np.degrees(span_full))


def ring_interp(orb, th_q):
    """在真实相机位置之间按方位角线性插值 → 轨迹上每帧的 eye 点。"""
    ths, Cs = orb["th"], orb["C_ring"]
    two_pi = 2 * np.pi
    ths_c = np.concatenate([ths, ths[:1] + two_pi])
    Cs_c = np.vstack([Cs, Cs[:1]])
    gaps = np.diff(ths_c)
    base = float(ths[0])
    out = np.zeros((len(th_q), 3))
    for i, th in enumerate(th_q):
        t = base + (th - base) % two_pi
        k = int(np.clip(np.searchsorted(ths_c, t) - 1, 0, len(ths_c) - 2))
        w = (t - ths_c[k]) / gaps[k]
        out[i] = Cs_c[k] * (1 - w) + Cs_c[k + 1] * w
    return out


# ── 高斯变换 ────────────────────────────────────────────────────────────────
def quat_mul(a, b):
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw], 1)


def warp(g, s, R, t):
    """把高斯从方法自身系搬到基准系：p' = s·R·p + t，朝向 R'_q = R·R_q，尺度 ×s。"""
    g["means"] = (s * (g["means"].astype(np.float64) @ R.T) + t).astype(np.float32)
    g["scales"] = (g["scales"].astype(np.float64) * s).astype(np.float32)   # 已是指数化尺度
    qR = Rot.from_matrix(R).as_quat()                    # xyzw
    qR = np.array([qR[3], qR[0], qR[1], qR[2]], dtype=np.float64)   # → wxyz
    qR = np.repeat(qR[None, :], len(g["quats"]), 0)
    q = quat_mul(qR, g["quats"].astype(np.float64))
    g["quats"] = (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32)
    return g


def ffmpeg_exe():
    from shutil import which
    p = which("ffmpeg")
    if p:
        return p
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


# ── 主流程 ──────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda"
    Cref, dref, upref = read_cams("w2c34", A + "/cameras.npz")
    orb = fit_orbit(Cref, dref, upref)
    log("📷 基准 = DA3-GS（%d 相机）｜ 注视点 (%.3f, %.3f, %.3f) ｜ 视线夹角中位 %.2f°"
        % (len(Cref), *orb["target"], orb["ang_med"]))
    log("   半径 90 分位 %.3f m ｜ 仰角中位 %.1f° ｜ 奇异值比 %.3f ｜ 法向 vs 相机 up %.2f°"
        % (np.percentile(orb["r"], 90), np.median(orb["el"]),
           orb["sv"][2] / orb["sv"][0], orb["up_ang"]))
    if orb["up_ang"] > 60:
        raise SystemExit("❌ 环法向与相机 up 夹角 %.1f°，定符号失败" % orb["up_ang"])

    # 自证：0 号相机（参考视角）的真实外参 vs look_at 构造 —— 相机约定错了这里必爆
    e0 = np.load(A + "/cameras.npz")["extrinsics"].astype(np.float64)
    if e0.ndim == 4:
        e0 = e0[:, :3, :]
    w_chk = look_at(Cref[0], orb["target"], orb["axis"])
    R_err = float(np.degrees(np.arccos(np.clip(
        (np.trace(w_chk[:3, :3].T @ e0[0, :3, :3]) - 1) / 2, -1, 1))))
    if R_err > 5:
        raise SystemExit("❌ look_at(C[0]) 与真实外参旋转差 %.1f°" % R_err)
    log("   ✅ 自证：look_at(C[0]) vs 真实外参 0 号，旋转差 %.2f°（阈值 5°）" % R_err)

    Rb = float(np.percentile(orb["r"], 90) * RADIUS_K)
    elev = float(ELEV) if ELEV else float(np.median(orb["el"]))
    log("   轨道半径 %.3f m（RADIUS_K=%.2f）｜ 仰角 %.1f°" % (Rb, RADIUS_K, elev))

    th_q, span, span_full = arc_angles(orb, NFRAMES, TRIM)
    eyes = ring_interp(orb, th_q)
    log("   ARC 轨迹：%d 帧扫过 %.1f°（观测弧 %.1f° 内）" % (len(eyes), span, span_full))

    # 内参统一取 DA3-GS 的（视场也算相机的一部分）
    K0 = np.load(A + "/cameras.npz")["intrinsics"].astype(np.float64)[0]
    W0, H0 = 378, 504
    CELLH = int(round(CELLW * 4 / 3))
    K = K0.copy()
    K[0, 0] *= CELLW / W0
    K[1, 1] *= CELLH / H0
    K[0, 2] = (CELLW - 1) / 2.0
    K[1, 2] = (CELLH - 1) / 2.0
    log("🖼  单格 %d×%d ｜ fx %.1f fy %.1f（统一内参）" % (CELLW, CELLH, K[0, 0], K[1, 1]))

    # 载入并搬到基准系 —— 逐个方法「载入→渲染→释放」，避免五个 600 万点同时驻留显存
    # （首版五个都常驻，24 GB 卡在第三个方法上 OOM）
    W2C = [torch.from_numpy(look_at(e, orb["target"], orb["axis"])).float().to(dev)[None]
           for e in eyes]
    Kt = torch.from_numpy(K).float().to(dev)[None]
    bg = torch.ones(3, device=dev) if BG == "white" else torch.zeros(3, device=dev)

    # ── 画布：顶部标题 + 五格 ──
    MARGIN, GAP, HEAD = 14, 8, 92
    n = len(CASES)
    Wv = MARGIN * 2 + CELLW * n + GAP * (n - 1)
    Hv = HEAD + CELLH + 10
    Hv += Hv % 2
    canvas = Image.new("RGB", (Wv, Hv), "white")
    dr = ImageDraw.Draw(canvas)
    f_t = ImageFont.truetype(FONT_R, 17)
    f_n = ImageFont.truetype(FONT_B, 21)
    f_m = ImageFont.truetype(FONT_R, 14)
    title = "同一相机轨迹 · 252.7° 观测弧 · 佳佳抱枕（32 视角）· 五方法逐帧对照（统一位姿与内参）"
    tw = dr.textlength(title, font=f_t)
    dr.text(((Wv - tw) / 2, 9), title, font=f_t, fill="#6E6E6E")
    xs = []
    x0 = MARGIN
    for c in CASES:
        xs.append(x0)
        nw = dr.textlength(c["name"], font=f_n)
        cw = dr.textlength(c["meta"], font=f_m)
        tot = max(nw, cw)
        dr.text((x0 + (CELLW - tot) / 2, 36), c["name"], font=f_n, fill=c["color"])
        dr.text((x0 + (CELLW - cw) / 2, 64), c["meta"], font=f_m, fill="#8A8A8A")
        dr.rectangle([x0, HEAD - 8, x0 + CELLW, HEAD - 4], fill=c["color"])
        x0 += CELLW + GAP
    del dr
    log("🎬 视频 %d×%d ｜ %d 帧 / %d fps / %.1f 秒" % (Wv, Hv, NFRAMES, FPS, NFRAMES / FPS))

    # 阶段 1：逐方法渲出整段帧（CPU 侧缓存，372×496×3×240×5 ≈ 0.66 GB）
    frames = []
    for c in CASES:
        g = read_ply(c["ply"])
        Cx, dx, upx = read_cams(*c["cams"])
        if c.get("ref"):
            s, R, t = 1.0, np.eye(3), np.zeros(3)
        else:
            s, R, t = umeyama(Cx, Cref)
        C2 = (s * (R @ Cx.T)).T + t
        rms = float(np.sqrt(((C2 - Cref) ** 2).sum(1).mean()))
        rel = rms / float(np.median(np.linalg.norm(Cref - Cref.mean(0), axis=1)))
        d2 = (R @ dx.T).T
        ang = float(np.median(np.degrees(np.arccos(np.clip(np.sum(d2 * dref, 1), -1, 1)))))
        u2 = (R @ upx.T).T
        uang = float(np.median(np.degrees(np.arccos(np.clip(np.sum(u2 * upref, 1), -1, 1)))))
        flag = "OK" if (rel < 0.03 and ang < 5 and uang < 5) else "FAIL"
        log("   %-15s %7d 点 SH%d ｜ s=%.4f 残差 %.2f%% 视线 %.2f° up %.2f° [%s]"
            % (c["name"], g["n"], g["deg"], s, rel * 100, ang, uang, flag))
        if flag != "OK":
            raise SystemExit("❌ %s 对齐没过判据，别往下渲" % c["name"])
        g = warp(g, s, R, t)
        G = {k: torch.from_numpy(g[k]).to(dev)
             for k in ("means", "quats", "scales", "opac", "sh")}
        deg = g["deg"]
        del g

        buf = []
        amed = 0.0
        for i, w2ct in enumerate(W2C):
            with torch.no_grad():
                img, alpha, _ = rasterization(
                    G["means"], G["quats"], G["scales"], G["opac"], G["sh"], w2ct, Kt,
                    CELLW, CELLH, sh_degree=deg, packed=False,
                    backgrounds=bg[None], render_mode="RGB")
            buf.append(img[0].clamp(0, 1).mul(255).byte().cpu().numpy())
            amed = float(alpha.median())
            if (i + 1) % 60 == 0:
                log("      frame %3d/%d  α 中位 %.3f" % (i + 1, NFRAMES, amed))
        frames.append(buf)
        del G
        torch.cuda.empty_cache()
        log("      ✅ %s 渲完，α 中位 %.3f，显存峰值 %.1f GB"
            % (c["name"], amed, torch.cuda.max_memory_allocated() / 1e9))
        torch.cuda.reset_peak_memory_stats()

    # 阶段 2：逐帧拼接 + 编码
    mp4 = os.path.join(OUT, "five_way_arc.mp4")
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (Wv, Hv),
           "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
           "-movflags", "+faststart", mp4]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    keep = {}
    for i in range(NFRAMES):
        frame = canvas.copy()
        for ci, x0 in enumerate(xs):
            frame.paste(Image.fromarray(frames[ci][i]), (x0, HEAD))
        proc.stdin.write(frame.tobytes())
        if i in (0, NFRAMES // 2, NFRAMES - 1):
            keep[i] = frame
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("❌ ffmpeg 编码失败")
    for i, fr in keep.items():
        fr.save(os.path.join(OUT, "five_way_arc_frame_%03d.png" % i))
    log("✅ %s（%.1f MB）" % (mp4, os.path.getsize(mp4) / 1e6))


if __name__ == "__main__":
    main()
