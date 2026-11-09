#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""audit_30_native_ghost.py — InfiniSplat 重影是否为复现管线引入的对照实验

问题：用户记得之前在 supersplat 里直接看 InfiniSplat 的 PLY 没有这么严重的重影，
怀疑 audit_24 的跨方法对齐（Umeyama warp 到 DA3-GS 系）把画面搞坏了。

判据（先写）：
  J1  若「原生相机 + 插值视角」也出现双脸/重影 → 分层是 PLY 固有属性，与 warp 无关；
  J2  若原生插值干净、只有 warp 后才双脸 → 复现管线有 bug，必须修；
  J3  同一插值视角拉远半径（×1.6，模拟 supersplat 默认拉远看全局）重影是否仍可读 —
      解释「为什么印象里 supersplat 看着还行」（脸占像素少，5 cm 层间距低于可读阈值）。

做法：完全不动 audit_24 的任何变换，直接用 InfiniSplat 自己的 cameras.npz（c2w+K）
拟合观测环（复用 m24.fit_orbit 的算法但喂原生相机），渲三个视角：
  A = 输入视角 0（应有 GT 对应 eval/render_000.png，作零假设）
  B = 相邻相机方位角中点的插值视角（原生 K、原生半径）
  C = 同 B 但半径 ×1.6（拉远）
另渲 D = DA3-GS 原生插值视角（同世界方位，作「单脸但糊」的参照）。
输出：全幅拼图 + 人脸 ×3 裁剪拼图。
"""
import os, sys, io

os.environ.setdefault("ONLY", "none")
os.environ.setdefault("NFRAMES", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import torch
import importlib.util

spec = importlib.util.spec_from_file_location("m24", os.path.join(HERE, "audit_24_five_way_video.py"))
m24 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m24)

from PIL import Image
from gsplat import rasterization

IN = os.environ.get("IN_DIR", os.path.expanduser("~/output/infinisplat_v2_results/jiajia_roi085_518"))
DA3 = os.path.expanduser("~/output/da3_gs_results/佳佳抱枕_p1m_off")
OUT = os.path.join(DA3, "orbit_video", "_ghost")
os.makedirs(OUT, exist_ok=True)

dev = "cuda" if torch.cuda.is_available() else "cpu"
log = lambda *a: print(*a, flush=True)


def load_native():
    d = np.load(os.path.join(IN, "cameras.npz"))
    c2w = d["c2w"].astype(np.float64)
    Ks = d["intrinsics"].astype(np.float64)
    C = c2w[:, :3, 3].copy()
    d_cam = c2w[:, :3, 2].copy()                      # OpenCV c2w：+z 前向
    up = c2w[:, :3, 1].copy()                          # +y 向下
    d = d_cam / np.linalg.norm(d_cam, axis=1, keepdims=True)
    up = up / np.linalg.norm(up, axis=1, keepdims=True)
    return C, d, up, c2w, Ks


def fit_orbit_native(C, d, up):
    Am = np.zeros((3, 3)); b = np.zeros(3)
    for Ci, di in zip(C, d):
        P = np.eye(3) - np.outer(di, di)
        Am += P; b += P @ Ci
    target = np.linalg.solve(Am, b)
    Cc = C.mean(0)
    _, sv, vt = np.linalg.svd(C - Cc, full_matrices=False)
    axis = vt[2]
    u = up.mean(0); u /= np.linalg.norm(u) + 1e-12
    if axis @ u < 0:
        axis = -axis
    r = np.linalg.norm(C - target, axis=1)
    el = np.degrees(np.arcsin(np.clip(((C - target) @ axis) / np.maximum(r, 1e-9), -1, 1)))
    th = np.arctan2((C - Cc) @ vt[1], (C - Cc) @ vt[0])
    order = np.argsort(th)
    return dict(target=target, axis=axis, e1=vt[0], e2=vt[1], r=r, el=el,
                th=th[order], C_ring=C[order])


def render_native(g, c2w_i, K_i, W, H, deg):
    w2c = np.linalg.inv(c2w_i)
    w2ct = torch.from_numpy(w2c).float().to(dev)[None]
    Kt = torch.from_numpy(K_i).float().to(dev)[None]
    G = {k: torch.from_numpy(g[k]).to(dev)
         for k in ("means", "quats", "scales", "opac", "sh")}
    with torch.no_grad():
        img, alpha, _ = rasterization(
            G["means"], G["quats"], G["scales"], G["opac"], G["sh"],
            w2ct, Kt, W, H, sh_degree=deg, packed=False,
            backgrounds=torch.ones(1, 3, device=dev), render_mode="RGB")
    a = img[0].clamp(0, 1).mul(255).byte().cpu().numpy()
    return a, float(alpha[0].mean())


def main():
    g = m24.read_ply(os.path.join(IN, "scene.ply"))
    log("InfiniSplat scene.ply：%d 点 SH%d" % (g["n"], g["deg"]))
    d_npz = np.load(os.path.join(IN, "cameras.npz"))
    c2w = d_npz["c2w"].astype(np.float64)
    Ks = d_npz["intrinsics"].astype(np.float64)
    C, d, up = m24.read_cams("c2w", os.path.join(IN, "cameras.npz"))
    orb = m24.fit_orbit(C, d, up)      # 与视频管线同一套（三道自证过）
    H, W = 518, 392
    log("原生轨道：半径中位 %.3f（rdnorm）仰角中位 %.1f° ｜ 视线夹角中位 %.2f°"
        % (np.median(orb["r"]), np.median(orb["el"]), orb["ang_med"]))

    ths = orb["th"]
    k = int(np.argmax(np.diff(np.concatenate([ths, ths[:1] + 2 * np.pi]))))
    pick = (k + 1 + len(ths) // 3) % len(ths)      # 与 audit_26 同一对相邻相机
    th_mid = float((ths[pick] + ths[(pick + 1) % len(ths)]) / 2)

    eyeB = m24.ring_interp(orb, np.array([th_mid]))[0]
    w2cB = m24.look_at(eyeB, orb["target"], orb["axis"])
    rB = float(np.linalg.norm(eyeB - orb["target"]))
    eyeC = orb["target"] + (eyeB - orb["target"]) / rB * rB * 1.6
    w2cC = m24.look_at(eyeC, orb["target"], orb["axis"])

    views = [("A 输入视角0", np.linalg.inv(c2w[0]), Ks[0]),
             ("B 插值@1.0r", w2cB, Ks[0]),
             ("C 插值@1.6r(拉远)", w2cC, Ks[0])]
    frames = []
    for tag, w2c_i, K_i in views:
        a, am = render_native(g, w2c_i, K_i, W, H, g["deg"])
        frames.append((tag, a))
        log("  %-18s α均值 %.4f" % (tag, am))
        Image.fromarray(a).save(os.path.join(OUT, "native_%s.png" % tag.split()[0]))

    # D：DA3-GS 参照 —— 同一物理相机环（provided poses），用 Umeyama 把插值 eye 搬到 metric 系
    g2 = m24.read_ply(os.path.join(DA3, "official_export/i12_fullpts_sh/0000.ply"))
    C3, d3, up3 = m24.read_cams("w2c34", os.path.join(DA3, "cameras.npz"))
    s, R, t = m24.umeyama(C, C3)
    eyeD = s * (R @ eyeB) + t
    orb3 = m24.fit_orbit(C3, d3, up3)
    w2cD = m24.look_at(eyeD, orb3["target"], orb3["axis"])
    Kd = np.array([[0.966 * 378, 0, (378 - 1) / 2], [0, 0.966 * 378, (504 - 1) / 2], [0, 0, 1]])
    aD, amD = render_native(g2, w2cD, Kd, 378, 504, g2["deg"])
    log("  D DA3-GS同方位     α均值 %.4f（s=%.4f）" % (amD, s))
    Image.fromarray(aD).save(os.path.join(OUT, "native_D_da3.png"))

    def crop(im, box=(0.30, 0.05, 0.75, 0.42), scale=2):
        h, w = im.shape[:2]
        c = im[int(box[1] * h):int(box[3] * h), int(box[0] * w):int(box[2] * w)]
        return np.array(Image.fromarray(c).resize((c.shape[1] * scale, c.shape[0] * scale), Image.LANCZOS))

    crops = [crop(a) for _, a in frames] + [crop(aD)]
    ch = max(c.shape[0] for c in crops)
    cw = [c.shape[1] for c in crops]
    row = np.concatenate([np.pad(c, ((0, ch - c.shape[0]), (6, 6), (0, 0)), constant_values=255) for c in crops], 1)
    Image.fromarray(row).save(os.path.join(OUT, "native_face_crops.png"))
    log("✅ 输出 %s（A/B/C/D 四张全幅 + 裁剪拼图）" % OUT)


if __name__ == "__main__":
    main()
