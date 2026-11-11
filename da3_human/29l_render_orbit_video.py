#!/usr/bin/env python3
"""29l：DA3 前馈直出 orbit 视频——「优化后口径」展示（峰哥 2026-11-11 要求）。

背景：峰哥要「先别训练，把优化后的 DA3 重建渲染个视频看看，再考虑高斯推理优化」。
「优化后」= 已决策的导出归位（不裁剪 + 全 SH，D1 实测空洞率 0.7% vs 34.8%）。

做法：完全复用 audit_32（compare_4methods）已验证的轨道与对齐管线
    （capture 系 32 相机拟合 five_way_arc 轨道 → 三对点 Umeyama 对齐 DA3 预测相机 →
    warp PLY 到 capture 系 → gsplat rasterization），
    只把五宫格换成上下两格：上=旧口径 FF_official（裁剪+DC），下=优化后 FF_max（不裁剪+全SH）。

用法（WSL）：
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate querysplat
    python /mnt/c/code/media_code/da3_human/29l_render_orbit_video.py
产物：
    <OUT>/orbit_da3_ff_official_vs_max.mp4 + 首/中/末帧 png
环境变量：OUT / NFRAMES / FPS / CELLW
"""
import os
import sys
import time
import subprocess
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/mnt/c/code/media_code/compare_4methods")
import audit_24_five_way_video as a24  # noqa: E402  read_ply / read_cams / umeyama / warp / fit_orbit / flat_ring / arc_angles / look_at / ffmpeg_exe

from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from gsplat import rasterization  # noqa: E402

HOME = os.path.expanduser("~")
O = HOME + "/output"
GT_CAMS = O + "/da3_results/_captures/佳佳抱枕/cameras.npz"          # 32v capture 系 c2w（轨道基准）
FF_DIR = O + "/da3_results/佳佳抱枕_3v/d1_ff_export"                 # D1 四档导出（含 cameras.npz）
OUT = os.environ.get("OUT", "/mnt/d/output/d1_diag_view")
NFRAMES = int(os.environ.get("NFRAMES", "240"))
FPS = int(os.environ.get("FPS", "30"))
CELLW = int(os.environ.get("CELLW", "560"))
RADIUS_K = 1.06
TRIM = 0.02

CASES = [
    dict(key="official",
         name="DA3 前馈直出 · 旧口径（默认导出：裁剪 + DC）",
         meta="背景空洞率 34.8% ｜ 问题现象",
         color=(200, 60, 60),
         ply=FF_DIR + "/gs_official/0000.ply",
         cams=("w2c34", FF_DIR + "/cameras.npz")),
    dict(key="max",
         name="DA3 前馈直出 · 优化后（不裁剪 + 全 SH，已决策）",
         meta="背景空洞率 0.7% ｜ 白洞消除，无需重训",
         color=(30, 150, 70),
         ply=FF_DIR + "/gs_max/0000.ply",
         cams=("w2c34", FF_DIR + "/cameras.npz")),
]

FONT_R = "/mnt/c/Windows/Fonts/msyh.ttc"
FONT_B = "/mnt/c/Windows/Fonts/msyhbd.ttc"


def log(m):
    print("[29l] " + m, flush=True)


def main():
    dev = "cuda"
    free_b, total_b = torch.cuda.mem_get_info()
    log("GPU 预检: 空闲 %.1f / %.1f GiB" % (free_b / 2**30, total_b / 2**30))
    if free_b < 4 * 2**30:
        raise SystemExit("❌ GPU 空闲显存 < 4 GiB，拒跑")

    # ── 轨道：与 audit_32/audit_24 同轨迹（capture 系）──
    Cref, dref, upref = a24.read_cams("c2w", GT_CAMS)
    orb = a24.fit_orbit(Cref, dref, upref)
    log("轨道基准 = capture %d 相机 ｜ 注视点 (%.3f, %.3f, %.3f)" % (len(Cref), *orb["target"]))

    # 对齐基准 idx3 = 3v 输入帧（0/40/81）在 32v 里的最近邻（audit_32 同款）
    base3 = np.load(O + "/infinisplat_v2_results/_captures/佳佳抱枕_3v/cameras.npz")["c2w"].astype(np.float64)
    Cb = base3[:, :3, 3]
    idx3 = [int(np.argmin(((Cref - c) ** 2).sum(1))) for c in Cb]

    Rb = float(np.percentile(orb["r"], 90) * RADIUS_K)
    elev = float(np.median(orb["el"]))
    th_q, span, span_full = a24.arc_angles(orb, NFRAMES, TRIM)
    eyes = a24.flat_ring(orb, th_q, Rb, elev)
    log("ARC 轨迹 %d 帧扫 %.1f°（观测弧 %.1f°）｜ 半径 %.2f m" % (len(eyes), span, span_full, Rb))

    # 内参：GT 原图 K 等比缩到单格（audit_32 同款）
    K0 = np.load(GT_CAMS)["intrinsics"].astype(np.float64)[0]
    W0, H0 = float(K0[0, 2] * 2), float(K0[1, 2] * 2)
    CELLH = int(round(CELLW * 4 / 3))
    K = K0.copy()
    K[0, 0] *= CELLW / W0
    K[1, 1] *= CELLH / H0
    K[0, 2] = (CELLW - 1) / 2.0
    K[1, 2] = (CELLH - 1) / 2.0

    W2C = [torch.from_numpy(a24.look_at(e, orb["target"], orb["axis"])).float().to(dev)[None]
           for e in eyes]
    Kt = torch.from_numpy(K).float().to(dev)[None]
    bg = torch.ones(3, device=dev)

    # ── 逐口径：载入→对齐→渲染→释放 ──
    panels = []
    for c in CASES:
        if not Path(c["ply"]).exists():
            raise SystemExit("❌ 缺产物 " + c["ply"])
        t0 = time.time()
        g = a24.read_ply(c["ply"])
        Cx, dx, upx = a24.read_cams(*c["cams"])
        s, R, t = a24.umeyama(Cx, Cref[idx3])
        C2 = (s * (R @ Cx.T)).T + t
        rms = float(np.sqrt(((C2 - Cref[idx3]) ** 2).sum(1).mean()))
        d2 = (R @ dx.T).T
        ang = float(np.median(np.degrees(np.arccos(np.clip(
            np.sum(d2 * dref[idx3], 1), -1, 1)))))
        log("%s: %d 点 SH%d ｜ s=%.4f rms=%.4f m 视线 %.2f° ｜ 载入 %.1fs"
            % (c["key"], g["n"], g["deg"], s, rms, ang, time.time() - t0))
        g = a24.warp(g, s, R, t)
        G = {k: torch.from_numpy(g[k]).to(dev)
             for k in ("means", "quats", "scales", "opac", "sh")}
        deg = g["deg"]
        del g

        t0 = time.time()
        buf = []
        amed = 0.0
        with torch.no_grad():
            for i, w2ct in enumerate(W2C):
                img, alpha, _ = rasterization(
                    G["means"], G["quats"], G["scales"], G["opac"], G["sh"], w2ct, Kt,
                    CELLW, CELLH, sh_degree=deg, packed=False,
                    backgrounds=bg[None], render_mode="RGB")
                buf.append(img[0].clamp(0, 1).mul(255).byte().cpu().numpy())
                amed = float(alpha.median())
        dt = (time.time() - t0) / len(W2C)
        log("   ✅ 渲完 %d 帧 ｜ %.0f ms/帧 ｜ α 中位 %.3f ｜ 显存峰值 %.1f GB"
            % (len(buf), dt * 1000, amed, torch.cuda.max_memory_allocated() / 1e9))
        panels.append(buf)
        del G
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # ── 逐帧拼横向两格 + 编码 ──
    os.makedirs(OUT, exist_ok=True)
    MARGIN, GAP, HEAD = 16, 12, 64
    Wv = MARGIN * 2 + CELLW * 2 + GAP
    Hv = HEAD + CELLH + 56 + MARGIN
    Hv += Hv % 2
    canvas0 = Image.new("RGB", (Wv, Hv), "white")
    dr = ImageDraw.Draw(canvas0)
    f_t = ImageFont.truetype(FONT_R, 20)
    f_n = ImageFont.truetype(FONT_B, 22)
    f_m = ImageFont.truetype(FONT_R, 16)
    title = ("DA3 前馈直出（3 视角输入）· 同一 orbit 轨迹 · 左=旧口径（裁剪+DC） 右=优化后（不裁剪+全 SH）")
    tw = dr.textlength(title, font=f_t)
    dr.text(((Wv - tw) / 2, 12), title, font=f_t, fill=(110, 110, 110))
    del dr

    mp4 = os.path.join(OUT, "orbit_da3_ff_official_vs_max.mp4")
    ff = a24.ffmpeg_exe()
    cmd = [ff, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (Wv, Hv),
           "-r", str(FPS), "-i", "-", "-pix_fmt", "yuv420p",
           "-c:v", "libx264", "-crf", "19", "-preset", "medium", mp4]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    snaps = {0, NFRAMES // 2, NFRAMES - 1}
    for i in range(NFRAMES):
        fr = canvas0.copy()
        dr = ImageDraw.Draw(fr)
        for ci, c in enumerate(CASES):
            x = MARGIN + ci * (CELLW + GAP)
            fr.paste(Image.fromarray(panels[ci][i]), (x, HEAD))
            nw = dr.textlength(c["name"], font=f_n)
            dr.text((x + (CELLW - nw) / 2, HEAD + CELLH + 6), c["name"], font=f_n, fill=c["color"])
            mw = dr.textlength(c["meta"], font=f_m)
            dr.text((x + (CELLW - mw) / 2, HEAD + CELLH + 36), c["meta"], font=f_m, fill=(120, 120, 120))
            dr.rectangle([x, HEAD - 6, x + CELLW, HEAD - 2], fill=c["color"])
        proc.stdin.write(np.asarray(fr, dtype=np.uint8).tobytes())
        if i in snaps:
            fr.save(os.path.join(OUT, "orbit_cmp_frame_%03d.png" % i))
    proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        raise SystemExit("❌ ffmpeg 编码失败 rc=%d" % rc)
    log("🎬 %s ｜ %d×%d ｜ %d 帧 / %d fps" % (mp4, Wv, Hv, NFRAMES, FPS))


if __name__ == "__main__":
    main()
