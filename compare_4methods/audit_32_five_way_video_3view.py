#!/usr/bin/env python3
"""三图输入（N=3）五方法对比视频 —— audit_24 的三图版。

与 32 图版（audit_24_five_way_video.py → video_five_way_arc.mp4）的关系
--------------------------------------------------------------------
· **轨迹完全相同**：轨道仍从 32v DA3-GS（佳佳抱枕_p1m_off）的 32 相机拟合
  （同一注视点 / 半径 / 仰角 / 252.7° 观测弧 / flat_ring 水平匀速圆环），
  内参也取同一颗。两版视频逐帧可比——领导能直接看到「同样的轨迹、少 29 张图」的代价。
· 区别只在高斯云与相机：全部换成 *_3v 产物（90a/90b 批量产出，FINDINGS §8）。
· 对齐：各方法 3v 预测相机 → GT 32 相机中「与 3 个输入帧（帧 0/40/81）最近的 3 个」
  （idx3，与 audit_31 评测同款取法），三对点 Umeyama（7 DoF），
  判据同 audit_24（残差<3%、视线<5°、up<5°），不过拒渲。
  ⚠️ 三对点是 Umeyama 下限配置，判据比 32 图版（32 对点）松不了，必须逐个看 [OK]。

预期观感（FINDINGS §8 已实测）：三图外推 all32 只有 9~12 dB，五格在轨迹中段
都会明显糊/雾/重影——这是真实结果，不是渲染 bug。3DGS 精化系（VGGT/DA3+3DGS）
是三图欠观测空间被 densify 填成半透雾，整格偏亮。

产出：<OUT>/five_way_arc_3view.mp4 + 首/中/末帧 png。

环境变量：OUT / NFRAMES / FPS / CELLW / RADIUS_K / ELEV / BG / ONLY
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_24_five_way_video as a24  # read_ply / read_cams / umeyama / warp / fit_orbit / flat_ring / arc_angles / look_at / ffmpeg_exe

HOME = "/home/velvet"
O = HOME + "/output"
# 轨道基准 = capture 系 32 相机（audit_31 评测的 GT 侧，c2w）。
# ⚠️ 不能用 p1m_off（audit_24 的 32v 基准）：那是 DA3 处理系，与 3v 输入帧位姿（BASE3，
# capture 系）差一个相似变换，跨系最近邻匹配会整体错位（实测 nn 距 2.03 m、视线夹角 83°）。
# 相似变换不改变角度与相对几何 → capture 系拟合的轨道与 32 图版逐帧同轨迹。
GT_CAMS = O + "/da3_results/_captures/佳佳抱枕/cameras.npz"
BASE3 = O + "/infinisplat_v2_results/_captures/佳佳抱枕_3v/cameras.npz"  # 3v 输入帧 GT 位姿（c2w）

OUT = os.environ.get("OUT", O + "/batch_4methods/orbit_video_3view")
NFRAMES = int(os.environ.get("NFRAMES", "240"))
FPS = int(os.environ.get("FPS", "30"))
CELLW = int(os.environ.get("CELLW", "372"))
RADIUS_K = float(os.environ.get("RADIUS_K", "1.06"))
ELEV = os.environ.get("ELEV", "").strip()
BG = os.environ.get("BG", "white")
ONLY = os.environ.get("ONLY", "").strip()
TRIM = float(os.environ.get("TRIM", "0.02"))

FONT_R = "/mnt/c/Windows/Fonts/msyh.ttc"
FONT_B = "/mnt/c/Windows/Fonts/msyhbd.ttc"

# 耗时/高斯数 = FINDINGS §8 三图实测（is 138 s / qs 438 s / vggt 1162 s / da3 3182 s）
CASES = [
    dict(name="DA3-GS", meta="13.5 秒 · 45 万高斯", color=(15, 110, 110),
         ply=O + "/da3_gs_results/佳佳抱枕_3v/gs_ply_full/0000.ply",
         cams=("w2c34", O + "/da3_gs_results/佳佳抱枕_3v_ff/cameras.npz")),
    dict(name="QuerySplat", meta="438 秒 · 19 万高斯", color=(91, 127, 181),
         ply=O + "/querysplat_results/佳佳抱枕_3v/gaussians.ply",
         cams=("c2w", O + "/querysplat_results/佳佳抱枕_3v/predicted_input_cameras.npz")),
    dict(name="InfiniSplat v2", meta="138 秒 · 59 万高斯", color=(139, 111, 168),
         ply=O + "/infinisplat_v2_results/佳佳抱枕_3v/scene.ply",
         cams=("c2w", O + "/infinisplat_v2_results/佳佳抱枕_3v/cameras.npz")),
    dict(name="VGGT-Ω + 3DGS", meta="19.4 分钟 · 46 万高斯", color=(168, 131, 79),
         ply=O + "/vggt_omega_results/佳佳抱枕_3v/04_model_3dgs/point_cloud/iteration_30000/point_cloud.ply",
         cams=("json", O + "/vggt_omega_results/佳佳抱枕_3v/04_model_3dgs/cameras.json")),
    dict(name="DA3 + 3DGS", meta="53 分钟 · 76 万高斯", color=(168, 131, 79),
         ply=O + "/da3_results/佳佳抱枕_3v/04_model_3dgs/point_cloud/iteration_30000/point_cloud.ply",
         cams=("json", O + "/da3_results/佳佳抱枕_3v/04_model_3dgs/cameras.json")),
]
if ONLY:
    CASES = [c for c in CASES if ONLY.lower() in c["name"].lower()]

log = a24.log


def main():
    from gsplat import rasterization

    os.makedirs(OUT, exist_ok=True)
    dev = "cuda"

    # ── 预检：WSL2 vGPU 双重度任务共存 → rasterization 活锁（2026-11-10 定案）──
    free_b, total_b = torch.cuda.mem_get_info()
    log("🩺 GPU 预检: 空闲 %.1f / %.1f GiB" % (free_b / 2**30, total_b / 2**30))
    if free_b < 4 * 2**30:
        raise SystemExit("❌ GPU 空闲显存 < 4 GiB——可能有其它任务在跑（或其残留），"
                         "活锁风险高，拒跑。等其退出或 wsl --shutdown 清残留后再试。")

    # ── 轨道基准 = capture 系 32 相机（c2w，与 audit_31 GT 侧同一颗）──
    Cref, dref, upref = a24.read_cams("c2w", GT_CAMS)
    orb = a24.fit_orbit(Cref, dref, upref)
    log("📷 基准 = capture %d 相机（c2w）｜ 注视点 (%.3f, %.3f, %.3f) ｜ 视线夹角中位 %.2f°"
        % (len(Cref), *orb["target"], orb["ang_med"]))
    if orb["up_ang"] > 60:
        raise SystemExit("❌ 环法向与相机 up 夹角 %.1f°" % orb["up_ang"])

    # 自证：look_at(C[0]) 的 w2c 旋转 vs inv(c2w[0]) 旋转
    c2w0 = np.load(GT_CAMS)["c2w"].astype(np.float64)[0]
    w_chk = a24.look_at(Cref[0], orb["target"], orb["axis"])
    R_err = float(np.degrees(np.arccos(np.clip(
        (np.trace(w_chk[:3, :3].T @ np.linalg.inv(c2w0)[:3, :3]) - 1) / 2, -1, 1))))
    if R_err > 5:
        raise SystemExit("❌ look_at(C[0]) 与 inv(c2w[0]) 旋转差 %.1f°" % R_err)
    log("   ✅ 自证：look_at(C[0]) vs inv(c2w[0])，旋转差 %.2f°" % R_err)

    # ── 对齐基准 idx3：3v 输入帧（帧 0/40/81）在 32v 集合里的最近邻（audit_31 同款）──
    Cb, _, _ = a24.read_cams("c2w", BASE3)
    idx3 = [int(np.argmin(((Cref - c) ** 2).sum(1))) for c in Cb]
    if len(set(idx3)) != 3:
        raise SystemExit("❌ idx3 去重后不足 3 个: %s" % idx3)
    nn_dist = [float(np.linalg.norm(Cref[i] - c)) for i, c in zip(idx3, Cb)]
    if max(nn_dist) > 0.5:
        raise SystemExit("❌ 输入帧最近邻太远（%s m）——capture 系假设错了"
                         % ["%.3f" % d for d in nn_dist])
    log("🎯 对齐基准 idx3 = %s（输入帧中心距 %s m）"
        % (idx3, ["%.3f" % d for d in nn_dist]))

    Rb = float(np.percentile(orb["r"], 90) * RADIUS_K)
    elev = float(ELEV) if ELEV else float(np.median(orb["el"]))
    log("   轨道半径 %.3f m ｜ 仰角 %.1f°" % (Rb, elev))

    th_q, span, span_full = a24.arc_angles(orb, NFRAMES, TRIM)
    eyes = a24.flat_ring(orb, th_q, Rb, elev)
    log("   ARC 轨迹：%d 帧扫过 %.1f°（观测弧 %.1f° 内）｜ 与 32 图版逐帧同轨迹" % (len(eyes), span, span_full))

    # 内参统一取 GT 的（capture npz 的 K 已是原图分辨率 1439×1919，audit_31 sanity 过），
    # 等比缩放到单格——视场保持不变
    z = np.load(GT_CAMS)
    K0 = z["intrinsics"].astype(np.float64)[0]
    # K 标定宽 ≈ 2×主点（audit_31 sanity：主点在中点附近）
    W0, H0 = float(K0[0, 2] * 2), float(K0[1, 2] * 2)
    CELLH = int(round(CELLW * 4 / 3))
    K = K0.copy()
    K[0, 0] *= CELLW / W0
    K[1, 1] *= CELLH / H0
    K[0, 2] = (CELLW - 1) / 2.0
    K[1, 2] = (CELLH - 1) / 2.0
    log("🖼  单格 %d×%d ｜ fx %.1f fy %.1f（原图 K 等比缩，统一内参）" % (CELLW, CELLH, K[0, 0], K[1, 1]))

    W2C = [torch.from_numpy(a24.look_at(e, orb["target"], orb["axis"])).float().to(dev)[None]
           for e in eyes]
    Kt = torch.from_numpy(K).float().to(dev)[None]
    bg = torch.ones(3, device=dev) if BG == "white" else torch.zeros(3, device=dev)

    # ── 画布：顶部标题 + 五格（版式与 audit_24 一致）──
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
    title = ("同一相机轨迹 · 252.7° 观测弧 · 佳佳抱枕（仅 3 视角输入）· "
             "五方法逐帧对照（统一位姿与内参，与 32 视角版同轨迹）")
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

    # 阶段 1：逐方法「载入→对齐→渲染→释放」
    frames = []
    for c in CASES:
        if not Path(c["ply"]).exists():
            raise SystemExit("❌ 缺产物 %s（对应步骤没跑完）" % c["ply"])
        g = a24.read_ply(c["ply"])
        Cx, dx, upx = a24.read_cams(*c["cams"])
        if len(Cx) != 3:
            raise SystemExit("❌ %s: 相机数 %d != 3（预期三图预测相机）" % (c["name"], len(Cx)))
        # 三对点 Umeyama：方法 3 相机 → GT 的 idx3 三个相机中心
        s, R, t = a24.umeyama(Cx, Cref[idx3])
        C2 = (s * (R @ Cx.T)).T + t
        rms = float(np.sqrt(((C2 - Cref[idx3]) ** 2).sum(1).mean()))
        rel = rms / float(np.median(np.linalg.norm(Cref - Cref.mean(0), axis=1)))
        d2 = (R @ dx.T).T
        ang = float(np.median(np.degrees(np.arccos(np.clip(
            np.sum(d2 * dref[idx3], 1), -1, 1)))))
        u2 = (R @ upx.T).T
        uang = float(np.median(np.degrees(np.arccos(np.clip(
            np.sum(u2 * upref[idx3], 1), -1, 1)))))
        flag = "OK" if (rel < 0.03 and ang < 5 and uang < 5) else "FAIL"
        log("   %-15s %7d 点 SH%d ｜ s=%.4f 残差 %.2f%% 视线 %.2f° up %.2f° [%s]"
            % (c["name"], g["n"], g["deg"], s, rel * 100, ang, uang, flag))
        if flag != "OK":
            raise SystemExit("❌ %s 三对点对齐没过判据，别往下渲" % c["name"])
        g = a24.warp(g, s, R, t)
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
            # 前 4 帧逐帧日志：再活锁也是几分钟暴露，不 silent 挂一晚
            if i < 4:
                log("      frame %3d  α 中位 %.3f" % (i + 1, amed))
            elif (i + 1) % 60 == 0:
                log("      frame %3d/%d  α 中位 %.3f" % (i + 1, NFRAMES, amed))
        frames.append(buf)
        del G
        torch.cuda.empty_cache()
        log("      ✅ %s 渲完，α 中位 %.3f，显存峰值 %.1f GB"
            % (c["name"], amed, torch.cuda.max_memory_allocated() / 1e9))
        torch.cuda.reset_peak_memory_stats()

    # 阶段 2：逐帧拼接 + 编码
    mp4 = os.path.join(OUT, "five_way_arc_3view.mp4")
    cmd = [a24.ffmpeg_exe(), "-y", "-loglevel", "error",
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
        fr.save(os.path.join(OUT, "five_way_arc_3view_frame_%03d.png" % i))
    log("✅ %s（%.1f MB）" % (mp4, os.path.getsize(mp4) / 1e6))


if __name__ == "__main__":
    main()
