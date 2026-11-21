#!/usr/bin/env python3
"""同一三图输入的 ZipSplat / SplatAnything 单向 360° 环绕对比。

与 eval_3view_r512_black 相同：原始 0/40/81 三输入、中心方形裁切、GT K、黑底、
只用三输入图做 Umeyama + Sim(3) 光度精化，留出图不进入对齐。视频轨迹则在
32 个 GT 相机定义的公共世界里生成，是目检轨迹，不是独立留出定量评测。

轨迹：注视点取 32 条视线的最小二乘交点，固定半径/仰角，从 GT cam0 的方位
开始单向 360°，不是官方 viz.turntable(sweep_deg=None) 的来回摆动。

WSL（querysplat env）：
  OUT=~/output/zipsplat_results/jiajia_3v_orbit_360 \
    NFRAMES=144 FPS=24 CELLW=368 python zipsplat/render_3view_orbit.py
  SMOKE=1 NFRAMES=3 ...  # 只做首/中/末帧，不覆盖正式 MP4

产物：3 个单独 MP4、3 联 MP4、首/中/末 PNG、相机与逐帧覆盖 JSON。
"""
import json
import os
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from gsplat import rasterization

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "compare_4methods"))
import audit_24_five_way_video as a24
from audit_31_eval_3view import read_cams_any
from sim3_refine import refine_sim3

O = Path(os.path.expanduser("~/output"))
OUT = Path(os.environ.get("OUT", O / "zipsplat_results/jiajia_3v_orbit_360")).expanduser()
GT_CAMS = O / "da3_results/_captures/佳佳抱枕/cameras.npz"
BASE3 = O / "infinisplat_v2_results/_captures/佳佳抱枕_3v/cameras.npz"
INPUT_IMAGES = BASE3.parent / "images"
NFRAMES = int(os.environ.get("NFRAMES", "144"))
FPS = int(os.environ.get("FPS", "24"))
CELLW = int(os.environ.get("CELLW", "368"))
SMOKE = os.environ.get("SMOKE", "") == "1"
FRAMES = [0, NFRAMES // 2, NFRAMES - 1] if SMOKE else list(range(NFRAMES))
CASES = [
    dict(name="SplatAnything", meta="三图 / pose 模式 / 深度反投影 + 体素合并",
         ply=O / "splatanything_results/eval381/jiajia_381/gaussians.ply",
         cams=("sa_json", O / "splatanything_results/eval381/jiajia_381/cameras.json"),
         color="#2F6BB5"),
    dict(name="ZipSplat FF", meta="三图 / 无位姿 / scene token 前馈",
         ply=O / "zipsplat_results/jiajia_3v_compare_feedforward/gaussians_feedforward.ply",
         cams=("c2w", O / "zipsplat_results/jiajia_3v_compare_feedforward/predicted_cameras.npz"),
         color="#237584"),
    dict(name="ZipSplat + TokenBA", meta="三图 / 无位姿 / token + 相机测试时优化",
         ply=O / "zipsplat_results/jiajia_3v_compare/gaussians.ply",
         cams=("c2w", O / "zipsplat_results/jiajia_3v_compare/predicted_cameras.npz"),
         color="#5A7CB7"),
]


def log(*args):
    print(*args, flush=True)


def ffmpeg_exe():
    from shutil import which
    return which("ffmpeg") or __import__("imageio_ffmpeg").get_ffmpeg_exe()


def load_common():
    """复刻 audit_31 的方形裁切/K，不重新实现另一套视场。"""
    c2w = np.load(GT_CAMS)["c2w"].astype(np.float64)
    Cref, dref, upref = a24.read_cams("c2w", str(GT_CAMS))
    K = np.load(GT_CAMS)["intrinsics"].astype(np.float64)[0].copy()
    first = cv2.imread(str(INPUT_IMAGES / sorted(p.name for p in INPUT_IMAGES.iterdir())[0]))
    h, w = first.shape[:2]
    assert (w, h) == (1439, 1919), (w, h)
    cx, cy = K[0, 2], K[1, 2]
    side = int(round(2 * min(cx, w - cx, cy, h - cy)))
    x0 = max(0, min(int(round(cx - side / 2)), w - side))
    y0 = max(0, min(int(round(cy - side / 2)), h - side))
    assert (side, x0, y0) == (1435, 0, 235), (side, x0, y0)
    K[0, 2] -= x0
    K[1, 2] -= y0
    K *= CELLW / side
    K[2, 2] = 1.0
    in_refs = sorted(p for p in INPUT_IMAGES.iterdir() if p.suffix.lower() in (".png", ".jpg"))
    assert [p.name for p in in_refs] == [
        "000_5093309607791922.png", "001_5093317707502922.png", "002_5093325907213922.png"]
    # 对齐函数内部负责缩到 512²。不能在这里预先降采样再让它缩第二次：
    # 像素值虽可相同，CUDA resize/梯度路径不同，实测 SA 精化会从 22.14 掉到 9.00 dB。
    in_rgb = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)[y0:y0 + side, x0:x0 + side]
              for p in in_refs]
    w2c_in = np.linalg.inv(np.load(BASE3)["c2w"].astype(np.float64))
    return c2w, Cref, dref, upref, K, in_rgb, w2c_in


def make_orbit(c2w, Cref, dref, upref):
    """单向完整环，不让 32 相机决定 360° 都有观测。"""
    orb = a24.fit_orbit(Cref, dref, upref)
    assert orb["up_ang"] < 30, orb["up_ang"]
    radius = float(np.percentile(orb["r"], 90) * 1.06)
    elev = float(np.median(orb["el"]))
    raw0 = float(np.arctan2((Cref[0] - orb["Cc"]) @ orb["e2"],
                            (Cref[0] - orb["Cc"]) @ orb["e1"]))
    th = raw0 + 2 * np.pi * np.arange(NFRAMES) / max(NFRAMES - 1, 1)
    eyes = a24.flat_ring(orb, th, radius, elev)
    w2cs = [a24.look_at(eye, orb["target"], orb["axis"]) for eye in eyes]
    # 首帧 roll/aim 与真实 cam0 作诊断；不据此断言所有环绕帧都是实测相机。
    delta = w2cs[0][:3, :3].T @ np.linalg.inv(c2w[0])[:3, :3]
    angle = float(np.degrees(np.arccos(np.clip((np.trace(delta) - 1) / 2, -1, 1))))
    log(f"🎥 单向 360°：半径 {radius:.3f} m / 仰角 {elev:.1f}°；首帧 GT cam0 旋转差 {angle:.2f}°")
    return w2cs, dict(direction="monotonic_positive", sweep_deg=360.0, closed_endpoint=True,
                      radius_m=radius, elevation_deg=elev, start_azimuth_rad=raw0,
                      target=orb["target"].tolist(), axis=orb["axis"].tolist(),
                      e1=orb["e1"].tolist(), e2=orb["e2"].tolist(), cam0_roll_aim_error_deg=angle)


def aligned_gaussians(case, in_rgb, w2c_in):
    """读取真实 PLY；source-only 精化后，给出与评分一致的公共 GT 系高斯。"""
    g = a24.read_ply(str(case["ply"]))
    Cx, dx, _ = read_cams_any(*case["cams"])
    Cbase, _, _ = a24.read_cams("c2w", str(BASE3))
    Cref, _, _ = a24.read_cams("c2w", str(GT_CAMS))
    idx3 = [int(np.argmin(((Cref - c) ** 2).sum(1))) for c in Cbase]
    assert idx3 == [0, 7, 31], idx3
    s, R, t = a24.umeyama(Cx, Cref[idx3])
    # audit_31 固定 512² refine 与原图裁切 K；视频 CELLW 只缩渲染 K，不改对齐。
    K0 = np.load(GT_CAMS)["intrinsics"].astype(np.float64)[0].copy()
    K0[0, 2] -= 0.0
    K0[1, 2] -= 235.0
    expected = {"SplatAnything": (12.935768621162515, 22.138206320121053, 3.087904930114746),
                "ZipSplat FF": (14.3982, 20.6387, 2.81),
                "ZipSplat + TokenBA": (13.8045, 21.3214, 2.90)}[case["name"]]
    info = refine_sim3(g, w2c_in, K0, in_rgb, (s, R, t), res=512, iters=200,
                       log=log, bg_value=0.0)
    assert abs(info["psnr_pre"] - expected[0]) < 0.03, (case["name"], "pre", info["psnr_pre"])
    assert abs(info["psnr_post"] - expected[1]) < 0.03, (case["name"], "post", info["psnr_post"])
    assert abs(info["delta_w_deg"] - expected[2]) < 0.04, (case["name"], "rotation", info["delta_w_deg"])
    assert 0.98 < info["delta_s"] < 1.02, (case["name"], "scale", info["delta_s"])
    log(f"🔧 {case['name']}: s0={s:.7f} | Kcx={K0[0,2]:.3f} Kcy={K0[1,2]:.3f} "
        f"source-only {info['psnr_pre']:.3f} → {info['psnr_post']:.3f} dB")
    g = a24.warp(g, info["s"], info["R"], info["t"])
    return g, {k: v for k, v in info.items() if k not in ("R", "t")}, idx3


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    for c in CASES:
        assert c["ply"].is_file() and Path(c["cams"][1]).is_file(), c["name"]
    c2w, Cref, dref, upref, K, in_rgb, w2c_in = load_common()
    w2cs, orbit = make_orbit(c2w, Cref, dref, upref)
    w2cs = [w2cs[i] for i in FRAMES]
    np.savez(OUT / "orbit_cameras.npz", w2c=np.asarray(w2cs), c2w=np.linalg.inv(w2cs), K=K,
             frame_indices=np.asarray(FRAMES))
    with (OUT / "protocol.json").open("w", encoding="utf-8", newline="\n") as f:
        json.dump(dict(input_frames=[0, 40, 81], input_images=[p.name for p in sorted(INPUT_IMAGES.iterdir())],
                       crop="1439x1919 -> center 1435x1435 (x0=0,y0=235) -> square video",
                       background="black", alignment="3-source Umeyama + source-only Sim(3) photometric refine",
                       target_images_used_for_alignment=False, video_is_heldout_evaluation=False,
                       orbit=orbit, frames=FRAMES, nframes=NFRAMES, fps=FPS, cellw=CELLW,
                       full360_has_unobserved_views=True), f, ensure_ascii=False, indent=2)
    dev = "cuda"
    free, _ = torch.cuda.mem_get_info()
    assert free >= 4 * 2**30, "GPU 空闲不足 4 GiB，拒绝并发渲染"
    kt = torch.from_numpy(K).float().to(dev)[None]
    bg = torch.zeros(3, device=dev)
    methods, buffers = [], []
    for case in CASES:
        g, alignment, idx3 = aligned_gaussians(case, in_rgb, w2c_in)
        n_gaussians = g["n"]
        G = {k: torch.from_numpy(g[k]).to(dev) for k in ("means", "quats", "scales", "opac", "sh")}
        deg = g["deg"]
        frames, coverage = [], []
        method_dir = OUT / case["name"].replace(" ", "_").replace("+", "plus")
        method_dir.mkdir(exist_ok=True)
        for j, (idx, w2c) in enumerate(zip(FRAMES, w2cs)):
            with torch.no_grad():
                img, alpha, _ = rasterization(
                    G["means"], G["quats"], G["scales"], G["opac"], G["sh"],
                    torch.from_numpy(w2c).float().to(dev)[None], kt, CELLW, CELLW,
                    sh_degree=deg, packed=False, backgrounds=bg[None], render_mode="RGB")
            rgb = img[0].clamp(0, 1).mul(255).byte().cpu().numpy()
            cov = float((alpha[0, ..., 0] >= 0.5).float().mean())
            frames.append(rgb)
            coverage.append(dict(frame=int(idx), alpha_gt_05=cov))
            if SMOKE:
                Image.fromarray(rgb).save(method_dir / f"frame_{idx:03d}.png")
            if j == 0 or (j + 1) % 36 == 0 or j + 1 == len(FRAMES):
                log(f"  {case['name']}: {j + 1}/{len(FRAMES)} 覆盖率 {cov:.3f}")
        del G, g
        torch.cuda.empty_cache()
        buffers.append(frames)
        methods.append(dict(name=case["name"], meta=case["meta"], ply=str(case["ply"]),
                            cams=[case["cams"][0], str(case["cams"][1])], color=case["color"], n_gaussians=n_gaussians,
                            alignment=alignment, input3_idx=idx3, coverage=coverage,
                            mean_coverage=float(np.mean([r["alpha_gt_05"] for r in coverage]))))
    margin, gap, head, foot = 12, 8, 76, 44
    vw = margin * 2 + CELLW * 3 + gap * 2
    vh = head + CELLW + foot
    font = "/mnt/c/Windows/Fonts/msyh.ttc"
    bold = "/mnt/c/Windows/Fonts/msyhbd.ttc"
    fn = ImageFont.truetype(bold, 19)
    fm = ImageFont.truetype(font, 12)
    ft = ImageFont.truetype(font, 14)
    ff = ImageFont.truetype(font, 13)
    exe = ffmpeg_exe()
    if not SMOKE:
        # 保留各方法独立视频，不把三联裁切后误称为另一套渲染。
        for c, frames in zip(methods, buffers):
            safe = c["name"].replace(" ", "_").replace("+", "plus")
            proc = subprocess.Popen([exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                     "-s", f"{CELLW}x{CELLW}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                                     "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart",
                                     str(OUT / f"{safe}.mp4")], stdin=subprocess.PIPE)
            for rgb in frames:
                proc.stdin.write(rgb.tobytes())
            proc.stdin.close()
            assert proc.wait() == 0, c["name"]
    combined = None if SMOKE else subprocess.Popen([
        exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{vw}x{vh}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "18", "-movflags", "+faststart", str(OUT / "three_view_orbit_360.mp4")], stdin=subprocess.PIPE)
    for j, idx in enumerate(FRAMES):
        canvas = Image.new("RGB", (vw, vh), "white")
        dr = ImageDraw.Draw(canvas)
        title = f"同一三图 0 / 40 / 81 · 同一相机单向 360° · {idx / max(NFRAMES - 1, 1) * 360:.0f}°"
        dr.text((margin + 6, 6), title, font=ft, fill="#3B4A5B")
        for ci, (c, frames) in enumerate(zip(methods, buffers)):
            x = margin + ci * (CELLW + gap)
            dr.text((x + 4, 28), c["name"], font=fn, fill=c["color"])
            dr.text((x + 4, 53), c["meta"], font=fm, fill="#64748B")
            canvas.paste(Image.fromarray(frames[j]), (x, head))
        dr.text((margin + 4, head + CELLW + 12), "固定半径/仰角、统一内参、黑底；背面无观测，视频仅供目检，不替代 29 留出评分。",
                font=ff, fill="#96631F")
        if combined:
            combined.stdin.write(canvas.tobytes())
        if idx in (0, NFRAMES // 2, NFRAMES - 1):
            canvas.save(OUT / f"comparison_frame_{idx:03d}.png")
    if combined:
        combined.stdin.close()
        assert combined.wait() == 0, "ffmpeg 三联编码失败"
    with (OUT / "render_metrics.json").open("w", encoding="utf-8", newline="\n") as f:
        json.dump(dict(seconds=time.time() - t0, smoke=SMOKE, methods=methods), f, ensure_ascii=False, indent=2)
    log(f"🎉 {OUT} / {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
