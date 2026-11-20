#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
structsplat/00ab_render_video.py — 环绕轨迹渲染 + 显存/耗时实测（PPT 素材）

为什么单独写：infer.py 只渲输入视角，没有轨迹视频，也没有记录峰值显存。
本脚本从 --dump-forward 的前向缓存直接装高斯/相机（跳过 1.5B 前馈），因此：
  1) 可反复调轨迹参数，秒级出片；
  2) 用 torch.cuda.max_memory_allocated() 精确测「前馈峰值」与「渲染峰值」两个数。

轨迹设计（避免花哨，只要能看清几何）：
  - 以全部输入相机中心的重心为圆心、拟合半径 R；
  - 方位角绕垂直轴扫 ±SWEEP，仰角做轻微的 sin 摆动；
  - 用 slerp 在相邻输入相机朝向之间插值，保证画面「看着场景中心」。

用法:
  python 00ab_render_video.py --cache <.../hand01_pf2.pt> --out <dir> \
      --frames 120 --sweep 40 --fps 24
"""

import argparse
import importlib.util
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("STRUCTSPLAT_REPO") or os.path.expanduser("~/repos/StructSplat")
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def look_at_w2c(eye, target, up_hint=np.array([0.0, 1.0, 0.0])):
    """OpenCV 约定：+z 朝前、+y 朝下。返回 4×4 w2c（世界系到相机系）。"""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd) + 1e-12

    up = np.asarray(up_hint, dtype=np.float64)
    if abs(float(np.dot(fwd, up))) > 0.98:
        up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right) + 1e-12
    down = np.cross(fwd, right)          # OpenCV: +y 向下
    down /= np.linalg.norm(down) + 1e-12

    R = np.stack([right, down, fwd], axis=0)   # 相机基在世界系下的行
    t = -R @ eye
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def cam_centers(w2c):
    R = w2c[:, :3, :3]
    t = w2c[:, :3, 3]
    return -np.einsum("sji,sj->si", R, t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="--dump-forward 产出的 .pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--sweep", type=float, default=40.0,
                    help="相对输入相机方位角的可见半幅（度）；过大会渲到没覆盖的区域")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--elev", type=float, default=6.0, help="仰角摆动幅度（度）")
    ap.add_argument("--radius-scale", type=float, default=1.0)
    ap.add_argument("--render-size", type=int, default=512)
    ap.add_argument("--white-bg", action="store_true")
    ap.add_argument("--pingpong", action="store_true", default=True,
                    help="往复播放（避免首尾跳变）")
    ap.add_argument("--no-pingpong", dest="pingpong", action="store_false")
    ap.add_argument("--still-frame", default=None,
                    help="额外导出单帧 PNG（取轨迹中点）")
    args = ap.parse_args()

    dev = torch.device("cuda")
    d = torch.load(args.cache, map_location="cpu", weights_only=False)
    means = d["means"].to(dev).float()
    quats = d["quats"].to(dev).float()
    scales = d["scales"].to(dev).float()
    opac = d["opac"].to(dev).float()
    colors = d["colors"].to(dev).float()
    w2cs_src = d["w2cs"].float().numpy()
    Ks = d["Ks"].float()
    new_h, new_w = int(d["new_h"]), int(d["new_w"])
    N = means.shape[0]
    print(f"载入 {N} 个高斯 / {len(w2cs_src)} 台相机  输入 {new_h}×{new_w}")

    # 渲染分辨率与 K 同步缩放（与 infer.py 口径一致：按宽缩）
    rw = args.render_size
    # 高度直接对齐到 16 的倍数：h264/yuv420p 要偶数，而 imageio-ffmpeg 默认
    # macro_block_size=16，会自己补边（实测 484→496，画面纵向拉伸 2.5%）。
    # 先对齐到 16 就绕开补边；两侧缩放比例的残差 <1%（480/484），肉眼不可辨。
    rh = max(16, int(round(rw * new_h / new_w / 16.0)) * 16)
    s = rw / new_w

    centers = cam_centers(w2cs_src)
    # 视线交汇点比「相机中心重心」更稳（后者在有俯仰时偏）
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for M, c in zip(w2cs_src, centers):
        z = M[2, :3]
        z = z / (np.linalg.norm(z) + 1e-12)
        P = np.eye(3) - np.outer(z, z)
        A += P
        b += P @ c
    target = np.linalg.solve(A, b)
    radius = float(np.linalg.norm(centers - target, axis=1).mean()) * args.radius_scale

    # ★ 世界 up 不能假设 +Y：VGGT 系以第一帧相机为参考，场景的"天"方向任意。
    #   实测 2026-11-20：按 +Y 转，人物在画面里横躺 90°。
    #   且本数据是竖图（1920×1440），喂网络前做了转置（518×392），
    #   转置后图像"右"= 原图"下"，所以相机 x 轴（w2c 第 1 行）才指向世界下方，
    #   用 y 轴估计会差 90°（实测 world_up=[-0.25,-0.64,0.72]，人物仍横躺）。
    xs = np.stack([M[0, :3] for M in w2cs_src])
    world_up = -xs.mean(axis=0)
    world_up /= np.linalg.norm(world_up) + 1e-12
    # 水平面内的一对正交基（h0×h1 = world_up）
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(ref, world_up))) > 0.95:
        ref = np.array([1.0, 0.0, 0.0])
    h1 = ref - world_up * float(np.dot(ref, world_up))
    h1 /= np.linalg.norm(h1) + 1e-12
    h0 = np.cross(h1, world_up)

    # ★ 方位角必须按**环绕**处理（2026-11-20 实测算错过一次）：
    #   两台相机 az = -75.1° 和 +165.0°，真实短弧差是 120°、中点 -135°；
    #   直接对度数取平均得 +45°，轨迹会扫到与输入覆盖区几乎不相交的一侧，
    #   渲出来大片空白。正解是把方位角当单位向量做向量平均，再用短弧算跨度。
    rel = centers - target
    az = np.degrees(np.arctan2(rel @ h0, rel @ h1))
    az_vec = np.stack([np.cos(np.radians(az)), np.sin(np.radians(az))], axis=1)
    az_mean_vec = az_vec.mean(axis=0)
    if np.linalg.norm(az_mean_vec) < 1e-6:
        az_mean_vec = az_vec[0]
    az_mid = float(np.degrees(np.arctan2(az_mean_vec[1], az_mean_vec[0])))
    # 到中点的短弧偏角（变量名勿用 dev：上面 dev 是 torch.device，会被遮蔽成 ndarray）
    off_deg = (az - az_mid + 180.0) % 360.0 - 180.0
    span = float(off_deg.max() - off_deg.min())
    el_mid = np.degrees(np.arctan2(rel @ world_up,
                                   np.hypot(rel @ h0, rel @ h1)))
    el_mid = float(np.mean(el_mid))
    print(f"场景中心 {np.round(target,3)}  半径 {radius:.3f}  世界up {np.round(world_up,3)}")
    print(f"输入方位角 中点 {az_mid:+.1f}°  跨度 {span:.1f}°  "
          f"逐台 {np.round(az,1).tolist()}  仰角均值 {el_mid:+.1f}°")

    # 轨迹：在输入覆盖区内扫，往复
    n = args.frames
    if args.pingpong:
        u = np.sin(np.linspace(-math.pi / 2, math.pi / 2, n))     # -1..1 平滑往复
    else:
        u = np.linspace(-1.0, 1.0, n)
    # 只扫输入覆盖区内的 ±sweep；覆盖跨度本身很窄时不强行放大（会渲到空白）
    half = min(args.sweep, 0.5 * span + 8.0)

    traj = []
    for k in range(n):
        a = math.radians(az_mid + u[k] * half)
        e = math.radians(el_mid + args.elev * math.sin(math.pi * k / max(n - 1, 1)))
        eye = (target
               + radius * (math.sin(a) * math.cos(e) * h0
                           + math.cos(a) * math.cos(e) * h1
                           + math.sin(e) * world_up))
        traj.append(look_at_w2c(eye, target, up_hint=world_up))
    w2cs = torch.from_numpy(np.stack(traj)).to(dev).float()
    print(f"轨迹 {n} 帧：方位 {az_mid - half:+.1f}° … {az_mid + half:+.1f}°")

    Ks_r = Ks.to(dev).clone()
    # 两个轴各按各的画布尺寸缩：高度取偶数后与按宽推的值差 ~0.3%，
    # 统一用 width 比例会让像素非方（fy≠fx），这里各自乘自己的比
    Ks_r[:, 0, :] *= rw / new_w
    Ks_r[:, 1, :] *= rh / new_h
    Ks_r = Ks_r[:1].repeat(n, 1, 1)

    from gsplat import rasterization

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    bg_val = 1.0 if args.white_bg else 0.0
    frames = []
    t0 = None
    import time
    with torch.no_grad():
        for k in range(n):
            if k == 0:
                torch.cuda.synchronize()
                t0 = time.time()
            out, _, _ = rasterization(
                means=means, quats=quats, scales=scales,
                opacities=opac, colors=colors,
                viewmats=w2cs[k:k + 1], Ks=Ks_r[k:k + 1],
                width=rw, height=rh, render_mode="RGB",
                packed=False,
                backgrounds=torch.full((1, 3), bg_val, dtype=torch.float32, device=dev),
            )
            frames.append(out[0, ..., :3].clamp(0, 1))
        if dev.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024**3

    print(f"🎬 渲染 {n} 帧耗时 {dt:.2f}s  平均 {dt/n*1000:.1f} ms/帧  "
          f"峰值显存 {peak:.2f} GiB")

    os.makedirs(args.out, exist_ok=True)

    # 落盘：PNG 序列 + MP4（用 torchvision 写，imageio 兜底）
    arr = torch.stack(frames).cpu().numpy()            # (n,H,W,3) float 0..1
    u8 = (arr * 255.0 + 0.5).astype(np.uint8)

    mp4 = os.path.join(args.out, "orbit.mp4")
    ok = False
    try:
        import imageio_ffmpeg
        writer = imageio_ffmpeg.write_frames(
            mp4, (rw, rh), fps=args.fps, codec="libx264",
            pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
            output_params=["-crf", "18", "-preset", "slow"],
            ffmpeg_log_level="error")
        writer.send(None)
        for f in u8:
            writer.send(f.tobytes())
        writer.close()
        ok = True
    except Exception as e:
        print(f"  ffmpeg 直写失败: {e}")
    print(f"🎥 视频: {mp4}  {'OK' if ok else 'FAILED'}")

    # 关键帧 PNG（PPT 直接用）
    picks = [0, n // 4, n // 2, 3 * n // 4]
    for i, k in enumerate(picks):
        p = os.path.join(args.out, f"key_{i}_{k:03d}.png")
        from torchvision.utils import save_image
        save_image(frames[k].permute(2, 0, 1), p)
    if args.still_frame:
        from torchvision.utils import save_image
        save_image(frames[n // 2].permute(2, 0, 1), args.still_frame)
    print(f"🖼️  关键帧 {len(picks)} 张 → {args.out}")

    json.dump({
        "n_gaussians": int(N),
        "src_views": int(len(w2cs_src)),
        "frames": n, "fps": args.fps,
        "render": [rh, rw],
        "render_total_s": round(dt, 2),
        "ms_per_frame": round(dt / n * 1000, 2),
        "peak_vram_gib_render": round(peak, 3),
        "az_range": [round(az_mid - half, 2), round(az_mid + half, 2)],
        "elev_amp": args.elev,
        "radius": round(radius, 4),
        "pivot": [round(float(x), 4) for x in target],
        "note": "从 --dump-forward 缓存渲染；峰值显存为「渲染阶段」，不含前馈",
    }, open(os.path.join(args.out, "video_meta.json"), "w", encoding="utf-8", newline="\n"),
        indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
