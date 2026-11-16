#!/usr/bin/env python3
"""用 gsplat 绕场景中心渲一圈轨道视频（mp4），输入是 02 的输出目录。

相机轨迹：以高斯质心（按 opacity 加权）为中心、源相机到中心的中位距离为半径，
水平 360° 均匀采 N 帧；俯仰角默认跟源相机一致（相对中心的仰角取中位），
up _hint 用源相机的 Y 轴均值（保持与输入一致的 roll）。

为什么不复用 99b：99b 走 UWA MP4 封装链（需要 gs_camera_params.json + 收窄视角），
是给重建产物打包用的；这里 cameras.json 本来就是同一坐标系的 OpenCV c2w（见
03_render_eval.py 头注释），直接 inv(c2w) 渲，不需要 Sim3 对齐。

Args:
  --result    02 的输出目录（含 gaussians.ply / cameras.json / rgb/）
  --out       输出 mp4 路径（默认 RESULT/orbit.mp4）
  --frames    帧数，默认 120
  --fps       帧率，默认 30
  --radius-scale  轨道半径 = 中位距离 × 此系数，默认 1.0
  --elevation-deg 固定俯仰角（度，高于水平面为正）；缺省 = 源相机中位仰角
  --eps2d     屏幕空间膨胀，默认 0.01（与 03 一致）
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np

# 03 里的 load_ply / load_cameras / render 直接复用（脚本名以数字开头，importlib 导入）
sys.path.insert(0, str(Path(__file__).resolve().parent))
m03 = importlib.import_module("03_render_eval")


def look_at_c2w(eye, target, y_hint):
    """OpenCV 约定（X 右 / Y 下 / Z 前）的 look-at c2w。y_hint = 世界"下"方向近似。"""
    z = target - eye
    z = z / (np.linalg.norm(z) + 1e-12)          # 前
    x = np.cross(y_hint, z)
    x = x / (np.linalg.norm(x) + 1e-12)          # 右
    y = np.cross(z, x)                           # 下（正交化后的）
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2] = x, y, z
    c2w[:3, 3] = eye
    return c2w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--frames', type=int, default=120)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--radius-scale', type=float, default=1.0)
    ap.add_argument('--elevation-deg', type=float, default=None)
    ap.add_argument('--eps2d', type=float, default=0.01)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    res = Path(args.result)
    out = Path(args.out) if args.out else res / 'orbit.mp4'
    out.parent.mkdir(parents=True, exist_ok=True)

    g = m03.load_ply(res / 'gaussians.ply')
    c2w_src, K_src, ids = m03.load_cameras(res / 'cameras.json')
    print(f'🚀 轨道渲染 | 高斯 {len(g["means"]):,} | 源视角 {len(ids)} | {args.frames} 帧')

    # 轨道参数：中心 = opacity 加权质心（比相机中点更贴被摄主体）
    w = g['opacities'].astype(np.float64)
    center = (g['means'] * w[:, None]).sum(0) / w.sum()
    eyes_src = c2w_src[:, :3, 3]
    rel = eyes_src - center
    dist = np.linalg.norm(rel, axis=1)
    radius = float(np.median(dist)) * args.radius_scale

    # 俯仰角：源相机相对中心的仰角中位（deg）；up_hint = 源相机 Y 轴均值
    elev = args.elevation_deg
    if elev is None:
        elev = float(np.degrees(np.median(np.arcsin(rel[:, 1] / (dist + 1e-12)))))
    y_hint = c2w_src[:, :3, 1].mean(0)
    if np.linalg.norm(y_hint) < 1e-6:
        y_hint = np.array([0.0, 1.0, 0.0])
    print(f'  🎯 中心 {np.round(center, 4)}  半径 {radius:.4f}  俯仰 {elev:.1f}°')

    # 方位角起点取第一个源相机的方位角，让视频从"输入视角附近"起转
    az0 = float(np.degrees(np.arctan2(rel[0, 0], rel[0, 2])))

    K = K_src[0]
    from PIL import Image
    with Image.open(res / 'rgb' / f'{ids[0]}.png') as im:
        w_px, h_px = im.size
    # 背景与 03 同口径：GT 四边像素中位数（orbit 没有 GT，借第一张的）
    first = np.asarray(Image.open(res / 'rgb' / f'{ids[0]}.png').convert('RGB'), np.float32) / 255.0
    bg = float(np.median(np.concatenate([first[0], first[-1], first[:, 0], first[:, -1]], axis=0)))

    import cv2
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (w_px, h_px))
    if not vw.isOpened():
        raise SystemExit(f'❌ cv2.VideoWriter 打不开 {out}')

    batch = 20  # gsplat 一次渲一批视角，比逐帧快
    c2w_all = []
    for i in range(args.frames):
        az = np.radians(az0 + 360.0 * i / args.frames)
        el = np.radians(elev)
        eye = center + radius * np.array([np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)])
        c2w_all.append(look_at_c2w(eye, center, y_hint))
    c2w_all = np.stack(c2w_all)

    for s in range(0, args.frames, batch):
        chunk = c2w_all[s:s + batch]
        rd, _al = m03.render(g, chunk, np.repeat(K[None], len(chunk), 0), w_px, h_px,
                             eps2d=args.eps2d, background=bg, device=args.device)
        for j in range(len(chunk)):
            vw.write(cv2.cvtColor((rd[j] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        print(f'  🎬 {min(s + batch, args.frames)}/{args.frames} 帧')
    vw.release()

    sz = out.stat().st_size / 1e6
    print(f'🎉 Done: {out} ({sz:.1f} MB)')
    meta = dict(result=str(res), out=str(out), frames=args.frames, fps=args.fps,
                resolution=[w_px, h_px], radius=radius, elevation_deg=elev,
                center=[float(v) for v in center], background=bg, eps2d=args.eps2d)
    out.with_suffix('.orbit.json').write_text(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
