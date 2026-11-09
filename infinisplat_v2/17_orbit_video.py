#!/usr/bin/env python3
"""把重建的 scene.ply 沿水平轨道环绕一圈渲染成 MP4。

与 compare_4methods/audit_22_orbit_video.py 同思路，但轨道/位姿直接复用
16_extrap_eval.py 里已对该场景标定过的那套（视线交汇球心 + 自检过的
make_c2w 约定）—— 那套位姿构造器是拿 32 台观测相机对拍验证过的（残差 p50 8.1°），
不要在这里另写一份 look-at（手写 cross 顺序的坑已经踩过两轮）。

轨道：水平环，仰角取观测俯仰中位，半径取观测半径中位（× --radius-k），
注视点 = 观测相机光轴最小二乘交点（同 16，不能用相机中心均值）。

⚠️ 方位只覆盖 ~201.5°（E-36 实测，背面 158.5° 没拍过）。默认走满 360°：
走进缺口段会看到拖影/半透明，那是 PLY 本身没数据，不是渲染 bug。
只想看"能看的部分"就 --arc，只在有观测的方位跨度内往返。

用法:
  python 17_orbit_video.py --result ~/output/.../jiajia --out <输出目录>
  python 17_orbit_video.py --result ... --tag ft --arc          # 只走观测弧
"""
import argparse
import json
import subprocess
import sys
import time
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_check import load_ply  # noqa: E402

_ex = import_module("16_extrap_eval")   # orbit_fit / orbit_basis / to_azel / make_c2w


def rasterize(g, w2c, Km, W, H, eps2d, background, dev):
    """张量常驻 GPU 的单帧渲染 —— 不能每帧调 render_check.render：
    那个函数每次调用都会把 5.9M 高斯重新 H2D 拷一遍（~320 MB/帧，240 帧 = 77 GB）。"""
    from gsplat import rasterization
    rgb, alpha, _ = rasterization(
        means=g["means_t"], quats=g["quats_t"], scales=g["scales_t"],
        opacities=g["opacities_t"], colors=g["colors_t"],
        viewmats=w2c, Ks=Km, width=W, height=H,
        sh_degree=0, eps2d=eps2d, near_plane=0.01, far_plane=200.0,
        backgrounds=background, render_mode="RGB")
    return rgb[0].clamp(0, 1), alpha[0]


def ffmpeg_exe():
    from shutil import which
    p = which("ffmpeg")
    if p:
        return p
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True, help="含 scene.ply / cameras.npz / result.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="orbit")
    ap.add_argument("--nframes", type=int, default=240)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--outw", type=int, default=540, help="输出宽度（高度按比例）")
    ap.add_argument("--radius-k", type=float, default=1.0, help="半径倍率（<1 推进 / >1 拉远）")
    ap.add_argument("--elev", type=float, default=None,
                    help="轨道仰角(度)，默认观测俯仰中位 —— 用户要\"水平\"就传 0")
    ap.add_argument("--arc", action="store_true", help="只走有观测的方位跨度（跳过缺口）")
    ap.add_argument("--eps2d", type=float, default=0.01)
    ap.add_argument("--background", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    res = Path(args.result)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cams = np.load(res / "cameras.npz", allow_pickle=False)
    c2w = np.array(cams["c2w"], np.float64)
    K = np.array(cams["intrinsics"], np.float64)
    meta = json.loads((res / "result.json").read_text())

    center, up, rad = _ex.orbit_fit(c2w)
    e1, e2 = _ex.orbit_basis(center, up, c2w)
    azel = np.array([_ex.to_azel(center, up, e1, e2, M[:3, 3]) for M in c2w])
    r_p50, r_max, pos_rms = _ex.calibrate_pose_builder(c2w, center, up, e1, e2)
    az_sorted = np.sort(azel[:, 0])
    gaps = np.diff(np.concatenate([az_sorted, [az_sorted[0] + 360.0]]))
    biggest_gap = float(gaps.max())

    elev = float(args.elev) if args.elev is not None else float(np.median(azel[:, 1]))
    radius = rad * args.radius_k

    print(f"🛰️  [17] 轨道 | 球心 {np.round(center, 4).tolist()}  半径 {radius:.4f}"
          f"（观测 p50 {rad:.4f} × {args.radius_k}）  仰角 {elev:+.1f}°"
          f"（观测 {azel[:, 1].min():+.1f}~{azel[:, 1].max():+.1f}°）")
    print(f"   位姿自检 p50 {r_p50:.2f}°（阈值 25°）  方位空洞 {biggest_gap:.1f}°")
    if r_p50 > 25.0:
        sys.exit("❌ 位姿构造器自检不通过 —— 别往下渲")

    wh = meta["inputs"][0].get("processed_wh", [392, 518])
    W0, H0 = int(wh[0]), int(wh[1])
    kw = args.outw / W0
    H = int(round(H0 * kw))
    H += H % 2
    W = args.outw
    Km = np.median(K, axis=0).copy()
    Km[0, 0] *= W / W0
    Km[1, 1] *= H / H0
    Km[0, 2] = (W - 1) / 2.0            # 轨道视图没有原始主点偏置，放正中
    Km[1, 2] = (H - 1) / 2.0

    g = load_ply(res / "scene.ply")
    n_pts = len(g["means"])
    print(f"📦 PLY {n_pts:,} 点 ｜ 输出 {W}×{H} ｜ fx {Km[0,0]:.1f}")

    dev = args.device
    gt = {k: torch.from_numpy(np.ascontiguousarray(g[k])).to(dev)
          for k in ("means", "quats", "scales", "opacities", "sh")}
    G = dict(means_t=gt["means"], quats_t=gt["quats"], scales_t=gt["scales"],
             opacities_t=gt["opacities"],
             colors_t=gt["sh"][None, :, None, :].contiguous())   # [1,N,1,3]，K=1 对应 SH0
    Km_t = torch.from_numpy(Km.astype(np.float32)).to(dev)[None]
    bg = torch.full((1, 3), float(args.background), device=dev)

    if args.arc:
        gap_lo = float(az_sorted[int(np.argmax(gaps))])
        start = (gap_lo + biggest_gap) % 360.0        # 缺口高端 = 观测弧起点
        span = 360.0 - biggest_gap
        print(f"🎞  ARC 模式：只走观测弧 {span:.1f}°（{start:.0f}° 起，缺口段跳过）")
        az_list = [start + span * i / max(args.nframes - 1, 1) for i in range(args.nframes)]
    else:
        print(f"🎞  360° 模式：满一圈；{biggest_gap:.0f}° 缺口段会露出拖影（PLY 本来没那块数据）")
        az_list = [360.0 * i / args.nframes for i in range(args.nframes)]

    mp4 = out / f"{args.tag}.mp4"
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(args.fps), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
           "-movflags", "+faststart", str(mp4)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    t0 = time.time()
    keep_at = {0, args.nframes // 4, args.nframes // 2, 3 * args.nframes // 4}
    for i, az in enumerate(az_list):
        M = _ex.make_c2w(center, up, e1, e2, az, elev, radius)
        w2c = torch.from_numpy(np.linalg.inv(M).astype(np.float32)).to(dev)[None]
        with torch.no_grad():
            rgb, alpha = rasterize(G, w2c, Km_t, W, H, args.eps2d, bg, dev)
        frame = (rgb.mul(255).byte().cpu().numpy())
        proc.stdin.write(frame.tobytes())
        if i in keep_at:
            Image.fromarray(frame).save(out / f"{args.tag}_frame_{i:03d}.png")
        if (i + 1) % 60 == 0 or i == 0:
            print(f"   frame {i + 1:3d}/{args.nframes}  α 中位 {float(alpha.median()):.3f}"
                  f"  ({(time.time() - t0):.1f}s)", flush=True)

    proc.stdin.close()
    if proc.wait() != 0:
        sys.exit("❌ ffmpeg 编码失败")
    print(f"✅ {mp4}（{args.nframes} 帧 / {args.fps} fps / {args.nframes/args.fps:.1f}s"
          f" / {mp4.stat().st_size/1e6:.1f} MB / 渲染耗时 {time.time()-t0:.1f}s）")


if __name__ == "__main__":
    main()
