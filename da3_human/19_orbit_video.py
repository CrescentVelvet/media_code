#!/usr/bin/env python3
"""把 DA3 高斯产物沿水平轨道环绕渲染成 MP4。

移植自 infinisplat_v2/17_orbit_video.py（2026-11-09 按峰哥指示抄入 da3_human），差异：
  1. 输入是 08_bench_official_gs.py 的产物目录（含 gs_max/0000.ply + cameras.npz）；
     npz 的 extrinsics 是 **w2c**（DA3 约定），轨道拟合前要求逆成 c2w；
  2. 渲染带**全 SH（sh_degree=2）**——max 档 PLY 有 DC+15 rest，infinisplat 版只渲 DC；
  3. load_ply 与轨道函数全部内嵌（load_ply 抄 07_eval_da3_gs.py；轨道五函数逐字移植
     infinisplat_v2/16_extrap_eval.py —— cross 顺序是拿 32 台 GT 相机对拍钉死的，勿改）。

用法:
  python 19_orbit_video.py --result <08 输出目录> --out <mp4 输出目录>
  python 19_orbit_video.py --result ... --tag orbit --arc    # 只走观测弧，跳过方位缺口
"""
import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------- load_ply（抄 07_eval_da3_gs.py）
def load_ply(path):
    with open(path, "rb") as f:
        raw = f.read(65536)
        end = raw.index(b"end_header") + len(b"end_header") + 1
        head = raw[:end].decode("ascii", "replace")
        n = int(re.search(r"element vertex (\d+)", head).group(1))
        props = [ln.split()[2] for ln in head.splitlines() if ln.startswith("property")]
        f.seek(end)
        arr = np.fromfile(f, dtype=np.float32).reshape(n, len(props))
    nm = {k: i for i, k in enumerate(props)}
    means = arr[:, [nm["x"], nm["y"], nm["z"]]]
    op = arr[:, nm["opacity"]]
    sc = arr[:, [nm["scale_0"], nm["scale_1"], nm["scale_2"]]]
    q = arr[:, [nm["rot_0"], nm["rot_1"], nm["rot_2"], nm["rot_3"]]]
    dc = arr[:, [nm["f_dc_0"], nm["f_dc_1"], nm["f_dc_2"]]]
    rest = sorted((int(k.split("_")[-1]), i) for k, i in nm.items() if k.startswith("f_rest_"))
    if rest:
        r = arr[:, [i for _, i in rest]]
        k = r.shape[1] // 3
        r = r.reshape(n, 3, k).transpose(0, 2, 1)  # channel-major → (N,K,3)
        sh = np.concatenate([dc[:, None, :], r], axis=1)
    else:
        sh = dc[:, None, :]
    return means, sh, op, sc, q


# ------------------------------------------- 轨道五函数（逐字移植 16_extrap_eval.py，勿改 cross 顺序）
def orbit_fit(c2w):
    """从观测相机的 c2w 拟合「球心 + 上轴 + 半径」。

    球心**必须用视线交汇**求：min Σ‖(I−zzᵀ)(c−p)‖² 的闭式解，等价于所有光轴的最近点。
    """
    R, cen = c2w[:, :3, :3], c2w[:, :3, 3]
    z = R[:, :, 2]
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for zi, pi in zip(z, cen):
        P = np.eye(3) - np.outer(zi, zi)
        A += P
        b += P @ pi
    if np.linalg.cond(A) < 1e6:
        center = np.linalg.solve(A, b)
    else:
        center = cen.mean(0)
        print("   ⚠️ 视线交汇病态（光轴近平行），退回相机中心均值")
    up = -R[:, :, 1].mean(0)                         # OpenCV 相机 +y 朝下 → -y 是世界"上"
    up = up / (np.linalg.norm(up) + 1e-12)
    radius = float(np.median(np.linalg.norm(cen - center, axis=1)))
    return center, up, radius


def orbit_basis(center, up, c2w):
    v = c2w[0, :3, 3] - center
    v = v - up * float(v @ up)
    e1 = v / (np.linalg.norm(v) + 1e-12)
    e2 = np.cross(up, e1)
    return e1, e2


def to_azel(center, up, e1, e2, p):
    v = p - center
    r = float(np.linalg.norm(v))
    u = v / (r + 1e-12)
    el = np.degrees(np.arcsin(np.clip(float(u @ up), -1, 1)))
    az = np.degrees(np.arctan2(float(u @ e2), float(u @ e1))) % 360.0
    return az, el, r


def make_c2w(center, up, e1, e2, az, el, radius):
    """球面 (az, el, radius) → look-at c2w（OpenCV：x右 y下 z前）。

    ⚠️ x = cross(z, up)、y = cross(z, x)——两处 cross 顺序是拿 GT 对拍钉死的，
    写成 cross(up, z) / cross(x, z) 渲染出来"有图"但内容镜像/倒置，单看发现不了。
    """
    a, e = np.radians(az), np.radians(el)
    d = np.cos(e) * (np.cos(a) * e1 + np.sin(a) * e2) + np.sin(e) * up
    p = center + radius * d
    z = center - p
    z = z / (np.linalg.norm(z) + 1e-12)
    x = np.cross(z, up)
    x = x / (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)
    M = np.eye(4, dtype=np.float64)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = x, y, z, p
    return M


def calibrate_pose_builder(c2w, center, up, e1, e2):
    """自检：观测相机按 (az,el,r) 重建 vs GT 的旋转偏差 p50。>25° 说明轨道系不可信。"""
    errs, poss = [], []
    for M in c2w:
        az, el, r = to_azel(center, up, e1, e2, M[:3, 3])
        Mb = make_c2w(center, up, e1, e2, az, el, r)
        cosang = np.clip((np.trace(Mb[:3, :3].T @ M[:3, :3]) - 1) / 2, -1, 1)
        errs.append(np.degrees(np.arccos(cosang)))
        poss.append(np.linalg.norm(Mb[:3, 3] - M[:3, 3]))
    return float(np.median(errs)), float(np.max(errs)), float(np.sqrt(np.mean(np.square(poss))))


# ---------------------------------------------------------------- 渲染
def rasterize(g, w2c, Km, W, H, eps2d, background, dev, sh_degree):
    """张量常驻 GPU 的单帧渲染，不能每帧重新 H2D 拷贝。"""
    from gsplat import rasterization
    rgb, alpha, _ = rasterization(
        means=g["means_t"], quats=g["quats_t"], scales=g["scales_t"],
        opacities=g["opacities_t"], colors=g["colors_t"],
        viewmats=w2c, Ks=Km, width=W, height=H,
        sh_degree=sh_degree, eps2d=eps2d, near_plane=0.01, far_plane=200.0,
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
    ap.add_argument("--result", required=True, help="08 产物目录（含 gs_max/0000.ply + cameras.npz）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="orbit")
    ap.add_argument("--nframes", type=int, default=240)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--outw", type=int, default=540, help="输出宽度（高度按 npz 深度图比例）")
    ap.add_argument("--radius-k", type=float, default=1.0, help="半径倍率（<1 推进 / >1 拉远）")
    ap.add_argument("--elev", type=float, default=None,
                    help="轨道仰角(度)，默认观测俯仰中位 —— 要水平就传 0")
    ap.add_argument("--arc", action="store_true", help="只走有观测的方位跨度（跳过缺口段）")
    ap.add_argument("--eps2d", type=float, default=0.01)
    ap.add_argument("--background", type=float, default=1.0)
    ap.add_argument("--sh-degree", type=int, default=2, help="DA3 max 档是全 SH(2)；DC-only 用 0")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    res = Path(args.result)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    z = np.load(res / "cameras.npz", allow_pickle=False)
    w2c = np.array(z["extrinsics"], np.float64)
    if w2c.ndim == 3 and w2c.shape[1:] == (3, 4):    # DA3 原始格式 R|t，补齐次行再求逆
        pad = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (w2c.shape[0], 1, 1))
        w2c = np.concatenate([w2c, pad], axis=1)
    c2w = np.linalg.inv(w2c)                          # DA3 npz 存 w2c，轨道拟合要 c2w
    K = np.array(z["intrinsics"], np.float64)
    H0, W0 = z["depth"].shape[-2:]                    # DA3 处理分辨率（504p）

    center, up, rad = orbit_fit(c2w)
    e1, e2 = orbit_basis(center, up, c2w)
    azel = np.array([to_azel(center, up, e1, e2, M[:3, 3]) for M in c2w])
    r_p50, r_max, pos_rms = calibrate_pose_builder(c2w, center, up, e1, e2)
    az_sorted = np.sort(azel[:, 0])
    gaps = np.diff(np.concatenate([az_sorted, [az_sorted[0] + 360.0]]))
    biggest_gap = float(gaps.max())

    elev = float(args.elev) if args.elev is not None else float(np.median(azel[:, 1]))
    radius = rad * args.radius_k

    print(f"🛰️  [19] 轨道 | 球心 {np.round(center, 4).tolist()}  半径 {radius:.4f}"
          f"（观测 p50 {rad:.4f} × {args.radius_k}）  仰角 {elev:+.1f}°"
          f"（观测 {azel[:, 1].min():+.1f}~{azel[:, 1].max():+.1f}°）")
    print(f"   位姿自检 p50 {r_p50:.2f}°（阈值 25°）  方位空洞 {biggest_gap:.1f}°")
    if r_p50 > 25.0:
        sys.exit("❌ 位姿构造器自检不通过 —— 别往下渲")

    kw = args.outw / W0
    H = int(round(H0 * kw))
    H += H % 2
    W = args.outw
    Km = np.median(K, axis=0).copy()
    Km[0, 0] *= W / W0
    Km[1, 1] *= H / H0
    Km[0, 2] = (W - 1) / 2.0            # 轨道视图没有原始主点偏置，放正中
    Km[1, 2] = (H - 1) / 2.0

    means, sh, op, sc, q = load_ply(res / "gs_max" / "0000.ply")
    n_pts = len(means)
    sh_degree = min(args.sh_degree, sh.shape[1] - 1)
    print(f"📦 PLY {n_pts:,} 点 ｜ SH{sh_degree} ｜ 输出 {W}×{H} ｜ fx {Km[0, 0]:.1f}")

    dev = args.device
    # PLY 存储约定：opacity 是 logit、scale 是 log —— 必须 sigmoid/exp（同 07_eval_da3_gs.py:113-116）
    G = dict(
        means_t=torch.from_numpy(np.ascontiguousarray(means)).to(dev),
        quats_t=torch.from_numpy(np.ascontiguousarray(q)).to(dev),
        scales_t=torch.exp(torch.from_numpy(np.ascontiguousarray(sc)).to(dev)),
        opacities_t=torch.sigmoid(torch.from_numpy(np.ascontiguousarray(op)).to(dev)),
        colors_t=torch.from_numpy(np.ascontiguousarray(sh)).to(dev),   # (N,K,3) 全 SH
    )
    Km_t = torch.from_numpy(Km.astype(np.float32)).to(dev)[None]
    bg = torch.full((3,), float(args.background), device=dev)   # 本版 gsplat 单视角 backgrounds=(C,)

    if args.arc:
        gap_lo = float(az_sorted[int(np.argmax(gaps))])
        start = (gap_lo + biggest_gap) % 360.0
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
        M = make_c2w(center, up, e1, e2, az, elev, radius)
        w2c_t = torch.from_numpy(np.linalg.inv(M).astype(np.float32)).to(dev)[None]
        with torch.no_grad():
            rgb, alpha = rasterize(G, w2c_t, Km_t, W, H, args.eps2d, bg, dev, sh_degree)
        frame = rgb.mul(255).byte().cpu().numpy()
        proc.stdin.write(frame.tobytes())
        if i in keep_at:
            Image.fromarray(frame).save(out / f"{args.tag}_frame_{i:03d}.png")
        if (i + 1) % 60 == 0 or i == 0:
            print(f"   frame {i + 1:3d}/{args.nframes}  α 中位 {float(alpha.median()):.3f}"
                  f"  ({(time.time() - t0):.1f}s)", flush=True)

    proc.stdin.close()
    if proc.wait() != 0:
        sys.exit("❌ ffmpeg 编码失败")
    print(f"✅ {mp4}（{args.nframes} 帧 / {args.fps} fps / {args.nframes / args.fps:.1f}s"
          f" / {mp4.stat().st_size / 1e6:.1f} MB / 渲染耗时 {time.time() - t0:.1f}s）")


if __name__ == "__main__":
    main()
