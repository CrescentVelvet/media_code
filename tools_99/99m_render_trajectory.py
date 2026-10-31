#!/usr/bin/env python3
"""99m — 按 COLMAP 轨迹渲染 3DGS PLY 为视频（gsplat 离屏渲染）。

输入：标准 3DGS PLY + 99l（或任何工具）写出的 COLMAP 文本三件套目录。
约定（与 99l 成对，改格式必须两边同步）：帧序 = IMAGE_ID 升序；images.txt 每条
记录占两行（位姿行 + POINTS2D 行）；只接受 PINHOLE / SIMPLE_PINHOLE 相机模型，
OPENCV 等带畸变的直接报错（静默按针孔渲会偏，宁可拒绝）。

activation 与原版 3DGS 训练端一致：scale=exp(raw)、opacity=sigmoid(raw)、
color 走 SH（gsplat 对 SH 输入内部做 sigmoid）。PLY 'rot' 默认按官方 3DGS 约定
存 (w,x,y,z)，gsplat 要 xyzw，加载时 roll；若渲染出来模型像被「拧散」，先怀疑
这个顺序，用 ROT_ORDER=xyzw 试。

运行环境：conda env infinisplat_v2（gsplat 1.5.2 + torch 2.6 cu124 + plyfile + cv2）。
本机无 ffmpeg，视频用 cv2.VideoWriter(mp4v) 直接写；要更小的文件可事后自行转码。

Env vars:
  PLY          输入 PLY（必填）
  TRAJ_DIR     含 cameras.txt + images.txt 的目录（必填）
  OUT          输出 MP4 路径，默认 <TRAJ_DIR>/render.mp4
  FPS          帧率，默认 30
  SH_DEGREE    渲染用的 SH 阶数，默认 -1 = 按 PLY 系数个数自动
  SSAA         超采样倍数（1=关，2=渲染 2x 再降到输出分辨率），默认 1
  EPS2D        gsplat 抗锯齿参数，默认 0.3（gsplat 1.x 的默认抗锯齿）
  NEAR / FAR   视锥裁剪，默认 0.01 / 200
  BG           背景色 "r,g,b"（0~1），默认 "0,0,0" 黑
  ROT_ORDER    PLY 'rot' 的顺序：wxyz（官方 3DGS 默认）| xyzw，默认 wxyz
  MAX_FRAMES   只渲前 N 帧（小样调试用），0=全部
  VIEW_CHUNK   一次 rasterization 并行渲染的视角数，默认 8
  KEEP_PNG     1=把逐帧 PNG 留在 <OUT>.frames/，默认 0
  W / H        0=跟随 cameras.txt；非 0=强制缩放所有帧到该分辨率
  DRY_RUN      1=只解析不渲染，打印轨迹/PLY 摘要
"""
import math
import os
import sys
from pathlib import Path


def die(msg):
    sys.exit(f"❌ {msg}")


def qvec2rotmat(q):
    # 与 99b_pack_ply_to_mp4.py:313 同式（COLMAP wxyz → w2c 旋转）
    w, x, y, z = q
    return [
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ]


def parse_cameras(path):
    """cameras.txt → {id: (w, h, fx, fy, cx, cy)}。只认 PINHOLE/SIMPLE_PINHOLE。"""
    cams = {}
    for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        model = p[1]
        if model == "PINHOLE" and len(p) >= 8:
            cams[int(p[0])] = (int(p[2]), int(p[3]),
                               float(p[4]), float(p[5]), float(p[6]), float(p[7]))
        elif model == "SIMPLE_PINHOLE" and len(p) >= 7:
            cams[int(p[0])] = (int(p[2]), int(p[3]),
                               float(p[4]), float(p[4]), float(p[5]), float(p[6]))
        else:
            die(f"cameras.txt 第 {ln} 行: 不支持相机模型 {model}"
                f"（只接受 PINHOLE / SIMPLE_PINHOLE；OPENCV 请先去畸变）")
    if not cams:
        die(f"{path} 未解析到任何相机")
    return cams


def parse_images(path):
    """images.txt → 按 IMAGE_ID 升序的 [(cam_id, viewmat[4x4])]。

    位姿行字段数 ≡ 1 (mod 3)（10 个固定位姿字段 + 3k 个 2D 点），POINTS2D 行
    字段数 ≡ 0 (mod 3)——两条判据一起用，防止漏第二行时错位吞行。
    """
    entries = {}
    pending = None  # 位姿行先缓存；紧随的空行 = 空的 POINTS2D 行（合法）
    for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.startswith("#"):
            continue
        if not line.strip():
            if pending is not None:
                pending = None  # 空 POINTS2D 行
            continue
        p = line.split()
        if len(p) % 3 == 1 and len(p) >= 10:
            image_id = int(p[0])
            R = qvec2rotmat([float(v) for v in p[1:5]])          # w2c 旋转
            t = [float(v) for v in p[5:8]]                        # w2c 平移
            viewmat = [
                [R[0][0], R[0][1], R[0][2], t[0]],
                [R[1][0], R[1][1], R[1][2], t[1]],
                [R[2][0], R[2][1], R[2][2], t[2]],
                [0.0, 0.0, 0.0, 1.0],
            ]
            if image_id in entries:
                die(f"images.txt 第 {ln} 行: IMAGE_ID {image_id} 重复")
            entries[image_id] = (int(p[8]), viewmat)
            pending = image_id
        elif len(p) % 3 == 0:
            pending = None  # POINTS2D 行，无需内容
        else:
            die(f"images.txt 第 {ln} 行字段数异常（{len(p)}）——"
                f"检查是否漏了 POINTS2D 第二行")
    if pending is not None:
        print("⚠️ 最后一条位姿记录后缺 POINTS2D 行（COLMAP 允许，已容忍）")
    if not entries:
        die(f"{path} 未解析到任何图像")
    return [entries[k] for k in sorted(entries)]


def load_gaussians(ply_path, rot_order, sh_degree_env, device):
    """标准 3DGS PLY → gsplat 输入张量。"""
    import numpy as np
    import torch
    from plyfile import PlyData

    v = PlyData.read(str(ply_path))["vertex"]
    names = set(v._property_lookup.keys()) if hasattr(v, "_property_lookup") else set(v.data.dtype.names)

    def col(name):
        if name not in names:
            die(f"PLY 缺少属性 {name!r}（不是标准 3DGS PLY？属性表: {sorted(names)}）")
        return np.asarray(v[name])

    def vec(base, k):
        """读向量属性：优先整体 'scale'/'rot'（向量型属性），退回标量 scale_0..2。

        不同导出器写法不同：官方 3DGS 写向量属性，infinisplat 等写逐分量标量。
        """
        if base in names:
            return col(base).astype(np.float32)
        parts = [f"{base}_{i}" for i in range(k)]
        if all(p in names for p in parts):
            return np.stack([col(p) for p in parts], axis=1).astype(np.float32)
        die(f"PLY 缺少 {base!r} 向量（也没有 {parts[0]}..{parts[-1]} 分量）")

    n = len(v["x"])
    xyz = np.stack([col("x"), col("y"), col("z")], axis=1).astype(np.float32)
    opacity = torch.sigmoid(torch.from_numpy(col("opacity").astype(np.float32)))
    scales = torch.exp(torch.from_numpy(vec("scale", 3)))
    rot = torch.from_numpy(vec("rot", 4))
    if rot_order == "wxyz":
        rot = torch.roll(rot, -1, dims=-1)  # 官方 3DGS 存 wxyz → gsplat 要 xyzw
    rot = torch.nn.functional.normalize(rot, dim=-1)

    dc = np.stack([col(f"f_dc_{i}") for i in range(3)], axis=1).astype(np.float32)
    rest_names = sorted((nm for nm in names if nm.startswith("f_rest_")),
                        key=lambda s: int(s.split("_")[-1]))
    if rest_names:
        rest = np.stack([col(nm) for nm in rest_names], axis=1).astype(np.float32)
        rest = rest.reshape(n, -1, 3)  # 3DGS 按 (N,(deg+1)^2-1,3) 展平存储
        shs = np.concatenate([dc[:, None, :], rest], axis=1)
    else:
        shs = dc[:, None, :]
    colors = torch.from_numpy(shs)

    max_sh = int(math.sqrt(colors.shape[1])) - 1
    sh_degree = max_sh if sh_degree_env < 0 else min(sh_degree_env, max_sh)

    print(f"🏋️ 高斯 {n:,} 个, SH 系数 {colors.shape[1]} 个/点（max degree {max_sh}，"
          f"渲染用 degree {sh_degree}）")
    means = torch.from_numpy(xyz).to(device)
    return {
        "means": means,
        "quats": rot.to(device),
        "scales": scales.to(device),
        "opacities": opacity.to(device),
        "colors": colors.to(device),
        "sh_degree": sh_degree,
        "center": means.mean(dim=0).cpu(),
    }


def main():
    ply_path = Path(os.environ.get("PLY", "").strip())
    traj_dir = Path(os.environ.get("TRAJ_DIR", "").strip())
    if not ply_path or not ply_path.exists():
        die(f"PLY 未设置或不存在: {ply_path}")
    if not traj_dir or not traj_dir.exists():
        die(f"TRAJ_DIR 未设置或不存在: {traj_dir}")
    cam_path = traj_dir / "cameras.txt"
    img_path = traj_dir / "images.txt"
    if not cam_path.exists() or not img_path.exists():
        die(f"{traj_dir} 缺 cameras.txt / images.txt")

    cams = parse_cameras(cam_path)
    frames = parse_images(img_path)

    fps = int(os.environ.get("FPS", "30"))
    ssaa = int(os.environ.get("SSAA", "1"))
    eps2d = float(os.environ.get("EPS2D", "0.3"))
    near = float(os.environ.get("NEAR", "0.01"))
    far = float(os.environ.get("FAR", "200"))
    bg = [float(x) for x in os.environ.get("BG", "0,0,0").split(",")]
    if len(bg) != 3:
        die("BG 需要 r,g,b 三个数")
    rot_order = os.environ.get("ROT_ORDER", "wxyz")
    if rot_order not in ("wxyz", "xyzw"):
        die("ROT_ORDER 只能是 wxyz | xyzw")
    max_frames = int(os.environ.get("MAX_FRAMES", "0"))
    chunk = int(os.environ.get("VIEW_CHUNK", "8"))
    keep_png = os.environ.get("KEEP_PNG", "0") == "1"
    force_w = int(os.environ.get("W", "0"))
    force_h = int(os.environ.get("H", "0"))
    dry_run = os.environ.get("DRY_RUN", "0") == "1"

    if max_frames > 0:
        frames = frames[:max_frames]
    n_frames = len(frames)
    w0, h0 = cams[frames[0][0]][0], cams[frames[0][0]][1]
    out_w = force_w or w0
    out_h = force_h or h0

    print(f"🚀 [99m] PLY={ply_path}")
    print(f"  📁 轨迹: {n_frames} 帧（cameras {len(cams)} 台）, "
          f"输出 {out_w}x{out_h}@{fps}fps, SSAA={ssaa}, eps2d={eps2d}")
    if dry_run:
        print("⏭️ DRY_RUN=1，只解析不渲染")
        return

    import cv2
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        die("torch.cuda 不可用——99m 必须在 WSL GPU 环境跑（infinisplat_v2）")
    device = "cuda"
    from gsplat import rasterization

    gs = load_gaussians(ply_path, rot_order, int(os.environ.get("SH_DEGREE", "-1")), device)

    out_path = Path(os.environ.get("OUT", "").strip() or str(traj_dir / "render.mp4"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = Path(str(out_path) + ".frames")
    if keep_png:
        frames_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (out_w, out_h))
    if not writer.isOpened():
        die(f"cv2.VideoWriter 打不开: {out_path}")

    bg_t = torch.tensor([bg], device=device, dtype=torch.float32)
    t0 = __import__("time").time()
    try:
        for s in range(0, n_frames, chunk):
            batch = frames[s:s + chunk]
            viewmats = torch.tensor([f[1] for f in batch], device=device, dtype=torch.float32)
            Ks = []
            for cam_id, _ in batch:
                w, h, fx, fy, cx, cy = cams[cam_id]
                # 统一换算到「实际渲染分辨率」的像素坐标系
                rw, rh = (force_w or w) * ssaa, (force_h or h) * ssaa
                sx, sy = rw / w, rh / h
                Ks.append([[fx * sx, 0.0, cx * sx], [0.0, fy * sy, cy * sy], [0.0, 0.0, 1.0]])
            Ks = torch.tensor(Ks, device=device, dtype=torch.float32)
            rw, rh = (force_w or cams[batch[0][0]][0]) * ssaa, (force_h or cams[batch[0][0]][1]) * ssaa
            # gsplat 1.5.x 返回 (render, alphas, info) 三元组
            renders, _alphas, _info = rasterization(
                means=gs["means"], quats=gs["quats"], scales=gs["scales"],
                # colors 要显式扩成 (C,N,K,3)：与 infinisplat_v2/render_check.py:124 对齐，
                # 不依赖 gsplat 对 batch 维的隐式广播
                opacities=gs["opacities"],
                colors=gs["colors"][None].expand(len(batch), -1, -1, -1).contiguous(),
                viewmats=viewmats,
                Ks=Ks, width=rw, height=rh, sh_degree=gs["sh_degree"],
                eps2d=eps2d, near_plane=near, far_plane=far,
                backgrounds=bg_t.expand(len(batch), -1),
            )
            imgs = (renders.clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()
            for j, img in enumerate(imgs):
                idx = s + j
                if ssaa > 1:
                    img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
                writer.write(img[..., ::-1])  # RGB → BGR
                if keep_png:
                    cv2.imwrite(str(frames_dir / f"frame_{idx:06d}.png"), img[..., ::-1])
            done = min(s + chunk, n_frames)
            print(f"  🎬 {done}/{n_frames} 帧（{(__import__('time').time() - t0):.1f}s）")
    finally:
        writer.release()
    print(f"🎉 完成: {out_path}（{n_frames / fps:.1f}s, "
          f"{(__import__('time').time() - t0):.1f}s 总耗时）")


if __name__ == "__main__":
    main()
