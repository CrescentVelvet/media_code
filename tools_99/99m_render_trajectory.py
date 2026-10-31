#!/usr/bin/env python3
"""99m_render_trajectory.py — 按 COLMAP 轨迹渲染 3DGS PLY 为视频（gsplat 离屏渲染）。

输入：标准 3DGS PLY + 99l（或任何工具）写出的 COLMAP 文本三件套目录。

环境准备
    conda env infinisplat_v2（gsplat 1.5.2 + torch 2.6 cu124 + plyfile + cv2），
    需要 NVIDIA GPU。服务器上按 tools_99/README.md 族 E「服务器运行」建
    gs_traj_render env。本机无 ffmpeg，视频用 cv2.VideoWriter(mp4v) 直接写。

与 99l 成对，轨迹格式约定（改格式必须两边同步）：帧序 = IMAGE_ID 升序；
images.txt 每条记录占两行（位姿行 + POINTS2D 行）；只接受 PINHOLE /
SIMPLE_PINHOLE 相机模型，OPENCV 等带畸变的直接报错（静默按针孔渲会偏，宁可拒绝）。

activation 与原版 3DGS 训练端一致：scale=exp(raw)、opacity=sigmoid(raw)、
color 走 SH（gsplat 对 SH 输入内部做 sigmoid）。PLY 'rot' 默认按官方 3DGS 约定
存 (w,x,y,z)，gsplat 要 xyzw，加载时 roll；若渲染出来模型像被「拧散」，先怀疑
这个顺序，用 ROT_ORDER=xyzw 试。

f_rest 有两种磁盘排布：系数主序（官方 3DGS，每基函数 RGB 连续 3 列）与通道
主序（部分 fork 导出少做一次 transpose，R/G/B 各一块）。SH_LAYOUT=auto 按块内
能量衰减自动区分；解析错的典型症状是 DC 颜色正常、一动相机就整屏彩虹光影。

用法:
    python tools_99/99m_render_trajectory.py               # 用 main() 里的预设批量跑
    TRAJ_DIR=<目录1>,<目录2> python ...                    # 批量：逗号分隔多条轨迹
    TRAJ_DIR=<目录> OUT=<路径.mp4> python ...              # 单条：OUT 可自定义文件名
    MAX_FRAMES=16 KEEP_PNG=1 python ...                    # 小样目检首中尾帧
    SSAA=2 python ...                                      # 超采样正式出片
    DRY_RUN=1 python ...                                   # 只解析不渲染

批量语义：PLY 只加载一次，逐条轨迹渲成独立 MP4；批量时输出固定为
OUT_DIR/<轨迹目录名>.mp4（轨迹目录名如 dolly_out → dolly_out.mp4），
此时 OUT 被忽略；单条（TRAJ_DIR 无逗号）时 OUT 生效、缺省为 OUT_DIR/<目录名>.mp4。

Env vars（不设则用下方 main() 里的默认值）:
  路径   PLY, TRAJ_DIR（单个目录，或逗号分隔多个）, OUT（单条时的输出）,
         OUT_DIR（批量输出目录）
  渲染   FPS, SH_DEGREE（-1=按 PLY 系数自动）, SSAA, EPS2D, NEAR, FAR, BG, W, H
  调试   ROT_ORDER（wxyz|xyzw）, SH_LAYOUT（auto|coef|channel）, MAX_FRAMES,
         VIEW_CHUNK, KEEP_PNG, DRY_RUN
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


def detect_sh_layout(rest):
    """按块内能量衰减区分 f_rest 磁盘排布（与 99n 的判据同源）。

    系数主序（官方 3DGS）：每基函数的 RGB 连续 3 列——按每通道 15 个切块时，
    块内各基函数同阶能量接近，至少有一块（同阶）近似水平；
    通道主序（部分 fork 导出）：R/G/B 各一块、块内按基函数阶衰减 → 每块尾部都趋零。
    取各块衰减比「最大值」判断：全块都衰减才认 channel。双统计量取「或」：
    中位数（抗重尾）阈值 0.30，均值（真实数据可能反而更低）阈值 0.25——
    任一显示全块衰减即认 channel；系数主序的同阶块在任何中心统计量下都近似水平，
    不会因为「或」规则被误判。
    解析错会在颜色通道间交叉污染 → 整屏彩虹光影。
    """
    import numpy as np
    x = np.abs(rest)
    k = x.shape[1]
    if k % 3:
        return "coef"
    per = k // 3  # 每通道系数数 = (deg+1)^2 - 1
    if math.isqrt(per + 1) ** 2 != per + 1:
        return "coef"

    def block_ratios(a):
        out = []
        for b in range(3):
            head = float(a[b * per: b * per + 3].mean())
            tail = float(a[(b + 1) * per - 3: (b + 1) * per].mean())
            out.append(tail / head if head > 0 else 1.0)
        return out

    r_med = block_ratios(np.median(x, axis=0))
    r_mean = block_ratios(x.mean(axis=0))
    is_ch = max(r_med) < 0.30 or max(r_mean) < 0.25
    print(f"  🔍 SH 排布检测: 块尾/头能量比 中位 {['%.3f' % r for r in r_med]} "
          f"均值 {['%.3f' % r for r in r_mean]} → "
          f"{'channel（通道主序）' if is_ch else 'coef（系数主序）'}")
    return "channel" if is_ch else "coef"


def load_gaussians(ply_path, rot_order, sh_degree_env, sh_layout_env, device):
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
        layout = detect_sh_layout(rest) if sh_layout_env == "auto" else sh_layout_env
        if layout == "channel":
            # 通道主序展平 (N,3,K-1) → 转回 gsplat 要的系数主序 (N,K-1,3)
            rest = rest.reshape(n, 3, -1).transpose(0, 2, 1)
        else:
            rest = rest.reshape(n, -1, 3)  # 系数主序：(N,(deg+1)^2-1,3)
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
    }


def render_traj(gs, cams, frames, out_path, P):
    """渲染一条轨迹为 MP4。gs 已加载（PLY 只读一次，批量时共用），P 为参数字典。"""
    import cv2
    import torch
    from gsplat import rasterization

    if P["max_frames"] > 0:
        frames = frames[:P["max_frames"]]
    n_frames = len(frames)
    w0, h0 = cams[frames[0][0]][0], cams[frames[0][0]][1]
    out_w, out_h = P["force_w"] or w0, P["force_h"] or h0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = Path(str(out_path) + ".frames")
    if P["keep_png"]:
        frames_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             P["fps"], (out_w, out_h))
    if not writer.isOpened():
        die(f"cv2.VideoWriter 打不开: {out_path}")

    bg_t = torch.tensor([P["bg"]], device=gs["means"].device, dtype=torch.float32)
    t0 = __import__("time").time()
    try:
        for s in range(0, n_frames, P["chunk"]):
            batch = frames[s:s + P["chunk"]]
            viewmats = torch.tensor([f[1] for f in batch], device=gs["means"].device,
                                    dtype=torch.float32)
            Ks = []
            for cam_id, _ in batch:
                w, h, fx, fy, cx, cy = cams[cam_id]
                # 统一换算到「实际渲染分辨率」的像素坐标系
                rw, rh = (P["force_w"] or w) * P["ssaa"], (P["force_h"] or h) * P["ssaa"]
                sx, sy = rw / w, rh / h
                Ks.append([[fx * sx, 0.0, cx * sx], [0.0, fy * sy, cy * sy],
                           [0.0, 0.0, 1.0]])
            Ks = torch.tensor(Ks, device=gs["means"].device, dtype=torch.float32)
            rw, rh = (P["force_w"] or w0) * P["ssaa"], (P["force_h"] or h0) * P["ssaa"]
            # gsplat 1.5.x 返回 (render, alphas, info) 三元组
            renders, _alphas, _info = rasterization(
                means=gs["means"], quats=gs["quats"], scales=gs["scales"],
                # colors 要显式扩成 (C,N,K,3)：与 infinisplat_v2/render_check.py:124 对齐，
                # 不依赖 gsplat 对 batch 维的隐式广播
                opacities=gs["opacities"],
                colors=gs["colors"][None].expand(len(batch), -1, -1, -1).contiguous(),
                viewmats=viewmats,
                Ks=Ks, width=rw, height=rh, sh_degree=gs["sh_degree"],
                eps2d=P["eps2d"], near_plane=P["near"], far_plane=P["far"],
                backgrounds=bg_t.expand(len(batch), -1),
            )
            imgs = (renders.clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()
            for j, img in enumerate(imgs):
                idx = s + j
                if P["ssaa"] > 1:
                    img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
                writer.write(img[..., ::-1])  # RGB → BGR
                if P["keep_png"]:
                    cv2.imwrite(str(frames_dir / f"frame_{idx:06d}.png"), img[..., ::-1])
            done = min(s + P["chunk"], n_frames)
            print(f"  🎬 {done}/{n_frames} 帧（{(__import__('time').time() - t0):.1f}s）")
    finally:
        writer.release()
    print(f"  🎉 {out_path.name}（{n_frames / P['fps']:.1f}s, "
          f"{(__import__('time').time() - t0):.1f}s）")


def main():
    # Windows 控制台默认 cp936，直接打印 emoji 会 UnicodeEncodeError；
    # WSL/Linux 下这个调用无害
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # ===== 预设路径与参数，改模型/批次时只改这几行 =====
    # 源：待渲染的 3DGS PLY + 99l 写出的轨迹目录列表（本机 WSL 示例路径；
    # 服务器上改成服务器路径）。批量：PLY 只加载一次，逐条轨迹渲成独立 MP4。
    PLY = Path(
        "../../output/2025.11.27-HMSD-Lisidani_out/"
        "action_01/result_gs/2025-11-28--00-27-09---4khq_scene/point_cloud/iteration_60/lisi01.ply"
    )
    TRAJ_DIRS = [Path(
        "../../output/2025.11.27-HMSD-Lisidani_out/"
        "action_01/result_gs/2025-11-28--00-27-09---4khq_scene/traj_camera/" + name
    ) for name in (
        "dolly_out", "grammy", "hitchcock",
        "orbit_left", "orbit_right", "orbit_up",
    )]
    OUT_DIR = Path("../../output/traj/videos")   # 批量输出目录：<轨迹目录名>.mp4
    # 常用渲染参数（其余走 env 覆盖，见文件头 Env vars 清单）
    FPS = 30
    SSAA = 1            # 2 = 渲染 2x 再降采样（抗 splats 缩放边缘闪烁）
    # ======================================================

    # env 覆盖（不设则用上面的预设）
    ply_path = Path(os.environ.get("PLY", str(PLY)).strip())
    if not ply_path.exists():
        die(f"PLY 不存在: {ply_path}")

    traj_env = os.environ.get("TRAJ_DIR", "").strip()
    if traj_env:
        # 逗号分隔 = 批量；单个 = 单条（此时 OUT 可自定义输出文件名）
        traj_dirs = [Path(p.strip()) for p in traj_env.split(",") if p.strip()]
    else:
        traj_dirs = list(TRAJ_DIRS)
    if not traj_dirs:
        die("TRAJ_DIRS 预设为空且 TRAJ_DIR 未设")

    out_env = os.environ.get("OUT", "").strip()
    out_dir = Path(os.environ.get("OUT_DIR", str(OUT_DIR)).strip())

    bg = [float(x) for x in os.environ.get("BG", "0,0,0").split(",")]
    if len(bg) != 3:
        die("BG 需要 r,g,b 三个数")
    rot_order = os.environ.get("ROT_ORDER", "wxyz")
    if rot_order not in ("wxyz", "xyzw"):
        die("ROT_ORDER 只能是 wxyz | xyzw")
    sh_layout = os.environ.get("SH_LAYOUT", "auto")
    if sh_layout not in ("auto", "coef", "channel"):
        die("SH_LAYOUT 只能是 auto | coef | channel")

    P = {
        "fps": int(os.environ.get("FPS", str(FPS))),
        "ssaa": int(os.environ.get("SSAA", str(SSAA))),
        "eps2d": float(os.environ.get("EPS2D", "0.3")),
        "near": float(os.environ.get("NEAR", "0.01")),
        "far": float(os.environ.get("FAR", "200")),
        "bg": bg,
        "max_frames": int(os.environ.get("MAX_FRAMES", "0")),
        "chunk": int(os.environ.get("VIEW_CHUNK", "8")),
        "keep_png": os.environ.get("KEEP_PNG", "0") == "1",
        "force_w": int(os.environ.get("W", "0")),
        "force_h": int(os.environ.get("H", "0")),
    }

    # 先全部解析校验（缺 cameras.txt / images.txt 的目录直接报错），
    # 避免渲到一半才发现某条轨迹有问题
    plans = []
    for td in traj_dirs:
        if not td.exists():
            die(f"TRAJ_DIR 不存在: {td}")
        cam_path, img_path = td / "cameras.txt", td / "images.txt"
        if not cam_path.exists() or not img_path.exists():
            die(f"{td} 缺 cameras.txt / images.txt")
        plans.append((td, parse_cameras(cam_path), parse_images(img_path)))

    print(f"🚀 [99m] PLY={ply_path}")
    print(f"  🗺️ {len(plans)} 条轨迹 → {out_dir}/<轨迹目录名>.mp4")
    for td, cams, frames in plans:
        print(f"    • {td.name}: {len(frames)} 帧（cameras {len(cams)} 台）")
    if os.environ.get("DRY_RUN", "0") == "1":
        print("⏭️ DRY_RUN=1，只解析不渲染")
        return

    import torch

    if not torch.cuda.is_available():
        die("torch.cuda 不可用——99m 必须在 GPU 环境跑（infinisplat_v2 / gs_traj_render）")

    gs = load_gaussians(ply_path, rot_order, int(os.environ.get("SH_DEGREE", "-1")),
                        sh_layout, "cuda")

    out_dir.mkdir(parents=True, exist_ok=True)
    t_all = __import__("time").time()
    for i, (td, cams, frames) in enumerate(plans, 1):
        if out_env and len(plans) == 1:
            out_path = Path(out_env)      # 单条时 OUT 可自定义文件名
        else:
            out_path = out_dir / f"{td.name}.mp4"
        print(f"  [{i}/{len(plans)}] {td.name} → {out_path.name}")
        render_traj(gs, cams, frames, out_path, P)
    print(f"🏁 全部完成（{(__import__('time').time() - t_all):.1f}s）")


if __name__ == "__main__":
    main()
