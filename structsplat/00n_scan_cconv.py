# -*- coding: utf-8 -*-
"""
00n_scan_cconv.py — GT 相机约定穷举扫描（离线，用前向缓存，每个候选渲一次算 PSNR）

为什么需要（2026-11-20）：端到端 PSNR 一直停在 6.4~7.3dB，而 pose-free 自洽有 17.5dB。
00m 诊断给出决定性线索 —— 对齐后 GT 与 pred 的**逐视角旋转差恒为 90.001°**，
四个视角完全一致。恒定 90° 不是估计误差，是坐标系差一个"轴交换"：
    M = [[0,1,0],[1,0,0],[0,0,1]]   （绕 z 轴交换 x/y，含镜像 → det=−1）
K 那边已经处理过转置（fx↔fy、cx↔cy 交换），外参如果再乘一次 M 就多转了 90°。

本脚本把「约定」拆成几个正交开关，穷举后每个候选真渲一次算 PSNR，
让数据选，而不是靠推理选：
    gl2cv      右乘 diag(1,-1,-1)
    transpose  右乘 M（轴交换）
    kswap      K 的 fx/cx ↔ fy/cy 交换
    kmirror_v  K 的 cy 用 H−cy（转置导致的 v 轴镜像）
三个误差源独立，正确组合必须自洽。

用法：
    python 00n_scan_cconv.py <cache.pt> <transforms.json>
"""
import os, sys, math, itertools
import numpy as np
import torch
import torch.nn.functional as F

SD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SD)
import infer as I


def build_gt(transforms_path, image_paths, new_h, new_w, undist,
             gl2cv=False, transpose=False, kswap=False, kmirror_v=False):
    """
    按开关重建 GT w2c + K。故意不复用 load_gt_cameras，方便逐开关打开对比。
    返回 (w2c (S,4,4) float32, K (S,3,3) float32)
    """
    import json
    with open(transforms_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    by_base = {}
    for fr in meta.get("frames", []):
        by_base[os.path.basename(fr.get("file_path", ""))] = fr

    _Kuse = undist[2] if undist is not None else None

    M = torch.tensor([[0., 1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=torch.float32)
    FLIP = torch.diag(torch.tensor([1., -1., -1.], dtype=torch.float32))

    c2ws, Ks = [], []
    for p in image_paths:
        fr = by_base[os.path.basename(p)]
        W0, H0 = fr["w"], fr["h"]
        if _Kuse is not None:
            fx, fy = float(_Kuse[0, 0]), float(_Kuse[1, 1])
            cx, cy = float(_Kuse[0, 2]), float(_Kuse[1, 2])
        else:
            fx, fy = fr["fl_x"], fr["fl_y"]
            cx, cy = fr["cx"], fr["cy"]
        transposed = (new_h < new_w) and (H0 > W0)
        if transposed:
            s_u, s_v = new_w / H0, new_h / W0
            # 转置后 u' 源自原 v
            a_f, a_c = fy * s_u, cy * s_u
            b_f, b_c = fx * s_v, cx * s_v
        else:
            sx, sy = new_w / W0, new_h / H0
            a_f, a_c = fx * sx, cx * sx
            b_f, b_c = fy * sy, cy * sy

        if kswap:
            fx2, fy2, cx2, cy2 = a_f, b_f, a_c, b_c
        else:
            fx2, fy2, cx2, cy2 = (b_f, a_f, b_c, a_c) if transposed else (a_f, b_f, a_c, b_c)
        if kmirror_v:
            cy2 = new_h - cy2
        K = torch.tensor([[fx2, 0., cx2], [0., fy2, cy2], [0., 0., 1.]], dtype=torch.float32)

        c2w = torch.tensor(fr["transform_matrix"], dtype=torch.float32).clone()
        if gl2cv:
            c2w[:3, :3] = c2w[:3, :3] @ FLIP
        if transposed and transpose:
            c2w[:3, :3] = c2w[:3, :3] @ M
        c2ws.append(c2w)
        Ks.append(K)
    return torch.linalg.inv(torch.stack(c2ws)), torch.stack(Ks)


def render_psnr(means, quats, scales, opac, colors, w2cs, Ks_r, src,
                render_h, render_w, device):
    from gsplat import rasterization
    with torch.no_grad():
        out, _, _ = rasterization(
            means=means, quats=quats, scales=scales,
            opacities=opac, colors=colors,
            viewmats=w2cs, Ks=Ks_r,
            width=render_w, height=render_h,
            render_mode="RGB+D", packed=False,
            backgrounds=torch.zeros((w2cs.shape[0], 3), dtype=torch.float32, device=device),
        )
    rgb = out[..., :3]
    psnrs = []
    for i in range(rgb.shape[0]):
        gt = F.interpolate(src[i].unsqueeze(0), (render_h, render_w),
                           mode="bilinear", align_corners=False, antialias=True)[0].clamp(0, 1)
        pr = rgb[i].permute(2, 0, 1).clamp(0, 1)
        mse = float(((gt - pr) ** 2).mean())
        psnrs.append(10 * math.log10(1.0 / max(mse, 1e-12)))
    return float(np.mean(psnrs)), psnrs


def main():
    cache = sys.argv[1]
    tpath = sys.argv[2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = torch.load(cache, map_location="cpu", weights_only=False)
    means = d["means"].to(device); quats = d["quats"].to(device)
    scales = d["scales"].to(device); opac = d["opac"].to(device)
    colors = d["colors"].to(device)
    src = d["src"].to(device)[0]
    paths = d["paths"]
    new_h, new_w = int(d["new_h"]), int(d["new_w"])
    pred_w2c = d["pred_w2c"].to(device)
    pred_K = d["pred_K"].to(device)
    render_h, render_w = int(d.get("render_h", 512 * new_h // new_w)), int(d.get("render_w", 512))
    print(f"高斯 {means.shape[0]}  输入 {new_h}×{new_w}  渲染 {render_h}×{render_w}  视角 {len(paths)}")

    undist = I.probe_distortion(tpath)

    # pose-free 自洽基线（同一批高斯 + 预测相机）作为上界参照
    s = render_w / new_w
    Kr_pf = pred_K.clone(); Kr_pf[:, 0, :] *= s; Kr_pf[:, 1, :] *= s
    base, _ = render_psnr(means, quats, scales, opac, colors, pred_w2c, Kr_pf,
                          src, render_h, render_w, device)
    print(f"\n★ pose-free 自洽基线（预测相机，上界参照）: {base:.3f} dB\n")

    # 逐开关扫描：先看四个主开关全组合（不做 Sim(3)，直接看约定对不对）
    rows = []
    for gl2cv, transpose, kswap, kmv in itertools.product([0, 1], repeat=4):
        w2c, K = build_gt(tpath, paths, new_h, new_w, undist,
                          gl2cv=bool(gl2cv), transpose=bool(transpose),
                          kswap=bool(kswap), kmirror_v=bool(kmv))
        w2c, K = w2c.to(device), K.to(device)
        Kr = K.clone(); Kr[:, 0, :] *= s; Kr[:, 1, :] *= s
        try:
            # 不对齐直接渲，只用来判断"约定本身"是否让画面出现
            p_raw, _ = render_psnr(means, quats, scales, opac, colors, w2c, Kr,
                                   src, render_h, render_w, device)
        except Exception as e:
            p_raw = float("nan")
        try:
            w2c_a, info = I.align_gt_to_model(pred_w2c, w2c, K, pred_K)
            p_al, _ = render_psnr(means, quats, scales, opac, colors, w2c_a, Kr,
                                  src, render_h, render_w, device)
        except Exception:
            p_al, info = float("nan"), {}
        rows.append((p_al, p_raw, gl2cv, transpose, kswap, kmv, p_al - base))
        print(f"  gl2cv={gl2cv} transp={transpose} kswap={kswap} kmv={kmv}"
              f"  → 对齐后 {p_al:6.2f} dB（原始 {p_raw:6.2f}）")

    rows.sort(key=lambda r: -r[0])
    print("\n" + "=" * 78)
    print("按「对齐后 PSNR」排序：")
    for r in rows[:6]:
        print(f"  {r[0]:6.2f} dB (Δ基线 {r[6]:+.2f})  gl2cv={r[2]} transp={r[3]} "
              f"kswap={r[4]} kmv={r[5]}")
    print("=" * 78)


if __name__ == "__main__":
    main()
