# -*- coding: utf-8 -*-
"""
00s_mirror_probe.py — 决定性检验：GT 分支的渲染图是不是「真值的某种几何变换」

为什么需要（2026-11-20）：
  误差预算已经把可能性排除到只剩一个：
    - K 不是问题（pred外参+GT_K 10.94 vs pred外参+pred_K 10.93，几乎无差）
    - 相对位姿是对的（3.37°，间距比 1.0087）
    - 3° 旋转误差只值 5.2dB（20.53→15.30）
    → 但 GT 分支实测 11.32，还差 4dB 没解释。

  这说明渲染图与 GT 之间可能存在一个**全局几何变换**（镜像 / 90° 旋转 /
  缩放 / 平移），而不是随机误差。随机误差只会均匀拉低 PSNR，
  全局变换则会让 PSNR 卡在一个"结构对但方位错"的平台（典型 8~12dB）。

  本脚本把渲染图与 GT 做二面体群（D4，8 种）+ 尺度 + 平移的穷举搜索：
    若某个变换把 PSNR 拉回到 15dB 以上 → 存在系统性几何错位，且已定位是哪种。
    若所有变换都救不回来 → 误差是弥散的，问题在几何本身（前馈质量）。

  为什么用互相关定平移：直接把渲染图和 GT 的灰度互相关，峰值位置就是最佳平移；
  比暴力搜索快 3 个数量级，且对噪声稳健。

用法： python 00s_mirror_probe.py <cache.pt> <transforms.json>
"""
import os, sys, math, itertools
import numpy as np
import torch
import torch.nn.functional as F

SD = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SD)
import infer as I
import importlib.util as ilu


def _mod(name, f):
    sp = ilu.spec_from_file_location(name, os.path.join(SD, f))
    m = ilu.module_from_spec(sp); sp.loader.exec_module(m)
    return m


k00 = _mod("k00", "00k_offline_align.py")
o00 = _mod("o00", "00o_joint_scan.py")


def render(d, w2cs, Ks_r, rh, rw, dev):
    from gsplat import rasterization
    with torch.no_grad():
        out, _, _ = rasterization(
            means=d["means"].to(dev), quats=d["quats"].to(dev),
            scales=d["scales"].to(dev), opacities=d["opac"].to(dev),
            colors=d["colors"].to(dev),
            viewmats=w2cs.to(dev).float(), Ks=Ks_r.to(dev).float(),
            width=rw, height=rh, render_mode="RGB+D", packed=False,
            backgrounds=torch.zeros((w2cs.shape[0], 3), dtype=torch.float32, device=dev))
    return out[..., :3]


def psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return 10 * math.log10(1.0 / max(mse, 1e-12))


def dihedral_fit(x, k, rh, rw):
    """对 x(C,H,W) 施加 D4 元素，再缩放到 (rh,rw)。
    90°/270° 旋转会交换 H/W，必须重采样对齐，否则形状不匹配。"""
    if k >= 4:
        x = torch.flip(x, dims=[-1])
    for _ in range(k % 4):
        x = torch.rot90(x, 1, dims=[-2, -1])
    if x.shape[-2] != rh or x.shape[-1] != rw:
        x = F.interpolate(x.unsqueeze(0), (rh, rw), mode="bilinear",
                          align_corners=False, antialias=True)[0]
    return x


def best_shift_psnr(pr, gt, max_shift=40):
    """用灰度互相关找最佳平移，返回 (psnr, dy, dx)"""
    a = pr.mean(0); b = gt.mean(0)
    A = torch.fft.rfft2(a - a.mean())
    B = torch.fft.rfft2(b - b.mean())
    cc = torch.fft.irfft2(A * torch.conj(B), s=a.shape)
    # 只在中心 ±max_shift 窗口里找峰（避免周期绕回）
    H, W = a.shape
    cc2 = torch.zeros_like(cc)
    cc2[:max_shift, :max_shift] = cc[:max_shift, :max_shift]
    cc2[-max_shift:, :max_shift] = cc[-max_shift:, :max_shift]
    cc2[:max_shift, -max_shift:] = cc[:max_shift, -max_shift:]
    cc2[-max_shift:, -max_shift:] = cc[-max_shift:, -max_shift:]
    idx = int(torch.argmax(cc2))
    dy, dx = idx // cc.shape[1], idx % cc.shape[1]
    if dy > H // 2: dy -= H
    if dx > W // 2: dx -= W
    shifted = torch.roll(pr, shifts=(dy, dx), dims=(-2, -1))
    return psnr(shifted, gt), dy, dx


def main():
    cache, tpath = sys.argv[1], sys.argv[2]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = torch.load(cache, map_location="cpu", weights_only=False)
    paths = d["paths"]; new_h, new_w = int(d["new_h"]), int(d["new_w"])
    rh, rw = int(d["render_h"]), int(d["render_w"])
    P = d["pred_w2c"].to(dev).float(); pred_K = d["pred_K"].to(dev).float()
    src = d["src"].to(dev)[0]
    s = rw / new_w

    def scale_K(K):
        K = K.clone(); K[:, 0, :] *= s; K[:, 1, :] *= s
        return K

    Kr_pred = scale_K(pred_K)
    base_rgb = render(d, P, Kr_pred, rh, rw, dev)
    gt_imgs = torch.stack([
        F.interpolate(src[i].unsqueeze(0), (rh, rw), mode="bilinear",
                      align_corners=False, antialias=True)[0].clamp(0, 1).permute(1, 2, 0)
        for i in range(src.shape[0])])
    print(f"pose-free 自洽基线: {np.mean([psnr(base_rgb[i], gt_imgs[i]) for i in range(len(gt_imgs))]):.2f} dB\n")

    undist = I.probe_distortion(tpath)
    G_raw, Kg = o00.build_gt(tpath, paths, new_h, new_w, undist,
                             gl2cv=True, transpose=True, kswap=False)
    G_raw = G_raw.to(dev).float(); Kg = Kg.to(dev).float()
    s_, R_, t_, _ = k00.solve_rot_from_axes(P, G_raw)
    Ga = k00.apply_sim3(G_raw, s_, R_.to(torch.float32), t_.to(torch.float32))
    Kr_gt = scale_K(Kg)
    g_rgb = render(d, Ga, Kr_gt, rh, rw, dev)

    print("=" * 78)
    print("① 原始对齐（无变换）逐视角 PSNR:")
    raw = [psnr(g_rgb[i], gt_imgs[i]) for i in range(len(gt_imgs))]
    print("   " + " ".join(f"v{i}={x:.2f}" for i, x in enumerate(raw)) + f"  均 {np.mean(raw):.2f}")
    print("=" * 78)

    # D4 搜索（每个视角各选最优），并用 view0 统一判断是否存在全局变换
    print("\n② D4 二面体群扫描（k=0..7: 0-3 旋转, 4-7 加左右翻）")
    print(f"   {'k':<4s} {'平均PSNR':>9s}  逐视角")
    best_k, best_p = -1, -1e9
    for k in range(8):
        ps = []
        for i in range(len(gt_imgs)):
            r = dihedral_fit(g_rgb[i].permute(2, 0, 1), k, rh, rw).permute(1, 2, 0)
            ps.append(psnr(r, gt_imgs[i]))
        m = float(np.mean(ps))
        if m > best_p:
            best_p, best_k = m, k
        print(f"   {k:<4d} {m:9.2f}  " + " ".join(f"{x:5.1f}" for x in ps))
    print(f"   → 最佳 k={best_k}  {best_p:.2f} dB")

    print("\n③ 平移搜索（在最佳 k 之上，用互相关定峰）")
    for i in range(len(gt_imgs)):
        r0 = g_rgb[i]
        r1 = dihedral_fit(r0.permute(2, 0, 1), best_k, rh, rw).permute(1, 2, 0)
        p0 = psnr(r1, gt_imgs[i])
        p1, dy, dx = best_shift_psnr(r1, gt_imgs[i])
        print(f"   v{i}: 无平移 {p0:6.2f} → 平移后 {p1:6.2f}  (dy={dy:+d}, dx={dx:+d})")

    print("\n④ 显著性判断")
    print(f"   pose-free 自洽        : {np.mean([psnr(base_rgb[i], gt_imgs[i]) for i in range(len(gt_imgs))]):.2f} dB")
    print(f"   GT 对齐 原始          : {np.mean(raw):.2f} dB")
    print(f"   GT 对齐 + 最佳 D4     : {best_p:.2f} dB")
    if best_p > 15.0 and best_p - np.mean(raw) > 3.0:
        print("   ⚠️  存在系统性几何错位：某个 D4 变换能显著救回 → 查镜像/转置")
    else:
        print("   ✅ 无系统性几何错位：误差弥散 → 瓶颈在前馈几何本身，不是相机约定")


if __name__ == "__main__":
    main()
