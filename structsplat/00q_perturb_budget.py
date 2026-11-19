# -*- coding: utf-8 -*-
"""
00q_perturb_budget.py — 相机误差预算：3° 相对旋转误差能吃掉多少 PSNR？

为什么需要（2026-11-20）：
  GT 分支最好只到 11.32 dB，pose-free 自洽 20.53 dB，差 9.2 dB。
  00p 已证明相机传递是**正确**的（相对旋转误差 3.37°、间距比 1.0087），
  所以差距不能归因于「相机约定错」。剩下两个候选：
      (a) 3.37° 的相机误差本身就足以吃掉 9 dB
      (b) 还有别的问题（K、畸变、几何本身）

  本脚本直接给 pose-free 相机注入**已知幅度**的旋转噪声，测 PSNR 曲线。
  若 3° → 约 11dB，则 (a) 成立，GT 分支已到合理上限，不是 bug。
  这是把「感觉差得多」变成「误差预算是否对得上」的唯一干净办法。

  关键点：噪声要作用在**每个视角各自**的旋转上（模拟预测相机的独立误差），
  而不是整体刚性旋转（后者会被 Sim(3) 对齐吸收，测不出影响）。

用法： python 00q_perturb_budget.py <cache.pt>
"""
import os, sys, math
import numpy as np
import torch
import torch.nn.functional as F

SD = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SD)


def rand_rot(deg, gen, dtype=torch.float32):
    """生成一个旋转角恰为 deg 的随机旋转矩阵（Rodrigues）。"""
    ax = torch.randn(3, generator=gen, dtype=torch.float64)
    ax = ax / ax.norm()
    th = math.radians(deg)
    Kx = torch.tensor([[0., -ax[2], ax[1]],
                       [ax[2], 0., -ax[0]],
                       [-ax[1], ax[0], 0.]], dtype=torch.float64)
    R = torch.eye(3, dtype=torch.float64) + math.sin(th) * Kx + (1 - math.cos(th)) * (Kx @ Kx)
    return R.to(dtype)


def render_psnr(d, w2cs, Ks_r, render_h, render_w, dev):
    from gsplat import rasterization
    means = d["means"].to(dev); quats = d["quats"].to(dev)
    scales = d["scales"].to(dev); opac = d["opac"].to(dev); colors = d["colors"].to(dev)
    src = d["src"].to(dev)[0]
    with torch.no_grad():
        out, _, _ = rasterization(
            means=means, quats=quats, scales=scales, opacities=opac, colors=colors,
            viewmats=w2cs.to(dev).float(), Ks=Ks_r.to(dev).float(),
            width=render_w, height=render_h, render_mode="RGB+D", packed=False,
            backgrounds=torch.zeros((w2cs.shape[0], 3), dtype=torch.float32, device=dev))
    rgb = out[..., :3]
    ps = []
    for i in range(rgb.shape[0]):
        gt = F.interpolate(src[i].unsqueeze(0), (render_h, render_w), mode="bilinear",
                           align_corners=False, antialias=True)[0].clamp(0, 1)
        pr = rgb[i].permute(2, 0, 1).clamp(0, 1)
        mse = float(((gt - pr) ** 2).mean())
        ps.append(10 * math.log10(1.0 / max(mse, 1e-12)))
    return float(np.mean(ps))


def main():
    cache = sys.argv[1]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = torch.load(cache, map_location="cpu", weights_only=False)
    P = d["pred_w2c"].to(dev).float()
    pred_K = d["pred_K"].to(dev).float()
    new_h, new_w = int(d["new_h"]), int(d["new_w"])
    render_h, render_w = int(d["render_h"]), int(d["render_w"])
    s = render_w / new_w
    Kr = pred_K.clone(); Kr[:, 0, :] *= s; Kr[:, 1, :] *= s

    base = render_psnr(d, P, Kr, render_h, render_w, dev)
    print(f"pose-free 自洽基线: {base:.3f} dB")
    print(f"\n{'注入旋转噪声':>12s} | {'PSNR':>8s} | {'Δ':>7s}")
    print("-" * 34)

    gen = torch.Generator().manual_seed(0)
    for deg in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0]:
        Pn = P.clone()
        for i in range(Pn.shape[0]):
            dR = rand_rot(deg, gen, dtype=torch.float32).to(dev)
            # 相机系右扰：w2c' = dR @ w2c  → 只改朝向，不改中心
            Pn[i, :3, :3] = dR @ Pn[i, :3, :3]
        p = render_psnr(d, Pn, Kr, render_h, render_w, dev)
        print(f"{deg:10.1f}° | {p:8.2f} | {p-base:+7.2f}")


if __name__ == "__main__":
    main()
