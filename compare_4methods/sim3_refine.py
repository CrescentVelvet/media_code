#!/usr/bin/env python3
"""Sim(3) 光度精化对齐 —— 修「三对点 Umeyama 精度不够」这个评测口径坑。

问题（2026-11-13 实测）
----------------------
三图输入的方法只预测 3 个相机，跨方法对齐只能拿这 3 个相机中心做 Umeyama：
3 点 × 3 坐标 = 9 个方程解 7 个未知数，**零冗余**，残差普遍 1.0~2.0% / 2.0~2.7°。
而本场景 GT 是 1435² 长焦（fx≈1381），角度误差的像素代价极高：

    平移  5 px → PSNR 天花板 18.7 dB
    旋转 1.0° → 15.9 dB
    旋转 2.7° → 13.7 dB

于是六个方法一起被压到 8.9~12.9 dB，**量到的是对齐精度不是模型差异**
（逐视角读数可证：view 0/7/31 就是输入视角本身，同样只有 9~14 dB，
而同一方法同一相机在 self3 自洽口径有 20.2 dB）。

做法
----
保持高斯基座不动，只学一个附着在 Umeyama 解上的小修正 Δ=(ΔR, Δτ, log Δs)，
最小化 **3 个输入视角** 的重建误差（这些图所有方法都见过，不泄漏 holdout）。
    x1 = s0·R0·x + t0                # Umeyama 粗解
    x2 = Δs·ΔR·(x1 − c) + c + Δτ     # 光度精修，c = 场景质心
合成后仍是一个 Sim(3)，交给 audit_24.warp 一次性搬到 GT 系：
    s = Δs·s0,  R = ΔR·R0,  t = Δs·ΔR·(t0 − c) + c + Δτ

为什么能收敛得这么快：初值已是 1~2% 邻域，7 个参数、3 个视角、512² 渲染，
实测 ~75 步到极小值，单方法 <5 s（RTX 3090）。

判据（写进台账前必须过）
------------------------
· 精化后输入视角 PSNR 应显著上升（实测 SA 12.18 → 21.62 dB，+9.4）
· |Δω| 应与 Umeyama 的 ang 残差同量级（实测 3.08° vs 2.72°，一致）
· Δs 应接近 1（实测 0.9983，证明没有沿光轴漂移）
"""
import time

import cv2
import numpy as np
import torch
from gsplat import rasterization


def _skew(w):
    z = torch.zeros_like(w[0])
    return torch.stack([torch.stack([z, -w[2], w[1]]),
                        torch.stack([w[2], z, -w[0]]),
                        torch.stack([-w[1], w[0], z])])


def _qmul(a, b):
    """左乘单个四元数 a(wxyz) 到 N 个四元数 b(N,4) 上。"""
    aw, ax, ay, az = a[0], a[1], a[2], a[3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw], dim=1)


def refine_sim3(g, w2c_in, K, refs_rgb, init, res=512, iters=200, log=print):
    """在输入视角上光度精化 Sim(3)。

    参数
    ----
    g           : read_ply 的高斯 dict（means N×3 / quats N×4 wxyz / scales N×3 /
                  opac N×1 / sh N×k / deg），**方法自身坐标系**，本函数不修改它
    w2c_in      : (V,4,4) 输入视角在 GT 系的 w2c（必须是输入图真实位姿，不是最近邻）
    K           : (3,3) GT 内参（原图分辨率），内部按 res 等比缩放
    refs_rgb    : V 张 HxWx3 uint8 输入图（**已裁成与 K 同一视场**）
    init        : (s, R, t) Umeyama 粗解
    res         : 优化分辨率（渲染低分辨率能显著提速，且对齐量级远大于像素级差别）
    iters       : Adam 步数（实测 200 足够，通常 ~75 步就停住）

    返回 dict(s, R, t, delta_w_deg, delta_tau, delta_s, psnr_pre, psnr_post, iters)
    """
    dev = "cuda"
    s0, R0, t0 = init
    H, W = refs_rgb[0].shape[:2]
    sc = res / float(W)
    Kt = torch.tensor([[[K[0, 0] * sc, 0.0, K[0, 2] * sc],
                        [0.0, K[1, 1] * sc, K[1, 2] * sc],
                        [0.0, 0.0, 1.0]]], dtype=torch.float32, device=dev)
    bg = torch.ones(3, device=dev)

    # 先把高斯搬到 GT 系（numpy，一次），精化只学小修正
    from audit_24_five_way_video import warp
    gw = warp({k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in g.items()}, s0, R0, t0)
    means0 = torch.from_numpy(gw["means"]).to(dev)
    quats0 = torch.from_numpy(gw["quats"]).to(dev)
    scales0 = torch.from_numpy(gw["scales"]).to(dev)
    opac = torch.from_numpy(gw["opac"]).to(dev)
    sh = torch.from_numpy(gw["sh"]).to(dev)
    deg = gw["deg"]
    ctr = means0.mean(0).detach()
    del gw

    refs = [torch.from_numpy(cv2.resize(im, (res, res), interpolation=cv2.INTER_AREA)
                             .astype(np.float32) / 255.0).to(dev) for im in refs_rgb]
    wm = torch.from_numpy(np.asarray(w2c_in, dtype=np.float64)).float().to(dev)

    def render(means, quats, scales):
        out = []
        for i in range(len(refs)):
            img, _, _ = rasterization(means, quats, scales, opac, sh, wm[i][None], Kt,
                                      res, res, sh_degree=deg, packed=False,
                                      backgrounds=bg[None], render_mode="RGB")
            out.append(img[0])
        return out

    def psnr(a, b):
        mse = float(torch.mean((a - b) ** 2))
        return 99.0 if mse <= 0 else 10.0 * np.log10(1.0 / mse)

    with torch.no_grad():
        pre = [psnr(a, b) for a, b in zip(render(means0, quats0, scales0), refs)]

    w_vec = torch.zeros(3, device=dev, requires_grad=True)
    tau = torch.zeros(3, device=dev, requires_grad=True)
    log_s = torch.zeros(1, device=dev, requires_grad=True)
    opt = torch.optim.Adam([w_vec, tau, log_s], lr=1e-3)

    t_start = time.time()
    for it in range(iters):
        # 余弦退火：先大步把 2~3° 拉回来，再小步精修到亚像素
        for grp in opt.param_groups:
            grp["lr"] = 3e-3 * (0.5 * (1 + np.cos(np.pi * it / iters))) + 1e-4
        opt.zero_grad(set_to_none=True)
        R = torch.matrix_exp(_skew(w_vec))
        s = torch.exp(log_s)
        m = s * ((means0 - ctr) @ R.T) + ctr + tau
        tr = R[0, 0] + R[1, 1] + R[2, 2]
        qw = torch.sqrt(torch.clamp(1 + tr, min=1e-8)) / 2
        q = torch.stack([qw, (R[2, 1] - R[1, 2]) / (4 * qw),
                         (R[0, 2] - R[2, 0]) / (4 * qw),
                         (R[1, 0] - R[0, 1]) / (4 * qw)])
        q = q / q.norm()
        qn = _qmul(q, quats0)
        qn = qn / qn.norm(dim=1, keepdim=True)
        imgs = render(m, qn, scales0 * s)
        loss = sum(torch.mean((a - b) ** 2) for a, b in zip(imgs, refs)) / len(refs)
        loss.backward()
        opt.step()
        if log and (it % 50 == 0 or it == iters - 1):
            log("     [refine] it %3d  PSNR %.2f  |Δω| %.2f°  |Δτ| %.4f  Δs %.4f  (%.0fs)"
                % (it, 10 * np.log10(1.0 / float(loss)), float(torch.rad2deg(w_vec.norm())),
                   float(tau.norm()), float(torch.exp(log_s)), time.time() - t_start))

    with torch.no_grad():
        R = torch.matrix_exp(_skew(w_vec))
        s1 = float(torch.exp(log_s))
        m = s1 * ((means0 - ctr) @ R.T) + ctr + tau
        tr = R[0, 0] + R[1, 1] + R[2, 2]
        qw = torch.sqrt(torch.clamp(1 + tr, min=1e-8)) / 2
        q = torch.stack([qw, (R[2, 1] - R[1, 2]) / (4 * qw),
                         (R[0, 2] - R[2, 0]) / (4 * qw),
                         (R[1, 0] - R[0, 1]) / (4 * qw)])
        q = q / q.norm()
        qn = _qmul(q, quats0)
        qn = qn / qn.norm(dim=1, keepdim=True)
        post = [psnr(a, b) for a, b in zip(render(m, qn, scales0 * s1), refs)]

    R1 = R.detach().cpu().numpy().astype(np.float64)
    tau1 = tau.detach().cpu().numpy().astype(np.float64)
    c = ctr.detach().cpu().numpy().astype(np.float64)
    # 合成：s = Δs·s0, R = ΔR·R0, t = Δs·ΔR·(t0 − c) + c + Δτ
    s = s1 * s0
    Rc = R1 @ R0
    tc = s1 * (R1 @ (t0 - c)) + c + tau1

    delta_w = float(torch.rad2deg(w_vec.norm()))
    return dict(s=float(s), R=Rc, t=tc,
                delta_w_deg=delta_w,
                delta_tau=tau1.tolist(),
                delta_s=float(s1),
                psnr_pre=float(np.mean(pre)),
                psnr_post=float(np.mean(post)),
                res=res, iters=iters)
