#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
structsplat/00k_offline_align.py — 用前向缓存离线对比 Sim(3) 对齐策略（不跑前馈）

为什么需要这个脚本（2026-11-20）：
  4 帧 GT 分支修好 Sim(3) 后 PSNR 只从 6.41 → 7.26 dB，仍远低于 pose-free 的 20.66。
  同时对齐自检报了「中心残差 RMS 0.0594」——**对齐本身没有完美闭合**，
  说明 (s,R,t) 拟合不准，而不是渲染或权重的问题。

  根因假设：现有 align_gt_to_model 用**相机中心**做 Kabsch 求旋转。
  但 4 个相机中心近乎共线（同一段轨迹上均匀抽 4 帧），
  此时旋转绕「连线轴」的自由度完全不可观测 → SVD 给出的解是任意的，
  残差 0.0594 正是这个欠定性的表现。

  更稳的做法：**旋转用相机朝向（光轴 + 上方向）估计**，只把尺度/平移交给中心。
  这与官方 get_tar_camera 用四元数均值定旋转是同一个思路（那边也是朝向，不是位置）。

本脚本读 infer.py --dump-forward 落盘的缓存，纯 CPU/GPU 渲染，秒级迭代。

对比四种策略：
  C0  identity        —— 不做对齐（复现 6.41dB 的错位基线）
  C1  center-kabsch   —— 现有实现（相机中心做 Kabsch）
  C2  rot-from-axes   —— 用相机朝向求 R，中心只定 s 与 t     ★ 待验证
  C3  procrustes-full —— 中心 + 朝向合并加权（朝向权重固定，避免尺度混淆）

判据：渲染输入视角 vs 输入图 的 PSNR/SSIM（与 infer.py 自检同口径）。

用法：
  python 00k_offline_align.py --cache /path/to/hand01_gt4.pt
"""

import argparse
import math
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# 三种 Sim(3) 估计
# ============================================================
def _centers(w2c):
    """相机中心 c = -Rᵀ·t"""
    Rw = w2c[:, :3, :3]
    tw = w2c[:, :3, 3]
    return -torch.einsum("sji,sj->si", Rw, tw)


def _axes(w2c):
    """
    相机在世界系里的三个朝向轴（单位向量），形状 (S,3,3)。

    为什么用轴而不是中心：
      w2c 的行向量就是相机基在世界系下的表示。Rw[0]=右、Rw[1]=下、Rw[2]=前(光轴)。
      这些是**纯方向量**，不受平移影响，也不像中心那样在窄基线轨迹上退化。
    """
    Rw = w2c[:, :3, :3]
    # 行向量归一化（预测相机可能带轻微非正交，先修一下）
    return Rw / Rw.norm(dim=2, keepdim=True).clamp_min(1e-9)


def _kabsch(A, B, w=None):
    """
    求最优正交 R 使 R·a ≈ b（a 取自 A，b 取自 B，均为 (N,3)）。
    w 为可选的逐点权重 (N,)。
    返回 (R, resid)
    """
    if w is None:
        w = torch.ones(A.shape[0], dtype=A.dtype, device=A.device)
    H = (A * w[:, None]).transpose(0, 1) @ B
    U, _, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.linalg.det(Vt.transpose(0, 1) @ U.transpose(0, 1)))
    D = torch.diag(torch.tensor([1.0, 1.0, float(d)],
                                dtype=A.dtype, device=A.device))
    R = Vt.transpose(0, 1) @ D @ U.transpose(0, 1)
    resid = ((A @ R.transpose(0, 1)) - B).norm(dim=1)
    return R, resid


def solve_center_kabsch(pred_w2c, gt_w2c):
    """C1：现有实现 —— 中心做 Kabsch 定旋转，中心距离比定尺度。"""
    P, G = _centers(pred_w2c).double(), _centers(gt_w2c).double()
    Pc, Gc = P.mean(0, keepdim=True), G.mean(0, keepdim=True)
    dP, dG = (P - Pc).norm(dim=1).mean(), (G - Gc).norm(dim=1).mean()
    s = float(dP / dG) if float(dG) > 1e-9 else 1.0
    R, resid = _kabsch((G - Gc) * s, P - Pc)
    t = Pc.squeeze(0) - s * (R @ Gc.squeeze(0))
    return s, R, t, {"fit_resid": float(resid.mean())}


def solve_rot_from_axes(pred_w2c, gt_w2c):
    """
    C2 ★：旋转从**相机朝向**估计，尺度/平移仍用中心。

    同一台相机在两个坐标系里的基向量满足  d_model = R · d_gt
    （因为模型系点 X_m = s·R·X_g + t，方向量不含 t，缩放 s 对单位向量无影响）。
    把 4 个相机的右/下/前三种轴全部堆起来做 Kabsch，样本量是中心法的 3 倍，
    且朝向在轨迹上变化大 → 不再欠定。
    """
    P, G = _centers(pred_w2c).double(), _centers(gt_w2c).double()
    Pc, Gc = P.mean(0, keepdim=True), G.mean(0, keepdim=True)
    dP, dG = (P - Pc).norm(dim=1).mean(), (G - Gc).norm(dim=1).mean()
    s = float(dP / dG) if float(dG) > 1e-9 else 1.0

    Am = _axes(gt_w2c).reshape(-1, 3).double()      # GT 系下的方向
    Bm = _axes(pred_w2c).reshape(-1, 3).double()    # 模型系下对应的方向
    R, resid = _kabsch(Am, Bm)

    # 平移：让中心质心对上（尺度用上面估的 s）
    t = Pc.squeeze(0) - s * (R @ Gc.squeeze(0))
    return s, R, t, {"fit_resid": float(resid.mean())}


def solve_procrustes_full(pred_w2c, gt_w2c, w_axis=1.0):
    """
    C3：中心 + 朝向联合 Kabsch。

    把中心与朝向拼成一个点集，各自标准化后再做一次加权 Kabsch。
    中心的权重按 1/尺度 归一化，防止「米级中心」把「无量纲朝向」压掉。
    """
    P, G = _centers(pred_w2c).double(), _centers(gt_w2c).double()
    Pc, Gc = P.mean(0, keepdim=True), G.mean(0, keepdim=True)
    dP, dG = (P - Pc).norm(dim=1).mean(), (G - Gc).norm(dim=1).mean()
    s = float(dP / dG) if float(dG) > 1e-9 else 1.0

    # 中心：中心化后除以各自尺度 → 无量纲
    Ap = ((G - Gc) * s) / max(dP, 1e-9)
    Bp = (P - Pc) / max(dP, 1e-9)
    # 朝向：本身就是单位向量
    Aa = _axes(gt_w2c).reshape(-1, 3).double()
    Ba = _axes(pred_w2c).reshape(-1, 3).double()

    A = torch.cat([Ap, Aa * w_axis], 0)
    B = torch.cat([Bp, Ba * w_axis], 0)
    wt = torch.cat([
        torch.full((Ap.shape[0],), 1.0, dtype=torch.float64, device=Ap.device),
        torch.full((Aa.shape[0],), 1.0, dtype=torch.float64, device=Aa.device),
    ])
    R, resid = _kabsch(A, B, wt)
    t = Pc.squeeze(0) - s * (R @ Gc.squeeze(0))
    return s, R, t, {"fit_resid": float(resid.mean())}


# ============================================================
# 把 (s,R,t) 作用到 GT w2c 上（保持 R 正交）
# ============================================================
def apply_sim3(gt_w2c, s, R, t):
    """
    X_m = s·R·X_g + t  ⇒  w2c' = R'·X_m + t',  R' = R_g·Rᵀ（正交）, t' = s·t_g − R'·t
    """
    Rm = gt_w2c[:, :3, :3] @ R.transpose(0, 1)
    tm = s * gt_w2c[:, :3, 3] - torch.einsum("sij,j->si", Rm, t)
    out = gt_w2c.clone()
    out[:, :3, :3] = Rm.to(gt_w2c.dtype)
    out[:, :3, 3] = tm.to(gt_w2c.dtype)
    return out


# ============================================================
# 渲染 + 指标
# ============================================================
def render_eval(d, w2cs, Ks, render_h, render_w, bg_val=0.0):
    from gsplat import rasterization
    import torch.nn.functional as F

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    means = d["means"].to(dev).float()
    quats = d["quats"].to(dev).float()
    scales = d["scales"].to(dev).float()
    opac = d["opac"].to(dev).float()
    colors = d["colors"].to(dev).float()
    src = d["src"].to(dev).float()          # (1,S,3,H,W)

    out, _a, _i = rasterization(
        means=means, quats=quats, scales=scales,
        opacities=opac, colors=colors,
        viewmats=w2cs.to(dev).float(), Ks=Ks.to(dev).float(),
        width=render_w, height=render_h,
        render_mode="RGB+D", packed=False,
        backgrounds=torch.full((w2cs.shape[0], 3), bg_val,
                               dtype=torch.float32, device=dev),
    )
    rgb = out[..., :3]
    S = rgb.shape[0]
    psnrs, ssims = [], []
    for i in range(S):
        gt = F.interpolate(src[0, i:i + 1], (render_h, render_w),
                           mode="bilinear", align_corners=False, antialias=True)[0]
        gt = gt.clamp(0, 1)
        pr = rgb[i].permute(2, 0, 1).clamp(0, 1)
        mse = float(((gt - pr) ** 2).mean())
        psnrs.append(90.0 if mse <= 1e-12 else 10 * math.log10(1.0 / mse))
        try:
            from skimage.metrics import structural_similarity as _ssim
            ssims.append(float(_ssim(gt.permute(1, 2, 0).cpu().numpy(),
                                     pr.permute(1, 2, 0).cpu().numpy(),
                                     channel_axis=2, data_range=1.0)))
        except Exception:
            ssims.append(float("nan"))
    return psnrs, ssims, rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="infer.py --dump-forward 产出的 .pt")
    ap.add_argument("--save-dir", default=None, help="把对比图存这里")
    args = ap.parse_args()

    d = torch.load(args.cache, map_location="cpu")
    need = ["means", "quats", "scales", "opac", "colors", "src",
            "pred_w2c", "pred_K", "gt_w2c_raw", "render_h", "render_w"]
    miss = [k for k in need if k not in d]
    if miss:
        raise SystemExit(f"❌ 缓存缺字段 {miss}；请用带 --dump-forward 的 gt 分支重跑")

    pred_w2c = d["pred_w2c"].double()
    gt_raw = d["gt_w2c_raw"].double()
    Ks_r = d["Ks"].float()          # 渲染分辨率下的 GT 内参（infer 已缩放）
    rh, rw = d["render_h"], d["render_w"]

    print("=" * 68)
    print(" Sim(3) 对齐策略离线对比（同一批高斯、同一批输入图）")
    print("=" * 68)
    print(f"  视角数 {pred_w2c.shape[0]}   渲染 {rh}×{rw}   "
          f"高斯 {d['means'].shape[0]}")
    P, G = _centers(pred_w2c), _centers(gt_raw)
    print(f"  预测系中心跨度 {float((P-P.mean(0)).norm(dim=1).max()):.4f}")
    print(f"  GT   系中心跨度 {float((G-G.mean(0)).norm(dim=1).max()):.4f}")

    strategies = [
        ("C0  identity", None),
        ("C1  center-kabsch", solve_center_kabsch),
        ("C2  rot-from-axes", solve_rot_from_axes),
        ("C3  procrustes", solve_procrustes_full),
    ]

    results = []
    for name, fn in strategies:
        if fn is None:
            w2c_used = gt_raw.float()
            s, resid, note = 1.0, float("nan"), "不对齐"
        else:
            s, R, t, info = fn(pred_w2c, gt_raw)
            w2c_used = apply_sim3(gt_raw, s, R, t).float()
            resid, note = info["fit_resid"], ""
            # 正交性自检
            Rm = w2c_used[:, :3, :3].double()
            orth = float((Rm.transpose(1, 2) @ Rm
                          - torch.eye(3, dtype=torch.float64)).abs().max())
            dets = torch.linalg.det(Rm)
            note = f"正交max {orth:.1e} det[{float(dets.min()):.6f},{float(dets.max()):.6f}]"

        ps, ss, rgb = render_eval(d, w2c_used, Ks_r, rh, rw)
        mps, mss = sum(ps) / len(ps), sum(ss) / len(ss)
        results.append((name, s, resid, mps, mss, ps))
        print(f"\n{name}")
        print(f"  尺度 s={s:.6f}  拟合残差={resid:.5f}")
        print(f"  {note}")
        print(f"  PSNR 均值 {mps:6.3f} dB  逐视角 "
              + " ".join(f"{v:.2f}" for v in ps))
        print(f"  SSIM 均值 {mss:.4f}")
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            try:
                from PIL import Image
                arr = (rgb.clamp(0, 1).cpu().numpy() * 255).astype("uint8")
                Image.fromarray(arr[0]).save(
                    os.path.join(args.save_dir, f"{name.split()[0]}.png"))
            except Exception as e:
                print(f"   (存图失败 {e})")

    print("\n" + "=" * 68)
    best = max(results, key=lambda r: r[3])
    base_pf = 20.66
    print(f" 最佳 {best[0].split()[0]}  PSNR {best[3]:.3f} dB")
    print(f" 参照 pose-free 端到端 {base_pf:.2f} dB "
          f"（差 {best[3] - base_pf:+.2f} dB）")
    print("=" * 68)


if __name__ == "__main__":
    main()
