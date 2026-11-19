#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
structsplat/00aa_alpha_diag.py — 判定「颜色偏暗」是 alpha 未满(预乘)还是几何/外观错

为什么需要（2026-11-20）：
  00y 报 far 组全图 11.14dB / 覆盖区 15.15dB；但刚做的图像分析发现：
  **即使在实心区**，把 pred 线性映射到 GT 也需要 a≈0.74、b≈0.24。
  这不是「最小二乘在迁就空白区」，而是实心区自己也偏暗、对比压缩。
  这恰好是 **alpha 未饱和时以黑底预乘** 的数学特征：
      rendered = alpha * color + (1-alpha) * 0
  → 若真是这个原因，则 rgb/alpha 反预乘后应显著提分，且 a→1、b→0。

判据（三者必须同时成立才认「预乘稀释」）：
  P1 实心区 mean(alpha) 明显 < 1（比如 0.80~0.95）
  P2 反预乘后 PSNR 提升 ≥ 1.0 dB
  P3 反预乘后线性拟合 a→≈1、b→≈0

用法：
  python 00aa_alpha_diag.py --cache ~/output/structsplat_results/_cache/hand01_src21.pt \
    --transforms ".../transforms.json" \
    --groups "far:21,42" "adj:1,2"
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

SD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SD)
import infer as I  # noqa: E402


def raster_rgbd(means, quats, scales, opac, colors, w2c, K, rh, rw, dev, bg=0.0):
    """一次渲染拿 RGB + alpha（alpha = 用全白颜色渲出来的 opacity 累积）。"""
    from gsplat import rasterization
    S = w2c.shape[0]
    with torch.no_grad():
        rgb, _, _ = rasterization(
            means=means.to(dev), quats=quats.to(dev), scales=scales.to(dev),
            opacities=opac.to(dev), colors=colors.to(dev),
            viewmats=w2c.to(dev).float(), Ks=K.to(dev).float(),
            width=rw, height=rh, render_mode="RGB", packed=False,
            backgrounds=torch.full((S, 3), bg, dtype=torch.float32, device=dev))
        a, _, _ = rasterization(
            means=means.to(dev), quats=quats.to(dev), scales=scales.to(dev),
            opacities=opac.to(dev),
            colors=torch.ones_like(colors).to(dev),
            viewmats=w2c.to(dev).float(), Ks=K.to(dev).float(),
            width=rw, height=rh, render_mode="RGB", packed=False,
            backgrounds=torch.zeros((S, 3), dtype=torch.float32, device=dev))
    return rgb.clamp(0, 1), a[..., 0].clamp(0, 1)


def psnr(a, b, m=None):
    d = (a - b) ** 2
    if m is not None:
        if int(m.sum()) < 64:
            return float("nan")
        d = d[m]
    return 10.0 * math.log10(1.0 / max(float(d.mean()), 1e-12))


def linfit_masked(pred_np, gt_np, m):
    xx = pred_np[m].reshape(-1, 3)
    yy = gt_np[m].reshape(-1, 3)
    xm, ym = xx.mean(0), yy.mean(0)
    a = ((xx - xm) * (yy - ym)).sum(0) / np.maximum(((xx - xm) ** 2).sum(0), 1e-9)
    b = ym - a * xm
    return float(a.mean()), float(b.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--transforms", required=True)
    ap.add_argument("--images", default=None)
    ap.add_argument("--groups", nargs="+", required=True)
    ap.add_argument("--no-undistort", action="store_true",
                    help="关闭去畸变（与 00y 的 --no-undistort 同义）")
    args = ap.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)

    d = torch.load(args.cache, map_location="cpu", weights_only=False)
    src_paths = d["paths"]                       # 缓存的键是 paths，不是 src_paths
    imgs_dir = args.images or os.path.dirname(src_paths[0])
    new_h, new_w = int(d["new_h"]), int(d["new_w"])
    # ⚠️ 必须与 00y 完全一致：00y 用 rh,rw = new_h,new_w（392×518），
    #    而缓存里的 render_h/render_w 是 387×512（infer.py 内部的渲染尺寸）。
    #    两者混用会导致 GT 与渲染图尺寸不匹配而算不出 PSNR。
    rh, rw = new_h, new_w
    # undist 是三件套 (K0, dist, newK)，不是布尔；与 00y 保持完全一致的构造方式
    undist = None if args.no_undistort else I.probe_distortion(args.transforms)
    print(f"缓存 {os.path.basename(args.cache)}  src={len(src_paths)}  "
          f"render={rh}x{rw}  undist={'ON' if undist is not None else 'OFF'}")

    with open(args.transforms, "r", encoding="utf-8") as f:
        tj = json.load(f)
    all_names = [os.path.basename(fr["file_path"]) for fr in tj["frames"]]

    gt_src, K_src, _ = I.load_gt_cameras(args.transforms, src_paths, new_h, new_w,
                                         undist=undist)
    pred_w2c = d["pred_w2c"].float()
    _, info = I.align_gt_to_model(pred_w2c, gt_src, K_src, d["pred_K"].float(),
                                  mode="axes", reflect="auto")
    s_, R_, t_ = info["scale_gt_to_model"], info["_R"].float(), info["_t"].float()
    print(f"Sim(3) s={s_:.5f}  残差 {info['rot_deg_mean']:.2f}°  det(R)={info['det_R']:+.2f}")

    means, quats = d["means"], d["quats"]
    scales, opac, colors = d["scales"], d["opac"], d["colors"]

    for spec in args.groups:
        gname, idxs = spec.split(":", 1)
        idxs = [int(x) for x in idxs.split(",") if x.strip()]
        tar_paths = [os.path.join(imgs_dir, all_names[i]) for i in idxs]
        _s = {os.path.basename(p) for p in src_paths}
        _t = {os.path.basename(p) for p in tar_paths}
        if _s & _t:
            raise SystemExit(f"❌ 组 {gname} 与 src 重叠：{_s & _t}")

        gts = []
        for p in tar_paths:
            t, _ = I.load_image_tensor(p, new_h, new_w, undist=undist)
            gts.append(t.clamp(0, 1).permute(1, 2, 0))
        gts = torch.stack(gts)
        gts_dev = gts.to(dev)        # PSNR 在 GPU 上算；numpy 用 gts[k].cpu()

        gt_tar, K_tar, m_tar = I.load_gt_cameras(args.transforms, tar_paths,
                                                 new_h, new_w, undist=undist)
        if len(m_tar) != len(tar_paths):
            raise SystemExit(f"❌ 组 {gname} 有图未匹配")

        Rt = gt_tar[:, :3, :3] @ R_.transpose(-1, -2)
        tt = s_ * gt_tar[:, :3, 3] - torch.einsum("sij,j->si", Rt, t_)
        tar_w2c = gt_tar.clone()
        tar_w2c[:, :3, :3] = Rt
        tar_w2c[:, :3, 3] = tt

        rgb, alpha = raster_rgbd(means, quats, scales, opac, colors, tar_w2c,
                                 K_tar, rh, rw, dev)

        print(f"\n── 组 {gname}  idx={idxs} ──")
        hdr = (f"  {'视角':<22s}{'cov%':>6s}{'α实心':>7s}{'α@cov':>7s}"
               f"{'全:原':>7s}{'全:反预乘':>9s}{'Δ':>6s}"
               f"{'核心:原':>8s}{'核心:反':>8s}{'边缘:原':>8s}")
        print(hdr)
        for k in range(len(tar_paths)):
            mk = alpha[k] > 0.5
            cov = float(mk.float().mean())
            # “核心”= alpha>0.95，真正饱和处；“边缘”= 0.5<alpha<=0.95，半透明晕圈
            core = alpha[k] > 0.95
            fringe = mk & (~core)
            a_solid = float(alpha[k][core].mean()) if int(core.sum()) > 64 else float("nan")
            a_cov = float(alpha[k][mk].mean()) if int(mk.sum()) > 64 else float("nan")

            gt_np = gts[k].cpu().numpy()
            raw_np = rgb[k].detach().cpu().numpy()
            p_raw = psnr(gts_dev[k], rgb[k])

            # 反预乘：rgb / alpha，alpha 下限保护避免爆亮
            al = alpha[k].clamp_min(1e-3)[..., None]
            un = (rgb[k] / al).clamp(0, 1)
            p_un = psnr(gts_dev[k], un)

            # 核心区（饱和处）与边缘区（半透明）分别打分：把「外观」与「晕圈」拆开
            p_core = psnr(gts_dev[k], rgb[k], core)
            p_core_un = psnr(gts_dev[k], un, core)
            p_fringe = psnr(gts_dev[k], rgb[k], fringe)

            m_np = mk.cpu().numpy()
            a0, b0 = linfit_masked(raw_np, gt_np, m_np) if int(mk.sum()) > 64 else (np.nan,) * 2
            un_np = un.detach().cpu().numpy()
            a1, b1 = linfit_masked(un_np, gt_np, m_np) if int(mk.sum()) > 64 else (np.nan,) * 2
            # ★ 核心区掩码下的拟合：若 a→1、b→0，则「曝光/色调」本身没问题，
            #   之前算出 a≈0.7~0.9 只是因为掩码含了半透明黑晕，拉低了斜率。
            core_np = core.cpu().numpy()
            a2, b2 = (linfit_masked(raw_np, gt_np, core_np)
                      if int(core.sum()) > 64 else (np.nan,) * 2)

            print(f"  {os.path.basename(tar_paths[k])[:20]:<22s}{cov*100:5.1f}%"
                  f"{a_solid:7.3f}{a_cov:7.3f}{p_raw:7.2f}{p_un:9.2f}"
                  f"{p_un-p_raw:+6.2f}{p_core:8.2f}{p_core_un:8.2f}{p_fringe:8.2f}")
            print(f"    拟合 cov掩码 a={a0:.2f} b={b0:+.3f} → 反预乘 a={a1:.2f} b={b1:+.3f}"
                  f"  ｜ 核心掩码 a={a2:.2f} b={b2:+.3f}"
                  f"  ｜ 半透明占比 {float(fringe.float().mean())*100:.1f}%")

        # src 自洽也看一眼 alpha 是否饱和（对照）
        print()
        r_s, a_s = raster_rgbd(means, quats, scales, opac, colors,
                               d["w2cs"].float(), d["Ks"].float(), rh, rw, dev)
        for k in range(len(src_paths)):
            t, _ = I.load_image_tensor(src_paths[k], new_h, new_w, undist=undist)
            gt_k = t.clamp(0, 1).permute(1, 2, 0).to(dev)
            solid = a_s[k] > 0.95
            print(f"  [src] {os.path.basename(src_paths[k])[:22]:<24s}"
                  f"PSNR {psnr(gt_k, r_s[k]):6.2f}  "
                  f"α实心 {float(a_s[k][solid].mean()) if int(solid.sum())>64 else float('nan'):.3f}  "
                  f"solid {float(solid.float().mean())*100:.1f}%")


if __name__ == "__main__":
    main()
