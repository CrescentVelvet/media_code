#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
structsplat/00y_target_sweep.py — target 评测的**覆盖诊断** + 多组 target 扫描

为什么需要（2026-11-20）：
  00w 报出 target 11.26dB，但 src-only Sim(3) 的朝向残差只有 0.66°（位姿没问题）。
  剩下唯一能解释低分的因素是**覆盖**：2 张图重建出的高斯只覆盖两锥相交的区域，
  大角度新视角必然大片空白。空白区的 PSNR 就是「黑图 vs 真值」，会把均值拖到 10~11dB，
  与「几何准不准」无关。
  → 必须把 alpha 覆盖率单独量出来，并给出「只在被覆盖像素上」的 PSNR。
    否则对外报的 target 数字是错的（把覆盖不足算成几何误差）。

核心判据：
  coverage = mean(alpha > 0.5)          # 有多少像素被高斯真正覆盖
  psnr_cov = 只在 alpha>0.5 上算 PSNR   # 覆盖区的几何/纹理是否准
  若 psnr_cov >> psnr_all 且 coverage 低 → 是覆盖问题，不是位姿/几何问题。

用法：
  python 00y_target_sweep.py --cache .../_cache/hand01_src21.pt \
    --transforms ".../transforms.json" \
    --groups "near_src0:1,80" "far:21,42" "mid:10,30,50,70"
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

SD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SD)
import infer as I  # noqa: E402


def raster(means, quats, scales, opac, colors, w2c, K, rh, rw, dev, bg=0.0):
    from gsplat import rasterization
    with torch.no_grad():
        out, _, _ = rasterization(
            means=means.to(dev), quats=quats.to(dev), scales=scales.to(dev),
            opacities=opac.to(dev), colors=colors.to(dev),
            viewmats=w2c.to(dev).float(), Ks=K.to(dev).float(),
            width=rw, height=rh, render_mode="RGB", packed=False,
            backgrounds=torch.full((w2c.shape[0], 3), bg,
                                   dtype=torch.float32, device=dev))
    return out[..., :3].clamp(0, 1)


def psnr(a, b, mask=None):
    if mask is not None:
        if int(mask.sum()) < 64:
            return float("nan")
        a, b = a[mask], b[mask]
    mse = float(((a - b) ** 2).mean())
    return float("inf") if mse <= 1e-12 else 10 * math.log10(1.0 / mse)


def ssim_of(a, b):
    try:
        from skimage.metrics import structural_similarity as s
        return float(s(a.cpu().numpy(), b.cpu().numpy(), channel_axis=2, data_range=1.0))
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--transforms", required=True)
    ap.add_argument("--images", required=True, help="image 目录（target 图从这里取）")
    ap.add_argument("--groups", nargs="+", required=True,
                    help='每组 "名字:idx,idx"（idx 是 82 帧里的顺序号）')
    ap.add_argument("--no-undistort", action="store_true")
    ap.add_argument("--save-dir", default=None)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = torch.load(args.cache, map_location="cpu", weights_only=False)
    src_paths = d["paths"]
    new_h, new_w = int(d["new_h"]), int(d["new_w"])
    rh, rw = new_h, new_w
    print(f"src {len(src_paths)} 视角  输入 {new_h}×{new_w}  渲染 {rh}×{rw}")
    print(f"  src: {[os.path.basename(p) for p in src_paths]}")

    # 全帧顺序（transforms.json 的 frames 顺序 = 82 帧）
    with open(args.transforms, "r", encoding="utf-8") as f:
        j = json.load(f)
    all_names = [os.path.basename(fr.get("file_path", fr.get("image", "")))
                 for fr in j["frames"]]
    print(f"  transforms 共 {len(all_names)} 帧")

    undist = None if args.no_undistort else I.probe_distortion(args.transforms)

    gt_src, K_src, m_src = I.load_gt_cameras(args.transforms, src_paths, new_h, new_w,
                                             undist=undist)
    pred_w2c = d["pred_w2c"].float()
    pred_K = d["pred_K"].float()
    _, info = I.align_gt_to_model(pred_w2c, gt_src, K_src, pred_K,
                                  mode="axes", reflect="auto")
    s_, R_, t_ = info["scale_gt_to_model"], info["_R"].float(), info["_t"].float()
    print(f"Sim(3) [src-only] s={s_:.5f} 朝向残差 {info['rot_deg_mean']:.2f}° "
          f"det(R)={info['det_R']:+.2f} reflect={info['reflect']}")

    means, quats = d["means"], d["quats"]
    scales, opac, colors = d["scales"], d["opac"], d["colors"]

    # ── src 自洽（上界参照）──
    src_imgs = []
    for p in src_paths:
        t, _ = I.load_image_tensor(p, new_h, new_w, undist=undist)
        src_imgs.append(t.clamp(0, 1).permute(1, 2, 0))
    g_src = torch.stack(src_imgs).to(dev)
    r_src = raster(means, quats, scales, opac, colors, d["w2cs"].float(),
                   d["Ks"].float(), rh, rw, dev)
    a_src = raster(means, quats, scales, opac,
                   torch.ones_like(colors[..., :3]), d["w2cs"].float(),
                   d["Ks"].float(), rh, rw, dev)[..., 0]
    print("\n── src 自洽（输入视角重渲，仅证明链路自洽）──")
    for k in range(len(src_paths)):
        cov = float((a_src[k] > 0.5).float().mean())
        print(f"  {os.path.basename(src_paths[k])[:34]:<34s} "
              f"PSNR {psnr(g_src[k], r_src[k]):6.2f}  cov {cov*100:5.1f}%")

    out_dir = args.save_dir or os.path.join(os.path.dirname(args.cache), "target_sweep")
    os.makedirs(out_dir, exist_ok=True)
    from torchvision.utils import save_image
    report = {}

    for spec in args.groups:
        if ":" not in spec:
            raise SystemExit(f'❌ 组格式应为 "名字:idx,idx"，收到 {spec}')
        gname, idxs = spec.split(":", 1)
        idxs = [int(x) for x in idxs.split(",") if x.strip()]
        tar_paths = [os.path.join(args.images, all_names[i]) for i in idxs]
        _s = {os.path.basename(p) for p in src_paths}
        _t = {os.path.basename(p) for p in tar_paths}
        if _s & _t:
            raise SystemExit(f"❌ 组 {gname} 与 src 重叠：{_s & _t}")
        if len(gt_src) < 2:
            raise SystemExit("❌ src < 2，Sim(3) 退化")

        gts = []
        for p in tar_paths:
            t, _ = I.load_image_tensor(p, new_h, new_w, undist=undist)
            gts.append(t.clamp(0, 1).permute(1, 2, 0))
        gts = torch.stack(gts)
        gts_dev = gts.to(dev)

        gt_tar, K_tar, m_tar = I.load_gt_cameras(args.transforms, tar_paths,
                                                 new_h, new_w, undist=undist)
        if len(m_tar) != len(tar_paths):
            raise SystemExit(f"❌ 组 {gname} 有图在 transforms.json 里没匹配上")
        Rt = gt_tar[:, :3, :3] @ R_.transpose(-1, -2)
        tt = s_ * gt_tar[:, :3, 3] - torch.einsum("sij,j->si", Rt, t_)
        tar_w2c = gt_tar.clone()
        tar_w2c[:, :3, :3] = Rt
        tar_w2c[:, :3, 3] = tt

        imgs = raster(means, quats, scales, opac, colors, tar_w2c, K_tar, rh, rw, dev)
        alpha = raster(means, quats, scales, opac, torch.ones_like(colors[..., :3]),
                       tar_w2c, K_tar, rh, rw, dev)[..., 0]

        print(f"\n── 组 {gname}  idx={idxs} ──")
        print(f"  {'视角':<30s} {'PSNR全':>8s} {'cov%':>7s} {'PSNR覆盖':>9s} {'SSIM':>7s}")
        rows = []
        for k in range(len(tar_paths)):
            mk = alpha[k] > 0.5
            cov = float(mk.float().mean())
            p_all = psnr(gts_dev[k], imgs[k])
            p_cov = psnr(gts_dev[k], imgs[k], mk)
            ss = ssim_of(gts[k], imgs[k].cpu())
            rows.append({"name": os.path.basename(tar_paths[k]), "idx": idxs[k],
                         "psnr_all": p_all, "coverage": cov,
                         "psnr_cov": p_cov, "ssim": ss})
            print(f"  {os.path.basename(tar_paths[k])[:30]:<30s} {p_all:>8.2f} "
                  f"{cov*100:>6.1f}% {p_cov:>9.2f} {ss:>7.4f}")
            nm = f"{gname}_{idxs[k]:03d}_{os.path.splitext(os.path.basename(tar_paths[k]))[0]}"
            save_image(imgs[k].permute(2, 0, 1), os.path.join(out_dir, nm + "_pred.png"))
            save_image(gts[k].permute(2, 0, 1), os.path.join(out_dir, nm + "_gt.png"))
        print(f"  {'均值':<30s} {np.mean([r['psnr_all'] for r in rows]):>8.2f} "
              f"{np.mean([r['coverage'] for r in rows])*100:>6.1f}% "
              f"{np.nanmean([r['psnr_cov'] for r in rows]):>9.2f} "
              f"{np.nanmean([r['ssim'] for r in rows]):>7.4f}")
        report[gname] = rows

    with open(os.path.join(out_dir, "sweep.json"), "w",
              encoding="utf-8", newline="\n") as f:
        json.dump({"src": [os.path.basename(p) for p in src_paths],
                   "align": {k: v for k, v in info.items() if not k.startswith("_")},
                   "groups": report}, f, ensure_ascii=False, indent=2)
    print(f"\n💾 {out_dir}")


if __name__ == "__main__":
    main()
