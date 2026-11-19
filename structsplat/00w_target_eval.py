#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
structsplat/00w_target_eval.py — 官方口径：src 重建 + 独立 held-out target 视角渲染

为什么必须单独做（2026-11-20）：
  infer.py 的自检是「渲染输入视角 vs 输入图」，只能证明链路自洽
  （相机/反投影/渲染三者一致）；输入视角本来就是高斯反投影的来源，重渲自己天然高分。
  官方评测（annotation src=2 / tar=2）是：喂 2 张重建，另取**数据集里别的帧**渲染。
  只有这个数字能对外报。

── 硬纪律（本项目已踩过）──
  ① Sim(3) 只用 **src 的 ≥2 台相机**估计，绝不碰 target。
     单台 src 会退化（无基线 → 尺度不可观测、旋转只剩朝向），实测 gt 分支塌到 8.92dB。
  ② target 图**不能来自高斯缓存**，否则等于泄漏（高斯已含该视角信息）。
     所以 target 图路径由 --tar-images 单独给，只复用 transforms.json 的相机。

── 口径边界（必须对外说清）──
  StructSplat 前馈只预测**输入视角**的相机，不能预测未见视角。所以 target 相机只能
  来自 transforms.json（GT），再用 src-only Sim(3) 搬进模型坐标系。
  → 这不是「纯 pose-free 新视角」，而是「模型几何 + GT target 相机」，
    衡量的是「前馈重建的场景几何有多准」，不是「位姿估计有多准」。
    位姿估计能力由 infer.py 的自洽 PSNR（输入视角重渲，2 图 22.46dB）体现。

用法：
  python 00w_target_eval.py \
    --cache ~/output/.../_cache/hand01_pf2.pt \
    --transforms ".../transforms.json" \
    --tar-images ".../image/B.heic,.../image/D.heic"
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


def render(means, quats, scales, opac, colors, w2c, K, rh, rw, dev, bg=0.0):
    from gsplat import rasterization
    with torch.no_grad():
        out, _, _ = rasterization(
            means=means.to(dev), quats=quats.to(dev), scales=scales.to(dev),
            opacities=opac.to(dev), colors=colors.to(dev),
            viewmats=w2c.to(dev).float(), Ks=K.to(dev).float(),
            width=rw, height=rh, render_mode="RGB",
            packed=False,
            backgrounds=torch.full((w2c.shape[0], 3), bg,
                                   dtype=torch.float32, device=dev))
    return out[..., :3].clamp(0, 1)


def psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return float("inf") if mse <= 1e-12 else 10 * math.log10(1.0 / mse)


def ssim_of(a, b):
    try:
        from skimage.metrics import structural_similarity as s
        return float(s(a.cpu().numpy(), b.cpu().numpy(), channel_axis=2, data_range=1.0))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True,
                    help="infer.py --dump-forward 的 .pt（只含 src 视角）")
    ap.add_argument("--transforms", required=True)
    ap.add_argument("--tar-images", required=True,
                    help="held-out target 图路径，逗号分隔；**不得**是缓存里的图")
    ap.add_argument("--no-undistort", action="store_true")
    ap.add_argument("--render-size", type=int, default=None)
    ap.add_argument("--save-dir", default=None)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = torch.load(args.cache, map_location="cpu", weights_only=False)
    src_paths = d["paths"]
    new_h, new_w = int(d["new_h"]), int(d["new_w"])
    print(f"src 缓存 {len(src_paths)} 视角  输入 {new_h}×{new_w}")
    print(f"  src: {[os.path.basename(p) for p in src_paths]}")

    tar_paths = [p.strip() for p in args.tar_images.split(",") if p.strip()]
    _s = {os.path.basename(p) for p in src_paths}
    _t = {os.path.basename(p) for p in tar_paths}
    if _s & _t:
        raise SystemExit(f"❌ target 与 src 重叠：{_s & _t}（会造成信息泄漏）")
    if len(src_paths) < 2:
        raise SystemExit("❌ src 少于 2 台相机，Sim(3) 退化（尺度与旋转不可观测）")
    print(f"tar {len(tar_paths)} 视角: {[os.path.basename(p) for p in tar_paths]}")

    rw = args.render_size or new_w
    rh = int(round(rw * new_h / new_w))
    print(f"  渲染 {rh}×{rw}")

    undist = None if args.no_undistort else I.probe_distortion(args.transforms)

    # ---------- target 真值图 ----------
    gts = []
    for p in tar_paths:
        t, _ = I.load_image_tensor(p, new_h, new_w, undist=undist)
        t = F.interpolate(t.unsqueeze(0), (rh, rw), mode="bilinear",
                          align_corners=False, antialias=True)[0]
        gts.append(t.clamp(0, 1).permute(1, 2, 0))
    gts = torch.stack(gts)
    gts_dev = gts.to(dev)          # 渲染在 cuda，真值必须同 device 才能算 PSNR
    print(f"  真值图 {tuple(gts.shape)}")

    # ---------- 相机 ----------
    gt_src, K_src, m_src = I.load_gt_cameras(args.transforms, src_paths, new_h, new_w,
                                             undist=undist)
    gt_tar, K_tar, m_tar = I.load_gt_cameras(args.transforms, tar_paths, new_h, new_w,
                                             undist=undist)
    print(f"  GT 相机匹配 src {len(m_src)}/{len(src_paths)}  tar {len(m_tar)}/{len(tar_paths)}")
    if len(m_tar) != len(tar_paths):
        raise SystemExit("❌ target 图在 transforms.json 里没全部匹配上，检查文件名")

    pred_w2c = d["pred_w2c"].float()
    pred_K = d["pred_K"].float()

    # Sim(3)：**仅 src**
    _, info = I.align_gt_to_model(pred_w2c, gt_src, K_src, pred_K,
                                  mode="axes", reflect="auto")
    s_, R_, t_ = info["scale_gt_to_model"], info["_R"].float(), info["_t"].float()
    print(f"  Sim(3) [src-only]: s={s_:.5f} 朝向残差 {info['rot_deg_mean']:.2f}° "
          f"det(R)={info['det_R']:+.2f} reflect={info['reflect']}")

    # 同一个变换作用到 target 相机
    Rt = gt_tar[:, :3, :3] @ R_.transpose(-1, -2)
    tt = s_ * gt_tar[:, :3, 3] - torch.einsum("sij,j->si", Rt, t_)
    tar_w2c = gt_tar.clone()
    tar_w2c[:, :3, :3] = Rt
    tar_w2c[:, :3, 3] = tt

    s_r = rw / new_w
    K_tar_r = K_tar.clone()
    K_tar_r[:, 0, :] *= s_r
    K_tar_r[:, 1, :] *= s_r

    # ---------- 渲染 + 指标 ----------
    imgs = render(d["means"], d["quats"], d["scales"], d["opac"], d["colors"],
                  tar_w2c, K_tar_r, rh, rw, dev)
    ps = [psnr(gts_dev[k], imgs[k]) for k in range(len(tar_paths))]
    ss = [ssim_of(gts[k], imgs[k].cpu()) for k in range(len(tar_paths))]

    print("\n" + "=" * 72)
    print(f"{'视角':<44s} {'PSNR':>8s} {'SSIM':>8s}")
    print("-" * 72)
    for k in range(len(tar_paths)):
        nm = os.path.basename(tar_paths[k])[:42]
        sv = f"{ss[k]:.4f}" if ss[k] is not None else "n/a"
        print(f"{nm:<44s} {ps[k]:>8.2f} {sv:>8s}")
    print("-" * 72)
    print(f"{'均值':<44s} {float(np.mean(ps)):>8.2f}")
    print("=" * 72)
    print("⚠️  口径：target 相机=GT(transforms.json) + src-only Sim(3)；")
    print("   前馈只预测输入视角相机，不预测未见视角 → 衡量的是重建几何精度，非位姿估计。")

    out_dir = args.save_dir or os.path.join(os.path.dirname(args.cache), "target_eval")
    os.makedirs(out_dir, exist_ok=True)
    from torchvision.utils import save_image
    for k, p in enumerate(tar_paths):
        nm = f"{k:03d}_{os.path.splitext(os.path.basename(p))[0]}.png"
        save_image(imgs[k].permute(2, 0, 1), os.path.join(out_dir, nm))
        save_image(gts[k].permute(2, 0, 1), os.path.join(out_dir, "gt_" + nm))
    meta = {
        "src": [os.path.basename(p) for p in src_paths],
        "tar": [os.path.basename(p) for p in tar_paths],
        "render": [rh, rw],
        "align": {k: v for k, v in info.items() if not k.startswith("_")},
        "psnr": ps, "psnr_mean": float(np.mean(ps)),
        "ssim": [x for x in ss if x is not None],
        "note": "target 相机=GT(transforms.json)+src-only Sim(3)；前馈不预测未见视角",
    }
    with open(os.path.join(out_dir, "metrics.json"), "w",
              encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"💾 {out_dir}")


if __name__ == "__main__":
    main()
