#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
structsplat/00z_oracle_align.py — 判定 target 低分是「对齐误差」还是「几何/覆盖」

为什么需要（2026-11-20）：
  00w/00y 拿到 target 11.07~11.26dB，但 src-only Sim(3) 朝向残差只有 0.66°。
  要对外报数，必须先排除「Sim(3) 自洽却仍然把 target 相机搬错了」这种可能。

── 为什么不能用「把 target 也放进 Sim(3) 拟合」做 oracle ──
  前馈**只预测输入视角的相机**（pred_w2c 只有 2 台）；target 没有对应的预测相机，
  所以 Sim(3) 无法把 target 纳入拟合（拟合需要 src↔dst 成对）。
  首版这么写直接崩：mat1 3x12 @ mat2 6x3（4 台 GT vs 2 台 pred）。

── 改用的正确 oracle：全局刚体修正量搜索 ──
  在 src-only Sim(3) 的结果上，左乘一个世界系（模型系）刚体修正 C = [Rc | tc]：
      w2c' = C · w2c_base
  这等价于「把模型在高斯空间里整体挪一下再拍」，正好覆盖 Sim(3) 的平移/旋转误差。
  对 C 做粗→细随机搜索，最大化 target PSNR，得到 **target 可达上界**。

判据：
  若 best_PSNR − 当前_PSNR < 0.8dB  → 对齐已到底，低分来自前馈几何/覆盖，不是位姿。
  若 gap 很大                    → src-only 对齐仍有系统误差，需回去查（不太可能）。

── 同时输出 coverage ──
  2 视角重建出的高斯只覆盖两锥相交处，大角度新视角必然大片空白。
  必须在「只在被覆盖像素上」的 PSNR 与全图 PSNR 之间分开看，否则会把覆盖不足
  误算成几何误差。

用法：
  python 00z_oracle_align.py --cache .../_cache/hand01_src21.pt \
    --transforms ".../transforms.json" --images ".../image" \
    --tar-idx 21,42 --iters 600
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


def apply_sim3(gt_w2c, s, R, t):
    """GT(真值系) → 模型系：R' = R_g·Rᵀ、t' = s·t_g − R'·t。"""
    Rt = gt_w2c[:, :3, :3] @ R.transpose(-1, -2)
    tt = s * gt_w2c[:, :3, 3] - torch.einsum("sij,j->si", Rt, t)
    out = gt_w2c.clone()
    out[:, :3, :3] = Rt
    out[:, :3, 3] = tt
    return out


def rodrigues(w):
    """axis-angle (3,) numpy → (3,3) 旋转矩阵。"""
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = np.asarray(w, dtype=np.float64) / th
    Kx = np.array([[0.0, -k[2], k[1]],
                   [k[2], 0.0, -k[0]],
                   [-k[1], k[0], 0.0]], dtype=np.float64)
    return np.eye(3) + math.sin(th) * Kx + (1.0 - math.cos(th)) * (Kx @ Kx)


def left_apply(w2c, C):
    """w2c' = C · w2c（C 为 4×4 numpy 世界系刚体变换）。"""
    C_ = torch.from_numpy(C).to(w2c.dtype)
    return C_.unsqueeze(0) @ w2c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--transforms", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--tar-idx", required=True, help="逗号分隔帧号")
    ap.add_argument("--no-undistort", action="store_true")
    ap.add_argument("--iters", type=int, default=600, help="随机搜索总评价次数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-dir", default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = torch.load(args.cache, map_location="cpu", weights_only=False)
    src_paths = d["paths"]
    new_h, new_w = int(d["new_h"]), int(d["new_w"])
    rh, rw = new_h, new_w

    with open(args.transforms, "r", encoding="utf-8") as f:
        j = json.load(f)
    all_names = [os.path.basename(fr.get("file_path", fr.get("image", "")))
                 for fr in j["frames"]]
    idxs = [int(x) for x in args.tar_idx.split(",") if x.strip()]
    tar_paths = [os.path.join(args.images, all_names[i]) for i in idxs]
    if {os.path.basename(p) for p in src_paths} & {os.path.basename(p) for p in tar_paths}:
        raise SystemExit("❌ target 与 src 重叠（信息泄漏）")
    if len(src_paths) < 2:
        raise SystemExit("❌ src < 2，Sim(3) 退化")

    print(f"src {len(src_paths)}  {[os.path.basename(p) for p in src_paths]}")
    print(f"tar idx={idxs}  {[os.path.basename(p) for p in tar_paths]}")

    undist = None if args.no_undistort else I.probe_distortion(args.transforms)
    gt_src, K_src, m_src = I.load_gt_cameras(args.transforms, src_paths, new_h, new_w,
                                             undist=undist)
    gt_tar, K_tar, m_tar = I.load_gt_cameras(args.transforms, tar_paths, new_h, new_w,
                                             undist=undist)
    if len(m_tar) != len(tar_paths) or len(m_src) != len(src_paths):
        raise SystemExit("❌ transforms.json 里有图没匹配上")

    gts = []
    for p in tar_paths:
        t, _ = I.load_image_tensor(p, new_h, new_w, undist=undist)
        gts.append(t.clamp(0, 1).permute(1, 2, 0))
    gts = torch.stack(gts)
    gts_dev = gts.to(dev)

    pred_w2c = d["pred_w2c"].float()
    pred_K = d["pred_K"].float()
    means, quats, scales, opac, colors = (d["means"], d["quats"], d["scales"],
                                          d["opac"], d["colors"])
    one = torch.ones_like(colors[..., :3])

    _, info = I.align_gt_to_model(pred_w2c, gt_src, K_src, pred_K,
                                  mode="axes", reflect="auto")
    s_, R_, t_ = info["scale_gt_to_model"], info["_R"].float(), info["_t"].float()
    print(f"\nsrc-only Sim(3): s={s_:.5f} fit_resid={info['fit_resid']:.5f} "
          f"rot {info['rot_deg_mean']:.2f}° det(R)={info['det_R']:+.2f} "
          f"reflect={info['reflect']}")

    base = apply_sim3(gt_tar, s_, R_, t_)

    # 相机运动尺度（用于设定平移搜索半径）
    Cc = I._cam_centers(base)
    radius = float(torch.linalg.norm(Cc - Cc.mean(0, keepdim=True), dim=1).mean())
    rot_scale = 1.0            # deg，最细一档
    tr_scale = 0.02 * max(radius, 1e-6)

    def score(C4):
        w = left_apply(base, C4).to(dev)
        img = raster(means, quats, scales, opac, colors, w, K_tar, rh, rw, dev)
        a = raster(means, quats, scales, opac, one, w, K_tar, rh, rw, dev)[..., 0]
        ps = [psnr(gts_dev[k], img[k]) for k in range(len(tar_paths))]
        cov = float((a > 0.5).float().mean())
        return float(np.mean(ps)), ps, cov, img, a

    I4 = np.eye(4)
    p0, ps0, cov0, img0, a0 = score(I4)
    print(f"\n── 基线（C=I，即正式口径）──")
    for k in range(len(tar_paths)):
        print(f"  {os.path.basename(tar_paths[k])[:32]:<32s} PSNR {ps0[k]:6.2f}")
    print(f"  均值 {p0:.2f} dB   coverage {cov0*100:.1f}%")

    # ── 粗→细随机搜索 ──
    best_p, best_C = p0, I4.copy()
    sched = [(rot_scale * 4, tr_scale * 8), (rot_scale * 2, tr_scale * 4),
             (rot_scale, tr_scale * 2), (rot_scale * 0.4, tr_scale),
             (rot_scale * 0.15, tr_scale * 0.4)]
    per = max(1, args.iters // len(sched))
    print(f"\n── 刚体修正量随机搜索（{args.iters} 次，{len(sched)} 档）"
          f"  相机半径 {radius:.4f} ──")

    def perturb(base_C, rs_deg, ts, shrink):
        """在 base_C 上叠加小扰动。"""
        w = rng.normal(0, math.radians(rs_deg) / 2.5 * shrink, size=3)
        tc = rng.normal(0, ts / 2.5 * shrink, size=3)
        C4 = np.eye(4)
        C4[:3, :3] = rodrigues(w)
        C4[:3, 3] = tc
        return C4 @ base_C

    # 第一档：绕单位阵大范围扫；后续档：在当前最优附近细化
    radii = [1.0] + [0.35, 0.15, 0.06, 0.02]
    for lvl, (rs_deg, ts) in enumerate(sched):
        anchor = I4 if lvl == 0 else best_C
        acc = 0
        for _ in range(per):
            C4 = perturb(anchor, rs_deg, ts, radii[lvl])
            p, _, _, _, _ = score(C4)
            if p > best_p + 1e-6:
                best_p, best_C = p, C4.copy()
                acc += 1
        print(f"  档 {lvl}: rot σ≈{rs_deg/2.5*radii[lvl]:.2f}° "
              f"trans σ≈{ts/2.5*radii[lvl]:.4f} → 更新 {acc}/{per} 次，"
              f"best {best_p:.2f} dB")

    gap = best_p - p0
    pb, psb, covb, imgb, ab = score(best_C)
    print(f"\n── 搜索到的最优修正 ──")
    wv = best_C[:3, :3]
    ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(wv) - 1) / 2))))
    print(f"  旋转 {ang:.2f}°  平移 {np.linalg.norm(best_C[:3,3]):.4f} "
          f"(= 相机半径的 {np.linalg.norm(best_C[:3,3])/(radius+1e-12)*100:.1f}%)")
    for k in range(len(tar_paths)):
        mk = ab[k] > 0.5
        print(f"  {os.path.basename(tar_paths[k])[:32]:<32s} PSNR {psb[k]:6.2f} "
              f"cov {float(mk.float().mean())*100:5.1f}%  "
              f"PSNR覆盖 {psnr(gts_dev[k], imgb[k], mk):6.2f}")

    print("\n" + "=" * 72)
    print(f"正式口径(C=I) {p0:.2f} dB  →  oracle 上界 {pb:.2f} dB   差 {gap:+.2f} dB")
    if abs(gap) < 0.8:
        print("  ✅ 对齐已到底：target 低分**不是对齐/位姿误差**，")
        print("     而是前馈几何在大角度新视角上的外推误差 + 覆盖不足。")
    else:
        print("  ⚠️ 存在更优修正量 → src-only 对齐仍有系统误差，需继续查。")
    print("=" * 72)
    print("⚠️ 口径边界：target 相机=GT(transforms.json)+src-only Sim(3)；")
    print("   前馈只预测输入视角相机 → 衡量重建几何精度，不是位姿估计能力。")

    out_dir = args.save_dir or os.path.join(os.path.dirname(args.cache), "oracle")
    os.makedirs(out_dir, exist_ok=True)
    from torchvision.utils import save_image
    for k, p in enumerate(tar_paths):
        nm = f"{idxs[k]:03d}_{os.path.splitext(os.path.basename(p))[0]}"
        save_image(img0[k].permute(2, 0, 1), os.path.join(out_dir, nm + "_base.png"))
        save_image(imgb[k].permute(2, 0, 1), os.path.join(out_dir, nm + "_oracle.png"))
        save_image(gts[k].permute(2, 0, 1), os.path.join(out_dir, nm + "_gt.png"))
    with open(os.path.join(out_dir, "oracle.json"), "w",
              encoding="utf-8", newline="\n") as f:
        json.dump({
            "src": [os.path.basename(p) for p in src_paths], "tar_idx": idxs,
            "align": {k: v for k, v in info.items() if not k.startswith("_")},
            "base_psnr_mean": p0, "base_psnr_each": ps0, "base_coverage": cov0,
            "oracle_psnr_mean": pb, "oracle_psnr_each": psb, "oracle_coverage": covb,
            "gap": gap, "cam_radius": radius,
            "C4": best_C.tolist(),
            "note": "target 相机=GT+src-only Sim(3)；oracle 为刚体修正量搜索的上界",
        }, f, ensure_ascii=False, indent=2)
    print(f"💾 {out_dir}")


if __name__ == "__main__":
    main()
