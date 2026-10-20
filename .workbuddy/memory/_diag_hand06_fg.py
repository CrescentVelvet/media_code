#!/usr/bin/env python3
"""hand06 per-person 模型的 FG PSNR + 手部区域 PSNR/锐度对比。"""
import os
import glob
import numpy as np
import cv2

R = "/home/velvet/output/som_human_results/hand06_nvsheng_xiezi"
PMASK = os.path.join(R, "colmap_refined", "persons", "p00", "masks")
HMASK = os.path.join(R, "colmap_refined", "hands", "merged")
MODELS = ["model_p00_base", "model_p00_hand3", "model_p00_hand5lr"]


def load_mask(d, name, hw):
    p = os.path.join(d, os.path.splitext(name)[0] + ".png")
    if not os.path.exists(p):
        return None
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if m.shape[:2] != hw:
        m = cv2.resize(m, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
    return m > 127


def psnr(a, b, m):
    if m.sum() < 50:
        return np.nan
    d = (a.astype(np.float32) - b.astype(np.float32))
    mse = (d[m] ** 2).mean()
    return 10 * np.log10(255.0 ** 2 / max(mse, 1e-9))


def sharp(img, m):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    return float(np.log10(np.var(lap[m]) + 1e-6))


for model in MODELS:
    for it in ["ours_10000"]:
        rd = os.path.join(R, model, "test", it, "renders")
        gd = os.path.join(R, model, "test", it, "gt")
        if not os.path.isdir(rd):
            print("{} {}: 无渲染".format(model, it))
            continue
        fg_ps, hd_ps, gt_hs, rd_hs = [], [], [], []
        for rp in sorted(glob.glob(os.path.join(rd, "*"))):
            n = os.path.basename(rp)
            gp = os.path.join(gd, n)
            if not os.path.exists(gp):
                continue
            r, g = cv2.imread(rp), cv2.imread(gp)
            if r is None or g is None:
                continue
            pm = load_mask(PMASK, n, g.shape[:2])
            hm = load_mask(HMASK, n, g.shape[:2])
            if pm is None:
                continue
            fg_ps.append(psnr(r, g, pm))
            if hm is not None and hm.sum() > 50:
                hd_ps.append(psnr(r, g, hm))
                gt_hs.append(sharp(g, hm))
                rd_hs.append(sharp(r, hm))
        print("{}: FG_PSNR {:.2f} | 手部 PSNR {:.2f} | 手部锐度 GT {:.2f} render {:.2f} (保留 {:.0%})".format(
            model, np.nanmean(fg_ps), np.nanmean(hd_ps),
            np.nanmean(gt_hs), np.nanmean(rd_hs),
            10 ** (np.nanmean(rd_hs) - np.nanmean(gt_hs))))
