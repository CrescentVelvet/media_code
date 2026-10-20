#!/usr/bin/env python3
"""对比 model_p07 (-r2,10k) vs model_p07_hi (-r1,30k) 的 mask 内锐度。"""
import os
import glob
import numpy as np
import cv2

R = "/home/velvet/output/som_human_results/hand10_shuangren_zhaoshou"
MASK_DIR = os.path.join(R, "colmap_refined", "persons", "p07", "masks")


def sharp_in_mask(img, mask):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    m = mask > 127
    if m.sum() < 100:
        return np.nan
    return float(np.log10(np.var(lap[m]) + 1e-6))


def iter_test(model_dir):
    base = os.path.join(R, model_dir, "test")
    for it in sorted(os.listdir(base)):
        rd = os.path.join(base, it, "renders")
        gd = os.path.join(base, it, "gt")
        if os.path.isdir(rd):
            yield it, rd, gd


def load_mask(name):
    for ext in (".png", ".jpg"):
        p = os.path.join(MASK_DIR, name.replace(".jpg", ext).replace(".png", ext))
        if os.path.exists(p):
            m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            return m
    return None


for model in ["model_p07", "model_p07_hi"]:
    for it, rd, gd in iter_test(model):
        names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(rd, "*")))
        gt_r, rd_r = [], []
        for n in names:
            rp = os.path.join(rd, n)
            gp = os.path.join(gd, n)
            if not os.path.exists(gp):
                continue
            r = cv2.imread(rp)
            g = cv2.imread(gp)
            if r is None or g is None:
                continue
            m = load_mask(n)
            if m is None:
                continue
            m = cv2.resize(m, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
            if r.shape[:2] != g.shape[:2]:
                r = cv2.resize(r, (g.shape[1], g.shape[0]))
            gt_r.append(sharp_in_mask(g, m))
            rd_r.append(sharp_in_mask(r, m))
        gt_r = np.array(gt_r)
        rd_r = np.array(rd_r)
        print("{} {}  n={}  GT {:.2f}  render {:.2f}  Δ {:+.2f}  保留率 {:.0%}".format(
            model, it, len(gt_r), np.nanmean(gt_r), np.nanmean(rd_r),
            np.nanmean(rd_r) - np.nanmean(gt_r),
            (10 ** np.nanmean(rd_r)) / (10 ** np.nanmean(gt_r))))
