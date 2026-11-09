#!/usr/bin/env python3
"""AUDIT-21 · 人脸窗口订正与重测（含非人脸对照）。

问题（2026-11-09 发现）：audit_14 起沿用的固定人脸窗
FACE = (0.35, 0.105, 0.565, 0.335)（归一化）只对 view 0 大致成立。
按 3dgs 渲染图 + 像素刻度网格人工标定后实测三个人头位置差别很大：
    view 0   x 168~212, y  78~140   （归一化 0.444~0.561 / 0.155~0.278）
    view 8   x 218~268, y  78~145   （归一化 0.577~0.709 / 0.155~0.288）
    view 16  x 178~235, y  85~150   （归一化 0.471~0.622 / 0.169~0.298）
旧框在 view 8 落在书架（x 132~213），**根本不是人脸**，而且书架纹理比人脸锐得多
→ 此前所有「人脸锐度」被整体抬高（+95 ~ +180）。

本脚本用逐视角人脸窗重测全部现有渲染 PNG（同一组视角、同一渲染口径），
并同时给出旧框读数，让口径错误的量级可见。

口径：
  · 人脸 PSNR / 锐度 = 逐视角人脸窗裁切上算（裁切后算 Laplacian 方差）
  · 非人脸 PSNR      = 同视角全图去掉人脸窗
  · 非人脸锐度       = 整图 Laplacian 后按非人脸掩膜取方差
                       （与裁切法略有差异，但各档同法，可比）
只读 PNG，不重新渲染。
"""
import glob
import json
import os

import cv2
import numpy as np
from PIL import Image

SC = os.environ["SC"]
REF = os.environ.get("REF_NPZ",
                     "/home/velvet/output/da3_gs_results/佳佳抱枕_p1m/official_dump.npz")
OUT = os.environ.get("OUT", os.path.join(SC, "face_recheck"))

FACE_V = {0: (0.415, 0.140, 0.585, 0.300),
          8: (0.515, 0.145, 0.665, 0.310),
          16: (0.430, 0.155, 0.600, 0.310)}
FACE_OLD = (0.35, 0.105, 0.565, 0.335)

CASES = [
    ("i12_fullpts_sh", "official_export", "ply_i12_fullpts_sh"),
    ("i01_fullpts_dc", "official_export", "ply_i01_fullpts_dc"),
    ("i02_pruned_sh", "official_export", "ply_i02_pruned_sh"),
    ("hist_pruned_dc", "official_export", "ply_hist_pruned_dc"),
    ("i03_3dgs", "i03_2dgs", "3dgs"),
    ("i03_2dgs_reorder", "i03_2dgs", "2dgs"),
    ("i03b_3dgs", "i03b_oracle", "3dgs"),
    ("i03b_2dgs_reorder", "i03b_oracle", "reorder"),
    ("i03b_2dgs_oracle", "i03b_oracle", "oracle2"),
    ("i03b_2dgs_perview", "i03b_oracle", "oracle1"),
]


def log(*a):
    print(*a, flush=True)


gt = np.load(REF)["processed_images"]
H, W = gt.shape[1:3]


def box(b):
    return (int(b[0] * W), int(b[1] * H), int(b[2] * W), int(b[3] * H))


def crop_of(b):
    x0, y0, x1, y1 = box(b)
    return (slice(y0, y1), slice(x0, x1))


def mask_of(b):
    m = np.zeros((H, W), bool)
    x0, y0, x1, y1 = box(b)
    m[y0:y1, x0:x1] = True
    return m


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 10 * np.log10(255.0 ** 2 / max(mse, 1e-12))


def lap(rgb):
    g = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)
    return cv2.Laplacian(g, cv2.CV_64F)


for v in FACE_V:
    log(f"  view {v:>2} 人脸窗 {box(FACE_V[v])}  裁切 {crop_of(FACE_V[v])}")
log(f"  旧框（统一）{box(FACE_OLD)}  ← view 8 落在书架上")

res = {}
log(f"\n{'档':>18} {'全图':>7} {'人脸':>7} {'非人脸':>7} {'旧框':>7}"
    f" {'脸锐':>6} {'非脸锐':>7} {'旧框锐':>7}")
for tag, sub, pref in CASES:
    d = os.path.join(SC, sub)
    full, fnew, nface, fold, sf, snf, sold = [], [], [], [], [], [], []
    for v in FACE_V:
        fs = glob.glob(os.path.join(d, f"{pref}_view{v:03d}.png"))
        if not fs:
            continue
        im = np.asarray(Image.open(fs[0])).astype(np.float64)
        gtv = gt[v].astype(np.float64)
        c = crop_of(FACE_V[v])
        m = mask_of(FACE_V[v])
        co = crop_of(FACE_OLD)
        L = lap(im)
        full.append(psnr(im, gtv))
        fnew.append(psnr(im[c], gtv[c]))
        nface.append(psnr(im[~m], gtv[~m]))
        fold.append(psnr(im[co], gtv[co]))
        sf.append(float(np.var(lap(im[c]))))
        snf.append(float(np.var(L[~m])))
        sold.append(float(np.var(lap(im[co]))))
    if not full:
        log(f"{tag:>18}  （缺 PNG，跳过）")
        continue
    row = dict(psnr=float(np.mean(full)), face=float(np.mean(fnew)),
               nonface=float(np.mean(nface)), face_old=float(np.mean(fold)),
               sharp_face=float(np.mean(sf)), sharp_nonface=float(np.mean(snf)),
               sharp_old=float(np.mean(sold)))
    res[tag] = row
    log(f"{tag:>18} {row['psnr']:>7.2f} {row['face']:>7.2f} {row['nonface']:>7.2f} "
        f"{row['face_old']:>7.2f} {row['sharp_face']:>6.0f} {row['sharp_nonface']:>7.0f} "
        f"{row['sharp_old']:>7.0f}")

log("\n订正幅度（旧框 − 新框）：")
for tag, r in res.items():
    log(f"  {tag:>18}: 人脸 PSNR {r['face_old'] - r['face']:+.2f} dB  "
        f"人脸锐度 {r['sharp_old'] - r['sharp_face']:+.0f}")

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "face_recheck.json"), "w", encoding="utf-8", newline="\n") as f:
    json.dump(dict(face_v={str(k): box(v) for k, v in FACE_V.items()},
                   face_old=box(FACE_OLD), rows=res), f, indent=1, ensure_ascii=False)
log(f"\n💾 {os.path.join(OUT, 'face_recheck.json')}")
