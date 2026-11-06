#!/usr/bin/env python3
"""R-3 的分频段诊断：PSNR 高但看着更糊，说明 PSNR 被"平滑"骗了。

为什么需要这一步：R-3 的原始结果是
    RES=1 @1439×1919  PSNR 26.04（最高）  但锐度只有 GT 的 48.5%
    RES=2 @1439×1919  PSNR 23.32（最低）  锐度却是 GT 的 86.4%
PSNR 与锐度指向相反。只有两种可能：
  (a) RES=1 靠过度平滑降低了逐像素误差 —— 高频全丢，所以肉眼更糊；
  (b) RES=2 的高锐度是渲染噪点（椒盐/颗粒），并非真细节。
判别方法：把图分成低/高频两段，分别算 PSNR，并算渲染高频与 GT 高频的相关系数。
  - 低频 PSNR 高、高频 PSNR 低 → (a) 平滑换分数
  - 高频相关系数低（≈0）而能量高 → (b) 噪声
  - 高频相关系数高（>0.5）→ 那份锐度是真细节

用法（querysplat env）：
  python compare_4methods/A7_analyze_freq.py
"""
import json
from pathlib import Path

import cv2
import numpy as np

BASE = Path("/home/velvet/output/da3_gs_results/佳佳抱枕_res_ab")
REF = Path("/home/velvet/output/da3_results/_captures/佳佳抱枕/images")
SIZES = ["378x504", "756x1008", "1439x1919"]
TAGS = ["res2", "res1"]
VIEWS = [0, 8, 16, 24]
FACE_BOX = (0.35, 0.105, 0.565, 0.335)   # 与 A6 一致


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def split_freq(img, sigma):
    """高通 = 原图 − 高斯低通。高频以 128 为中性再算 PSNR，避免直流分量主导。"""
    low = cv2.GaussianBlur(img, (0, 0), sigma)
    high = img - low + 128.0
    return low, high


def main():
    refs = sorted(p for p in REF.iterdir() if p.suffix.lower() == ".png")
    rows = []
    for tag in TAGS:
        for sz in SIZES:
            acc = {"psnr": [], "psnr_low": [], "psnr_high": [], "corr": [],
                   "hf_ratio": [], "face_psnr": [], "face_psnr_low": [],
                   "face_psnr_high": [], "face_corr": []}
            for v in VIEWS:
                rp = BASE / f"final_{tag}" / f"render_{sz}" / f"view_{v:03d}.png"
                if not rp.exists():
                    continue
                rd = cv2.cvtColor(cv2.imread(str(rp)), cv2.COLOR_BGR2RGB).astype(np.float64)
                gt = cv2.cvtColor(cv2.imread(str(refs[v])), cv2.COLOR_BGR2RGB).astype(np.float64)
                if rd.shape != gt.shape:
                    rd = cv2.resize(rd, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LANCZOS4)
                h, w = gt.shape[:2]

                # 低通核按分辨率缩放，保证"低频"在所有档位对应同一物理尺度
                sigma = 3.0 * (w / 378.0)
                lo_g, hi_g = split_freq(gt, sigma)
                lo_r, hi_r = split_freq(rd, sigma)

                acc["psnr"].append(psnr(rd, gt))
                acc["psnr_low"].append(psnr(lo_r, lo_g))
                acc["psnr_high"].append(psnr(hi_r, hi_g))
                g = hi_g - 128.0
                r = hi_r - 128.0
                acc["corr"].append(float(np.corrcoef(g.ravel(), r.ravel())[0, 1]))
                acc["hf_ratio"].append(float(r.std() / max(g.std(), 1e-9)))

                # 人脸区
                box = (int(FACE_BOX[0] * w), int(FACE_BOX[1] * h),
                       int(FACE_BOX[2] * w), int(FACE_BOX[3] * h))
                fg, fr = gt[box[1]:box[3], box[0]:box[2]], rd[box[1]:box[3], box[0]:box[2]]
                fl_g, fh_g = split_freq(fg, sigma)
                fl_r, fh_r = split_freq(fr, sigma)
                acc["face_psnr"].append(psnr(fr, fg))
                acc["face_psnr_low"].append(psnr(fl_r, fl_g))
                acc["face_psnr_high"].append(psnr(fh_r, fh_g))
                acc["face_corr"].append(float(np.corrcoef((fh_g - 128).ravel(), (fh_r - 128).ravel())[0, 1]))

            m = {k: round(float(np.mean(v)), 4) for k, v in acc.items()}
            rows.append(dict(tag=tag, size=sz, **m))

    hdr = f"{'组':<6}{'评测':<12}{'全图':>7}{'低频':>7}{'高频':>7}{'HF相关':>8}{'HF能量比':>9}" \
          f"{'人脸':>7}{'脸低频':>7}{'脸高频':>7}{'脸HF相关':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['tag']:<6}{r['size']:<12}{r['psnr']:>7.2f}{r['psnr_low']:>7.2f}"
              f"{r['psnr_high']:>7.2f}{r['corr']:>8.3f}{r['hf_ratio']:>9.2f}"
              f"{r['face_psnr']:>7.2f}{r['face_psnr_low']:>7.2f}{r['face_psnr_high']:>7.2f}"
              f"{r['face_corr']:>9.3f}")

    print("\nΔ(RES=1 − RES=2)")
    by = {(r["tag"], r["size"]): r for r in rows}
    for sz in SIZES:
        a, b = by.get(("res1", sz)), by.get(("res2", sz))
        if not (a and b):
            continue
        print(f"  @{sz:<12} 全图 {a['psnr']-b['psnr']:+.2f}  低频 {a['psnr_low']-b['psnr_low']:+.2f}  "
              f"高频 {a['psnr_high']-b['psnr_high']:+.2f}  HF相关 {a['corr']-b['corr']:+.3f}  "
              f"| 脸 {a['face_psnr']-b['face_psnr']:+.2f}  脸高频 {a['face_psnr_high']-b['face_psnr_high']:+.2f}  "
              f"脸HF相关 {a['face_corr']-b['face_corr']:+.3f}")

    out = Path(__file__).resolve().parent / "ppt_assets" / "res_ab_freq.json"
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\n✅ {out}")


if __name__ == "__main__":
    main()
