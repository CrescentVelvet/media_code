#!/usr/bin/env python3
"""多尺度带通诊断：定位"PSNR 更高但看着更糊"到底丢在哪个尺度。

A7 用 sigma≈11 px 切低/高频，结论是 RES=1 在中高频上反而更好（高频 PSNR +1.70，
HF 相关 0.841→0.889），但它 @1439 的 Laplacian 锐度只有 RES=2 的 56%。
两个数不冲突的唯一解释：**RES=1 丢的是最高频**（Lap 对单像素级最敏感），
而 A7 的"高频"其实是 5~30 px 的中高频。

这里做真正的带通分解：band_k = G(sigma_k) − G(sigma_{k+1})，
对每个 band 算能量比（渲染/GT）与相关系数。能量比低 = 该尺度细节缺失；
能量比高但相关低 = 该尺度是噪声。物理尺度约 2π·sigma 像素。

用法（querysplat env）：
  python compare_4methods/A7b_bands.py
"""
from pathlib import Path

import cv2
import numpy as np

BASE = Path("/home/velvet/output/da3_gs_results/佳佳抱枕_res_ab")
REF = Path("/home/velvet/output/da3_results/_captures/佳佳抱枕/images")
FACE_BOX = (0.35, 0.105, 0.565, 0.335)
VIEWS = [0, 8, 16, 24]
# 每个 sigma 是"像素"，在 1439 口径下换算成物理尺度约 2πσ
SIGMAS = [0.6, 1.2, 2.5, 5.0, 10.0, 20.0, 40.0]


def bands(img):
    """返回 (band_list, lowpass_last)。band_k 对应 sigma_k 到 sigma_{k+1} 之间的频率。"""
    gs = [cv2.GaussianBlur(img, (0, 0), s) for s in SIGMAS]
    out = [gs[i] - gs[i + 1] for i in range(len(SIGMAS) - 1)]
    out.append(img - gs[-1])          # 最后一个：比最小 sigma 还细，即最高频
    return out


def main():
    refs = sorted(p for p in REF.iterdir() if p.suffix.lower() == ".png")
    labels = [f"{SIGMAS[i]:g}–{SIGMAS[i+1]:g}" for i in range(len(SIGMAS) - 1)]
    labels.append(f"<{SIGMAS[-1]:g}")
    # 显示顺序：最高频在前
    order = list(range(len(labels)))[::-1]

    for region in ("face", "full"):
        print(f"\n=== {'人脸区' if region == 'face' else '全图'}（1439×1919 口径，4 视角均值）===")
        hdr = f"{'频段(px)':<12}" + "".join(f"{'R2比':>7}{'R2相':>7}{'R1比':>7}{'R1相':>7}  " for _ in [0])
        print(f"{'频段 σ':<12}{'R2 能量比':>10}{'R2 相关':>8}{'R1 能量比':>10}{'R1 相关':>8}"
              f"{'  谁更准':<10}")
        print("-" * 70)

        acc: dict = {}
        for tag in ("res2", "res1"):
            ratios = np.zeros(len(labels))
            corrs = np.zeros(len(labels))
            n = 0
            for v in VIEWS:
                rp = BASE / f"final_{tag}" / "render_1439x1919" / f"view_{v:03d}.png"
                if not rp.exists():
                    continue
                rd = cv2.cvtColor(cv2.imread(str(rp)), cv2.COLOR_BGR2RGB).astype(np.float64)
                gt = cv2.cvtColor(cv2.imread(str(refs[v])), cv2.COLOR_BGR2RGB).astype(np.float64)
                if region == "face":
                    h, w = gt.shape[:2]
                    box = (int(FACE_BOX[0] * w), int(FACE_BOX[1] * h),
                           int(FACE_BOX[2] * w), int(FACE_BOX[3] * h))
                    gt = gt[box[1]:box[3], box[0]:box[2]]
                    rd = rd[box[1]:box[3], box[0]:box[2]]
                bg, br = bands(gt), bands(rd)
                for i in range(len(labels)):
                    g, r = bg[i], br[i]
                    gs, rs = g.std(), r.std()
                    ratios[i] += rs / max(gs, 1e-9)
                    # 能量过低时相关系数无意义，记 NaN
                    corrs[i] += (np.corrcoef(g.ravel(), r.ravel())[0, 1]
                                 if gs > 0.05 and rs > 0.05 else np.nan)
                n += 1
            acc[tag] = (ratios / max(n, 1), corrs / max(n, 1))

        for i in order:
            r2, c2 = acc["res2"][0][i], acc["res2"][1][i]
            r1, c1 = acc["res1"][0][i], acc["res1"][1][i]
            # "更准"：能量比接近 1 且相关系数更高
            score2 = c2 * min(r2, 1.0)
            score1 = c1 * min(r1, 1.0)
            win = "RES=1" if score1 > score2 else "RES=2"
            if abs(score1 - score2) < 0.02:
                win = "持平"
            print(f"{labels[i]:<12}{r2:>10.2f}{c2:>8.3f}{r1:>10.2f}{c1:>8.3f}{'  ' + win:<10}")

        print("说明：能量比 = 渲染该频段 std / GT 该频段 std（1.0 = 能量匹配，<1 = 细节缺失）；"
              "相关 = 该频段波形相关系数（低 = 是噪声不是细节）")


if __name__ == "__main__":
    main()
