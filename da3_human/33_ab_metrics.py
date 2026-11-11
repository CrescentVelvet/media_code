#!/usr/bin/env python
# 33_ab_metrics.py — 两版批量产物的 30 号评测指标 A/B 对比（依赖 31 号已跑出的 multiview_report.json）
#
# 用法: python 33_ab_metrics.py <根A> <根B>
# 输出: 各指标 mean(A) / mean(B) / 逐包 Δ 统计 + Δ 极值包
import glob
import json
import os
import statistics as st
import sys

KEYS = ["psnr", "fg", "ssim", "lpips", "sharp"]
KEYMAP = {"psnr": ("psnr", "mean"), "fg": ("fg_psnr", "mean"),
          "ssim": ("ssim", "mean"), "lpips": ("lpips", "mean"),
          "sharp": ("sharp_ratio", None)}


def load(root):
    out = {}
    for p in sorted(glob.glob(os.path.join(root, "*", "eval", "multiview_report.json"))):
        d = json.load(open(p))
        a = d["aggregate"]
        row = {}
        for k, (m, sub) in KEYMAP.items():
            row[k] = a[m][sub] if sub else a[m]
        out[d["scene"]] = row
    return out


def main():
    ra, rb = sys.argv[1], sys.argv[2]
    va, vb = load(ra), load(rb)
    common = sorted(set(va) & set(vb))
    print(f"A={ra} ({len(va)})  B={rb} ({len(vb)})  共有={len(common)}")
    only_a = sorted(set(va) - set(vb))
    only_b = sorted(set(vb) - set(va))
    if only_a:
        print(f"⚠️ 仅 A 有: {only_a}")
    if only_b:
        print(f"⚠️ 仅 B 有: {only_b}")
    print()
    hdr = ("metric", "mean(A)", "mean(B)", "mean_dB", "max|Δ|", "n|Δ|>0.05")
    print(f"{hdr[0]:>8} {hdr[1]:>8} {hdr[2]:>8} {hdr[3]:>9} {hdr[4]:>8} {hdr[5]:>10}")
    for k in KEYS:
        d = [vb[s][k] - va[s][k] for s in common]
        n = sum(1 for x in d if abs(x) > 0.05)
        print(f"{k:>8} {st.mean(va[s][k] for s in common):8.3f} "
              f"{st.mean(vb[s][k] for s in common):8.3f} "
              f"{st.mean(d):+9.4f} {max(abs(x) for x in d):8.4f} {n:>8}/55")
    d = sorted(((vb[s]["psnr"] - va[s]["psnr"], s[:12]) for s in common))
    print()
    print("PSNR Δ 最小3:", [(s, round(x, 3)) for x, s in d[:3]])
    print("PSNR Δ 最大3:", [(s, round(x, 3)) for x, s in d[-3:]])


if __name__ == "__main__":
    main()
