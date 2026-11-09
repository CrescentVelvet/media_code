#!/usr/bin/env python3
"""AUDIT-18 · 盘厚度体检：DA3 的"薄盘"到底有多薄？2DGS 的假设成立吗？

动机：I-03b 给了 oracle 法向，2DGS 反而更差（20.92 vs 21.82）。若法向不是瓶颈，
就得回到更基础的假设：**2DGS 把每个高斯当成零厚度 surfel**
（`Projection2DGSFused.cu:152` 第三维硬编码 1.0，即该轴完全不参与投影）。

这个假设只在 s_min / s_mid ≪ 1 时成立。本脚本直接量：
  T1  s_min / s_max、s_min / s_mid 的分布（分位数）
  T2  按 s_min/s_mid 分档统计占比：<0.05（合格 surfel）/ 0.05~0.3 / >0.3（块状）
  T3  被 2DGS 丢掉的那一维，折算到屏幕像素是多少（@378 与 @1439 两个分辨率）
  T4  opacity 与厚度的关系（薄的地方是不是本来就没权重）

判据（写死）：若 s_min/s_mid 中位 > 0.15 → 「不是薄盘」成立，
2DGS 的零厚度假设不成立，I-03/I-03b 的负结果与法向无关，I-04 必须先让头变薄。

只读，不改任何东西。
"""
import os
import re

import numpy as np

PLY = os.environ["PLY"]
RES = float(os.environ.get("RES_SCALE", "3.806"))   # 378 → 1439 的倍率（1/scale_factor 近似）


def log(*a):
    print(*a, flush=True)


def read_header(path):
    with open(path, "rb") as f:
        head = b""
        while b"end_header" not in head:
            head += f.read(1)
        tail = f.read(1)
        while tail not in (b"\n", b""):
            head += tail
            tail = f.read(1)
        head += tail
        return head


head = read_header(PLY)
text = head.decode("ascii", "replace")
n = int(re.search(r"element vertex (\d+)", text).group(1))
names = re.findall(r"^property float (\w+)$", text, re.M)
dt = np.dtype([(nm, "<f4") for nm in names])
with open(PLY, "rb") as f:
    f.seek(len(head))
    raw = np.fromfile(f, dtype=dt, count=n)

scales = np.exp(np.stack([raw[f"scale_{i}"] for i in range(3)], 1).astype(np.float64))
opac = 1.0 / (1.0 + np.exp(-raw["opacity"].astype(np.float64)))

s_min = scales.min(1)
s_max = scales.max(1)
s_mid = np.median(scales, 1)

r_mid = s_min / np.maximum(s_mid, 1e-12)
r_max = s_min / np.maximum(s_max, 1e-12)

log(f"📦 {os.path.basename(os.path.dirname(PLY))}/{os.path.basename(PLY)}: {n:,} 点")
log(f"\n[T1] 尺寸（单位 m，scene 尺度）")
for nm, v in (("s_min", s_min), ("s_mid", s_mid), ("s_max", s_max)):
    log(f"  {nm}: 中位 {np.median(v)*1000:8.3f} mm   "
        f"p10 {np.percentile(v,10)*1000:8.3f}  p90 {np.percentile(v,90)*1000:8.3f}")

log(f"\n[T2] 厚度比 s_min/s_mid（2DGS 的零厚度假设要求 ≪1）")
for nm, v in (("s_min/s_mid", r_mid), ("s_min/s_max", r_max)):
    log(f"  {nm}: 中位 {np.median(v):.4f}  "
        f"p10 {np.percentile(v,10):.4f}  p90 {np.percentile(v,90):.4f}  "
        f"p99 {np.percentile(v,99):.4f}")
bins = [(0, 0.05), (0.05, 0.15), (0.15, 0.3), (0.3, 0.6), (0.6, 1.01)]
log("  分档占比：")
for lo, hi in bins:
    m = (r_mid >= lo) & (r_mid < hi)
    log(f"    [{lo:.2f}, {hi:.2f})  {m.mean():7.2%}   （{m.sum():>9,} 点）")

thin = r_mid < 0.05
log(f"\n[T3] 被 2DGS 丢掉的那一维（该轴完全不参与投影），屏幕当量")
focal_px = float(os.environ.get("FOCAL", "375.62"))
z_med = float(os.environ.get("Z_MED", "1.5"))
log(f"  口径：px = s_min × focal / z（focal {focal_px:.1f}px @378，z 取 {z_med} m 典型深度）")
for tag, sc in (("378", 1.0), ("1439", RES)):
    px = s_min * focal_px * sc / z_med
    log(f"    @{tag:<5} 厚度中位 {np.median(px):7.3f} px  p90 {np.percentile(px,90):7.3f} px"
        f"  （>1px 占 {(px > 1).mean():.1%}）")

log(f"\n[T4] 厚度 vs 不透明度")
for lo, hi in bins:
    m = (r_mid >= lo) & (r_mid < hi)
    if m.sum() < 100:
        continue
    log(f"  s_min/s_mid [{lo:.2f},{hi:.2f})  α 中位 {np.median(opac[m]):.4f}  "
        f"点数 {m.sum():>9,}")

verdict = ("不是薄盘（2DGS 零厚度假设不成立）" if np.median(r_mid) > 0.15
           else "确实是薄盘（零厚度假设成立）")
log(f"\n⇒ 判定：s_min/s_mid 中位 {np.median(r_mid):.4f} → {verdict}")
