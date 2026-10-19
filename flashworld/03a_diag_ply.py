#!/usr/bin/env python3
"""FlashWorld PLY 快速体检：顶点数、坐标分布、opacity/scale 分布、空间占用。

用于快速判断一个生成的 PLY「乱不乱」：
  - opacity p50 极低（<0.01）说明大部分高斯近透明 → 普通 viewer 里是一团雾
  - 单一体素集中 >25% 高斯 → 有坍缩
  - xyz 长尾远超 p99 → 有飞点（viewer 按包围盒取景会把主体缩到看不见）

用法:
  python flashworld/03a_diag_ply.py <ply>
"""
import re
import sys

import numpy as np

p = sys.argv[1]
with open(p, "rb") as f:
    raw = f.read(8192)
    end = raw.index(b"end_header") + len(b"end_header") + 1
    head = raw[:end].decode("ascii", "replace")
    n = int(re.search(r"element vertex (\d+)", head).group(1))
    props = [l.split()[2] for l in head.splitlines() if l.startswith("property")]
    f.seek(end)
    d = np.fromfile(f, dtype=np.float32).reshape(n, len(props))
nm = {k: i for i, k in enumerate(props)}

print("N =", n, " props =", len(props))
xyz = d[:, 0:3]
print("xyz min:", np.round(xyz.min(0), 3), " max:", np.round(xyz.max(0), 3))
for i, ax in enumerate("xyz"):
    q = np.percentile(xyz[:, i], [1, 50, 99])
    print("  %s p1/p50/p99: %s" % (ax, np.round(q, 3)))

op = 1.0 / (1.0 + np.exp(-d[:, nm["opacity"]]))
print("opacity: p50 %.4f  mean %.4f  >=0.5: %.1f%%" % (
    np.percentile(op, 50), op.mean(), (op >= 0.5).mean() * 100))
sc = np.exp(d[:, nm["scale_0"]:nm["scale_0"] + 3])
print("scale(exp): p50 %.5f  p99 %.5f  max %.5f" % (
    np.percentile(sc, 50), np.percentile(sc, 99), sc.max()))

sh = d[:, nm["f_dc_0"]:nm["f_dc_0"] + 3]
rgb = (sh * 0.28209479177387814 + 0.5).clip(0, 1)
print("RGB 均值:", np.round(rgb.mean(0) * 255, 1), " 饱和度均值 %.3f" % (rgb.max(1) - rgb.min(1)).mean())

lo, hi = xyz.min(0), xyz.max(0)
idx = np.clip(((xyz - lo) / np.maximum(hi - lo, 1e-9) * 16).astype(np.int32), 0, 15)
key = idx[:, 0] * 256 + idx[:, 1] * 16 + idx[:, 2]
uniq, cnt = np.unique(key, return_counts=True)
print("16^3 体素占用 %d/4096；最密体素 %.1f%% 高斯" % (len(uniq), cnt.max() / n * 100))
