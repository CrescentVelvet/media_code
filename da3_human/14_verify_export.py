#!/usr/bin/env python3
"""校验 _fast_export_ply 与官方 export_ply 的输出逐字节/逐字段等价。"""
import re
import sys
import numpy as np

A, B = sys.argv[1], sys.argv[2]


def load(path):
    with open(path, "rb") as f:
        raw = f.read(1 << 20)
        end = raw.index(b"end_header") + len(b"end_header") + 1
        head = raw[:end].decode("ascii", "replace")
        n = int(re.search(r"element vertex (\d+)", head).group(1))
        props = [ln.split()[2] for ln in head.splitlines() if ln.startswith("property")]
        f.seek(end)
        arr = np.fromfile(f, dtype="<f4").reshape(n, len(props))
    return head, props, arr


ha, pa, aa = load(A)
hb, pb, ab = load(B)
print(f"A: {aa.shape} 官方   B: {ab.shape} 快速")
print("属性名完全一致:", pa == pb)
if pa != pb:
    print("  A:", pa)
    print("  B:", pb)
    sys.exit(1)
print("形状一致:", aa.shape == ab.shape)
if aa.shape != ab.shape:
    sys.exit(1)

bad = []
for i, nm in enumerate(pa):
    ca, cb = np.isnan(aa[:, i]).sum(), np.isnan(ab[:, i]).sum()
    if ca or cb:
        print(f"  ⚠️ {nm:12s} NaN 数: 官方 {ca} / 快速 {cb}")
        if ca != cb:
            bad.append((nm, float("nan")))
        continue
    d = np.abs(aa[:, i] - ab[:, i]).max()
    if d > 0:
        bad.append((nm, float(d)))
print(f"\n{len(pa)} 列中不一致的列数: {len(bad)}")
for nm, d in bad[:15]:
    print(f"  {nm:12s} max|Δ| = {d:.3e}")
