#!/usr/bin/env python3
"""AUDIT-23：采集视角的方位角覆盖 —— 这批相机到底是不是完整环绕？

起因（2026-11-09）
------------------
渲染 PLY 的环绕视频时，转到 180° 附近画面出现大面积拖影（人物头部被横向拉成条纹）。
第一反应是 look_at 构造错了，但两处自证都通过（指向注视点 vs 真实视线夹角中位 1.97°；
look_at(C[0]) vs 真实外参旋转差 2.88°）。量了方位角分布才明白：

    **32 个相机只覆盖 252.7°，中间有一个 107.3° 的缺口（−147° → −40°）。**

缺口里没有任何观测，渲染进去只能把两侧的高斯"拉"过来 —— 拖影是数据边界，不是 bug。
在 supersplat 里手动转同一圈会看到一模一样的现象（同一个 PLY）。

这条对 PPT 有直接后果：讲"32 视角环绕"时必须说明它是个**约 253° 的弧**，
否则观众转到背面会以为重建失败。

判据
----
· 最大缺口 > 30° → 判「非完整环绕」，任何 360° 展示都要标注缺口位置。
· 相机到注视点距离的极差 / 中位 > 0.5 → 轨迹半径不稳，插值轨道要按真实位置走。
"""
import json
import os

import numpy as np

NPZ = os.environ["NPZ"]
out_json = os.environ.get("OUT", "")

d = np.load(NPZ)
extr = d["extrinsics"].astype(np.float64)
if extr.ndim == 4:
    extr = extr[:, :3, :]
Rwc, twc = extr[:, :3, :3], extr[:, :3, 3]
C = (-np.transpose(Rwc, (0, 2, 1)) @ twc[..., None])[..., 0]
V = len(C)

dvec = np.transpose(Rwc, (0, 2, 1)) @ np.array([0.0, 0.0, 1.0])
dvec /= np.maximum(np.linalg.norm(dvec, axis=1, keepdims=True), 1e-12)
A = np.zeros((3, 3))
b = np.zeros(3)
for Ci, di in zip(C, dvec):
    P = np.eye(3) - np.outer(di, di)
    A += P
    b += P @ Ci
target = np.linalg.solve(A, b)

v = target[None, :] - C
v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
ang = np.degrees(np.arccos(np.clip(np.sum(v * dvec, 1), -1, 1)))
if np.median(ang) > 90:
    dvec = -dvec
    ang = 180.0 - ang

Cc = C.mean(0)
_, sv, vt = np.linalg.svd(C - Cc, full_matrices=False)
e1, e2 = vt[0], vt[1]
th = np.degrees(np.arctan2((C - Cc) @ e2, (C - Cc) @ e1))
order = np.argsort(th)
ths = th[order]
gaps = np.diff(np.concatenate([ths, ths[:1] + 360.0]))
k = int(np.argmax(gaps))
r = np.linalg.norm(C - target, axis=1)

print("=" * 74)
print("AUDIT-23 采集方位角覆盖")
print("=" * 74)
print("  PLY 视角数 %d　视线交点 (%.3f, %.3f, %.3f) m" % (V, *target))
print("  ✅ 视线自检：指向交点 vs 相机轴 夹角中位 %.2f°" % np.median(ang))
print("  相机到注视点：中位 %.3f m　范围 %.3f ~ %.3f m　极差/中位 %.3f"
      % (np.median(r), r.min(), r.max(), (r.max() - r.min()) / np.median(r)))
print("  环平面共面度（奇异值比）%.4f" % (sv[2] / max(sv[0], 1e-12)))
print("\n  方位角（度，排序）：")
print("   " + "  ".join("%6.1f" % x for x in ths))
print("\n  相邻间隔：中位 %.2f°　最大 %.2f°　最小 %.2f°"
      % (np.median(gaps), gaps.max(), gaps.min()))
print("  最大缺口 %.1f°：从 %.1f° 到 %.1f°（跨越 %d 个相机位）没有任何观测"
      % (gaps[k], ths[k], ths[(k + 1) % V], 0))
print("  实际覆盖弧长 %.1f°（= 360 − 缺口），占整圈 %.1f%%"
      % (360.0 - gaps[k], (360.0 - gaps[k]) / 3.6))

flag_cov = bool(gaps.max() > 30)
flag_rad = bool((r.max() - r.min()) / np.median(r) > 0.5)
print("\n  判据 1 最大缺口 ≤30°：%.1f° → %s"
      % (gaps.max(), "过" if not flag_cov else "未过（非完整环绕）"))
print("  判据 2 半径极差/中位 ≤0.5：%.3f → %s"
      % ((r.max() - r.min()) / np.median(r), "过" if not flag_rad else "未过"))
print("\n  ⇒ 结论：%s" % ("完整环绕，可放心做 360° 展示"
                          if not flag_cov else
                          "**非完整环绕**：360° 视频必然在缺口段出现拖影/空洞，"
                          "展示时须标注缺口区间"))

if out_json:
    import json
    with open(out_json, "w", encoding="utf-8", newline="\n") as f:
        json.dump(dict(npz=NPZ, n_views=int(V), target=[float(x) for x in target],
                       view_angle_med_deg=float(np.median(ang)),
                       az_deg=sorted(float(x) for x in ths),
                       gap_max_deg=float(gaps.max()),
                       gap_from_deg=float(ths[k]), gap_to_deg=float(ths[(k + 1) % V]),
                       covered_arc_deg=float(360 - gaps[k]),
                       gap_pass=bool(not flag_cov), radius_pass=bool(not flag_rad)),
                  f, ensure_ascii=False, indent=2)
    print("\n  json →", out_json)
