#!/usr/bin/env python3
"""R1 分层评测：把 holdout 按「与最近输入视角的光轴夹角」分桶，看各方法的衰减曲线。

为什么必须分桶
--------------
三图输入的方法在 29 个 holdout 上一平均，插值区和外推区被混成一个数，排名没有信噪比。
只有分开看「近角插值 / 中角 / 大角外推」，才能回答「这个方法的几何到底能撑多远」。

角距定义
--------
对每个 GT 相机 i，取它与 3 个输入视角（idx3）光轴夹角的最小值：
    ang_i = min_k  arccos(d_i · d_k)
用光轴夹角而不是方位角差，因为本场景相机不在同一水平圈上（俯仰从 +85° 扫到 −87°），
方位角在两个极区会失去意义（极点处方位角任意）。两种度量都算，输出里都留。

实测分布（佳佳抱枕 32 相机 / 输入 idx3=[0,7,31]）
    holdout 光轴夹角: min 0.8°  median 20.9°  max 44.7°
    → 计划里写的 >70° 外推桶在本场景**不存在**，任何人谈"大段零观测"都要先看这张分布

用法
----
    python audit_32_angle_buckets.py <eval.json> [<eval2.json> ...]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_24_five_way_video as a24

BUCKETS = [(0, 15), (15, 30), (30, 45), (45, 70), (70, 181)]


def angle_table(gt_cams, idx3):
    """返回 (光轴夹角 (N,), 方位角差 (N,))：每个 GT 相机到最近输入视角。"""
    C, d, _ = a24.read_cams("c2w", gt_cams)
    ctr = C.mean(0)
    R = C - ctr
    az = np.degrees(np.arctan2(R[:, 1], R[:, 0]))
    axis = np.array([min(float(np.degrees(np.arccos(np.clip(d[i] @ d[k], -1, 1)))) for k in idx3)
                     for i in range(len(C))])
    azd = []
    for i in range(len(C)):
        x = np.abs(az[i] - az[list(idx3)]) % 360
        azd.append(float(np.min(np.minimum(x, 360 - x))))
    return axis, np.array(azd)


def report(path):
    j = json.load(open(path))
    idx3 = j["gt"]["input3_idx"]
    axis, azd = angle_table(j["gt"]["cams"], idx3)
    ho = [i for i in range(len(axis)) if i not in idx3]

    print("\n" + "=" * 78)
    print("📄 %s" % os.path.basename(path))
    print("   视场 %s²  输入 idx %s  精化 %s  评测分辨率 %s"
          % (j["gt"]["size"], idx3,
             "是" if any(m.get("refine") for m in j["methods"].values()) else "否",
             j["gt"]["size"][0]))
    print("   角距分布（holdout %d 个）: 光轴 min %.1f / 中位 %.1f / max %.1f°, 方位角 max %.1f°"
          % (len(ho), axis[ho].min(), np.median(axis[ho]), axis[ho].max(), azd[ho].max()))

    print("\n   分桶（光轴夹角）:")
    hdr = "     %-18s" % "方法"
    for lo, hi in BUCKETS:
        n = sum(1 for i in ho if lo <= axis[i] < hi)
        hdr += " %10s" % ("%d-%d°(n=%d)" % (lo, min(hi, 90), n))
    hdr += " %8s %8s" % ("holdout", "斜率")
    print(hdr)
    rows = {}
    for name, m in j["methods"].items():
        if "per_view" not in m:
            print("     %-18s  %s" % (name, m.get("align", "无数据")))
            continue
        pv = {r["view"]: r for r in m["per_view"]}
        line = "     %-18s" % name[:18]
        for lo, hi in BUCKETS:
            ids = [i for i in ho if lo <= axis[i] < hi]
            line += " %10s" % ("%.2f" % np.mean([pv[i]["psnr"] for i in ids]) if ids else "—")
        allho = float(np.mean([pv[i]["psnr"] for i in ho]))
        # 斜率：holdout 上 PSNR 对角度的线性回归（dB / 10°）
        a = np.array([axis[i] for i in ho])
        p = np.array([pv[i]["psnr"] for i in ho])
        slope = float(np.polyfit(a, p, 1)[0]) * 10
        line += " %8.2f %+7.2f" % (allho, slope)
        print(line)
        rows[name] = dict(buckets=[float(np.mean([pv[i]["psnr"] for i in ho if lo <= axis[i] < hi]))
                                   if any(lo <= axis[i] < hi for i in ho) else None
                                   for lo, hi in BUCKETS],
                          holdout=allho, slope_per_10deg=slope,
                          self3=m.get("self3", {}).get("psnr"),
                          input3=m.get("input3", {}).get("psnr"),
                          n_pts=m.get("n_pts"), refine=m.get("refine"))
    return dict(json=path, axis=axis.tolist(), azdiff=azd.tolist(), idx3=idx3, methods=rows)


if __name__ == "__main__":
    out = [report(p) for p in (sys.argv[1:] or [
        os.path.expanduser("~/output/batch_4methods/eval_3view_jiajia_r512.json")])]
    dst = os.path.join(os.path.dirname(out[0]["json"]), "angle_buckets.json")
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n✅ 写出 %s" % dst)
