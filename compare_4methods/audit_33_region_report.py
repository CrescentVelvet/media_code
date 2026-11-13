#!/usr/bin/env python3
"""把 audit_31 的区域指标结果摊成人能读的表（含「非 PSNR」维度的小结）。

用法：python compare_4methods/audit_33_region_report.py [eval.json] [--holdout]
默认读 ~/output/batch_4methods/eval_3view_jiajia_bgmask.json。
"""
import json
import os
import sys

O = os.path.expanduser("~/output")
PATH = O + "/batch_4methods/eval_3view_jiajia_bgmask.json"
HOLDOUT = "--holdout" in sys.argv
for a in sys.argv[1:]:
    if a.endswith(".json"):
        PATH = a

d = json.load(open(PATH))
key = "regions_holdout" if HOLDOUT else "regions_all32"
tag = "holdout 29 视角" if HOLDOUT else "全 32 视角"
print(f"# {os.path.basename(PATH)} ｜ 区域：{tag} ｜ {d['gt']['size']} ｜ "
      f"底色策略见各方法 bg\n")

names = [n for n, v in d["methods"].items() if v.get("align") == "OK"]
fail = [n for n, v in d["methods"].items() if v.get("align") != "OK"]


def f(v, fmt, dash="—"):
    if v is None:
        return dash
    try:
        if isinstance(v, float) and v != v:
            return dash
    except Exception:
        pass
    return fmt % v


hdr = ("| 方法 | 覆盖% | 全区 PSNR | 人气 PSNR | 人脸 PSNR | 手部 PSNR |"
       " 人气锐度比 | 人脸锐度比 | 人气高频比 | 人脸高频比 |"
       " 人气边缘F1 | 人脸边缘F1 | 人气鬼影率 | 人脸鬼影率 |")
print(hdr)
print("|" + "---|" * 14)
for n in names:
    m = d["methods"][n]
    g = m["all32"]
    r = m.get(key) or {}
    print("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
        n, f(g.get("coverage", 0) * 100, "%.1f"), f(g["psnr"], "%.2f"),
        f((r.get("person") or {}).get("psnr"), "%.2f"),
        f((r.get("face") or {}).get("psnr"), "%.2f"),
        f((r.get("hand") or {}).get("psnr"), "%.2f"),
        f((r.get("person") or {}).get("sharp_ratio"), "%.2f"),
        f((r.get("face") or {}).get("sharp_ratio"), "%.2f"),
        f((r.get("person") or {}).get("hf_ratio"), "%.2f"),
        f((r.get("face") or {}).get("hf_ratio"), "%.2f"),
        f((r.get("person") or {}).get("edge_f1"), "%.3f"),
        f((r.get("face") or {}).get("edge_f1"), "%.3f"),
        f((r.get("person") or {}).get("ghost_rate"), "%.2f"),
        f((r.get("face") or {}).get("ghost_rate"), "%.2f")))
if fail:
    print("\n评测未通过（不参与排名）：" + "、".join(fail))

print("\n## 位移/散度（GT→渲染 光流，区域中位位移 px / 散度；散度>0 = 局部被撑大）")
print("| 方法 | 人气 | 人脸 | 手部 |")
print("|---|---|---|---|")
for n in names:
    r = d["methods"][n].get(key) or {}
    cells = []
    for reg in ("person", "face", "hand"):
        v = r.get(reg)
        cells.append("—" if not v else "%.2f px / %+.3f" % (v["flow_med_px"], v["flow_div"]))
    print("| %s | %s | %s | %s |" % (n, *cells))

print("\n## 底色与对齐")
print("| 方法 | bg | 来源 | 点云 | 3点残差 | 精化后视线 |")
print("|---|---|---|---|---|---|")
for n in names:
    m = d["methods"][n]
    rf = m.get("refine") or {}
    print("| %s | %.0f | %s | %s | %.2f%% | %s |" % (
        n, m.get("bg", -1), m.get("bg_source", "?"),
        ("%d" % m["n_pts"]).replace("0000", "0k") if m.get("n_pts") else "—",
        m.get("rel", 0) * 100,
        ("%.2f°" % rf["ang_after"]) if rf.get("ang_after") is not None else "%.2f°" % m.get("ang_deg", 0)))
