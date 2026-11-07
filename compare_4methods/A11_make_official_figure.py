#!/usr/bin/env python3
"""fig22：DA3 官方效果复现——同一份前向结果，四种导出/坐标口径的对照。

左：view 0 的渲染（GT 用模型实际看到的那张 378x504 输入）
右：32 视角均值 PSNR 的横向条形图
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "ppt_assets")
AUD = os.environ.get(
    "AUD", "/home/velvet/output/da3_gs_results/佳佳抱枕_p1m/audit_official")
NPZ = os.environ.get(
    "NPZ", "/home/velvet/output/da3_gs_results/佳佳抱枕_p1m/official_dump.npz")

meta = json.load(open(os.path.join(AUD, "audit_official_render.json")))
mean = meta["mean_psnr"]
n_view = len({r["view"] for r in meta["rows"]})

plt.rcParams["axes.unicode_minus"] = False

# 中文字体：与 A9 同源（Windows 的微软雅黑），Linux 侧没有中文字体时也能出图
_MSYH = "/mnt/c/Windows/Fonts/msyh.ttc"
try:
    from matplotlib import font_manager

    font_manager.fontManager.addfont(_MSYH)
    _CN = font_manager.FontProperties(fname=_MSYH).get_name()
except Exception:
    _CN = "Microsoft YaHei"
plt.rcParams["font.sans-serif"] = [_CN, "Noto Sans CJK SC", "DejaVu Sans"]

GT = np.load(NPZ)["processed_images"]

CASES = [
    ("A_official", "官方等价\n610 万点 + 全 SH", "#2f6f4e"),
    ("A_dc", "610 万点 + 仅 DC", "#6b8f7a"),
    ("A_pruned_full", "裁剪到 484 万 + 全 SH", "#c08a3e"),
    ("A_pruned_dc", "裁剪 + 仅 DC\n（我们的 PLY 口径）", "#a85b3a"),
    ("B_metric_full", "帧错：缩放相机\n配未缩放高斯", "#9a9a9a"),
]

fig = plt.figure(figsize=(13.4, 7.0), facecolor="white")
gs = gridspec.GridSpec(1, 2, width_ratios=[0.80, 1.50], wspace=0.36,
                       left=0.025, right=0.975, top=0.795, bottom=0.175)

ax = fig.add_subplot(gs[0, 0])
ax.imshow(GT[0])
ax.set_xticks([])
ax.set_yticks([])
for s in ax.spines.values():
    s.set_color("#cccccc")
ax.set_title("GT：模型实际看到的输入帧（378×504）", fontsize=12, pad=8)

ax2 = fig.add_subplot(gs[0, 1])
names = [c[1] for c in CASES]
vals = [mean[c[0]] for c in CASES]
cols = [c[2] for c in CASES]
y = np.arange(len(CASES))[::-1]
bars = ax2.barh(y, vals, color=cols, height=0.62)
for yy, v in zip(y, vals):
    ax2.text(v + 0.18, yy, f"{v:.2f}", va="center", fontsize=12, weight="bold")
ax2.set_yticks(y)
ax2.set_yticklabels(names, fontsize=10.5)
ax2.set_xlim(0, 26)
ax2.set_title(f"{n_view} 视角平均 PSNR / dB（对模型实际看到的输入帧）",
              fontsize=11, pad=10)
ax2.grid(axis="x", color="#e8e8e8", lw=0.8)
ax2.set_axisbelow(True)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)

fig.text(0.5, 0.935,
         "DA3 官方效果 = 22.11 dB；我们之前报的 18.97 dB 是「裁剪 + 仅 DC」的 PLY 交付口径",
         ha="center", va="center", fontsize=15, weight="bold")
fig.text(0.5, 0.885,
         "同一份前向结果、同一个 gsplat 1.5.3、同一组相机——3.14 dB 的差全部来自导出选项，与模型和渲染器无关",
         ha="center", va="center", fontsize=11, color="#555555")
fig.text(0.5, 0.075,
         "全 SH 值 +1.15 dB（全点）/ +0.80 dB（裁剪）；不裁剪值 +2.34 dB（全 SH）/ +1.99 dB（DC）"
         "　→　点集裁剪的代价比 SH 更大。",
         ha="center", va="center", fontsize=10.2, color="#444444")
fig.text(0.5, 0.030,
         "帧错那档（缩放相机配未缩放高斯）掉 10.8 dB，即 E-01：高斯在未缩放帧，相机必须除回 scale_factor（或把高斯乘回来）。",
         ha="center", va="center", fontsize=10.2, color="#666666")

p = os.path.join(OUT, "fig22_official_repro.png")
fig.savefig(p, facecolor="white", dpi=130)
print(f"💾 {p}")
