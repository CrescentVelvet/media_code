#!/usr/bin/env python3
"""fig24：侧视角（view 8 / 16）的归位前后对照 —— 重影最容易显形的地方。

每格标注该视角的 PSNR，便于看"归位在哪一档视角上最有效"。
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "ppt_assets")
EX = os.environ.get(
    "EX", "/home/velvet/output/da3_gs_results/佳佳抱枕_p1m_off/official_export")
REF_NPZ = os.environ.get(
    "REF_NPZ", "/home/velvet/output/da3_gs_results/佳佳抱枕_p1m/official_dump.npz")

plt.rcParams["axes.unicode_minus"] = False
_MSYH = "/mnt/c/Windows/Fonts/msyh.ttc"
try:
    from matplotlib import font_manager
    font_manager.fontManager.addfont(_MSYH)
    _CN = font_manager.FontProperties(fname=_MSYH).get_name()
except Exception:
    _CN = "Microsoft YaHei"
plt.rcParams["font.sans-serif"] = [_CN, "Noto Sans CJK SC", "DejaVu Sans"]

met = json.load(open(os.path.join(EX, "ply_metrics.json")))
PSNR = {}
for r in met["rows"]:
    if r["psnr"] is not None:
        PSNR[(r["tag"], r["view"])] = r["psnr"]

GT = np.load(REF_NPZ)["processed_images"]

CASES = [
    ("i12_fullpts_sh", "I-01+I-02 官方等价\n610 万 + 全 SH", "#2f6f4e"),
    ("i02_pruned_sh", "I-02 单独\n484 万 + 全 SH", "#6b8f7a"),
    ("i01_fullpts_dc", "I-01 单独\n610 万 + 仅 DC", "#c08a3e"),
    ("hist_pruned_dc", "归位前\n484 万 + 仅 DC", "#a85b3a"),
]
VIEWS = [(8, "view 8　侧转约 90°"), (16, "view 16　侧转约 180°")]

fig = plt.figure(figsize=(15.0, 8.6), facecolor="white")
gs = gridspec.GridSpec(2, 5, left=0.030, right=0.986, top=0.845, bottom=0.105,
                       wspace=0.05, hspace=0.18)


def show(ax, im, title=None, color="#333333", fs=10.5):
    ax.imshow(im)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#d5d5d5")
        s.set_linewidth(1.0)
    if title:
        ax.set_title(title, fontsize=fs, pad=6, color=color)


for row, (v, vlabel) in enumerate(VIEWS):
    ax = fig.add_subplot(gs[row, 0])
    show(ax, GT[v], f"{vlabel}\nGT（模型看到的输入帧）" if row == 0 else "GT", "#555555")
    for i, (tag, label, col) in enumerate(CASES, start=1):
        im = np.asarray(Image.open(os.path.join(EX, f"ply_{tag}_view{v:03d}.png")))
        ax = fig.add_subplot(gs[row, i])
        show(ax, im, label if row == 0 else None, "#333333", 9.8)
        p = PSNR.get((tag, v))
        txt = f"{p:.2f} dB" if p is not None else "—"
        ax.text(0.97, 0.035, txt, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=11.5, weight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.26", fc=col, ec="none", alpha=0.92))

fig.text(0.5, 0.955,
         "归位前后在侧视角上的差别：重影来自「点数被砍 26%」，糊来自「只有 DC 颜色」",
         ha="center", va="center", fontsize=15.5, weight="bold")
fig.text(0.5, 0.905,
         "两行分别是 view 8（侧转约 90°）与 view 16（约 180°）——DA3 的跨视角几何误差在这两个视角上最容易被看到",
         ha="center", va="center", fontsize=11.2, color="#555555")

d_hist = {v: PSNR[("hist_pruned_dc", v)] for v in range(32)}
d_i12 = {v: PSNR[("i12_fullpts_sh", v)] for v in range(32)}
gains = {v: d_i12[v] - d_hist[v] for v in range(32)}
vmax = max(gains, key=gains.get)
vmin = min(gains, key=gains.get)
fig.text(0.5, 0.062,
         f"view 8：{d_hist[8]:.2f} → {d_i12[8]:.2f}（{gains[8]:+.2f} dB）　　"
         f"view 16：{d_hist[16]:.2f} → {d_i12[16]:.2f}（{gains[16]:+.2f} dB）　　"
         f"全 32 视角：{np.mean(list(d_hist.values())):.2f} → {np.mean(list(d_i12.values())):.2f}"
         f"（{np.mean(list(gains.values())):+.2f} dB）",
         ha="center", va="center", fontsize=10.6, color="#444444")
fig.text(0.5, 0.021,
         f"但收益在视角间并不均匀：最大 {gains[vmax]:+.2f}（view {vmax}），最小 {gains[vmin]:+.2f}"
         f"（view {vmin}）——裁剪是按「逐视角深度 q90」做的，被砍得多的视角代价就大。"
         "本口径是自一致性 PSNR，会奖励平滑；判「更清楚」请看人脸锐度 268 → 489",
         ha="center", va="center", fontsize=9.8, color="#777777")

p = os.path.join(OUT, "fig24_ply_official_sideviews.png")
fig.savefig(p, facecolor="white", dpi=135)
print(f"💾 {p}")
