#!/usr/bin/env python3
"""fig23：I-01/I-02 归位 —— PLY 交付口径从 18.97 抬到官方 22.11。

上排：view 0 全图（GT + 4 档 PLY 渲染）
下排：同一位置的人脸放大（FACE_BOX 与 A6/A7 同源）
PSNR/SSIM/锐度写在列标题里，全部是 32 视角均值。
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
SC = os.environ.get("SC", "/home/velvet/output/da3_gs_results/佳佳抱枕_p1m_off")
EX = os.path.join(SC, "official_export")
REF_NPZ = os.environ.get(
    "REF_NPZ", "/home/velvet/output/da3_gs_results/佳佳抱枕_p1m/official_dump.npz")

FACE_BOX = (0.35, 0.105, 0.565, 0.335)   # 归一化 (l, t, r, b)，与 A6/A7 同源

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
mean = met["mean"]

CASES = [
    ("i12_fullpts_sh", "I-01+I-02　官方等价\n610 万点 + 全 SH　　训练用初值", "#2f6f4e"),
    ("i02_pruned_sh", "I-02 单独\n484 万点 + 全 SH", "#6b8f7a"),
    ("i01_fullpts_dc", "I-01 单独\n610 万点 + 仅 DC", "#c08a3e"),
    ("hist_pruned_dc", "归位前（历史 PLY 口径）\n484 万点 + 仅 DC", "#a85b3a"),
]

gt = np.load(REF_NPZ)["processed_images"][0]
h, w = gt.shape[:2]
box = (int(FACE_BOX[0] * w), int(FACE_BOX[1] * h),
       int(FACE_BOX[2] * w), int(FACE_BOX[3] * h))

fig = plt.figure(figsize=(15.0, 9.2), facecolor="white")
gs = gridspec.GridSpec(2, 5, left=0.028, right=0.986, top=0.805, bottom=0.160,
                       wspace=0.055, hspace=0.145, height_ratios=[1.0, 0.90])


def show(ax, im, title=None, color="#333333"):
    ax.imshow(im)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#d5d5d5")
        s.set_linewidth(1.0)
    if title:
        ax.set_title(title, fontsize=10.8, pad=7, color=color)


# 上排
ax = fig.add_subplot(gs[0, 0])
show(ax, gt, "GT：模型实际看到的输入帧\n（378×504，无渲染）", "#555555")

for i, (tag, label, col) in enumerate(CASES, start=1):
    a = mean[tag]
    im = np.asarray(Image.open(os.path.join(EX, f"ply_{tag}_view000.png")))
    ax = fig.add_subplot(gs[0, i])
    show(ax, im)
    ax.set_title(f"{label}\n{tag.replace('_', ' ')}", fontsize=10.0, pad=7, color="#333333")
    # 右上角 PSNR 徽标
    ax.text(0.97, 0.035, f"{a['psnr']:.2f} dB", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=12.5, weight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.28", fc=col, ec="none", alpha=0.92))

# 下排：脸区放大
ax = fig.add_subplot(gs[1, 0])
show(ax, gt[box[1]:box[3], box[0]:box[2]], "脸区 ×4.6（下排同位置）", "#777777")

for i, (tag, label, col) in enumerate(CASES, start=1):
    im = np.asarray(Image.open(os.path.join(EX, f"ply_{tag}_view000.png")))
    crop = im[box[1]:box[3], box[0]:box[2]]
    a = mean[tag]
    ax = fig.add_subplot(gs[1, i])
    show(ax, crop)
    ax.text(0.5, -0.055, f"SSIM {a['ssim']:.3f}　锐度 {a['sharp']:.0f}　"
                         f"{a['n']/1e4:.0f} 万点 · SH deg {a['deg']}",
            transform=ax.transAxes, ha="center", va="top", fontsize=9.6, color="#555555")

met_r = {}
_rp = os.path.join(EX, "ply_region_metrics.json")
if os.path.exists(_rp):
    met_r = json.load(open(_rp))["per_case"]

fig.text(0.5, 0.958,
         "I-01/I-02 归位完成：PLY 交付口径 18.97 → 22.11 dB，与官方内存口径逐档吻合（差 ≤ 0.005 dB）",
         ha="center", va="center", fontsize=15.5, weight="bold")
fig.text(0.5, 0.905,
         "同一次前向、同一个 gsplat 1.5.3、同一组相机 —— 4 档之间只差「裁剪 / 不裁剪」与「DC / 全 SH」两个导出开关",
         ha="center", va="center", fontsize=11.2, color="#555555")

if met_r:
    b = met_r["hist_pruned_dc"]
    sh = met_r["i02_pruned_sh"]
    pt = met_r["i01_fullpts_dc"]
    al = met_r["i12_fullpts_sh"]
    fig.text(0.5, 0.098,
             f"两个开关分工明确：全 SH 买「清晰度」（人脸锐度 {sh['face_sharp']-b['face_sharp']:+.0f}、"
             f"PSNR {sh['face']-b['face']:+.2f} dB）　｜　不裁剪买「准确度」"
             f"（人脸 PSNR {pt['face']-b['face']:+.2f} dB、锐度 {pt['face_sharp']-b['face_sharp']:+.0f}）",
             ha="center", va="center", fontsize=10.4, color="#444444")
    fig.text(0.5, 0.045,
             f"叠加后人脸 {b['face']:.2f} → {al['face']:.2f} dB（{al['face']-b['face']:+.2f}）、"
             f"锐度 {b['face_sharp']:.0f} → {al['face_sharp']:.0f}（{al['face_sharp']-b['face_sharp']:+.0f}）"
             "　—　归位前那种「既糊又重影」，是两个不同的缺失各占一半",
             ha="center", va="center", fontsize=10.0, color="#666666")
else:
    fig.text(0.5, 0.075,
             "全 SH：+1.15 dB（全点口径）/ +0.80 dB（裁剪口径）　　不裁剪：+1.99 dB（仅 DC）/ +2.34 dB（全 SH）",
             ha="center", va="center", fontsize=10.4, color="#444444")

p = os.path.join(OUT, "fig23_ply_official.png")
fig.savefig(p, facecolor="white", dpi=135)
print(f"💾 {p}")
