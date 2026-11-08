#!/usr/bin/env python3
"""fig25 / fig26：I-03 只换 2DGS 光栅器的结果图。

fig25 视觉对照：3 个视角 × (GT ｜ 3DGS EWA ｜ 2DGS ray-splat ｜ 误差差热图)
                热图 = |2DGS−GT| − |3DGS−GT|，红 = 2DGS 更差，蓝 = 更好。
fig26 量化对照：深度一致性分档 / MAE 分档 / 关键指标 + E7 重排代价。
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.colors import TwoSlopeNorm
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "ppt_assets")
SC = os.environ.get("SC", "/home/velvet/output/da3_gs_results/佳佳抱枕_p1m_off")
I3 = os.path.join(SC, "i03_2dgs")
REF = os.environ.get("REF_NPZ",
                     "/home/velvet/output/da3_gs_results/佳佳抱枕_p1m/official_dump.npz")

plt.rcParams["axes.unicode_minus"] = False
_MSYH = "/mnt/c/Windows/Fonts/msyh.ttc"
try:
    from matplotlib import font_manager
    font_manager.fontManager.addfont(_MSYH)
    _CN = font_manager.FontProperties(fname=_MSYH).get_name()
except Exception:
    _CN = "Microsoft YaHei"
plt.rcParams["font.sans-serif"] = [_CN, "Noto Sans CJK SC", "DejaVu Sans"]

os.makedirs(OUT, exist_ok=True)
j = json.load(open(os.path.join(I3, "i03_2dgs.json")))
dg = json.load(open(os.path.join(I3, "i03_diag.json")))
gt_all = np.load(REF)["processed_images"]
c3, c2, cm = j["cases"]["3dgs"], j["cases"]["2dgs_reorder"], j["cases"]["2dgs_median"]
g = j["gate"]

C3, C2 = "#2f6f4e", "#8a4a86"          # 3DGS 绿 / 2DGS 紫

# ══════════════════════════════════════════ fig25
VIEWS = (0, 8, 16)
fig = plt.figure(figsize=(15.4, 12.4), facecolor="white")
gs = gridspec.GridSpec(3, 4, left=0.052, right=0.984, top=0.842, bottom=0.128,
                       wspace=0.045, hspace=0.135, height_ratios=[1, 1, 1])


def show(ax, im, title=None, color="#333333", fs=11.0):
    ax.imshow(im, aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#d5d5d5")
        s.set_linewidth(1.0)
    if title:
        ax.set_title(title, fontsize=fs, pad=8, color=color)


norm = TwoSlopeNorm(vmin=-45, vcenter=0, vmax=45)
im_hm = None
for r, v in enumerate(VIEWS):
    gt = gt_all[v].astype(np.float64)
    im3 = np.asarray(Image.open(os.path.join(I3, f"3dgs_view{v:03d}.png"))).astype(np.float64)
    im2 = np.asarray(Image.open(os.path.join(I3, f"2dgs_view{v:03d}.png"))).astype(np.float64)
    e3 = np.abs(im3 - gt).mean(-1)
    e2 = np.abs(im2 - gt).mean(-1)

    ax = fig.add_subplot(gs[r, 0])
    show(ax, gt.astype(np.uint8),
         "GT：模型实际看到的输入帧" if r == 0 else None, "#555555")
    ax.set_ylabel(f"view {v}", fontsize=12.5, color="#444444", labelpad=9)

    ax = fig.add_subplot(gs[r, 1])
    show(ax, im3.astype(np.uint8),
         f"3DGS · EWA 光栅器\n全图 {c3['psnr']:.2f} dB　锐度 {c3['sharp']:.0f}"
         if r == 0 else None, C3)

    ax = fig.add_subplot(gs[r, 2])
    show(ax, im2.astype(np.uint8),
         f"2DGS · ray-splat（轴序已重排）\n全图 {c2['psnr']:.2f} dB　锐度 {c2['sharp']:.0f}"
         if r == 0 else None, C2)

    ax = fig.add_subplot(gs[r, 3])
    im_hm = ax.imshow(e2 - e3, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#d5d5d5")
        s.set_linewidth(1.0)
    if r == 0:
        ax.set_title(f"误差差：|2DGS−GT| − |3DGS−GT|\n平均 {g['d_psnr_all']:+.2f} dB"
                     "（红＝2DGS 更差）", fontsize=10.4, pad=8, color="#444444")

cax = fig.add_axes([0.815, 0.052, 0.155, 0.013])
cb = fig.colorbar(im_hm, cax=cax, orientation="horizontal")
cb.set_label("MAE 差（0–255）", fontsize=9.0, labelpad=2)
cb.ax.tick_params(labelsize=8.5)

fig.text(0.5, 0.962,
         "I-03：只换光栅器（头不动、不重训）—— 表示与渲染器对齐了，观感却没有变好",
         ha="center", va="center", fontsize=16.0, weight="bold")
fig.text(0.5, 0.918,
         "同一个 PLY（610 万点 · 全 SH · 官方等价 22.11 dB）、同一组相机、同 sh_degree=2、同 eps2d=0.3 —— 唯一变量是光栅器",
         ha="center", va="center", fontsize=11.2, color="#555555")
fig.text(0.5, 0.887,
         "陷阱：2DGS 只用 scales[0]、scales[1]（第三维硬编码 1.0），而 DA3 的最小轴 99.69% 落在 index 0 —— 不重排直接得到零宽刀片，实测 −5.14 dB",
         ha="center", va="center", fontsize=10.4, color="#8a4a86")

fig.text(0.40, 0.082,
         f"全图 {c3['psnr']:.2f} → {c2['psnr']:.2f} dB（{g['d_psnr_all']:+.2f}）　"
         f"锐度 {c3['sharp']:.0f} → {c2['sharp']:.0f}（{g['d_sharp']:+.0f}）　"
         f"人脸锐度 {c3['face_sharp']:.0f} → {c2['face_sharp']:.0f}（{g['d_face_sharp']:+.0f}）",
         ha="center", va="center", fontsize=11.6, color="#333333")
fig.text(0.5, 0.038,
         "误差差热图：变差不是局部的，而是整幅均匀加噪 —— 连正对相机的平坦区（墙面、地毯）也在变差，这与「掠射病态」的预期相反",
         ha="center", va="center", fontsize=10.6, color="#8a4a86")

p1 = os.path.join(OUT, "fig25_i03_2dgs_visual.png")
fig.savefig(p1, facecolor="white", dpi=130)
print(f"💾 {p1}")
plt.close(fig)

# ══════════════════════════════════════════ fig26
TIERS = ["0-30", "30-60", "60-90", "all"]
LBL = ["0–30°\n正对", "30–60°", "60–90°\n掠射", "全图"]
GT_REF = {"0-30": 0.38, "30-60": 0.33, "60-90": 0.80, "all": 0.40}   # audit_15b

fig = plt.figure(figsize=(15.4, 6.5), facecolor="white")
gs = gridspec.GridSpec(1, 3, left=0.055, right=0.980, top=0.760, bottom=0.205,
                       wspace=0.255, width_ratios=[1.32, 1.0, 1.05])

# (a) 深度一致性
ax = fig.add_subplot(gs[0, 0])
x = np.arange(len(TIERS))
w = 0.205
a3 = [c3[t]["med"] for t in TIERS]
a2 = [c2[t]["med"] for t in TIERS]
am = [cm[t]["med"] for t in TIERS]
ag = [GT_REF[t] for t in TIERS]
ax.bar(x - 1.5 * w, ag, w, label="DA3 输入深度（单视角）", color="#c9c9c9", ec="white")
ax.bar(x - 0.5 * w, a3, w, label="3DGS expected", color=C3, ec="white")
ax.bar(x + 0.5 * w, a2, w, label="2DGS expected", color=C2, ec="white")
ax.bar(x + 1.5 * w, am, w, label="2DGS median", color="#c98ac4", ec="white")
for xi, vals in zip(x, zip(ag, a3, a2, am)):
    for k, vv in enumerate(vals):
        ax.text(xi + (k - 1.5) * w, vv + 0.035, f"{vv:.2f}",
                ha="center", va="bottom", fontsize=7.6, color="#444444")
ax.set_xticks(x)
ax.set_xticklabels(LBL, fontsize=9.6)
ax.set_ylabel("跨视角深度一致性（中位相对误差 %）\n越低越好", fontsize=10.0)
ax.set_title("(a) 深度一致性：expected 变差，median 反而好一半",
             fontsize=11.4, pad=9, loc="left")
ax.legend(fontsize=8.6, frameon=False, ncol=2, loc="upper left")
ax.set_ylim(0, 1.42)
ax.grid(axis="y", color="#e6e6e6", lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# (b) MAE 分档
ax = fig.add_subplot(gs[0, 1])
m3 = [dg["per_tier"][t]["mae_3dgs"] for t in TIERS]
m2 = [dg["per_tier"][t]["mae_2dgs"] for t in TIERS]
ax.bar(x - w / 2, m3, w, label="3DGS", color=C3, ec="white")
ax.bar(x + w / 2, m2, w, label="2DGS", color=C2, ec="white")
for xi, (v3, v2) in enumerate(zip(m3, m2)):
    ax.text(xi - w / 2, v3 + 0.22, f"{v3:.2f}", ha="center", fontsize=8.2, color="#333333")
    ax.text(xi + w / 2, v2 + 0.22, f"{v2:.2f}", ha="center", fontsize=8.2, color="#333333")
    ax.text(xi, max(v3, v2) + 1.05, f"{(v2-v3)/v3:+.2%}", ha="center",
            fontsize=9.4, weight="bold", color="#b03a2e")
ax.set_xticks(x)
ax.set_xticklabels(LBL, fontsize=9.6)
ax.set_ylabel("RGB 绝对误差 MAE（0–255）\n越低越好", fontsize=10.0)
ax.set_title("(b) RGB 误差：正对区损失最大，不是掠射区", fontsize=11.4, pad=9, loc="left")
ax.legend(fontsize=9.0, frameon=False, loc="upper left")
ax.set_ylim(0, 20.5)
ax.grid(axis="y", color="#e6e6e6", lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# (c) 判据仪表
ax = fig.add_subplot(gs[0, 2])
ax.axis("off")
rows = [
    ("P-D   掠射 expected 比值", f"{g['P_D_ratio']:.2f}", "≤0.90", g["P_D_pass"]),
    ("P-D2  2DGS 掠射绝对值", f"{g['P_D2_abs']:.2f}%", "<2.40%", g["P_D2_pass"]),
    ("P-D3  掠射 median 比值", f"{g['P_D3_ratio']:.2f}", "≤0.70", g["P_D3_pass"]),
    ("P-COV 深度覆盖比", f"{g['P_COV_ratio']:.2f}", "≥0.95", g["P_COV_pass"]),
    ("P-E7  重排 ΔPSNR", f"{g['P_E7_delta']:+.2f} dB", "≤−1.0", g["P_E7_pass"]),
]
ax.set_title("(c) 预注册判据（跑前写死）", fontsize=11.4, pad=9, loc="left")
for i, (nm, val, thr, ok) in enumerate(rows):
    y = 0.845 - i * 0.150
    ax.text(0.00, y, nm, fontsize=9.5, color="#333333", va="center")
    ax.text(0.635, y, val, fontsize=10.6, weight="bold", va="center",
            color="#2f6f4e" if ok else "#b03a2e", ha="right")
    ax.text(0.66, y, thr, fontsize=8.8, color="#888888", va="center")
    ax.text(0.985, y, "过" if ok else "未过", fontsize=10.0, weight="bold",
            va="center", ha="right", color="#2f6f4e" if ok else "#b03a2e")
ax.text(0.0, 0.055, "判定：B 档 —— 能力存在（median 深度），但未转化为观感",
        fontsize=10.2, weight="bold", color="#8a4a86", va="center")

fig.text(0.5, 0.945,
         "I-03 量化结果：A2 成立（median 深度 −48%），A1 不成立（更糊）",
         ha="center", va="center", fontsize=15.6, weight="bold")
fig.text(0.5, 0.882,
         "解释：损失在正对区最大（+7.54%）、掠射区反而更小（+5.14%）→ 不是透视病态，是盘的朝向本身没有几何意义",
         ha="center", va="center", fontsize=11.0, color="#555555")
fig.text(0.5, 0.108,
         "DA3 的盘法向与真实表面夹角 56.9~66.4°（随机基线 57.3°）：2DGS 拿随机法向定求交平面 = 屏幕足迹被随机拉伸；"
         "3DGS 的 EWA 靠 add_blur(eps2d) 兜底，对退化轴不敏感，反而稳。→ 法向正则（I-04）是 2DGS 有意义的前提",
         ha="center", va="center", fontsize=9.8, color="#777777")

p2 = os.path.join(OUT, "fig26_i03_2dgs_quant.png")
fig.savefig(p2, facecolor="white", dpi=135)
print(f"💾 {p2}")
