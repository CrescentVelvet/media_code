#!/usr/bin/env python3
"""fig27 / fig28：I-03b —— 把盘法向换成 oracle（GT 深度图法向）后，2DGS 反而更糊。

fig27 视觉：view 0 / 16 全图 + view 8 脸部放大 × (GT ｜ 3DGS ｜ 2DGS 重排 ｜ 2DGS oracle)
fig28 量化：PSNR / 锐度条形 + 判据表

结论口径（写死）：oracle 法向没有把 2DGS 拉回 3DGS，H2 证伪 → 渲染回归原版 DA3-GS。
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
IB = os.path.join(SC, "i03b_oracle")
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
j = json.load(open(os.path.join(IB, "i03b_oracle.json")))
gt_all = np.load(REF)["processed_images"]
c3, cr, co = j["cases"]["3dgs"], j["cases"]["2dgs_reorder"], j["cases"]["2dgs_oracle"]
cp = j["cases"]["2dgs_oracle_perview"]
J = j["judge"]
OR = j["oracle"]

C3, CR, CO = "#2f6f4e", "#c07a2c", "#8a4a86"

# 逐视角人脸窗（audit_21 实测标定）：view 8 的人头在 x 209~277，旧统一框会落到书架上
FACE = (0.515, 0.145, 0.665, 0.310)


def load(tag, v):
    return np.asarray(Image.open(os.path.join(IB, f"{tag}_view{v:03d}.png")))


# ══════════════════════════════════════════ fig27
fig = plt.figure(figsize=(15.6, 12.4), facecolor="white")
gs = gridspec.GridSpec(3, 4, left=0.028, right=0.986, top=0.845, bottom=0.135,
                       wspace=0.045, hspace=0.13)

COLS = [("GT", None, "#555555"),
        ("3DGS · EWA", "3dgs", C3),
        ("2DGS · 重排（I-03）", "reorder", CR),
        ("2DGS · oracle 法向（I-03b）", "oracle2", CO)]


def show(ax, im, title, color, fs=10.8):
    ax.imshow(im)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#d8d8d8")
        s.set_linewidth(1.0)
    if title:
        ax.set_title(title, fontsize=fs, pad=6, color=color)


for r, v in enumerate((0, 16)):
    for cidx, (lab, tag, col) in enumerate(COLS):
        ax = fig.add_subplot(gs[r, cidx])
        im = gt_all[v].astype(np.uint8) if tag is None else load(tag, v)
        title = None
        if r == 0:
            title = f"view {v}　GT（模型输入帧）" if tag is None else f"view {v}　{lab}"
        show(ax, im, title, col)

# 第三行：view 8 脸部放大
v = 8
y0, y1 = int(FACE[1] * 504), int(FACE[3] * 504)
x0, x1 = int(FACE[0] * 378), int(FACE[2] * 378)
for cidx, (lab, tag, col) in enumerate(COLS):
    ax = fig.add_subplot(gs[2, cidx])
    im = gt_all[v].astype(np.uint8) if tag is None else load(tag, v)
    crop = im[y0:y1, x0:x1]
    ax.imshow(crop, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#d8d8d8")
        s.set_linewidth(1.0)
    if cidx == 0:
        ax.set_title(f"view {v}　脸部 4× 放大", fontsize=10.8, pad=6, color="#555555")

fig.text(0.5, 0.963,
         "I-03b：把盘法向换成 oracle（GT 深度图表面法向）—— 2DGS 反而更糊，H2 证伪",
         ha="center", va="center", fontsize=15.6, weight="bold")
fig.text(0.5, 0.922,
         "同一 PLY / 同一组相机 / 同 sh_degree=2 / 同轴序约定；三档 2DGS 只差旋转矩阵第三列，"
         "其余（位置·尺度·颜色·透明度）逐字节相同",
         ha="center", va="center", fontsize=11.0, color="#555555")
fig.text(0.5, 0.893,
         f"oracle 覆盖率 {OR['used_oracle']:.0%}（闸门 |Δz|/z<3%，未命中回落重排法向）"
         f"　｜　oracle 法向 vs DA3 盘法向夹角中位 {OR['angle_med']:.0f}°"
         f"　｜　DA3 的 s_min/s_mid 中位 0.0000（98.4% <0.05，确实是薄盘）",
         ha="center", va="center", fontsize=10.2, color="#8a4a86")

fig.text(0.5, 0.098,
         f"全图 PSNR　3DGS {c3['psnr']:.2f} ｜ 重排 {cr['psnr']:.2f}"
         f"（{cr['psnr']-c3['psnr']:+.2f}）｜ oracle {co['psnr']:.2f}"
         f"（{co['psnr']-c3['psnr']:+.2f}）",
         ha="center", va="center", fontsize=11.6, color="#333333")
fig.text(0.5, 0.066,
         f"人脸 PSNR　{c3['face_psnr']:.2f} ｜ {cr['face_psnr']:.2f} ｜ {co['face_psnr']:.2f}"
         f"　　｜　　锐度　{c3['sharp']:.0f} ｜ {cr['sharp']:.0f} ｜ {co['sharp']:.0f}"
         f"　　（人脸 {c3['face_sharp']:.0f} ｜ {cr['face_sharp']:.0f} ｜ {co['face_sharp']:.0f}）",
         ha="center", va="center", fontsize=11.0, color="#333333")
fig.text(0.5, 0.030,
         "法向给正确答案后，清晰度不升反降 —— 说明「盘法向没学对」不是 2DGS 落后的原因；"
         "2DGS 在「不重训」前提下不可能超过原版 DA3-GS",
         ha="center", va="center", fontsize=10.6, color="#8a4a86")

p27 = os.path.join(OUT, "fig27_i03b_oracle_visual.png")
fig.savefig(p27, dpi=155, facecolor="white")
plt.close(fig)
print("💾", p27)

# ══════════════════════════════════════════ fig28
fig = plt.figure(figsize=(15.0, 6.9), facecolor="white")
gs = gridspec.GridSpec(1, 3, left=0.062, right=0.982, top=0.775, bottom=0.205,
                       wspace=0.30, width_ratios=[1.12, 1.0, 1.42])

names = ["3DGS\n（原版 DA3-GS）", "2DGS 重排\n（I-03）", "2DGS oracle\n（I-03b）"]
vals = [c3["psnr"], cr["psnr"], co["psnr"]]
cols = [C3, CR, CO]

ax = fig.add_subplot(gs[0, 0])
base = c3["psnr"]
d_vals = [v - base for v in vals]
ax.axhline(0, color="#555555", lw=1.1)
for i, (nm, d, cc) in enumerate(zip(names, d_vals, cols)):
    ax.plot([i, i], [0, d], color=cc, lw=2.2, solid_capstyle="round")
    ax.plot([i], [d], "o", ms=11, color=cc, zorder=5)
    if i == 0:
        ax.text(i, 0.055, f"{vals[i]:.2f} dB\n（基线）", ha="center", va="bottom",
                fontsize=10.4, weight="bold", color=cc)
    else:
        ax.text(i, d - 0.075, f"{vals[i]:.2f} dB\n（{d:+.2f}）", ha="center", va="top",
                fontsize=10.4, weight="bold", color=cc)
ax.set_xticks(range(3))
ax.set_xticklabels(names, fontsize=9.8)
ax.set_xlim(-0.62, 2.62)
ax.set_ylim(min(d_vals) - 0.34, 0.34)
ax.set_ylabel("相对 3DGS 基线的 ΔPSNR（dB）", fontsize=10.6)
ax.set_title("全图 PSNR 增量：给对法向也没追回来", fontsize=12, pad=10)
ax.grid(axis="y", ls=":", lw=0.8, color="#cccccc")
ax.set_axisbelow(True)
ax.tick_params(labelsize=9.8)

ax = fig.add_subplot(gs[0, 1])
x = np.arange(3)
w = 0.38
sp = [c3["sharp"], cr["sharp"], co["sharp"]]
fsp = [c3["face_sharp"], cr["face_sharp"], co["face_sharp"]]
b1 = ax.bar(x - w / 2, sp, w, color="#7f9db3", label="全图锐度")
b2 = ax.bar(x + w / 2, fsp, w, color="#c58b9a", label="人脸锐度")
ax.set_xticks(x)
ax.set_xticklabels(["3DGS", "2DGS 重排", "2DGS oracle"], fontsize=10.2)
ax.set_ylabel("Laplacian 方差", fontsize=11)
ax.set_title("清晰度：一档比一档低", fontsize=12, pad=10)
ax.legend(fontsize=9.2, frameon=False, loc="upper right", ncol=2,
          handlelength=1.2, columnspacing=1.0, bbox_to_anchor=(1.0, 1.02))
ax.grid(axis="y", ls=":", lw=0.8, color="#cccccc")
ax.set_axisbelow(True)
ax.set_ylim(0, max(sp + fsp) * 1.22)
for bars_ in (b1, b2):
    for b_ in bars_:
        ax.text(b_.get_x() + b_.get_width() / 2, b_.get_height() + 6,
                f"{b_.get_height():.0f}", ha="center", va="bottom", fontsize=9.2)

ax = fig.add_subplot(gs[0, 2])
ax.axis("off")
ax.set_title("判据（§3.5.2，跑前写死）", fontsize=12, pad=10)
rows = [
    ("J-PSNR  全图 ≥ 3DGS −0.15 dB", f"{J['J_psnr_gap']:+.2f}", J["J_psnr_pass"]),
    ("J-FACE  人脸 ≥ 3DGS −0.15 dB", f"{J['J_face_gap']:+.2f}", J["J_face_pass"]),
    ("前提闸门  oracle 覆盖率 ≥60%", f"{OR['used_oracle']:.0%}", OR["used_oracle"] >= 0.6),
    ("法向净贡献 > 0（法向是否值得修）", f"{J['J_oracle_vs_reorder']:+.2f} dB",
     J["J_oracle_vs_reorder"] > 0),
]
yb = 0.80
ax.text(0.02, yb + 0.16, "判据", fontsize=10, color="#666666")
ax.text(0.80, yb + 0.16, "实测", fontsize=10, color="#666666", ha="center")
ax.text(0.98, yb + 0.16, "结论", fontsize=10, color="#666666", ha="right")
for lab, val, ok in rows:
    ax.text(0.02, yb, lab, fontsize=10.2, va="center")
    ax.text(0.80, yb, val, fontsize=11.0, va="center", ha="center", weight="bold",
            color="#2f6f4e" if ok else "#b03a2e")
    ax.text(0.98, yb, "过" if ok else "未过", fontsize=10.4, va="center", ha="right",
            weight="bold", color="#2f6f4e" if ok else "#b03a2e")
    ax.axhline(yb - 0.105, xmin=0.02, xmax=0.98, color="#e2e2e2", lw=0.8)
    yb -= 0.21
ax.text(0.02, yb - 0.02, J["verdict"], fontsize=10.6, color="#8a4a86", va="top", wrap=True)

fig.text(0.5, 0.935,
         "I-03b 量化：法向不是 2DGS 的瓶颈 —— oracle 法向反而让全图与人脸同时变糊",
         ha="center", va="center", fontsize=15.2, weight="bold")
fig.text(0.5, 0.882,
         "三档 2DGS 之间只差旋转矩阵第三列；oracle 覆盖率不足的部分回落重排法向",
         ha="center", va="center", fontsize=10.6, color="#555555")
fig.text(0.5, 0.075,
         f"结论：H2 证伪 → 2DGS 路线在「不重训」前提下到此为止，"
         f"渲染与交付回归原版 DA3-GS（3DGS EWA + 官方等价导出，{c3['psnr']:.2f} dB）",
         ha="center", va="center", fontsize=10.8, color="#8a4a86")

p28 = os.path.join(OUT, "fig28_i03b_oracle_quant.png")
fig.savefig(p28, dpi=155, facecolor="white")
plt.close(fig)
print("💾", p28)
