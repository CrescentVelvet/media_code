#!/usr/bin/env python3
"""四方法（+DA3 原生高斯头）横评 PPT 配图 —— 修正口径版（2026-11-06 晚）。

本版修正了两个旧问题：
  1. 旧图把 DA3/VGGT-Ω 的"前馈"与"接 3DGS"混在一条柱上，且漏了 DA3 自带的
     高斯头（13.5 秒直出 ply）——本版按「前馈直出 / 两阶段优化」分组。
  2. 旧图把 3DGS 训练时间与纯前馈时间放在同一量级叙述；实际差 ~100 倍，
     前馈段在柱状图里几乎不可见——本版按量级拆成两个子图。

产出（compare_4methods/ppt_assets/）：
  fig1_render_compare.png  六联：GT + 四方法 + DA3 原生高斯头（同视角）
  fig2_time.png            左：前馈直出组（秒）｜右：两阶段全链路（秒，标注分钟）
  fig3_gaussians.png       高斯数与 PLY 体积

数字来源（佳佳抱枕 32 视角，RTX 3090，见 .workbuddy/memory/2026-11-05.md）：
  DA3 原生 GS 头   纯前馈 13.5s（前向 6.2 + 导出 6.8），4.8~6.1M 高斯 / 314MB
  InfiniSplat v2   前馈 33.9s + 短精调 163s = 3.3min，5.94M / 386MB
  QuerySplat       前馈 38.5s（16.3+22.1），论文口径 37.1 万 / 33MB
  VGGT-Ω + 3DGS    前馈 35s + 训练 1189s = 20.4min，1.75M / 414MB
  DA3 + 3DGS       前馈 130s + 训练 1242s = 22.9min，1.89M / 448MB
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "ppt_assets"
OUT.mkdir(exist_ok=True)
WSL = Path.home() / "output"

FONT = "/mnt/c/Windows/Fonts/msyh.ttc"
if Path(FONT).exists():
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False

# ---------- 统一定义 ----------
FF = "#2E8B7F"    # 前馈直出（核心）
HY = "#5B7FB5"    # 前馈 + 短精调（混合）
OP = "#C77C2E"    # 两阶段优化

# 名称, 纯前馈秒, 短精调秒, 3DGS训练秒, 高斯数, 体积MB, 颜色, 分组
METHODS = [
    ("DA3 原生高斯头", 13.5, 0, 0, 6_100_000, 314, FF, "ff"),
    ("QuerySplat", 38.5, 0, 0, 371_000, 33, FF, "ff"),
    ("InfiniSplat v2", 33.9, 163, 0, 5_940_842, 386, HY, "hy"),
    ("VGGT-Ω + 3DGS", 35, 0, 1189, 1_746_358, 414, OP, "op"),
    ("DA3 + 3DGS", 130, 0, 1242, 1_894_132, 448, OP, "op"),
]

# ---------- fig1: 渲染六联（同视角） ----------
RENDERS = [
    (WSL / "vggt_omega_results/佳佳抱枕/04_model_3dgs/train/ours_30000/gt/00000.png",
     "GT 参考帧", "训练视角原图", "#333333"),
    (WSL / "da3_gs_results/佳佳抱枕_clean/render/view_000.png",
     "DA3 原生高斯头", "一次前向 13.5 秒 · 6.1M 高斯", FF),
    (WSL / "querysplat_results/佳佳抱枕_paper/rendered/render_view0.png",
     "QuerySplat", "一次前向 38.5 秒 · 37.1 万高斯", FF),
    (WSL / "ppt_assets/is_renders/render_000.png",
     "InfiniSplat v2", "前馈 34 秒 + 精调 163 秒", HY),
    (WSL / "vggt_omega_results/佳佳抱枕/04_model_3dgs/train/ours_30000/renders/00000.png",
     "VGGT-Ω + 传统3DGS", "前馈 35 秒 + 训练 20 分钟", OP),
    (WSL / "da3_results/佳佳抱枕/04_model_3dgs/train/ours_30000/renders/00000.png",
     "DA3 + 传统3DGS", "前馈 2.2 分钟 + 训练 21 分钟", OP),
]

H = 520
imgs = []
for path, title, sub, color in RENDERS:
    if not path.exists():
        raise SystemExit(f"❌ 缺图: {path}")
    im = Image.open(path).convert("RGB")
    w = int(im.width * H / im.height)
    imgs.append((im.resize((w, H), Image.LANCZOS), title, sub, color))

pad, label_h, foot_h = 12, 112, 40
band_h = 36
W_total = sum(im.width for im, _, _, _ in imgs) + pad * (len(imgs) + 1)
canvas = Image.new("RGB", (W_total, H + label_h + foot_h), "white")
draw = ImageDraw.Draw(canvas)
f_title = ImageFont.truetype(FONT, 26)
f_sub = ImageFont.truetype(FONT, 16)
f_band = ImageFont.truetype(FONT, 18)
f_foot = ImageFont.truetype(FONT, 15)

# 列坐标
xs = []
x = pad
for im, *_ in imgs:
    xs.append(x)
    x += im.width + pad

# 分组带：索引区间 → 标签（GT 单列不参与分组）
GROUPS = [
    ((1, 2), "前馈直出（一次前向即可渲染）", FF),
    ((3, 3), "前馈 + 短精调", HY),
    ((4, 5), "前馈几何 + 逐场景优化", OP),
]
for (i0, i1), tag, color in GROUPS:
    x0, x1 = xs[i0], xs[i1] + imgs[i1][0].width
    band_col = tuple(int(color.lstrip("#")[k:k + 2], 16) for k in (0, 2, 4))
    light = tuple(int(c + (255 - c) * 0.86) for c in band_col)
    draw.rectangle([x0, 0, x1, band_h], fill=light)
    tw = draw.textlength(tag, font=f_band)
    bb = draw.textbbox((0, 0), tag, font=f_band)          # 按实测字形盒垂直居中，避免上缘被裁
    ty = (band_h - (bb[3] - bb[1])) / 2 - bb[1]
    draw.text(((x0 + x1) / 2 - tw / 2, ty), tag, fill=color, font=f_band)

for x, (im, title, sub, color) in zip(xs, imgs):
    canvas.paste(im, (x, label_h))
    draw.text((x + (im.width - draw.textlength(title, font=f_title)) / 2, 44),
              title, fill=color, font=f_title)
    draw.text((x + (im.width - draw.textlength(sub, font=f_sub)) / 2, 78),
              sub, fill=(110, 110, 110), font=f_sub)
    draw.rectangle([x, label_h - 5, x + im.width, label_h - 2], fill=color)

foot = ("注：各方法导出分辨率不同（DA3-GS 378×504 / QuerySplat 512×512 / IS v2 392×518 / 3DGS 系列 720×960），"
        "此处按等高缩放；QuerySplat 为方形视场，构图相对更紧。同相机重渲染留给统一评测环节。")
ftw = draw.textlength(foot, font=f_foot)
draw.text((W_total / 2 - ftw / 2, label_h + H + 10), foot, fill=(130, 130, 130), font=f_foot)
canvas.save(OUT / "fig1_render_compare.png", quality=92)
print("✅ fig1_render_compare.png", canvas.size)

# ---------- fig2: 耗时拆解（两组分开画） ----------
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.2, 4.4), dpi=150,
                               gridspec_kw={"width_ratios": [1, 1.25]})

# 左：前馈直出组（秒级）
left = [m for m in METHODS if m[7] in ("ff", "hy")]
ls = [m[0] for m in left]
lf = [m[1] for m in left]
lp = [m[2] for m in left]
lc = [m[6] for m in left]
y = np.arange(len(ls))
axL.barh(y, lf, color=lc, label="前馈推理")
axL.barh(y, lp, left=lf, color=lc, alpha=0.55, hatch="..", edgecolor="white",
         label="短精调（3000 步，仅外观属性）")
for i, (f, p) in enumerate(zip(lf, lp)):
    tot = f + p
    txt = f"{tot:.0f} 秒" if tot < 120 else f"{tot/60:.1f} 分钟"
    axL.text(tot + 4, i, f"{f:.1f}s" if p == 0 else f"{f:.1f}+{p:.0f} = {txt}",
             va="center", fontsize=10.5)
axL.set_yticks(y, ls, fontsize=11.5)
axL.invert_yaxis()
axL.set_xlim(0, 250)
axL.set_xlabel("耗时（秒）", fontsize=11)
axL.set_title("① 前馈直出组：一次前向即可出 ply", fontsize=12.5, loc="left")
axL.legend(loc="lower right", fontsize=9)
axL.spines[["top", "right"]].set_visible(False)

# 右：两阶段全链路（秒级，但标注分钟）
right = [m for m in METHODS if m[7] == "op"]
rs = [m[0] for m in right]
rf = [m[1] for m in right]
rt = [m[3] for m in right]
rc = [m[6] for m in right]
y2 = np.arange(len(rs))
axR.barh(y2, rf, color=rc, label="前馈推理（35 / 130 秒）")
axR.barh(y2, rt, left=rf, color=rc, alpha=0.38, hatch="//", edgecolor="white",
         label="传统 3DGS 训练（30k it）")
for i, (f, t) in enumerate(zip(rf, rt)):
    tot = f + t
    axR.text(tot + 25, i, f"{tot/60:.1f} 分钟", va="center", fontsize=11)
axR.set_yticks(y2, rs, fontsize=11.5)
axR.invert_yaxis()
axR.set_xlim(0, 1750)
axR.set_xlabel("耗时（秒）", fontsize=11)
axR.set_title("② 两阶段组：前馈只占 3~10%，其余是逐场景优化", fontsize=12.5, loc="left")
axR.legend(loc="lower right", fontsize=9)
axR.spines[["top", "right"]].set_visible(False)
axR.annotate("", xy=(1400, 1.62), xytext=(60, 1.62),
             arrowprops=dict(arrowstyle="<->", color="#888888", lw=1))
axR.text(730, 1.72, "13.5 秒 : 22.9 分钟 ≈ 100 倍", ha="center", fontsize=10.5, color="#555555")
axR.set_ylim(1.75, -0.6)

fig.suptitle("端到端耗时：前馈直出 13.5 秒 vs 接传统 3DGS 20+ 分钟（佳佳抱枕 · 32 视角 · RTX 3090）",
             fontsize=13.5, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "fig2_time.png", bbox_inches="tight")
plt.close(fig)
print("✅ fig2_time.png")

# ---------- fig3: 高斯数与体积 ----------
counts = [m[4] for m in METHODS]
sizes = [m[5] for m in METHODS]
colors = [m[6] for m in METHODS]
short = ["DA3\n原生高斯头", "Query\nSplat", "InfiniSplat\nv2", "VGGT-Ω\n+3DGS", "DA3\n+3DGS"]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.4), dpi=150)
x = np.arange(len(METHODS))
b1 = a1.bar(x, counts, color=colors)
a1.set_yscale("log")
a1.set_ylim(1e5, 2e7)
a1.set_ylabel("高斯数（log 刻度）", fontsize=11)
labels = ["4.8~6.1M", "37.1万", "594万", "175万", "189万"]
for r, c, lb in zip(b1, counts, labels):
    a1.text(r.get_x() + r.get_width() / 2, c * 1.25, lb, ha="center", fontsize=10)
a1.set_xticks(x, short, fontsize=9.5)
a1.set_title("重建高斯数", fontsize=12.5)
b2 = a2.bar(x, sizes, color=colors)
for r, s in zip(b2, sizes):
    a2.text(r.get_x() + r.get_width() / 2, s + 8, f"{s} MB", ha="center", fontsize=10)
a2.set_ylabel("PLY 文件体积（MB）", fontsize=11)
a2.set_xticks(x, short, fontsize=9.5)
a2.set_title("PLY 文件体积", fontsize=12.5)
a2.set_ylim(0, 520)
for a in (a1, a2):
    a.spines[["top", "right"]].set_visible(False)
fig.suptitle("重建规模：QuerySplat 靠「查询预算」把高斯数压到 1/16，其余方法 175 万～610 万",
             fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "fig3_gaussians.png", bbox_inches="tight")
plt.close(fig)
print("✅ fig3_gaussians.png")
print("全部输出 →", OUT)
