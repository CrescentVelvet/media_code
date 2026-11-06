#!/usr/bin/env python3
"""生成四方法横评 PPT 配图（佳佳抱枕 32 视角实测）。

产出（写到 compare_4methods/ppt_assets/）：
  fig1_render_compare.png  五联图：GT 参考 + 四方法同场景渲染
  fig2_time.png            耗时对比（堆叠：前馈推理 / 3DGS 训练）
  fig3_gaussians.png       高斯数与 PLY 体积对比

字体用 Windows 微软雅黑（WSL 下 /mnt/c/Windows/Fonts/msyh.ttc）。
数字来自 2026-11-06 佳佳抱枕 32v 首跑实测（见 memory/2026-11-05.md）。
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "ppt_assets"
OUT.mkdir(exist_ok=True)
WSL = Path.home() / "output"

FONT = "/mnt/c/Windows/Fonts/msyh.ttc"
if Path(FONT).exists():
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False

METHODS = ["InfiniSplat v2", "QuerySplat", "VGGT-Ω+3DGS", "DA3+3DGS"]
COLORS = ["#4C7FB8", "#8A62AC", "#D98E32", "#3D9A8B"]

# ---------- fig1: 渲染五联图 ----------
GT_IMG = WSL / "vggt_omega_results/佳佳抱枕/04_model_3dgs/train/ours_30000/gt/00000.png"
IS_IMG = WSL / "ppt_assets/is_renders/render_000.png"
QS_IMG = WSL / "querysplat_results/佳佳抱枕/rendered/render_view0.png"
VO_IMG = WSL / "vggt_omega_results/佳佳抱枕/04_model_3dgs/train/ours_30000/renders/00000.png"
DA3_IMG = WSL / "da3_results/佳佳抱枕/04_model_3dgs/train/ours_30000/renders/00000.png"

cells = [
    (GT_IMG, "GT 参考帧", "训练视角原图"),
    (IS_IMG, "InfiniSplat v2", "纯前馈 · 34 秒（+3 分钟精调另计）"),
    (QS_IMG, "QuerySplat", "纯前馈 · 39 秒"),
    (VO_IMG, "VGGT-Ω + 传统3DGS", "前馈 35 秒 + 训练 20 分钟"),
    (DA3_IMG, "DA3 + 传统3DGS", "前馈 2 分钟 + 训练 21 分钟"),
]
H = 520
imgs = []
for path, title, sub in cells:
    if not path.exists():
        raise SystemExit(f"❌ 缺图: {path}")
    im = Image.open(path).convert("RGB")
    w = int(im.width * H / im.height)
    imgs.append((im.resize((w, H), Image.LANCZOS), title, sub))

pad, label_h = 12, 64
W_total = sum(im.width for im, _, _ in imgs) + pad * (len(imgs) + 1)
canvas = Image.new("RGB", (W_total, H + label_h), "white")
from PIL import ImageDraw, ImageFont
draw = ImageDraw.Draw(canvas)
try:
    f_title = ImageFont.truetype(FONT, 26)
    f_sub = ImageFont.truetype(FONT, 17)
except Exception:
    f_title = f_sub = None
x = pad
for im, title, sub in imgs:
    canvas.paste(im, (x, label_h))
    tw = draw.textlength(title, font=f_title)
    draw.text((x + (im.width - tw) / 2, 8), title, fill=(20, 20, 20), font=f_title)
    sw = draw.textlength(sub, font=f_sub)
    draw.text((x + (im.width - sw) / 2, 40), sub, fill=(110, 110, 110), font=f_sub)
    x += im.width + pad
canvas.save(OUT / "fig1_render_compare.png", quality=92)
print("✅ fig1_render_compare.png", canvas.size)

# ---------- fig2: 耗时对比（拆三段：前馈 / 精调 / 3DGS 训练）----------
feedfwd = [34, 39, 35, 130]        # 纯前馈推理（秒）：IS 33.9 / QS 16.3+22.1 / VO 35 / DA3 130
finetune = [163, 0, 0, 0]          # IS 默认串的 3000 步后处理精调（只调外观属性）
train = [0, 0, 1189, 1242]         # 传统 3DGS 30k it
fig, ax = plt.subplots(figsize=(9.6, 4.6), dpi=150)
y = np.arange(len(METHODS))
ax.barh(y, feedfwd, color=COLORS, label="前馈推理")
ax.barh(y, finetune, left=feedfwd, color=COLORS, alpha=0.75, hatch="..", edgecolor="white",
        label="后处理精调（3000 步，只调外观）")
ax.barh(y, train, left=[f + t for f, t in zip(feedfwd, finetune)], color=COLORS, alpha=0.4,
        hatch="//", edgecolor="white", label="传统 3DGS 训练（30k it）")
for i, (f, ft, t) in enumerate(zip(feedfwd, finetune, train)):
    total = f + ft + t
    txt = f"{total/60:.1f} 分钟" if total >= 120 else f"{total} 秒"
    ax.text(total + 25, i, txt, va="center", fontsize=11)
ax.set_yticks(y, METHODS, fontsize=12)
ax.invert_yaxis()
ax.set_xlabel("端到端耗时（秒）", fontsize=11)
ax.set_title("四方法耗时拆解：纯前馈 30~40 秒 vs 接 3DGS 训练 20 分钟（佳佳抱枕 · 32 视角）", fontsize=13)
ax.legend(loc="upper right", fontsize=9.5)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(0, 1750)
fig.tight_layout()
fig.savefig(OUT / "fig2_time.png", bbox_inches="tight")
plt.close(fig)
print("✅ fig2_time.png")

# ---------- fig3: 高斯数与体积 ----------
counts = [5_940_842, 222_301, 1_746_358, 1_894_132]
sizes_mb = [386, 20, 414, 448]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150)
x = np.arange(len(METHODS))
b1 = a1.bar(x, counts, color=COLORS)
a1.set_yscale("log")
a1.set_ylabel("高斯数（log 刻度）", fontsize=11)
for r, c in zip(b1, counts):
    a1.text(r.get_x() + r.get_width() / 2, c * 1.15, f"{c/1e4:.0f}万" if c >= 1e4 else f"{c}",
            ha="center", fontsize=10)
a1.set_xticks(x, ["InfiniSplat\nv2", "Query\nSplat", "VGGT-Ω\n+3DGS", "DA3\n+3DGS"], fontsize=10)
a1.set_title("重建高斯数", fontsize=13)
b2 = a2.bar(x, sizes_mb, color=COLORS)
for r, s in zip(b2, sizes_mb):
    a2.text(r.get_x() + r.get_width() / 2, s + 8, f"{s} MB", ha="center", fontsize=10)
a2.set_ylabel("PLY 文件体积（MB）", fontsize=11)
a2.set_xticks(x, ["InfiniSplat\nv2", "Query\nSplat", "VGGT-Ω\n+3DGS", "DA3\n+3DGS"], fontsize=10)
a2.set_title("PLY 文件体积", fontsize=13)
a2.set_ylim(0, 520)
for a in (a1, a2):
    a.spines[["top", "right"]].set_visible(False)
fig.suptitle("四方法重建规模对比（佳佳抱枕 · 32 视角）", fontsize=14)
fig.tight_layout()
fig.savefig(OUT / "fig3_gaussians.png", bbox_inches="tight")
plt.close(fig)
print("✅ fig3_gaussians.png")
print("全部输出 →", OUT)
