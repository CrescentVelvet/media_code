# -*- coding: utf-8 -*-
"""E3 scale 封顶对照的汇报配图生成（Windows，workbuddy python，自带 pillow）。

输入：D:\\output\\e3_scale_view\\（29n 产物 + GT_v*.png + e3_report.json）
输出：D:\\output\\e3_scale_view\\ppt\\ppt_e3_*.png
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\wangyufeng\.workbuddy\binaries\python\envs\default\Lib\site-packages")

from PIL import Image, ImageDraw, ImageFont

SRC = Path(r"D:\output\e3_scale_view")
OUT = SRC / "ppt"
OUT.mkdir(parents=True, exist_ok=True)

FONT_PATHS = [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyhbd.ttc"]

rep = json.loads((SRC / "e3_report.json").read_text(encoding="utf-8"))
base, clamp = rep["arms"]["base"], rep["arms"]["clamp"]


def font(size, bold=False):
    for p in FONT_PATHS[::-1] if bold else FONT_PATHS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def label(draw, xy, text, size=24, bold=False, fill=(255, 255, 255), anchor="la"):
    draw.text(xy, text, font=font(size, bold), fill=fill, anchor=anchor)


def header(draw, W, text, size=34):
    label(draw, (W // 2, 16), text, size=size, bold=True, anchor="ma")
    draw.line([(40, 64), (W - 40, 64)], fill=(160, 160, 160), width=2)


def load(name):
    return Image.open(SRC / name).convert("RGB")


def panel_stats(arm, k):
    v = arm["views"][f"v{k}"]
    return v


# ---------- 图 0：判决总览（3 视角 × 原始/受限/GT） ----------
PW, PH = 252, 336  # 378x504 等比缩小
COLS, ROWS = 3, 3
COL_GAP, ROW_CAP = 16, 46
W = 60 * 2 + COLS * PW + (COLS - 1) * COL_GAP
H = 100 + ROWS * (PH + ROW_CAP) + 30 + 130
canvas = Image.new("RGB", (W, H), (28, 28, 32))
d = ImageDraw.Draw(canvas)
header(d, W, "E3：σ 封顶到 1.5 px —— 锐度数字暴涨，但画面碎了", size=29)
view_names = ["v0（近参考）", "v1（189°，对跖）", "v2（99°）"]
row_names = [("base", "原始尺度", (120, 200, 255)), ("clamp", "σ≤1.5px 封顶", (255, 200, 90))]
for j, vn in enumerate(view_names):
    x = 60 + j * (PW + COL_GAP) + PW // 2
    label(d, (x, 74), vn, 22, bold=True, anchor="ma", fill=(230, 230, 230))
for i, (tag, rname, col) in enumerate(row_names):
    arm = rep["arms"][tag]
    for j in range(COLS):
        x = 60 + j * (PW + COL_GAP)
        y = 100 + i * (PH + ROW_CAP)
        img = load(f"{tag}_v{j}_render.png").resize((PW, PH))
        canvas.paste(img, (x, y + ROW_CAP))
        v = panel_stats(arm, j)
        label(d, (x + PW // 2, y), f"{rname} · PSNR {v['psnr']}", 17, anchor="ma", fill=col)
        label(d, (x + PW // 2, y + 22), f"锐度 {v['sharp_whole']:.0f} · 人物空洞 {v['hole_pct_fg']}%",
              17, anchor="ma", fill=col)
for j in range(COLS):
    x = 60 + j * (PW + COL_GAP)
    y = 100 + 2 * (PH + ROW_CAP)
    canvas.paste(load(f"GT_v{j}.png").resize((PW, PH)), (x, y + ROW_CAP))
    label(d, (x + PW // 2, y), "GT（输入）", 18, anchor="ma", fill=(180, 255, 180))
fy = 100 + 3 * (PH + ROW_CAP) + 14
label(d, (W // 2, fy),
      "判决：85.2% 高斯被钳制，σ 中位 2.7~4.3→0.7~1.4 px；锐度×3~6 全是颗粒，",
      20, anchor="ma", fill=(255, 120, 120))
label(d, (W // 2, fy + 32),
      "人物空洞率飙到 63~81%，PSNR 掉 8~14 dB。",
      20, anchor="ma", fill=(255, 120, 120))
label(d, (W // 2, fy + 70),
      "尺度是覆盖的承重墙，糊不能靠压 σ 修复 → 归因回跨视角几何不一致",
      20, anchor="ma", fill=(200, 200, 200))
label(d, (W // 2, fy + 98),
      "（E3 一次性因果对照 = 负结果，支持 D3 重训几何头）",
      20, anchor="ma", fill=(200, 200, 200))
canvas.save(OUT / "ppt_e3_0_判决总览.png")
print("ppt_e3_0 ok", canvas.size)

# ---------- 图 1：alpha 覆盖对比（空洞可视化） ----------
H2 = 100 + 2 * (PH + ROW_CAP) + 90
canvas = Image.new("RGB", (W, H2), (28, 28, 32))
d = ImageDraw.Draw(canvas)
header(d, W, "Alpha 覆盖：尺度封顶后大面积透底（黑 = 无高斯覆盖）")
for i, (tag, rname, col) in enumerate(row_names):
    arm = rep["arms"][tag]
    for j in range(COLS):
        x = 60 + j * (PW + COL_GAP)
        y = 100 + i * (PH + ROW_CAP)
        a = Image.open(SRC / f"{tag}_v{j}_alpha.png").convert("L").resize((PW, PH))
        canvas.paste(a.convert("RGB"), (x, y + ROW_CAP))
        v = panel_stats(arm, j)
        label(d, (x + PW // 2, y), f"{rname} · 人物α中位 {v['alpha_median_fg']}",
              17, anchor="ma", fill=col)
        label(d, (x + PW // 2, y + 22), f"背景空洞 {v['hole_pct_bg05']}%",
              17, anchor="ma", fill=col)
fy = 100 + 2 * (PH + ROW_CAP) + 14
label(d, (W // 2, fy),
      "原始：三层叠加 α 偏低，但人物区 α 中位 0.96~0.99 盖得住；",
      21, anchor="ma", fill=(200, 200, 200))
label(d, (W // 2, fy + 32),
      "封顶后 footprint 缩太小，571k 高斯也铺不满 378×504（人物空洞 63~81%）。",
      21, anchor="ma", fill=(200, 200, 200))
canvas.save(OUT / "ppt_e3_1_alpha空洞.png")
print("ppt_e3_1 ok", canvas.size)

# ---------- 图 2：σ 分布 ----------
hist = load("sigma_hist.png")
W2 = 1080
H3 = 100 + 44 + hist.height + 90
canvas = Image.new("RGB", (W2, H3), (255, 255, 255))
d = ImageDraw.Draw(canvas)
label(d, (W2 // 2, 16), "投影 σ 分布（最坏输入视角，0~12 px，对数计数）", 32, bold=True,
      fill=(20, 20, 20), anchor="ma")
d.line([(40, 62), (W2 - 40, 62)], fill=(160, 160, 160), width=2)
q_b = rep["sigma_worstview_quantiles_px"]["base"]
q_c = rep["sigma_worstview_quantiles_px"]["clamp"]
d.rectangle([80, 78, 116, 98], fill=(150, 150, 150))
label(d, (124, 76), f"原始：p50/p90/p99 = {q_b[0]}/{q_b[1]}/{q_b[2]} px", 24, fill=(40, 40, 40))
d.rectangle([640, 78, 676, 98], fill=(60, 170, 90))
label(d, (684, 76), f"封顶后：p50/p90/p99 = {q_c[0]}/{q_c[1]}/{q_c[2]} px", 24, fill=(40, 40, 40))
canvas.paste(hist, ((W2 - hist.width) // 2, 124))
label(d, (W2 // 2, 124 + hist.height + 16),
      f"{rep['n_clamped']:,}/{rep['n_points']:,} 高斯（{rep['clamp_frac_pct']}%）被钳制。原始分布尾巴极长（p99 = {q_b[2]} px）——",
      22, fill=(60, 60, 60), anchor="ma")
label(d, (W2 // 2, 124 + hist.height + 46),
      "大 σ 高斯数量不多但承担覆盖；全压到 1.5 px 后覆盖崩坏。判 B：尺度不是独立病因。",
      22, fill=(60, 60, 60), anchor="ma")
canvas.save(OUT / "ppt_e3_2_sigma分布.png")
print("ppt_e3_2 ok", canvas.size)

# ---------- 图 3：v0 人物区放大（糊 vs 颗粒） ----------
BOX = (94, 130, 294, 430)  # v0 人物大致区域
ZW, ZH = (BOX[2] - BOX[0]) * 2, (BOX[3] - BOX[1]) * 2
W4 = 60 * 2 + 2 * ZW + 40
H4 = 100 + ZH + 60 + 90
canvas = Image.new("RGB", (W4, H4), (28, 28, 32))
d = ImageDraw.Draw(canvas)
header(d, W4, "v0 人物区 2× 放大：原始=糊成一团，封顶=碎成颗粒")
for i, (tag, rname, col) in enumerate(row_names):
    x = 60 + i * (ZW + 40)
    img = load(f"{tag}_v0_render.png").crop(BOX).resize((ZW, ZH), Image.LANCZOS)
    canvas.paste(img, (x, 100))
    v = panel_stats(rep["arms"][tag], 0)
    label(d, (x + ZW // 2, 100 + ZH + 14),
          f"{rname} | PSNR {v['psnr']} · 锐度 {v['sharp_whole']:.0f}", 22, anchor="ma", fill=col)
label(d, (W4 // 2, 100 + ZH + 56),
      "Laplacian 锐度对颗粒同样敏感——“锐度提升”假象由空洞边缘的高频贡献，不是真实细节。",
      21, anchor="ma", fill=(200, 200, 200))
canvas.save(OUT / "ppt_e3_3_人物放大.png")
print("ppt_e3_3 ok", canvas.size)
