# -*- coding: utf-8 -*-
"""E4 几何审计的汇报配图生成（Windows，workbuddy python，自带 pillow）。

输入：D:\\output\\e4_geometry_view\\（29p 产物 + base_v*_render.png + e4_report.json）
输出：D:\\output\\e4_geometry_view\\ppt\\ppt_e4_*.png
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\wangyufeng\.workbuddy\binaries\python\envs\default\Lib\site-packages")

from PIL import Image, ImageDraw, ImageFont

SRC = Path(r"D:\output\e4_geometry_view")
OUT = SRC / "ppt"
OUT.mkdir(parents=True, exist_ok=True)

FONT_PATHS = [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyhbd.ttc"]

rep = json.loads((SRC / "e4_report.json").read_text(encoding="utf-8"))


def font(size, bold=False):
    for p in FONT_PATHS[::-1] if bold else FONT_PATHS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def label(draw, xy, text, size=24, bold=False, fill=(255, 255, 255), anchor="la"):
    draw.text(xy, text, font=font(size, bold), fill=fill, anchor=anchor)


def header(draw, W, text, size=32):
    label(draw, (W // 2, 16), text, size=size, bold=True, anchor="ma")
    draw.line([(40, 62), (W - 40, 62)], fill=(160, 160, 160), width=2)


def load(name):
    return Image.open(SRC / name).convert("RGB")


LAYER_COL = [(230, 60, 60), (60, 180, 80), (70, 120, 240)]
LAYER_NAME = ["L0（v0 预测层）", "L1（v1 预测层）", "L2（v2 预测层）"]
VIEW_NAME = ["v0（近参考）", "v1（189°，对跖）", "v2（99°）"]

# ---------- 图 0：逐层渲染分解（谁贡献了哪个视角） ----------
PW, PH = 200, 267
CAP = 24
COLS, ROWS = 4, 3
MARGIN_L = 150
W = MARGIN_L + 40 + COLS * PW + (COLS - 1) * 14
H = 96 + ROWS * (PH + CAP) + 190
canvas = Image.new("RGB", (W, H), (28, 28, 32))
d = ImageDraw.Draw(canvas)
header(d, W, "E4：逐视角层渲染分解 —— 每层只认识自己那个视角", size=30)
col_titles = ["全部层（=正式口径）", "L0 单独渲", "L1 单独渲", "L2 单独渲"]
for j, ct in enumerate(col_titles):
    x = MARGIN_L + j * (PW + 14) + PW // 2
    label(d, (x, 72), ct, 18, bold=True, anchor="ma",
          fill=(230, 230, 230) if j == 0 else LAYER_COL[j - 1])
for i in range(ROWS):
    y = 96 + i * (PH + CAP)
    label(d, (MARGIN_L - 10, y + PH // 2), VIEW_NAME[i], 17, bold=True, anchor="rm",
          fill=(230, 230, 230))
    canvas.paste(load(f"base_v{i}_render.png").resize((PW, PH)), (MARGIN_L, y))
    for k in range(3):
        x = MARGIN_L + (k + 1) * (PW + 14)
        canvas.paste(load(f"L{k}_at_v{i}_render.png").resize((PW, PH)), (x, y))
        v = rep["views"][f"v{i}"][f"L{k}"]
        col = LAYER_COL[k]
        label(d, (x + PW // 2, y + PH + 2), f"PSNR {v['psnr_vs_gt']} · 人物α {v['alpha_mass_fg']}",
              15, anchor="ma", fill=col)
fy = 96 + 3 * (PH + CAP) + 8
label(d, (W // 2, fy), "对角线（L_k 渲 v_k）PSNR 19.5 / 10.1 / 13.0、人物 α 量 0.83~0.95 ——",
      20, anchor="ma", fill=(200, 200, 200))
label(d, (W // 2, fy + 30), "每个视角的人物由本层自己贡献；非对角全部 ≤8.7 dB。",
      20, anchor="ma", fill=(200, 200, 200))
label(d, (W // 2, fy + 60), "遮挡融合权重没有分错层，错的是各层几何本身互不重合。",
      20, anchor="ma", fill=(200, 200, 200))
label(d, (W // 2, fy + 100), "判决：center 跨视角错位是主因（非遮挡融合错误）。", 22, bold=True,
      anchor="ma", fill=(255, 210, 120))
canvas.save(OUT / "ppt_e4_0_逐层分解.png")
print("ppt_e4_0 ok", canvas.size)

# ---------- 图 1：人物高斯三层 3D 云团 ----------
RAW_W, RAW_H = 760, 560  # scatter_*.png 原始尺寸
W = 60 * 2 + RAW_W * 2 + 40
H = 100 + RAW_H + 150
canvas = Image.new("RGB", (W, H), (250, 250, 250))
d = ImageDraw.Draw(canvas)
label(d, (W // 2, 16), "v0 选出的 31.5 万人物高斯，三层在 3D 里不重合（center 错位实锤）",
      30, bold=True, fill=(20, 20, 20), anchor="ma")
d.line([(40, 62), (W - 40, 62)], fill=(160, 160, 160), width=2)
canvas.paste(load("scatter_person_top.png"), (60, 92))
canvas.paste(load("scatter_person_side.png"), (60 + RAW_W + 40, 92))
label(d, (60 + RAW_W // 2, 92 + RAW_H + 10), "俯视（x-z 平面）", 20, fill=(40, 40, 40), anchor="ma")
label(d, (60 + RAW_W + 40 + RAW_W // 2, 92 + RAW_H + 10), "侧视（y-z 平面）", 20,
      fill=(40, 40, 40), anchor="ma")
for k in range(3):
    x = 120 + k * 420
    d.rectangle([x, 92 + RAW_H + 46, x + 28, 92 + RAW_H + 70], fill=LAYER_COL[k])
    label(d, (x + 38, 92 + RAW_H + 44), LAYER_NAME[k], 20, fill=(40, 40, 40))
label(d, (W // 2, 92 + RAW_H + 92), "三个视角各自预测了一份“人物表面”，三份在 3D 中分成三簇、质心错开 ——",
      20, anchor="ma", fill=(120, 40, 40))
label(d, (W // 2, 92 + RAW_H + 122), "合成渲染时跨层互相投射出幽灵/出框。",
      20, anchor="ma", fill=(120, 40, 40))
canvas.save(OUT / "ppt_e4_1_人物云团.png")
print("ppt_e4_1 ok", canvas.size)

# ---------- 图 2：跨视角重投影（v0 人物高斯 → v1/v2） ----------
W = 40 * 2 + 378 * 2 + 40
H = 96 + 504 + 150
canvas = Image.new("RGB", (W, H), (28, 28, 32))
d = ImageDraw.Draw(canvas)
header(d, W, "跨视角重投影：v0 的人物高斯，从 v1/v2 看大半不在框内", size=29)
rp = rep["person_reprojection"]
for i, j in enumerate((1, 2)):
    x = 40 + i * (378 + 40)
    canvas.paste(load(f"person_proj_v{j}.png"), (x, 92))
    r = rp[f"v{j}"]
    label(d, (x + 189, 92 + 508), f"v0 人物高斯 → v{j}", 20, bold=True, anchor="ma",
          fill=(255, 120, 120))
    label(d, (x + 189, 92 + 532), f"在框 {r['inframe_pct']}% · 人物命中 {r['person_hit_pct']}% · 质心位移 {r['centroid_shift_px']:.0f}px",
          18, anchor="ma", fill=(255, 170, 170))
label(d, (W // 2, 92 + 574), "红点 = 31.5 万个 v0 人物高斯投影到 v1/v2 的位置（人物在三张 GT 里都在画面内）。",
      19, anchor="ma", fill=(200, 200, 200))
label(d, (W // 2, 92 + 602), "62.3%（v1）/ 85.9%（v2）直接投出画面或到相机后方 —— 几何错位无法用融合权重补救。",
      19, anchor="ma", fill=(200, 200, 200))
canvas.save(OUT / "ppt_e4_2_重投影.png")
print("ppt_e4_2 ok", canvas.size)

# ---------- 图 3：判决汇总（四个假说逐项） ----------
W, H = 1180, 760
canvas = Image.new("RGB", (W, H), (248, 249, 251))
d = ImageDraw.Draw(canvas)
label(d, (W // 2, 18), "E4 判决：糊/出框/幽灵 = 跨视角 center 错位（Phase 1 收口）",
      32, bold=True, fill=(20, 20, 20), anchor="ma")
d.line([(40, 64), (W - 40, 64)], fill=(180, 180, 180), width=2)
rows = [
    ("center 错位（跨视角几何不一致）", "实锤", (190, 40, 40),
     "① 三层人物云团 3D 不重合（图1） ② v0 人物高斯投 v1/v2 出框率 62/86%（图2）"
     " ③ 逐层渲染非对角 ≤8.7 dB（图0）"),
    ("遮挡融合错误", "排除", (40, 130, 60),
     "每视角人物由本层贡献（人物 α 量 0.83~0.95）、权重分配正常；错在各层几何不重合，非权重"),
    ("scale 过大（独立病因）", "已排除", (40, 130, 60),
     "E3 一次性因果对照判 B：σ 封顶到 1.5px 即塌覆盖（人物空洞 63~81%），尺度是承重墙"),
    ("rotation 不匹配", "非主因", (150, 110, 30),
     "各向异性短轴 vs 预测深度法向 |cos| 中位 0.40~0.54、≥0.7 仅 23~33% —— "
     "高斯本就非表面定向，旋转不含可靠几何信息，调它无从修起"),
]
y = 92
for name, verdict, vcol, detail in rows:
    d.rounded_rectangle([40, y, W - 40, y + 128], radius=12, fill=(255, 255, 255),
                        outline=(200, 200, 205), width=2)
    d.rounded_rectangle([W - 210, y + 18, W - 66, y + 62], radius=8, fill=vcol)
    label(d, (W - 138, y + 40), verdict, 24, bold=True, fill=(255, 255, 255), anchor="mm")
    label(d, (64, y + 16), name, 25, bold=True, fill=(30, 30, 30))
    label(d, (64, y + 56), detail[:52], 19, fill=(70, 70, 70))
    label(d, (64, y + 84), detail[52:], 19, fill=(70, 70, 70))
    y += 146
label(d, (W // 2, y + 6), "→ 冻结 DA3-Baseline-V1，进入 Phase 2：G0 官方几何头复刻 → G1 跨视角几何一致性重训",
      25, bold=True, fill=(30, 60, 150), anchor="ma")
canvas.save(OUT / "ppt_e4_3_判决汇总.png")
print("ppt_e4_3 ok", canvas.size)
