# -*- coding: utf-8 -*-
"""D1 诊断 PPT 配图生成：把 d1_diag_view 里的单张产物拼成 4 张汇报图。

用法（Windows，workbuddy python，自带 pillow）：
    python 29b_make_ppt_images.py

输入：D:\\output\\d1_diag_view\\  （GT_v*.png / FF_*_v*_render.png / FF_*_v*_alpha.png）
输出：D:\\output\\d1_diag_view\\ppt\\ppt_*.png
"""
import json
import os
import sys
from pathlib import Path

# workbuddy 内置 env 的 site-packages（Windows 侧跑，自带 pillow）
sys.path.insert(0, r"C:\Users\wangyufeng\.workbuddy\binaries\python\envs\default\Lib\site-packages")

from PIL import Image, ImageDraw, ImageFont

SRC = Path(r"D:\output\d1_diag_view")
OUT = SRC / "ppt"
OUT.mkdir(parents=True, exist_ok=True)

FONT_PATHS = [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyhbd.ttc"]


def font(size, bold=False):
    for p in FONT_PATHS[::-1] if bold else FONT_PATHS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def label(draw, xy, text, size=26, bold=False, fill=(255, 255, 255), anchor="la"):
    draw.text(xy, text, font=font(size, bold), fill=fill, anchor=anchor)


def header(draw, W, text, size=34):
    label(draw, (W // 2, 14), text, size=size, bold=True, anchor="ma")
    draw.line([(40, 62), (W - 40, 62)], fill=(160, 160, 160), width=2)


def load(name):
    return Image.open(SRC / name).convert("RGB")


# ---------- 图 0：五方法总览（直接复用已有五宫格） ----------
img = load("five_way_arc_3view_frame_120.png")
img.save(OUT / "ppt_0_五方法总览.png")
print("ppt_0 ok", img.size)

# ---------- 图 1：白洞归因（official vs max，同视角 v0） ----------
a = load("FF_official_v0_render.png")   # 378x504
b = load("FF_max_v0_render.png")
H = 700
scale = (H - 130) / a.height
aw, ah = int(a.width * scale), int(a.height * scale)
gap = 60
W = 80 + aw * 2 + gap + 80
canvas = Image.new("RGB", (W, H + 50), (30, 30, 34))
d = ImageDraw.Draw(canvas)
header(d, W, "背景白色空洞 = 导出裁剪所致（同一前馈模型、同一视角）")
canvas.paste(a.resize((aw, ah)), (80, 100))
canvas.paste(b.resize((aw, ah)), (80 + aw + gap, 100))
label(d, (80 + aw // 2, 76), "默认导出（逐视角深度裁剪+边框裁剪）", 24, anchor="ma")
label(d, (80 + aw // 2, 108 + ah), "背景空洞率 34.79%", 26, bold=True, anchor="ma", fill=(255, 120, 120))
label(d, (80 + aw + gap + aw // 2, 76), "不裁剪 + 全 SH 导出", 24, anchor="ma")
label(d, (80 + aw + gap + aw // 2, 108 + ah), "背景空洞率 0.72%", 26, bold=True, anchor="ma", fill=(120, 255, 140))
label(d, (W // 2, H + 2), "黑色/白色区域 = 无 Gaussian 覆盖（透出底色）。取消裁剪后空洞基本消失，无需重训。",
      22, anchor="ma", fill=(200, 200, 200))
canvas.save(OUT / "ppt_1_白洞归因.png")
print("ppt_1 ok", canvas.size)

# ---------- 图 2：灰/糊归因（GT vs FF_max 三视角，含 PSNR） ----------
rep = json.loads((SRC / "diag_report.json").read_text(encoding="utf-8"))
ff = rep["pipelines"]["FF_max"]["views"]
panels = []
for i in range(3):
    panels.append((load(f"GT_v{i}.png"), f"GT v{i}（输入）", None))
    v = ff[f"v{i}"]
    panels.append((load(f"FF_max_v{i}_render.png"),
                   f"DA3 前馈直出 v{i}",
                   f"PSNR {v['psnr']} dB · 锐度 {v['sharp_render']}（GT {v['sharp_gt']}）"))
ph = 420
pw = int(panels[0][0].width * ph / panels[0][0].height)
cols = 6
W = 60 + (pw + 20) * cols - 20 + 60
H = 130 + ph + 90
canvas = Image.new("RGB", (W, H), (30, 30, 34))
d = ImageDraw.Draw(canvas)
header(d, W, "模糊与偏灰 = 前馈直出头跨视角一致性不足（离参考视角越远越差）")
for j, (im, t1, t2) in enumerate(panels):
    x = 60 + j * (pw + 20)
    canvas.paste(im.resize((pw, ph)), (x, 110))
    label(d, (x + pw // 2, 82), t1, 22, bold=(j % 2 == 1), anchor="ma",
          fill=(255, 255, 255) if j % 2 else (170, 220, 255))
    if t2:
        label(d, (x + pw // 2, 118 + ph + 8), t2, 20, anchor="ma", fill=(255, 170, 170))
label(d, (W // 2, H - 40), "v0 = 参考视角（23.08 dB，基本正常）；v1/v2 = 外推视角（10~13 dB，人物糊、饱和度骤降 -71%）",
      22, anchor="ma", fill=(200, 200, 200))
canvas.save(OUT / "ppt_2_灰糊归因.png")
print("ppt_2 ok", canvas.size)

# ---------- 图 3：alpha 对比（空洞=α 叠不满的机制证据） ----------
a = load("FF_official_v0_alpha.png").convert("L")
b = load("FF_max_v0_alpha.png").convert("L")
H = 720
scale = (H - 140) / 504
aw, ah = int(378 * scale), int(504 * scale)
gap = 60
W = 80 + aw * 2 + gap + 80
canvas = Image.new("RGB", (W, H + 50), (30, 30, 34))
d = ImageDraw.Draw(canvas)
header(d, W, "α 机制证据：3 视角仅 3 层叠加，不透明度中位 0.10")
for j, (im, t) in enumerate([(a, "默认导出 alpha（裁剪后）"), (b, "不裁剪导出 alpha")]):
    x = 80 + j * (aw + gap)
    heat = Image.merge("RGB", (im, im, im)).resize((aw, ah))
    canvas.paste(heat, (x, 100))
    label(d, (x + aw // 2, 76), t, 24, anchor="ma")
    label(d, (x + aw // 2, 108 + ah),
          "暗 = 覆盖不足" if j == 0 else "亮 = 覆盖充分", 22, anchor="ma",
          fill=(255, 170, 170) if j == 0 else (120, 255, 140))
label(d, (W // 2, H + 2), "灰度 = 累计不透明度 α。α 叠不满时底色透出 → 画面发灰 + 白洞。", 22, anchor="ma", fill=(200, 200, 200))
canvas.save(OUT / "ppt_3_alpha机制.png")
print("ppt_3 ok", canvas.size)

# ---------- 图 4：指标汇总条形图 ----------
pipelines = ["FF_official", "FF_max", "FT_30k"]
colors = {"FF_official": (255, 120, 120), "FF_max": (120, 200, 255), "FT_30k": (140, 255, 150)}
names = {"FF_official": "前馈直出·默认导出", "FF_max": "前馈直出·不裁剪全SH",
         "FT_30k": "30k 步精化·train 视角（对照）"}
W, H = 1680, 760
canvas = Image.new("RGB", (W, H), (245, 245, 248))
d = ImageDraw.Draw(canvas)
label(d, (W // 2, 20), "三口径定量对比（3 视角输入，378×504 统一渲染口径）", 34, bold=True,
      fill=(30, 30, 34), anchor="ma")

# 左：PSNR 分组条形图（v0/v1/v2）；FT_30k 用 03 官方 train renders（55~59dB 过拟合对照），
# 不用本脚本 gsplat 重渲值（相机口径不一致，仅 5~28dB，会误导）
views = ["v0", "v1", "v2"]
bx0, by0, bw, bh = 120, 140, 640, 420
d.rectangle([bx0, by0, bx0 + bw, by0 + bh], outline=(120, 120, 130), width=2)
label(d, (bx0 + bw // 2, by0 - 46), "输入视角重放 PSNR (dB)", 26, bold=True, fill=(30, 30, 34), anchor="ma")
ymax = 62.0
for gy in range(0, 63, 10):
    y = by0 + bh - bh * gy / ymax
    d.line([(bx0, y), (bx0 + bw, y)], fill=(220, 220, 226), width=1)
    label(d, (bx0 - 14, y - 12), str(gy), 20, fill=(90, 90, 100), anchor="ra")
group_w = bw / len(views)
bar_w = 46
ft_train = rep["pipelines"].get("FT_30k_train_renders", {}).get("views", {})
for gi, v in enumerate(views):
    cx = bx0 + group_w * gi + group_w / 2
    vals = []
    for p in pipelines:
        if p == "FT_30k":
            m = ft_train.get(v)
        else:
            m = rep["pipelines"][p]["views"].get(v)
        vals.append(m["psnr"] if m else None)
    for bi, (p, val) in enumerate(zip(pipelines, vals)):
        if val is None:
            continue
        x = int(cx - (bar_w * 3 + 12) / 2 + bi * (bar_w + 6))
        h = bh * val / ymax
        d.rectangle([x, by0 + bh - h, x + bar_w, by0 + bh], fill=colors[p])
        label(d, (x + bar_w // 2, by0 + bh - h - 30), f"{val:.1f}", 20, bold=True,
              fill=(30, 30, 34), anchor="ma")
    label(d, (int(cx), by0 + bh + 16), f"{v}（{'参考' if v == 'v0' else '外推'}）", 22,
          fill=(30, 30, 34), anchor="ma")

# 右：背景空洞率 + 关键统计
rx = 900
label(d, (rx + 300, by0 - 46), "背景空洞率 / 模型统计", 26, bold=True, fill=(30, 30, 34), anchor="ma")
holes = {"FF_official": 34.79, "FF_max": 0.72}
bar_max_w, bar_h = 380, 44
for i, (p, hv) in enumerate(holes.items()):
    y = by0 + 40 + i * (bar_h + 26)
    label(d, (rx, y + bar_h // 2 - 12), names[p], 22, fill=(30, 30, 34))
    w = bar_max_w * hv / 40.0
    d.rectangle([rx + 300, y, rx + 300 + w, y + bar_h], fill=colors[p])
    d.rectangle([rx + 300, y, rx + 300 + bar_max_w, y + bar_h], outline=(150, 150, 160), width=1)
    label(d, (rx + 300 + bar_max_w + 12, y + bar_h // 2 - 12), f"{hv}%", 22, bold=True, fill=(30, 30, 34))
stats = [
    "前馈模型加载 87 s · 纯前向 2.0 s · 导出 0.3 s（RTX 3090）",
    "3 视角高斯数 571,536（57 万，每视角 ~19 万）",
    "渲染显存峰值 7.6 GB",
    "前馈直出 opacity 中位 0.10（α 叠不满 → 发灰）",
    "前馈直出 σ 折像素 2.7~4.5 px（大 footprint → 糊）",
    "30k 精化 train 视角 55~59 dB（过拟合，仅作对照非目标）",
]
y = by0 + 200
for s in stats:
    label(d, (rx, y), "· " + s, 22, fill=(60, 60, 70))
    y += 42

# 图例
lx = bx0
ly = by0 + bh + 64
for p in pipelines:
    d.rectangle([lx, ly, lx + 26, ly + 26], fill=colors[p])
    label(d, (lx + 36, ly + 1), names[p], 22, fill=(30, 30, 34))
    lx += 36 + 24 * len(names[p]) + 60
canvas.save(OUT / "ppt_4_指标汇总.png")
print("ppt_4 ok", canvas.size)

print("ALL ->", OUT)
