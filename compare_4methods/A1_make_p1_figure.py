#!/usr/bin/env python3
"""P1 结论图：前馈 init + 短精调，到底换到了什么。

两个口径必须同时给，否则会误导：
  - 纯训练耗时：只算 3DGS 优化那一段
  - 端到端耗时：DA3 前向 + 导出（约 35 s）也计入，因为三种路线都要先跑一次前馈

图 1（fig9）：端到端耗时 vs PSNR，看"多花的时间买到了多少 dB"
图 2（fig10）：四组在同一视角的渲染对照
"""
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# WSL 里 matplotlib 找不到 Windows 的中文字体（rcParams 名字匹配不到 .ttc），
# 必须显式把字体文件注册进 font_manager，否则中文全渲染成方块
_MSYH = "/mnt/c/Windows/Fonts/msyh.ttc"
try:
    from matplotlib import font_manager

    font_manager.fontManager.addfont(_MSYH)
    _CN = font_manager.FontProperties(fname=_MSYH).get_name()
except Exception:                                    # 非 WSL 环境退回系统字体
    _CN = "Microsoft YaHei"
matplotlib.rcParams["font.sans-serif"] = [_CN, "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "ppt_assets"
OUT.mkdir(exist_ok=True)

# 端到端 = DA3 加载 17 s + 前向 6.2 s + 导出 28.2 s ≈ 35 s（与是否出高斯无关）
FF = 35.0

# (标签, 训练秒, PSNR, 颜色, 是否主结论点)
# (标签, 训练秒, PSNR, 颜色, 是否强调, 标签偏移)
# (标签, 训练秒, PSNR, 颜色)
PTS = [
    ("DA3 直出\n(不训练)\n0.6 min", 0, 11.38, "#8C8C8C"),
    ("COLMAP 稀疏\n+ 5k\n2.6 min", 123, 22.23, "#2E8B7F"),
    ("前馈 init 57万\n+ 5k\n3.4 min", 169, 21.72, "#9B6BA8"),
    ("前馈 init 415万\n+ 5k\n8.3 min", 466, 20.14, "#C77C2E"),
    ("COLMAP 稀疏\n+ 30k\n21.5 min", 1253, 24.33, "#5B7FB5"),
]
REF30 = 24.33


def fig9():
    """柱状图而不是散点：五个配置的耗时从 0.6 到 21.5 min，线性轴上点会挤在左侧，
    标签互相压。改成有序类别 + 柱顶标 dB + 柱底标耗时，一眼能读出"多花的时间买了多少 dB"。"""
    fig, ax = plt.subplots(figsize=(10.2, 5.6), dpi=170)
    xs = np.arange(len(PTS))
    vals = [p[2] for p in PTS]
    cols = [p[3] for p in PTS]
    bars = ax.bar(xs, vals, color=cols, width=0.6, edgecolor="white", linewidth=1.2, zorder=2)

    for i, (b, v, c) in enumerate(zip(bars, vals, cols)):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.2f}", ha="center",
                va="bottom", fontsize=14, fontweight="bold", color=c, zorder=3)
        d = v - REF30
        if abs(d) > 0.01:
            ax.text(b.get_x() + b.get_width() / 2, v / 2, f"{d:+.2f} dB", ha="center",
                    va="center", fontsize=11.5, color="white", fontweight="bold", zorder=3)

    ax.axhline(REF30, ls=":", lw=1.3, color="#5B7FB5", zorder=1)
    # 基线标注放左侧空白区（DA3 柱上方），否则会与最右柱顶的数值打架
    ax.text(0.42, REF30 + 0.5, f"30k 基线 {REF30} dB", ha="left",
            fontsize=10, color="#5B7FB5")

    ax.set_xticks(xs)
    ax.set_xticklabels([p[0] for p in PTS], fontsize=10.5, linespacing=1.5)
    ax.set_ylabel("PSNR vs 输入帧（32 视角均值，dB）", fontsize=11.5)
    ax.set_ylim(0, 28.5)
    ax.set_title("短精调的收益来自「少跑步数」，不是「前馈初值」",
                 fontsize=14, loc="left", pad=14)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", ls=":", lw=0.7, alpha=0.4, zorder=0)
    ax.tick_params(axis="x", length=0, pad=8)

    fig.text(0.012, 0.015,
             "端到端耗时 = DA3 前向 + 导出 35 s + 3DGS 训练；每组均用 32 视角训练、32 视角评测",
             fontsize=9.5, color="#888888")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    p = OUT / "fig9_p1_tradeoff.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("✅", p)


def fig10():
    from PIL import Image, ImageDraw, ImageFont

    D = Path("/home/velvet/output/da3_gs_results/佳佳抱枕_p1")
    B30 = Path("/home/velvet/output/da3_results/佳佳抱枕/04_model_3dgs")
    REF = Path("/home/velvet/output/da3_results/_captures/佳佳抱枕/images")
    refs = sorted(p for p in REF.iterdir() if p.suffix.lower() == ".png")

    cols = [
        ("GT", lambda v: Image.open(refs[v])),
        ("DA3 直出", lambda v: Image.open(D/f"ffinit_ckpt/render_it0/view_{v:03d}.png")),
        ("COLMAP+5k\n2.6 min / 22.2 dB", lambda v: Image.open(D/f"final_B1_colmap5k/render_all/view_{v:03d}.png")),
        ("前馈+5k\n8.4 min / 20.1 dB", lambda v: Image.open(D/f"final_A1_ff5k/render_all/view_{v:03d}.png")),
        ("COLMAP+30k\n21.5 min / 24.3 dB", lambda v: Image.open(B30/f"render_all/view_{v:03d}.png")),
    ]
    views = [0, 16]
    W, H, pad, top = 320, 427, 8, 74
    canvas = Image.new("RGB", (len(cols) * (W + pad) + pad, top + len(views) * (H + pad + 26) + pad), "white")
    dr = ImageDraw.Draw(canvas)
    f_h = ImageFont.truetype("/mnt/c/Windows/Fonts/msyh.ttc", 17)
    f_r = ImageFont.truetype("/mnt/c/Windows/Fonts/msyh.ttc", 15)

    for j, (title, get) in enumerate(cols):
        x = pad + j * (W + pad)
        for k, line in enumerate(title.split("\n")):
            dr.text((x + 3, 8 + k * 20), line, fill=(30, 30, 30), font=f_h if k == 0 else f_r)
    for i, v in enumerate(views):
        y = top + i * (H + pad + 26)
        dr.text((pad + 3, y - 22), f"视角 {v}", fill=(110, 110, 110), font=f_r)
        for j, (_, get) in enumerate(cols):
            im = get(v).convert("RGB")
            im = im.resize((W, H), Image.LANCZOS)
            canvas.paste(im, (pad + j * (W + pad), y))

    p = OUT / "fig10_p1_renders.png"
    canvas.save(p)
    print("✅", p, canvas.size)


if __name__ == "__main__":
    fig9()
    fig10()
