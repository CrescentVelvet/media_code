#!/usr/bin/env python3
"""E-01 重测出图：前馈 init 在「错尺度」与「米制」两种初值下的渲染与指标。

fig12：同一视角的渲染对照 —— 重影是否消失，是 E-01 归因的第二个独立验证
fig13：四组配置 PSNR 新旧并排（同超参、只差初值是否乘 1.9988）

数据：
  旧口径 ~/output/da3_gs_results/佳佳抱枕_p1   （PLY 未乘 scale_factor）
  米制   ~/output/da3_gs_results/佳佳抱枕_p1m （已乘，语义正确）
"""
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

_MSYH = "/mnt/c/Windows/Fonts/msyh.ttc"
try:
    from matplotlib import font_manager

    font_manager.fontManager.addfont(_MSYH)
    _CN = font_manager.FontProperties(fname=_MSYH).get_name()
except Exception:
    _CN = "Microsoft YaHei"
matplotlib.rcParams["font.sans-serif"] = [_CN, "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "ppt_assets"
OUT.mkdir(exist_ok=True)

OLD = Path("/home/velvet/output/da3_gs_results/佳佳抱枕_p1")
NEW = Path("/home/velvet/output/da3_gs_results/佳佳抱枕_p1m")
REF = Path("/home/velvet/output/da3_results/_captures/佳佳抱枕/images")

COLMAP5K, COLMAP30K = 22.23, 24.33


def fig12():
    from PIL import Image, ImageDraw, ImageFont

    refs = sorted(p for p in REF.iterdir() if p.suffix.lower() == ".png")
    cols = [
        ("GT\n输入帧", lambda v: Image.open(refs[v])),
        ("DA3 直出\n错尺度 11.4", lambda v: Image.open(OLD / f"render_all32/view_{v:03d}.png")),
        ("DA3 直出\n米制 19.0", lambda v: Image.open(NEW / f"render_all32/view_{v:03d}.png")),
        ("前馈415万+5k\n错尺度 20.1", lambda v: Image.open(OLD / f"final_A1_ff5k/render_all/view_{v:03d}.png")),
        ("前馈415万+5k\n米制 25.5", lambda v: Image.open(NEW / f"final_A1_ff5k/render_all32/view_{v:03d}.png")),
        ("前馈57万+5k\n错尺度 21.7", lambda v: Image.open(OLD / f"final_A2_ff500k_5k/render_all/view_{v:03d}.png")),
        ("前馈57万+5k\n米制 25.2", lambda v: Image.open(NEW / f"final_A2_ff500k_5k/render_all32/view_{v:03d}.png")),
    ]
    views = [0, 16]
    W, H, pad, top = 262, 349, 6, 74
    canvas = Image.new(
        "RGB", (len(cols) * (W + pad) + pad, top + len(views) * (H + pad + 26) + pad), "white"
    )
    dr = ImageDraw.Draw(canvas)
    f_h = ImageFont.truetype(_MSYH, 15)
    f_r = ImageFont.truetype(_MSYH, 14)

    for j, (title, _) in enumerate(cols):
        x = pad + j * (W + pad)
        for k, line in enumerate(title.split("\n")):
            dr.text((x + 2, 8 + k * 18), line, fill=(30, 30, 30), font=f_h if k == 0 else f_r)

    for i, v in enumerate(views):
        y = top + i * (H + pad + 26)
        dr.text((pad + 2, y - 21), f"视角 {v}", fill=(110, 110, 110), font=f_r)
        for j, (_, get) in enumerate(cols):
            try:
                im = get(v).convert("RGB").resize((W, H), Image.LANCZOS)
            except FileNotFoundError:
                im = Image.new("RGB", (W, H), (235, 235, 235))
            canvas.paste(im, (pad + j * (W + pad), y))

    p = OUT / "fig12_e01_retest_renders.png"
    canvas.save(p)
    print("✅", p, canvas.size)


def fig13():
    data = json.load(open(OUT / "p1_retest_metrics.json", encoding="utf-8"))
    rows = [
        ("DA3 直出\n(0 步)", 11.38, 18.97, "0 s", "#8C8C8C"),
        ("前馈415万\n+2k", data["A0 前馈415万+2k"]["old_psnr"], data["A0 前馈415万+2k"]["new_psnr"],
         f"{data['A0 前馈415万+2k']['train_seconds']} s", "#C77C2E"),
        ("前馈415万\n+5k", data["A1 前馈415万+5k"]["old_psnr"], data["A1 前馈415万+5k"]["new_psnr"],
         f"{data['A1 前馈415万+5k']['train_seconds']} s", "#C77C2E"),
        ("前馈57万\n+5k", data["A2 前馈57万+5k"]["old_psnr"], data["A2 前馈57万+5k"]["new_psnr"],
         f"{data['A2 前馈57万+5k']['train_seconds']} s", "#9B6BA8"),
    ]
    labels = [r[0] for r in rows]
    old = np.array([r[1] for r in rows])
    new = np.array([r[2] for r in rows])
    secs = [r[3] for r in rows]
    cols = [r[4] for r in rows]

    fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=170)
    x = np.arange(len(rows))
    w = 0.36
    b1 = ax.bar(x - w / 2, old, w, color="#D8D5CE", edgecolor="white", lw=1.0,
                label="错尺度初值（PLY 未乘 1.9988）", zorder=2)
    b2 = ax.bar(x + w / 2, new, w, color=cols, edgecolor="white", lw=1.0,
                label="米制初值（正确口径）", zorder=2)

    for b, v in zip(b1, old):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.35, f"{v:.2f}", ha="center", va="bottom",
                fontsize=10.5, color="#7A776F", zorder=3)
    for b, v, c in zip(b2, new, cols):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.35, f"{v:.2f}", ha="center", va="bottom",
                fontsize=11.5, fontweight="bold", color=c, zorder=3)

    for i, (o, n) in enumerate(zip(old, new)):
        ax.annotate("", xy=(x[i] + w / 2, n), xytext=(x[i] - w / 2, o),
                    arrowprops=dict(arrowstyle="->", lw=1.1, color="#B5651D", alpha=0.7))
        ax.text(x[i], max(o, n) + 2.55, f"{n - o:+.2f}", ha="center", fontsize=11.5,
                color="#B5651D", fontweight="bold")

    ax.axhline(COLMAP5K, ls="--", lw=1.2, color="#2E8B7F", zorder=1)
    ax.axhline(COLMAP30K, ls=":", lw=1.4, color="#5B7FB5", zorder=1)
    # 两条参考线统一标在右侧留白里：标在左侧会被图例压，标在柱内会被柱子压
    ax.set_xlim(-0.7, len(rows) + 1.15)
    ax.text(len(rows) - 0.03, COLMAP30K + 0.32, f"30k 基线 {COLMAP30K}", ha="left",
            va="bottom", fontsize=9.5, color="#5B7FB5")
    ax.text(len(rows) - 0.03, COLMAP5K - 0.34, f"稀疏+5k {COLMAP5K}", ha="left",
            va="top", fontsize=9.5, color="#2E8B7F")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n训练 {s}" for l, s in zip(labels, secs)], fontsize=10.5,
                       linespacing=1.5)
    ax.set_ylabel("PSNR vs 输入帧（32 视角均值，dB）", fontsize=11.5)
    ax.set_ylim(0, 31)
    ax.set_title("初值口径修对后，前馈 init 从「不如稀疏点云」变成「全面领先」",
                 fontsize=13.5, loc="left", pad=12)
    ax.legend(fontsize=10, loc="upper left", frameon=False, ncol=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", ls=":", lw=0.7, alpha=0.35, zorder=0)
    ax.tick_params(axis="x", length=0, pad=6)

    fig.text(0.012, 0.015,
             "同场景、同超参、同 32 视角，唯一差别是导出 PLY 时是否乘 DA3 的 scale_factor=1.9988",
             fontsize=9.5, color="#888888")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    p = OUT / "fig13_p1_retest_bars.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("✅", p)


if __name__ == "__main__":
    fig12()
    fig13()
