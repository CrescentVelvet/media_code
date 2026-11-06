#!/usr/bin/env python3
"""DA3 两条路线正面对比：官方 GSDPT 高斯头（一次前向）vs 接传统 3DGS（30k 训练）。

同一场景、同一批 32 视角输入、同样三个视角（0/8/16）。
上排 = DA3 原生 GS 头，forward 6.2s + 导出 6.8s ≈ 13.5s
下排 = DA3 深度/位姿 → COLMAP → 3DGS 30k 训练 ≈ 21min

产出 compare_4methods/ppt_assets/fig6_da3_native_gs.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

NATIVE = WSL / "da3_gs_results/佳佳抱枕_clean/render"
TRAINED = WSL / "da3_results/佳佳抱枕/04_model_3dgs/train/ours_30000/renders"
VIEWS = [0, 8, 16]
ROWS = [
    ("DA3 官方 GS 高斯头 · 一次前向 13.5s（6.1M 高斯）", "#3D9A8B",
     [NATIVE / f"view_{v:03d}.png" for v in VIEWS]),
    ("DA3 深度/位姿 → COLMAP → 传统 3DGS 30k it · 约 21min", "#D98E32",
     [TRAINED / f"{v:05d}.png" for v in VIEWS]),
]

fig, axes = plt.subplots(2, 3, figsize=(11.5, 8.6), dpi=150)
for r, (title, color, paths) in enumerate(ROWS):
    axes[r][0].set_title(f"{title}", fontsize=11.5, color=color, loc="left", pad=22)
    for c, p in enumerate(paths):
        ax = axes[r][c]
        if p.exists():
            ax.imshow(Image.open(p).convert("RGB"))
        else:
            ax.text(0.5, 0.5, f"缺图\n{p.name}", ha="center", va="center", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(color); s.set_linewidth(2)
        if r == 0:
            ax.set_title(f"视角 {VIEWS[c]}", fontsize=10.5, pad=4, color="#444444")
fig.suptitle("DA3 同一次前馈：原生高斯头 13.5 秒 vs 接传统 3DGS 21 分钟（佳佳抱枕 32 视角 · RTX 3090）",
             fontsize=13.5, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = OUT / "fig6_da3_native_gs.png"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print("✅", out)
