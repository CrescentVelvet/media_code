#!/usr/bin/env python3
"""DA3 两条路线正面对比：官方 GSDPT 高斯头（一次前向）vs 接传统 3DGS（30k 训练）。

同一场景、同一批 32 视角输入、同样三个视角（0/8/16）。
上排 = DA3 原生 GS 头，forward 6.9s + 导出 28.8s ≈ 36s（484 万高斯，DC-only）
下排 = DA3 深度/位姿 → COLMAP → 3DGS 30k 训练 ≈ 21min

⚠️ 2026-11-07（E-01）：上排原来取 `佳佳抱枕_clean/render`，那是**错尺度**的渲染
（高斯漏乘 scale_factor=1.9988），视角 8/16 整幅融化，属导出 bug 不属模型能力。
现改取 `佳佳抱枕_p1m/render_all32`（米制），32 视角自一致性 18.97 dB。

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

NATIVE = WSL / "da3_gs_results/佳佳抱枕_p1m/render_all32"
TRAINED = WSL / "da3_results/佳佳抱枕/04_model_3dgs/train/ours_30000/renders"
VIEWS = [0, 8, 16]
ROWS = [
    ("DA3 原生高斯头\n36 s · 484 万", "#3D9A8B",
     [NATIVE / f"view_{v:03d}.png" for v in VIEWS]),
    ("接 3DGS\n30k 步 · 21 min", "#D98E32",
     [TRAINED / f"{v:05d}.png" for v in VIEWS]),
]

fig, axes = plt.subplots(2, 3, figsize=(11.2, 9.0), dpi=150,
                         gridspec_kw=dict(wspace=0.04, hspace=0.10))
for r, (title, color, paths) in enumerate(ROWS):
    # 行标题竖排在左侧：横排会与上方的「视角 N」列标题抢同一块空间（旧版正是压在这里）
    axes[r][0].set_ylabel(title, fontsize=10.5, color=color, rotation=90,
                          labelpad=12, va="center", linespacing=1.6)
    for c, p in enumerate(paths):
        ax = axes[r][c]
        if p.exists():
            ax.imshow(Image.open(p).convert("RGB"))
        else:
            ax.text(0.5, 0.5, f"缺图\n{p.name}", ha="center", va="center", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(color)
            s.set_linewidth(2)
        if r == 0:
            ax.set_title(f"视角 {VIEWS[c]}", fontsize=10.5, pad=5, color="#444444")

fig.suptitle("DA3 同一次前馈：原生高斯头 36 秒 vs 接传统 3DGS 21 分钟\n"
             "32 视角自一致性 18.97 dB vs 24.33 dB（佳佳抱枕 · RTX 3090）",
             fontsize=12.5, y=0.985, linespacing=1.5)
fig.tight_layout(rect=[0.02, 0, 1, 0.955])

out = OUT / "fig6_da3_native_gs.png"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print("✅", out)
