#!/usr/bin/env python3
"""生成 da3_human PPT 配图（框架图 / 基线对照 / anchor 漏斗 / 实验进度）。

输出: $FIG_OUT_DIR（默认 /mnt/c/code/media_code/da3_human/ppt_figures/）
"""
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# 微软雅黑在 Windows 字体目录（WSL 经 /mnt/c 可见）
_YAHEI = "/mnt/c/Windows/Fonts/msyh.ttc"
if Path(_YAHEI).exists():
    font_manager.fontManager.addfont(_YAHEI)
plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(os.environ.get("FIG_OUT_DIR", "/mnt/c/code/media_code/da3_human/ppt_figures"))
OUT.mkdir(parents=True, exist_ok=True)

# 统一配色（靛蓝=数据/核心，青=步骤，琥珀=说明，红=负结果）
C_IN = "#3B4CC0"   # 输入/数据
C_CORE = "#0E9594"  # 处理核心
C_OUT = "#2E7D32"   # 输出
C_WARN = "#D9822B"  # 说明/警告
C_NEG = "#C0392B"   # 负结果
C_GREY = "#6B7280"


def box(ax, x, y, w, h, text, fc, ec=None, fs=11, tc="white", lw=1.5):
    ec = ec or fc
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                       fc=fc, ec=ec, lw=lw, mutation_aspect=1)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, linespacing=1.45)


def arrow(ax, x1, y1, x2, y2, color=C_GREY, lw=2.2, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=16, color=color, lw=lw))


# ---------------------------------------------------------------- fig1 框架图
def fig_framework():
    fig, ax = plt.subplots(figsize=(13.33, 7.5), dpi=150)
    ax.set_xlim(0, 100); ax.set_ylim(0, 56); ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(50, 54, "Per-Scene DA3 + 2DGS 冻结几何方案（da3_human）",
            ha="center", fontsize=17, weight="bold", color="#1F2937")
    ax.text(50, 51.2, "冻结 DA3 出度量深度+位姿 → anchor 点云 → 有界 offset 双分支 2DGS，目标替代 scene_gs 的 COLMAP 初始化",
            ha="center", fontsize=10.5, color=C_GREY)

    # 第一行：输入 → 冻结骨干 → 几何验收
    box(ax, 2, 40, 16, 8, "输入\n125 视角图像\n1440×1920 采集", C_IN, fs=11)
    box(ax, 24, 40, 22, 8, "冻结 DA3 骨干（推理 only）\n度量深度 504p + conf\n相机位姿 + 内参", C_CORE, fs=11)
    box(ax, 52, 40, 20, 8, "几何验收通过\n深度 median 1.80m / NaN=0\n位姿对齐 mean 4.9mm", "#2E7D32", fs=10.5)
    box(ax, 78, 40, 20, 8, "DA3 内参偏差已排除\n（B-04b 对照实验）\nfx≠fy 非性能主因", C_WARN, fs=10.5)
    arrow(ax, 18, 44, 24, 44); arrow(ax, 46, 44, 52, 44); arrow(ax, 72, 44, 78, 44)

    # 第二行：anchor 构建三步
    box(ax, 4, 24, 26, 9, "① 反投影\n14,334,402 点\nconf ≥ P40（阈值1.49）", C_CORE, fs=10.5)
    box(ax, 37, 24, 26, 9, "② 多视角一致性筛选\n8 最近邻 · τ=3%+5cm · ≥3视角\n保留 13,841,023（96.6%）", C_CORE, fs=10.5)
    box(ax, 70, 24, 26, 9, "③ 体素去重 7.2mm\n→ 683,879 anchors\n（surfel 点云）", C_CORE, fs=10.5)
    arrow(ax, 30, 28.5, 37, 28.5); arrow(ax, 63, 28.5, 70, 28.5)
    ax.text(50, 21.2, "C-05 anchor 构建（置信度 × 跨视角一致性 × 几何去重）",
            ha="center", fontsize=9.5, color=C_GREY)

    # 第三行：双分支 2DGS
    box(ax, 14, 7, 34, 9, "几何分支（粗糙层）\nanchor + 有界 offset：tanh 参数化\noffset 界防深度噪声漂移", "#1D6FA5", fs=10.5)
    box(ax, 52, 7, 34, 9, "外观分支（精细层）\n高斯残差偏移 + 纹理/颜色\n高分辨率细节补偿", "#1D6FA5", fs=10.5)
    arrow(ax, 83, 24, 31, 16.2); arrow(ax, 83, 24, 69, 16.2)

    # 输出
    box(ax, 88, 7, 10, 9, "渲染\n+\n指标", C_OUT, fs=11)
    arrow(ax, 86, 11.5, 88, 11.5)

    ax.text(50, 2.8, "证据闭环：QuerySplat 冻结骨干 ｜ 高分辨率纹理 −4.25dB ｜ InfiniSplat v2 有界 offset（全开有害→取界）",
            ha="center", fontsize=9, color=C_GREY, style="italic")

    fig.savefig(OUT / "fig1_framework.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("✅ fig1_framework.png")


# ------------------------------------------------------------ fig2 基线对照
def fig_baselines():
    names = ["03 COLMAP init\n(7,163 点)", "04 DA3 init\n(459k 点)", "04b DA3几何\n+COLMAP内参"]
    psnr = [27.573, 25.505, 25.374]
    ssim = [0.8854, 0.8613, 0.8564]
    lpips = [0.1642, 0.1870, 0.1946]
    colors = ["#2E7D32", C_NEG, "#E67E22"]

    fig, axes = plt.subplots(1, 3, figsize=(13.33, 4.6), dpi=150)
    fig.patch.set_facecolor("white")
    titles = ["PSNR ↑ (dB)", "SSIM ↑", "LPIPS ↓"]
    data = [psnr, ssim, lpips]
    for ax, vals, title in zip(axes, data, titles):
        bars = ax.bar(names, vals, color=colors, width=0.55, zorder=3)
        ax.set_title(title, fontsize=13, weight="bold")
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.tick_params(axis="x", labelsize=9.5)
        lo = min(vals)
        ax.set_ylim(lo * 0.985 if lo > 0 else lo * 1.05, max(vals) * 1.012)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}" if v < 1 else f"{v:.3f}",
                    ha="center", va="bottom", fontsize=10, weight="bold")
    axes[0].annotate("−2.07dB", xy=(1, psnr[1]), xytext=(0.55, 26.9),
                     fontsize=12, color=C_NEG, weight="bold",
                     arrowprops=dict(arrowstyle="->", color=C_NEG, lw=1.8))
    fig.suptitle("阶段 B 基线对照（train 125/125 in-sample · 30K iters 全分辨率 · RTX 3090）",
                 fontsize=14, weight="bold", y=1.0)
    fig.text(0.5, -0.04, "结论：DA3 几何直出替换 COLMAP 不成立；内参假设排除（04b），主因 = 深度噪声经 densify 放大",
             ha="center", fontsize=10.5, color=C_NEG)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_baselines.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("✅ fig2_baselines.png")


# ------------------------------------------------------------- fig3 anchor 漏斗
def fig_funnel():
    stages = ["深度反投影\nconf≥P40", "多视角一致性\n(8邻域·τ=3%+5cm)", "体素去重\n7.2mm"]
    vals = [14334402, 13841023, 683879]
    labels = ["14,334,402", "13,841,023（96.6%）", "683,879"]
    colors = [C_IN, C_CORE, C_OUT]

    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=150)
    fig.patch.set_facecolor("white")
    widths = np.array(vals) / vals[0]
    y = np.arange(len(stages))[::-1]
    for i, (w, c) in enumerate(zip(widths, colors)):
        ax.barh(y[i], w, left=(1 - w) / 2, height=0.62, color=c, zorder=3)
        if w > 0.35:  # 条形够宽：白字放条内；太窄则放条外右侧，避免白字落在白底上
            ax.text(0.5, y[i], f"{stages[i]}\n{labels[i]}", ha="center", va="center",
                    fontsize=12, color="white", weight="bold", linespacing=1.5)
        else:
            ax.text(0.5 - w / 2 - 0.03, y[i], stages[i], ha="right", va="center",
                    fontsize=12, color="#1F2937", weight="bold", linespacing=1.5)
            ax.text(0.5 + w / 2 + 0.03, y[i], labels[i], ha="left", va="center",
                    fontsize=12, color="#1F2937", weight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(-0.6, 2.6)
    ax.axis("off")
    ax.set_title("C-05 Anchor 点云构建漏斗（125 视角 · 深度 504p · 一致性保留率 96.6%）",
                 fontsize=14, weight="bold", pad=14)
    ax.annotate("", xy=(0.82, 0.9), xytext=(0.82, 1.9),
                arrowprops=dict(arrowstyle="->", color=C_GREY, lw=2))
    ax.text(0.86, 1.4, "96.6%\n仅剔遮挡边界", fontsize=10, color=C_GREY, va="center")
    ax.annotate("", xy=(0.55, -0.1), xytext=(0.82, 0.75),
                arrowprops=dict(arrowstyle="->", color=C_GREY, lw=2))
    ax.text(0.60, 0.25, "跨视角重复合并\n−95.1%", fontsize=10, color=C_GREY)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_anchor_funnel.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("✅ fig3_anchor_funnel.png")


# ------------------------------------------------------------- fig4 实验进度
def fig_timeline():
    # (标签, 开始日, 持续天, 状态)  状态: done / running / todo
    items = [
        ("A 环境搭建", 0.0, 0.3, "done"),
        ("B-01 权重下载", 0.3, 0.6, "done"),
        ("B-02 DA3 推理+验收", 0.8, 0.3, "done"),
        ("B-03 COLMAP 基线", 1.0, 0.8, "done"),
        ("B-04 DA3-init 基线", 1.5, 1.0, "done"),
        ("B-04b 内参隔离", 2.3, 0.9, "done"),
        ("C-05 anchor 构建", 1.0, 0.4, "done"),
        ("C-06 双分支2DGS训练器", 3.0, 1.5, "running"),
        ("D 消融(τ/offset界/分辨率)", 4.5, 1.5, "todo"),
        ("E 文档定稿", 6.0, 0.8, "todo"),
    ]
    c = {"done": C_OUT, "running": C_WARN, "todo": "#B9C0CC"}

    fig, ax = plt.subplots(figsize=(13.33, 5.6), dpi=150)
    fig.patch.set_facecolor("white")
    for i, (name, s, d, st) in enumerate(items):
        ax.barh(len(items) - i, d, left=s, height=0.6, color=c[st], zorder=3,
                alpha=1.0 if st != "todo" else 0.7)
        ax.text(s - 0.06, len(items) - i, name, ha="right", va="center", fontsize=10.5)
    ax.set_yticks([]); ax.set_xlim(-0.1, 7.2)
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
    ax.set_xticklabels(["11-03", "11-04", "11-05", "11-06", "11-07", "11-08", "11-09", "11-10"], fontsize=10)
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.set_title("da3_human 实验计划进度（截至 2026-11-05）", fontsize=14, weight="bold", pad=12)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc=C_OUT, label="已完成"), Patch(fc=C_WARN, label="进行中"),
                       Patch(fc="#B9C0CC", label="待做")],
              loc="lower right", fontsize=10, framealpha=0.9)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_timeline.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("✅ fig4_timeline.png")


if __name__ == "__main__":
    fig_framework()
    fig_baselines()
    fig_funnel()
    fig_timeline()
    print(f"🎉 全部输出: {OUT}")
