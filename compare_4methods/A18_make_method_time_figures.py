#!/usr/bin/env python3
"""PPT 配图：方法对比矩阵（fig33）+ 耗时对比（fig34）。

角色划分（2026-11-09 峰哥确认）
------------------------------
  主角    DA3-GS（DA3 原生高斯头，一次前向 13.5 s 直出）
  同档    QuerySplat / InfiniSplat v2（同为秒级~分钟级前馈）
  参照    DA3+3DGS / VGGT-Ω+3DGS（20+ 分钟，只说明"多花 100 倍时间买到什么"）

口径纪律
--------
· PSNR 只在 DA3-GS 上标：其余四方法没有在同一批相机上重渲染，
  拿不同口径的 PSNR 硬排是错的（ERRATA E6）。本图因此**不排质量名次**，
  只排「机制」与「耗时」。
· DA3-GS 的高斯数 / 体积给两档：完整输出 610 万 / 953 MB（22.11 dB），
  轻量交付 484 万 / 314 MB（18.97 dB）。差 3.14 dB 全部来自导出设置，模型未变。
· 耗时均为 RTX 3090 单卡、同一台机器、同一场景（佳佳抱枕 32 视角）。
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent / "ppt_assets"
OUT.mkdir(exist_ok=True)

FONT = "/mnt/c/Windows/Fonts/msyh.ttc"
FONT_B = "/mnt/c/Windows/Fonts/msyhbd.ttc"
for f in (FONT, FONT_B):
    if Path(f).exists():
        font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False

C_HERO = "#0F6E6E"      # 主角
C_FF = "#5B7FB5"        # 前馈直出（同档）
C_HY = "#8B6FA8"        # 前馈 + 短精调
C_OP = "#B5763A"        # 前馈 + 逐场景优化（参照）
C_GRAY = "#6E6E6E"
C_LINE = "#DCD8D2"

ROWS = [
    dict(name="DA3-GS", tag="主角 · 一次前向直出", color=C_HERO, hero=True,
         geom="DA3 单次前向\n直接给出米制点云、相机与深度\n（不经 COLMAP、不需逐场景拟合）",
         gauss="原生高斯头\n逐像素 1 个高斯，颜色含 SH 高阶\n6.2 s 前向 + 6.8 s 导出",
         opt="无\n网络跑完即可渲染",
         size="610 万\n953 MB", time="13.5 秒"),
    dict(name="QuerySplat", tag="同档 · 前馈直出", color=C_FF,
         geom="冻结 VGGT-Ω 骨干\n复用其几何先验",
         gauss="查询式高斯头\n沿参考视角射线采样解码",
         opt="无\n一次前向即可渲染",
         size="37.1 万\n33 MB", time="38.5 秒"),
    dict(name="InfiniSplat v2", tag="同档 · 前馈 + 短精调", color=C_HY,
         geom="单次前向\n生成稠密高斯场",
         gauss="前馈高斯场\n594 万点，位置/尺度/颜色一次给出",
         opt="3000 步短精调\n位置冻结，只调外观",
         size="594 万\n386 MB", time="前馈 33.9 秒\n+ 精调 163 秒\n≈ 3.3 分钟", time_fs=11.0),
    dict(name="VGGT-Ω + 3DGS", tag="参照 · 前馈几何 + 逐场景优化", color=C_OP,
         geom="VGGT-Ω 前向出点云\n→ COLMAP 精化位姿",
         gauss="3DGS 从点云初始化\n在 COLMAP 场景内重建",
         opt="30k 步训练\n约 20 分钟",
         size="175 万\n414 MB", time="20.4 分钟"),
    dict(name="DA3 + 3DGS", tag="参照 · 前馈几何 + 逐场景优化", color=C_OP,
         geom="DA3 前向出点云\n→ COLMAP 精化位姿",
         gauss="3DGS 从点云初始化\n在 COLMAP 场景内重建",
         opt="30k 步训练\n约 20 分钟",
         size="189 万\n448 MB", time="22.9 分钟"),
]

COLS = [
    ("", 0.000, 0.178),
    ("① 几何与位姿从哪来", 0.184, 0.404),
    ("② 高斯从哪来", 0.410, 0.622),
    ("③ 要不要逐场景优化", 0.628, 0.768),
    ("④ 高斯数 / 体积", 0.774, 0.874),
    ("⑤ 端到端耗时", 0.880, 0.998),
]


def cell(ax, x0, x1, y0, y1, text, fc, tc="#2B2B2B", fs=9.6, bold=False, ec=None, lw=1.0):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0,rounding_size=0.006",
        linewidth=lw, edgecolor=ec or C_LINE, facecolor=fc))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, weight="bold" if bold else "normal",
            linespacing=1.55)


def fig_matrix():
    fig = plt.figure(figsize=(17.0, 9.4), facecolor="white")
    ax = fig.add_axes([0.012, 0.132, 0.976, 0.700])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.text(0.5, 0.965,
             "五方法机制对比：DA3-GS 一次前向直接出高斯，另外两条路线要多花 100 倍时间",
             ha="center", va="center", fontsize=21, weight="bold", color="#1B1B1B")
    fig.text(0.5, 0.918,
             "同一场景（佳佳抱枕 · 32 视角）· 同一台 RTX 3090　｜　"
             "上三行是候选方案，下两行只作「多花的 100 倍时间买到了什么」的参照",
             ha="center", va="center", fontsize=12.4, color="#5E5E5E")

    # 表头
    yh0, yh1 = 0.945, 1.0
    for lab, x0, x1 in COLS:
        if lab:
            ax.text((x0 + x1) / 2, (yh0 + yh1) / 2, lab, ha="center", va="center",
                    fontsize=12.0, weight="bold", color="#4A4A4A")

    top = 0.930
    gap = 0.011
    rh = (top - 4 * gap) / 5
    for i, r in enumerate(ROWS):
        y1 = top - i * (rh + gap)
        y0 = y1 - rh
        hero = r.get("hero")
        base = "#F3FAF9" if hero else ("#FAFAF8" if r["color"] != C_OP else "#FBF7F2")

        # 方法名 + 角色标签
        cell(ax, COLS[0][1], COLS[0][2], y0, y1, "", base,
             ec=r["color"] if hero else C_LINE, lw=2.4 if hero else 1.0)
        ax.text((COLS[0][1] + COLS[0][2]) / 2, y0 + rh * 0.60, r["name"],
                ha="center", va="center", fontsize=15.0 if hero else 14.0,
                weight="bold", color=r["color"])
        ax.text((COLS[0][1] + COLS[0][2]) / 2, y0 + rh * 0.235, r["tag"],
                ha="center", va="center", fontsize=9.4, color="#7A7A7A")

        for (lab, x0, x1), key in zip(COLS[1:4], ("geom", "gauss", "opt")):
            cell(ax, x0, x1, y0, y1, r[key], base, fs=9.8)
        cell(ax, COLS[4][1], COLS[4][2], y0, y1, r["size"], base, fs=10.6, bold=True)
        # 耗时：主角用实心强调
        if hero:
            cell(ax, COLS[5][1], COLS[5][2], y0, y1, r["time"], r["color"],
                 tc="white", fs=15.0, bold=True, ec=r["color"], lw=2.4)
        else:
            cell(ax, COLS[5][1], COLS[5][2], y0, y1, r["time"], base,
                 tc=r["color"], fs=r.get("time_fs", 13.0), bold=True)

    fig.text(0.5, 0.082,
             "耗时构成：DA3-GS = 前向 6.2 s + 导出 6.8 s　｜　QuerySplat = 16.3 + 22.1 s　｜　"
             "InfiniSplat v2 = 前馈 33.9 s + 精调 163 s　｜　VGGT-Ω+3DGS = 前馈 35 s + 训练 1189 s　｜　"
             "DA3+3DGS = 前馈 130 s + 训练 1242 s",
             ha="center", va="center", fontsize=10.2, color="#6E6E6E")
    fig.text(0.5, 0.049,
             "质量只对 DA3-GS 报：完整输出 22.11 dB（610 万点 + 全 SH + 不裁剪，953 MB）；"
             "轻量交付 18.97 dB（裁剪到 484 万 + 仅 DC，314 MB）—— 两者是同一份前向结果的两种导出设置。",
             ha="center", va="center", fontsize=10.2, color="#8A4A86")
    fig.text(0.5, 0.018,
             "其余四方法没有在同一批相机上重渲染，故不标 PSNR、不排名次 —— 跨口径比质量会得出错误结论。",
             ha="center", va="center", fontsize=10.2, color="#909090")

    p = OUT / "fig33_method_matrix.png"
    fig.savefig(p, dpi=200, facecolor="white")
    plt.close(fig)
    print("✅", p)


def fig_time():
    # 名称, 前馈秒, 短精调秒, 3DGS训练秒, 颜色, 是否参照
    M = [
        ("DA3-GS", 13.5, 0, 0, C_HERO, False),
        ("QuerySplat", 38.5, 0, 0, C_FF, False),
        ("InfiniSplat v2", 33.9, 163, 0, C_HY, False),
        ("VGGT-Ω + 3DGS", 35, 0, 1189, C_OP, True),
        ("DA3 + 3DGS", 130, 0, 1242, C_OP, True),
    ]
    base = M[0][1]

    fig = plt.figure(figsize=(15.4, 7.8), facecolor="white")
    ax = fig.add_axes([0.185, 0.175, 0.535, 0.585])

    ys = np.arange(len(M))[::-1]          # y 越大越靠上，故 M[0] 落在最上面
    for y, (nm, ff, hy, op, col, is_ref) in zip(ys, M):
        left = 0.0
        for val, c in ((ff, col), (hy, C_HY), (op, C_OP)):
            if val <= 0:
                continue
            ax.barh(y, val, left=left, height=0.52, color=c,
                    alpha=0.92 if not is_ref else 0.72,
                    edgecolor="white", linewidth=1.2)
            left += val
        tot = ff + hy + op
        mult = tot / base
        lab = ("%.1f 秒" % tot) if tot < 90 else ("%.1f 分钟" % (tot / 60))
        tag = "%s（%s×）" % (lab, ("%.0f" % mult) if mult >= 10 else ("%.1f" % mult))
        if tot < 400:                     # 短条：标签放条外
            ax.text(tot + 18, y, tag, va="center", ha="left",
                    fontsize=12.0, weight="bold", color=col)
        else:                             # 长条：标签放条内右端，省出右侧版面
            ax.text(tot - 20, y, tag, va="center", ha="right",
                    fontsize=12.0, weight="bold", color="white")

    ax.set_yticks(ys)
    ax.set_yticklabels([m[0] for m in M], fontsize=13.5)   # 顺序必须与 ys 一一对应，别反转
    for t, m in zip(ax.get_yticklabels(), M):
        t.set_color(m[4])
        if m[0] == "DA3-GS":
            t.set_weight("bold")
    ax.set_xlim(0, 1440)
    ax.set_xlabel("端到端耗时（秒，含前馈 + 精调/训练 + 导出）", fontsize=12.5)
    ax.grid(axis="x", ls=":", lw=0.8, color="#D5D5D5")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=11)

    # 分段图例放 ax 上方，横向排 —— 放图内会压住长条
    hs = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (C_HERO, C_HY, C_OP)]
    fig.legend(hs, ["前馈（一次前向）", "短精调（位置冻结）", "30k 步逐场景优化"],
               fontsize=10.8, frameon=False, ncol=3,
               loc="upper left", bbox_to_anchor=(0.183, 0.862), handlelength=1.6)

    # 放大插图：0–250 s
    axi = fig.add_axes([0.745, 0.435, 0.235, 0.275])
    sub = M[:3]
    ys3 = np.arange(len(sub))[::-1]
    for y, (nm, ff, hy, op, col, _) in zip(ys3, sub):
        left = 0.0
        for val, c in ((ff, col), (hy, C_HY), (op, C_OP)):
            if val > 0:
                axi.barh(y, val, left=left, height=0.5, color=c, edgecolor="white", linewidth=1.0)
                left += val
        axi.text(left + 4, y, ("%.1f s" % left), va="center", fontsize=9.6,
                 weight="bold", color=col)
    axi.set_yticks(ys3)
    axi.set_yticklabels([m[0] for m in sub], fontsize=9.6)
    axi.set_xlim(0, 260)
    axi.set_title("放大：候选方案的秒级差异", fontsize=11.0, pad=6, color="#444444")
    axi.grid(axis="x", ls=":", lw=0.7, color="#DDDDDD")
    axi.set_axisbelow(True)
    for s in ("top", "right"):
        axi.spines[s].set_visible(False)
    axi.tick_params(labelsize=8.6)
    axi.axvline(base, color=C_HERO, ls="--", lw=1.2, alpha=0.7)

    fig.text(0.5, 0.945, "耗时对比：13.5 秒 vs 20~23 分钟 —— 差 90~100 倍",
             ha="center", va="center", fontsize=21, weight="bold", color="#1B1B1B")
    fig.text(0.5, 0.893,
             "横条按构成分段：前馈是一次前向的成本，后两段才是「为了更准而额外付的钱」",
             ha="center", va="center", fontsize=12.4, color="#5E5E5E")
    fig.text(0.5, 0.085,
             "两根橙色条里，前馈只占 2.5%（35 s / 1224 s）；即 97.5% 的时间花在逐场景的 30k 步优化上 —— "
             "这段开销与场景绑定，不能复用到下一个场景。",
             ha="center", va="center", fontsize=11.0, color="#8A4A86")
    fig.text(0.5, 0.040,
             "数据：RTX 3090 单卡，佳佳抱枕 32 视角。DA3-GS 的体积/质量给两档导出（953 MB / 22.11 dB 与 314 MB / 18.97 dB），"
             "耗时两档相同。",
             ha="center", va="center", fontsize=10.4, color="#909090")

    p = OUT / "fig34_time_compare.png"
    fig.savefig(p, dpi=200, facecolor="white")
    plt.close(fig)
    print("✅", p)


if __name__ == "__main__":
    fig_matrix()
    fig_time()
