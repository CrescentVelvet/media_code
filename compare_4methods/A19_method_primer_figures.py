#!/usr/bin/env python3
"""PPT 科普配图：给完全不懂技术的读者讲清五方法对比（2026-11-09 峰哥要求）。

四张图（素材口径全部来自 media_paper 调研笔记，见各图脚注）：
  fig37_gaussian_101.png   什么是 3D 高斯：一颗椭球的 5 个属性 → 投影成色斑 → 610 万颗成场景
  fig38_five_pipelines.png 五方法泳道总览：看懂场景 → 生成高斯 → 要不要回炉 → 耗时
  fig39_da3gs_deepdive.png 主角 DA3-GS 内部机制深挖（DINOv2 / depth-ray / GS-DPT）
  fig40_gaussian_philosophy.png 三种"生成高斯"的哲学：逐像素 / 查询式 / 逐场景雕塑

口径纪律：技术细节以 media_paper 笔记为准 ——
  DA3:    3D-20251113-Depth-Anything-3.html（depth-ray / Dual-DPT / GS-DPT / teacher-student）
  QuerySplat: 3D-20261105-QuerySplat.html（几何/外观双查询解耦、8192 查询 × 64 颗）
  InfiniSplat v2: 3D-20261014-InfiniSplat-v2.html（冻结几何、位置钉死、3000 步精调）
  VGGT-Ω: 3D-20260514-VGGT-Omega.html（CVPR2026 Oral，几何 SOTA 骨干）
  3DGS:   3D-20230808-3DGS基础与改进综述.html（显式椭球、可微光栅化、30 分钟训练）
耗时/规模数字与 FINDINGS §2 实测表一致。
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parent / "ppt_assets"
OUT.mkdir(exist_ok=True)

FONT = "/mnt/c/Windows/Fonts/msyh.ttc"
FONT_B = "/mnt/c/Windows/Fonts/msyhbd.ttc"
for f in (FONT, FONT_B):
    if Path(f).exists():
        font_manager.fontManager.addfont(f)
FP = font_manager.FontProperties(fname=FONT)
FPB = font_manager.FontProperties(fname=FONT_B)
plt.rcParams["font.family"] = FP.get_name()
plt.rcParams["axes.unicode_minus"] = False

C_HERO = "#0F6E6E"      # DA3-GS 主角
C_FF = "#5B7FB5"        # QuerySplat
C_HY = "#8B6FA8"        # InfiniSplat v2
C_OP = "#B5763A"        # 参照（3DGS 系）
C_INK = "#2B2B2B"
C_SUB = "#5E5E5E"
C_FAINT = "#909090"
C_LINE = "#DCD8D2"

# 一张真实渲染图（fig37 第三格用）：DA3-GS 单方法环视视频的抽帧（单格 540×720）
REAL_FRAME = Path("/home/velvet/output/da3_gs_results/佳佳抱枕_p1m_off/orbit_video/video_orbit_arc_frame_150.png")


def txt(ax, x, y, s, fs=11, c=C_INK, w="normal", ha="center", va="center", ls=1.5):
    ax.text(x, y, s, fontsize=fs, color=c, ha=ha, va=va,
            fontproperties=(FPB if w == "bold" else FP),
            fontweight=("bold" if w == "bold" else "normal"),
            linespacing=ls)


def box(ax, x0, y0, x1, y1, fc="#FFFFFF", ec=C_LINE, lw=1.2, r=0.010):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0,rounding_size=%f" % r,
                                linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2))


def arrow(ax, x0, y0, x1, y1, c=C_SUB, lw=2.2, style="-|>", ms=16):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=ms, linewidth=lw, color=c, zorder=3))


def gaussian_blob(ax, cx, cy, w, h, color, n=7, amax=0.55, angle=0):
    """同心椭圆模拟一颗高斯的透明度衰减。"""
    for i in range(n, 0, -1):
        f = i / n
        ax.add_patch(Ellipse((cx, cy), w * f, h * f, angle=angle,
                             facecolor=color, alpha=amax * (1 - f) ** 1.6 + 0.06,
                             edgecolor="none", zorder=4))


# ────────────────────────────────────────────────────────────────────────────
def fig_gaussian_101():
    fig = plt.figure(figsize=(16.4, 7.6), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    txt(ax, 0.5, 0.955, "先认识唯一的\"积木\"：3D 高斯 —— 一颗半透明的彩色椭球", 21, w="bold")
    txt(ax, 0.5, 0.905,
        "五方法殊途同归，最终都在摆放同一种东西；区别只在\"谁来摆、摆多快\"", 12.5, c=C_SUB)

    P = [  # (x0, x1, 标题, 颜色)
        (0.030, 0.345, "① 一颗高斯 = 5 个数字", C_HERO),
        (0.375, 0.640, "② 渲染 = 把椭球\"拍扁\"成色斑", C_FF),
        (0.670, 0.972, "③ 610 万颗叠在一起 = 整个房间", C_OP),
    ]
    for x0, x1, t, c in P:
        box(ax, x0, 0.075, x1, 0.855, fc="#FDFDFC", ec=c, lw=1.6)
        txt(ax, (x0 + x1) / 2, 0.815, t, 13.5, c=c, w="bold")

    # ── 格 1：一颗椭球 + 属性标注 ──
    cx, cy = 0.187, 0.470
    gaussian_blob(ax, cx, cy, 0.16, 0.30, C_HERO, n=9, amax=0.62, angle=18)
    ax.add_patch(Ellipse((cx, cy), 0.16, 0.30, angle=18, fill=False,
                         edgecolor=C_HERO, lw=2.0, zorder=5))
    ax.plot(cx, cy, "+", color="white", ms=11, mew=2.2, zorder=6)
    attrs = [
        ("中心位置 μ", "椭球摆在空间的哪里"),
        ("三个轴长 σ", "胖瘦高矮（各向异性）"),
        ("旋转 R", "朝向哪个方向"),
        ("不透明度 α", "越透明越像一层薄雾"),
        ("颜色 SH", "球谐系数，可随视角变化"),
    ]
    for i, (k, v) in enumerate(attrs):
        y = 0.310 - i * 0.052
        txt(ax, 0.052, y, k, 10.5, c=C_HERO, w="bold", ha="left")
        txt(ax, 0.052, y - 0.023, v, 9.2, c=C_SUB, ha="left")

    # ── 格 2：相机 → 投影 → 屏幕色斑 ──
    ax.plot(0.432, 0.575, "o", ms=13, color=C_INK, zorder=6)
    txt(ax, 0.432, 0.640, "相机", 10.5, w="bold")
    ax.plot([0.432, 0.545], [0.575, 0.545], color=C_SUB, lw=1.4, ls=":")
    ax.plot([0.432, 0.545], [0.575, 0.415], color=C_SUB, lw=1.4, ls=":")
    ax.add_patch(Rectangle((0.545, 0.300), 0.062, 0.36, facecolor="#FFFFFF",
                           edgecolor=C_FF, lw=1.8, zorder=4))
    gaussian_blob(ax, 0.576, 0.505, 0.075, 0.115, C_FF, n=7, amax=0.6)
    txt(ax, 0.576, 0.215, "屏幕上的色斑\n近大 · 远小 · 可叠加", 10, c=C_FF)
    arrow(ax, 0.470, 0.575, 0.540, 0.560, c=C_SUB, lw=1.6, ms=12)
    txt(ax, 0.505, 0.640, "splatting\n（拍扁投影）", 9.5, c=C_SUB)
    txt(ax, 0.508, 0.120,
        "一颗高斯投到屏幕上是一个渐变色斑；\n前后遮挡按透明度混合，就像水彩叠色", 9.8, c=C_SUB)

    # ── 格 3：真实渲染 + 椭球叠加示意 ──
    if REAL_FRAME.exists():
        try:
            img = plt.imread(str(REAL_FRAME))
            ax.imshow(img, extent=(0.690, 0.952, 0.360, 0.760), zorder=4,
                      aspect="auto")
            box(ax, 0.690, 0.360, 0.952, 0.760, fc="none", ec=C_OP, lw=1.6, r=0.004)
        except Exception:
            box(ax, 0.690, 0.360, 0.952, 0.760, fc="#F5EFE8", ec=C_OP, lw=1.6)
    rng = np.random.default_rng(7)
    for _ in range(26):
        ex = 0.700 + rng.random() * 0.24
        ey = 0.375 + rng.random() * 0.36
        # 只撒在人物身上（画面中部偏下），白圈才有"这几颗属于人"的示意感
        if not (0.755 < ex < 0.890 and 0.43 < ey < 0.72):
            continue
        ax.add_patch(Ellipse((ex, ey), 0.016 + rng.random() * 0.012,
                             0.010 + rng.random() * 0.008,
                             angle=rng.random() * 180, fill=False,
                             edgecolor="white", lw=0.9, alpha=0.85, zorder=5))
    txt(ax, 0.821, 0.315, "白圈 = 其中几颗高斯（示意）", 9.2, c=C_SUB)
    txt(ax, 0.821, 0.205,
        "场景里没有网格、没有神经网络，\n只有 610 万颗椭球各就各位；\n"
        "\"换个角度看\" = 把每颗重新投影一次", 10, c=C_SUB)
    txt(ax, 0.5, 0.032,
        "3D Gaussian Splatting（SIGGRAPH 2023，INRIA）开创的表示：显式、可编辑、可实时渲染 —— 五方法对比的本质是\"这 610 万颗分别由谁、用多快摆出来\"",
        10.2, c=C_FAINT)

    p = OUT / "fig37_gaussian_101.png"
    fig.savefig(p, dpi=200, facecolor="white")
    plt.close(fig)
    print("✅", p)


# ────────────────────────────────────────────────────────────────────────────
def fig_five_pipelines():
    fig = plt.figure(figsize=(17.2, 9.8), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    txt(ax, 0.5, 0.962, "同一份输入（32 张照片），五条路线怎么把它变成可旋转的 3D 模型", 20.5, w="bold")
    txt(ax, 0.5, 0.922, "每一步都在回答一个问题：① 谁来\"看懂\"场景　② 高斯怎么生出来　③ 生完要不要回炉　｜　耗时为实测（RTX 3090）", 12, c=C_SUB)

    # 列定义：(标题, x0, x1)
    COLH = 0.880
    cols = [
        ("① 谁来\"看懂\"场景\n（几何骨干：估深度、相机、点云）", 0.135, 0.400),
        ("② 高斯怎么生出来", 0.415, 0.640),
        ("③ 生完要不要回炉", 0.655, 0.830),
        ("端到端耗时", 0.845, 0.975),
    ]
    for t, x0, x1 in cols:
        txt(ax, (x0 + x1) / 2, COLH, t, 11.5, c="#4A4A4A", w="bold")

    # 输入图标列
    txt(ax, 0.070, COLH, "输入", 11.5, c="#4A4A4A", w="bold")

    ROWS = [
        dict(name="DA3-GS", tag="主角 · 前馈直出", c=C_HERO, hero=True,
             understand="DA3 大模型一次前向\n每像素算出深度 + 光线，\n点云、相机一次全给（不找特征匹配点）",
             gen="同一模型接 GS 头：\n每个像素直接吐 1 颗高斯的\n位置·大小·朝向·透明度·颜色",
             refine="不回炉\n网络跑完即成品",
             time="13.5 秒"),
        dict(name="QuerySplat", tag="同档 · 前馈直出", c=C_FF,
             understand="冻结 VGGT-Ω 大模型\n复用它\"看懂\"的几何先验",
             gen="放 8192 个\"查询槽\"，\n每槽解码 64 颗高斯：\n几何查询管摆放、外观查询管上色",
             refine="不回炉\n一次前向即成品",
             time="38.5 秒"),
        dict(name="InfiniSplat v2", tag="同档 · 前馈+短精调", c=C_HY,
             understand="把场景切块，每块各跑\n一遍 DA3 大模型，\n拼成稠密点云",
             gen="每块逐像素出高斯，\n体素合并去重\n（594 万颗）",
             refine="回炉 3000 步，但\n位置钉死，只调颜色\n外观（163 秒）",
             time="≈ 3.3 分钟"),
        dict(name="VGGT-Ω + 3DGS", tag="参照 · 逐场景优化", c=C_OP,
             understand="VGGT-Ω 前向出点云\n再用 COLMAP 传统算法\n精化相机位姿",
             gen="从点云撒初始高斯\n（179 万颗）",
             refine="回炉 30,000 步：\n反复\"渲染→和照片比对→\n全部参数微调\"（≈20 分钟）",
             time="20.4 分钟"),
        dict(name="DA3 + 3DGS", tag="参照 · 逐场景优化", c=C_OP,
             understand="DA3 前向出点云\n再用 COLMAP 传统算法\n精化相机位姿",
             gen="从点云撒初始高斯\n（189 万颗）",
             refine="回炉 30,000 步：\n反复\"渲染→和照片比对→\n全部参数微调\"（≈20 分钟）",
             time="22.9 分钟"),
    ]

    top, gap = 0.845, 0.012
    rh = (top - 4 * gap - 0.055) / 5
    for i, r in enumerate(ROWS):
        y1 = top - i * (rh + gap)
        y0 = y1 - rh
        ym = (y0 + y1) / 2
        bgc = "#F3FAF9" if r.get("hero") else ("#FBF7F2" if r["c"] == C_OP else "#FAFAF8")
        box(ax, 0.008, y0, 0.992, y1, fc=bgc,
            ec=r["c"] if r.get("hero") else C_LINE, lw=2.2 if r.get("hero") else 1.0)
        # 方法名
        txt(ax, 0.070, ym + 0.028, r["name"], 13 if r.get("hero") else 12, c=r["c"], w="bold")
        txt(ax, 0.070, ym - 0.030, r["tag"], 8.6, c=C_FAINT)
        # 输入小图（照片堆示意）
        for k in range(3):
            box(ax, 0.024 + k * 0.008, ym - 0.035 + k * 0.010,
                0.058 + k * 0.008, ym + 0.005 + k * 0.010,
                fc="#FFFFFF", ec=C_SUB, lw=0.9, r=0.003)
        # 三个内容格 + 箭头
        xs = [(0.135, 0.400), (0.415, 0.640), (0.655, 0.830)]
        for (x0, x1), key in zip(xs, ("understand", "gen", "refine")):
            txt(ax, (x0 + x1) / 2, ym, r[key], 9.6 if len(r[key]) < 40 else 9.0,
                c=C_INK, ls=1.45)
        arrow(ax, 0.405, ym, 0.413, ym, lw=1.6, ms=10)
        arrow(ax, 0.645, ym, 0.653, ym, lw=1.6, ms=10)
        arrow(ax, 0.100, ym, 0.130, ym, lw=1.6, ms=10)
        arrow(ax, 0.833, ym, 0.842, ym, lw=1.6, ms=10)
        # 耗时
        if r.get("hero"):
            box(ax, 0.845, y0 + 0.030, 0.975, y1 - 0.030, fc=r["c"], ec=r["c"], lw=1.5)
            txt(ax, 0.910, ym, r["time"], 14.5, c="white", w="bold")
        else:
            txt(ax, 0.910, ym, r["time"], 13, c=r["c"], w="bold")

    txt(ax, 0.5, 0.030,
        "路线差别不在\"用哪个大模型\"，而在第③列：DA3-GS / QuerySplat 一次前向直接出成品；InfiniSplat 只回炉 3000 步且位置不动；两条 3DGS 参照要回炉 30,000 步全参数精雕 —— 这就是 13 秒 vs 20 分钟的来源",
        10.6, c="#8A4A86")

    p = OUT / "fig38_five_pipelines.png"
    fig.savefig(p, dpi=200, facecolor="white")
    plt.close(fig)
    print("✅", p)


# ────────────────────────────────────────────────────────────────────────────
def fig_da3gs_deepdive():
    fig = plt.figure(figsize=(17.2, 9.2), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    txt(ax, 0.5, 0.960, "主角深挖：DA3-GS 凭什么一次前向 13.5 秒就交出 610 万颗高斯？", 20.5, w="bold")
    txt(ax, 0.5, 0.918, "三个反直觉设计：不找匹配点、不单独估相机、没有迭代循环 —— 来自 Depth Anything 3（字节 Seed，ICLR 2026 Oral）", 12, c=C_SUB)

    # ── 主流程 6 节点（两行各 3 个，之字形）──
    N = [  # (x0,y0,x1,y1, 标题, 正文, 颜色)
        (0.030, 0.560, 0.200, 0.845, "输入：32 张照片",
         "随便拍的一组环绕照片\n（数量、顺序都不挑）", C_INK),
        (0.235, 0.560, 0.445, 0.845, "单一预训练大模型\n（DINOv2 transformer）",
         "32 张照片的每个小块(patch)\n在模型里互相\"对眼神\"\n（交替注意力：先各自看、再一起看）",
         C_HERO),
        (0.480, 0.560, 0.690, 0.845, "每像素两个预测：\n深度 D + 光线 (t, d)",
         "深度：这个像素离相机多远\n光线：它从哪个点、沿哪个方向\n射出来的（6 个数）",
         C_HERO),
        (0.480, 0.205, 0.690, 0.460, "点云免费得到\nP = t + D·d",
         "深度乘光线方向逐像素还原 3D 点\n—— 全程不做特征匹配，\n这是和 COLMAP 路线的分水岭",
         C_HERO),
        (0.235, 0.205, 0.445, 0.460, "相机也不用单独估",
         "光线原点就是相机中心；\n方向里做一道最小二乘\n(DLT+RANSAC) 直接解出位姿 ——\n\"学深度\"被迫把相机也学对",
         C_HERO),
        (0.030, 0.205, 0.200, 0.460, "GS-DPT 头：\n每像素吐 1 颗高斯",
         "冻结大模型，只训练这个小头：\n每个像素再给出高斯的\n颜色·大小·朝向·透明度\n→ 610 万颗，写成 PLY 文件",
         C_HERO),
    ]
    for x0, y0, x1, y1, t, b, c in N:
        box(ax, x0, y0, x1, y1, fc="#FDFEFD", ec=c, lw=1.8)
        txt(ax, (x0 + x1) / 2, y1 - 0.043, t, 12, c=c, w="bold")
        txt(ax, (x0 + x1) / 2, (y0 + y1) / 2 - 0.030, b, 10, c=C_INK)

    # 节点间箭头（之字）
    arrow(ax, 0.202, 0.700, 0.232, 0.700, c=C_HERO)          # 1→2
    arrow(ax, 0.447, 0.700, 0.477, 0.700, c=C_HERO)          # 2→3
    arrow(ax, 0.585, 0.557, 0.585, 0.463, c=C_HERO)          # 3→4
    arrow(ax, 0.477, 0.330, 0.447, 0.330, c=C_HERO)          # 4→5
    arrow(ax, 0.232, 0.330, 0.202, 0.330, c=C_HERO)          # 5→6

    # ── 右上：teacher-student 小卡 ──
    box(ax, 0.725, 0.560, 0.975, 0.845, fc="#F7F4EF", ec=C_OP, lw=1.4)
    txt(ax, 0.850, 0.812, "模型为什么「懂」几何？—— 训练期的事，与推理无关", 11.5, c=C_OP, w="bold")
    txt(ax, 0.850, 0.690,
        "先在 20 个合成数据集上训一个\n\"单目深度老师\"，再用 RANSAC 把它的\n"
        "预测对齐到真实数据的稀疏实测值，\n教出\"多视角学生\"。\n\n"
        "一句话：推理时喂照片就完事，\n昂贵的训练只做一次，\n摊薄到每个场景 ≈ 0 成本。",
        9.8, c=C_INK, ls=1.5)

    # ── 右下：三个关键数字 ──
    box(ax, 0.725, 0.205, 0.975, 0.460, fc="#F3FAF9", ec=C_HERO, lw=1.8)
    txt(ax, 0.850, 0.428, "一次前向的账本（实测）", 11.5, c=C_HERO, w="bold")
    for i, (k, v) in enumerate([("看懂场景 + 出高斯（前向）", "6.2 秒"),
                                ("写成 PLY 文件（导出）", "6.8 秒"),
                                ("回炉优化", "0 —— 没有这一步"),
                                ("产出", "610 万颗 / 953 MB / 22.11 dB")]):
        y = 0.378 - i * 0.047
        txt(ax, 0.745, y, k, 10, c=C_INK, ha="left")
        txt(ax, 0.955, y, v, 10.5, c=C_HERO, w="bold", ha="right")

    txt(ax, 0.5, 0.120,
        "对照两条 3DGS 参照路线：它们到\"点云\"这一步就停了，接下来要新开一个优化器把高斯反复打磨 3 万步 —— DA3-GS 把\"打磨\"也变成了网络前向的一部分，所以能快 100 倍",
        11.5, c="#8A4A86")
    txt(ax, 0.5, 0.055,
        "机制出处：Depth Anything 3 论文 §3（depth-ray 表示）、§5（GS-DPT 前馈 3DGS）；训练范式 §4（teacher-student）。实测口径同 FINDINGS §2。",
        9.5, c=C_FAINT)

    p = OUT / "fig39_da3gs_deepdive.png"
    fig.savefig(p, dpi=200, facecolor="white")
    plt.close(fig)
    print("✅", p)


# ────────────────────────────────────────────────────────────────────────────
def fig_gaussian_philosophy():
    fig = plt.figure(figsize=(17.2, 8.8), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    txt(ax, 0.5, 0.955, "三种\"摆放高斯\"的哲学：像素网格 × 查询槽 × 逐场景雕塑", 20.5, w="bold")
    txt(ax, 0.5, 0.910, "同样摆 610 万颗椭球，三条路线对\"自由度\"和\"时间\"的取舍完全不同", 12, c=C_SUB)

    COLS = [
        (0.030, 0.345, "A · 逐像素直出", "DA3-GS（13.5 秒）", C_HERO),
        (0.375, 0.640, "B · 查询式", "QuerySplat（38.5 秒）", C_FF),
        (0.670, 0.972, "C · 逐场景雕塑", "InfiniSplat 精调 + 两条 3DGS 参照（3.3 分钟 ~ 23 分钟）", C_OP),
    ]
    top = 0.860
    for x0, x1, t, sub, c in COLS:
        box(ax, x0, 0.070, x1, 0.875, fc="#FDFDFC", ec=c, lw=1.6)
        txt(ax, (x0 + x1) / 2, 0.838, t, 14, c=c, w="bold")
        txt(ax, (x0 + x1) / 2, 0.798, sub, 9.6, c=C_SUB)

    # A 格：像素网格 → 每格一颗高斯
    gx0, gy0, cell = 0.055, 0.520, 0.030
    rng = np.random.default_rng(3)
    for i in range(5):
        for j in range(5):
            x = gx0 + i * cell * 1.35
            y = gy0 + j * cell * 1.6
            ax.add_patch(Rectangle((x, y), cell * 1.1, cell * 1.35,
                                   facecolor="#EAF4F3", edgecolor=C_HERO, lw=0.8))
            gaussian_blob(ax, x + cell * 0.55, y + cell * 0.67, cell * 0.75,
                          cell * 0.95, C_HERO, n=4, amax=0.6, angle=rng.random() * 60)
    txt(ax, 0.190, 0.470, "每个像素固定对应 1 颗高斯：\n像素网格 = 高斯网格", 10, c=C_HERO)
    txt(ax, 0.190, 0.300,
        "√ 结构整齐、一次前向最快\n× 高斯被相机射线\"绑死\"：\n　深度估错一点，高斯就飘到\n　错误的位置（视角间重影）",
        9.6, c=C_INK, ha="center", va="top", ls=1.6)
    txt(ax, 0.190, 0.108, "610 万颗 = 32 视角 × 每像素 1 颗", 9.2, c=C_SUB)

    # B 格：查询槽
    qx, qy = 0.415, 0.560
    for i in range(8):
        for j in range(3):
            x = qx + i * 0.026
            y = qy + j * 0.075
            box(ax, x, y, x + 0.020, y + 0.058, fc="#EDF1F8", ec=C_FF, lw=1.0, r=0.004)
            ax.plot(x + 0.010, y + 0.029, "o", ms=2.6, color=C_FF)
    txt(ax, 0.508, 0.828 - 0.046, "", 1)  # spacer noop
    txt(ax, 0.508, 0.510, "8192 个\"查询槽\"，每个槽解码 64 颗\n＝ 52 万颗，落在连续空间里", 10, c=C_FF)
    txt(ax, 0.508, 0.340,
        "√ 解开像素网格：高斯想去哪去哪\n√ 分工防糊：几何查询管\"摆放\"，\n　外观查询管\"上色\"（纠缠才会糊）\n× 查询预算有限，表达力有天花板",
        9.6, c=C_INK, va="top", ls=1.6)
    txt(ax, 0.508, 0.108, "52 万颗 远少于 610 万颗 —— 本场景实测它更糊", 9.2, c=C_SUB)

    # C 格：雕塑循环
    steps = [("① 给一坨初始点云", 0.700, 0.700),
             ("② 渲染出图，和照片比对", 0.862, 0.560),
             ("③ 哪里不像调哪里\n（位置·颜色·大小全可调）", 0.700, 0.420),
             ("④ 重复 3 万次", 0.862, 0.280)]
    for t, x, y in steps:
        box(ax, x - 0.075, y - 0.042, x + 0.075, y + 0.042, fc="#FBF3EA", ec=C_OP, lw=1.3)
        txt(ax, x, y, t, 9.4, c=C_INK)
    arrow(ax, 0.775, 0.660, 0.830, 0.606, c=C_OP, lw=1.8)
    arrow(ax, 0.830, 0.520, 0.775, 0.464, c=C_OP, lw=1.8)
    arrow(ax, 0.775, 0.378, 0.830, 0.324, c=C_OP, lw=1.8)
    ax.add_patch(FancyArrowPatch((0.927, 0.240), (0.927, 0.660),
                                 connectionstyle="arc3,rad=-0.92",
                                 arrowstyle="-|>", mutation_scale=15,
                                 linewidth=1.8, color=C_OP, zorder=3))
    txt(ax, 0.821, 0.175, "", 1)
    txt(ax, 0.821, 0.108,
        "InfiniSplat 是\"轻雕塑\"：只调颜色外观 3000 步、位置钉死\n两条 3DGS 参照是\"全雕塑\"：30,000 步全参数",
        9.2, c=C_SUB)

    txt(ax, 0.5, 0.030,
        "自由度和时间成正比：A 最快但被网格锁死，B 解开网格但预算有限，C 无拘无束但每个场景都要单独精雕 —— 大规模生产只能选 A 或 B",
        10.6, c="#8A4A86")

    p = OUT / "fig40_gaussian_philosophy.png"
    fig.savefig(p, dpi=200, facecolor="white")
    plt.close(fig)
    print("✅", p)


if __name__ == "__main__":
    fig_gaussian_101()
    fig_five_pipelines()
    fig_da3gs_deepdive()
    fig_gaussian_philosophy()
