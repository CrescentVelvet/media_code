#!/usr/bin/env python3
"""fig46~48：「X 在 Y 基础上改了什么」系列架构对比图（PPT 用，1920×1080）。

fig46 QuerySplat ← VGGT：backbone 冻结，改动全在 VGM 之后（双查询解码器）。
fig47 InfiniSplat v2 ← DA3：DA3 拆双分支，位置锁死，属性全学，外加 Refiner。
fig48 DA3 ↔ VGGT：两代 pose-free 前馈几何模型的架构分歧（纯绘制版式）。

事实口径：media_paper 笔记 3D-20261105(QuerySplat §4/§6/§7/§13 表4)、
3D-20261014(InfiniSplat v2 §4/§7/§11/§13 消融)、3D-20251113(DA3 §5/§13)、
3D-20250314(VGGT 架构节)；原图复用 media_paper/images。
字体坑同 A20：雅黑无 ✓✗ 字形，用 √×；显式 fontfamily 每个 text。

用法（WSL）：bash compare_4methods/_run_a21_wrapped.sh
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle
from matplotlib import font_manager

ROOT = os.path.dirname(os.path.abspath(__file__))
if os.path.exists("/mnt/c/code/media_paper"):
    PAPER = "/mnt/c/code/media_paper/images"
    FONT_R = "/mnt/c/Windows/Fonts/msyh.ttc"
    FONT_B = "/mnt/c/Windows/Fonts/msyhbd.ttc"
else:
    PAPER = r"C:\code\media_paper\images"
    FONT_R = r"C:\Windows\Fonts\msyh.ttc"
    FONT_B = r"C:\Windows\Fonts\msyhbd.ttc"
OUT = os.path.join(ROOT, "ppt_assets")

YAHEI = None
for f in (FONT_R, FONT_B):
    if os.path.exists(f):
        font_manager.fontManager.addfont(f)
        if YAHEI is None:
            YAHEI = font_manager.FontProperties(fname=f).get_name()
plt.rcParams["font.family"] = YAHEI or "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

C_MAIN = "#0F6E6E"    # DA3 / DA3-GS 青
C_BLUE = "#5B7FB5"    # QuerySplat 蓝
C_PURPLE = "#8B6FA8"  # InfiniSplat 紫
C_VGGT = "#C0764B"    # VGGT 暖棕
C_GREEN = "#4E7A4E"   # 冻结/保留
C_RED = "#B0503C"     # 弃用/锁死警示
C_GRAY = "#707070"

TAG_STYLE = {
    "冻结": ("#E8F0E8", C_GREEN),
    "新增": ("#E9EFF7", C_BLUE),
    "弃用": ("#F6E9E5", C_RED),
    "锁死": ("#F6E9E5", C_RED),
    "拆分": ("#E9EFF7", C_BLUE),
    "改输出": ("#E9EFF7", C_BLUE),
    "训练": ("#F3EEE3", "#8A6D3B"),
    "可替换": ("#F0F0ED", C_GRAY),
    "分歧": ("#F0F0ED", C_GRAY),
}


def _t(ax, *a, **kw):
    kw.setdefault("fontfamily", YAHEI or "sans-serif")
    return ax.text(*a, **kw)


def _canvas(title, subtitle, color):
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1920)
    ax.set_ylim(1080, 0)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 1920, 1080, fc="#FBFBFA", ec="none", zorder=0))
    ax.add_patch(Rectangle((0, 0), 1920, 118, fc="white", ec="#E2E2E0", zorder=1))
    ax.add_patch(Rectangle((0, 0), 14, 118, fc=color, ec="none", zorder=2))
    _t(ax, 40, 36, title, fontsize=29, fontweight="bold", color=color, va="center")
    _t(ax, 40, 82, subtitle, fontsize=16, color="#404040", va="center")
    return fig, ax


def _img_card(ax, x, y, w, h, img_path, caption, cap_color):
    """白底卡片 + 等比嵌论文原图 + 底部图注条。"""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc="white", ec="#D8D8D5", lw=1.2, zorder=1))
    img = mpimg.imread(img_path)
    ih, iw = img.shape[:2]
    top_h = 44
    scale = min((w - 32) / iw, (h - top_h - 44) / ih)
    dw, dh = iw * scale, ih * scale
    ax.imshow(img, extent=[x + (w - dw) / 2, x + (w + dw) / 2,
                           y + top_h + (h - top_h - 44 + dh) / 2,
                           y + top_h + (h - top_h - 44 - dh) / 2], zorder=3)
    ax.add_patch(Rectangle((x + 1, y + h - 42), w - 2, 41, fc=cap_color,
                           ec="none", zorder=2, alpha=0.92))
    _t(ax, x + w / 2, y + h - 21, caption, fontsize=14.5, fontweight="bold",
       color="white", ha="center", va="center", zorder=4)


def _change_rows(ax, rows, y0=700, note=None):
    """编号 + 标签 + 说明行。rows = [(tag, text), ...]"""
    ry = y0
    for i, (tag, text) in enumerate(rows, 1):
        ax.add_patch(Circle((64, ry), 17, fc="#404040", ec="none", zorder=3))
        _t(ax, 64, ry, str(i), fontsize=15, fontweight="bold", color="white",
           ha="center", va="center", zorder=4)
        fc, tc = TAG_STYLE[tag]
        ax.add_patch(FancyBboxPatch((96, ry - 15), 92, 30, boxstyle="round,pad=0.02",
                                    fc=fc, ec="none", zorder=3))
        _t(ax, 142, ry, tag, fontsize=13.5, fontweight="bold", color=tc,
           ha="center", va="center", zorder=4)
        _t(ax, 210, ry, text, fontsize=16.5, color="#262626", va="center", zorder=4)
        ry += 56
    if note:
        _t(ax, 40, ry + 16, note, fontsize=13, color="#808080", va="center")
    return ry


def _arrow_between(ax, x0, x1, y, label, color):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                                 mutation_scale=30, lw=4, color=color, zorder=5))
    _t(ax, (x0 + x1) / 2, y - 34, label, fontsize=14, fontweight="bold",
       color=color, ha="center", va="center", zorder=5)


# ── fig46：QuerySplat ← VGGT ────────────────────────────────────────────────
def fig46():
    fig, ax = _canvas(
        "QuerySplat：在 VGGT 基础上改了什么",
        "backbone 一行没动 —— 改动全部发生在冻结的 VGM 之后（arXiv:2608.01186，笔记 3D-20261105）",
        C_BLUE)
    _img_card(ax, 40, 150, 840, 510,
              os.path.join(PAPER, "2605.15195", "architecture.png"),
              "基座：VGGT-Ω（1B 级）｜ 原图：VGGT-Ω architecture", C_VGGT)
    _img_card(ax, 1040, 150, 840, 510,
              os.path.join(PAPER, "2608.01186", "Overview.png"),
              "新增：双查询解码器 ｜ 原图：QuerySplat overview", C_BLUE)
    _arrow_between(ax, 895, 1035, 400, "冻结，只出特征", C_GREEN)
    y = _change_rows(ax, [
        ("冻结", "VGM 全部 requires_grad=False + eval()：梯度不回流，深度/相机头只用于定义坐标系与早期 CD 正则"),
        ("新增", "几何查询解码器：8192 个可学习查询 + 12×DecoderBlock，cross-attn 到第 4/11/17/23 层混合特征"),
        ("新增", "  → 每个查询展开 64 颗高斯的中心/旋转/尺度（z+1.0、scale 上限 0.075）→ 共 52.4 万颗"),
        ("新增", "外观查询解码器：6×DecoderBlock，以几何 token 为基态，只读 RGB+Plücker 局部内存 → SH 颜色 + α"),
        ("弃用", "VGGT 的点图/深度/相机多任务头不再参与推理输出 —— 推理只走高斯解码器一路"),
        ("可替换", "消融表 4：VGM 换成 VGGT 或 VGGT-Ω 结果打平（23.07 vs 22.90）→ 增益来自「怎么消费几何先验」而非具体 VGM"),
    ], y0=712)
    _t(ax, 1880, 1046, "原图出处：media_paper/images ｜ 输入 pose-free 2~300 视角，输出非像素对齐 3DGS",
       fontsize=12, color="#A0A0A0", ha="right", va="center")
    out = os.path.join(OUT, "fig46_querysplat_vs_vggt.png")
    fig.savefig(out, dpi=100, facecolor="#FBFBFA")
    plt.close(fig)
    print("OK %s（%.2f MB）" % (out, os.path.getsize(out) / 1e6))


# ── fig47：InfiniSplat v2 ← DA3 ────────────────────────────────────────────
def fig47():
    fig, ax = _canvas(
        "InfiniSplat v2：在 DA3 基础上改了什么",
        "同一几何基座 DA3NESTED-GIANT-LARGE-1.1（commit 钉死）—— 差别全在「高斯属性怎么学」（arXiv:2610.20341）",
        C_PURPLE)
    _img_card(ax, 40, 150, 840, 510,
              os.path.join(PAPER, "2511.10647", "pipeline.png"),
              "基座：DA3 ｜ 原图：DA3 论文 pipeline", C_MAIN)
    _img_card(ax, 1040, 150, 840, 510,
              os.path.join(PAPER, "2610.20341", "pipelines.png"),
              "v2：双分支 + Refiner ｜ 原图：InfiniSplat v2 pipelines", C_PURPLE)
    _arrow_between(ax, 895, 1035, 400, "几何冻结，只学属性", C_GREEN)
    y = _change_rows(ax, [
        ("拆分", "DA3 backbone 拆两支：几何分支（Φ_geo）整支冻结；属性分支（Φ_attr）同权重独立初始化、只微调最后 10 个 block"),
        ("锁死", "高斯位置零自由度：GaussianHead 仅 11 通道（rot4+scale3+α1+SH0×3），无位置通道，中心 100% 来自深度反投影"),
        ("锁死", "  → 消融 w/ O（加 3D offset）全面变差 —— offset 会重开「移点补偿属性误差」的退化旁路"),
        ("改输出", "属性全学：scale clamp 0.3、颜色以输入 RGB 为基底预测残差 → 体素合并 + opacity<0.01 剪枝（命中 51.8% 坏点）"),
        ("新增", "Refiner：把渲染梯度当特征喂 4×KNN Transformer block，训练展开 1~3 步、推理 3 步，中心与点数固定"),
        ("训练", "两条训练机关：scale-aligned GT 相机反投影（RDNorm + Umeyama 只取尺度）+ 大角度目标视角监督（0.7/0.3 分池）"),
    ], y0=712)
    _t(ax, 40, y + 16, "血统：v1 = 单图 DepthPro + 隐式解码 + 有界位置更新 → v2 整个放弃隐式解码、位置从「有界更新」变「完全锁死」、几何源换成 DA3",
       fontsize=13, color="#808080", va="center")
    _t(ax, 1880, 1046, "原图出处：media_paper/images ｜ pose-conditioned / pose-free 双模式同一权重",
       fontsize=12, color="#A0A0A0", ha="right", va="center")
    out = os.path.join(OUT, "fig47_infinisplat_v2_vs_da3.png")
    fig.savefig(out, dpi=100, facecolor="#FBFBFA")
    plt.close(fig)
    print("OK %s（%.2f MB）" % (out, os.path.getsize(out) / 1e6))


# ── fig48：DA3 ↔ VGGT 架构对比（纯绘制）────────────────────────────────────
def _col_schematic(ax, x, w, color, name, sub, boxes, foots):
    ax.add_patch(FancyBboxPatch((x, 150), w, 620, boxstyle="round,pad=0.012",
                                fc="white", ec="#D8D8D5", lw=1.2, zorder=1))
    ax.add_patch(Rectangle((x + 1, 150), w - 2, 52, fc=color, ec="none",
                           zorder=2, alpha=0.92))
    _t(ax, x + 20, 176, name, fontsize=19, fontweight="bold", color="white",
       va="center", zorder=4)
    _t(ax, x + w - 20, 176, sub, fontsize=12.5, color="white", ha="right",
       va="center", zorder=4)
    by = 226
    for title, lines, h in boxes:
        ax.add_patch(FancyBboxPatch((x + 24, by), w - 48, h,
                                    boxstyle="round,pad=0.02", fc="#F7F7F4",
                                    ec=color, lw=1.4, zorder=2))
        _t(ax, x + 44, by + 24, title, fontsize=16, fontweight="bold",
           color=color, va="center", zorder=4)
        ty = by + 52
        for ln in lines:
            _t(ax, x + 44, ty, ln, fontsize=13.5, color="#383838", va="center",
               zorder=4)
            ty += 25
        by += h + 22
    fy = 150 + 620 - 24
    for ln in reversed(foots):
        _t(ax, x + 24, fy, ln, fontsize=12.5, color=color, va="center", zorder=4)
        fy -= 26


def fig48():
    fig, ax = _canvas(
        "DA3 vs VGGT：两代 pose-free 前馈几何模型的架构分歧",
        "同一目标（无位姿多图 → 相机+稠密几何）、两种路线：定制双注意力多任务 vs 预训练权重+最小几何目标",
        C_MAIN)
    _col_schematic(ax, 40, 900, C_VGGT, "VGGT", "arXiv:2503.11651 · CVPR 2025 Best Paper",
        boxes=[
            ("输入与接地", ["N 张无位姿 RGB（1~数百张）",
                       "Plücker 射线坐标 (d, o×d) 注入 token 位置嵌入"], 78),
            ("图像编码器 + 交替注意力 ×24 层", ["DINOv2 ViT-L 负责 patchify（冻结），后接 24 层 transformer",
                             "帧内自注意力 frame-wise 与全局自注意力 global 交替（专门注意力算子）",
                             "可学习 camera token 拼在 token 序列里一起进注意力"], 104),
            ("多任务头（过完整预测）", ["Camera Head → R(6D)/t/f",
                        "DPT Head → 深度图 + 点图 + 特征图（点追踪）",
                        "深度与点图各自独立预测、再互相补充"], 114),
        ],
        foots=["表示 = point map：深度与相机耦合在同一张量",
               "2/3 的 block 未经预训练；多任务联合损失，权重靠经验平衡"])
    _col_schematic(ax, 980, 900, C_MAIN, "DA3", "arXiv:2511.10647 · 笔记 3D-20251113",
        boxes=[
            ("输入与 token 调度", ["N 张无位姿 RGB（Nv∈[1,∞)，位姿可选注入）",
                       "camera token（已知位姿经 MLP 编码 9 维 t/q/f）+ 参考视图重排 + RoPE 双坐标"], 78),
            ("原版 DINOv2 block，零修改 ×N 层", ["local（视图内）与 global（全部视图拼接）交替 ——",
                             "完全靠 reshape 表达，「没有专门注意力算子」",
                             "cat_token 双流拼接（head 输入通道 ×2）"], 104),
            ("Dual-DPT 双分支头", ["depth 分支：exp 激活回归深度 + 置信度",
                      "ray 分支：6 维光线（原点 t + 方向 d=RK⁻¹p）+ 置信度",
                      "相机 = ray map 解析解（置信度加权 DLT+RQ），pose 是几何的副产品"], 114),
        ],
        foots=["表示 = depth-ray 单目标：深度管尺度、光线管相机（消融表 6 全胜）",
               "全量 DINOv2 预训练权重 + teacher（DA2 系）蒸馏多视图学生"])
    # 中间 vs
    _t(ax, 960, 430, "VS", fontsize=30, fontweight="bold", color="#B0B0AA",
       ha="center", va="center")
    # 底部分歧表
    rows = [
        ("注意力实现", "定制 frame-wise/global 交替算子 + Plücker 接地",
         "reshape 级 local/global 交替，block 零修改（消融：VGGT 式双 transformer 仅 79.8%）"),
        ("预训练利用", "仅图像编码器用 DINOv2，2/3 block 未预训练",
         "全量 DINOv2 权重接入 —— 「强预训练面前架构特化是负资产」"),
        ("几何表示", "point map（深度与相机耦合在一个张量）",
         "depth-ray 正交分解（深度管相对尺度，光线管相机）"),
        ("相机来源", "camera token 经 Camera Head 回归 R/t/f",
         "ray map 解析恢复（DLT+RQ），另有 camera head 作备选支路"),
        ("训练范式", "多任务联合（pose+depth+point+track）",
         "teacher-student：单目 teacher 伪标签教多视图学生"),
    ]
    TY, RH = 796, 40
    _t(ax, 40, TY - 26, "关键分歧一览：", fontsize=16, fontweight="bold", color="#404040")
    ax.add_patch(Rectangle((40, TY), 280, RH, fc="#6B6B66", ec="none", zorder=2))
    ax.add_patch(Rectangle((320, TY), 790, RH, fc=C_VGGT, ec="none", zorder=2))
    ax.add_patch(Rectangle((1110, TY), 770, RH, fc=C_MAIN, ec="none", zorder=2))
    _t(ax, 60, TY + RH / 2, "维度", fontsize=15, fontweight="bold", color="white", va="center", zorder=4)
    _t(ax, 715, TY + RH / 2, "VGGT", fontsize=15, fontweight="bold", color="white", ha="center", va="center", zorder=4)
    _t(ax, 1495, TY + RH / 2, "DA3", fontsize=15, fontweight="bold", color="white", ha="center", va="center", zorder=4)
    ry = TY + RH
    for i, (dim, vg, da) in enumerate(rows):
        bgc = "white" if i % 2 == 0 else "#F3F3EF"
        ax.add_patch(Rectangle((40, ry), 280, RH, fc=bgc, ec="#DDDDD8", lw=0.6, zorder=1))
        ax.add_patch(Rectangle((320, ry), 790, RH, fc=bgc, ec="#DDDDD8", lw=0.6, zorder=1))
        ax.add_patch(Rectangle((1110, ry), 770, RH, fc=bgc, ec="#DDDDD8", lw=0.6, zorder=1))
        _t(ax, 60, ry + RH / 2, dim, fontsize=14, fontweight="bold", color="#404040", va="center")
        _t(ax, 336, ry + RH / 2, vg, fontsize=12.5, color="#333333", va="center")
        _t(ax, 1126, ry + RH / 2, da, fontsize=12.5, color="#333333", va="center")
        ry += RH
    _t(ax, 1880, 1062, "DA3 笔记自述 VGGT 为「最直接前身」：统一几何目标下，表示/架构/预训练利用三处收敛",
       fontsize=12, color="#A0A0A0", ha="right", va="center")
    out = os.path.join(OUT, "fig48_da3_vs_vggt.png")
    fig.savefig(out, dpi=100, facecolor="#FBFBFA")
    plt.close(fig)
    print("OK %s（%.2f MB）" % (out, os.path.getsize(out) / 1e6))


def main():
    os.makedirs(OUT, exist_ok=True)
    for img in ("2605.15195/architecture.png", "2608.01186/Overview.png",
                "2511.10647/pipeline.png", "2610.20341/pipelines.png"):
        p = os.path.join(PAPER, img)
        if not os.path.exists(p):
            raise SystemExit("missing paper image: %s" % p)
    fig46()
    fig47()
    fig48()


if __name__ == "__main__":
    main()
