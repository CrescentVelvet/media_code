#!/usr/bin/env python3
"""fig41~45：五方法算法框架详解图（PPT 用，1920×1080）。

版式（每张）：
  顶部   方法名 + 一句话定位 + 徽章行（耗时 / 高斯数 / SH 阶数 / 输入视角）
  中部   **论文原始框架图**（media_paper/images/ 已核图，白底卡片等比嵌入）
  右侧   两卡对照：「与其它四家相同」/「这家独有」
  底部   五行泳道迷你条（fig38 同款五格）高亮自己 → 一眼看到在谱系里的位置

口径（严谨，全部可回查）：
  耗时/高斯数/SH = FINDINGS §6 五方法表（32 视角实测，佳佳抱枕）；
  算法细节 = media_paper 笔记 3D-20251113(DA3)/3D-20261105(QuerySplat)/
  3D-20261014(InfiniSplat v2)/3D-20260514(VGGT-Ω)/3D-20230808(3DGS)。
  字体坑：微软雅黑无 ✓✗ 字形 → 一律用 √×。

用法（Windows 本机，WorkBuddy default python 带 matplotlib/PIL）：
  python compare_4methods/A20_make_framework_figures.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib import font_manager

ROOT = os.path.dirname(os.path.abspath(__file__))
if os.path.exists("/mnt/c/code/media_paper"):      # WSL 跑法（A18/A19 同款）
    PAPER = "/mnt/c/code/media_paper/images"
    FONT_R = "/mnt/c/Windows/Fonts/msyh.ttc"
    FONT_B = "/mnt/c/Windows/Fonts/msyhbd.ttc"
else:                                               # Windows 本机跑法
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

# 与 fig35/38 一致的方法配色
C_MAIN = "#0F6E6E"   # DA3-GS 主角青
C_BLUE = "#5B7FB5"   # QuerySplat
C_PURPLE = "#8B6FA8" # InfiniSplat
C_ORANGE = "#A8834F" # 3DGS 精化系
C_GRAY = "#8A8A8A"

# 五家泳道（底部迷你条用，顺序固定）
LANES = [
    ("DA3-GS", C_MAIN), ("QuerySplat", C_BLUE), ("InfiniSplat v2", C_PURPLE),
    ("VGGT-Ω+3DGS", C_ORANGE), ("DA3+3DGS", C_ORANGE),
]

# ── 每张图的内容 ────────────────────────────────────────────────────────────
# img: 论文原图；pos: 定位句；badges: 徽章行；same/diff: 异同点（每条 ≤ 22 字为佳）
CARDS = [
    dict(
        file="fig41_da3gs_framework.png", lane=0,
        name="DA3-GS（Depth Anything 3 前馈高斯头）", color=C_MAIN,
        src="原图：DA3 论文 pipeline ｜ arXiv:2511.10647",
        img=os.path.join(PAPER, "2511.10647", "pipeline.png"),
        pos="一次前向，逐像素「点云即高斯」——五家里唯一不挑场景 Sculpt 的方法",
        badges=["13.5 秒", "610 万高斯", "SH 2 阶", "3~32 图均可"],
        same=[
            "多张无位姿照片输入，相机靠网络自己估",
            "DINOv2 系 ViT 提特征（四家同源）",
            "输出带相机的 3D 高斯场，可直接渲染",
        ],
        diff=[
            "逐像素一一对应：每个像素点直接",
            "  变一颗高斯，不剪枝不融合",
            "高斯数 ∝ 像素数×视角数（610 万）",
            "显影机制把跨视角不一致「摊平成糊」",
            "  （而非 InfiniSplat 式的错位重影）",
        ],
        note=" ours 复现口径：全 SH 导出不裁剪，全图 22.11 dB（DA3 官方管线口径，32 视角）",
    ),
    dict(
        file="fig42_querysplat_framework.png", lane=1,
        name="QuerySplat（查询槽式高斯生成）", color=C_BLUE,
        src="原图：QuerySplat 论文 overview ｜ arXiv:2608.01186",
        img=os.path.join(PAPER, "2608.01186", "Overview.png"),
        pos="「先定放哪，再填属性」——8192 个查询槽竞标高斯摆放位置",
        badges=["38.5 秒", "19~37 万高斯", "SH 1 阶", "视角数变化最钝感"],
        same=[
            "VGGT 系几何先验（与 VGGT-Ω 同门）",
            "查询-注意力解码器出高斯（Transformer）",
            "Plücker 射线编码注入视角信息",
        ],
        diff=[
            "稀疏固定预算：8192 查询×64 token，",
            "  场景再复杂也只放约 19 万颗",
            "「先几何槽位、后外观颜色」两段式",
            "三图实测仅掉 3.0 dB（五家最抗缩减）",
            "  ——代价是绝对上限低、画面偏糊",
        ],
        note=" ours 画幅修正实验 E18：等比缩焦距后 16.55 dB（自家渲染判据，勿与 22.11 并比）",
    ),
    dict(
        file="fig43_infinisplat_framework.png", lane=2,
        name="InfiniSplat v2（DA3 血统 + 轻量回炉）", color=C_PURPLE,
        src="原图：InfiniSplat v2 论文 pipelines ｜ arXiv:2610.20341",
        img=os.path.join(PAPER, "2610.20341", "pipelines.png"),
        pos="内核就是 DA3——但多了 163 秒「回炉精修」和三大训练机关",
        badges=["前馈 34 秒", "+精化 163 秒", "594 万高斯", "SH 0 阶"],
        same=[
            "DA3 前馈内核：多视角 backbone",
            "  出高斯/深度/相机三张图",
            "同样一次前向出可渲染高斯场",
        ],
        diff=[
            "唯一带「回炉」：163 秒精化步",
            "  （opacity 剪枝 + 大角度视角监督）",
            "双分支训练：可训分支+冻结分支同架构",
            "尺度对齐 GT 位姿，深度图重标定",
            "opacity 原值显影 → 层间小错位呈重影",
            "  （DA3-GS 同分层则摊平成糊，E19）",
        ],
        note=" ours 复现口径：jiajia 全分辨率输入；§8 同口径（GT 相机渲自家产物）全图 19.81 dB",
    ),
    dict(
        file="fig44_vggt_omega_framework.png", lane=3,
        name="VGGT-Ω + 3DGS（最大几何先验 + 回炉雕塑）", color=C_ORANGE,
        src="原图：VGGT-Ω 论文 architecture ｜ arXiv:2605.15195",
        img=os.path.join(PAPER, "2605.15195", "architecture.png"),
        pos="先馈送最强几何（CVPR 2026 Oral），再交 3DGS 回炉 30k 步精雕",
        badges=["前馈约 2 分", "+3DGS 20.4 分", "175 万高斯", "SH 3 阶"],
        same=[
            "与 DA3+3DGS 共用同一条「回炉」流水线",
            "  （3DGS 30k 步，见下一张）",
            "DINO 特征 + 相机 token 估位姿",
            "点云 init → 逐场景优化雕塑",
        ],
        diff=[
            "register 注意力：帧间信息只走寄存器",
            "  （替全局注意力，训练省 70% 显存）",
            "单一 dense head 多任务出相机/深度/点",
            "五家里参数量最大的几何先验",
            "视角缺失时几何最稳（dynamic 场景也训）",
        ],
        note=" ours 复现口径：VGGT-Ω 点云 init 3DGS 30k 步；§8 同口径全图 11.29 dB",
    ),
    dict(
        file="fig45_3dgs_framework.png", lane=4,
        name="DA3 + 3DGS（前馈点云 + 经典回炉雕塑）", color=C_ORANGE,
        src="原图：3DGS 论文 overview ｜ arXiv:2308.04079 (SIGGRAPH 2023)",
        img=os.path.join(PAPER, "2308.04079", "overview.png"),
        pos="回炉段的祖师爷：3DGS 本身——前馈点云进，30k 步雕塑出精品",
        badges=["前馈 13.5 秒", "+3DGS 22.9 分", "189 万高斯", "SH 3 阶"],
        same=[
            "与 VGGT-Ω+3DGS 共用同一条回炉段",
            "Splatting 渲染引擎五家同宗（EWA）",
            "梯度：光栅化回传 → 打磨高斯参数",
        ],
        diff=[
            "起点从「SfM 点」换成「DA3 前馈点云」",
            "  ——省掉 COLMAP，但位姿也改用 DA3 的",
            "自适应密度控制：clone/split/prune",
            "  欠观测空间会填成半透雾（三图实证）",
            "30k 步不随视角数缩短 → 三图也要 53 分",
            "32 视角全图 11.71 dB（§8 GT 相机口径）",
        ],
        note=" ours 复现口径：DA3 点云+相机 init 3DGS 30k 步；init 增益 +3.1 dB（§2）",
    ),
]


def _t(ax, *a, **kw):
    """统一走微软雅黑（WSL 里 font.family 名解析偶发失败，显式传 family 最稳）。"""
    kw.setdefault("fontfamily", YAHEI or "sans-serif")
    return ax.text(*a, **kw)


def draw_card(spec):
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1920)
    ax.set_ylim(1080, 0)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 1920, 1080, fc="#FBFBFA", ec="none", zorder=0))

    # ── 顶部标题区 ──
    ax.add_patch(Rectangle((0, 0), 1920, 132, fc="white", ec="#E2E2E0", zorder=1))
    ax.add_patch(Rectangle((0, 0), 14, 132, fc=spec["color"], ec="none", zorder=2))
    _t(ax, 40, 30, spec["name"], fontsize=30, fontweight="bold",
            color=spec["color"], va="center", fontfamily="Microsoft YaHei")
    _t(ax, 40, 78, spec["pos"], fontsize=17, color="#404040", va="center",
            fontfamily="Microsoft YaHei")
    # 徽章行（右对齐，从右往左排）
    bx = 1880
    for b in reversed(spec["badges"]):
        t = ax.text(bx, 30, b, fontsize=14, color="white", ha="right", va="center",
                    fontweight="bold", fontfamily="Microsoft YaHei",
                    bbox=dict(boxstyle="round,pad=0.4", fc=spec["color"], ec="none", alpha=0.9))
        bx -= (len(b) * 15 + 46)
    _t(ax, 1880, 82, spec["src"], fontsize=12.5, color="#909090", ha="right",
            va="center", fontfamily="Microsoft YaHei")

    # ── 中部：论文原图卡片（fit 到 1380×660）──
    IX, IY, IW, IH = 40, 190, 1380, 660
    ax.add_patch(FancyBboxPatch((IX, IY), IW, IH, boxstyle="round,pad=0.012",
                                fc="white", ec="#D8D8D5", lw=1.2, zorder=1))
    img = mpimg.imread(spec["img"])
    h, w = img.shape[:2]
    scale = min((IW - 40) / w, (IH - 60) / h)
    dw, dh = w * scale, h * scale
    ax.imshow(img, extent=[IX + (IW - dw) / 2, IX + (IW + dw) / 2,
                           IY + (IH + dh) / 2, IY + (IH - dh) / 2], zorder=3)

    # ── 右侧：相同 / 不同 两卡 ──
    RX, RW = 1450, 430
    def side_card(y, h, title, lines, tcolor):
        ax.add_patch(FancyBboxPatch((RX, y), RW, h, boxstyle="round,pad=0.012",
                                    fc="white", ec="#D8D8D5", lw=1.2, zorder=1))
        ax.add_patch(Rectangle((RX + 14, y + 14), 6, h - 28, fc=tcolor, ec="none", zorder=2))
        _t(ax, RX + 34, y + 32, title, fontsize=17, fontweight="bold",
                color=tcolor, va="center", fontfamily="Microsoft YaHei")
        ty = y + 66
        for ln in lines:
            if ln.startswith(" "):
                _t(ax, RX + 52, ty, ln.strip(), fontsize=12.5, color="#505050",
                        va="center", fontfamily="Microsoft YaHei")
            else:
                _t(ax, RX + 36, ty, "√ " + ln, fontsize=12.5, color="#303030",
                        va="center", fontfamily="Microsoft YaHei")
            ty += 27
    same_h = 66 + 27 * max(len(spec["same"]), 3) + 14
    diff_h = 66 + 27 * len(spec["diff"]) + 14
    side_card(190, same_h, "与其它四家相同", spec["same"], "#4E7A4E")
    side_card(190 + same_h + 18, diff_h, "这家独有", spec["diff"], spec["color"])

    # ── 底部：迷你泳道（五家定位）+ 复现口径注 ──
    _t(ax, 40, 902, "在五方法谱系中的位置：", fontsize=15, fontweight="bold",
            color="#404040", va="center", fontfamily="Microsoft YaHei")
    lx, lw_, gap = 250, 296, 14
    for i, (nm, col) in enumerate(LANES):
        hl = (i == spec["lane"])
        ax.add_patch(FancyBboxPatch((lx + i * (lw_ + gap), 880), lw_, 44,
                                    boxstyle="round,pad=0.02",
                                    fc=col if hl else "#EDEDEA",
                                    ec=col if hl else "#CFCFCB",
                                    lw=2 if hl else 1, zorder=2,
                                    alpha=1.0 if hl else 0.6))
        _t(ax, lx + i * (lw_ + gap) + lw_ / 2, 902, nm, fontsize=14,
                fontweight="bold" if hl else "normal",
                color="white" if hl else "#707070", ha="center", va="center",
                fontfamily="Microsoft YaHei")
    _t(ax, 40, 962, spec["note"], fontsize=13.5, color="#606060", va="center",
            fontfamily="Microsoft YaHei")
    _t(ax, 1880, 962, "佳佳抱枕 32 视角对比 ｜ ours = 本仓复现与实测",
            fontsize=12, color="#A0A0A0", ha="right", va="center",
            fontfamily="Microsoft YaHei")

    out = os.path.join(OUT, spec["file"])
    fig.savefig(out, dpi=100, facecolor="#FBFBFA")
    plt.close(fig)
    print("✅ %s（%.2f MB）" % (out, os.path.getsize(out) / 1e6))


def main():
    os.makedirs(OUT, exist_ok=True)
    for spec in CARDS:
        if not os.path.exists(spec["img"]):
            raise SystemExit("❌ 缺原图 %s" % spec["img"])
        draw_card(spec)


if __name__ == "__main__":
    main()
