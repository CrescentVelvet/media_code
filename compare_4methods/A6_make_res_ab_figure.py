#!/usr/bin/env python3
"""R-3 出图：训练分辨率 RES=4/2/1（360×480 / 720×960 / 1439×1919），同 init 同 2k 步。

核心现象是"对角线"：每一档训练的 PSNR 峰值都出现在与其训练分辨率相近的评测尺寸上。
  fig16  3×3 矩阵（行=训练分辨率，列=评测分辨率）—— 对角线规律 + 错配代价 + 成本表
  fig20  378 口径下三种训练的人脸放大对照 —— 对角线最优档到底长什么样
  fig17  三档评测分辨率下的人脸放大（含 GT）—— 看"脸糊"是否随分辨率改善
  fig18  1439 原生口径的全身对照
  fig19  误差定位图（由 A8 生成）

数据：~/output/da3_gs_results/佳佳抱枕_res_ab/
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

BASE = Path("/home/velvet/output/da3_gs_results/佳佳抱枕_res_ab")
REF = Path("/home/velvet/output/da3_results/_captures/佳佳抱枕/images")

SIZES = ["378x504", "756x1008", "1439x1919"]
# 从上到下：训练分辨率由低到高（行），与评测分辨率由低到高（列）构成对角线
TRAINS = [
    ("res4", "RES=4\n360×480", "#5B7FB5"),
    ("res2", "RES=2\n720×960", "#C77C2E"),
    ("res1", "RES=1\n1439×1919", "#2E8B7F"),
]
FACE_BOX = (0.35, 0.105, 0.565, 0.335)
MAIN_SIZE = "1439x1919"
FACE_VIEWS = (0, 8, 16, 24)


def load(tag, size):
    p = BASE / f"final_{tag}" / f"metrics_eval{size}.json"
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def train_info(tag):
    p = BASE / f"final_{tag}" / "short_train_summary.json"
    d = json.load(open(p, encoding="utf-8")) if p.exists() else {}
    mem = BASE / f"mem_{tag}.csv"
    peak = None
    if mem.exists():
        vals = [int(x) for x in mem.read_text().split() if x.strip().isdigit()]
        peak = max(vals) if vals else None
    return d.get("train_seconds"), peak


def face_psnr(tag, size, view):
    from PIL import Image

    refs = sorted(p for p in REF.iterdir() if p.suffix.lower() == ".png")
    rp = BASE / f"final_{tag}" / f"render_{size}" / f"view_{view:03d}.png"
    if not rp.exists():
        return None
    gt = Image.open(refs[view]).convert("RGB")
    rd = Image.open(rp).convert("RGB")
    if rd.size != gt.size:
        rd = rd.resize(gt.size, Image.LANCZOS)
    w, h = gt.size
    box = (int(FACE_BOX[0] * w), int(FACE_BOX[1] * h), int(FACE_BOX[2] * w), int(FACE_BOX[3] * h))
    g = np.asarray(gt.crop(box), dtype=np.float64)
    r = np.asarray(rd.crop(box), dtype=np.float64)
    mse = float(np.mean((g - r) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def fig16():
    """3×3 矩阵热力图：对角线是"训练分辨率 ≈ 评测分辨率"，一格看完整规律。"""
    M = np.full((len(TRAINS), len(SIZES)), np.nan)
    S = np.full_like(M, np.nan)
    for i, (tag, _, _) in enumerate(TRAINS):
        for j, sz in enumerate(SIZES):
            d = load(tag, sz)
            if d:
                M[i, j] = d["mean"]["psnr"]
                S[i, j] = d["mean"]["ssim"]

    fig = plt.figure(figsize=(13.6, 6.2), dpi=170)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.28, 1.0], wspace=0.26)
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(M, cmap="YlGnBu", aspect="auto", vmin=np.nanmin(M) - 1, vmax=np.nanmax(M))

    lo, hi = np.nanmin(M), np.nanmax(M)
    for i in range(len(TRAINS)):
        for j in range(len(SIZES)):
            if np.isnan(M[i, j]):
                continue
            diag = i == j
            dark = M[i, j] > lo + 0.62 * (hi - lo)
            main_c = "white" if dark else "#243B4A"
            sub_c = "#CFE0EC" if dark else "#5A5A5A"
            ax.text(j, i - 0.15, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=17 if diag else 14.5,
                    fontweight="bold" if diag else "normal", color=main_c)
            ax.text(j, i + 0.26, f"SSIM {S[i, j]:.3f}", ha="center", va="center",
                    fontsize=9, color=sub_c)
            if diag:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="#D64545", lw=2.6, zorder=5))

    ax.set_xticks(range(len(SIZES)))
    ax.set_xticklabels([s.replace("x", "×") for s in SIZES], fontsize=11.5)
    ax.set_yticks(range(len(TRAINS)))
    ax.set_yticklabels([t[1] for t in TRAINS], fontsize=10.5, linespacing=1.5)
    ax.set_xlabel("渲染 + 评测分辨率  →", fontsize=12)
    ax.set_ylabel("←  训练分辨率（3DGS 的 RES）", fontsize=12)
    ax.set_title("3×3 矩阵：PSNR 峰值全部落在「训练 ≈ 评测」的对角线上",
                 fontsize=13.5, loc="left", pad=12)
    cb = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.02)
    cb.set_label("PSNR (dB)", fontsize=10)

    # 右：成本表
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off")
    rows = []
    for i, (tag, lab, _) in enumerate(TRAINS):
        sec, peak = train_info(tag)
        px = {4: "360×480", 2: "720×960", 1: "1439×1919"}[int(tag[-1])]
        npx = {"360×480": 360 * 480, "720×960": 720 * 960, "1439×1919": 1439 * 1919}[px]
        dens = 4_838_397 / npx
        rows.append([px, f"{sec} s" if sec else "—",
                     f"{peak:,} MiB" if peak else "—", f"{dens:.1f}"])

    table = ax2.table(
        cellText=rows,
        colLabels=["训练分辨率", "训练耗时", "峰值显存", "点/像素"],
        cellLoc="center", loc="upper center",
        colWidths=[0.29, 0.24, 0.24, 0.20],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11.5)
    table.scale(1, 2.05)
    for j in range(4):
        table[0, j].set_facecolor("#E8EEF3")
        table[0, j].set_text_props(fontweight="bold")
    table[(1, 0)].set_facecolor("#DCE9F5")
    table[(1, 3)].set_facecolor("#FBE3E3")

    ax2.text(0.5, 0.655,
             "点/像素 = 415 万高斯 ÷ 训练像素数\n"
             "→ 它决定训练出的高斯尺度有多大：\n\n"
             "    24 点/px（RES=4）高斯小，细节足\n"
             "      6 点/px（RES=2）居中\n"
             "    1.5 点/px（RES=1）高斯被摊大 → 软\n\n"
             "所以最优档不是「分辨率越高越好」，\n"
             "而是「训练分辨率贴近交付分辨率」。",
             transform=ax2.transAxes, ha="center", va="top", fontsize=11,
             linespacing=1.75, color="#2B2B2B",
             bbox=dict(fc="#F7F7F5", ec="#DDDDD8", pad=14))
    ax2.text(0.5, 0.025,
             "自一致性口径（渲染 vs 该视角输入帧），非 NVS holdout；2k 步、关 densify、同一 init。",
             transform=ax2.transAxes, ha="center", fontsize=9, color="#888888")

    p = OUT / "fig16_res_ab_matrix.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("✅", p)


def fig20(view=0, cell=290):
    """378 口径下三种训练的人脸放大——对角线最优档到底长什么样。"""
    from PIL import Image, ImageDraw, ImageFont

    refs = sorted(p for p in REF.iterdir() if p.suffix.lower() == ".png")
    cols = [("GT（1439×1919 原图）", None)]
    cols += [(f"{lab.replace(chr(10), ' · ')}", tag) for tag, lab, _ in TRAINS]

    pad, top, lab_h = 8, 66, 24
    canvas = Image.new("RGB", (len(cols) * (cell + pad) + pad, top + cell + lab_h), "white")
    dr = ImageDraw.Draw(canvas)
    f_t = ImageFont.truetype(_MSYH, 16)
    f_s = ImageFont.truetype(_MSYH, 13)

    psnr = {tag: load(tag, "378x504")["mean"]["psnr"] for tag, _, _ in TRAINS}
    for j, (lab, tag) in enumerate(cols):
        x = pad + j * (cell + pad)
        dr.text((x + 3, 14), lab, fill=(30, 30, 30), font=f_t)
        if tag:
            dr.text((x + 3, 38), f"378×504 口径  {psnr[tag]:.2f} dB", fill=(150, 90, 40), font=f_s)
        im = Image.open(refs[view]).convert("RGB") if tag is None else \
            Image.open(BASE / f"final_{tag}" / "render_378x504" / f"view_{view:03d}.png").convert("RGB")
        w, h = im.size
        box = (int(FACE_BOX[0] * w), int(FACE_BOX[1] * h), int(FACE_BOX[2] * w), int(FACE_BOX[3] * h))
        canvas.paste(im.crop(box).resize((cell, cell), Image.LANCZOS), (x, top))
    dr.text((pad + 3, top + cell + 2),
            "view 0 ｜ 全部按 378×504 渲染后裁人脸放大 ｜ 训练分辨率越低反而越锐，因为高斯尺度匹配了像素密度",
            fill=(120, 120, 120), font=f_s)

    p = OUT / "fig20_res_face_378.png"
    canvas.save(p)
    print("✅", p)


def fig17(view=0, cell=(280, 280)):
    """人脸区放大对照：行=评测分辨率，列=GT / RES=4 / RES=2 / RES=1。"""
    from PIL import Image, ImageDraw, ImageFont

    refs = sorted(p for p in REF.iterdir() if p.suffix.lower() == ".png")
    gt = Image.open(refs[view]).convert("RGB")
    gw, gh = gt.size
    gbox = (int(FACE_BOX[0] * gw), int(FACE_BOX[1] * gh),
            int(FACE_BOX[2] * gw), int(FACE_BOX[3] * gh))

    cw, ch = cell
    pad, top, lab = 8, 74, 26
    cols = ["GT（1439×1919 原图）"] + [f"{t[1].replace(chr(10), ' ')}" for t in TRAINS]
    canvas = Image.new("RGB", (len(cols) * (cw + pad) + pad,
                               top + len(SIZES) * (ch + pad + lab) + pad), "white")
    dr = ImageDraw.Draw(canvas)
    f_t = ImageFont.truetype(_MSYH, 16)
    f_s = ImageFont.truetype(_MSYH, 13)

    for j, t in enumerate(cols):
        dr.text((pad + j * (cw + pad) + 3, 18), t, fill=(30, 30, 30), font=f_t)
    dr.text((pad + 3, 44), "同一视角、同一相对人脸框；渲染与评测在同一分辨率下进行",
            fill=(120, 120, 120), font=f_s)

    for i, sz in enumerate(SIZES):
        y = top + i * (ch + pad + lab)
        dr.text((pad + 3, y - 20), f"渲染/评测 {sz.replace('x', '×')}", fill=(110, 110, 110), font=f_s)
        canvas.paste(gt.crop(gbox).resize((cw, ch), Image.LANCZOS), (pad, y))
        for j, (tag, _, _) in enumerate(TRAINS):
            rp = BASE / f"final_{tag}" / f"render_{sz}" / f"view_{view:03d}.png"
            if not rp.exists():
                dr.text((pad + (j + 1) * (cw + pad) + 6, y + ch // 2), "缺图",
                        fill=(200, 60, 60), font=f_t)
                continue
            im = Image.open(rp).convert("RGB")
            if im.size != gt.size:
                im = im.resize(gt.size, Image.LANCZOS)
            canvas.paste(im.crop(gbox).resize((cw, ch), Image.LANCZOS),
                         (pad + (j + 1) * (cw + pad), y))

    p = OUT / "fig17_res_face_zoom.png"
    canvas.save(p)
    print("✅", p)


def fig18(view=0):
    """1439×1919 原生口径的全身对照。"""
    from PIL import Image, ImageDraw, ImageFont

    refs = sorted(p for p in REF.iterdir() if p.suffix.lower() == ".png")
    cols = [("GT", lambda: Image.open(refs[view]).convert("RGB"))]
    for tag, lab, _ in TRAINS:
        rp = BASE / f"final_{tag}" / f"render_{MAIN_SIZE}" / f"view_{view:03d}.png"
        cols.append((lab.replace("\n", " "), (lambda p=rp: Image.open(p).convert("RGB"))))

    W, H, pad, top = 272, 363, 8, 58
    canvas = Image.new("RGB", (len(cols) * (W + pad) + pad, top + H + pad), "white")
    dr = ImageDraw.Draw(canvas)
    f_t = ImageFont.truetype(_MSYH, 15)
    for j, (t, get) in enumerate(cols):
        x = pad + j * (W + pad)
        dr.text((x + 3, 14), t, fill=(30, 30, 30), font=f_t)
        canvas.paste(get().resize((W, H), Image.LANCZOS), (x, top))
    p = OUT / "fig18_res_full_1439.png"
    canvas.save(p)
    print("✅", p)


if __name__ == "__main__":
    fig16()
    fig20()
    fig17()
    fig18()
