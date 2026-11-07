#!/usr/bin/env python3
"""AUDIT 出图：把「重影/模糊」拆成「本视角几何」与「跨视角叠加」

fig21  5 列 × 2 行：GT / 全部高斯 / 仅本视角 / 仅本视角÷α / 去掉本视角
       同一像素集（self 覆盖到的像素）上的 PSNR 直接标在每列下面。
       结论看三件事：
         ① 仅本视角很暗 → 单视角 alpha 叠不满（每像素要 ~25 层）
         ② 仅本视角÷α 也不如其余几列 → 单视角几何本身偏离 GT 4 dB 左右
         ③ 去掉本视角 ≈ 全部 → 成品几乎是"其它 31 个视角的平均"

数据：~/output/da3_gs_results/佳佳抱枕_res_ab/audit_res1/
"""
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

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
AUD = Path("/home/velvet/output/da3_gs_results/佳佳抱枕_res_ab/audit_res1")
REF = Path("/home/velvet/output/da3_results/_captures/佳佳抱枕/images")
VIEWS = [0, 16]
CROP_Y = (40, 478)          # 去掉上下空白

COLS = [
    ("GT（输入帧）", None, None),
    ("全部高斯", "render_all", "psnr_all_masked"),
    ("仅本视角", "render_self", "psnr_self_masked"),
    ("仅本视角 ÷ α", "render_self_n", "psnr_self_n_masked"),
    ("去掉本视角", "render_other", "psnr_other_masked"),
]

met = json.load(open(AUD / "audit_render_decomp.json", encoding="utf-8"))
per = {r["view"]: r for r in met["per_view"]}
refs = sorted(p for p in REF.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))

rows, cols = len(VIEWS), len(COLS)
fig = plt.figure(figsize=(15.2, 8.6), dpi=150)
gs = fig.add_gridspec(rows, cols, left=0.012, right=0.988, top=0.90, bottom=0.20,
                      wspace=0.035, hspace=0.10)

for ri, v in enumerate(VIEWS):
    gt = Image.open(refs[v]).convert("RGB")
    w0, h0 = gt.size
    for ci, (title, sub, key) in enumerate(COLS):
        ax = fig.add_subplot(gs[ri, ci])
        if sub is None:
            im = gt
        else:
            im = Image.open(AUD / sub / f"view_{v:03d}.png").convert("RGB")
        iw, ih = im.size
        box = (0, int(CROP_Y[0] / 504 * ih), iw, int(CROP_Y[1] / 504 * ih))
        im = im.crop(box)
        ax.imshow(np.asarray(im))
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.8)
            s.set_color("#BBBBBB")
        if ri == 0:
            ax.set_title(title, fontsize=13, pad=6)
        if key is not None:
            ax.text(0.5, -0.055, f"{per[v][key]:.2f} dB", transform=ax.transAxes,
                    ha="center", va="top", fontsize=11.5,
                    color="#222222" if key != "psnr_self_masked" else "#B03A2E")
        if ci == 0:
            ax.set_ylabel(f"view {v}", fontsize=11.5)

m = met["mean"]
fig.text(0.5, 0.945,
         "重影的来源：成品 ≈ 「其它视角的平均」，单视角几何各自偏离 GT 约 4 dB（RES=1 训练产物，378×504 口径）",
         ha="center", va="center", fontsize=14.5, weight="bold")
fig.text(0.5, 0.115,
         "同一像素集（仅本视角覆盖到的像素，占全图 43%）上比："
         f"全部 {m['psnr_all_masked']:.2f} dB ｜ 仅本视角 {m['psnr_self_masked']:.2f} dB（暗是因为 α 只有 ~0.07）"
         f" ｜ ÷α 后 {m['psnr_self_n_masked']:.2f} dB ｜ 去掉本视角 {m['psnr_other_masked']:.2f} dB",
         ha="center", va="center", fontsize=11.5)
fig.text(0.5, 0.062,
         "读法：① 单视角 α 中位 0.073，一个像素要 ~25 个视角叠起来才不透光；"
         "② 每个视角的壳各自偏离 GT（÷α 后仍差 "
         f"{m['psnr_all_masked'] - m['psnr_self_n_masked']:.1f} dB）；"
         "③ 31 层平均把各自的误差平均掉（所以全图 PSNR 不低）但也把细节抹平 → 模糊 + 淡重影",
         ha="center", va="center", fontsize=10.5, color="#444444")
fig.text(0.5, 0.018,
         "渲染器与交付图逐像素一致（闸门 99.0 dB）；位姿 npz 与 COLMAP 一致（ΔR 2.4e-7 / Δt 0）；"
         "四元数 wxyz 序正确（薄盘最短轴与视线夹角中位 20.5°）；opacity/scale 全局增益扫掠峰在 1.0~2.0（无成倍口径错）",
         ha="center", va="center", fontsize=9.5, color="#666666")

p = OUT / "fig21_audit_decomp.png"
fig.savefig(p, facecolor="white")
print(f"💾 {p}")
