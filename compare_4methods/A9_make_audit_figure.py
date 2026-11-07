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
         "「仅本视角」的重影不是 bug：一个视角的壳本身就是稀疏、半透明的点画"
         "（RES=1 训练产物，378×504 口径）",
         ha="center", va="center", fontsize=14.5, weight="bold")
fig.text(0.5, 0.120,
         f"同一像素集（仅本视角覆盖到的像素，占全图 43%）上比：全部 {m['psnr_all_masked']:.2f} dB"
         f" ｜ 仅本视角 {m['psnr_self_masked']:.2f} dB（暗，因为单层 α 只有 0.073）"
         f" ｜ ÷α 后 {m['psnr_self_n_masked']:.2f} dB ｜ 去掉本视角 {m['psnr_other_masked']:.2f} dB",
         ha="center", va="center", fontsize=11.5)
fig.text(0.5, 0.087,
         "① 每视角仅 0.79 个高斯/像素、PLY 只保留 79.4% 的像素、单层 α 中位 0.036"
         " → 单看一层必然是稀疏 + 半透明的点画；成品靠 32 视角叠到 α≈1.000",
         ha="center", va="center", fontsize=10.0, color="#444444")
fig.text(0.5, 0.055,
         "② 去掉 1/32 的贡献只动 0.5 dB，属正常稀释，不能据此说「本视角没用」；"
         "③ ÷α 会把噪声放大 ~14 倍，不能当几何误差读（真实残差 ~1 px 横向 + 26~42 mm 沿视线）",
         ha="center", va="center", fontsize=10.0, color="#444444")
fig.text(0.5, 0.024,
         "官方等价口径（610 万点 + 全 SH + 正确帧）= 22.11 dB；本图 18.97 dB 是「裁剪 + 仅 DC」的 PLY 交付口径，"
         "差 3.14 dB 全在导出选项；帧错（缩放相机配未缩放高斯）会掉到 11.27 dB",
         ha="center", va="center", fontsize=9.3, color="#666666")

p = OUT / "fig21_audit_decomp.png"
fig.savefig(p, facecolor="white")
print(f"💾 {p}")
