#!/usr/bin/env python3
"""A2 · E-01 修复前后对比图（fig11）

上半：视角 0 / 视角 16，三列（GT · 修复前 · 修复后），单元格内标该视角 PSNR。
下半：32 视角的 PSNR ~ 到参考视角距离散点 + 前后均值，用来看"距离衰减"还剩多少。

数据来源（WSL）：
  ~/output/da3_gs_results/佳佳抱枕_p1/render_all32   修复前（PLY raw + 米制相机）
  ~/output/da3_gs_results/佳佳抱枕_p1m/render_all32  修复后（PLY ×2.0 + 米制相机）
  e01_before.json / e01_after.json                   逐视角指标

用法（WSL，querysplat env，有 cv2/matplotlib）：
  python compare_4methods/A2_make_errata_figure.py
"""
import json
import os
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

# WSL 下按名字匹配不到 msyh.ttc，必须显式注册（否则中文全是方块）
_MSYH = "/mnt/c/Windows/Fonts/msyh.ttc"
if os.path.exists(_MSYH):
    font_manager.fontManager.addfont(_MSYH)
    _CN = font_manager.FontProperties(fname=_MSYH).get_name()
    matplotlib.rcParams["font.sans-serif"] = [_CN, "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path("/home/velvet/output")
BEFORE = ROOT / "da3_gs_results/佳佳抱枕_p1"
AFTER = ROOT / "da3_gs_results/佳佳抱枕_p1m"
REF = ROOT / "da3_results/_captures/佳佳抱枕/images"
OUT = Path("/mnt/c/code/media_code/compare_4methods/ppt_assets/fig11_e01_before_after.png")
MIRROR = Path("/mnt/c/code/media_code/compare_4methods/ppt_assets")


def load_ref(idx, size):
    files = sorted(p for p in REF.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    img = cv2.cvtColor(cv2.imread(str(files[idx])), cv2.COLOR_BGR2RGB)
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def load_render(d, v, size):
    img = cv2.cvtColor(cv2.imread(str(Path(d) / f"view_{v:03d}.png")), cv2.COLOR_BGR2RGB)
    return img if img.shape[1::-1] == size else cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def main():
    jb = json.load(open(BEFORE / "e01_before.json", encoding="utf-8"))
    ja = json.load(open(AFTER / "e01_after.json", encoding="utf-8"))
    pb = {r["view"]: r["psnr"] for r in jb["per_view"]}
    pa = {r["view"]: r["psnr"] for r in ja["per_view"]}

    cam = np.load(AFTER / "cameras.npz")
    e = cam["extrinsics"].astype(np.float64)
    C = -np.einsum("nji,nj->ni", e[:, :, :3], e[:, :, 3])
    dist = np.linalg.norm(C - C[0], axis=1)

    views = [0, 16]
    size = (378, 504)
    fig = plt.figure(figsize=(12.6, 13.4))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.78], hspace=0.16, wspace=0.04)

    for r, v in enumerate(views):
        imgs = [load_ref(v, size), load_render(BEFORE / "render_all32", v, size),
                load_render(AFTER / "render_all32", v, size)]
        titles = [f"GT（输入帧）",
                  f"修复前：{pb[v]:.2f} dB",
                  f"修复后：{pa[v]:.2f} dB  (+{pa[v]-pb[v]:.2f})"]
        for c in range(3):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(imgs[c])
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=13, pad=7,
                             color=("#0C447C" if c == 0 else ("#8A5A2B" if c == 1 else "#1E6B3A")))
            else:
                ax.set_title(titles[c], fontsize=12, pad=5,
                             color=("#0C447C" if c == 0 else ("#8A5A2B" if c == 1 else "#1E6B3A")))
            if c == 0:
                ax.set_ylabel(f"视角 {v}\n（帧 0 距离 {dist[v]:.2f} m）", fontsize=12)
            for s in ax.spines.values():
                s.set_color("#888780")
                s.set_linewidth(0.5)

    # ── 下半：逐视角 PSNR ~ 距离 ──────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, :])
    xs = np.arange(32)
    ax.plot(xs, [pb[i] for i in xs], "o-", color="#C77A45", lw=1.2, ms=3.4,
            label=f"修复前（均值 {jb['mean']['psnr']:.2f} dB）")
    ax.plot(xs, [pa[i] for i in xs], "o-", color="#1E6B3A", lw=1.4, ms=3.4,
            label=f"修复后（均值 {ja['mean']['psnr']:.2f} dB）")
    ax.axvline(0, color="#888780", lw=0.8, ls=":")
    ax.annotate("参考视角：径向缩放不改变投影\n→ 预测 +0.00，实测 +0.07 dB",
                xy=(0, pa[0]), xytext=(1.9, 22.9), fontsize=10.5, color="#444441",
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#888780"))
    ax.annotate(f"最差视角（帧 16）：{pb[16]:.2f} → {pa[16]:.2f} dB",
                xy=(16, pa[16]), xytext=(17.2, 23.4), fontsize=10.5, color="#444441",
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#888780"))
    ax.set_xlabel("视角序号（0 = 参考视角）", fontsize=11, labelpad=7)
    ax.set_ylabel("自一致性 PSNR (dB)", fontsize=11)
    ax.set_xticks(xs[::2])
    ax.grid(alpha=0.22, lw=0.6)
    ax.legend(fontsize=10.5, loc="lower left", frameon=False)
    ax.set_ylim(6, 26.5)
    ax.spines[["top", "right"]].set_visible(False)

    near = dist <= 1.0
    txt = (f"PSNR 均值　近≤1 m：{np.mean([pb[i] for i in xs[near]]):.1f} → "
           f"{np.mean([pa[i] for i in xs[near]]):.1f} dB　　>1 m："
           f"{np.mean([pb[i] for i in xs[~near]]):.1f} → {np.mean([pa[i] for i in xs[~near]]):.1f} dB"
           f"　　近远差距 3.31 → 1.42 dB")
    ax.text(0.988, 0.055, txt, transform=ax.transAxes, ha="right", fontsize=10,
            color="#444441")
    fig.suptitle("E-01 修复：DA3 高斯补乘 scale_factor=2.0 后，同一 PLY 同一相机的前后对比",
                 fontsize=15, y=0.995)
    fig.text(0.5, 0.012,
             "口径：与输入帧比（自一致性，非 NVS holdout），统一 378×504；两组仅差“高斯是否乘 2.0”，相机与分辨率完全相同",
             ha="center", fontsize=10, color="#5F5E5A")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    for src, dst in ((BEFORE / "e01_before.json", "e01_before.json"),
                     (AFTER / "e01_after.json", "e01_after.json")):
        json.dump(json.load(open(src, encoding="utf-8")),
                  open(MIRROR / dst, "w", encoding="utf-8", newline="\n"),
                  ensure_ascii=False, indent=1)
    print(f"✅ {OUT}")


if __name__ == "__main__":
    main()
