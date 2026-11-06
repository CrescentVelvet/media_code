#!/usr/bin/env python3
"""R-3 误差定位图：整脸统计会把局部错误平均掉，必须看误差在空间上落在哪。

背景：fig17 里 RES=1 的脸有明显块状污渍，但 A7/A7b 的整脸统计显示两组几乎一致
（相关系数 0.93~0.99）。这与 P0 的老教训同构：mask 内的均值指标抓不到局部缺陷。
所以这里直接画 |diff| 热图 + 灰度放大，让"污渍在哪、谁的错误更成块"变成可见证据。

fig19：3 行（GT / RES=2 / RES=1）× 3 列（渲染 / |误差|×6 / 误差灰度）
"""
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
BASE = Path("/home/velvet/output/da3_gs_results/佳佳抱枕_res_ab")
REF = Path("/home/velvet/output/da3_results/_captures/佳佳抱枕/images")
FACE_BOX = (0.35, 0.105, 0.565, 0.335)
VIEW = 0
GAIN = 6.0


def crop(im):
    h, w = im.shape[:2]
    box = (int(FACE_BOX[0] * w), int(FACE_BOX[1] * h),
           int(FACE_BOX[2] * w), int(FACE_BOX[3] * h))
    return im[box[1]:box[3], box[0]:box[2]]


def main():
    from PIL import Image

    refs = sorted(p for p in REF.iterdir() if p.suffix.lower() == ".png")
    gt = np.asarray(Image.open(refs[VIEW]).convert("RGB"), dtype=np.float64)

    rows = [("GT", None)]
    for tag, lab in (("res2", "RES=2 训练（720×960）"), ("res1", "RES=1 训练（1439×1919）")):
        p = BASE / f"final_{tag}" / "render_1439x1919" / f"view_{VIEW:03d}.png"
        rows.append((lab, np.asarray(Image.open(p).convert("RGB"), dtype=np.float64)))

    fig, axes = plt.subplots(len(rows), 3, figsize=(9.6, 3.1 * len(rows)), dpi=160)
    for i, (lab, img) in enumerate(rows):
        g = crop(gt)
        if img is None:
            axes[i, 0].imshow(g.astype(np.uint8))
            axes[i, 1].imshow(np.zeros_like(g, dtype=np.uint8) + 255)
            axes[i, 2].imshow(np.zeros_like(g, dtype=np.uint8) + 255)
            sub = "—"
        else:
            r = crop(img)
            d = np.abs(g - r).mean(axis=2)
            axes[i, 0].imshow(r.astype(np.uint8))
            hm = axes[i, 1].imshow(np.clip(d * GAIN, 0, 255), cmap="inferno", vmin=0, vmax=110)
            axes[i, 2].imshow(np.clip(d * GAIN, 0, 255).astype(np.uint8), cmap="gray", vmin=0, vmax=255)
            mse = float((d ** 2).mean())
            sub = f"PSNR {10 * np.log10(255.0 ** 2 / mse):.2f} dB ｜ 平均|误差| {d.mean():.2f}"
        axes[i, 0].set_ylabel(lab, fontsize=11)
        axes[i, 0].text(0.02, 0.97, sub, transform=axes[i, 0].transAxes, fontsize=9,
                        va="top", color="white" if img is not None else "#333333",
                        bbox=dict(fc="#00000099" if img is not None else "#ffffff00",
                                  ec="none", pad=2.5))
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)

    for ax, t in zip(axes[0], ("渲染（人脸区放大）", f"|误差| ×{GAIN:g}", f"|误差| 灰度 ×{GAIN:g}")):
        ax.set_title(t, fontsize=11.5, pad=7)

    cb = fig.colorbar(hm, ax=axes[1:, 1].tolist(), fraction=0.022, pad=0.015)
    cb.set_label("放大后的误差灰度", fontsize=9)
    fig.suptitle("误差落在哪：整脸统计一致，但错误的空间分布完全不同（view 0，1439×1919 原生口径）",
                 fontsize=12.5, y=0.995)
    p = OUT / "fig19_res_error_maps.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("✅", p)


if __name__ == "__main__":
    main()
