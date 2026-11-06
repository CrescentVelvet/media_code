#!/usr/bin/env python3
"""R-2 出图：前馈初值「DC-only」vs「保留 SH rest」，在米制 + 2k 短精调下的 A/B。

fig14：渲染对照（GT / DC-only / +SH rest / 差值放大）—— 看 SH rest 到底改了什么
fig15：四组 PSNR 与逐视角 Δ —— 看收益是"整体抬升"还是"个别视角的局部修补"

两组口径：
  dc / full        active_sh_degree 从 0 起（3DGS 原生，2k 步里 SH 只在后 1000 步生效）
  dc_d2 / full_d2  初始 degree=2（SH 全程生效，用于排除"SH 没被用够"这个解释）

数据：~/output/da3_gs_results/佳佳抱枕_sh_ab/
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

BASE = Path("/home/velvet/output/da3_gs_results/佳佳抱枕_sh_ab")
REF = Path("/home/velvet/output/da3_results/_captures/佳佳抱枕/images")

# (key, 列标题, 颜色)
GROUPS = [
    ("dc", "DC-only 初值\n25.21 dB", "#8C8C8C"),
    ("full", "+SH rest 初值\n25.44 dB", "#C77C2E"),
    ("dc_d2", "DC-only · deg=2\n25.14 dB", "#B0A99C"),
    ("full_d2", "+SH rest · deg=2\n25.40 dB", "#D9A35B"),
]


def load(key):
    p = BASE / f"final_{key}" / "metrics_sh_ab.json"
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def fig14(views=(0, 16), gain=10.0):
    from PIL import Image, ImageDraw, ImageFont

    refs = sorted(p for p in REF.iterdir() if p.suffix.lower() == ".png")
    W, H, pad, top = 268, 357, 6, 74
    ncol = 4
    canvas = Image.new(
        "RGB", (ncol * (W + pad) + pad, top + len(views) * (H + pad + 26) + pad), "white"
    )
    dr = ImageDraw.Draw(canvas)
    f_h = ImageFont.truetype(_MSYH, 15)
    f_r = ImageFont.truetype(_MSYH, 14)
    f_s = ImageFont.truetype(_MSYH, 13)

    titles = ["GT\n输入帧", "DC-only 初值\n(对照)", "+SH rest 初值\n(实验)",
              f"|差值| ×{gain:g}\n暖色 = SH 改变了这里"]
    for j, t in enumerate(titles):
        x = pad + j * (W + pad)
        for k, line in enumerate(t.split("\n")):
            dr.text((x + 2, 6 + k * 19), line, fill=(30, 30, 30), font=f_h if k == 0 else f_s)

    for i, v in enumerate(views):
        y = top + i * (H + pad + 26)
        dr.text((pad + 2, y - 21), f"视角 {v}", fill=(110, 110, 110), font=f_r)
        try:
            ims = [Image.open(BASE / "final_dc" / "render_all32" / f"view_{v:03d}.png").convert("RGB"),
                   Image.open(BASE / "final_full" / "render_all32" / f"view_{v:03d}.png").convert("RGB")]
        except FileNotFoundError:
            continue
        rects = [Image.open(refs[v]).convert("RGB")] + ims
        imgs = [r.resize((W, H), Image.LANCZOS) for r in rects]
        for j, im in enumerate(imgs):
            canvas.paste(im, (pad + j * (W + pad), y))

        a = np.asarray(imgs[1], dtype=np.float32)
        b = np.asarray(imgs[2], dtype=np.float32)
        d = np.clip(np.abs(a - b).mean(axis=2) * gain, 0, 255)
        # 差值图用暖色：白→黄→红，纯黑表示 SH 没有改变该像素
        rgb = np.stack([np.clip(d * 1.0, 0, 255), np.clip(d * 0.55, 0, 255), np.zeros_like(d)], axis=2)
        canvas.paste(Image.fromarray(rgb.astype(np.uint8)), (pad + 3 * (W + pad), y))
        dr.text((pad + 3 * (W + pad) + 6, y + H - 22),
                f"均值差 {np.abs(a - b).mean():.2f}", fill=(255, 255, 255), font=f_s)

    p = OUT / "fig14_sh_ab_renders.png"
    canvas.save(p)
    print("✅", p, canvas.size)


def fig15():
    got = {k: load(k) for k, _, _ in GROUPS}
    keys = [k for k in got if got[k] is not None]
    if not keys:
        print("⏭️ 还没有任何 metrics，跳过 fig15")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6), dpi=170,
                            gridspec_kw={"width_ratios": [1.55, 1]})
    ax = axes[0]
    xs = np.arange(32)
    style = {"dc": dict(ls="-", lw=1.9), "full": dict(ls="-", lw=1.9),
             "dc_d2": dict(ls="--", lw=1.3), "full_d2": dict(ls="--", lw=1.3)}
    for k, lab, c in GROUPS:
        if k not in keys:
            continue
        v = [x["psnr"] for x in got[k]["per_view"]]
        ax.plot(xs, v, color=c, label=lab.replace("\n", " · "), **style[k])
    ax.set_xlabel("视角序号（0 = 参考视角）", fontsize=11)
    ax.set_ylabel("自一致性 PSNR (dB)", fontsize=11)
    ax.set_title("逐视角 PSNR：实线 = 原生 degree 时间表；虚线 = 初始 degree=2", fontsize=12,
                 loc="left", pad=10)
    ax.legend(fontsize=9, frameon=False, ncol=4, loc="lower center",
              bbox_to_anchor=(0.5, -0.34))
    ax.grid(alpha=0.25, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    # 两组 Δ 并排：若两条几乎重合，说明"SH 是否全程生效"不影响结论
    pairs = [("dc", "full", "原生 degree 时间表", "#C77C2E", 0.0),
             ("dc_d2", "full_d2", "初始 degree=2", "#D9A35B", 0.0)]
    pairs = [p for p in pairs if p[0] in keys and p[1] in keys]
    for i, (a, b, lab, c, _) in enumerate(pairs):
        dl = np.array([f["psnr"] - d["psnr"] for d, f in zip(got[a]["per_view"], got[b]["per_view"])])
        off = (i - (len(pairs) - 1) / 2) * 0.42
        ax.bar(xs + off, dl, width=0.4, color=c, alpha=1.0 if i == 0 else 0.55, label=lab)
        if i == 0:
            ax.axhline(np.mean(dl), ls=":", lw=1.3, color="#B5651D")
            ax.text(0.015, 0.045, f"均值 {np.mean(dl):+.2f} dB（±{np.std(dl):.2f}，"
                    f"范围 {dl.min():+.2f}~{dl.max():+.2f}）", transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=9.5, color="#B5651D")
    ax.axhline(0, color="#888780", lw=0.9)
    ax.set_xlabel("视角序号", fontsize=11)
    ax.set_ylabel("ΔPSNR = +SH rest − DC-only (dB)", fontsize=11)
    ax.margins(y=0.16)
    ax.set_title("两种时间表的 Δ 几乎重合（结论不依赖 degree），但逐视角波动远大于均值",
                 fontsize=12, loc="left", pad=10)
    ax.legend(fontsize=9, frameon=False, loc="upper left", ncol=1)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    fig.text(0.012, 0.015,
             "同场景/同超参/同一次前向导出的两份 PLY（顶点数完全一致），"
             "唯一差别是 f_rest 是否为 DA3 的 sh_degree=2 系数；32 训练视角自一致性口径",
             fontsize=9.5, color="#888888")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    p = OUT / "fig15_sh_ab_bars.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("✅", p)


def face_region():
    """C3：全图指标会掩盖局部。按视角手工框住人脸（相对坐标，从 fig14 目视定位）。

    ⚠️ 框必须贴着人脸：早先用的 40%×40% 大框把书架和地面都圈进来，
    算出的 9.5 dB 是"背景差异"，不能用来判断人脸。
    """
    from PIL import Image

    refs = sorted(p for p in REF.iterdir() if p.suffix.lower() == ".png")
    boxes = {0: (0.395, 0.145, 0.505, 0.275), 16: (0.275, 0.285, 0.395, 0.415)}
    rows = []
    for v, bx in boxes.items():
        gt = Image.open(refs[v]).convert("RGB")
        w, h = gt.size
        box = (int(bx[0] * w), int(bx[1] * h), int(bx[2] * w), int(bx[3] * h))
        g = np.asarray(gt.crop(box).resize((160, 160), Image.LANCZOS), dtype=np.float64)
        seg = [f"v{v}"]
        for k, _, _ in GROUPS:
            p = BASE / f"final_{k}" / "render_all32" / f"view_{v:03d}.png"
            if not p.exists():
                continue
            r_img = Image.open(p).convert("RGB")
            rw, rh = r_img.size
            r = np.asarray(r_img.crop((int(bx[0] * rw), int(bx[1] * rh),
                                       int(bx[2] * rw), int(bx[3] * rh))
                                      ).resize((160, 160), Image.LANCZOS), dtype=np.float64)
            mse = ((g - r) ** 2).mean()
            seg.append(f"{k}={10 * np.log10(255 ** 2 / mse):.2f}")
        rows.append("  ".join(seg))
    print("🖼️ 人脸区 PSNR（贴脸小框：v0 40×40 px 等效 / v16 42×42）")
    for r in rows:
        print("  ", r)


if __name__ == "__main__":
    fig14()
    fig15()
    face_region()
