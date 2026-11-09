#!/usr/bin/env python3
"""效果展示图（PPT 用，PIL 拼接，等高缩放）。

角色划分（2026-11-09 峰哥确认）
------------------------------
  主角    DA3-GS（DA3 原生高斯头，13.5 s 一次前向直出）
  同档    QuerySplat / InfiniSplat v2
  参照    VGGT-Ω+3DGS / DA3+3DGS —— 20+ 分钟，只作「多花的 100 倍时间买到什么」的对照，
          画面降饱和 45% + 列头用灰橙色，一眼分得出主次

产出：ppt_assets/fig35_showcase.png   6 列（GT + 5 方法）× 2 行（视角 0 / 16）

口径
  · DA3-GS 用「完整输出」档（610 万点 + 全 SH + 不裁剪）= 22.11 dB；
    轻量交付档 18.97 dB / 314 MB 只在脚注出现。
  · 各方法按各自预测内参渲染，视场/分辨率不同（DA3-GS 378×504、QuerySplat 512²、
    InfiniSplat 392×518、训练类 720×960），此处等高缩放 —— 只比观感与失败模式，
    不做像素级 PSNR 对比（ERRATA E6）。
  · 视角 0 / 16 已逐列目视核对，六格确为同一相机。
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HOME = Path("/home/velvet")
OUT = Path("/mnt/c/code/media_code/compare_4methods/ppt_assets")
FONT_R = "/mnt/c/Windows/Fonts/msyh.ttc"
FONT_B = "/mnt/c/Windows/Fonts/msyhbd.ttc"

O = HOME / "output"
P_GT = O / "vggt_omega_results/佳佳抱枕/04_model_3dgs/train/ours_30000/gt"
P_D3 = O / "da3_gs_results/佳佳抱枕_p1m_off/official_export"
P_QS = O / "querysplat_results/佳佳抱枕_paper/rendered"
P_IS = O / "ppt_assets/is_renders"
P_D3T = O / "da3_results/佳佳抱枕/04_model_3dgs/train/ours_30000/renders"
P_VG = O / "vggt_omega_results/佳佳抱枕/04_model_3dgs/train/ours_30000/renders"

C_HERO = "#0F6E6E"
C_QS = "#5B7FB5"
C_IS = "#8B6FA8"
C_REF = "#A8834F"

CASES = [
    dict(name="GT（训练视角原图）", sub="模型输入帧 · 720×960", color="#444444",
         f={0: P_GT / "00000.png", 16: P_GT / "00016.png"}, psnr=""),
    dict(name="DA3-GS", sub="13.5 秒 · 610 万高斯 · 22.11 dB", color=C_HERO,
         f={0: P_D3 / "ply_i12_fullpts_sh_view000.png",
            16: P_D3 / "ply_i12_fullpts_sh_view016.png"},
         psnr="一次前向直出（原生高斯头）"),
    dict(name="QuerySplat", sub="38.5 秒 · 37.1 万高斯", color=C_QS,
         f={0: P_QS / "render_view0.png", 16: P_QS / "render_view16.png"},
         psnr="查询式高斯头（冻结 VGGT-Ω）"),
    dict(name="InfiniSplat v2", sub="3.3 分钟 · 594 万高斯", color=C_IS,
         f={0: P_IS / "render_000.png", 16: P_IS / "render_016.png"},
         psnr="前馈 + 3000 步短精调"),
    dict(name="VGGT-Ω + 3DGS", sub="20.4 分钟 · 175 万高斯", color=C_REF,
         f={0: P_VG / "00000.png", 16: P_VG / "00016.png"}, gray=True,
         psnr="前馈 35 s + 训练 30k 步"),
    dict(name="DA3 + 3DGS", sub="22.9 分钟 · 189 万高斯", color=C_REF,
         f={0: P_D3T / "00000.png", 16: P_D3T / "00016.png"}, gray=True,
         psnr="前馈 130 s + 训练 30k 步"),
]


def font(path, size):
    return ImageFont.truetype(path, size)


def load(path, h, gray=False):
    im = Image.open(path).convert("RGB")
    w = max(1, int(round(im.width * h / im.height)))
    im = im.resize((w, h), Image.LANCZOS)
    if gray:
        im = Image.blend(im, im.convert("L").convert("RGB"), 0.25)   # 轻降饱和：标识配角，别读成"质量差"
    return im


def center_text(d, cx, y, text, f, fill):
    w = d.textlength(text, font=f)
    d.text((cx - w / 2, y), text, font=f, fill=fill)


def compose(cases, views, out_path, title, subtitle, foot, concl=None, h=470):
    cols = []
    for c in cases:
        ims = {v: load(c["f"][v], h, c.get("gray", False)) for v in views}
        cols.append((c, ims, max(im.width for im in ims.values())))

    pad, lab_w = 14, 86
    head_h, row_gap, foot_h = 240, 46, 84
    concl_h = 62 if concl else 0
    W = lab_w + sum(w for _, _, w in cols) + pad * (len(cols) + 1)
    H = head_h + h * len(views) + row_gap * (len(views) - 1) + 30 + concl_h + foot_h
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)

    f_h1 = font(FONT_B, 42)
    f_h2 = font(FONT_R, 21)
    f_title = font(FONT_B, 24)
    f_sub = font(FONT_R, 17)
    f_sub2 = font(FONT_R, 16)
    f_lab = font(FONT_B, 23)
    f_foot = font(FONT_R, 19)
    f_concl = font(FONT_R, 21)

    center_text(d, W / 2, 12, title, f_h1, "#1B1B1B")
    center_text(d, W / 2, 78, subtitle, f_h2, "#6E6E6E")

    xs, x = [], lab_w + pad
    for _, _, w in cols:
        xs.append(x)
        x += w + pad

    for (c, ims, w), x0 in zip(cols, xs):
        top = head_h - 34
        d.rectangle([x0, top, x0 + w, top + 4], fill=c["color"])
        center_text(d, x0 + w / 2, top - 76, c["name"], f_title, c["color"])
        center_text(d, x0 + w / 2, top - 40, c["sub"], f_sub, "#6E6E6E")

    for ri, v in enumerate(views):
        y = head_h + ri * (h + row_gap)
        d.rectangle([0, y, lab_w - 10, y + h], fill="#F5F4F2")
        lab = "视角 %d" % v
        lw = d.textlength(lab, font=f_lab)
        d.text(((lab_w - 10 - lw) / 2, y + h / 2 - 16), lab, font=f_lab, fill="#5E5E5E")
        for (c, ims, w), x0 in zip(cols, xs):
            im = ims[v]
            canvas.paste(im, (x0 + int((w - im.width) / 2), y))
            d.rectangle([x0, y, x0 + w - 1, y + h - 1], outline="#DDDAD5", width=1)
        if ri == 0:
            for (c, ims, w), x0 in zip(cols, xs):
                if c.get("psnr"):
                    center_text(d, x0 + w / 2, y + h + 10, c["psnr"], f_sub2, "#8A8A8A")

    yb = head_h + len(views) * h + row_gap * (len(views) - 1)
    if concl:
        d.rectangle([0, yb + 16, W, yb + 16 + 44], fill="#FAF8F4")
        center_text(d, W / 2, yb + 26, concl, f_concl, "#3A3A3A")
    for i, line in enumerate(foot):
        center_text(d, W / 2, H - foot_h + 18 + i * 27, line, f_foot, "#909090")

    canvas.save(out_path, quality=95)
    print("✅", out_path, canvas.size)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    compose(
        CASES, [0, 16], OUT / "fig35_showcase.png",
        "效果对比：DA3-GS 一次前向 13.5 秒，四条基线要多花 3~100 倍时间",
        "同一数据（佳佳抱枕 · 32 视角）· 同一批帧 · 同一相机视角 · RTX 3090 单卡　｜　"
        "上：视角 0　下：视角 16　｜　右侧两列降饱和＝参照档",
        [
            "注：各方法按各自预测内参渲染，视场与分辨率不同（DA3-GS 378×504 / QuerySplat 512² / "
            "InfiniSplat v2 392×518 / 训练类 720×960），此处按等高缩放 —— 只比观感与失败模式，不做像素级 PSNR 对比。",
            "DA3-GS 为完整输出档：610 万点 + 全 SH + 不裁剪 = 22.11 dB；"
            "轻量交付档（裁剪到 484 万 + 仅 DC）为 18.97 dB / 314 MB，体积小 3 倍。",
        ],
        concl="20.4 / 22.9 分钟 vs 13.5 秒：多出的时间里 97.5% 是逐场景的 30k 步优化，"
              "这段开销换不来可复用的能力",
    )
