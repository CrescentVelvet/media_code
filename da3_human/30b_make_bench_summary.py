#!/usr/bin/env python3
"""30b_make_bench_summary.py — 视角数分桶 + 多包基线摸底 的 PPT 汇总图。

输入：D:\\output\\bench_viewcount_view\\<tag>\\eval\\multiview_report.json（WSL 侧拷出）
输出：同目录 ppt\\ 下汇总图（不入 git）。

图 0：视角数 scaling（PSNR mean/min、LPIPS、FG PSNR、高斯数）
图 1：多包基线总表（逐包指标 + 视角覆盖登记）
图 2：视角数分桶的逐视角 PSNR 分布（spread 是否收敛）
"""
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SRC = Path(r"D:\output\bench_viewcount_view")
OUT = SRC / "ppt"
OUT.mkdir(parents=True, exist_ok=True)

IS_BUCKET = re.compile(r"^佳佳抱枕_\d+v$")

FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"


def font(sz, bold=False):
    p = r"C:\Windows\Fonts\msyhbd.ttc" if bold else FONT_PATH
    return ImageFont.truetype(p, sz)


def load_reports():
    reps = {}
    for d in sorted(SRC.iterdir()):
        rp = d / "eval" / "multiview_report.json"
        if rp.exists():
            reps[d.name] = json.loads(rp.read_text(encoding="utf-8"))
    return reps


def label(d, xy, s, sz, bold=False, anchor="la", fill=(30, 30, 30)):
    d.text(xy, s, font=font(sz, bold), anchor=anchor, fill=fill)


def header(d, W, s, size=34):
    label(d, (W // 2, 16), s, size, bold=True, anchor="ma")
    d.line([(40, 62), (W - 40, 62)], fill=(170, 170, 170), width=2)


def draw_panel(d, box, title, lines, tcol=(30, 60, 150), bg=(248, 249, 251), ln=(210, 215, 225)):
    d.rounded_rectangle(box, radius=12, fill=bg, outline=ln, width=2)
    x0, y0, x1, y1 = box
    label(d, (x0 + 16, y0 + 10), title, 21, bold=True, fill=tcol)
    for i, s in enumerate(lines):
        label(d, (x0 + 16, y0 + 42 + i * 26), s, 17)


def polyline(d, pts, col, width=4):
    d.line(pts, fill=col, width=width, joint="curve")


def main():
    reps = load_reports()
    vc = {k: v for k, v in reps.items() if IS_BUCKET.match(k)}
    pk = {k: v for k, v in reps.items() if not IS_BUCKET.match(k)}
    print(f"视角数桶 {len(vc)} 条，多包 {len(pk)} 条")

    def nv_key(name):
        return int(name.split("_")[-1].replace("v", ""))

    vc_names = sorted(vc, key=nv_key)

    # ================= 图 0：视角数 scaling =================
    W, H = 1280, 780
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    header(d, W, "同一采集包（佳佳抱枕 32v 源）前馈直出：输入视角数 scaling")
    xs = [nv_key(n) for n in vc_names]
    # 左：PSNR mean/min 折线
    box_l = (50, 90, 660, 430)
    draw_panel(d, box_l, "输入视角自一致性 PSNR（越高越好）",
               ["mean = 全部输入视角平均；min = 最差视角",
                "mean 下降：新增视角离参考更远；min 不升：不一致不收敛"])
    x0, y0, x1, y1 = box_l
    px0, py0, px1, py1 = x0 + 60, y0 + 104, x1 - 30, y1 - 60
    lo, hi = 8.0, 25.0
    for gval in (10, 15, 20, 25):
        y = py1 - (gval - lo) / (hi - lo) * (py1 - py0)
        d.line([(px0, y), (px1, y)], fill=(230, 230, 230), width=1)
        label(d, (px0 - 8, y - 9), f"{gval}", 14, anchor="ra", fill=(90, 90, 90))
    xslot = (px1 - px0) / (len(xs) - 1)
    def pt(i, v):
        return (px0 + i * xslot, py1 - (v - lo) / (hi - lo) * (py1 - py0))
    mean_pts = [pt(i, vc[n]["aggregate"]["psnr"]["mean"]) for i, n in enumerate(vc_names)]
    min_pts = [pt(i, vc[n]["aggregate"]["psnr"]["min"]) for i, n in enumerate(vc_names)]
    polyline(d, mean_pts, (60, 120, 220))
    polyline(d, min_pts, (230, 110, 90))
    for i, n in enumerate(vc_names):
        a = vc[n]["aggregate"]["psnr"]
        label(d, (px0 + i * xslot, py1 + 8), f"{xs[i]}v", 16, anchor="ma")
        label(d, (px0 + i * xslot, pt(i, a["mean"])[1] - 22), f"{a['mean']:.1f}", 14,
              anchor="ma", fill=(60, 120, 220))
        label(d, (px0 + i * xslot, pt(i, a["min"])[1] + 8), f"{a['min']:.1f}", 14,
              anchor="ma", fill=(230, 110, 90))
    label(d, (px0 + 4, py0 + 6), "● mean", 16, fill=(60, 120, 220))
    label(d, (px0 + 96, py0 + 6), "● min", 16, fill=(230, 110, 90))

    # 右：LPIPS + 空洞 + 高斯数
    box_r = (700, 90, W - 50, 430)
    rows = ["视角数 | LPIPS↓ | SSIM↑ | 空洞率% | 高斯数 | 前向渲染",
            "-" * 60]
    for n in vc_names:
        a = vc[n]["aggregate"]
        rows.append(f"{nv_key(n):>2}v   | {a['lpips']['mean']:.3f}  | {a['ssim']['mean']:.3f}   | "
                    f"{vc[n]['hole_pct_all']:>5.2f}   | {vc[n]['n_gaussians']/1e6:.2f}M | {vc[n]['render_seconds']:.1f}s")
    rows += ["", "LPIPS/SSIM 同步变差：新增输入视角自身渲染质量下降，",
             "与 PSNR mean 下降同向；min 不升 → 跨视角几何不一致",
             "是结构性问题，加视角不能自愈，必须 G1 几何头重训。"]
    draw_panel(d, box_r, "逐桶汇总（正式口径：不裁剪+全 SH）", rows)

    # 底部：判决
    d.rounded_rectangle([50, 460, W - 50, 740], radius=12, fill=(40, 44, 52))
    label(d, (W // 2, 480), "判决：视角数 3→18，最差视角 PSNR 不升（10.4~10.8 dB 平台），跨视角几何不一致不随视角数收敛；",
          22, bold=True, anchor="ma", fill=(255, 210, 120))
    label(d, (W // 2, 516), "训练目标分布必须覆盖 3/8/13/18 各桶且以几何一致性损失为主（与 EXPERIMENT_PLAN Phase 2 一致）。",
          22, anchor="ma", fill=(210, 210, 210))
    # 底部嵌入逐视角 PSNR 小图（每桶）
    bx0, by0, bx1, by1 = 80, 566, W - 80, 716
    n_max = max(vc[n]["n_views"] for n in vc_names)
    bucket_cols = [(80 + i * 40, 140, 240 - i * 30) for i in range(len(vc_names))]
    for i, n in enumerate(vc_names):
        v = vc[n]
        pvs = [v["views"][f"v{k}"]["psnr"] for k in range(v["n_views"])]
        pts = []
        for k, p in enumerate(pvs):
            x = bx0 + (k / max(1, n_max - 1)) * (bx1 - bx0)
            y = by1 - (p - 8) / (25 - 8) * (by1 - by0)
            pts.append((x, y))
        polyline(d, pts, bucket_cols[i], 3)
    for i, n in enumerate(vc_names):
        label(d, (bx0 + 30 + i * ((bx1 - bx0 - 200) / 3), by0 - 24), f"{nv_key(n)}v",
              16, bold=True, fill=bucket_cols[i])
    d.rectangle([bx0, by0, bx1, by1], outline=(120, 120, 120), width=1)
    label(d, (W // 2, 724), "各桶逐视角 PSNR（横轴=输入顺序，纵轴 8~25 dB）：最差视角始终停在 10 dB 平台",
          14, anchor="ma", fill=(190, 190, 190))
    canvas.save(OUT / "ppt_vc_0_scaling.png")
    print("ppt_vc_0 ok")

    # ================= 图 1：多包基线总表 =================
    W2 = 1280
    row_h = 44
    H2 = 100 + (len(pk) + 2) * row_h + 150
    canvas = Image.new("RGB", (W2, H2), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    header(d, W2, "多包基线摸底：DA3-Baseline-V1（官方权重，32v 全视角，不裁剪+全 SH）")
    cols = [("包名", 50, "l"), ("PSNR mean", 290, "r"), ("PSNR min", 410, "r"),
            ("FG PSNR", 525, "r"), ("SSIM", 635, "r"), ("LPIPS", 735, "r"),
            ("空洞%", 830, "r"), ("相邻角距最大°", 930, "r"), ("方位角跨度°", 1100, "r")]
    hy = 92
    for name, x, al in cols:
        label(d, (x, hy), name, 17, bold=True)
    az_span = lambda v: round(max(v["azimuth_deg"]) - min(v["azimuth_deg"]), 0)
    for r, (n, v) in enumerate(sorted(pk.items(), key=lambda kv: kv[1]["aggregate"]["psnr"]["mean"])):
        y = hy + 34 + r * row_h
        a = v["aggregate"]
        vals = [re.sub(r"_0v$", "", n), f"{a['psnr']['mean']:.2f}", f"{a['psnr']['min']:.2f}",
                f"{a['fg_psnr']['mean']:.2f}", f"{a['ssim']['mean']:.3f}",
                f"{a['lpips']['mean']:.3f}", f"{v['hole_pct_all']:.2f}",
                f"{v['pairwise_gap_max_deg']:.0f}", f"{az_span(v):.0f}"]
        for (name, x, al), val in zip(cols, vals):
            label(d, (x, y), str(val), 17)
        d.line([(50, y + 30), (W2 - 50, y + 30)], fill=(225, 228, 232), width=1)
    fy = hy + 34 + len(pk) * row_h + 14
    label(d, (W2 // 2, fy), "读法：每包最差视角 PSNR 仅 8~11.5 dB —— 每包都有 1~2 个崩坏视角（跨视角几何不一致的表现）；",
          20, anchor="ma", fill=(120, 40, 40))
    label(d, (W2 // 2, fy + 32), "空洞率全部 ≤0.01%（多视角叠加 + 不裁剪口径，覆盖无虞）；相邻角距最大 77~166° —— 采集轨迹不均匀，",
          20, anchor="ma", fill=(60, 60, 60))
    label(d, (W2 // 2, fy + 64), "存在折返/大缺口，训练集登记必须逐包记录方位角，不能假设均匀环绕。",
          20, anchor="ma", fill=(60, 60, 60))
    label(d, (W2 // 2, fy + 108), "结论：7/7 包稳定复现同一病因；训练集筛选建议：保留含难例视角的包，并按 3/8/13/18 桶各抽子集。",
          21, bold=True, anchor="ma", fill=(30, 60, 150))
    canvas.save(OUT / "ppt_vc_1_multipackage.png")
    print("ppt_vc_1 ok", H2)

    # ================= 图 2：各包最差视角（崩坏视角实样） =================
    tw, th = 300, 200  # 每格 pair 缩略（760x504 → 300x200）
    ncol = 4
    nrow = (len(pk) + ncol - 1) // ncol
    W3 = 40 * 2 + ncol * (tw + 14) - 14
    H3 = 96 + nrow * (th + 46) + 90
    canvas = Image.new("RGB", (W3, H3), (28, 28, 32))
    d = ImageDraw.Draw(canvas)
    header(d, W3, "每包最差输入视角实样：左=前馈渲染，右=GT（崩坏 = 跨视角几何不一致）", size=30)
    for r, (n, v) in enumerate(sorted(pk.items(), key=lambda kv: kv[1]["aggregate"]["psnr"]["min"])):
        # 找 PSNR 最差的视角
        worst_k = min(range(v["n_views"]), key=lambda k: v["views"][f"v{k}"]["psnr"])
        pair = Image.open(SRC / n / "eval" / f"v{worst_k:02d}_pair.png").resize((tw, th))
        rr, cc = divmod(r, ncol)
        x = 40 + cc * (tw + 14)
        y = 96 + rr * (th + 46)
        canvas.paste(pair, (x, y + 24))
        vv = v["views"][f"v{worst_k}"]
        label(d, (x, y), f"{re.sub(r'_0v$', '', n)} v{worst_k} 帧{vv['src_frame']} "
              f"{vv['psnr']:.1f}dB az{vv['azimuth_deg']:+.0f}°", 16, bold=True,
              fill=(255, 150, 120))
    label(d, (W3 // 2, H3 - 72), "最差视角的人物区域与 GT 存在可见差异（错位 / 灰化 / 细节丢失），对应 E4 的两条结论：",
          20, anchor="ma", fill=(220, 220, 220))
    label(d, (W3 // 2, H3 - 40), "center 跨视角错位 + 外观跨视角退化；且每包最差视角方位角各不相同 —— 系统性问题，不是固定坏帧。",
          20, anchor="ma", fill=(220, 220, 220))
    canvas.save(OUT / "ppt_vc_2_worst_views.png")
    print("ppt_vc_2 ok", W3, H3)


if __name__ == "__main__":
    main()
