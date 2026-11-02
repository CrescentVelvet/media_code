#!/usr/bin/env python3
"""量化对比「整图重建」与「ROI 重建」的手部清晰度。

为什么需要独立脚本：Step 2 已证明 σ_eff(px) 不可跨分辨率比较 —— 同样是 1.5 px，
在 392 宽的整图上和在 380 宽的 ROI 图上物理含义完全不同。本脚本给两个量纲无关的量：

  σ_rel      = σ_eff(px) / 手部像素宽       原生分辨率下的相对模糊（越小越锐）
  σ_eff@160  = 手部区域重采样到 160 px 宽后的 σ_eff      同观看尺寸下谁更锐

第二个量才是"用户凑近看手时"的真实体验：整图组的手只有约 80 px，放大到 160 px
必然插值发虚；ROI 组的手有约 260 px，缩小反而更实。

手部区域定义：SAM3 双手 bbox 并集（整图归一化坐标），按 --pad 扩边后裁切。
ROI 组需要经 meta.json 的 roi_px / src_wh 换算出该组图像内的 bbox。

用法:
  python 13_roi_eval.py --mask-manifest .../hand_32v/masks_manifest.json \
    --group full32=<cap>/jiajia=<res>/jiajia \
    --group roi030=<cap>/jiajia_roi030_518=<res>/jiajia_roi030_518 \
    --out .../roi_eval.json
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

import ghost_metrics as gm

SIGMAS = tuple(np.arange(0.0, 6.01, 0.25))


def load_hand_boxes(manifest):
    mf = json.loads(Path(manifest).read_text(encoding="utf-8"))
    boxes = {}
    for stem, insts in mf.get("frames", {}).items():
        if not insts:
            continue
        key = stem.split("_", 1)[1] if "_" in stem else stem
        x0 = min(it["box_xywh"][0] for it in insts)
        y0 = min(it["box_xywh"][1] for it in insts)
        x1 = max(it["box_xywh"][0] + it["box_xywh"][2] for it in insts)
        y1 = max(it["box_xywh"][1] + it["box_xywh"][3] for it in insts)
        boxes[key] = (x0, y0, x1, y1)
    return boxes


def bbox_in_group(box, meta_frame, is_roi, src_wh):
    """整图归一化 bbox → 该组图像归一化 bbox。

    整图组直接用；ROI 组按 12 的 meta（源像素坐标 roi_px + 源尺寸 src_wh）换算。
    ⚠️ 换算必须先在**像素**域做减法再归一化，直接在归一化域减 roi_frac 会错，
    因为 ROI 框是矩形（宽高各 0.30），而归一化的 u/v 分母不同。
    """
    if not is_roi:
        return box
    W, H = src_wh
    x0, y0, x1, y1 = meta_frame["roi_px"]
    ow, oh = meta_frame["out_wh"]
    return ((box[0] * W - x0) / (x1 - x0), (box[1] * H - y0) / (y1 - y0),
            (box[2] * W - x0) / (x1 - x0), (box[3] * H - y0) / (y1 - y0))


def crop_resize(img, box, target_w, pad):
    """按 bbox（归一化）扩边裁切并重采样到固定显示宽度，保持宽高比。"""
    H, W = img.shape[:2]
    cx = 0.5 * (box[0] + box[2]) * W
    cy = 0.5 * (box[1] + box[3]) * H
    hw = 0.5 * (box[2] - box[0]) * W * pad
    hh = 0.5 * (box[3] - box[1]) * H * pad
    x0 = int(round(cx - hw))
    x1 = int(round(cx + hw))
    y0 = int(round(cy - hh))
    y1 = int(round(cy + hh))
    # 越界时整体平移而不是补黑边（补黑边会制造强边缘，污染 σ_eff）
    if x0 < 0:
        x1 -= x0; x0 = 0
    if y0 < 0:
        y1 -= y0; y0 = 0
    x1, y1 = min(x1, W), min(y1, H)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None, 0.0
    patch = img[y0:y1, x0:x1]
    hand_px = max(1.0, (box[2] - box[0]) * W)
    scale = target_w / patch.shape[1]
    out_h = max(8, int(round(patch.shape[0] * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    patch = cv2.resize(patch, (target_w, out_h), interpolation=interp)
    return patch, hand_px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", action="append", required=True,
                    metavar="NAME=CAPTURE_DIR=RESULT_DIR",
                    help="可重复。CAPTURE_DIR 需含 images/ + meta.json")
    ap.add_argument("--mask-manifest", required=True)
    ap.add_argument("--hand-width", type=int, default=160)
    ap.add_argument("--pad", type=float, default=1.8)
    ap.add_argument("--max-views", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    boxes = load_hand_boxes(args.mask_manifest)
    print(f"🖐️ 手部 bbox {len(boxes)} 帧，pad={args.pad}，显示宽度 {args.hand_width}px")

    all_rows = {}
    for spec in args.group:
        parts = spec.split("=")
        if len(parts) != 3:
            sys.exit(f"❌ --group 格式应为 NAME=CAPTURE_DIR=RESULT_DIR，收到 {spec}")
        name, cap, res = parts
        cap, res = Path(cap), Path(res)
        meta = json.loads((cap / "meta.json").read_text(encoding="utf-8"))
        is_roi = meta.get("mode") == "roi"
        frame_meta = {Path(f["src"]).stem: f for f in meta["frames"]}
        imgs = sorted(glob.glob(str(cap / "images" / "*.png")))[:args.max_views]
        renders = sorted(glob.glob(str(res / "eval" / "render_*.png")))[:len(imgs)]
        if len(renders) != len(imgs):
            print(f"⚠️ {name}: GT {len(imgs)} vs 渲染 {len(renders)}，取较小者")
        n = min(len(imgs), len(renders))

        rows = []
        for k in range(n):
            stem = Path(imgs[k]).stem.split("_", 1)[1] if "_" in Path(imgs[k]).stem else Path(imgs[k]).stem
            if stem not in boxes:
                continue
            fm = frame_meta.get(stem)
            if fm is None:
                print(f"⚠️ {name}: 第 {k} 帧在 meta 里找不到 {stem}")
                continue
            src_wh = tuple(fm.get("src_wh") or (0, 0))
            if is_roi and not src_wh:
                sys.exit("❌ ROI 组的 meta 缺 src_wh（旧版 12 脚本产物），请重新转换")
            box = bbox_in_group(boxes[stem], fm, is_roi, src_wh)

            gt = cv2.cvtColor(cv2.imread(imgs[k]), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            rd = cv2.cvtColor(cv2.imread(renders[k]), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            # ⚠️ 方向必须是「GT 降到渲染分辨率」，不能反过来放大渲染图。
            # 07/12 存的是去畸变原图（约 1352x1878），而 pipeline 实际处理到长边 518 ——
            # 重建的有效分辨率是渲染图的分辨率。把渲染放大去凑 GT 会凭空注入 3.4x
            # 上采样模糊，实测让整图组 σ 从 1.5 虚高到 3.1，结论直接反向。
            if gt.shape[:2] != rd.shape[:2]:
                interp = cv2.INTER_AREA if rd.shape[1] < gt.shape[1] else cv2.INTER_LINEAR
                gt = cv2.resize(gt, (rd.shape[1], rd.shape[0]), interpolation=interp)

            # 原生口径：直接在原图分辨率下裁（不需要重采样）
            gp, hand_px = crop_resize(gt, box, args.hand_width, args.pad)
            rp, _ = crop_resize(rd, box, args.hand_width, args.pad)
            if gp is None or rp is None or gp.shape != rp.shape:
                continue

            sig_disp, psnr_disp, _ = gm.equivalent_blur_sigma(gp, rp, sigmas=SIGMAS)
            # σ_native：把统一显示口径的 σ 换算回该组原生像素（交叉验证用 ——
            # 整图组换算后应接近 Step 0 实测的 1.5 px，否则说明口径有问题）。
            # σ_rel = σ_native / 手部像素宽 = σ_disp / hand_width，即"模糊占手宽的比例"。
            sig_native = sig_disp * hand_px / args.hand_width
            rows.append(dict(view=k, hand_px=round(hand_px, 1),
                             psnr_disp=round(gm.psnr(gp, rp), 3),
                             sigma_disp=round(sig_disp, 2),
                             sigma_native=round(sig_native, 3),
                             sigma_rel=round(sig_disp / args.hand_width, 5)))

        if not rows:
            print(f"❌ {name}: 没有有效帧")
            continue
        agg = {k: float(np.median([r[k] for r in rows]))
               for k in ("hand_px", "psnr_disp", "sigma_disp", "sigma_native", "sigma_rel")}
        all_rows[name] = dict(mode=meta.get("mode", "full"), n=len(rows), agg=agg, per_view=rows)
        print(f"  [{name}] n={len(rows):2d} 手部{agg['hand_px']:6.1f}px  "
              f"PSNR@{args.hand_width} {agg['psnr_disp']:6.3f}  "
              f"σ@{args.hand_width} {agg['sigma_disp']:5.2f}  "
              f"σ_native {agg['sigma_native']:5.3f}  σ_rel {agg['sigma_rel']*100:5.2f}%")

    Path(args.out).write_text(json.dumps(dict(hand_width=args.hand_width, pad=args.pad,
                                              groups=all_rows), indent=2), encoding="utf-8")
    print(f"\n✅ 写出 {args.out}")
    print(f"\n{'组':<14}{'模式':<7}{'手部px':>8}{'PSNR':>9}{'σ@160':>8}{'σ_rel':>9}")
    for name, d in all_rows.items():
        a = d["agg"]
        print(f"{name:<14}{d['mode']:<7}{a['hand_px']:>8.1f}{a['psnr_disp']:>9.3f}"
              f"{a['sigma_disp']:>8.2f}{a['sigma_rel']*100:>8.2f}%")


if __name__ == "__main__":
    main()
