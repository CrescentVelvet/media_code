#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""03b_fg_metrics.py — 前景专项指标（官方 metrics.py 只有全图 PSNR/SSIM/LPIPS）

动机（方案 E 验收）：深度先验/手部去模糊的增益会被「整图 PSNR」平均掉——
背景占了大部分像素且本来就压得住。要看前景（人）是否真的变好，必须在
fg mask 内单独算 PSNR，并同时报背景（作为「没有牺牲背景换前景」的对照）。

输入（03_render.sh MODE=render 的产物 + 01d 的 mask）：
  <model>/<split>/ours_<iter>/renders/NNNNN.png   render_som.py 写的渲染
  <model>/<split>/ours_<iter>/gt/NNNNN.png        真值
  <scene>/masks/<stem>.png                        01d 软 mask（0-255）

⚠️ 索引映射：render.py 用 enumerate(getTestCameras()) 命名 NNNNN，而 test 集是
`sorted(stems)[0::8]`（llffhold=8）→ 第 i 张渲染对应 stems[i*8]。本脚本按同一
规则重建映射，不靠猜。

输出：每视角 fg/bg/all 三个 PSNR + 均值；--montage 时把 fg PSNR 最差的 N 张
拼成「GT | Render | 绝对误差×4」三联图，便于手部区域目视定位。

用法：
  python 03b_fg_metrics.py --scene <scene> --model <model> [--iteration 20000]
                           [--split test] [--top 6] [--montage out.png]
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image


def psnr(a, b, mask=None):
    """a/b: float32 [H,W,3] in [0,1]；mask: bool [H,W] 或 None"""
    d = (a - b) ** 2
    if mask is not None:
        if mask.sum() == 0:
            return float("nan")
        d = d[mask]                      # [n,3]
    mse = float(d.mean())
    if mse <= 0:
        return 99.0
    return 10.0 * np.log10(1.0 / mse)


def load_rgb(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


def load_mask(p, size_hw):
    if not os.path.exists(p):
        return None
    m = Image.open(p).convert("L")
    if (m.size[1], m.size[0]) != size_hw:
        m = m.resize((size_hw[1], size_hw[0]), Image.BILINEAR)
    return np.asarray(m, dtype=np.float32) > 127.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="COLMAP 场景目录（含 images/ 与 masks/）")
    ap.add_argument("--model", required=True, help="训练输出目录（含 test/ours_<iter>/）")
    ap.add_argument("--iteration", type=int, default=20000)
    ap.add_argument("--split", default="test")
    ap.add_argument("--mask-dir", default=None, help="默认 <scene>/masks")
    ap.add_argument("--top", type=int, default=0, help="拼图取 fg PSNR 最差的 N 张")
    ap.add_argument("--montage", default=None, help="拼图输出路径（.png）")
    ap.add_argument("--json", default=None, help="把逐视角结果写 JSON")
    args = ap.parse_args()

    img_dir = os.path.join(args.scene, "images")
    mask_dir = args.mask_dir or os.path.join(args.scene, "masks")
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(img_dir))
    if args.split == "test":
        stems = stems[0::8]              # llffhold=8，与 dataset_readers.py:193 一致
    else:
        stems = [s for i, s in enumerate(stems) if i % 8 != 0]

    rdir = os.path.join(args.model, args.split,
                        "ours_{}".format(args.iteration), "renders")
    gdir = os.path.join(args.model, args.split,
                        "ours_{}".format(args.iteration), "gt")
    if not os.path.isdir(rdir):
        print("❌ 找不到渲染目录: {}".format(rdir), file=sys.stderr)
        return 2

    n_mask_missing = 0
    rows = []
    for i, stem in enumerate(stems):
        rp = os.path.join(rdir, "{:05d}.png".format(i))
        gp = os.path.join(gdir, "{:05d}.png".format(i))
        if not (os.path.exists(rp) and os.path.exists(gp)):
            continue
        r = load_rgb(rp)
        g = load_rgb(gp)
        m = load_mask(os.path.join(mask_dir, stem + ".png"), r.shape[:2])
        if m is None:
            n_mask_missing += 1
        rows.append({
            "i": i, "stem": stem,
            "all": psnr(r, g, None),
            "fg": psnr(r, g, m) if m is not None else float("nan"),
            "bg": psnr(r, g, ~m) if m is not None else float("nan"),
            "fg_frac": float(m.mean()) if m is not None else float("nan"),
        })

    if not rows:
        print("❌ 没有可比对的视角（渲染图缺失？）", file=sys.stderr)
        return 2

    def mean(k):
        v = np.array([r[k] for r in rows], dtype=np.float64)
        v = v[np.isfinite(v)]
        return float(v.mean()) if v.size else float("nan")

    print("📊 前景专项指标  split={}  views={}  mask 缺失={}".format(
        args.split, len(rows), n_mask_missing))
    print("   fg 像素占比均值: {:.1%}".format(
        np.nanmean([r["fg_frac"] for r in rows])))
    print("   ALL PSNR = {:.3f}   FG PSNR = {:.3f}   BG PSNR = {:.3f}".format(
        mean("all"), mean("fg"), mean("bg")))
    print()
    print("   逐视角 FG PSNR（升序，最差在前）：")
    for r in sorted(rows, key=lambda x: (x["fg"] if np.isfinite(x["fg"]) else 1e9))[:12]:
        print("     #{:2d} stem {}  fg={:6.3f}  bg={:6.3f}  all={:6.3f}".format(
            r["i"], r["stem"], r["fg"], r["bg"], r["all"]))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"summary": {"all": mean("all"), "fg": mean("fg"),
                                   "bg": mean("bg"), "views": len(rows)},
                       "per_view": rows}, f, ensure_ascii=False, indent=1)
        print("\n   JSON → {}".format(args.json))

    if args.top > 0 and args.montage:
        worst = sorted([r for r in rows if np.isfinite(r["fg"])],
                       key=lambda x: x["fg"])[:args.top]
        make_montage(rdir, gdir, mask_dir, worst, args.montage)
        print("   Montage → {}（GT | Render | |diff|×4 三联，红框=fg mask 边界）"
              .format(args.montage))
    return 0


def make_montage(rdir, gdir, mask_dir, worst, out_path):
    """GT | Render | |diff|×4 三联竖排（每行一个视角）"""
    rows = []
    for r in worst:
        rp = os.path.join(rdir, "{:05d}.png".format(r["i"]))
        gp = os.path.join(gdir, "{:05d}.png".format(r["i"]))
        gt = load_rgb(gp)
        rd = load_rgb(rp)
        diff = np.clip(np.abs(rd - gt) * 4.0, 0, 1)
        h, w = gt.shape[:2]
        m = load_mask(os.path.join(mask_dir, r["stem"] + ".png"), (h, w))
        # fg 之外压暗，便于看清「误差是否集中在人身上」
        if m is not None:
            dim = 0.25
            gt_v = gt.copy(); rd_v = rd.copy()
            gt_v[~m] *= dim; rd_v[~m] *= dim
            diff_v = diff.copy(); diff_v[~m] *= dim
        else:
            gt_v, rd_v, diff_v = gt, rd, diff
        sep = np.ones((h, 3, 3), dtype=np.float32)
        rows.append(np.concatenate([gt_v, sep, rd_v, sep, diff_v], axis=1))
    sep_h = np.ones((3, rows[0].shape[1], 3), dtype=np.float32)
    out = rows[0]
    for rr in rows[1:]:
        out = np.concatenate([out, sep_h, rr], axis=0)
    Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).save(out_path)


if __name__ == "__main__":
    sys.exit(main())
