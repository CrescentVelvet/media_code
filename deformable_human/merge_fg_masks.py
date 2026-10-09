#!/usr/bin/env python3
"""merge_fg_masks.py — 合并 SAM3 两遍分割为单张前景软 mask。

规则：
  fg = union(person 全部实例) ∪ { 与 person 邻接的 extra 实例 }
「邻接」判定：extra 实例二值像素中落在 dilate(person) 内的比例 ≥ overlap_min。
  ——实测 SAM3 把背景白雕塑也检成 "dinosaur plush toy"（静态误检），
    持有物永远贴着人，用邻接过滤掉远处误检，见 2026-10-09 日志。

输入命名（vggt_human/sam3_face_masks_worker.py 约定）：
  {stem}.p{pid:02d}.alpha.png  每实例软 mask；单实例帧另有 {stem}.alpha.png
输出：
  out_dir/{stem}.png           uint8 软 mask（0-255，羽毛边保留）
  out_dir/_qc_overlay.jpg      抽检帧红色叠加图（目视 QC）
  out_dir/_merge_report.json   逐帧覆盖率/接收/拒绝统计
"""
import os
import json
import glob
import argparse

import numpy as np
from PIL import Image


def dilate_bool(mask, ksize):
    """二值膨胀，ksize 为奇数核边长。cv2 优先，scipy 兜底。"""
    try:
        import cv2
        kernel = np.ones((ksize, ksize), np.uint8)
        return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    except ImportError:
        from scipy.ndimage import binary_dilation
        return binary_dilation(mask, structure=np.ones((ksize, ksize)))


def load_alpha(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def union_alphas(pattern):
    """glob 一组 alpha png 并逐像素取 max；无文件返回 None。"""
    out = None
    for f in sorted(glob.glob(pattern)):
        a = load_alpha(f)
        out = a if out is None else np.maximum(out, a)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--person_dir", required=True)
    ap.add_argument("--extra_dirs", nargs="*", default=[])
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--overlap_min", type=float, default=0.3)
    ap.add_argument("--dilate_frac", type=float, default=0.03)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(args.images_dir)
                   if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png"))

    report = {}
    n_missing = n_reject = n_accept = 0
    qc_rows = []

    for stem in stems:
        person = union_alphas(os.path.join(args.person_dir, stem + "*.alpha.png"))
        if person is None:
            person = union_alphas(os.path.join(args.person_dir, stem + "*.mask.png"))
        if person is None:
            n_missing += 1
            report[stem] = {"error": "no person mask"}
            continue

        H, W = person.shape
        ksize = max(3, int(min(H, W) * args.dilate_frac) | 1)
        person_bin = person > 0.5
        person_dil = dilate_bool(person_bin, ksize)

        fg = person
        accepted, rejected = [], []
        for d in args.extra_dirs:
            for f in sorted(glob.glob(os.path.join(d, stem + ".p*.alpha.png"))):
                inst = load_alpha(f)
                inst_bin = inst > 0.5
                npx = int(inst_bin.sum())
                if npx == 0:
                    continue
                ov = float((inst_bin & person_dil).sum()) / npx
                tag = os.path.basename(d) + "/" + os.path.basename(f)
                if ov >= args.overlap_min:
                    fg = np.maximum(fg, inst)
                    accepted.append(tag)
                    n_accept += 1
                else:
                    rejected.append("%s(ov=%.2f)" % (tag, ov))
                    n_reject += 1

        Image.fromarray((fg * 255).astype(np.uint8)).save(
            os.path.join(args.out_dir, stem + ".png"))
        report[stem] = {
            "fg_coverage_pct": round(float((fg > 0.5).mean()) * 100, 2),
            "accepted": accepted, "rejected": rejected,
        }
        qc_rows.append(stem)

    with open(os.path.join(args.out_dir, "_merge_report.json"), "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)

    # QC 叠加图：均匀取最多 6 帧，红通道叠加 fg
    picks = [qc_rows[int(i * (len(qc_rows) - 1) / 5)] for i in range(6)] if qc_rows else []
    tiles = []
    for stem in picks:
        img_path = None
        for ext in (".jpg", ".png", ".jpeg"):
            p = os.path.join(args.images_dir, stem + ext)
            if os.path.isfile(p):
                img_path = p
                break
        if img_path is None:
            continue
        img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32)
        m = load_alpha(os.path.join(args.out_dir, stem + ".png"))
        if m.shape[:2] != img.shape[:2]:
            m = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize(
                (img.shape[1], img.shape[0])), dtype=np.float32) / 255.0
        over = img.copy()
        over[..., 0] = np.clip(over[..., 0] + 255 * 0.45 * m, 0, 255)
        tiles.append(over.astype(np.uint8))
    if tiles:
        rows = [np.concatenate(tiles[i:i + 3], axis=1) for i in range(0, len(tiles), 3)]
        sheet = np.concatenate(rows, axis=0)
        Image.fromarray(sheet).save(os.path.join(args.out_dir, "_qc_overlay.jpg"),
                                    quality=90)

    print("merge done: %d frames, %d missing-person, %d extra accepted, %d extra rejected"
          % (len(stems), n_missing, n_accept, n_reject))
    covs = [v["fg_coverage_pct"] for v in report.values() if "fg_coverage_pct" in v]
    if covs:
        print("fg coverage: min %.1f%%  median %.1f%%  max %.1f%%"
              % (min(covs), float(np.median(covs)), max(covs)))


if __name__ == "__main__":
    main()
