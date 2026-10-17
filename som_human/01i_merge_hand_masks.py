#!/usr/bin/env python3
"""01i_merge_hand_masks.py — 把 SAM3 多实例手部 mask 合并为逐帧单张 mask。

01g（SAM3_PROMPT=hand）产出的是多实例命名：{stem}.p{pid:02d}.mask.png
（左手/右手各一个 track）。train_som.py 的 FGMaskBank 约定是 {stem}.png，
本脚本把所有 pid 的 mask 取逐像素 max 合并，写到 merged/ 子目录。

用法：
    python 01i_merge_hand_masks.py --masks <scene>/hands/masks [--out <scene>/hands/merged]
"""
import argparse
import os
import re
import sys

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", required=True, help="hands/masks 目录")
    ap.add_argument("--out", default="", help="输出目录（默认 <masks>/../merged）")
    args = ap.parse_args()

    out_dir = args.out or os.path.join(os.path.dirname(os.path.normpath(args.masks)), "merged")
    os.makedirs(out_dir, exist_ok=True)

    pat = re.compile(r"^(?P<stem>.+)\.p\d{2}\.mask\.png$")
    by_stem = {}
    for f in sorted(os.listdir(args.masks)):
        m = pat.match(f)
        if m:
            by_stem.setdefault(m.group("stem"), []).append(f)

    n = 0
    for stem, files in by_stem.items():
        acc = None
        for f in files:
            m = np.asarray(Image.open(os.path.join(args.masks, f)).convert("L"))
            acc = m if acc is None else np.maximum(acc, m)
        Image.fromarray(acc).save(os.path.join(out_dir, stem + ".png"))
        n += 1
    print("merged {} frames -> {}".format(n, out_dir))
    # 覆盖率统计（用于确认合并合理）
    covs = []
    for stem in list(by_stem)[:200]:
        m = np.asarray(Image.open(os.path.join(out_dir, stem + ".png")).convert("L"))
        covs.append((m > 127).mean())
    print("coverage mean {:.2%}  min {:.2%}  max {:.2%}".format(
        np.mean(covs), np.min(covs), np.max(covs)))


if __name__ == "__main__":
    main()
