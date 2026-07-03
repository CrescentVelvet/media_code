#!/usr/bin/env python3
"""Build a PAIRED training parquet for HYPIR from an HQ folder and an LQ folder.

Pairs are matched by FILENAME (the crop_faces_paired.py script already saves
HQ and LQ crops with identical names under hq/ and lq/). Output columns:
``hq_path``, ``lq_path``, ``prompt`` (all absolute), consumed by
``PairedFaceDataset`` via ``file_meta`` in sd2_train_paired.yaml.

Env:
  HQ_DIR       (required) folder of high-quality face crops (e.g. .../hq)
  LQ_DIR       (required) folder of low-quality face crops  (e.g. .../lq)
  PARQUET_OUT  (required) output .parquet path
  PROMPT       (default "")  caption for every pair ("" = null-text training)
  MIN_SIDE     (default 0)   skip pairs whose LQ smaller side is below this (px)
"""

import os
import sys
from pathlib import Path

import polars as pl
from PIL import Image

HQ_DIR = os.environ.get("HQ_DIR", "")
LQ_DIR = os.environ.get("LQ_DIR", "")
PARQUET_OUT = os.environ.get("PARQUET_OUT", "")
PROMPT = os.environ.get("PROMPT", "")
MIN_SIDE = int(os.environ.get("MIN_SIDE", "0"))

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def scan(root):
    return {Path(p).name: p for p in [
        os.path.join(r, f) for r, _, fs in os.walk(root) for f in fs
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    ]}


def main():
    for name, val in (("HQ_DIR", HQ_DIR), ("LQ_DIR", LQ_DIR), ("PARQUET_OUT", PARQUET_OUT)):
        if not val:
            sys.exit(f"ERROR: set {name}.")
    for name, val in (("HQ_DIR", HQ_DIR), ("LQ_DIR", LQ_DIR)):
        if not os.path.isdir(val):
            sys.exit(f"ERROR: {name} not found: {val}")

    hq = scan(HQ_DIR)
    lq = scan(LQ_DIR)
    common = sorted(set(hq) & set(lq))
    only_hq = sorted(set(hq) - set(lq))
    only_lq = sorted(set(lq) - set(hq))
    print(f"[*] HQ={len(hq)}  LQ={len(lq)}  paired={len(common)}  "
          f"only-HQ={len(only_hq)}  only-LQ={len(only_lq)}")
    if only_hq[:3]:
        print(f"    sample HQ-without-LQ: {only_hq[:3]}")
    if only_lq[:3]:
        print(f"    sample LQ-without-HQ: {only_lq[:3]}")
    if not common:
        sys.exit("ERROR: no filename-matched HQ/LQ pairs — check that both folders "
                 "hold the same crop names (crop_faces_paired.py does this).")

    hq_paths, lq_paths, dropped = [], [], 0
    for name in common:
        if MIN_SIDE > 0:
            try:
                w, h = Image.open(lq[name]).size
                if min(w, h) < MIN_SIDE:
                    dropped += 1
                    continue
            except Exception:
                dropped += 1
                continue
        hq_paths.append(os.path.abspath(hq[name]))
        lq_paths.append(os.path.abspath(lq[name]))

    out = os.path.abspath(PARQUET_OUT)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df = pl.from_dict({
        "hq_path": hq_paths,
        "lq_path": lq_paths,
        "prompt": [PROMPT] * len(hq_paths),
    })
    df.write_parquet(out)
    print(f"[*] wrote {len(hq_paths)} pairs -> {out}")
    if dropped:
        print(f"    dropped {dropped} pairs (LQ < {MIN_SIDE}px or unreadable)")
    print(f"[*] prompt={'<empty>' if PROMPT == '' else repr(PROMPT)}")
    print(f"[*] next: PARQUET_PATH={out} bash hypir/04_train_paired.sh")


if __name__ == "__main__":
    main()
