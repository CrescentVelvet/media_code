#!/usr/bin/env python3
"""Build an HQ-only parquet for HYPIR's on-the-fly RealESRGAN degradation.

Scans HQ_DIR recursively for images and writes a parquet with columns
`image_path` (absolute) + `prompt` (PROMPT, "" by default). The trainer's
RealESRGANDataset reads this parquet and synthesizes LQ on-the-fly each epoch
(HYPIR's default degradation: two-stage blur/sinc/noise/jpeg) — no LQ files
are stored, augmentation is fresh every epoch.

This is the "synthetic degradation" counterpart to build_paired_dataset.py
(which uses real LQ+HQ pairs). Train with 04_train.sh (official SD2Trainer,
from-scratch LoRA) — or warm-start from the released LoRA, see README.

RealESRGANDataset needs HQ >= OUT_SIZE(512): use CROP_TYPE=random (crops 512
patches on-the-fly) for large HQ, or pre-resize to 512 for CROP_TYPE=none.
This script just lists paths (no resize); it warns about images < 512px.

Env:
  HQ_DIR       (required) folder of high-quality images
  PARQUET_OUT  (required) output .parquet path
  PROMPT       (default "")  caption for every image ("" = null-text training)
"""
import os
import sys

import polars as pl
from PIL import Image

HQ_DIR = os.environ.get("HQ_DIR", "")
PARQUET_OUT = os.environ.get("PARQUET_OUT", "")
PROMPT = os.environ.get("PROMPT", "")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def scan(root):
    out = []
    for r, _, fs in os.walk(root):
        for f in fs:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                out.append(os.path.join(r, f))
    return sorted(out)


def main():
    if not HQ_DIR:
        sys.exit("ERROR: set HQ_DIR.")
    if not PARQUET_OUT:
        sys.exit("ERROR: set PARQUET_OUT.")
    if not os.path.isdir(HQ_DIR):
        sys.exit(f"ERROR: HQ_DIR not found: {HQ_DIR}")

    hq = [os.path.abspath(p) for p in scan(HQ_DIR)]
    if not hq:
        sys.exit(f"ERROR: no images in {HQ_DIR}")

    # RealESRGANDataset needs HQ >= 512 (it crops 512 patches on-the-fly, or
    # asserts exactly 512 for crop_type=none). Warn about small images.
    small = 0
    for p in hq[:200]:
        try:
            w, h = Image.open(p).size
            if min(w, h) < 512:
                small += 1
        except Exception:
            pass
    if small:
        print(f"[!] {small}/200 sampled images have min side < 512px.")
        print("    RealESRGANDataset needs HQ >= 512 (CROP_TYPE=random crops 512 on-the-fly);")
        print("    pre-resize small images to >=512 before training.")

    out = os.path.abspath(PARQUET_OUT)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df = pl.from_dict({"image_path": hq, "prompt": [PROMPT] * len(hq)})
    df.write_parquet(out)
    print(f"[*] wrote {len(hq)} HQ images -> {out}  (HQ-only; LQ synthesized on-the-fly)")
    print(f"[*] prompt={'<empty>' if PROMPT == '' else repr(PROMPT)}")
    print(f"[*] next: PARQUET_PATH={out} CROP_TYPE=random bash hypir/04_train.sh")


if __name__ == "__main__":
    main()
