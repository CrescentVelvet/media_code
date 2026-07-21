#!/usr/bin/env python3
"""Scan a folder for corrupt/truncated images; optionally delete them.

A "corrupt" image is one that PIL cannot fully decode — e.g. a truncated JPEG
(OSError: image file is truncated, the exact failure that crashes
build_beauty_dataset.py at `Image.open(fp).convert("RGB")`). This script walks
INPUT_DIR recursively, tries to open+convert every image, and lists the ones
that raise. Set DELETE=1 to remove them in place (good one-time cleanup before
running 03d, so the source folder has no broken files for later runs).

build_beauty_dataset.py already skips corrupt images (try/except + half-pair
cleanup), so this scanner is OPTIONAL — use it when you want to clean the
SOURCE folder rather than just skip-and-carry-on.

Env:
  INPUT_DIR  (required) folder of images (walked recursively)
  DELETE     (default 0) 1 = unlink corrupt files; 0 = just list them

Usage:
  INPUT_DIR=/data/faces python hypir/scan_corrupt_images.py                 # list only
  INPUT_DIR=/data/faces DELETE=1 python hypir/scan_corrupt_images.py        # list + delete
"""
import os
import sys
from pathlib import Path

from PIL import Image, ImageFile

# Do NOT set LOAD_TRUNCATED_IMAGES — we WANT truncated images to raise so we
# can detect them. (build_beauty_dataset.py also leaves this unset, so the two
# scripts agree on what counts as "corrupt".)
INPUT_DIR = os.environ.get("INPUT_DIR", "")
DELETE = os.environ.get("DELETE", "0") == "1"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif", ".ppm"}


def is_corrupt(fp: Path) -> str:
    """Return '' if the image decodes fine, else the error message."""
    try:
        with Image.open(fp) as im:
            im.convert("RGB")      # forces full decode -> raises on truncation
            im.load()
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    return ""


def main():
    if not INPUT_DIR:
        sys.exit("ERROR: set INPUT_DIR (folder of images to scan).")
    if not os.path.isdir(INPUT_DIR):
        sys.exit(f"ERROR: INPUT_DIR not found: {INPUT_DIR}")

    images = []
    for root, _, files in os.walk(INPUT_DIR):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                images.append(Path(root) / f)
    images.sort(key=lambda x: str(x.relative_to(INPUT_DIR)))
    if not images:
        sys.exit(f"ERROR: no images in {INPUT_DIR}")

    print(f"[*] scanning {len(images)} image(s) under {INPUT_DIR}  (DELETE={DELETE})")
    corrupt = []
    for i, fp in enumerate(images, 1):
        rel = fp.relative_to(INPUT_DIR)
        err = is_corrupt(fp)
        if err:
            corrupt.append(fp)
            print(f"[{i}/{len(images)}] CORRUPT  {rel.as_posix()}  ({err})")
            if DELETE:
                try:
                    fp.unlink()
                    print(f"           deleted.")
                except Exception as de:
                    print(f"           ! delete failed: {de}", file=sys.stderr)
        elif i % 200 == 0 or i == len(images):
            print(f"[{i}/{len(images)}] ok so far, {len(corrupt)} corrupt ...")

    print(f"\n[*] === summary ===")
    print(f"    scanned:  {len(images)}")
    print(f"    corrupt: {len(corrupt)}")
    if DELETE:
        print(f"    deleted:  {len(corrupt)} (in place)")
        print(f"    (rerun 03d on the cleaned {INPUT_DIR} — no more truncated images to skip.)")
    elif corrupt:
        print(f"    (not deleted — rerun with DELETE=1 to remove them.)")
        print(f"    corrupt paths (first 10):")
        for p in corrupt[:10]:
            print(f"      {p.relative_to(INPUT_DIR).as_posix()}")
    else:
        print(f"    all good — no corrupt images found.")


if __name__ == "__main__":
    main()
