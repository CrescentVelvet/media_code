#!/usr/bin/env python3
"""Build a HYPIR training parquet from a folder of images.

Scans DATA_DIR recursively for images and writes a parquet with columns
`image_path` (absolute) and `prompt` (PROMPT, "" by default). The resulting
parquet is consumed by HYPIR.dataset.realesrgan.RealESRGANDataset via
file_meta: {file_list, image_path_prefix:"", image_path_key:"image_path",
prompt_key:"prompt"} — see configs/sd2_train.yaml.

With CROP=1 (default), each image is first sliced into CROP_SIZE×CROP_SIZE
patches (non-overlapping by default; set CROP_STRIDE < CROP_SIZE for overlap)
saved under CROP_OUT, and the parquet points at the patches. This is needed
because the default dataset config uses crop_type=none + out_size=512, which
asserts every GT image is exactly 512×512.

Env:
  DATA_DIR       (required) folder of high-quality images
  PARQUET_OUT    (required) output .parquet path
  PROMPT         (default "")  caption for every image (use "" for null-text training)
  CROP           (default 1)   1 = slice into patches first; 0 = use images as-is
  CROP_SIZE      (default 512) patch size
  CROP_STRIDE    (default =CROP_SIZE) patch stride (smaller => overlap)
  CROP_OUT       (default <parquet_dir>/patches) where patches are saved
  IMAGE_DIR      alias for DATA_DIR (DATA_DIR wins if both set)
"""
import os
import sys
from pathlib import Path

import polars as pl
from PIL import Image

DATA_DIR = os.environ.get("DATA_DIR") or os.environ.get("IMAGE_DIR")
PARQUET_OUT = os.environ.get("PARQUET_OUT")
PROMPT = os.environ.get("PROMPT", "")
CROP = os.environ.get("CROP", "1") == "1"
CROP_SIZE = int(os.environ.get("CROP_SIZE", "512"))
CROP_STRIDE = int(os.environ.get("CROP_STRIDE", str(CROP_SIZE)))  # default non-overlapping
CROP_OUT = os.environ.get("CROP_OUT")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def scan_images(root):
    out = []
    for r, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                out.append(os.path.join(r, f))
    return sorted(out)


def main():
    if not DATA_DIR:
        sys.exit("ERROR: set DATA_DIR (folder of images).")
    if not PARQUET_OUT:
        sys.exit("ERROR: set PARQUET_OUT (output .parquet path).")
    if not os.path.isdir(DATA_DIR):
        sys.exit(f"ERROR: DATA_DIR not found: {DATA_DIR}")

    data_dir = os.path.abspath(DATA_DIR)
    parquet_out = os.path.abspath(PARQUET_OUT)
    os.makedirs(os.path.dirname(parquet_out) or ".", exist_ok=True)

    paths = []
    if CROP:
        if not CROP_OUT:
            CROP_OUT = os.path.join(os.path.dirname(parquet_out) or ".", "patches")
        crop_out = os.path.abspath(CROP_OUT)
        os.makedirs(crop_out, exist_ok=True)
        files = scan_images(data_dir)
        print(f"[*] cropping {len(files)} image(s) -> {CROP_SIZE}px patches "
              f"(stride {CROP_STRIDE}) into {crop_out}")
        n_img = 0
        for f in files:
            stem = Path(f).stem
            try:
                img = Image.open(f).convert("RGB")
            except Exception as e:
                print(f"    skip {os.path.basename(f)} (load failed: {e})")
                continue
            W, H = img.size
            if H < CROP_SIZE or W < CROP_SIZE:
                print(f"    skip {os.path.basename(f)} ({W}x{H} < {CROP_SIZE})")
                continue
            idx = 0
            top = 0
            while top + CROP_SIZE <= H:
                left = 0
                while left + CROP_SIZE <= W:
                    patch = img.crop((left, top, left + CROP_SIZE, top + CROP_SIZE))
                    out_path = os.path.join(crop_out, f"{stem}_{idx:05d}.png")
                    patch.save(out_path)
                    paths.append(os.path.abspath(out_path))
                    idx += 1
                    left += CROP_STRIDE
                top += CROP_STRIDE
            n_img += 1
            if n_img % 200 == 0:
                print(f"    ... {n_img}/{len(files)} images, {len(paths)} patches")
        print(f"[*] cropped {n_img} image(s) -> {len(paths)} patches")
    else:
        paths = [os.path.abspath(p) for p in scan_images(data_dir)]
        print(f"[*] using {len(paths)} image(s) as-is (CROP=0) from {data_dir}")
        # Sanity-check sizes when crop_type=none is intended.
        bad = 0
        for p in paths[:50]:
            try:
                with Image.open(p) as im:
                    if im.size != (CROP_SIZE, CROP_SIZE):
                        bad += 1
            except Exception:
                bad += 1
        if bad:
            print(f"[!] warning: {bad}/50 sampled images are not {CROP_SIZE}x{CROP_SIZE}.")
            print("    RealESRGANDataset with crop_type=none asserts exact out_size.")
            print("    Either set CROP=1 here, or pass CROP_TYPE=random to 04_train.sh.")

    if not paths:
        sys.exit("ERROR: no usable images found (or none large enough to crop).")

    df = pl.from_dict({"image_path": paths, "prompt": [PROMPT] * len(paths)})
    df.write_parquet(parquet_out)
    print(f"[*] wrote {len(paths)} rows -> {parquet_out}")
    print(f"[*] columns: image_path (absolute), prompt={'<empty>' if PROMPT == '' else repr(PROMPT)}")
    print(f"[*] next: PARQUET_PATH={parquet_out} bash hypir/04_train.sh")


if __name__ == "__main__":
    main()
