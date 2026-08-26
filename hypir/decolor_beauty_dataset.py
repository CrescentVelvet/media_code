#!/usr/bin/env python3
"""Decolor beauty dataset: keep beauty's high-freq (smoothing) + orig's low-freq (color).

Reads pixel-aligned ``hq_orig/`` + ``hq_beauty/`` (produced by
``03d_build_beauty_dataset.sh``) and writes ``hq_beauty_decolor/`` =
``wavelet_reconstruction(content=hq_beauty, style=hq_orig)`` — i.e. the beautified
image's HIGH frequency (skin-smoothing texture, blemish removal) blended with the
ORIGINAL image's LOW frequency (true skin tone / color).

Why: RetouchFormer's output tends to look reddish (肤色红润) — a low-frequency
color bias baked into the released weights (the VRT selective self-attention
pulls blemish regions toward the surrounding skin statistics, which skew warm).
Smoothing/blemish-removal, however, is a HIGH-frequency change. So:
    result = beauty_high + orig_low
keeps the smoothing (high-freq from beauty) while restoring the true skin color
(low-freq from orig) — "去红润、保留磨皮".

The three wavelet_* functions are COPIED VERBATIM from
``HYPIR/utils/common.py:32-80`` (self-contained — torch only, no cv2 / no
diffusers). Keeping them inline means this script has zero HYPIR import
dependency and can run in any env that has torch (retouchformer env, hypir env,
or a plain torch env alike) — easy to drop into any face-enhancement pipeline.

Because the HQ *training target* is now de-colored, a LoRA trained on the
resulting parquet (via the UNCHANGED ``04b_train_paired.sh``) learns to restore
blurry LQ faces toward de-colored beauty. Inference with that LoRA (via the
UNCHANGED ``02_run_inference.sh``) thus produces "去红润 + 磨皮" results — neither
the training nor the inference script needs any modification; only the HQ target
folder / parquet path differs.

Env vars (set by 03e_decolor_beauty_dataset.sh):
  BEAUTY_DIR        (required) 03d output root (contains hq_orig/ + hq_beauty/ [+ lq_gauss/])
  HQ_ORIG_NAME      (default "hq_orig")       original (un-beautified) subdir name
  HQ_BEAUTY_NAME    (default "hq_beauty")     beautified (reddish) subdir name
  HQ_DECOLOR_NAME   (default "hq_beauty_decolor")  output subdir name
  DEVICE            (default "cuda")          falls back to CPU if CUDA unavailable
  SAVE_COMPARE      (default "0")             1 = also save compare_decolor/<name>.png
                                                = [orig | beauty(reddish) | decolor]
"""
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image


# ── BEGIN verbatim copy from HYPIR/utils/common.py:32-80 ──────────────────────
# (self-contained — only torch.nn.functional; kept inline so this script has no
#  HYPIR import dependency and runs in any torch env. Do not "fix" the kernel
#  layout / dilation — it implements the multi-scale Gaussian wavelet used by
#  HYPIR/SUPIR/CCSR; matching it byte-for-byte guarantees the same fusion as
#  HYPIR's own inference-time wavelet_reconstruction.)
def wavelet_blur(image, radius):
    """
    Apply wavelet blur to the input tensor.
    """
    # input shape: (1, 3, H, W)
    # convolution kernel
    kernel_vals = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device)
    # add channel dimensions to the kernel to make it a 4D tensor
    kernel = kernel[None, None]
    # repeat the kernel across all input channels
    kernel = kernel.repeat(3, 1, 1, 1)
    image = torch.nn.functional.pad(image, (radius, radius, radius, radius), mode='replicate')
    # apply convolution
    output = torch.nn.functional.conv2d(image, kernel, groups=3, dilation=radius)
    return output


def wavelet_decomposition(image, levels=5):
    """
    Apply wavelet decomposition to the input tensor.
    This function only returns the low frequency & the high frequency.
    """
    high_freq = torch.zeros_like(image)
    for i in range(levels):
        radius = 2 ** i
        low_freq = wavelet_blur(image, radius)
        high_freq += (image - low_freq)
        image = low_freq

    return high_freq, low_freq


def wavelet_reconstruction(content_feat, style_feat):
    """
    Apply wavelet decomposition, so that the content will have the same color as the style.
    """
    # calculate the wavelet decomposition of the content feature
    content_high_freq, content_low_freq = wavelet_decomposition(content_feat)
    del content_low_freq
    # calculate the wavelet decomposition of the style feature
    style_high_freq, style_low_freq = wavelet_decomposition(style_feat)
    del style_high_freq
    # reconstruct the content feature with the style's high frequency
    return content_high_freq + style_low_freq
# ── END verbatim copy (HYPIR/utils/common.py) ─────────────────────────────────


BEAUTY_DIR = os.environ.get("BEAUTY_DIR")
HQ_ORIG_NAME = os.environ.get("HQ_ORIG_NAME", "hq_orig")
HQ_BEAUTY_NAME = os.environ.get("HQ_BEAUTY_NAME", "hq_beauty")
HQ_DECOLOR_NAME = os.environ.get("HQ_DECOLOR_NAME", "hq_beauty_decolor")
DEVICE = os.environ.get("DEVICE", "cuda")
SAVE_COMPARE = os.environ.get("SAVE_COMPARE", "0") == "1"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif", ".ppm"}

# 03d stores hq_orig/hq_beauty via save_image(normalize=True, value_range=(-1,1)),
# which maps [-1,1] -> [0,255] on disk — i.e. a standard sRGB PNG. Reading it back
# with ToTensor() gives [0,1]. Wavelet ops are linear (conv/pad/subtract), so
# doing them in [0,1] is equivalent to doing them in [-1,1] (any constant offset
# lands in the low-frequency term). Output is clamped to [0,1] and saved as a
# standard PNG — byte-for-byte compatible with 03d's hq_orig/hq_beauty format, so
# 03b's build_paired_dataset.py pairs them by filename without any change.
tfm = transforms.Compose([transforms.ToTensor()])


def main():
    if not BEAUTY_DIR:
        sys.exit("ERROR: set BEAUTY_DIR (03d output root, contains hq_orig/ + hq_beauty/).")

    beauty_root = Path(BEAUTY_DIR)
    orig_dir = beauty_root / HQ_ORIG_NAME
    beauty_in_dir = beauty_root / HQ_BEAUTY_NAME
    decolor_dir = beauty_root / HQ_DECOLOR_NAME
    compare_dir = beauty_root / "compare_decolor" if SAVE_COMPARE else None

    for d in (orig_dir, beauty_in_dir):
        if not d.is_dir():
            sys.exit(f"ERROR: {d} not found (run 03d_build_beauty_dataset first to produce "
                     f"{HQ_ORIG_NAME}/ + {HQ_BEAUTY_NAME}/).")
    decolor_dir.mkdir(parents=True, exist_ok=True)
    if compare_dir:
        compare_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(DEVICE if (DEVICE == "cuda" and torch.cuda.is_available()) else "cpu")
    if DEVICE == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available — falling back to CPU (still fast; wavelet is "
              "a few conv2d on 512x512).", file=sys.stderr)

    # Collect hq_orig images as the pairing basis (same names => aligned, by 03d construction).
    images = []
    for root, _, files in os.walk(orig_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                images.append(Path(root) / f)
    images.sort(key=lambda x: str(x.relative_to(orig_dir)))
    if not images:
        sys.exit(f"ERROR: no images in {orig_dir}")
    print(f"[*] {len(images)} image(s):")
    print(f"    orig   (style,  low-freq): {orig_dir}")
    print(f"    beauty (content, high-freq): {beauty_in_dir}")
    print(f"    -> decolor               : {decolor_dir}")
    print(f"[*] params: device={device} save_compare={SAVE_COMPARE} "
          f"(wavelet levels=5, fixed by HYPIR copy)")

    times = []
    ok = 0
    miss = 0
    t0 = time.time()
    with torch.no_grad():
        for i, orig_fp in enumerate(images, 1):
            rel = orig_fp.relative_to(orig_dir)
            stem = rel.with_suffix(".png")
            beauty_fp = beauty_in_dir / orig_fp.name   # same filename => aligned
            out_fp = decolor_dir / stem
            if not beauty_fp.exists():
                print(f"[{i}/{len(images)}] {orig_fp.name}  ! beauty missing, skipped: {beauty_fp}",
                      file=sys.stderr)
                miss += 1
                continue
            try:
                out_fp.parent.mkdir(parents=True, exist_ok=True)
                orig = tfm(Image.open(orig_fp).convert("RGB")).unsqueeze(0).to(device)
                beauty = tfm(Image.open(beauty_fp).convert("RGB")).unsqueeze(0).to(device)
                if orig.shape != beauty.shape:
                    print(f"[{i}/{len(images)}] {orig_fp.name}  ! shape mismatch "
                          f"{tuple(orig.shape)} vs {tuple(beauty.shape)}, skipped", file=sys.stderr)
                    miss += 1
                    continue

                t1 = time.time()
                # content=beauty (take its HIGH-freq: smoothing/blemish-removal)
                # style=orig   (take its LOW-freq: true skin tone, kills the reddish bias)
                decolor = wavelet_reconstruction(beauty, orig)
                dt = time.time() - t1
                decolor = decolor.clamp(0, 1)

                save_image(decolor, str(out_fp))           # [0,1] -> standard PNG

                if SAVE_COMPARE:
                    # [orig | beauty(reddish) | decolor] — visual check of de-colorization
                    cmp = torch.cat([orig, beauty, decolor], dim=3)
                    cmp_fp = compare_dir / stem
                    cmp_fp.parent.mkdir(parents=True, exist_ok=True)
                    save_image(cmp, str(cmp_fp))

                times.append(dt)
                ok += 1
                if i <= 3 or i % 50 == 0:
                    print(f"[{i}/{len(images)}] {orig_fp.name}  ->  {stem.as_posix()} | {dt:.3f}s")
            except Exception as e:
                print(f"[{i}/{len(images)}] {orig_fp.name}  ! failed (skipped): {e}",
                      file=sys.stderr)
                # remove half-written output so a later 03b pairing doesn't see a dangling file
                try:
                    if out_fp.exists():
                        out_fp.unlink()
                except Exception:
                    pass

    loop = time.time() - t0
    pure = sum(times)
    print(f"[*] done. {ok}/{len(images)} succeeded, {miss} missing/skipped. "
          f"loop {loop:.2f}s (fusion {pure:.2f}s)")
    if times:
        print(f"[*] per-image: avg {pure / len(times):.3f}s, "
              f"min {min(times):.3f}s, max {max(times):.3f}s, n={len(times)}")
    print(f"[*] hq_beauty_decolor: {decolor_dir}")
    if SAVE_COMPARE:
        print(f"[*] compare (orig|beauty|decolor): {compare_dir}")


if __name__ == "__main__":
    main()
