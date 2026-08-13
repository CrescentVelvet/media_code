#!/usr/bin/env python3
"""Segment the person in EVERY image of an orbit shoot onto a white background.

Unlike wan22_rotate/pick_and_segment.py (which picks ONE best front-facing image),
this keeps ALL views — PDF-GS needs a multi-view set to triangulate the body and
filter micro-motion distractors. Lighter than pick_and_segment.py: no SAM 3D Body
/ detectron2 / GATED weights — just SAM2 automatic mask generation (largest salient
mask = person) with a rembg fallback. Runs fully inside the `pdfgs` env.

Pipeline per image:
  1. SAM2 SAM2AutomaticMaskGenerator -> list of masks; pick largest by area.
     (Falls back to rembg `remove` -> alpha channel if SAM2 unavailable/fails.)
  2. Apply mask: person pixels kept, background -> white (255,255,255).
  3. Save as <stem>.png under OUTPUT_DIR, preserving the input rel subpath.

Env vars (set by 01_segment_all.sh):
  INPUT_DIR, OUTPUT_DIR, SAM2_CHECKPOINT, SAM2_CONFIG, DEVICE,
  WHITE_BG, MIN_MASK_FRAC, JPG_QUALITY, SEGMENTOR
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import cv2

INPUT_DIR = os.environ.get("INPUT_DIR", "")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "")
SAM2_CHECKPOINT = os.environ.get("SAM2_CHECKPOINT", "")
SAM2_CONFIG = os.environ.get("SAM2_CONFIG", "configs/sam2.1/sam2.1_hiera_large.yaml")
DEVICE = os.environ.get("DEVICE", "cuda")
WHITE_BG = os.environ.get("WHITE_BG", "1") == "1"
MIN_MASK_FRAC = float(os.environ.get("MIN_MASK_FRAC", "0.02"))
JPG_QUALITY = int(os.environ.get("JPG_QUALITY", "95"))
SEGMENTOR = os.environ.get("SEGMENTOR", "auto")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

_sam2_mask_gen = None
_sam2_disabled = False
_rembg_session = None


def _device():
    if DEVICE == "cuda" and not _torch_cuda_available():
        print("⚠️ CUDA not available -- falling back to CPU (will be slow).")
        return "cpu"
    return DEVICE


def _torch_cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _load_sam2():
    global _sam2_mask_gen, _sam2_disabled
    if _sam2_mask_gen is not None or _sam2_disabled:
        return _sam2_mask_gen
    if not SAM2_CHECKPOINT or not os.path.isfile(SAM2_CHECKPOINT):
        print("⚠️ SAM2 checkpoint not found, disabling SAM2 path.")
        _sam2_disabled = True
        return None
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_automatic_mask_generator import SAM2AutomaticMaskGenerator
    except Exception as e:
        print(f"⚠️ sam2 import failed: {e} — disabling SAM2 path.")
        _sam2_disabled = True
        return None
    device = _device()
    sam2 = None
    for cfg in (SAM2_CONFIG, _abs_config(SAM2_CONFIG)):
        try:
            sam2 = build_sam2(cfg, SAM2_CHECKPOINT, device=device)
            print(f"🏋️ SAM2 loaded (cfg={cfg}, device={device})")
            break
        except Exception as e:
            last = e
    if sam2 is None:
        print(f"⚠️ build_sam2 failed: {last} — disabling SAM2 path.")
        _sam2_disabled = True
        return None
    try:
        # Use SAM2 defaults (points_per_side / IoU / stability thresholds). The
        # post-hoc MIN_MASK_FRAC check below handles "largest mask too small".
        _sam2_mask_gen = SAM2AutomaticMaskGenerator(sam2)
    except Exception as e:
        print(f"⚠️ SAM2AutomaticMaskGenerator init failed: {e}")
        _sam2_disabled = True
        return None
    return _sam2_mask_gen


def _abs_config(rel):
    # rel already includes "configs/..." (e.g. "configs/sam2.1/sam2.1_hiera_large.yaml");
    # the sam2 repo layout is <SAM2_DIR>/sam2/configs/... , so abs = <SAM2_DIR>/sam2/<rel>.
    sam2_dir = os.environ.get("SAM2_DIR", "")
    return str(Path(sam2_dir) / "sam2" / rel)


def segment_sam2(image_bgr):
    gen = _load_sam2()
    if gen is None:
        return None
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    try:
        masks = gen.generate(rgb)
    except Exception as e:
        print(f"    ⚠️ SAM2 generate failed: {e}")
        return None
    if not masks:
        return None
    masks.sort(key=lambda m: m.get("area", 0), reverse=True)
    best = masks[0]
    seg = best["segmentation"]
    if seg.ndim == 3:
        seg = seg[..., 0]
    mask = seg.astype(np.uint8)
    frac = mask.sum() / (mask.shape[0] * mask.shape[1])
    if frac < MIN_MASK_FRAC:
        print(f"    ⚠️ SAM2 largest mask only {frac*100:.1f}% of image (< {MIN_MASK_FRAC*100:.0f}%)")
        return None
    return mask


def _load_rembg():
    global _rembg_session
    if _rembg_session is not None:
        return _rembg_session
    try:
        from rembg import new_session
    except Exception as e:
        print(f"⚠️ rembg import failed: {e}")
        return None
    try:
        _rembg_session = new_session("u2net")
        print("🏋️ rembg session loaded (u2net)")
    except Exception as e:
        print(f"⚠️ rembg session failed (model download blocked?): {e}")
        return None
    return _rembg_session


def segment_rembg(image_bgr):
    session = _load_rembg()
    if session is None:
        return None
    try:
        from rembg import remove
        from PIL import Image
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        out = remove(Image.fromarray(rgb), session=session)
        arr = np.array(out)
    except Exception as e:
        print(f"    ⚠️ rembg remove failed: {e}")
        return None
    if arr.ndim != 3 or arr.shape[-1] < 4:
        return None
    alpha = arr[..., 3]
    mask = (alpha > 16).astype(np.uint8)
    frac = mask.sum() / (mask.shape[0] * mask.shape[1])
    if frac < MIN_MASK_FRAC:
        print(f"    ⚠️ rembg mask only {frac*100:.1f}% of image (< {MIN_MASK_FRAC*100:.0f}%)")
        return None
    return mask


def apply_mask(image_bgr, mask, white_bg=True):
    result = image_bgr.copy()
    bg = mask == 0
    result[bg] = (255, 255, 255) if white_bg else (0, 0, 0)
    return result


def collect_images():
    if not INPUT_DIR:
        sys.exit("❌ INPUT_DIR not set")
    root = Path(INPUT_DIR)
    if (root / "image").is_dir():
        root = root / "image"
    if not root.is_dir():
        sys.exit(f"❌ image dir not found: {root}")
    images = []
    for r, _, files in os.walk(root):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                images.append(Path(r) / f)
    images.sort(key=lambda x: str(x.relative_to(root)))
    return root, images


def main():
    if not OUTPUT_DIR:
        sys.exit("❌ OUTPUT_DIR not set")
    root, images = collect_images()
    if not images:
        sys.exit(f"❌ no images in {root}")
    print(f"🖼️ {len(images)} images in {root}")
    print(f"  💾 output: {OUTPUT_DIR}")
    print(f"  ✂️ segmentor: {SEGMENTOR}  white_bg: {WHITE_BG}  min_mask: {MIN_MASK_FRAC*100:.0f}%")

    use_sam2 = SEGMENTOR in ("auto", "sam2")
    use_rembg = SEGMENTOR in ("auto", "rembg")
    if use_sam2:
        _load_sam2()
    if use_rembg and _sam2_disabled:
        _load_rembg()

    out_root = Path(OUTPUT_DIR)
    n_ok = n_fail = 0
    t0 = time.time()
    for i, fp in enumerate(images, 1):
        rel = fp.relative_to(root)
        t1 = time.time()
        bgr = cv2.imread(str(fp))
        if bgr is None:
            print(f"[{i}/{len(images)}] {rel}  ❌ unreadable")
            n_fail += 1
            continue
        h, w = bgr.shape[:2]

        mask = None
        method = None
        if use_sam2 and not _sam2_disabled:
            mask = segment_sam2(bgr)
            method = "sam2"
        if mask is None and use_rembg:
            mask = segment_rembg(bgr)
            method = "rembg"

        if mask is None:
            print(f"[{i}/{len(images)}] {rel}  ❌ no mask ({time.time()-t1:.1f}s)")
            n_fail += 1
            continue

        out_img = apply_mask(bgr, mask, white_bg=WHITE_BG)
        out_path = out_root / rel.with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), out_img, [cv2.IMWRITE_PNG_COMPRESSION, 6]
                    if out_path.suffix.lower() == ".png"
                    else [cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY])
        frac = mask.sum() / (h * w)
        print(f"[{i}/{len(images)}] {rel}  ✅ {method} {frac*100:.0f}%  ({time.time()-t1:.1f}s)")
        n_ok += 1

    print(f"\n⏱️ {n_ok}/{len(images)} segmented, {n_fail} failed in {time.time()-t0:.1f}s")
    if n_ok == 0:
        sys.exit("❌ no image was segmented. Check SAM2/rembg setup (INSTALL_DEPS=1 bash 00_setup_env.sh).")
    print(f"📁 saved: {out_root}/*.png")
    print(f"\n下一步:")
    print(f"  INPUT={out_root} bash {Path(__file__).resolve().parent}/02_pi3_colmap.sh")


if __name__ == "__main__":
    main()
