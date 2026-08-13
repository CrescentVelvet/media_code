#!/usr/bin/env python3
"""Segment the person in EVERY image of an orbit shoot onto a white background.

Unlike wan22_rotate/pick_and_segment.py (which picks ONE best front-facing image),
this keeps ALL views — PDF-GS needs a multi-view set to triangulate the body and
filter micro-motion distractors. Self-contained in the `pdfgs` env: no SAM 3D Body
/ detectron2 / GATED weights.

Segmentation strategy (SEGMENTOR=auto, the default) mirrors
wan22_rotate/pick_and_segment_mediapipe.py's "person bbox -> SAM2" approach, but
uses rembg as the lightweight person LOCATOR (ViTDet-equivalent) since this env
has no detectron2:

Pipeline per image:
  1. rembg coarse mask -> tight person bbox (lightweight locator).
  2. SAM2 image predictor prompted with that bbox -> crisp person-only mask.
     The bbox prompt constrains SAM2 to the person, so edges are crisp with no
     dark halo (this is the technique that keeps wan22_rotate's output clean).
     Fallbacks: SAM2 automatic mask generation (largest salient mask), then
     rembg direct (alpha channel).
  3. Apply mask: person pixels kept, background -> white (255,255,255).
  4. Save as <stem>.png under OUTPUT_DIR, preserving the input rel subpath.

Black-border note: the old rembg path used alpha>16, which kept the soft
semi-transparent edge band (mostly dark background) -> a dark halo around the
person on a white bg. REMBG_ALPHA_THRESH now defaults to 128 (excludes that
band), and the bbox-prompted SAM2 path produces clean edges by construction.

Env vars (set by 01_segment_all.sh):
  INPUT_DIR, OUTPUT_DIR, SAM2_CHECKPOINT, SAM2_CONFIG, DEVICE,
  WHITE_BG, MIN_MASK_FRAC, JPG_QUALITY, SEGMENTOR,
  REMBG_ALPHA_THRESH, BBOX_PADDING
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
# rembg alpha threshold: 16 (old) kept the soft semi-transparent edge band
# (mostly dark background) -> dark halo / black border on white bg. 128 excludes
# that band, keeping only clearly-opaque (definitely-person) pixels.
REMBG_ALPHA_THRESH = int(os.environ.get("REMBG_ALPHA_THRESH", "128"))
# padding around the rembg-derived person bbox before prompting SAM2 (fraction of
# bbox w/h). Ensures hair/limb extremities sit inside the prompt box so SAM2
# segments the full person (compensates for a tight rembg mask at low alpha).
BBOX_PADDING = float(os.environ.get("BBOX_PADDING", "0.05"))

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

_sam2_model = None       # built once, shared by automatic-gen and predictor
_sam2_mask_gen = None    # SAM2AutomaticMaskGenerator (fallback: no bbox prompt)
_sam2_predictor = None   # SAM2ImagePredictor (primary: bbox-prompted, like wan22)
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


def _build_sam2_model():
    """Build the SAM2 model once (shared by the automatic-gen and predictor paths)."""
    global _sam2_model, _sam2_disabled
    if _sam2_model is not None or _sam2_disabled:
        return _sam2_model
    if not SAM2_CHECKPOINT or not os.path.isfile(SAM2_CHECKPOINT):
        print("⚠️ SAM2 checkpoint not found, disabling SAM2 path.")
        _sam2_disabled = True
        return None
    try:
        from sam2.build_sam import build_sam2
    except Exception as e:
        print(f"⚠️ sam2 import failed: {e} — disabling SAM2 path.")
        _sam2_disabled = True
        return None
    device = _device()
    sam2 = None
    last = None
    for cfg in (SAM2_CONFIG, _abs_config(SAM2_CONFIG)):
        try:
            sam2 = build_sam2(cfg, SAM2_CHECKPOINT, device=device)
            print(f"🏋️ SAM2 model loaded (cfg={cfg}, device={device})")
            break
        except Exception as e:
            last = e
    if sam2 is None:
        print(f"⚠️ build_sam2 failed: {last} — disabling SAM2 path.")
        _sam2_disabled = True
        return None
    _sam2_model = sam2
    return _sam2_model


def _load_sam2_automatic():
    """SAM2AutomaticMaskGenerator — fallback when no person bbox is available.

    No prompt; picks the largest salient mask. Edges can be slightly generous
    vs the bbox-prompted path, so prefer _load_sam2_predictor() for clean output.
    """
    global _sam2_mask_gen
    if _sam2_mask_gen is not None:
        return _sam2_mask_gen
    sam2 = _build_sam2_model()
    if sam2 is None:
        return None
    try:
        from sam2.sam2_automatic_mask_generator import SAM2AutomaticMaskGenerator
        # Use SAM2 defaults (points_per_side / IoU / stability thresholds). The
        # post-hoc MIN_MASK_FRAC check below handles "largest mask too small".
        _sam2_mask_gen = SAM2AutomaticMaskGenerator(sam2)
        print("🏋️ SAM2 automatic mask generator loaded (fallback path)")
    except Exception as e:
        print(f"⚠️ SAM2AutomaticMaskGenerator init failed: {e}")
        return None
    return _sam2_mask_gen


def _load_sam2_predictor():
    """SAM2ImagePredictor — bbox-prompted, mirrors wan22_rotate's HumanSegmentor.

    wan22_rotate does ViTDet -> person bbox -> human_segmentor.run_sam(rgb, bboxes)
    (SAM2 prompted with a box). That yields a crisp, person-only mask with no
    dark halo. This env has no detectron2/ViTDet, so rembg supplies the bbox
    instead (see segment_sam2_bbox). The SAM2 side is identical: prompt with a
    box, get a tight instance mask.
    """
    global _sam2_predictor
    if _sam2_predictor is not None:
        return _sam2_predictor
    sam2 = _build_sam2_model()
    if sam2 is None:
        return None
    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        _sam2_predictor = SAM2ImagePredictor(sam2)
        print("🏋️ SAM2 image predictor loaded (bbox-prompted, primary path)")
    except Exception as e:
        print(f"⚠️ SAM2ImagePredictor init failed: {e} — will use automatic/rembg")
        return None
    return _sam2_predictor


def _abs_config(rel):
    # rel already includes "configs/..." (e.g. "configs/sam2.1/sam2.1_hiera_large.yaml");
    # the sam2 repo layout is <SAM2_DIR>/sam2/configs/... , so abs = <SAM2_DIR>/sam2/<rel>.
    sam2_dir = os.environ.get("SAM2_DIR", "")
    return str(Path(sam2_dir) / "sam2" / rel)


def segment_sam2(image_bgr):
    """Fallback: SAM2 automatic mask generation (no bbox prompt). Picks the
    largest salient region. Kept for SEGMENTOR=sam2 and as a 2nd fallback in auto."""
    gen = _load_sam2_automatic()
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


def _mask_to_bbox(mask, padding=0.05):
    """Tight [x1,y1,x2,y2] bbox from a binary mask, padded outwards so SAM2's
    prompt box comfortably contains the whole person (incl. hair/limbs)."""
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    h, w = mask.shape
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * padding))
    y1 = max(0, int(y1 - bh * padding))
    x2 = min(w, int(x2 + bw * padding))
    y2 = min(h, int(y2 + bh * padding))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def segment_sam2_bbox(image_bgr):
    """Primary path — mirrors wan22_rotate/pick_and_segment_mediapipe.py's
    "person bbox -> SAM2" segmentation, which produces clean edges (no black
    border). Since this env has no ViTDet/detectron2, rembg supplies the coarse
    person mask -> bbox (lightweight locator), then SAM2's image predictor is
    prompted with that box for a crisp person-only mask.

    Why this beats automatic generation / rembg direct: a bbox prompt tells SAM2
    exactly where the person is, so the mask boundary follows the true silhouette
    instead of a slightly-generous salient region or rembg's soft alpha band
    (the soft band is what shows up as a dark halo on white backgrounds).
    """
    predictor = _load_sam2_predictor()
    if predictor is None:
        return None
    # 1. rembg -> coarse person mask -> tight bbox (the ViTDet-equivalent locator).
    coarse = segment_rembg(image_bgr)
    if coarse is None:
        return None
    bbox = _mask_to_bbox(coarse, padding=BBOX_PADDING)
    if bbox is None:
        return None
    # 2. SAM2 image predictor prompted with the bbox -> crisp person mask.
    #    Try both box shapes ((4,) for SAM-v1, (1,4) per SAM2 docs) — mirrors the
    #    defensive multi-call style in wan22_rotate/pick_and_segment_mediapipe.py.
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    masks = None
    last_err = None
    try:
        predictor.set_image(rgb)
    except Exception as e:
        print(f"    ⚠️ SAM2 predictor.set_image failed: {e}")
        return None
    for box_arg in (bbox, bbox[None, :]):
        try:
            masks, scores, _ = predictor.predict(box=box_arg, multimask_output=False)
            break
        except Exception as e:
            last_err = e
    if masks is None:
        print(f"    ⚠️ SAM2 predictor.predict failed: {last_err}")
        return None
    m_arr = np.asarray(masks)
    if hasattr(m_arr, "cpu"):
        m_arr = m_arr.cpu().numpy()
    while m_arr.ndim > 2:        # collapse leading batch/multimask dims -> (H, W)
        m_arr = m_arr[0]
    mask = (m_arr > 0.5).astype(np.uint8)
    frac = mask.sum() / (mask.shape[0] * mask.shape[1])
    if frac < MIN_MASK_FRAC:
        print(f"    ⚠️ SAM2 bbox mask only {frac*100:.1f}% of image (< {MIN_MASK_FRAC*100:.0f}%)")
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
    # rembg reads the model from $U2NET_HOME (set by _env.sh -> $MODEL_DIR/u2net).
    # If u2net.onnx is missing there, rembg tries to download (blocked by corporate
    # proxy). Pre-check and point the user at where to put it.
    _u2home = os.environ.get("U2NET_HOME", os.path.expanduser("~/.u2net"))
    if not os.path.isfile(os.path.join(_u2home, "u2net.onnx")):
        print(f"⚠️ rembg u2net.onnx not found at {_u2home}/u2net.onnx")
        print(f"   Put it there (or set U2NET_HOME); rembg will otherwise try to download (proxy blocks it).")
    try:
        _rembg_session = new_session("u2net")
        print("🏋️ rembg session loaded (u2net)")
    except Exception as e:
        print(f"⚠️ rembg session failed (model download blocked?): {e}")
        print(f"   Expected u2net.onnx at {_u2home}/u2net.onnx")
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
    # 16 (old) kept the soft semi-transparent edge band -> dark halo / black
    # border on white bg. 128 keeps only clearly-opaque (definitely-person)
    # pixels, dropping that band.
    mask = (alpha > REMBG_ALPHA_THRESH).astype(np.uint8)
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
    print(f"  🎯 primary: SAM2 bbox-prompted (rembg locates) — mirrors wan22_rotate")

    # --- one-time model pre-load so per-image cost is just inference ---
    # auto (default): rembg -> bbox -> SAM2 predictor (clean edges) -> SAM2
    #   automatic -> rembg direct. sam2: automatic only (no rembg locator).
    #   rembg: rembg direct only.
    if SEGMENTOR in ("auto", "sam2"):
        _build_sam2_model()
    if SEGMENTOR == "auto":
        _load_sam2_predictor()   # primary: bbox-prompted (needs rembg to locate)
        _load_rembg()
    elif SEGMENTOR == "sam2":
        _load_sam2_automatic()   # sam2-only (no rembg locator) -> automatic gen
    elif SEGMENTOR == "rembg":
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
        # auto: bbox-prompted SAM2 (best) -> automatic SAM2 -> rembg direct.
        if SEGMENTOR == "auto":
            mask = segment_sam2_bbox(bgr); method = "sam2_bbox"
            if mask is None and not _sam2_disabled:
                mask = segment_sam2(bgr); method = "sam2_auto"
            if mask is None:
                mask = segment_rembg(bgr); method = "rembg"
        elif SEGMENTOR == "sam2":
            if not _sam2_disabled:
                mask = segment_sam2(bgr); method = "sam2_auto"
        elif SEGMENTOR == "rembg":
            mask = segment_rembg(bgr); method = "rembg"

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
