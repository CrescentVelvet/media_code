#!/usr/bin/env python3
"""dynamic_mask.py — Generate per-frame dynamic object masks via text-prompted segmentation.

Uses GroundingDINO (text→boxes) + SAM2/SAM (boxes→masks) to detect dynamic objects
(persons, TV screens, etc.) in training images. Outputs per-image binary masks
(H×W uint8, 1=dynamic, 0=static) as initial priors for dynamic interference removal.

Segmenter: SAM2 (sam2 package, primary) → SAM (transformers, fallback).
Detector:  GroundingDINO (transformers, always needed for text prompts).
If neither segmenter available → sys.exit with install hint.
If GroundingDINO unavailable → sys.exit with install hint.

Env vars:
  SAM2_MODEL_PATH    : SAM2 checkpoint dir or .pt file (default: from _env.sh)
  SAM2_CONFIG        : SAM2 config yaml (default: configs/sam2.1/sam2.1_hiera_large.yaml)
  SAM2_DIR           : sam2 repo dir for config path resolution (default: ../sam2)
  SAM_MODEL_ID       : SAM (fallback) HuggingFace model ID (default: facebook/sam-vit-base)
  GROUNDING_DINO_ID  : GroundingDINO HF model ID (default: IDEA-Research/grounding-dino-tiny)
  DEVICE             : cuda | cpu (default: cuda)
  BOX_THRESHOLD      : GroundingDINO box score threshold (default: 0.3)
  TEXT_THRESHOLD     : GroundingDINO text score threshold (default: 0.25)

CLI:
  python dynamic_mask.py --images_dir <dir> --output_dir <dir> [--prompts person "TV screen"]
"""
import os
import sys
import glob
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SAM2_MODEL_PATH = os.environ.get("SAM2_MODEL_PATH", "")
SAM2_CONFIG = os.environ.get("SAM2_CONFIG", "configs/sam2.1/sam2.1_hiera_large.yaml")
SAM2_DIR = os.environ.get("SAM2_DIR", "")
SAM_MODEL_ID = os.environ.get("SAM_MODEL_ID", "facebook/sam-vit-base")
GROUNDING_DINO_ID = os.environ.get("GROUNDING_DINO_ID", "IDEA-Research/grounding-dino-tiny")
DEVICE = os.environ.get("DEVICE", "cuda")
BOX_THRESHOLD = float(os.environ.get("BOX_THRESHOLD", "0.3"))
TEXT_THRESHOLD = float(os.environ.get("TEXT_THRESHOLD", "0.25"))

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif"}

# Lazy-loaded singletons
_grounded_dino = None      # (model, processor) or None
_sam2_predictor = None     # SAM2ImagePredictor or None
_sam_model = None          # (model, processor) for SAM transformers, or None
_segmenter_type = None     # "sam2" | "sam"


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------
def _device():
    if DEVICE == "cuda":
        if not torch.cuda.is_available():
            print("⚠️ CUDA not available — falling back to CPU (will be slow).")
            return "cpu"
    return DEVICE


# ---------------------------------------------------------------------------
# SAM2 checkpoint resolution
# ---------------------------------------------------------------------------
def _find_sam2_checkpoint():
    """Resolve SAM2 checkpoint .pt file from SAM2_MODEL_PATH."""
    if not SAM2_MODEL_PATH:
        return ""
    if os.path.isfile(SAM2_MODEL_PATH):
        return SAM2_MODEL_PATH
    if os.path.isdir(SAM2_MODEL_PATH):
        # Look for known SAM2 checkpoint files
        for name in (
            "sam2.1_hiera_large.pt",
            "sam2.1_hiera_base_plus.pt",
            "sam2.1_hiera_small.pt",
            "sam2.1_hiera_tiny.pt",
            "sam2_hiera_large.pt",
            "sam2_hiera_base_plus.pt",
            "sam2_hiera_small.pt",
            "sam2_hiera_tiny.pt",
        ):
            p = os.path.join(SAM2_MODEL_PATH, name)
            if os.path.isfile(p):
                return p
        # Glob fallback: any .pt in the dir
        pts = sorted(glob.glob(os.path.join(SAM2_MODEL_PATH, "*.pt")))
        if pts:
            return pts[0]
    return ""


def _abs_config(rel):
    """Resolve SAM2 config path relative to SAM2_DIR (mirrors segment_all.py)."""
    if os.path.isabs(rel) and os.path.isfile(rel):
        return rel
    sam2_dir = SAM2_DIR
    if sam2_dir:
        abs_path = str(Path(sam2_dir) / "sam2" / rel)
        if os.path.isfile(abs_path):
            return abs_path
    return rel


# ---------------------------------------------------------------------------
# Model loaders (lazy)
# ---------------------------------------------------------------------------
def _load_detector():
    """Load GroundingDINO for text→boxes detection. Returns (model, processor) or None."""
    global _grounded_dino
    if _grounded_dino is not None:
        return _grounded_dino
    try:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    except ImportError:
        print("⚠️ transformers not installed — GroundingDINO unavailable.")
        return None
    try:
        processor = AutoProcessor.from_pretrained(GROUNDING_DINO_ID)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_DINO_ID)
        model = model.to(_device())
        model.eval()
        _grounded_dino = (model, processor)
        print(f"🏋️ GroundingDINO loaded: {GROUNDING_DINO_ID} (device={_device()})")
    except Exception as e:
        print(f"⚠️ GroundingDINO load failed: {e}")
        _grounded_dino = None
    return _grounded_dino


def _load_segmenter():
    """Load SAM2 (primary) or SAM (fallback) for box→mask segmentation.

    Returns "sam2" | "sam" (the segmenter type), or sys.exit if neither available.
    """
    global _sam2_predictor, _sam_model, _segmenter_type
    if _segmenter_type is not None:
        return _segmenter_type

    # ── Primary: SAM2 (sam2 package, following segment_all.py pattern) ──
    ckpt = _find_sam2_checkpoint()
    if ckpt:
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            device = _device()
            sam2 = None
            last_err = None
            # checkpoint 名 → 官方 config 名映射（repo 里是缩写：large→l、base_plus→b+）
            cfg_candidates = [SAM2_CONFIG, _abs_config(SAM2_CONFIG)]
            ckpt_base = os.path.basename(ckpt).lower()
            for full, short in (("large", "l"), ("base_plus", "b+"),
                                ("small", "s"), ("tiny", "t")):
                if full in ckpt_base:
                    d = os.path.basename(SAM2_CONFIG).replace(full, short)
                    cfg_candidates += [f"configs/sam2.1/{d}", d,
                                       f"configs/sam2.1/{d}".replace("sam2.1", "sam2")]
                    break
            for cfg in dict.fromkeys(cfg_candidates):   # 去重保序
                try:
                    sam2 = build_sam2(cfg, ckpt, device=device)
                    print(f"🏋️ SAM2 loaded (ckpt={ckpt}, cfg={cfg}, device={device})")
                    break
                except Exception as e:
                    last_err = e
            if sam2 is not None:
                _sam2_predictor = SAM2ImagePredictor(sam2)
                _segmenter_type = "sam2"
                return _segmenter_type
            print(f"⚠️ build_sam2 failed: {last_err}")
        except ImportError:
            print("⚠️ sam2 package not installed — trying SAM (transformers) fallback.")
        except Exception as e:
            print(f"⚠️ SAM2 load failed: {e}")
    else:
        print("⚠️ SAM2 checkpoint not found — trying SAM (transformers) fallback.")

    # ── Fallback: original SAM (transformers) ──
    try:
        from transformers import SamModel, SamProcessor
        model = SamModel.from_pretrained(SAM_MODEL_ID).to(_device())
        processor = SamProcessor.from_pretrained(SAM_MODEL_ID)
        _sam_model = (model, processor)
        _segmenter_type = "sam"
        print(f"🏋️ SAM (fallback) loaded: {SAM_MODEL_ID} (device={_device()})")
        return _segmenter_type
    except Exception as e:
        print(f"⚠️ SAM unavailable: {e}")

    sys.exit(
        "❌ No segmenter available (SAM2 or SAM). Install one of:\n"
        "  pip install git+https://github.com/facebookresearch/segment-anything-2.git\n"
        "  (and set SAM2_MODEL_PATH to the checkpoint dir)\n"
        "  OR: pip install transformers  # for SAM via HuggingFace"
    )


# ---------------------------------------------------------------------------
# Detection (text → boxes via GroundingDINO)
# ---------------------------------------------------------------------------
def _detect_boxes(pil_image, text_prompt):
    """GroundingDINO: text → list of [x1, y1, x2, y2] boxes (numpy float32)."""
    det = _load_detector()
    if det is None:
        sys.exit(
            "❌ GroundingDINO not available. Install:\n"
            "  pip install transformers  # for GroundingDINO + SAM fallback"
        )
    model, processor = det
    device = _device()
    W, H = pil_image.size  # PIL: (W, H)

    # GroundingDINO requires a period at the end of the text prompt
    text = text_prompt.strip()
    if not text.endswith("."):
        text += " ."

    inputs = processor(images=pil_image, text=text, return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process (API varies by transformers version: <4.46 用 box_threshold=，
    # >=4.46 改名 threshold=，参数名不对抛 TypeError 而非 AttributeError)
    results = None
    for kwargs in (
        dict(threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD),   # >=4.46
        dict(box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD),  # 旧版
        dict(threshold=BOX_THRESHOLD),                                   # 兜底
    ):
        try:
            results = processor.post_process_grounded_object_detection(
                outputs, inputs["input_ids"],
                target_sizes=[(H, W)],  # target_sizes expects (H, W)
                **kwargs,
            )
            break
        except TypeError:
            continue
    if results is None:
        sys.exit("❌ post_process_grounded_object_detection: 无兼容参数组合")

    if not results or len(results) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    boxes = results[0]["boxes"].cpu().numpy()  # (N, 4) xyxy
    return boxes


# ---------------------------------------------------------------------------
# Segmentation (boxes → masks via SAM2 or SAM)
# ---------------------------------------------------------------------------
def _segment_boxes_sam2(rgb_np, boxes):
    """SAM2: boxes → list of (H, W) bool arrays."""
    predictor = _sam2_predictor
    predictor.set_image(rgb_np)
    masks = []
    for box in boxes:
        # Try both (4,) and (1,4) shapes (mirrors segment_all.py defensive style)
        m = None
        for box_arg in (box, box[None, :]):
            try:
                m, scores, _ = predictor.predict(
                    box=box_arg, multimask_output=False
                )
                break
            except Exception:
                continue
        if m is None:
            print(f"  ⚠️ SAM2 predict failed for box {box.tolist()}, skipping")
            continue
        m_arr = np.asarray(m)
        if hasattr(m_arr, "cpu"):
            m_arr = m_arr.cpu().numpy()
        while m_arr.ndim > 2:
            m_arr = m_arr[0]
        masks.append(m_arr > 0.5)
    return masks


def _segment_boxes_sam(pil_image, boxes):
    """SAM (transformers): boxes → list of (H, W) bool arrays."""
    model, processor = _sam_model
    device = _device()
    # SAM (transformers) processor 期望 input_boxes=[[box, ...]]，box 为
    # [x1, y1, x2, y2] 四元 XYXY（官方示例格式），不是角点对。
    input_boxes = [[[float(v) for v in b] for b in boxes]]
    inputs = processor(pil_image, input_boxes=[input_boxes], return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )
    result = []
    if masks and len(masks) > 0:
        tensor = masks[0]  # (num_boxes, 1, H, W) or similar
        for i in range(tensor.shape[0]):
            m = tensor[i]
            while m.ndim > 2:
                m = m[0]
            result.append(m.numpy() > 0.5)
    return result


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------
def generate_dynamic_masks(images_dir, output_dir, prompts=None):
    """Generate per-image dynamic object masks via text-prompted segmentation.

    Args:
        images_dir: training image directory (loose images).
        output_dir: output root (debug visualizations saved to <output_dir>/dynamic_mask/).
        prompts: list of text prompts (default: ["person", "TV screen"]).

    Returns:
        dict[str, np.ndarray]: {image_name: union_mask(H, W) uint8}, 1=dynamic, 0=static.
    """
    if prompts is None:
        prompts = ["person", "TV screen"]

    # Find images
    images = sorted([f for f in os.listdir(images_dir)
                     if os.path.splitext(f)[1].lower() in IMG_EXTS])
    if not images:
        sys.exit(f"❌ no images in {images_dir}")

    print(f"🚀 generating dynamic masks for {len(images)} images")
    print(f"  📂 images: {images_dir}")
    print(f"  🎯 prompts: {prompts}")

    # Load models (lazy, builds once)
    seg_type = _load_segmenter()
    _load_detector()

    # Debug output dir
    debug_dir = os.path.join(output_dir, "dynamic_mask")
    os.makedirs(debug_dir, exist_ok=True)

    masks_dict = {}
    t0 = time.time()

    for i, name in enumerate(images):
        path = os.path.join(images_dir, name)
        pil = Image.open(path).convert("RGB")
        W, H = pil.size

        all_masks = []
        for prompt in prompts:
            boxes = _detect_boxes(pil, prompt)
            if len(boxes) == 0:
                continue
            if seg_type == "sam2":
                rgb_np = np.array(pil)  # (H, W, 3) RGB
                seg_masks = _segment_boxes_sam2(rgb_np, boxes)
            else:
                seg_masks = _segment_boxes_sam(pil, boxes)
            all_masks.extend(seg_masks)

        # Union all masks for this image
        if all_masks:
            union = np.any(np.stack(all_masks), axis=0).astype(np.uint8)
        else:
            union = np.zeros((H, W), dtype=np.uint8)

        masks_dict[name] = union

        # Debug: save mask×255 as PNG
        stem = os.path.splitext(name)[0]
        debug_path = os.path.join(debug_dir, stem + ".png")
        Image.fromarray(union * 255).save(debug_path)

        ratio = union.sum() / union.size * 100
        n_masks = len(all_masks)
        print(f"  [{i + 1}/{len(images)}] {name}: {ratio:.1f}% dynamic ({n_masks} masks)")

    elapsed = time.time() - t0
    total_dynamic = sum(m.sum() for m in masks_dict.values())
    total_pixels = sum(m.size for m in masks_dict.values())
    overall = total_dynamic / max(total_pixels, 1) * 100
    print(f"\n✅ {len(masks_dict)} masks in {elapsed:.1f}s. Overall: {overall:.1f}% dynamic")
    print(f"  debug masks: {debug_dir}/")

    return masks_dict


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate dynamic object masks via text-prompted segmentation.")
    parser.add_argument("--images_dir", required=True, help="training image directory")
    parser.add_argument("--output_dir", required=True, help="output root (debug masks saved here)")
    parser.add_argument("--prompts", nargs="*", default=["person", "TV screen"],
                        help="detection prompts (default: person 'TV screen')")
    args = parser.parse_args()

    masks = generate_dynamic_masks(args.images_dir, args.output_dir, args.prompts)
