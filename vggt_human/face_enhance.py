#!/usr/bin/env python3
"""face_enhance.py — MediaPipe face detection + HYPIR enhancement + gradient blending.

For each image in the input directory:
  1. MediaPipe BlazeFace detects face bounding boxes.
  2. Each bbox is enlarged by FACE_PADDING (default 0.2 = 20%).
  3. The crop is fed to the HYPIR model (SD2Enhancer with LoRA checkpoint)
     for face restoration/beautification (upscale=1, no super-resolution).
  4. The enhanced crop is blended back into the original image using a
     quadratic falloff mask (1 at center, 0 at edges) → seamless transition.
  5. The COLMAP sparse/ directory is copied unchanged (only images are enhanced).

Uses the hypir conda env (has diffusers/transformers/peft for HYPIR).

Env vars (set by 05_face_enhance.sh):
  HYPIR_DIR, HYPIR_BASE_MODEL, HYPIR_WEIGHT, INPUT_SOURCE_DIR, SOURCE_FACE_DIR,
  FACE_PADDING, UPSCALE, PATCH_SIZE, STRIDE, DEVICE
"""
import os
import sys
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image

HYPIR_DIR = os.environ.get("HYPIR_DIR", "../HYPIR")
BASE_MODEL_PATH = os.environ.get("HYPIR_BASE_MODEL", "")
WEIGHT_PATH = os.environ.get("HYPIR_WEIGHT", "")
INPUT_SOURCE_DIR = os.environ.get("INPUT_SOURCE_DIR", "")
SOURCE_FACE_DIR = os.environ.get("SOURCE_FACE_DIR", "")
FACE_PADDING = float(os.environ.get("FACE_PADDING", "0.2"))
UPSCALE = int(os.environ.get("UPSCALE", "1"))
PATCH_SIZE = int(os.environ.get("PATCH_SIZE", "512"))
STRIDE = int(os.environ.get("STRIDE", "256"))
DEVICE = os.environ.get("DEVICE", "cuda")
LORA_RANK = int(os.environ.get("LORA_RANK", "256"))
LORA_MODULES = os.environ.get(
    "LORA_MODULES",
    "to_k,to_q,to_v,to_out.0,conv,conv1,conv2,conv_shortcut,conv_out,proj_in,proj_out,ff.net.2,ff.net.0.proj",
).split(",")
MODEL_T = int(os.environ.get("MODEL_T", "200"))
COEFF_T = int(os.environ.get("COEFF_T", "200"))

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


# ---------------------------------------------------------------------------
# Feather mask: quadratic falloff (1 at center, 0 at edges).
# ---------------------------------------------------------------------------
def create_feather_mask(h, w):
    """Create a 2D mask with quadratic falloff from center to edges."""
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    y = np.arange(h, dtype=np.float32) - cy
    x = np.arange(w, dtype=np.float32) - cx
    yy, xx = np.meshgrid(y, x, indexing="ij")
    r = np.sqrt(xx ** 2 + yy ** 2)
    r_max = max(min(cx, cy), 1.0)
    return np.clip(1.0 - (r / r_max) ** 2, 0.0, 1.0)


# ---------------------------------------------------------------------------
# MediaPipe face detection.
# ---------------------------------------------------------------------------
def load_face_detector():
    import mediapipe as mp
    mp_face = mp.solutions.face_detection
    detector = mp_face.FaceDetection(
        model_selection=1,  # 1 = short-range (better for close-up faces)
        min_detection_confidence=0.3,
    )
    return detector


def detect_faces(detector, image_rgb):
    """Returns list of (x1, y1, x2, y2) in pixel coordinates."""
    import cv2
    h, w = image_rgb.shape[:2]
    results = detector.process(image_rgb)
    boxes = []
    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            x1 = max(int(bbox.xmin * w), 0)
            y1 = max(int(bbox.ymin * h), 0)
            x2 = min(int((bbox.xmin + bbox.width) * w), w)
            y2 = min(int((bbox.ymin + bbox.height) * h), h)
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
    return boxes


def enlarge_bbox(bbox, padding, img_w, img_h):
    """Enlarge bbox by padding fraction (e.g., 0.2 = 20%)."""
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    nw, nh = bw * (1 + padding), bh * (1 + padding)
    nx1 = max(int(cx - nw / 2), 0)
    ny1 = max(int(cy - nh / 2), 0)
    nx2 = min(int(cx + nw / 2), img_w)
    ny2 = min(int(cy + nh / 2), img_h)
    return nx1, ny1, nx2, ny2


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main():
    if not INPUT_SOURCE_DIR:
        sys.exit("❌ INPUT_SOURCE_DIR not set")
    if not SOURCE_FACE_DIR:
        sys.exit("❌ SOURCE_FACE_DIR not set")
    if not WEIGHT_PATH:
        sys.exit("❌ HYPIR_WEIGHT not set")
    if not os.path.isfile(WEIGHT_PATH):
        sys.exit(f"❌ HYPIR weight not found: {WEIGHT_PATH}")
    if not BASE_MODEL_PATH:
        sys.exit("❌ HYPIR_BASE_MODEL not set")
    if not os.path.isdir(BASE_MODEL_PATH):
        sys.exit(f"❌ HYPIR base model not found: {BASE_MODEL_PATH}")

    t0 = time.time()

    # ── 1. Load MediaPipe face detector ────────────────────────────────────
    print("🔍 [1/3] loading MediaPipe face detector...")
    detector = load_face_detector()
    print("  ✅ MediaPipe loaded")

    # ── 2. Load HYPIR model ──────────────────────────────────────────────────
    print("🏋️ [2/3] loading HYPIR model (SD2Enhancer + LoRA)...")
    sys.path.insert(0, HYPIR_DIR)
    from HYPIR.enhancer.sd2 import SD2Enhancer  # noqa: E402
    from accelerate.utils import set_seed  # noqa: E402
    from torchvision import transforms  # noqa: E402

    set_seed(231)
    model = SD2Enhancer(
        base_model_path=BASE_MODEL_PATH,
        weight_path=WEIGHT_PATH,
        lora_modules=LORA_MODULES,
        lora_rank=LORA_RANK,
        model_t=MODEL_T,
        coeff_t=COEFF_T,
        device=DEVICE,
    )
    model.init_models()
    print(f"  ✅ HYPIR loaded (weight={WEIGHT_PATH})")

    # ── 3. Process images ────────────────────────────────────────────────────
    input_images_dir = os.path.join(INPUT_SOURCE_DIR, "images")
    output_images_dir = os.path.join(SOURCE_FACE_DIR, "images")
    output_sparse_dir = os.path.join(SOURCE_FACE_DIR, "sparse", "0")
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_sparse_dir, exist_ok=True)

    images = sorted([f for f in os.listdir(input_images_dir)
                     if os.path.splitext(f)[1].lower() in IMG_EXTS])
    if not images:
        sys.exit(f"❌ no images in {input_images_dir}")

    print(f"🖼️ [3/3] enhancing faces in {len(images)} images...")
    to_tensor = transforms.ToTensor()

    total_faces = 0
    for i, name in enumerate(images, 1):
        img_path = os.path.join(input_images_dir, name)
        img_pil = Image.open(img_path).convert("RGB")
        img_np = np.array(img_pil)
        h, w = img_np.shape[:2]

        # Detect faces
        import cv2
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        # MediaPipe expects RGB input actually; let me check...
        # Actually MediaPipe's FaceDetection.process expects RGB.
        img_rgb_mp = img_np  # already RGB (PIL → numpy)
        boxes = detect_faces(detector, img_rgb_mp)

        if not boxes:
            # No face → copy as-is
            img_pil.save(os.path.join(output_images_dir, name))
            print(f"  [{i}/{len(images)}] {name} — no face, copy as-is")
            continue

        # Enhance each face
        enhanced = img_np.copy()
        face_count = 0
        for bbox in boxes:
            x1, y1, x2, y2 = enlarge_bbox(bbox, FACE_PADDING, w, h)
            crop_w, crop_h = x2 - x1, y2 - y1
            if crop_w < 32 or crop_h < 32:
                continue

            crop_pil = img_pil.crop((x1, y1, x2, y2))
            crop_tensor = to_tensor(crop_pil).unsqueeze(0)

            try:
                result = model.enhance(
                    lq=crop_tensor,
                    prompt="",
                    scale_by="factor",
                    upscale=UPSCALE,
                    patch_size=min(PATCH_SIZE, max(crop_w, crop_h)),
                    stride=min(STRIDE, max(crop_w, crop_h) // 2),
                    return_type="pil",
                )[0]
            except Exception as e:
                print(f"  [{i}] ⚠️ HYPIR failed on face: {e}", file=sys.stderr)
                continue

            # Resize back to crop size if HYPIR changed resolution
            if result.size != (crop_w, crop_h):
                result = result.resize((crop_w, crop_h), Image.LANCZOS)
            result_np = np.array(result).astype(np.float32)

            # Gradient blend: feathered mask
            mask = create_feather_mask(crop_h, crop_w)
            mask_3d = mask[..., np.newaxis]  # (H, W, 1)
            original_crop = img_np[y1:y2, x1:x2].astype(np.float32)
            blended = original_crop * (1 - mask_3d) + result_np * mask_3d
            enhanced[y1:y2, x1:x2] = blended.astype(np.uint8)
            face_count += 1

        total_faces += face_count
        Image.fromarray(enhanced).save(os.path.join(output_images_dir, name))
        print(f"  [{i}/{len(images)}] {name} — {face_count} face(s) enhanced")

    # ── 4. Copy COLMAP sparse/ (unchanged) ──────────────────────────────────
    input_sparse = os.path.join(INPUT_SOURCE_DIR, "sparse", "0")
    if os.path.isdir(input_sparse):
        for fname in os.listdir(input_sparse):
            shutil.copy(os.path.join(input_sparse, fname),
                        os.path.join(output_sparse_dir, fname))
        print(f"  ✅ copied sparse/0/ ({len(os.listdir(output_sparse_dir))} files)")

    # Free GPU
    del model
    import torch
    torch.cuda.empty_cache()

    print(f"\n🎉 Done. {total_faces} faces enhanced in {len(images)} images. "
          f"{time.time() - t0:.1f}s")
    print(f"  output: {SOURCE_FACE_DIR}")
    print(f"  Next: bash vggt_human/06_train_denoise.sh")


if __name__ == "__main__":
    main()
