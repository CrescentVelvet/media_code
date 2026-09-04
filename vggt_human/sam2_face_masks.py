#!/usr/bin/env python3
"""sam2_face_masks.py — SAM2 pixel-level face segmentation (replaces MediaPipe bbox).

Uses SAM2.1 hiera_large for pixel-accurate face masks, replacing the MediaPipe
bbox-based face_masks.py. SAM2 provides precise face contours (not rectangles),
which is critical for:
  - Face 3D center computation (point cloud projection filtering)
  - Closeup view coverage calculation (40% threshold)
  - HYPIR finetune face mask (more accurate than bbox)

Strategy:
  1. MediaPipe BlazeFace for initial face detection (fast, provides box prompt)
  2. SAM2 with box prompt → pixel-level mask (precise contour)
  3. Output: <stem>.alpha.png (feathered soft) + <stem>.mask.png (binary)
     Same naming as face_masks.py for drop-in replacement.

Env vars:
  IMAGES_DIR   : input images folder
  OUTPUT_DIR    : output masks folder
  SAM2_CKPT     : SAM2.1 checkpoint path
  SAM2_CFG      : SAM2 config name (default: configs/sam2.1/sam2.1_hiera_l.yaml)
  SOFT_FEATHER  : 1 = apply feather to alpha (default 1)
  ERODE_PX      : erosion pixels for binary mask (default 2)
  MIN_SCORE     : minimum SAM2 mask score to accept (default 0.5)
"""
import os
import sys
import time
import argparse

import numpy as np
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

SAM2_CKPT = os.environ.get("SAM2_CKPT", "/mnt/d/wheel/vggt_human_ms/sam2/sam2.1_hiera_large.pt")
SAM2_CFG = os.environ.get("SAM2_CFG", "configs/sam2.1/sam2.1_hiera_l.yaml")
SOFT_FEATHER = os.environ.get("SOFT_FEATHER", "1") == "1"
ERODE_PX = int(os.environ.get("ERODE_PX", "2"))
MIN_SCORE = float(os.environ.get("MIN_SCORE", "0.5"))
# MediaPipe 检不出人脸时, 用画面中心框作为 SAM2 box prompt 兜底。
# 依据: render_closeup 的 look-at 保证人脸落在画面正中 (offset 0px),
# 3DGS 近景渲染人脸过糊、MediaPipe 检不出是常态, 兜底才能拿到 mask。
CENTER_BOX_FALLBACK = os.environ.get("CENTER_BOX_FALLBACK", "0") == "1"


def create_feather_alpha(mask, padding_px=5):
    """Create feathered alpha from binary mask (blur edges)."""
    try:
        import cv2
        kernel = np.ones((padding_px * 2 + 1, padding_px * 2 + 1), np.uint8)
        # Dilate then blur for feather
        dilated = cv2.dilate(mask.astype(np.uint8) * 255, kernel, iterations=1)
        blurred = cv2.GaussianBlur(dilated.astype(np.float32), (padding_px * 4 + 1,) * 2, 0)
        return np.clip(blurred / 255.0, 0, 1)
    except ImportError:
        # Fallback: simple distance-based feather
        from scipy.ndimage import distance_transform_edt
        dist = distance_transform_edt(mask)  # distance to boundary (inside)
        inv_dist = distance_transform_edt(~mask)  # distance to boundary (outside)
        alpha = np.clip(dist / (padding_px + 1), 0, 1)
        # Add soft transition outside
        transition = np.clip(inv_dist / (padding_px + 1), 0, 1)
        alpha = np.where(mask, 1.0 - np.clip(1.0 - alpha, 0, 1), 1.0 - transition)
        return np.clip(alpha, 0, 1)


def erode_mask(binary, px):
    """Binary erosion by px pixels."""
    try:
        import cv2
        kernel = np.ones((3, 3), np.uint8)
        return cv2.erode(binary, kernel, iterations=px)
    except ImportError:
        out = binary.copy()
        for _ in range(px):
            p = np.pad(out, 1, mode="constant", constant_values=0)
            out = np.min(np.stack([p[0:-2, 0:-2], p[0:-2, 1:-1], p[0:-2, 2:],
                                   p[1:-1, 0:-2], p[1:-1, 1:-1], p[1:-1, 2:],
                                   p[2:, 0:-2], p[2:, 1:-1], p[2:, 2:]]), axis=0)
        return out


def detect_faces_mediapipe(img_rgb):
    """Use MediaPipe for initial face detection (provides box prompt for SAM2)."""
    import mediapipe as mp
    mp_face = mp.solutions.face_detection
    detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.3)
    results = detector.process(img_rgb)
    boxes = []
    if results.detections:
        h, w = img_rgb.shape[:2]
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            x1 = max(int(bbox.xmin * w), 0)
            y1 = max(int(bbox.ymin * h), 0)
            x2 = min(int((bbox.xmin + bbox.width) * w), w)
            y2 = min(int((bbox.ymin + bbox.height) * h), h)
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
    detector.close() if hasattr(detector, 'close') else None
    return boxes


def main():
    ap = argparse.ArgumentParser(description="SAM2 pixel-level face segmentation.")
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    print(f"🧑 SAM2 face segmentation")
    print(f"  📂 images: {args.images_dir}")
    print(f"  💾 output: {args.output_dir}")
    print(f"  🔧 SAM2 ckpt: {SAM2_CKPT}")
    print(f"  🔧 SAM2 cfg:  {SAM2_CFG}")
    print(f"  📐 min_score: {MIN_SCORE}, erode: {ERODE_PX}px, feather: {SOFT_FEATHER}")
    print("")

    os.makedirs(args.output_dir, exist_ok=True)

    names = sorted(f for f in os.listdir(args.images_dir)
                   if os.path.splitext(f)[1].lower() in IMG_EXTS)
    if not names:
        sys.exit(f"no images in {args.images_dir}")

    print(f"  found {len(names)} images")

    # Load SAM2
    print("\n📥 loading SAM2...")
    t0 = time.time()
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    import torch

    model = build_sam2(config_file=SAM2_CFG, ckpt_path=SAM2_CKPT, device="cuda")
    predictor = SAM2ImagePredictor(model)
    print(f"  loaded in {time.time()-t0:.1f}s")

    total = len(names)
    processed = 0
    skipped = 0
    t_start = time.time()

    for idx, name in enumerate(names):
        stem = os.path.splitext(name)[0]
        img_path = os.path.join(args.images_dir, name)
        img = np.array(Image.open(img_path).convert("RGB"))
        h, w = img.shape[:2]

        # Step 1: MediaPipe detection (box prompt source)
        boxes = detect_faces_mediapipe(img)
        if not boxes and CENTER_BOX_FALLBACK:
            # 中心框兜底: 近景相机 look-at 对准人脸, 人脸必在画面中央区域
            boxes = [(int(0.25 * w), int(0.15 * h), int(0.75 * w), int(0.75 * h))]
        if not boxes:
            skipped += 1
            if (idx + 1) % 20 == 0 or idx == 0:
                print(f"  [{idx+1}/{total}] {name}: no face (skip)")
            continue

        # Step 2: SAM2 segmentation with box prompts
        predictor.set_image(img)
        best_mask = None
        best_score = 0
        for box in boxes:
            masks, scores, _ = predictor.predict(
                box=np.array(box),
                multimask_output=True,
            )
            for mi, (m, s) in enumerate(zip(masks, scores)):
                if s > best_score and s >= MIN_SCORE:
                    best_score = s
                    best_mask = m

        if best_mask is None:
            skipped += 1
            if (idx + 1) % 20 == 0 or idx == 0:
                print(f"  [{idx+1}/{total}] {name}: no mask above {MIN_SCORE} (skip)")
            continue

        # Step 3: Build alpha + binary
        binary = best_mask.astype(np.uint8) * 255
        if SOFT_FEATHER:
            alpha = create_feather_alpha(best_mask, padding_px=5)
            alpha_uint8 = (alpha * 255).astype(np.uint8)
        else:
            alpha_uint8 = binary.copy()

        binary_eroded = erode_mask(binary, ERODE_PX)

        # Save
        Image.fromarray(alpha_uint8).save(os.path.join(args.output_dir, f"{stem}.alpha.png"))
        Image.fromarray(binary_eroded).save(os.path.join(args.output_dir, f"{stem}.mask.png"))
        processed += 1

        coverage = best_mask.sum() / best_mask.size * 100
        if (idx + 1) % 20 == 0 or idx == 0:
            print(f"  [{idx+1}/{total}] {name}: {len(boxes)} face(s), score={best_score:.3f}, cov={coverage:.1f}%")

    elapsed = time.time() - t_start
    print(f"\n✅ done: {processed} masks, {skipped} skipped, {elapsed:.1f}s ({elapsed/max(total,1):.1f}s/img)")
    print(f"  output: {args.output_dir}")


if __name__ == "__main__":
    main()
