#!/usr/bin/env python3
"""face_masks.py — Generate face-region loss masks for HYPIR finetune (Task #16).

Replicates face_enhance.py's exact face regions so the finetune's
complementary supervision matches what HYPIR actually changed:
  1. MediaPipe BlazeFace detection (model_selection=1, conf=0.3) — same as face_enhance.py.
  2. bbox enlarged by FACE_PADDING (default 0.2) — same as face_enhance.py.
  3. Quadratic feather mask per bbox (max over overlapping faces).
  4. alpha saved as 8-bit PNG (soft weight, for optional soft-mask mode).
  5. binary mask = alpha > MASK_THRESHOLD, eroded ERODE_PX pixels (design: 2-3px)
     → pushes the ambiguous transition zone to the "original" side of the
       complementary loss.

Frames with no detected face get NO mask file → downstream finetune treats
them as full-original supervision (design: 失败帧跳过).

Outputs (per image <name>.png in --output_dir):
  <name>.alpha.png  — feather alpha (0-255)
  <name>.mask.png   — binary {0,255}, thresholded + eroded
"""
import os
import sys
import argparse

import numpy as np
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


# ── identical to face_enhance.py ──────────────────────────────────────────
def create_feather_mask(h, w):
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    y = np.arange(h, dtype=np.float32) - cy
    x = np.arange(w, dtype=np.float32) - cx
    yy, xx = np.meshgrid(y, x, indexing="ij")
    r = np.sqrt(xx ** 2 + yy ** 2)
    r_max = max(min(cx, cy), 1.0)
    return np.clip(1.0 - (r / r_max) ** 2, 0.0, 1.0)


def load_face_detector():
    import mediapipe as mp
    mp_face = mp.solutions.face_detection
    return mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.3)


def detect_faces(detector, image_rgb):
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
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    nw, nh = bw * (1 + padding), bh * (1 + padding)
    nx1 = max(int(cx - nw / 2), 0)
    ny1 = max(int(cy - nh / 2), 0)
    nx2 = min(int(cx + nw / 2), img_w)
    ny2 = min(int(cy + nh / 2), img_h)
    return nx1, ny1, nx2, ny2


# ── mask construction ─────────────────────────────────────────────────────
def build_alpha(h, w, boxes, padding):
    """Per-image feather alpha: max over per-face feather masks."""
    alpha = np.zeros((h, w), dtype=np.float32)
    for bbox in boxes:
        x1, y1, x2, y2 = enlarge_bbox(bbox, padding, w, h)
        cw, ch = x2 - x1, y2 - y1
        if cw < 32 or ch < 32:  # same min-size gate as face_enhance.py
            continue
        alpha[y1:y2, x1:x2] = np.maximum(
            alpha[y1:y2, x1:x2], create_feather_mask(ch, cw)
        )
    return alpha


def erode_mask(binary, px):
    """Binary erosion by px pixels (3x3 kernel, px iterations). OpenCV first,
    pure-numpy fallback (shifted-min) if cv2 unavailable."""
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
                                   p[2:,   0:-2], p[2:,   1:-1], p[2:,   2:]]), axis=0)
        return out


def process_image(path, detector, padding, threshold, erode_px):
    img = np.array(Image.open(path).convert("RGB"))
    h, w = img.shape[:2]
    boxes = detect_faces(detector, img)
    if not boxes:
        return None, 0
    alpha = build_alpha(h, w, boxes, padding)
    if alpha.max() <= 0:
        return None, 0
    binary = (alpha > threshold).astype(np.uint8) * 255
    binary = erode_mask(binary, erode_px)
    return alpha, binary, len(boxes)


def main():
    ap = argparse.ArgumentParser(description="Generate face loss masks for HYPIR finetune.")
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--padding", type=float, default=0.2, help="bbox enlarge fraction (must match face_enhance.py)")
    ap.add_argument("--threshold", type=float, default=0.1, help="alpha threshold for binary mask")
    ap.add_argument("--erode_px", type=int, default=2, help="erosion pixels (design: 2-3)")
    args = ap.parse_args()

    detector = load_face_detector()
    os.makedirs(args.output_dir, exist_ok=True)

    names = sorted(f for f in os.listdir(args.images_dir)
                   if os.path.splitext(f)[1].lower() in IMG_EXTS)
    if not names:
        sys.exit(f"❌ no images in {args.images_dir}")

    n_face, n_skip = 0, 0
    for i, name in enumerate(names, 1):
        stem = os.path.splitext(name)[0]
        alpha_path = os.path.join(args.output_dir, f"{stem}.alpha.png")
        mask_path = os.path.join(args.output_dir, f"{stem}.mask.png")
        if os.path.isfile(alpha_path) and os.path.isfile(mask_path):
            n_face += 1
            continue  # cached
        try:
            res = process_image(os.path.join(args.images_dir, name),
                                detector, args.padding, args.threshold, args.erode_px)
        except Exception as e:
            print(f"  [{i}/{len(names)}] ⚠️ {name}: {e}", file=sys.stderr)
            res = None
        if res is None or res[0] is None:
            n_skip += 1
            print(f"  [{i}/{len(names)}] {name} — no face, skip")
            continue
        alpha, binary, k = res
        Image.fromarray((alpha * 255).astype(np.uint8)).save(alpha_path)
        Image.fromarray(binary).save(mask_path)
        n_face += 1
        print(f"  [{i}/{len(names)}] {name} — {k} face(s)")

    print(f"\n🎉 done: {n_face} mask(s), {n_skip} skipped (no face) → {args.output_dir}")


if __name__ == "__main__":
    main()
