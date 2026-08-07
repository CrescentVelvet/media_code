#!/usr/bin/env python3
"""Simplified pick + segment: detect person -> SAM segment -> largest area -> white bg.

No 3D body model, no global_rot, no FOV. Just:
1. ViTDet detects person bbox in each image (or full-image fallback)
2. SAM2/SAM3 segments person from bbox (or bbox fallback)
3. Pick the image with the largest person mask area
4. Apply mask: person kept, background -> white
"""
import os
import sys
import time
import csv
from pathlib import Path

import numpy as np
import cv2

SAM3D_DIR = os.environ.get("SAM3D_DIR", "../sam-3d-body")
sys.path.insert(0, SAM3D_DIR)

import torch  # noqa: E402

DEVICE = os.environ.get("DEVICE", "cuda")
DETECTOR_NAME = os.environ.get("DETECTOR_NAME", "vitdet")
DETECTOR_PATH = os.environ.get("DETECTOR_PATH", "")
SEGMENTOR_NAME = os.environ.get("SEGMENTOR_NAME", "sam2")
SEGMENTOR_PATH = os.environ.get("SEGMENTOR_PATH", "")
INPUT_DIR = os.environ.get("INPUT_DIR", "")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "")
BBOX_THRESH = float(os.environ.get("BBOX_THRESH", "0.8"))
WHITE_BG = os.environ.get("WHITE_BG", "1") == "1"
PADDING = float(os.environ.get("PADDING", "0.1"))

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def build_tools():
    device = DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        print("[!] CUDA not available, using CPU")
        device = "cpu"

    human_detector = human_segmentor = None

    if DETECTOR_NAME:
        print(f"[*] loading detector: {DETECTOR_NAME}")
        from tools.build_detector import HumanDetector
        human_detector = HumanDetector(name=DETECTOR_NAME, device=device, path=DETECTOR_PATH)

    need_seg = SEGMENTOR_NAME and (SEGMENTOR_NAME != "sam2" or len(SEGMENTOR_PATH))
    if need_seg:
        print(f"[*] loading segmentor: {SEGMENTOR_NAME}")
        from tools.build_sam import HumanSegmentor
        human_segmentor = HumanSegmentor(name=SEGMENTOR_NAME, device=device, path=SEGMENTOR_PATH)

    return human_detector, human_segmentor


def detect_persons(human_detector, image_bgr):
    if human_detector is None:
        h, w = image_bgr.shape[:2]
        return [[0, 0, w, h]]

    # HumanDetector.run_human_detection(image, score_threshold, device)
    for desc, fn in [
        ("run_human_detection(img)", lambda: human_detector.run_human_detection(image_bgr)),
        ("run_human_detection(img, thr)", lambda: human_detector.run_human_detection(image_bgr, BBOX_THRESH)),
        ("run_human_detection(img, thr, dev)", lambda: human_detector.run_human_detection(image_bgr, BBOX_THRESH, DEVICE)),
    ]:
        try:
            result = fn()
            bboxes = _extract_bboxes(result)
            if bboxes:
                return bboxes
            print(f"    ⚠️ {desc} returned {type(result).__name__}, no bboxes")
        except Exception as e:
            print(f"    ⚠️ {desc} failed: {e}")
    return []


def _extract_bboxes(result):
    if result is None:
        return []
    # detectron2 DefaultPredictor returns dict {"instances": Instances(...)}
    if isinstance(result, dict) and "instances" in result:
        result = result["instances"]
    # detectron2 Instances object
    if hasattr(result, "pred_boxes") and hasattr(result, "scores"):
        boxes = result.pred_boxes.tensor.cpu().numpy()
        scores = result.scores.cpu().numpy() if hasattr(result.scores, "cpu") else np.asarray(result.scores)
        classes = result.pred_classes.cpu().numpy() if hasattr(result, "pred_classes") and hasattr(result.pred_classes, "cpu") else \
            (np.asarray(result.pred_classes) if hasattr(result, "pred_classes") else np.zeros(len(boxes)))
        # COCO class 0 = person
        person_mask = (classes == 0) & (scores >= BBOX_THRESH)
        return boxes[person_mask].tolist() if person_mask.any() else []
    # numpy array: (N, 5) [x1, y1, x2, y2, score] or (N, 4)
    if isinstance(result, np.ndarray):
        arr = result.squeeze()
        if arr.ndim == 1:
            arr = arr[None]
        if arr.shape[-1] >= 4:
            return arr[:, :4].tolist()
    if isinstance(result, (list, tuple)):
        bboxes = []
        for item in result:
            if isinstance(item, (list, tuple, np.ndarray)):
                arr = np.asarray(item).squeeze()
                if arr.shape[-1] >= 4:
                    bboxes.append(arr[:4].tolist())
        return bboxes
    return []


def _extract_mask(result):
    if result is None:
        return None
    if isinstance(result, np.ndarray):
        m = result.squeeze()
        if m.ndim == 2:
            return (m > 0).astype(np.uint8)
        if m.ndim == 3 and m.shape[0] == 1:
            return (m[0] > 0).astype(np.uint8)
    if isinstance(result, (list, tuple)):
        for item in result:
            mask = _extract_mask(item)
            if mask is not None:
                return mask
    return None


def segment_with_segmentor(human_segmentor, image_bgr, bboxes):
    if human_segmentor is None:
        print("    ⚠️ segmentor not loaded (SEGMENTOR_PATH not set?), using bbox fallback")
        return None
    if not bboxes:
        return None

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    np_bboxes = np.array(bboxes, dtype=np.float32)

    # HumanSegmentor.run_sam(image, bboxes, ...) — try different arg patterns
    for desc, fn in [
        ("run_sam(img_rgb, bboxes)", lambda: human_segmentor.run_sam(image_rgb, np_bboxes)),
        ("run_sam(img_bgr, bboxes)", lambda: human_segmentor.run_sam(image_bgr, np_bboxes)),
        ("run_sam(img_rgb, bboxes, dev)", lambda: human_segmentor.run_sam(image_rgb, np_bboxes, DEVICE)),
        ("run_sam(img_bgr, bboxes, dev)", lambda: human_segmentor.run_sam(image_bgr, np_bboxes, DEVICE)),
    ]:
        try:
            result = fn()
            mask = _extract_mask(result)
            if mask is not None:
                print(f"    ✅ segmentor OK via {desc}")
                return mask
            print(f"    ⚠️ {desc} returned {type(result).__name__}, no mask extracted")
        except Exception as e:
            print(f"    ⚠️ {desc} failed: {e}")
    return None


def bbox_to_mask(bbox, img_h, img_w, padding=0.1):
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * padding))
    y1 = max(0, int(y1 - bh * padding))
    x2 = min(img_w, int(x2 + bw * padding))
    y2 = min(img_h, int(y2 + bh * padding))
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    return mask


def apply_mask(image_bgr, mask, white_bg=True):
    result = image_bgr.copy()
    result[mask == 0] = (255, 255, 255) if white_bg else (0, 0, 0)
    return result


def center_person(image_bgr, mask, white_bg=True):
    """Shift the person to the image center, fill gaps with background color."""
    h, w = image_bgr.shape[:2]
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return image_bgr

    person_cx = (xs.min() + xs.max()) / 2
    person_cy = (ys.min() + ys.max()) / 2
    dx = int(w / 2 - person_cx)
    dy = int(h / 2 - person_cy)

    if dx == 0 and dy == 0:
        return image_bgr

    bg_color = (255, 255, 255) if white_bg else (0, 0, 0)
    result = np.full_like(image_bgr, bg_color)

    src_x1, src_y1 = max(0, -dx), max(0, -dy)
    src_x2, src_y2 = min(w, w - dx), min(h, h - dy)
    dst_x1, dst_y1 = max(0, dx), max(0, dy)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    result[dst_y1:dst_y2, dst_x1:dst_x2] = image_bgr[src_y1:src_y2, src_x1:src_x2]
    print(f"    📐 centered: shift dx={dx} dy={dy}")
    return result


def main():
    if not INPUT_DIR:
        sys.exit("ERROR: INPUT_DIR not set.")
    if not OUTPUT_DIR:
        sys.exit("ERROR: OUTPUT_DIR not set.")

    image_dir = Path(INPUT_DIR) / "image"
    if not image_dir.is_dir():
        image_dir = Path(INPUT_DIR)
    if not image_dir.is_dir():
        sys.exit(f"ERROR: image dir not found: {image_dir}")

    images = []
    for root, _, files in os.walk(image_dir):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                images.append(Path(root) / f)
    images.sort(key=lambda x: str(x.relative_to(image_dir)))

    if not images:
        sys.exit(f"ERROR: no images in {image_dir}")
    print(f"🔍 {len(images)} images in {image_dir}")

    human_detector, human_segmentor = build_tools()

    results = []
    t0 = time.time()

    for i, fp in enumerate(images, 1):
        rel = fp.relative_to(image_dir)
        t1 = time.time()
        try:
            image_bgr = cv2.imread(str(fp))
            if image_bgr is None:
                print(f"[{i}/{len(images)}] {fp.name}  ⚠️ cannot read")
                continue
            img_h, img_w = image_bgr.shape[:2]

            bboxes = detect_persons(human_detector, image_bgr)
            if not bboxes:
                print(f"[{i}/{len(images)}] {fp.name}  ⚪ no person")
                continue

            best_bbox = max(bboxes, key=lambda b: max(0, b[2]-b[0]) * max(0, b[3]-b[1]))

            mask = None
            if human_segmentor is not None:
                mask = segment_with_segmentor(human_segmentor, image_bgr, [best_bbox])
            if mask is None:
                mask = bbox_to_mask(best_bbox, img_h, img_w, PADDING)

            if mask is None:
                print(f"[{i}/{len(images)}] {fp.name}  ⚪ no mask")
                continue

            area = int(mask.sum())
            results.append({"path": str(fp), "rel": str(rel), "area": area, "mask": mask, "image": image_bgr})
            print(f"[{i}/{len(images)}] {fp.name}  ✂️ area={area}  ({time.time()-t1:.2f}s)")
        except Exception as e:
            print(f"[{i}/{len(images)}] {fp.name}  ❌ {e}", file=sys.stderr)

    print(f"⏱️ processed {len(results)}/{len(images)} in {time.time()-t0:.1f}s")
    if not results:
        sys.exit("❌ no person detected in any image.")

    best = max(results, key=lambda r: r["area"])
    print(f"\n🎯 picked: {best['rel']}  (area={best['area']})")

    result_img = apply_mask(best["image"], best["mask"], white_bg=WHITE_BG)
    centered_img = center_person(result_img, best["mask"], white_bg=WHITE_BG)

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    seg_path = out_dir / "segmented_image.png"
    cv2.imwrite(str(seg_path), result_img)
    print(f"✅ saved: {seg_path}")

    seg_centered_path = out_dir / "segmented_image_centered.png"
    cv2.imwrite(str(seg_centered_path), centered_img)
    print(f"✅ saved: {seg_centered_path} (person centered)")

    cv2.imwrite(str(out_dir / "front_facing_original.jpg"), best["image"])
    cv2.imwrite(str(out_dir / "debug_mask.png"), best["mask"] * 255)

    with open(out_dir / "areas.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "person_area"])
        for r in sorted(results, key=lambda x: -x["area"]):
            writer.writerow([r["rel"], r["area"]])

    print(f"\n🎉 Done. Next:")
    print(f"   WEIGHT_PATH=<lora> bash {Path(__file__).resolve().parent}/02_generate_video.sh")


if __name__ == "__main__":
    main()
