#!/usr/bin/env python3
"""Pick the front-facing image via MediaPipe Face Mesh + segment person (SAM2 or bbox).

Lighter alternative to 01 (SAM 3D Body 3D pose) and 01b (ViTDet + largest area):
- 01:  needs SAM 3D Body GATED weights + 3D body model + global_rot math.
- 01b: needs ViTDet (detectron2) + picks largest person area (not always frontal).
- 01c (this): only needs MediaPipe (pip install, CPU-friendly, ~no extra weights).
  Front-facing => nose centered between eyes + widest eye distance.
  Back-facing => no face detected (auto-excluded).
  Side-facing => nose offset, eye distance shrinks => low score.

Pipeline:
1. Walk INPUT_DIR/image/ recursively -> collect all images.
2. For each image: MediaPipe Face Mesh -> 468 landmarks (if a face is found).
3. Frontal score = symmetry * (eye_dist + 0.3).
   - symmetry = 1 - |nose.x - midpoint(left_eye.x, right_eye.x)|  (1 = centered)
   - eye_dist  = |right_eye.x - left_eye.x|  (max when front-facing)
4. Pick the image with the best score.
5. Segment the person:
   a. SAM2 via sam_3d_body HumanSegmentor (if SEGMENTOR_PATH set) -- pixel-accurate.
   b. Bounding box from ViTDet person detection (if DETECTOR_NAME set) -- rough.
   c. Full-image fallback (white-bg the whole frame, rarely useful).
   (Steps a/b reuse 01b's machinery; if both unset, only the picked image is saved
    without segmentation -- handy for a quick "which frame is frontal" check.)
6. Apply mask: person pixels kept, background -> white (or black).
7. Save: segmented_image.png + segmented_image_centered.png
        + front_facing_original.jpg + debug_mask.png + frontal_scores.csv.

Env vars (set by 01c_pick_and_segment.sh):
  SAM3D_DIR, DETECTOR_NAME, DETECTOR_PATH, SEGMENTOR_NAME, SEGMENTOR_PATH, DEVICE,
  INPUT_DIR, OUTPUT_DIR, WHITE_BG, PADDING,
  MP_MIN_CONFIDENCE, SKIP_SEGMENTATION
"""
import os
import sys
import time
import csv
from pathlib import Path

import numpy as np
import cv2

SAM3D_DIR = os.environ.get("SAM3D_DIR", "../sam-3d-body")
DEVICE = os.environ.get("DEVICE", "cpu")
DETECTOR_NAME = os.environ.get("DETECTOR_NAME", "vitdet")
DETECTOR_PATH = os.environ.get("DETECTOR_PATH", "")
SEGMENTOR_NAME = os.environ.get("SEGMENTOR_NAME", "sam2")
SEGMENTOR_PATH = os.environ.get("SEGMENTOR_PATH", "")
INPUT_DIR = os.environ.get("INPUT_DIR", "")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "")
WHITE_BG = os.environ.get("WHITE_BG", "1") == "1"
PADDING = float(os.environ.get("PADDING", "0.1"))
MP_MIN_CONFIDENCE = float(os.environ.get("MP_MIN_CONFIDENCE", "0.5"))
SKIP_SEGMENTATION = os.environ.get("SKIP_SEGMENTATION", "0") == "1"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# --- MediaPipe Face Mesh (imported lazily so the rest of the script can run
#     even if mediapipe isn't installed, e.g. for env checks) ---
_mp_face_mesh = None


def get_face_mesh():
    global _mp_face_mesh
    if _mp_face_mesh is None:
        import mediapipe as mp  # noqa: E402
        _mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,           # only the main subject matters
            refine_landmarks=False,    # 468 landmarks is enough; iris not needed
            min_detection_confidence=MP_MIN_CONFIDENCE,
        )
        print(f"🤖 MediaPipe FaceMesh loaded (min_conf={MP_MIN_CONFIDENCE})")
    return _mp_face_mesh


def compute_frontal_score(image_bgr):
    """Run MediaPipe Face Mesh on one image; return (score, landmarks) or (0.0, None).

    score = symmetry * (eye_dist + 0.3)
      symmetry in [0, 1]  (1 = nose centered between eyes)
      eye_dist in [0, 1]  (normalized x; max when front-facing)
    No face -> (0.0, None). Multi-face -> take the largest by bbox area.
    """
    fm = get_face_mesh()
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = fm.process(rgb)

    if not results.multi_face_landmarks:
        return 0.0, None

    best_score = 0.0
    best_lm = None
    for face_lm in results.multi_face_landmarks:
        lm = face_lm.landmark  # 468 NormalizedLandmark (x, y, z in [0,1])

        nose = lm[1]             # nose tip
        l_eye = lm[33]           # left eye outer corner
        r_eye = lm[263]          # right eye outer corner

        mid_x = (l_eye.x + r_eye.x) / 2.0
        symmetry = 1.0 - abs(nose.x - mid_x)
        eye_dist = abs(r_eye.x - l_eye.x)
        score = symmetry * (eye_dist + 0.3)

        if score > best_score:
            best_score = score
            best_lm = face_lm

    return best_score, best_lm


# ── person segmentation (reuses 01b's tools, optional) ──

def build_tools():
    """Load ViTDet detector + SAM2 segmentor from sam_3d_body code (optional)."""
    if SKIP_SEGMENTATION:
        print("⏭️  SKIP_SEGMENTATION=1 -> no detector/segmentor loaded")
        return None, None

    try:
        sys.path.insert(0, SAM3D_DIR)
        import torch  # noqa: E402
    except ImportError as e:
        print(f"⚠️  sam_3d_body/torch not importable ({e}); skipping segmentation")
        return None, None

    device = DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        print("[!] CUDA not available, using CPU")
        device = "cpu"

    human_detector = human_segmentor = None

    if DETECTOR_NAME:
        try:
            print(f"🔍 loading detector: {DETECTOR_NAME}")
            from tools.build_detector import HumanDetector  # noqa: E402
            human_detector = HumanDetector(
                name=DETECTOR_NAME, device=device, path=DETECTOR_PATH
            )
        except Exception as e:
            print(f"⚠️  detector load failed: {e}")

    need_seg = SEGMENTOR_NAME and (SEGMENTOR_NAME != "sam2" or len(SEGMENTOR_PATH))
    if need_seg:
        try:
            print(f"✂️  loading segmentor: {SEGMENTOR_NAME}")
            from tools.build_sam import HumanSegmentor  # noqa: E402
            human_segmentor = HumanSegmentor(
                name=SEGMENTOR_NAME, device=device, path=SEGMENTOR_PATH
            )
        except Exception as e:
            print(f"⚠️  segmentor load failed: {e}")

    return human_detector, human_segmentor


def detect_persons(human_detector, image_bgr):
    """Return list of [x1, y1, x2, y2] bboxes (largest-first by area)."""
    if human_detector is None:
        h, w = image_bgr.shape[:2]
        return [[0, 0, w, h]]

    for desc, fn in [
        ("run_human_detection(img)", lambda: human_detector.run_human_detection(image_bgr)),
        ("run_human_detection(img, thr)", lambda: human_detector.run_human_detection(image_bgr, 0.8)),
        ("run_human_detection(img, thr, dev)", lambda: human_detector.run_human_detection(image_bgr, 0.8, DEVICE)),
    ]:
        try:
            result = fn()
            bboxes = _extract_bboxes(result)
            if bboxes:
                return bboxes
            print(f"    ⚠️  {desc} returned {type(result).__name__}, no bboxes")
        except Exception as e:
            print(f"    ⚠️  {desc} failed: {e}")
    return []


def _extract_bboxes(result):
    if result is None:
        return []
    if isinstance(result, dict) and "instances" in result:
        result = result["instances"]
    if hasattr(result, "pred_boxes") and hasattr(result, "scores"):
        boxes = result.pred_boxes.tensor.cpu().numpy()
        scores = result.scores.cpu().numpy() if hasattr(result.scores, "cpu") else np.asarray(result.scores)
        classes = (result.pred_classes.cpu().numpy()
                   if hasattr(result, "pred_classes") and hasattr(result.pred_classes, "cpu")
                   else (np.asarray(result.pred_classes) if hasattr(result, "pred_classes")
                         else np.zeros(len(boxes))))
        person_mask = (classes == 0) & (scores >= 0.8)
        return boxes[person_mask].tolist() if person_mask.any() else []
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
    if human_segmentor is None or not bboxes:
        return None
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    np_bboxes = np.array(bboxes, dtype=np.float32)

    for desc, fn in [
        ("run_sam(img_rgb, bboxes)", lambda: human_segmentor.run_sam(image_rgb, np_bboxes)),
        ("run_sam(img_bgr, bboxes)", lambda: human_segmentor.run_sam(image_bgr, np_bboxes)),
        ("run_sam(img_rgb, bboxes, dev)", lambda: human_segmentor.run_sam(image_rgb, np_bboxes, DEVICE)),
        ("run_sam(img_bgr, bboxes, dev)", lambda: human_segmentor.run_sam(image_bgr, np_bboxes, DEVICE)),
        ("__call__(img_rgb, bboxes)", lambda: human_segmentor(image_rgb, np_bboxes)),
        ("__call__(img_bgr, bboxes)", lambda: human_segmentor(image_bgr, np_bboxes)),
    ]:
        try:
            result = fn()
            mask = _extract_mask(result)
            if mask is not None:
                print(f"    ✅ segmentor OK via {desc}")
                return mask
            print(f"    ⚠️  {desc} returned {type(result).__name__}, no mask extracted")
        except Exception as e:
            print(f"    ⚠️  {desc} failed: {e}")
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
    print(f"    📐 centered: dx={dx} dy={dy}")
    return result


def main():
    if not INPUT_DIR:
        sys.exit("❌ INPUT_DIR not set")
    if not OUTPUT_DIR:
        sys.exit("❌ OUTPUT_DIR not set")

    image_dir = Path(INPUT_DIR) / "image"
    if not image_dir.is_dir():
        image_dir = Path(INPUT_DIR)
    if not image_dir.is_dir():
        sys.exit(f"❌ image dir not found: {image_dir}")

    images = []
    for root, _, files in os.walk(image_dir):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                images.append(Path(root) / f)
    images.sort(key=lambda x: str(x.relative_to(image_dir)))

    if not images:
        sys.exit(f"❌ no images in {image_dir}")
    print(f"🖼️  {len(images)} images in {image_dir}")

    # Trigger MediaPipe import early so a missing install fails fast.
    get_face_mesh()

    human_detector, human_segmentor = build_tools()

    results = []
    t0 = time.time()

    for i, fp in enumerate(images, 1):
        rel = fp.relative_to(image_dir)
        t1 = time.time()
        try:
            image_bgr = cv2.imread(str(fp))
            if image_bgr is None:
                print(f"[{i}/{len(images)}] {fp.name}  ⚠️  cannot read")
                continue

            score, lm = compute_frontal_score(image_bgr)
            if lm is None:
                print(f"[{i}/{len(images)}] {fp.name}  ⚪  no face (back/side?)")
                continue

            results.append({
                "path": str(fp),
                "rel": str(rel),
                "score": score,
                "image": image_bgr,
            })
            print(f"[{i}/{len(images)}] {fp.name}  🎯 score={score:.4f}  ({time.time()-t1:.2f}s)")
        except Exception as e:
            print(f"[{i}/{len(images)}] {fp.name}  ❌ {e}", file=sys.stderr)

    print(f"⏱️  processed {len(results)}/{len(images)} images in {time.time()-t0:.1f}s")

    if not results:
        sys.exit("❌ no face detected in any image (are these back-facing shots?)")

    best = max(results, key=lambda r: r["score"])
    print(f"\n🎯 picked: {best['rel']}  (score={best['score']:.4f})")

    image_bgr = best["image"]
    img_h, img_w = image_bgr.shape[:2]

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save the picked original (always, even if segmentation is skipped)
    cv2.imwrite(str(out_dir / "front_facing_original.jpg"), image_bgr)
    print(f"✅ saved: {out_dir/'front_facing_original.jpg'}")

    # Save scores CSV (always — handy for debugging the pick)
    with open(out_dir / "frontal_scores.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "frontal_score"])
        for r in sorted(results, key=lambda x: -x["score"]):
            writer.writerow([r["rel"], f"{r['score']:.4f}"])
    print(f"✅ saved: {out_dir/'frontal_scores.csv'}")

    if SKIP_SEGMENTATION:
        print("⏭️  SKIP_SEGMENTATION=1 -> done (no segmentation)")
        print(f"\n🎉 Done. Next:")
        print(f"   WEIGHT_PATH=<lora> bash {Path(__file__).resolve().parent}/02_generate_video.sh")
        return

    # Segment the person in the picked image.
    bboxes = detect_persons(human_detector, image_bgr)
    best_bbox = max(bboxes, key=lambda b: max(0, b[2]-b[0]) * max(0, b[3]-b[1])) if bboxes else None

    mask = None
    if human_segmentor is not None and best_bbox is not None:
        mask = segment_with_segmentor(human_segmentor, image_bgr, [best_bbox])
    if mask is None and best_bbox is not None:
        mask = bbox_to_mask(best_bbox, img_h, img_w, PADDING)

    if mask is None:
        print("⚠️  no mask produced (no detector/segmentor configured).")
        print("    Set DETECTOR_NAME=vitdet + SEGMENTOR_PATH=<sam2_repo> for segmentation,")
        print("    or use SKIP_SEGMENTATION=1 to skip silently.")
        print(f"\n🎉 Done (front-facing pick only). Next:")
        print(f"   WEIGHT_PATH=<lora> bash {Path(__file__).resolve().parent}/02_generate_video.sh")
        return

    result_img = apply_mask(image_bgr, mask, white_bg=WHITE_BG)
    centered_img = center_person(result_img, mask, white_bg=WHITE_BG)

    seg_path = out_dir / "segmented_image.png"
    cv2.imwrite(str(seg_path), result_img)
    print(f"✅ saved: {seg_path}")

    seg_centered_path = out_dir / "segmented_image_centered.png"
    cv2.imwrite(str(seg_centered_path), centered_img)
    print(f"✅ saved: {seg_centered_path} (person centered)")

    cv2.imwrite(str(out_dir / "debug_mask.png"), mask * 255)

    print(f"\n🎉 Done. Next:")
    print(f"   WEIGHT_PATH=<lora> bash {Path(__file__).resolve().parent}/02_generate_video.sh")


if __name__ == "__main__":
    main()
