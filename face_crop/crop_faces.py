#!/usr/bin/env python3
"""Batch face detection & cropping with MediaPipe (Tasks API).

Reads all images from an input directory, detects (possibly multiple) faces per
image using MediaPipe's BlazeFace FaceDetector, crops each face, and saves each
crop as a JPG into the output directory. Crops are named "<original_stem>_face1.jpg",
"_face2.jpg", ... sorted left-to-right by horizontal position.

A CSV log (faces_log.csv in the output dir) records every processed file so the
run is resumable: re-running skips files already marked "ok". Delete the CSV
(or the output dir) to reprocess.

Usage:
    python crop_faces.py
    python crop_faces.py --input C:\\code\\target_c --output C:\\code\\target_c_faces
    python crop_faces.py --model-kind short --min-confidence 0.5 --pad 0.15
"""

import argparse
import csv
import logging
import os
import sys
import time
import urllib.request
from pathlib import Path

# Silence MediaPipe / TFLite C++ verbose logs (INFO/WARNING) before import.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
from PIL import Image

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("face_crop")

DEFAULT_INPUT = r"C:\code\target_c"
DEFAULT_OUTPUT = r"C:\code\target_c_faces"
IMAGE_EXTS = (".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jp2")
LOG_NAME = "faces_log.csv"

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODELS = {
    "short": {
        "file": "blaze_face_short_range.tflite",
        "url": "https://storage.googleapis.com/mediapipe-models/face_detector/"
               "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
    },
    "full": {
        "file": "blaze_face_full_range.tflite",
        "url": "https://storage.googleapis.com/mediapipe-models/face_detector/"
               "blaze_face_full_range/float16/latest/blaze_face_full_range.tflite",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch face detection & cropping (MediaPipe Tasks API).")
    p.add_argument("--input", "-i", default=DEFAULT_INPUT, help="Input image directory.")
    p.add_argument("--output", "-o", default=DEFAULT_OUTPUT, help="Output directory for cropped JPGs.")
    p.add_argument("--model-kind", default="short", choices=("short", "full"),
                   help="BlazeFace model: 'short' (<2m, close/large faces, default) "
                        "or 'full' (<5m, smaller/farther faces).")
    p.add_argument("--model-path", default=None,
                   help="Override model: path to a .tflite face detector. Takes precedence over --model-kind.")
    p.add_argument("--min-confidence", type=float, default=0.5,
                   help="Minimum detection confidence [0,1]. Lower finds more but more false positives. Default 0.5.")
    p.add_argument("--pad", type=float, default=0.15,
                   help="Expand each face box by this fraction on every side (clamped to image). Default 0.15.")
    p.add_argument("--min-size", type=int, default=24,
                   help="Skip crops smaller than this many pixels on either side. Default 24.")
    p.add_argument("--quality", type=int, default=95, help="JPEG save quality 1-100. Default 95.")
    p.add_argument("--exts", default=",".join(IMAGE_EXTS),
                   help="Comma-separated image extensions to process.")
    p.add_argument("--log", default=None, help="CSV log path. Default <output>/faces_log.csv.")
    p.add_argument("--limit", type=int, default=0,
                   help="Process only the first N images (0 = all). Useful for smoke tests.")
    p.add_argument("--overwrite", action="store_true",
                   help="Ignore existing CSV log and reprocess everything (existing JPGs are overwritten).")
    return p.parse_args()


def resolve_model(args) -> Path:
    if args.model_path:
        p = Path(args.model_path)
        if not p.is_file():
            raise SystemExit(f"model not found: {p}")
        return p
    info = MODELS[args.model_kind]
    p = MODEL_DIR / info["file"]
    if p.is_file():
        return p
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s-range model -> %s", args.model_kind, p)
    try:
        urllib.request.urlretrieve(info["url"], p)
    except Exception as e:
        raise SystemExit(f"failed to download model from {info['url']}: {e}")
    log.info("downloaded %d bytes", p.stat().st_size)
    return p


def load_done(log_csv: Path) -> set:
    done = set()
    if not log_csv.exists():
        return done
    with log_csv.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "ok" and row.get("filename"):
                done.add(row["filename"])
    return done


def iter_images(input_dir: Path, exts):
    exts = tuple(e.lower().lstrip(".") for e in exts)
    files = [p for p in input_dir.iterdir()
             if p.is_file() and p.suffix.lower().lstrip(".") in exts]
    files.sort(key=lambda p: p.name)
    return files


def clamp_box(origin_x, origin_y, w, h, iw, ih, pad, min_size):
    """Expand a pixel box by `pad` fraction on each side, clamp to image, drop tiny ones."""
    cx = origin_x + w / 2.0
    cy = origin_y + h / 2.0
    nw = w * (1 + 2 * pad)
    nh = h * (1 + 2 * pad)
    x0 = cx - nw / 2.0
    y0 = cy - nh / 2.0
    x1 = cx + nw / 2.0
    y1 = cy + nh / 2.0
    x0 = max(0, int(round(x0)))
    y0 = max(0, int(round(y0)))
    x1 = min(iw, int(round(x1)))
    y1 = min(ih, int(round(y1)))
    if (x1 - x0) < min_size or (y1 - y0) < min_size:
        return None
    return x0, y0, x1, y1


def process_one(detector, path: Path, out_dir: Path, args):
    """Detect faces in one image; save crops; return (n_faces, [out_names], status)."""
    try:
        with Image.open(path) as im:
            rgb = np.array(im.convert("RGB"))
    except Exception as e:
        return 0, [], f"error: read failed: {e}"

    ih, iw = rgb.shape[:2]
    try:
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_img)
    except Exception as e:
        return 0, [], f"error: detect failed: {e}"

    dets = result.detections or []
    # sort left-to-right by box center x
    dets = sorted(dets, key=lambda d: d.bounding_box.origin_x + d.bounding_box.width / 2.0)

    saved = []
    for d in dets:
        if d.categories:
            score = d.categories[0].score
            if score < args.min_confidence:
                continue
        b = d.bounding_box
        box = clamp_box(b.origin_x, b.origin_y, b.width, b.height, iw, ih, args.pad, args.min_size)
        if box is None:
            continue
        x0, y0, x1, y1 = box
        crop = rgb[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        out_name = f"{path.stem}_face{len(saved) + 1}.jpg"
        out_path = out_dir / out_name
        try:
            Image.fromarray(crop).save(out_path, "JPEG", quality=args.quality)
            saved.append(out_name)
        except Exception as e:
            return len(saved), saved, f"error: save failed: {e}"
    return len(saved), saved, "ok"


def main() -> int:
    args = parse_args()
    in_dir = Path(args.input)
    out_dir = Path(args.output)
    if not in_dir.is_dir():
        log.error("input directory not found: %s", in_dir)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    log_csv = Path(args.log) if args.log else (out_dir / LOG_NAME)

    exts = [e.strip() for e in args.exts.split(",") if e.strip()]
    files = iter_images(in_dir, exts)
    if args.limit > 0:
        files = files[: args.limit]
    total = len(files)
    if total == 0:
        log.error("no images found in %s with extensions %s", in_dir, exts)
        return 2

    done = set() if args.overwrite else load_done(log_csv)
    todo = [p for p in files if p.name not in done]
    model_path = resolve_model(args)
    log.info("input=%s", in_dir)
    log.info("output=%s", out_dir)
    log.info("model=%s (%s)", model_path.name, args.model_kind)
    log.info("min_conf=%.2f pad=%.2f min_size=%d quality=%d",
             args.min_confidence, args.pad, args.min_size, args.quality)
    log.info("total=%d already_done=%d todo=%d", total, len(done), len(todo))

    write_header = (not log_csv.exists()) or args.overwrite
    if write_header and log_csv.exists() and args.overwrite:
        log_csv.unlink()
    log_f = log_csv.open("a", newline="", encoding="utf-8")
    writer = csv.writer(log_f)
    if write_header:
        writer.writerow(["filename", "n_faces", "outputs", "status"])
        log_f.flush()

    base_opts = mp_python.BaseOptions(model_asset_path=str(model_path))
    det_opts = vision.FaceDetectorOptions(
        base_options=base_opts,
        min_detection_confidence=args.min_confidence,
    )

    t0 = time.time()
    processed = 0
    total_faces = 0
    n_no_face = 0
    errors = 0
    ret = 0
    try:
        with vision.FaceDetector.create_from_options(det_opts) as detector:
            for p in todo:
                n_faces, saved, status = process_one(detector, p, out_dir, args)
                writer.writerow([p.name, n_faces, ";".join(saved), status])
                log_f.flush()
                processed += 1
                if status == "ok":
                    total_faces += n_faces
                    if n_faces == 0:
                        n_no_face += 1
                else:
                    errors += 1
                    log.warning("%s -> %s", p.name, status)
                if processed % 100 == 0 or processed == len(todo):
                    el = time.time() - t0
                    rate = processed / el if el > 0 else 0
                    rem = (len(todo) - processed) / rate if rate > 0 else 0
                    log.info("progress %d/%d  faces=%d  no_face=%d  err=%d  %.1f img/s  eta=%.0fs",
                             processed, len(todo), total_faces, n_no_face, errors, rate, rem)
    except KeyboardInterrupt:
        log.warning("interrupted by user (partial results saved)")
        ret = 130
    finally:
        log_f.close()

    el = time.time() - t0
    log.info("done: processed=%d faces=%d images_with_no_face=%d errors=%d elapsed=%.1fs",
             processed, total_faces, n_no_face, errors, el)
    log.info("log: %s", log_csv)
    return ret


if __name__ == "__main__":
    sys.exit(main())
