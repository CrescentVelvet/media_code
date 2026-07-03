#!/usr/bin/env python3
"""Paired HQ/LQ face-crop dataset builder for face super-resolution.

Given a high-quality RAW image folder (CR2/NEF/ARW/RAF/RW2, decoded via rawpy
with EXIF orientation applied) and a low-quality 360p .tif folder of the SAME
11161 images (same stems, same content, only resolution differs), this script:

  1. Detects faces on the LQ image (MediaPipe BlazeFace) — the LQ is in the
     detector's sweet spot and guarantees the LQ crop is meaningful.
  2. Converts each (padded, clamped) face box to RELATIVE [0,1] coordinates.
  3. Applies the SAME relative region to the decoded HQ image (scaled by the
     per-axis pixel ratio) so HQ and LQ crops are pixel-aligned pairs.
  4. Saves paired PNGs:  <out>/hq/<stem>_faceN.png  and  <out>/lq/<stem>_faceN.png.

Images whose HQ/LQ aspect ratios disagree (e.g. a few 4:3 Panasonic RW2 vs the
2:3 LQ) are SKIPPED to keep every produced pair geometrically consistent.

Multi-process + resumable (faces_paired_log.csv). Re-run skips stems already
marked "ok".

Usage:
    python crop_faces_paired.py
    python crop_faces_paired.py --workers 6 --min-confidence 0.5 --pad 0.15
    python crop_faces_paired.py --half-size          # faster, ~5x scale (default: full ~10.7x)
"""

import argparse
import csv
import logging
import os
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# --- env must be set before importing mediapipe / rawpy ----------------------
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OMP_NUM_THREADS", "1")  # rawpy/LibRaw OpenMP: 1 thread per worker

import numpy as np
from PIL import Image
import rawpy
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("paired")

DEFAULT_HQ = r"C:\baidunetdiskdownload\PPR0K_all_files_11161_zip\raw_zips\raw"
DEFAULT_LQ = r"C:\baidunetdiskdownload\PPR0K_all_files_11161_zip\train_val_images_tif_360p\source"
DEFAULT_OUT = r"C:\code\ppr10k_faces"
LOG_NAME = "faces_paired_log.csv"
RAW_EXTS = {".cr2", ".nef", ".arw", ".raf", ".rw2", ".dng"}
AR_TOL = 0.02  # accept up to 2% aspect-ratio mismatch

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


def parse_args():
    p = argparse.ArgumentParser(description="Paired HQ/LQ face-crop dataset builder (SR training).")
    p.add_argument("--hq", default=DEFAULT_HQ, help="HQ RAW image directory (CR2/NEF/ARW/RAF/RW2).")
    p.add_argument("--lq", default=DEFAULT_LQ, help="LQ image directory (.tif 360p).")
    p.add_argument("--output", "-o", default=DEFAULT_OUT, help="Output root: creates <out>/hq and <out>/lq.")
    p.add_argument("--model-kind", default="short", choices=("short", "full"))
    p.add_argument("--model-path", default=None, help="Override .tflite path.")
    p.add_argument("--min-confidence", type=float, default=0.5)
    p.add_argument("--pad", type=float, default=0.15, help="Expand each face box by this fraction per side.")
    p.add_argument("--min-size", type=int, default=24, help="Min LQ face crop side (px).")
    p.add_argument("--png-level", type=int, default=3, help="PNG compress_level (0-9). Lower=faster+larger.")
    p.add_argument("--half-size", action="store_true", help="Decode HQ at half resolution (~2x faster, ~5x scale).")
    p.add_argument("--workers", type=int, default=0, help="Process pool size (0 = min(cpu,8)).")
    p.add_argument("--limit", type=int, default=0, help="Process only first N pairs (smoke test).")
    p.add_argument("--overwrite", action="store_true", help="Ignore log; reprocess everything.")
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
    urllib.request.urlretrieve(info["url"], p)
    return p


def load_done(log_csv: Path) -> set:
    if not log_csv.exists():
        return set()
    done = set()
    with log_csv.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "ok" and row.get("stem"):
                done.add(row["stem"])
    return done


def build_pairs(hq_dir: Path, lq_dir: Path):
    pairs = []
    for p in hq_dir.iterdir():
        if not p.is_file() or p.suffix.lower() not in RAW_EXTS:
            continue
        lq = lq_dir / (p.stem + ".tif")
        if lq.exists():
            pairs.append((p.stem, p, lq))
    pairs.sort(key=lambda x: x[0])
    return pairs


# --- worker state ------------------------------------------------------------
_det = None  # per-process MediaPipe FaceDetector


def worker_init(model_path, min_confidence):
    global _det
    base = mp_python.BaseOptions(model_asset_path=str(model_path))
    opts = vision.FaceDetectorOptions(base_options=base, min_detection_confidence=min_confidence)
    _det = vision.FaceDetector.create_from_options(opts)


def lq_face_boxes(rgb):
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = _det.detect(img)
    dets = res.detections or []
    # left-to-right by box center x
    return sorted(dets, key=lambda d: d.bounding_box.origin_x + d.bounding_box.width / 2.0)


def padded_clamped_box(b, w, h, pad, min_size):
    """Expand box by `pad` on each side, clamp to image, drop tiny ones. Returns int px box."""
    cx = b.origin_x + b.width / 2.0
    cy = b.origin_y + b.height / 2.0
    nw = b.width * (1 + 2 * pad)
    nh = b.height * (1 + 2 * pad)
    x0 = max(0, int(round(cx - nw / 2.0)))
    y0 = max(0, int(round(cy - nh / 2.0)))
    x1 = min(w, int(round(cx + nw / 2.0)))
    y1 = min(h, int(round(cy + nh / 2.0)))
    if (x1 - x0) < min_size or (y1 - y0) < min_size:
        return None
    return x0, y0, x1, y1


def process_one(task):
    stem, hq_path, lq_path, out_hq, out_lq, pad, min_size, min_conf, half_size, png_level = task
    # 1) LQ load + detect
    try:
        lq = np.array(Image.open(lq_path).convert("RGB"))
    except Exception as e:
        return stem, 0, [], f"error: lq read: {e}"
    lq_h, lq_w = lq.shape[:2]
    dets = lq_face_boxes(lq)
    # 2) HQ decode (auto-oriented per EXIF)
    try:
        with rawpy.imread(str(hq_path)) as rp:
            hq = rp.postprocess(output_bps=8, half_size=half_size)
    except Exception as e:
        return stem, 0, [], f"error: hq decode: {e}"
    hq_h, hq_w = hq.shape[:2]
    # 3) aspect gate
    if abs((hq_w / hq_h) - (lq_w / lq_h)) > AR_TOL:
        return stem, 0, [], "skip:aspect_mismatch"
    # 4) paired crops via shared relative region
    saved = []
    for d in dets:
        if d.categories and d.categories[0].score < min_conf:
            continue
        box = padded_clamped_box(d.bounding_box, lq_w, lq_h, pad, min_size)
        if box is None:
            continue
        x0, y0, x1, y1 = box
        rx0, ry0, rx1, ry1 = x0 / lq_w, y0 / lq_h, x1 / lq_w, y1 / lq_h
        lq_crop = lq[y0:y1, x0:x1]
        HX0 = max(0, int(round(rx0 * hq_w)))
        HY0 = max(0, int(round(ry0 * hq_h)))
        HX1 = min(hq_w, int(round(rx1 * hq_w)))
        HY1 = min(hq_h, int(round(ry1 * hq_h)))
        hq_crop = hq[HY0:HY1, HX0:HX1]
        if lq_crop.size == 0 or hq_crop.size == 0:
            continue
        n = len(saved) + 1
        name = f"{stem}_face{n}.png"
        try:
            Image.fromarray(lq_crop).save(out_lq / name, compress_level=png_level)
            Image.fromarray(hq_crop).save(out_hq / name, compress_level=png_level)
        except Exception as e:
            return stem, len(saved), saved, f"error: save: {e}"
        saved.append(name)
    return stem, len(saved), saved, "ok"


def main() -> int:
    args = parse_args()
    hq_dir, lq_dir, out_root = Path(args.hq), Path(args.lq), Path(args.output)
    if not hq_dir.is_dir():
        log.error("HQ dir not found: %s", hq_dir); return 2
    if not lq_dir.is_dir():
        log.error("LQ dir not found: %s", lq_dir); return 2
    out_hq = out_root / "hq"; out_lq = out_root / "lq"
    out_hq.mkdir(parents=True, exist_ok=True)
    out_lq.mkdir(parents=True, exist_ok=True)
    log_csv = out_root / LOG_NAME

    pairs = build_pairs(hq_dir, lq_dir)
    if args.limit > 0:
        pairs = pairs[: args.limit]
    if not pairs:
        log.error("no matching HQ/LQ pairs found"); return 2

    done = set() if args.overwrite else load_done(log_csv)
    todo = [pp for pp in pairs if pp[0] not in done]
    model_path = resolve_model(args)
    workers = args.workers or min(os.cpu_count() or 4, 8)

    log.info("HQ=%s", hq_dir)
    log.info("LQ=%s", lq_dir)
    log.info("out=%s/{hq,lq}  png_level=%d  half_size=%s  workers=%d", out_root, args.png_level, args.half_size, workers)
    log.info("model=%s  min_conf=%.2f  pad=%.2f  min_size=%d", model_path.name, args.min_confidence, args.pad, args.min_size)
    log.info("pairs=%d  done=%d  todo=%d", len(pairs), len(done), len(todo))

    tasks = [
        (stem, str(hq), str(lq), out_hq, out_lq, args.pad, args.min_size,
         args.min_confidence, args.half_size, args.png_level)
        for (stem, hq, lq) in todo
    ]

    write_header = (not log_csv.exists()) or args.overwrite
    if write_header and log_csv.exists() and args.overwrite:
        log_csv.unlink()
    log_f = log_csv.open("a", newline="", encoding="utf-8")
    writer = csv.writer(log_f)
    if write_header:
        writer.writerow(["stem", "n_faces", "outputs", "status"])
        log_f.flush()

    t0 = time.time()
    processed = faces_total = n_no_face = n_skip = errors = 0
    ret = 0
    try:
        with ProcessPoolExecutor(max_workers=workers, initializer=worker_init,
                                 initargs=(str(model_path), args.min_confidence)) as ex:
            for stem, n, saved, status in ex.map(process_one, tasks, chunksize=4):
                writer.writerow([stem, n, ";".join(saved), status])
                log_f.flush()
                processed += 1
                if status == "ok":
                    faces_total += n
                    if n == 0:
                        n_no_face += 1
                elif status.startswith("skip"):
                    n_skip += 1
                else:
                    errors += 1
                    log.warning("%s -> %s", stem, status)
                if processed % 100 == 0 or processed == len(tasks):
                    el = time.time() - t0
                    rate = processed / el if el > 0 else 0
                    rem = (len(tasks) - processed) / rate if rate > 0 else 0
                    log.info("progress %d/%d  faces=%d  no_face=%d  skip=%d  err=%d  %.1f/s  eta=%.0fs",
                             processed, len(tasks), faces_total, n_no_face, n_skip, errors, rate, rem)
    except KeyboardInterrupt:
        log.warning("interrupted (partial results saved)"); ret = 130
    finally:
        log_f.close()

    el = time.time() - t0
    log.info("done: processed=%d faces=%d no_face=%d skip=%d errors=%d elapsed=%.1fs",
             processed, faces_total, n_no_face, n_skip, errors, el)
    log.info("log: %s", log_csv)
    return ret


if __name__ == "__main__":
    sys.exit(main())
