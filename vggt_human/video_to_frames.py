#!/usr/bin/env python3
"""Extract frames from a video to an image folder (loose PNGs under image/).

Output structure matches the test_task input format that 01_face_enhance.sh expects:
  <OUTPUT_DIR>/image/000000.png, 000001.png, ...
So 01's INPUT_DIR can point at <OUTPUT_DIR> and face_enhance.py auto-detects the
image/ subfolder. Full pipeline works unchanged:
  01a (video -> frames) -> 01 (face enhance) -> 02 (VGGT-Omega) -> 03 -> 04

Quality gate (blur detection): each sampled frame's Laplacian variance is
computed; frames below BLUR_THRESHOLD are skipped (catches defocus / zoom
transition frames that harm downstream reconstruction). Set BLUR_THRESHOLD=0
to disable. A histogram of kept-frame sharpness is printed for tuning.

Env vars (set by 01a_video_to_frames.sh):
  INPUT_DIR       : video file path (.mp4/.mov/.avi/.mkv)
  OUTPUT_DIR      : output parent folder (frames go to <OUTPUT_DIR>/image/)
  VIDEO_FPS       : frame sampling fps (default 2)
  BLUR_THRESHOLD  : Laplacian variance below this = skip (default 100; 0=off)
"""
import os
import sys

import cv2
import numpy as np

INPUT_DIR = os.environ.get("INPUT_DIR", "")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./frames")
VIDEO_FPS = float(os.environ.get("VIDEO_FPS", "2"))
BLUR_THRESHOLD = float(os.environ.get("BLUR_THRESHOLD", "100"))

VID_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


def laplacian_variance(gray):
    """Variance of Laplacian — higher = sharper. float64 for cv2 compatibility."""
    gray = np.asarray(gray, dtype=np.float64)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def main():
    if not INPUT_DIR:
        sys.exit("❌ INPUT_DIR not set (video file path)")
    if not os.path.isfile(INPUT_DIR):
        sys.exit(f"❌ input video not found: {INPUT_DIR}")
    ext = os.path.splitext(INPUT_DIR)[1].lower()
    if ext not in VID_EXTS:
        sys.exit(f"❌ not a video file ({ext}): {INPUT_DIR}")

    # Write frames to <OUTPUT_DIR>/image/ to match test_task input structure.
    frames_dir = os.path.join(OUTPUT_DIR, "image")
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(INPUT_DIR)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    step = max(int(round(src_fps / max(VIDEO_FPS, 0.1))), 1)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    print(f"🎬 input: {INPUT_DIR}")
    print(f"   src fps: {src_fps:.2f}, sample every {step} frame(s) -> ~{VIDEO_FPS} fps")
    if total:
        est = total // step
        print(f"   total frames: {total}, will sample ~{est}")
    print(f"💾 output: {frames_dir}")
    if BLUR_THRESHOLD > 0:
        print(f"🔍 blur gate: BLUR_THRESHOLD={BLUR_THRESHOLD:.0f} (Laplacian var < this = skip)")
    else:
        print(f"🔍 blur gate: disabled (BLUR_THRESHOLD=0)")

    idx, saved, skipped_blur = 0, 0, 0
    sharpness_kept = []  # for histogram
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            # Blur detection: compute Laplacian variance on grayscale.
            if BLUR_THRESHOLD > 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                lap_var = laplacian_variance(gray)
                if lap_var < BLUR_THRESHOLD:
                    skipped_blur += 1
                    idx += 1
                    continue
                sharpness_kept.append(lap_var)
            cv2.imwrite(os.path.join(frames_dir, f"{saved:06}.png"), frame)
            saved += 1
        idx += 1
    cap.release()

    if saved == 0:
        sys.exit("❌ no frames extracted (video may be corrupt or empty, or all frames below blur threshold)")

    print(f"✅ saved {saved} frames -> {frames_dir}")
    if BLUR_THRESHOLD > 0 and skipped_blur > 0:
        print(f"   skipped {skipped_blur} blurry frame(s) (Laplacian var < {BLUR_THRESHOLD:.0f})")
    if sharpness_kept:
        arr = np.array(sharpness_kept)
        print(f"   sharpness (kept frames): min={arr.min():.1f} median={np.median(arr):.1f} "
              f"mean={arr.mean():.1f} max={arr.max():.1f}")
        # Suggest a threshold for tuning: 10th percentile of kept frames.
        p10 = float(np.percentile(arr, 10))
        print(f"   (for tuning: 10th percentile of kept = {p10:.1f})")


if __name__ == "__main__":
    main()
