#!/usr/bin/env python3
"""extract_frames.py — split a video into JPG frames under <video_name>/image/.

Output structure (alongside the source video; matches the standard
INPUT_DIR/image/ pattern that wan22_rotate step 01 / sam_3d_body / pi3_3dgs
all accept as input):

  <video_dir>/
    <video_name>.mp4          # source video (unchanged)
    <video_name>/             # NEW: same-name folder
      image/
        00000.jpg
        00001.jpg
        ...

So a `rotate_360.mp4` produced by step 02 becomes a `rotate_360/` folder that
can itself be fed back as INPUT_DIR to step 01, or as INPUT to pi3_3dgs.

Env vars (set by 03_extract_frames.sh):
  VIDEO_PATH, FPS, JPG_QUALITY
"""
import os
import sys
import argparse

import cv2


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--video", default=os.environ.get("VIDEO_PATH"),
                   help="Path to the source video (env: VIDEO_PATH)")
    p.add_argument("--fps", type=float,
                   default=float(os.environ.get("FPS", "0")),
                   help="Target sampling fps (0 = every frame at source fps; default 0)")
    p.add_argument("--quality", type=int,
                   default=int(os.environ.get("JPG_QUALITY", "95")),
                   help="JPG quality 1-100 (default 95, visually lossless)")
    p.add_argument("--start", type=int, default=0,
                   help="Start frame index (default 0)")
    p.add_argument("--end", type=int, default=-1,
                   help="End frame index, -1 = until end (default -1)")
    args = p.parse_args()

    if not args.video:
        sys.exit("ERROR: --video (or env VIDEO_PATH) is required")
    if not os.path.isfile(args.video):
        sys.exit(f"ERROR: video not found: {args.video}")

    video_path = os.path.abspath(args.video)
    video_dir = os.path.dirname(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    image_dir = os.path.join(video_dir, video_name, "image")
    os.makedirs(image_dir, exist_ok=True)

    print(f"🚀 [03] extract frames: {video_path}")
    print(f"  📁 output: {image_dir}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"❌ ERROR: cannot open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Decide sampling step. FPS=0 means "every frame at source fps".
    if args.fps > 0 and args.fps < src_fps:
        step = max(int(round(src_fps / args.fps)), 1)
        label = f"@ {args.fps:.1f}fps (step={step}, src={src_fps:.1f}fps)"
    else:
        step = 1
        label = f"@ src {src_fps:.1f}fps (every frame)"
    print(f"  🎬 {total} frames {label}  quality={args.quality}")

    saved = 0
    idx = 0
    end = args.end if args.end >= 0 else total
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx < args.start:
            idx += 1
            continue
        if idx > end and end >= 0:
            break
        if (idx - args.start) % step == 0:
            out_path = os.path.join(image_dir, f"{saved:05d}.jpg")
            cv2.imwrite(out_path, frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
            saved += 1
            if saved % 20 == 0:
                print(f"  ... {saved} frames saved")
        idx += 1
    cap.release()

    if saved == 0:
        sys.exit("❌ ERROR: no frames extracted (video may be empty or unreadable)")

    print(f"✅ saved {saved} JPGs to {image_dir}")
    print(f"🎉 Done. Same-name folder: {os.path.dirname(image_dir)}")
    print(f"   Use as INPUT_DIR for step 01:  INPUT_DIR={os.path.dirname(image_dir)} bash wan22_rotate/01_pick_and_segment.sh")
    print(f"   Use as INPUT for pi3_3dgs:       INPUT={os.path.dirname(image_dir)} bash pi3_3dgs/run_all.sh")


if __name__ == "__main__":
    main()
