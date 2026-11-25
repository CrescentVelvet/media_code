#!/usr/bin/env python3
"""海思静止视角 MP4 -> 可检查的十图 SplatAnything 场景。

MP4 是重复静止画面，sourceIndex 是原采集序列编号，不是 MP4 帧号。
先按缩略图相邻差识别 viewCount 个连续静止段，跳过段首编码抖动后选一帧。
不修改原始 MP4、相机、bounds 或竞品 PLY；所有新产物放 CASE/splatanything/。
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FLIP_CV = np.diag([1.0, -1.0, -1.0, 1.0])  # 输入 OpenGL c2w -> SA OpenCV c2w


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def font(size):
    path = "/mnt/c/Windows/Fonts/msyh.ttc"
    return ImageFont.truetype(path, size) if Path(path).exists() else ImageFont.load_default()


def find_static_segments(video, n_views, threshold=None):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot decode video: {video}")
    thumbs = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        thumbs.append(cv2.resize(im, (88, 88), interpolation=cv2.INTER_AREA).astype(np.float32))
    cap.release()
    if len(thumbs) < n_views:
        raise RuntimeError(f"Only {len(thumbs)} decoded frames for {n_views} views")
    differences = np.abs(np.diff(np.stack(thumbs), axis=0)).mean(axis=(1, 2, 3))
    # 静止段内部是编码噪声，视角切换是数量级更大的画面差。阈值由数据分离，不硬编码 15 帧。
    if threshold is None:
        # 视角切换不超过视图数-1，按最大的边界簇与非边界簇的几何中点定阈值。
        ordered = np.sort(differences)[::-1]
        top = ordered[:n_views - 1]
        floor = float(np.median(ordered[n_views - 1:]))
        threshold = max(2.0, float(top[-1] * .25), floor * 20)
    starts = np.flatnonzero(differences > threshold).astype(int) + 1
    starts = np.concatenate([[0], starts])
    if len(starts) != n_views:
        raise RuntimeError(
            f"Expected {n_views} static segments, found {len(starts)} at {starts.tolist()}; "
            f"threshold={threshold:.4g}, largest={np.sort(differences)[-20:].tolist()}")
    ends = np.concatenate([starts[1:], [len(thumbs)]])
    if np.any(ends - starts < 4):
        raise RuntimeError(f"Static segment shorter than 4 frames: {list(zip(starts, ends))}")
    return list(zip(starts.tolist(), ends.tolist())), differences.tolist(), float(threshold), len(thumbs)


def contact_sheet(frames, records, output):
    tile_w, tile_h, label_h = 352, 352, 62
    canvas = Image.new("RGB", (tile_w * 5, (tile_h + label_h) * 2), (19, 25, 32))
    draw = ImageDraw.Draw(canvas)
    for i, (im, rec) in enumerate(zip(frames, records)):
        x, y = (i % 5) * tile_w, (i // 5) * (tile_h + label_h)
        tile = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).resize(
            (tile_w, tile_h), Image.Resampling.LANCZOS)
        canvas.paste(tile, (x, y))
        draw.text((x + 10, y + tile_h + 5),
                  f"View {i:02d} | sourceIndex {rec['sourceIndex']}",
                  fill=(235, 240, 245), font=font(17))
        draw.text((x + 10, y + tile_h + 30),
                  f"MP4 zero-based frame {rec['video_frame']}",
                  fill=(153, 180, 200), font=font(15))
    canvas.save(output, quality=94)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--segment-offset", type=int, default=8,
                    help="每个静止段跳过的开头帧数，避免段首编码抖动")
    ap.add_argument("--threshold", type=float, default=None,
                    help="静止/切换 MAE 阈值（0..255）；缺省自动分离")
    args = ap.parse_args()
    case = args.case.resolve()
    videos = sorted(case.glob("*.mp4"))
    cameras_files = sorted(case.glob("*cameras.json"))
    bounds_files = sorted(case.glob("*bounds.json"))
    competitor_files = sorted(p for p in case.glob("*.ply") if p.name != "splatanything.ply")
    if not all(len(x) == 1 for x in (videos, cameras_files, bounds_files, competitor_files)):
        raise RuntimeError("Each case must contain exactly one MP4/cameras/bounds/competitor PLY")
    video, camera_file, bounds_file, competitor = (
        videos[0], cameras_files[0], bounds_files[0], competitor_files[0])
    camera_data = json.loads(camera_file.read_text(encoding="utf-8"))
    views = camera_data["views"]
    n_views = int(camera_data["viewCount"])
    if len(views) != n_views:
        raise RuntimeError("viewCount does not match camera entries")
    segments, differences, threshold, n_frames = find_static_segments(
        video, n_views, args.threshold)
    out = args.out.resolve() if args.out else case / "splatanything"
    if out != case / "splatanything" and case not in out.parents:
        raise RuntimeError("Extraction output must stay inside its case directory")
    images_dir = out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    records, frames, extrinsics, intrinsics = [], [], [], []
    for i, (view, (start, end)) in enumerate(zip(views, segments)):
        picked = min(start + args.segment_offset, end - 1)
        if picked - start < 3:
            raise RuntimeError("Segment offset leaves fewer than three leading frames")
        cap.set(cv2.CAP_PROP_POS_FRAMES, picked)
        ok, im = cap.read()
        if not ok:
            raise RuntimeError(f"Failed to read frame {picked}")
        h, w = im.shape[:2]
        if (w, h) != (int(camera_data["imageWidth"]), int(camera_data["imageHeight"])):
            raise RuntimeError(f"Image/K size mismatch: {(w, h)} vs camera JSON")
        rotation = np.asarray(view["rotation3x3"], dtype=np.float64).reshape(3, 3)
        position = np.asarray(view["position"], dtype=np.float64)
        c2w_gl = np.eye(4)
        c2w_gl[:3, :3], c2w_gl[:3, 3] = rotation, position
        c2w_cv = c2w_gl @ FLIP_CV
        K = np.array([[view["fx"], 0, view["cx"]],
                      [0, view["fy"], view["cy"]], [0, 0, 1]], dtype=np.float64)
        if not (0 <= K[0, 2] < w and 0 <= K[1, 2] < h):
            raise RuntimeError("Camera principal point outside image")
        name = f"view_{i:02d}_source_{int(view['sourceIndex']):03d}"
        Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).save(images_dir / f"{name}.png")
        extrinsics.append({"camera_id": name, "matrix": c2w_cv.tolist()})
        intrinsics.append({"camera_id": name, "matrix": K.tolist()})
        records.append({"view": i, "sourceIndex": int(view["sourceIndex"]),
                        "video_frame": int(picked), "segment_start": start,
                        "segment_end_exclusive": end, "image": f"images/{name}.png",
                        "sha256": sha256(images_dir / f"{name}.png")})
        frames.append(im)
        print(f"  view {i:02d}: sourceIndex={view['sourceIndex']} -> MP4 frame {picked}")
    cap.release()
    pairwise_mae = [float(np.abs(frames[i].astype(np.float32) - frames[j].astype(np.float32)).mean())
                    for i in range(n_views) for j in range(i + 1, n_views)]
    if min(pairwise_mae) < 2.0:
        raise RuntimeError(f"Two extracted views are near-duplicates: min pairwise MAE={min(pairwise_mae)}")
    dump(out / "prior_camera.json", {"num_cameras": n_views,
                                    "extrinsics": extrinsics, "intrinsics": intrinsics})
    manifest = {
        "case": case.name, "video": video.name, "video_sha256": sha256(video),
        "video_frame_count": n_frames, "source_indices_are_not_video_frames": True,
        "segments_zero_based_half_open": segments, "difference_threshold": threshold,
        "selection_rule": "detect static segments, skip leading frames, one frame per segment",
        "segment_offset": args.segment_offset, "adjacent_thumbnail_mae": differences,
        "min_selected_pairwise_mae": min(pairwise_mae),
        "input_camera_convention": "OpenGL camera-to-world (-Z forward, +Y up)",
        "output_camera_convention": "OpenCV camera-to-world (+Z forward, +Y down)",
        "camera_file": camera_file.name, "bounds_file": bounds_file.name,
        "competitor_ply": competitor.name, "competitor_sha256": sha256(competitor),
        "frames": records}
    dump(out / "extraction.json", manifest)
    (out / "prior_camera_input.json").write_text(json.dumps(
        {"num_cameras": n_views,
         "extrinsics": [{"camera_id": r["camera_id"], "matrix": np.asarray(r["matrix"]).tolist()}
                        for r in extrinsics],
         "intrinsics": intrinsics}, indent=2) + "\n", encoding="utf-8", newline="\n")
    contact_sheet(frames, records, out / "contact_sheet.jpg")
    print(f"✅ {n_views} distinct views, min pairwise MAE={min(pairwise_mae):.3f}")
    print(f"✅ Inspect: {out / 'contact_sheet.jpg'}")
    print(f"✅ Inference input: {out}")


if __name__ == "__main__":
    main()
