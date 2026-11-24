#!/usr/bin/env python3
"""Run official Depth Pro 2 buffered streaming on every frame of one video.

Env vars: VIDEO_PATH, OUT_DIR, DEPTH_PRO_2_DIR, CKPT_PATH, FOV_CKPT_PATH,
WINDOW (16), OVERLAP (4), FRAME_STRIDE (1), MAX_SIDE (1280), OUTPUT_FPS (source).
No source-video modification. No camera poses or world-space registration.
"""
import dataclasses
import hashlib
import json
import os
import time
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import torch
from PIL import Image


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")


def main():
    import sys
    sys.path.insert(0, str(Path(os.environ["DEPTH_PRO_2_DIR"]) / "src"))
    from depth_pro_2 import PRESETS, create_model_and_transforms

    video_path = Path(os.environ["VIDEO_PATH"])
    out = Path(os.environ["OUT_DIR"])
    window = int(os.environ.get("WINDOW", "16"))
    overlap = int(os.environ.get("OVERLAP", "4"))
    stride = int(os.environ.get("FRAME_STRIDE", "1"))
    max_side = int(os.environ.get("MAX_SIDE", "1280"))
    assert 0 < stride and 0 <= overlap < window
    assert video_path.is_file() and not video_path.is_symlink()
    for folder in (out, out / "frames", out / "depth", out / "normalized_inverse_depth", out / "pcd", out / "previews"):
        assert not folder.is_symlink()
        folder.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), video_path
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    source_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    scale = min(1.0, max_side / max(width, height))
    ow, oh = round(width * scale), round(height * scale)
    source = {"path": str(video_path), "sha256": sha256(video_path), "width": width, "height": height,
              "fps": source_fps, "frame_count": source_count, "duration_s": source_count / source_fps}
    write_json(out / "video_source.json", source)
    write_json(out / "ffmpeg_path.json", {"ffmpeg": imageio_ffmpeg.get_ffmpeg_exe()})

    config = dataclasses.replace(PRESETS["depthpro2-h"], checkpoint_uri=os.environ["CKPT_PATH"],
                                fov_checkpoint_uri=os.environ["FOV_CKPT_PATH"])
    model, transform = create_model_and_transforms(config, device="cuda")
    frame_indices, selected = [], []
    for i in range(source_count):
        ok, bgr = cap.read()
        if not ok:
            raise ValueError(f"Decode stopped before declared frame count: {i}/{source_count}")
        if i % stride == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if (ow, oh) != (width, height):
                rgb = np.asarray(Image.fromarray(rgb).resize((ow, oh), Image.Resampling.LANCZOS))
            Image.fromarray(rgb).save(out / "frames" / f"{len(frame_indices):06d}.jpg", quality=96)
            frame_indices.append(i)
            selected.append(torch.from_numpy(rgb))
    cap.release()
    assert selected and len(selected) == len(frame_indices)
    # Defer the CPU->GPU copy to the lazy generator: a 12s 720p video otherwise
    # unnecessarily occupies gigabytes of CPU memory before inference.
    def inputs():
        for rgb in selected:
            yield transform(rgb.numpy())
    metadata = {
        "model": "depthpro2-h", "official_repo": os.environ["DEPTH_PRO_2_DIR"],
        "official_revision": os.environ.get("OFFICIAL_REVISION", "unrecorded"),
        "checkpoint_sha256": sha256(Path(os.environ["CKPT_PATH"])),
        "fov_checkpoint_sha256": sha256(Path(os.environ["FOV_CKPT_PATH"])),
        "mode": "buffered", "window": window, "overlap": overlap,
        "frame_stride": stride, "output_width": ow, "output_height": oh,
        "focal_source": "official first-window FOV estimator; no known K; one focal shared by video",
        "camera_poses": "unavailable; each exported PCD is local camera coordinates, not world-aligned",
        "source_video_sha256": source["sha256"], "source_frame_indices": frame_indices,
        "frames": [],
    }
    write_json(out / "inference.json", metadata)
    start = time.time()
    total = 0
    peak = 0.0
    for i, prediction in enumerate(model.infer_video(inputs(), window=window, overlap=overlap, mode="buffered")):
        inv = prediction["inverse_depth"].float().cpu().numpy()
        depth = prediction["depth"].float().cpu().numpy()
        focal = float(prediction["focallength_px"].cpu())
        assert inv.shape == depth.shape == (oh, ow)
        assert np.all(inv >= 0)  # ReLU head: sky may legitimately be inv=0, Z=inf.
        assert np.all(depth > 0)
        expected = np.divide(focal, ow * inv, out=np.full_like(inv, np.inf, dtype=np.float32), where=inv > 0)
        finite = np.isfinite(depth) & (depth > 0)
        assert np.allclose(depth[finite], expected[finite], rtol=1e-5, atol=0)
        assert np.all(np.isinf(depth[~finite]))
        np.savez_compressed(out / "depth" / f"{i:06d}.npz", depth=depth.astype(np.float32),
                            source_index=frame_indices[i], focal_px=focal)
        np.save(out / "normalized_inverse_depth" / f"{i:06d}.npy", inv.astype(np.float32))
        finite = depth[np.isfinite(depth) & (depth > 0)]
        assert len(finite) > 0
        record = {"index": i, "source_index": frame_indices[i], "focal_px": focal,
                  "nonfinite_depth_fraction": float((~np.isfinite(depth)).mean()),
                  "depth_quantiles_m": {str(q): float(np.quantile(finite, q)) for q in (0.01, 0.5, 0.99)}}
        metadata["frames"].append(record)
        total += 1
        if i % 10 == 0 or i == len(selected) - 1:
            peak = max(peak, torch.cuda.max_memory_allocated() / 2**30)
            metadata["processed_frames"] = total
            metadata["inference_seconds_so_far"] = time.time() - start
            metadata["peak_vram_gib_so_far"] = peak
            write_json(out / "inference.json", metadata)
        print(f"✅ {i + 1}/{len(selected)} | {time.time() - start:.1f}s | focal={focal:.2f}px | VRAM={peak:.2f}GiB", flush=True)
    assert total == len(selected)
    metadata.update(processed_frames=total, inference_seconds=time.time() - start,
                    peak_vram_gib=torch.cuda.max_memory_allocated() / 2**30,
                    output_fps=float(os.environ.get("OUTPUT_FPS", source_fps / stride)))
    write_json(out / "inference.json", metadata)
    print(f"🎉 Video depth: {total} frames in {metadata['inference_seconds']:.1f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
