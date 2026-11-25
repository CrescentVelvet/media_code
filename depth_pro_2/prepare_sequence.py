#!/usr/bin/env python3
"""Decode and calibrate all 82 sequential Remy cameras without altering the source.

Env vars: RAW_DIR, OUT_DIR, MAX_SIDE (1536), PLAYBACK_FPS (12).
The saved cameras, not DP2, provide world registration. No capture timestamps are
invented: PLAYBACK_FPS is a display rate for the ordered still sequence.
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener
from scipy.spatial.transform import Rotation

from io_utils import sha256, write_json

register_heif_opener()


def main():
    raw = Path(os.environ["RAW_DIR"])
    out = Path(os.environ["OUT_DIR"])
    max_side = int(os.environ.get("MAX_SIDE", "1536"))
    fps = float(os.environ.get("PLAYBACK_FPS", "12"))
    assert raw.is_dir() and not raw.is_symlink() and fps > 0
    metadata = json.loads((raw / "transforms.json").read_text(encoding="utf-8"))
    records = sorted(metadata["frames"], key=lambda f: f["file_path"])
    if len(records) < 16:
        raise ValueError("This pipeline expects a continuous multi-view sequence")
    for folder in (out, out / "frames"):
        assert not folder.is_symlink()
        folder.mkdir(parents=True, exist_ok=True)
    manifest = {
        "scene": "human_sequence_82", "source_dir": str(raw),
        "transforms_sha256": sha256(raw / "transforms.json"),
        "coordinate_convention": "OpenCV; c2w_cv = supplied OpenGL c2w @ diag(1,-1,-1,1)",
        "frame_count": len(records), "playback_fps": fps,
        "fps_note": "Ordered still-camera playback; no capture timestamps or physical motion speed inferred.",
        "frames": [], "adjacent_motions": [],
    }
    previous = None
    for index, record in enumerate(records):
        source = raw / record["file_path"]
        rgb = np.asarray(Image.open(source).convert("RGB"))
        size = (int(record["w"]), int(record["h"]))
        if rgb.shape[1::-1] != size:
            raise ValueError(f"RGB/metadata size mismatch: {source.name}")
        K = np.array([[record["fl_x"], 0, record["cx"]],
                      [0, record["fl_y"], record["cy"]], [0, 0, 1]], np.float64)
        dist = np.array([record[k] for k in ("k1", "k2", "p1", "p2", "k3")], np.float64)
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, size, 0)
        rgb = cv2.undistort(rgb, K, dist, None, new_K)
        x, y, w, h = map(int, roi)
        rgb = rgb[y:y + h, x:x + w]
        new_K[0, 2] -= x
        new_K[1, 2] -= y
        scale = min(1.0, max_side / max(w, h))
        ow, oh = round(w * scale), round(h * scale)
        new_K[0, :] *= ow / w
        new_K[1, :] *= oh / h
        rgb = np.asarray(Image.fromarray(rgb).resize((ow, oh), Image.Resampling.LANCZOS))
        Image.fromarray(rgb).save(out / "frames" / f"{index:06d}.jpg", quality=96)
        c2w = np.asarray(record["transform_matrix"], np.float64) @ np.diag([1, -1, -1, 1])
        R = c2w[:3, :3]
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(R), 1, atol=1e-5):
            raise ValueError(f"Invalid c2w rotation: {source.name}")
        frame = {
            "index": index, "name": source.name, "source": str(source),
            "source_sha256": sha256(source), "width": ow, "height": oh,
            "K": new_K.tolist(), "c2w": c2w.tolist(), "w2c": np.linalg.inv(c2w).tolist(),
            "undistort_roi": [x, y, w, h], "dist": dist.tolist(),
            "pose_source": "Supplied transforms.json; post-inference world transform, not DP2 input",
        }
        if previous is not None:
            relative = np.linalg.inv(previous) @ c2w
            manifest["adjacent_motions"].append({
                "from": index - 1, "to": index,
                "baseline_m": float(np.linalg.norm(c2w[:3, 3] - previous[:3, 3])),
                "rotation_degrees": float(np.degrees(np.linalg.norm(Rotation.from_matrix(relative[:3, :3]).as_rotvec()))),
            })
        manifest["frames"].append(frame)
        previous = c2w
        print(f"✅ {index + 1}/{len(records)}: {source.name}, {ow}x{oh}", flush=True)
    if len({(f["width"], f["height"]) for f in manifest["frames"]}) != 1:
        raise ValueError("Sequence inference requires one frame size")
    write_json(out / "manifest.json", manifest)
    print(f"🎉 {len(records)} continuous calibrated cameras -> {out}", flush=True)


if __name__ == "__main__":
    main()
