#!/usr/bin/env python3
"""Select three frames, verify RGB identity, and export pinhole cameras.

Env vars: SCENE (human|router), RAW_DIR, POSE_DIR (router), OUT_DIR, MAX_SIDE.
Human poses: Remy c2w OpenGL -> c2w OpenCV with diag(1,-1,-1,1).
Router poses: COLMAP w2c OpenCV, recovered from a byte-identical longer sequence.
"""
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ])


def colmap_records(pose_dir, wanted):
    lines = (pose_dir / "sparse/0/cameras.txt").read_text().splitlines()
    row = next(s.split() for s in lines if s and not s.startswith("#"))
    if row[1] != "PINHOLE":
        raise ValueError("Only undistorted PINHOLE COLMAP cameras are supported")
    width, height = int(row[2]), int(row[3])
    fx, fy, cx, cy = map(float, row[4:8])
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    records = {}
    for line in (pose_dir / "sparse/0/images.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) == 10 and fields[-1] in wanted:
            name = fields[-1]
            R = qvec2rotmat(np.array(list(map(float, fields[1:5]))))
            t = np.array(list(map(float, fields[5:8])))
            c2w = np.eye(4)
            c2w[:3, :3] = R.T
            c2w[:3, 3] = -R.T @ t
            records[name] = {
                "source": str(pose_dir / "image" / name), "K": K.tolist(),
                "dist": [0, 0, 0, 0, 0], "c2w_cv": c2w.tolist(),
                "pose_source": "COLMAP w2c OpenCV; arbitrary SfM metric unit",
            }
    return records, (width, height)


def human_records(raw_dir):
    meta = json.loads((raw_dir / "transforms.json").read_text(encoding="utf-8"))
    frames = sorted(meta["frames"], key=lambda x: x["file_path"])
    indices = [0, len(frames) // 2, len(frames) - 1]
    records = {}
    for i in indices:
        f = frames[i]
        name = Path(f["file_path"]).name
        K = np.array([[f["fl_x"], 0, f["cx"]], [0, f["fl_y"], f["cy"]], [0, 0, 1]])
        c2w = np.asarray(f["transform_matrix"]) @ np.diag([1, -1, -1, 1])
        records[name] = {
            "source": str(raw_dir / f["file_path"]), "K": K.tolist(),
            "dist": [f["k1"], f["k2"], f["p1"], f["p2"], f["k3"]],
            "c2w_cv": c2w.tolist(), "source_index": i,
            "pose_source": "Remy transform_matrix=c2w OpenGL, converted to c2w OpenCV",
        }
    return records, (int(frames[0]["w"]), int(frames[0]["h"]))


def main():
    scene = os.environ["SCENE"]
    raw = Path(os.environ["RAW_DIR"])
    out = Path(os.environ["OUT_DIR"])
    max_side = int(os.environ.get("MAX_SIDE", "1536"))
    if scene == "human":
        records, size = human_records(raw)
    elif scene == "router":
        pose_dir = Path(os.environ["POSE_DIR"])
        names = [p.name for p in sorted(raw.glob("*.jpg"))]
        if len(names) != 3:
            raise ValueError(f"Expected exactly 3 router JPGs, got {names}")
        records, size = colmap_records(pose_dir, set(names))
        for name in names:
            if sha256(raw / name) != sha256(pose_dir / "image" / name):
                raise ValueError(f"RGB differs from the pose sequence: {name}")
    else:
        raise ValueError("SCENE must be human or router")
    if len(records) != 3:
        raise ValueError(f"Expected 3 cameras, got {len(records)}")
    (out / "rgb").mkdir(parents=True, exist_ok=True)
    manifest = {
        "scene": scene, "source_dir": str(raw), "source_size": list(size),
        "max_side": max_side, "coordinate_convention": "OpenCV: x right, y down, z forward",
        "metric_scale": scene == "human", "frames": [],
    }
    for i, (name, rec) in enumerate(sorted(records.items())):
        im = Image.open(rec["source"]).convert("RGB")
        if im.size != tuple(size):
            raise ValueError(f"Unexpected RGB size: {name}: {im.size} vs {size}")
        K = np.asarray(rec["K"], dtype=np.float64)
        dist = np.asarray(rec["dist"], dtype=np.float64)
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, size, 0)
        rgb = cv2.undistort(np.asarray(im), K, dist, None, new_K)
        x, y, w, h = map(int, roi)
        rgb = rgb[y:y + h, x:x + w]
        new_K[0, 2] -= x
        new_K[1, 2] -= y
        scale = min(1.0, max_side / max(w, h))
        ow, oh = round(w * scale), round(h * scale)
        sx, sy = ow / w, oh / h
        rgb = np.asarray(Image.fromarray(rgb).resize((ow, oh), Image.Resampling.LANCZOS))
        new_K[0, :] *= sx
        new_K[1, :] *= sy
        rgb_path = out / "rgb" / f"{i:02d}.png"
        Image.fromarray(rgb).save(rgb_path)
        c2w = np.asarray(rec["c2w_cv"], dtype=np.float64)
        w2c = np.linalg.inv(c2w)
        manifest["frames"].append({
            "index": i, "name": name, "source": rec["source"],
            "source_sha256": sha256(Path(rec["source"])),
            "source_index": rec.get("source_index"), "width": ow, "height": oh,
            "K": new_K.tolist(), "c2w": c2w.tolist(), "w2c": w2c.tolist(),
            "undistort_roi": [x, y, w, h], "resize_scale": [sx, sy],
            "pose_source": rec["pose_source"],
        })
        print(f"✅ {scene} {i}: {name}, {ow}x{oh}, fx={new_K[0, 0]:.3f}")
    reference_pcd = raw / "pcd.ply"
    if reference_pcd.is_file():
        manifest["reference_pcd"] = str(reference_pcd)
    write_json(out / "manifest.json", manifest)
    print(f"🎉 Camera manifest: {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
