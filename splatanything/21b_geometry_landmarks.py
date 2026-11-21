#!/usr/bin/env python3
"""21b_geometry_landmarks.py -- N0 的原生 GT 图 FaceMesh 地标。

只检测，不训练，不改图片。使用 manifest 的 file_rgb，覆盖 source/target。
输出 468 个原生像素坐标、检测尺度及带 ROI 标记的审核图。sam3 env 已有
mediapipe 0.10.14；本脚本不加载 SAM3。
"""
import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw

REGIONS = {
    "nose_tip": [4, 1, 2, 168, 197],
    "nose_ala": [48, 64, 98, 97, 278, 294, 327, 326],
    "eye_l": [263, 362, 386, 387, 373, 380, 374, 398, 466, 388],
    "eye_r": [33, 133, 159, 160, 158, 144, 145, 153, 246, 161],
    "brow_l": [105, 107, 66, 52, 65, 70, 63],
    "brow_r": [334, 336, 296, 282, 295, 300, 293],
}


def detect(rgb):
    h, w = rgb.shape[:2]
    best = None
    for scale in (1.0, 0.75, 0.5, 0.35, 0.25):
        small = rgb if scale == 1 else cv2.resize(
            rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                             min_detection_confidence=0.3) as fm:
            out = fm.process(small)
        if not out.multi_face_landmarks:
            continue
        points = np.array([[p.x * w, p.y * h]
                           for p in out.multi_face_landmarks[0].landmark], np.float64)
        oval = sorted({i for edge in mp.solutions.face_mesh.FACEMESH_FACE_OVAL for i in edge})
        side = float(np.max(np.ptp(points[oval], axis=0)))
        if best is None or side > best[0]:
            best = (side, scale, points)
    if best is None:
        raise RuntimeError("No FaceMesh; refusing fixed-box or approximate landmarks")
    return best[2], best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--source-scene", required=True)
    ap.add_argument("--target-scene", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pkg = Path(args.pkg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((pkg / "cameras.json").read_text(encoding="utf-8"))
    cameras = {int(c["id"]): c for c in manifest["cameras"]}
    priors = []
    for p in (Path(args.source_scene) / "prior_camera.json",
              Path(args.target_scene) / "prior_camera.json"):
        priors.extend(json.loads(p.read_text(encoding="utf-8"))["extrinsics"])
    result = {"coordinate": "native GT RGB pixel; integer index corresponds to pixel center",
              "views": {}}
    for c in priors:
        stem = c["camera_id"]
        numeric = int(stem.rsplit("_", 1)[1])
        mc = cameras[numeric]
        im = Image.open(pkg / mc["file_rgb"]).convert("RGB")
        rgb = np.asarray(im)
        points, factor = detect(rgb)
        result["views"][stem] = {"id": numeric, "file_rgb": mc["file_rgb"],
                                 "file_depth": mc["file_depth"],
                                 "detection_scale": factor,
                                 "points": points.tolist()}
        annotated = im.copy()
        draw = ImageDraw.Draw(annotated)
        for name, inds in REGIONS.items():
            color = (255, 40, 40) if name.startswith("nose") else (20, 160, 255)
            for x, y in points[inds]:
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
            draw.text(tuple(points[inds].mean(axis=0)), name, fill=color)
        annotated.save(out / (stem + "_landmarks.png"))
        print(stem, "scale", factor, "nose", points[4].tolist(), flush=True)
    with open(out / "landmarks_native.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("DONE", out)


if __name__ == "__main__":
    main()
