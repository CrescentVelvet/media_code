#!/usr/bin/env python3
"""detect_faces.py — 阶段一：全图人脸检测，输出 bbox + 关键点 + 置信度。

与 vggt_human 的 detect_face_landmarks.py 不同：那个是「先用 SAM3 face mask 定位再裁剪检测」，
本脚本是**全图直接检测**，归属由阶段二（match_faces.py）用三层判据决定。
这样做的原因：mask 引导会把 mask 自身的错误带进来，且 mask 漏了就整帧丢。

关键点（MediaPipe FaceDetection 输出 6 点：双眼/鼻尖/嘴中/双耳廓）**下游不使用**——
设计文档明确 5 点 landmark 只是副产物，仿射对齐裁剪用的是 bbox，观测用阶段三的 468 点。
这里存下来只为调试可视化。

检测器选型：默认 MediaPipe FaceDetection（full-range，零额外依赖，与阶段三的
MediaPipe 468 同源）。远景小脸召回不够时换 SCRFD/RetinaFace——换这里即可，
下游只消费 bbox。

Env:
  IMAGES_DIR      输入图像目录（默认 $SOURCE_DIR/images）
  OUT_JSON        输出（默认 $RESULTS_DIR/02_faces/face_boxes.json）
  MIN_DET_CONF    检测阈值（默认 0.5）
  MODEL_SELECTION 0=近景(2m) / 1=full-range(5m，远景小脸更好)（默认 1）
  MAX_FACES       单帧最多人脸（默认 6）
"""
import os
import sys
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def log(m):
    print(m, flush=True)


def main():
    source_dir = os.environ.get("SOURCE_DIR", "")
    results_dir = os.environ.get("RESULTS_DIR", "")
    images_dir = Path(os.environ.get("IMAGES_DIR", f"{source_dir}/images"))
    out_json = Path(os.environ.get(
        "OUT_JSON", f"{results_dir}/02_faces/face_boxes.json"))
    min_conf = float(os.environ.get("MIN_DET_CONF", "0.5"))
    model_sel = int(os.environ.get("MODEL_SELECTION", "1"))
    max_faces = int(os.environ.get("MAX_FACES", "6"))

    if not images_dir.is_dir():
        sys.exit(f"❌ 图像目录不存在: {images_dir}")

    imgs = sorted(p for p in images_dir.iterdir() if p.suffix in IMG_EXTS)
    if not imgs:
        sys.exit(f"❌ 目录里没有图像: {images_dir}")
    log(f"🎯 [阶段一] 人脸检测: {len(imgs)} 张图 -> {out_json}")

    import mediapipe as mp
    det = mp.solutions.face_detection.FaceDetection(
        model_selection=model_sel,
        min_detection_confidence=min_conf,
    )

    out = {"frames": {}}
    n_face = 0
    t0 = time.time()
    for i, img_path in enumerate(imgs):
        stem = img_path.stem
        img = np.array(Image.open(img_path).convert("RGB"))
        H, W = img.shape[:2]
        rec = {"image": img_path.name, "W": W, "H": H, "faces": []}

        res = det.process(img)
        if res.detections:
            # MediaPipe 不保证按置信度排序，且 MAX_FACES 需自己截断
            dets = sorted(res.detections,
                          key=lambda d: -d.score[0])[:max_faces]
            for d in dets:
                bb = d.location_data.relative_bounding_box
                x0 = bb.xmin * W
                y0 = bb.ymin * H
                w = bb.width * W
                h = bb.height * H
                bbox = [float(x0), float(y0), float(x0 + w), float(y0 + h)]
                kps = []
                for kp in d.location_data.relative_keypoints:
                    kps.append([float(kp.x * W), float(kp.y * H)])
                rec["faces"].append({
                    "bbox": [max(0.0, v) for v in bbox],
                    "kps": kps,
                    "score": float(d.score[0]),
                })
        n_face += len(rec["faces"])
        out["frames"][stem] = rec

        if (i + 1) % 50 == 0 or i == len(imgs) - 1:
            el = time.time() - t0
            log(f"  … {i+1}/{len(imgs)} ({el:.0f}s, {el/(i+1):.3f}s/张) "
                f"累计人脸 {n_face}")

    det.close()

    n_with = sum(1 for r in out["frames"].values() if r["faces"])
    out["meta"] = {
        "images_dir": str(images_dir),
        "min_det_conf": min_conf,
        "model_selection": model_sel,
        "max_faces": max_faces,
        "n_frames": len(out["frames"]),
        "n_frames_with_face": n_with,
        "n_faces_total": n_face,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out))
    log(f"💾 {out_json}")
    log(f"📊 {n_with}/{len(imgs)} 帧检出人脸，共 {n_face} 张脸")
    if n_with < len(imgs) * 0.5:
        log("  ⚠️ 检出率偏低：确认 MODEL_SELECTION=1（full-range）"
            "或降低 MIN_DET_CONF；小脸场景建议换 SCRFD/RetinaFace")


if __name__ == "__main__":
    main()
