#!/usr/bin/env python3
"""detect_face_landmarks.py — 多视角人脸 landmark 检测（MediaPipe FaceMesh 468）。

头/身/景拆分 step 2：给 3DMM 拟合提供 2D 观测。

为什么不用「整图跑一遍」：
  - 场景里有 3 个人，整图检测会返回 3 张脸，无法区分属于哪个 SAM3 track；
  - 侧脸/小脸整图检测率低。
所以：**用 SAM3 的 face mask 定位人脸框 → 裁剪 → 在裁剪区里检测**，
多张脸时用 mask 质心挑最近的那个。这样 landmark 天然带了 pid 归属。

输出 landmark 是 468 点（与 canonical_face_model 一致），
refine_landmarks 关闭（开启会多 10 个虹膜点，与模板对不上）。

Output JSON:
  {meta:{...}, frames:{<stem>: {p00: {ok, lm:[[x,y]...], mask_centroid, bbox}, ...}}}

Env:
  IMAGES_DIR    输入图像（默认 $RESULTS_DIR/03_source/images）
  MASKS_DIR     SAM3 face mask（默认 $RESULTS_DIR/03_sam3_face_masks）
  OUT_JSON      输出（默认 $RESULTS_DIR/03e_head_3dmm/face_landmarks.json）
  PAD_RATIO     crop 外扩比例（默认 0.6，相对 bbox 尺寸）
  MIN_MASK_PX   mask 像素数下限（默认 400）
  MAX_FACES     MediaPipe 单帧最多人脸（默认 3）
  MIN_DET_CONF  MediaPipe 检测阈值（默认 0.5）
  FALLBACK_FULL 裁剪失败时退回整图检测（默认 1）
"""
import os
import sys
import json
import glob
import time
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def log(msg):
    print(msg, flush=True)


def mask_bbox(mask_path, min_px):
    """读 mask → (bbox, centroid, n_px)；mask 太小返回 None。"""
    m = np.array(Image.open(mask_path).convert("L"))
    ys, xs = np.nonzero(m > 127)
    if len(xs) < min_px:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return (x0, y0, x1, y1), (float(xs.mean()), float(ys.mean())), len(xs)


def crop_with_pad(img, bbox, pad_ratio):
    x0, y0, x1, y1 = bbox
    H, W = img.shape[:2]
    pw = int((x1 - x0) * 0.5 * pad_ratio)
    ph = int((y1 - y0) * 0.5 * pad_ratio)
    cx0, cy0 = max(0, x0 - pw), max(0, y0 - ph)
    cx1, cy1 = min(W, x1 + pw), min(H, y1 + ph)
    return img[cy0:cy1, cx0:cx1], (cx0, cy0, cx1, cy1)


def pick_nearest(face_lms, img_shape, target_xy):
    """多张脸时挑质心离 target 最近的（target 是全图坐标 → 需转 crop 坐标）。"""
    H, W = img_shape[:2]
    best, best_d = None, np.inf
    for fl in face_lms:
        pts = np.array([[lm.x * W, lm.y * H] for lm in fl.landmark])
        c = pts.mean(0)
        d = np.linalg.norm(c - target_xy)
        if d < best_d:
            best, best_d = pts, d
    return best, best_d


def main():
    results_dir = os.environ.get("RESULTS_DIR", "")
    images_dir = Path(os.environ.get("IMAGES_DIR", f"{results_dir}/03_source/images"))
    masks_dir = Path(os.environ.get("MASKS_DIR", f"{results_dir}/03_sam3_face_masks"))
    out_json = Path(
        os.environ.get("OUT_JSON", f"{results_dir}/03e_head_3dmm/face_landmarks.json")
    )
    pad_ratio = float(os.environ.get("PAD_RATIO", "0.6"))
    min_mask_px = int(os.environ.get("MIN_MASK_PX", "400"))
    max_faces = int(os.environ.get("MAX_FACES", "3"))
    min_conf = float(os.environ.get("MIN_DET_CONF", "0.5"))
    fallback_full = int(os.environ.get("FALLBACK_FULL", "1"))

    if not images_dir.is_dir():
        log(f"❌ 图像目录不存在: {images_dir}")
        return 1
    if not masks_dir.is_dir():
        log(f"❌ mask 目录不存在: {masks_dir}")
        return 1

    import mediapipe as mp

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=max_faces,
        refine_landmarks=False,  # 468 点（与 canonical_face_model 对齐）
        min_detection_confidence=min_conf,
    )

    # 收集 (pid, stem) 任务
    tasks = {}
    for mp_path in sorted(masks_dir.glob("*.p*.mask.png")):
        name = mp_path.name
        stem = name.split(".p")[0]
        pid = name.split(".p")[1].split(".")[0]
        tasks.setdefault(stem, []).append((pid, mp_path))
    log(f"🎯 待处理: {len(tasks)} 帧, {sum(len(v) for v in tasks.values())} 个 (frame,pid) 对")

    out = {"frames": {}}
    stats = {}
    t0 = time.time()
    for i, (stem, items) in enumerate(sorted(tasks.items())):
        # 找图像（stem 可能对应 .jpg/.png）
        img_path = None
        for ext in IMG_EXTS:
            p = images_dir / f"{stem}{ext}"
            if p.exists():
                img_path = p
                break
        if img_path is None:
            log(f"  ⚠️ 无图像: {stem}")
            continue
        img = np.array(Image.open(img_path).convert("RGB"))
        H, W = img.shape[:2]
        out["frames"][stem] = {"image": img_path.name, "W": W, "H": H, "persons": {}}

        for pid, mp_path in items:
            rec = {"ok": False, "lm": None, "reason": ""}
            bbox_info = mask_bbox(mp_path, min_mask_px)
            if bbox_info is None:
                rec["reason"] = "mask_too_small"
                out["frames"][stem]["persons"][pid] = rec
                stats.setdefault(pid, {}).setdefault("mask_small", 0)
                stats[pid]["mask_small"] += 1
                continue
            bbox, centroid, npx = bbox_info
            rec["mask_centroid"] = centroid
            rec["mask_bbox"] = bbox
            rec["mask_px"] = int(npx)

            crop, (cx0, cy0, cx1, cy1) = crop_with_pad(img, bbox, pad_ratio)
            if crop.size == 0 or min(crop.shape[:2]) < 16:
                rec["reason"] = "crop_too_small"
                out["frames"][stem]["persons"][pid] = rec
                continue

            # 在 crop 坐标里的 mask 质心（用于多脸挑选）
            tgt = np.array([centroid[0] - cx0, centroid[1] - cy0])
            res = face_mesh.process(crop)
            pts = None
            if res.multi_face_landmarks:
                pts, _ = pick_nearest(res.multi_face_landmarks, crop.shape, tgt)
                rec["source"] = "crop"
            if pts is None and fallback_full:
                res2 = face_mesh.process(img)
                if res2.multi_face_landmarks:
                    pts, _ = pick_nearest(
                        res2.multi_face_landmarks, img.shape, np.array(centroid)
                    )
                    rec["source"] = "full"
                    cx0, cy0 = 0, 0
            if pts is None:
                rec["reason"] = "no_detection"
                out["frames"][stem]["persons"][pid] = rec
                stats.setdefault(pid, {}).setdefault("no_detection", 0)
                stats[pid]["no_detection"] += 1
                continue

            pts = pts + np.array([cx0, cy0])  # crop → 全图
            rec["ok"] = True
            rec["lm"] = [[float(x), float(y)] for x, y in pts]
            stats.setdefault(pid, {}).setdefault("ok", 0)
            stats[pid]["ok"] += 1
            out["frames"][stem]["persons"][pid] = rec

        if (i + 1) % 20 == 0 or i == len(tasks) - 1:
            el = time.time() - t0
            log(
                f"  … {i+1}/{len(tasks)} 帧 ({el:.0f}s, {el/(i+1):.2f}s/帧) "
                f"ok={ {p: s.get('ok',0) for p, s in sorted(stats.items())} }"
            )

    out["meta"] = {
        "images_dir": str(images_dir),
        "masks_dir": str(masks_dir),
        "pad_ratio": pad_ratio,
        "min_mask_px": min_mask_px,
        "n_frames": len(out["frames"]),
        "stats": stats,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f)
    log(f"💾 {out_json}")
    log("📊 每 pid 成功帧数:")
    for pid in sorted(stats):
        s = stats[pid]
        log(f"   {pid}: ok={s.get('ok',0)} no_detection={s.get('no_detection',0)} "
            f"mask_small={s.get('mask_small',0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
