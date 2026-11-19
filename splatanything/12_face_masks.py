#!/usr/bin/env python3
"""12_face_masks.py — 生成脸部局部优化所需的 2D mask（face / band / person）。

对应 da3_human/EXPERIMENT_PLAN.md P0-1："只从 source 获取人物 mask、脸/头发分区与
遮挡标注"。产物全部在 SA 处理后的 rgb/ 分辨率（如 756²）上，与渲染像素一一对齐：

  <out>/<view_id>_face.png    脸部核心 mask（mediapipe FaceMesh 脸轮廓凸包；
                              失败时退化为 face_detection 框内椭圆）
  <out>/<view_id>_band.png    边界带 = dilate(face) & ~erode(face)（发际/下颌过渡区，
                              损失里单独计权）
  <out>/<view_id>_person.png  人物 mask（SAM3 "person" prompt，外层保护区）
  <out>/<view_id>_overlay.png 审核叠加图（face 绿 / band 黄 / person 蓝描边）
  <out>/masks_meta.json       每视角检测分数、框、面积占比

运行环境：sam3 env（mediapipe 0.10.14 + sam3）。用法：
  python 12_face_masks.py --result <02输出目录> [--out <目录>] [--no-sam3]

--no-sam3 只用 mediapipe（face/band 仍有，person 退化为 dilate(face) 的椭圆外接，
仅用于应急；正式实验必须有 SAM3 person mask）。
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SAM3_CKPT = os.environ.get("SAM3_CKPT", "/mnt/d/wheel/vggt_human_ms/sam3/sam3.pt")
SAM3_BPE = os.environ.get(
    "SAM3_BPE", "/home/velvet/repos/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz")
MIN_SCORE = float(os.environ.get("MIN_SCORE", "0.5"))


# ---------------------------------------------------------------- mediapipe
def detect_face_mesh(arr):
    """返回 (face_mask[H,W] bool, bbox(x0,y0,x1,y1), score, source)。"""
    import mediapipe as mp
    h, w = arr.shape[:2]
    # 首选 FaceMesh：468 点的脸轮廓比检测框更贴合发际/下颌
    fm = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=False,
        min_detection_confidence=0.3)
    out = fm.process(arr)
    fm.close()
    if out.multi_face_landmarks:
        lm = out.multi_face_landmarks[0].landmark
        # FACEMESH_FACE_OVAL 的索引（mediapipe 常量，36 点轮廓环）
        oval = sorted({i for conn in mp.solutions.face_mesh.FACEMESH_FACE_OVAL
                       for i in conn})
        pts = np.array([[lm[i].x * w, lm[i].y * h] for i in oval], np.int32)
        hull = cv2.convexHull(pts)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask, hull, 1)
        x, y, bw, bh = cv2.boundingRect(hull)
        return mask.astype(bool), (x, y, x + bw, y + bh), 1.0, "face_mesh"
    # 退化：检测框内接椭圆
    det = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.3)
    res = det.process(arr)
    det.close()
    if not res.detections:
        return None, None, 0.0, "none"
    d = max(res.detections, key=lambda x: float(x.score[0]))
    b = d.location_data.relative_bounding_box
    x0, y0 = int(b.xmin * w), int(b.ymin * h)
    x1, y1 = int((b.xmin + b.width) * w), int((b.ymin + b.height) * h)
    mask = np.zeros((h, w), np.uint8)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    ax, ay = max(1, (x1 - x0) // 2), max(1, (y1 - y0) // 2)
    cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 1, -1)
    return mask.astype(bool), (x0, y0, x1, y1), float(d.score[0]), "detection_box"


# ------------------------------------------------------------------- SAM3
def sam3_person_masks(images):
    """对一组 PIL 图跑 SAM3 'person'，返回 list[bool mask or None]。"""
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    t0 = time.time()
    model = build_sam3_image_model(
        bpe_path=SAM3_BPE, checkpoint_path=SAM3_CKPT,
        load_from_HF=False, device="cuda", eval_mode=True)
    processor = Sam3Processor(model)
    print(f"  🤖 SAM3 ready in {time.time()-t0:.1f}s")
    masks = []
    for name, img in images:
        state = processor.set_image(img)
        out = processor.set_text_prompt(state=state, prompt="person")
        best, best_s = None, 0.0
        for i in range(len(out["scores"])):
            s = float(out["scores"][i])
            if s < MIN_SCORE:
                continue
            m = out["masks"][i]
            if hasattr(m, "detach"):
                m = m.detach().cpu().numpy()
            m = np.asarray(m).squeeze().astype(bool)
            if m.sum() == 0:
                continue
            if s > best_s:
                best, best_s = m, s
        print(f"    {name}: person score={best_s:.3f} "
              f"area={100*best.sum()/best.size:.1f}%" if best is not None
              else f"    {name}: ⚠️ no person >= {MIN_SCORE}")
        masks.append(best)
    return masks


def write_mask(path, m):
    Image.fromarray((m.astype(np.uint8)) * 255).save(str(path))


def overlay(arr, face, band, person):
    """face 绿填充半透明 / band 黄 / person 蓝描边。"""
    img = arr.copy().astype(np.float32)
    if face is not None:
        img[face] = 0.65 * img[face] + 0.35 * np.array([0, 255, 0], np.float32)
    if band is not None:
        img[band] = 0.5 * img[band] + 0.5 * np.array([255, 220, 0], np.float32)
    img = img.astype(np.uint8)
    if person is not None:
        cnt, _ = cv2.findContours(person.astype(np.uint8), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnt, -1, (255, 80, 0), 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True, help="02 的输出目录（含 rgb/）")
    ap.add_argument("--out", default=None, help="默认 <result>/face_opt/masks")
    ap.add_argument("--no-sam3", action="store_true")
    ap.add_argument("--band-inner", type=int, default=6, help="band 内缩 px")
    ap.add_argument("--band-outer", type=int, default=10, help="band 外扩 px")
    args = ap.parse_args()

    res = Path(args.result)
    rgb_dir = res / "rgb"
    out = Path(args.out) if args.out else res / "face_opt" / "masks"
    out.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in rgb_dir.glob("*.png"))
    if not names:
        sys.exit(f"❌ 缺 rgb/*.png：{rgb_dir}")
    print(f"🚀 [12] 生成 mask | {len(names)} 视角 | out={out}")

    imgs = [(n, Image.open(rgb_dir / n).convert("RGB")) for n in names]
    arrs = {n: np.asarray(im) for n, im in imgs}

    # 1) face + band（mediapipe）
    face_masks, meta = {}, {}
    for n, arr in arrs.items():
        face, box, score, src = detect_face_mesh(arr)
        stem = Path(n).stem
        if face is None:
            print(f"  ⚠️ {n}: 未检出脸")
            meta[stem] = dict(source="none")
            continue
        k_in = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * args.band_inner + 1,) * 2)
        k_out = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * args.band_outer + 1,) * 2)
        er = cv2.erode(face.astype(np.uint8), k_in).astype(bool)
        di = cv2.dilate(face.astype(np.uint8), k_out).astype(bool)
        band = di & ~er
        face_masks[n] = dict(face=face, band=band, box=box)
        write_mask(out / f"{stem}_face.png", face)
        write_mask(out / f"{stem}_band.png", band)
        h, w = face.shape
        meta[stem] = dict(
            source=src, score=round(score, 4),
            box=[int(v) for v in box],
            face_px=int(face.sum()), face_pct=round(100 * face.sum() / (h * w), 2),
            band_px=int(band.sum()))
        print(f"  ✅ {n}: face {src} score={score:.2f} "
              f"{face.sum()}px ({100*face.sum()/(h*w):.2f}%) band {band.sum()}px")

    # 2) person（SAM3）
    if args.no_sam3:
        person_masks = [None] * len(imgs)
        print("  ⚠️ --no-sam3：无 person mask（仅应急）")
    else:
        person_masks = sam3_person_masks(imgs)
    for (n, im), pm in zip(imgs, person_masks):
        stem = Path(n).stem
        if pm is not None:
            write_mask(out / f"{stem}_person.png", pm)
            if stem in meta:
                meta[stem]["person_px"] = int(pm.sum())
        ov = overlay(arrs[n],
                     face_masks.get(n, {}).get("face"),
                     face_masks.get(n, {}).get("band"), pm)
        Image.fromarray(ov).save(out / f"{stem}_overlay.png")

    with open(out / "masks_meta.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"🎉 Done: {out}")


if __name__ == "__main__":
    main()
