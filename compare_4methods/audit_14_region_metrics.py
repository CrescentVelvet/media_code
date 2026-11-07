"""分项 PSNR：人脸区 vs 非人脸区（view 0/8/16），回答"归位到底让人脸清楚了没有"。"""
import json
import os

import numpy as np
from PIL import Image

REF = "/home/velvet/output/da3_gs_results/佳佳抱枕_p1m/official_dump.npz"
EX = "/home/velvet/output/da3_gs_results/佳佳抱枕_p1m_off/official_export"
gt_all = np.load(REF)["processed_images"]
H, W = gt_all.shape[1:3]
FACE = (0.35, 0.105, 0.565, 0.335)          # 与 A6/A7/A12 同源（view 0 的脸）
box = (int(FACE[0] * W), int(FACE[1] * H), int(FACE[2] * W), int(FACE[3] * H))

mask = np.zeros((H, W), bool)
mask[box[1]:box[3], box[0]:box[2]] = True


def psnr(a, b, m=None):
    if m is not None:
        a, b = a[m], b[m]
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 10 * np.log10(255.0 ** 2 / max(mse, 1e-12))


def sharp(rgb):
    import cv2
    g = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


TAGS = ("i12_fullpts_sh", "i02_pruned_sh", "i01_fullpts_dc", "hist_pruned_dc")
VIEWS = (0, 8, 16)
rep = {}
for tag in TAGS:
    face, other, full = [], [], []
    fsh, osh = [], []
    for v in VIEWS:
        im = np.asarray(Image.open(os.path.join(EX, f"ply_{tag}_view{v:03d}.png")))
        g = gt_all[v]
        face.append(psnr(im, g, mask))
        other.append(psnr(im, g, ~mask))
        full.append(psnr(im, g))
        fsh.append(sharp(im[box[1]:box[3], box[0]:box[2]]))
        om = np.array(im, copy=True)
        om[mask] = 0
        osh.append(sharp(om))
    rep[tag] = dict(face=float(np.mean(face)), nonface=float(np.mean(other)),
                    full=float(np.mean(full)),
                    face_sharp=float(np.mean(fsh)), nonface_sharp=float(np.mean(osh)))
    print(f"{tag:<16} 人脸 {rep[tag]['face']:6.2f}  非人脸 {rep[tag]['nonface']:6.2f}  "
          f"全图 {rep[tag]['full']:6.2f} ｜锐度 人脸 {rep[tag]['face_sharp']:6.0f} "
          f"非人脸 {rep[tag]['nonface_sharp']:6.0f}   (人脸面积 {mask.mean()*100:.1f}%)")

base = rep["hist_pruned_dc"]
print("\n相对归位前（hist_pruned_dc）的增量：")
for tag in TAGS[:-1]:
    r = rep[tag]
    print(f"  {tag:<16} 人脸 {r['face']-base['face']:+.2f}  非人脸 "
          f"{r['nonface']-base['nonface']:+.2f}  全图 {r['full']-base['full']:+.2f}"
          f"  ｜锐度 人脸 {r['face_sharp']-base['face_sharp']:+.0f}  "
          f"非人脸 {r['nonface_sharp']-base['nonface_sharp']:+.0f}")

with open(os.path.join(EX, "ply_region_metrics.json"), "w", encoding="utf-8", newline="\n") as f:
    json.dump(dict(views=list(VIEWS), face_box=list(box), face_area=float(mask.mean()),
                   per_case=rep), f, indent=1, ensure_ascii=False)
print("\nsaved ply_region_metrics.json")
