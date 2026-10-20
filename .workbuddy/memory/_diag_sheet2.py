#!/usr/bin/env python3
"""生成帧序列缩略图：只看人体上半身区域，观察手部姿态随时间变化。"""
import os
import glob
import numpy as np
import cv2

SCENES = {
    "xiezi": "/home/velvet/output/som_human_results/hand06_nvsheng_xiezi/colmap_refined/images",
    "tanqin": "/home/velvet/output/som_human_results/hand08_nvsheng_tanqin/colmap_refined/images",
}
OUT = "/mnt/d/output/som_human_hand/diag"
os.makedirs(OUT, exist_ok=True)


def sheet(root, tag, n=24, cols=8):
    imgs = sorted(glob.glob(os.path.join(root, "*")))
    idx = np.linspace(0, len(imgs) - 1, min(n, len(imgs))).astype(int)
    tiles = []
    for i in idx:
        im = cv2.imread(imgs[i], cv2.IMREAD_COLOR)
        if im is None:
            continue
        h, w = im.shape[:2]
        # 裁掉上下：保留中部 30%~95%（人体与手所在带）
        y0, y1 = int(h * 0.28), int(h * 0.95)
        crop = im[y0:y1]
        tw = 260
        crop = cv2.resize(crop, (tw, int(crop.shape[0] * tw / crop.shape[1])))
        cv2.putText(crop, str(i), (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2)
        tiles.append(crop)
    if not tiles:
        return
    hh = max(t.shape[0] for t in tiles)
    fixed = []
    for t in tiles:
        if t.shape[0] < hh:
            pad = np.zeros((hh - t.shape[0], t.shape[1], 3), np.uint8)
            t = np.vstack([t, pad])
        fixed.append(t)
    rows = []
    for r in range(0, len(fixed), cols):
        chunk = fixed[r:r + cols]
        while len(chunk) < cols:
            chunk.append(np.zeros_like(fixed[0]))
        rows.append(np.hstack(chunk))
    out = np.vstack(rows)
    path = os.path.join(OUT, "sheet_{}.jpg".format(tag))
    cv2.imwrite(path, out, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(path, out.shape)


for tag, root in SCENES.items():
    sheet(root, tag)
