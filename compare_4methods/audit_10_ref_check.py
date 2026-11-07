#!/usr/bin/env python3
"""评测参考图口径核对：模型实际看到的输入(processed_images) vs 我们用 cv2 缩放的 GT。

若两者 PSNR 不高，说明"糊"的判断里混进了 resize 口径差，必须先统一再比。
"""
import glob
import os

import cv2
import numpy as np

NPZ = os.environ["NPZ"]
IMG = os.path.expanduser(os.environ["IMAGES_DIR"])
d = np.load(NPZ)
GT = d["processed_images"]                       # (V,H,W,3) uint8，模型看到的那张
V, H, W = GT.shape[:3]
files = sorted(sum([glob.glob(os.path.join(IMG, e)) for e in ("*.png", "*.jpg", "*.jpeg")], []))


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 10 * np.log10(255.0 ** 2 / max(mse, 1e-12))


def rs(img, w, h, interp):
    return cv2.resize(img, (w, h), interpolation=interp)


print(f"{'view':>5} {'INTER_AREA':>11} {'INTER_LINEAR':>13} {'LANCZOS4':>10} {'原图尺寸':>16}")
for v in (0, 8, 16):
    ref = cv2.cvtColor(cv2.imread(files[v]), cv2.COLOR_BGR2RGB)
    src = f"{ref.shape[1]}x{ref.shape[0]}"
    print(f"{v:>5} {psnr(rs(ref,W,H,cv2.INTER_AREA), GT[v]):>11.2f} "
          f"{psnr(rs(ref,W,H,cv2.INTER_LINEAR), GT[v]):>13.2f} "
          f"{psnr(rs(ref,W,H,cv2.INTER_LANCZOS4), GT[v]):>10.2f} {src:>16}")
print("\n（数越高说明该缩放方式越接近模型内部用的那套；>35 dB 可视为同一张图）")
