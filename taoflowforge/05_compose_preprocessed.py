#!/usr/bin/env python3
"""05_compose_preprocessed.py — SAM3 mask → TaoFlowForge 输入图（白底 + 主体居中）。

对每张原图：
  1. 读 worker 产出的最大置信实例（{stem}.p00.alpha.png / .mask.png）
  2. 羽化 alpha 合成到纯白底（掩码外像素 → 白）
  3. 按 mask 包围盒裁正方形（边长 = max(w,h)×(1+2·MARGIN)，防贴边截断）
  4. 缩放到 SIZE×SIZE（默认 1024²，与 TaoFlowForge 内部预处理一致）

找不到 mask 的图：原图直接缩放输出并在清单里标 fallback_original
（批量推理照常跑，但预期质量差——和 E-03 耳机同类问题）。

用法（由 05_preprocess_sam3.sh 调用，三个位置参数）：
  python 05_compose_preprocessed.py <MASKS_DIR> <IMG_DIR> <OUT_DIR>

Env: MARGIN=0.15  SIZE=1024
输出: OUT_DIR/<帧号>.png + OUT_DIR/_compose_report.json
"""
import json
import os
import sys

import numpy as np
from PIL import Image

MASKS_DIR, IMG_DIR, OUT_DIR = sys.argv[1], sys.argv[2], sys.argv[3]
MARGIN = float(os.environ.get("MARGIN", "0.15"))
SIZE = int(os.environ.get("SIZE", "1024"))
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

os.makedirs(OUT_DIR, exist_ok=True)

manifest_path = os.path.join(MASKS_DIR, "masks_manifest.json")
frames = {}
if os.path.exists(manifest_path):
    with open(manifest_path, encoding="utf-8") as f:
        frames = json.load(f).get("frames", {})

report = {}
for fname in sorted(os.listdir(IMG_DIR)):
    stem, ext = os.path.splitext(fname)
    if ext.lower() not in IMG_EXTS:
        continue
    img = Image.open(os.path.join(IMG_DIR, fname)).convert("RGB")
    W, H = img.size

    alpha_p = os.path.join(MASKS_DIR, f"{stem}.p00.alpha.png")
    mask_p = os.path.join(MASKS_DIR, f"{stem}.p00.mask.png")
    insts = frames.get(stem, [])

    if not insts or not os.path.exists(alpha_p):
        img.resize((SIZE, SIZE), Image.LANCZOS).save(
            os.path.join(OUT_DIR, f"{stem}.png"))
        report[stem] = {"status": "fallback_original",
                        "reason": "no SAM3 instance >= MIN_SCORE"}
        print(f"  ⚪ {fname}: 无 mask → 原图直出（预期质量差）")
        continue

    alpha = Image.open(alpha_p).convert("L")
    if alpha.size != (W, H):
        alpha = alpha.resize((W, H), Image.BILINEAR)
    alpha = np.asarray(alpha).astype(np.float32) / 255.0

    m_img = Image.open(mask_p).convert("L")
    if m_img.size != (W, H):
        m_img = m_img.resize((W, H), Image.BILINEAR)
    m = np.asarray(m_img) > 127

    # 羽化合成 → 白底
    rgb = np.asarray(img).astype(np.float32)
    comp = (rgb * alpha[..., None] + 255.0 * (1.0 - alpha[..., None])).astype(np.uint8)

    # 包围盒 → 正方形裁剪（居中于 bbox 中心）
    ys, xs = np.where(m)
    x1, x2, y1, y2 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    bw, bh = x2 - x1 + 1, y2 - y1 + 1
    side = int(max(bw, bh) * (1.0 + 2.0 * MARGIN))
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    sx1, sy1 = max(0, cx - side // 2), max(0, cy - side // 2)
    sx2, sy2 = min(W, sx1 + side), min(H, sy1 + side)
    crop = comp[sy1:sy2, sx1:sx2]

    canvas = np.full((side, side, 3), 255, dtype=np.uint8)
    canvas[:crop.shape[0], :crop.shape[1]] = crop
    out = Image.fromarray(canvas).resize((SIZE, SIZE), Image.LANCZOS)
    out.save(os.path.join(OUT_DIR, f"{stem}.png"))

    cov = insts[0].get("coverage")
    report[stem] = {"status": "ok", "coverage": cov, "score": insts[0].get("score")}
    print(f"  ✂️ {fname}: cov={cov}% score={insts[0].get('score')} → {stem}.png")

with open(os.path.join(OUT_DIR, "_compose_report.json"), "w",
          encoding="utf-8", newline="\n") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(f"🎉 compose done: {OUT_DIR} ({len(report)} 张)")
