#!/usr/bin/env python3
"""Remy 采集包 image/*.heic → DA3 输入用 images/*.jpg（文件名保持时间戳序）。

Env vars:
  SRC_DIR   采集包目录（含 image/ 子目录）
  OUT_DIR   输出目录（生成 OUT_DIR/images/*.jpg）
"""
import os
import sys
from pathlib import Path

SRC = Path(os.environ["SRC_DIR"])
OUT = Path(os.environ["OUT_DIR"])
IMG_OUT = OUT / "images"
IMG_OUT.mkdir(parents=True, exist_ok=True)

from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

heics = sorted((SRC / "image").glob("*.heic"))
if not heics:
    sys.exit(f"❌ {SRC}/image 下没有 heic")

n_skip = 0
for i, p in enumerate(heics):
    dst = IMG_OUT / (p.stem + ".jpg")
    if dst.exists():
        n_skip += 1
        continue
    im = Image.open(p).convert("RGB")
    im.save(dst, "JPEG", quality=92)
    if (i + 1) % 50 == 0:
        print(f"   {i + 1}/{len(heics)}", flush=True)

print(f"✅ {len(heics)} 张（跳过已存在 {n_skip}）→ {IMG_OUT}")
