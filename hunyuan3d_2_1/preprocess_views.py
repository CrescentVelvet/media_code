#!/usr/bin/env python3
"""Batch background removal: INPUT_ROOT/<object>/* -> PREP_DIR/<object>/rgba/<stem>.png.

Hunyuan3D's pipeline does NOT rembg path/string inputs (only the demo does),
so RGB photos with cluttered backgrounds must be pre-cut to RGBA PNGs, whose
alpha channel the image processor then uses as the object mask.

Uses the official repo's BackgroundRemover (u2net; auto-downloads ~170MB on
first run). Resumable: existing PNGs are skipped.

Env vars: HY3D_DIR, INPUT_ROOT, PREP_DIR
"""
import json
import os
import sys

from PIL import Image

HY3D_DIR = os.environ["HY3D_DIR"]
sys.path.insert(0, os.path.join(HY3D_DIR, "hy3dshape"))
from hy3dshape.rembg import BackgroundRemover

INPUT_ROOT = os.environ["INPUT_ROOT"]
PREP_DIR = os.environ["PREP_DIR"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main():
    objects = sorted(
        d for d in os.listdir(INPUT_ROOT) if os.path.isdir(os.path.join(INPUT_ROOT, d))
    )
    if not objects:
        sys.exit("❌ no object folders in " + INPUT_ROOT)

    print(f"✂️ [rembg] {len(objects)} object(s), u2net session loading ...")
    remover = BackgroundRemover()

    for obj in objects:
        src_dir = os.path.join(INPUT_ROOT, obj)
        files = sorted(
            f for f in os.listdir(src_dir) if os.path.splitext(f)[1].lower() in IMG_EXTS
        )
        out_dir = os.path.join(PREP_DIR, obj, "rgba")
        os.makedirs(out_dir, exist_ok=True)
        stems = []
        for f in files:
            stem = os.path.splitext(f)[0]
            out_path = os.path.join(out_dir, stem + ".png")
            stems.append(stem)
            if os.path.exists(out_path):
                continue  # resumable
            img = Image.open(os.path.join(src_dir, f)).convert("RGB")
            remover(img).save(out_path)
        manifest = {"object": obj, "source_dir": src_dir, "images": stems}
        with open(
            os.path.join(PREP_DIR, obj, "prep_manifest.json"), "w",
            encoding="utf-8", newline="\n",
        ) as fp:
            json.dump(manifest, fp, ensure_ascii=False, indent=2)
        print(f"  ✅ {obj}: {len(stems)} view(s)")
    print("🎉 [rembg] done ->", PREP_DIR)


if __name__ == "__main__":
    main()
