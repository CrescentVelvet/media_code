#!/usr/bin/env python3
"""将 HEIC/HEIF 图像批量转换为 JPG。

Env vars:
  INPUT_DIR   输入目录（含 .heic 文件）
  OUTPUT_DIR  输出目录（默认 INPUT_DIR 同级 image_jpg/）
"""
import os, sys, time, glob
from pathlib import Path

INPUT_DIR = os.environ.get("INPUT_DIR", "")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "")

if not INPUT_DIR:
    sys.exit("❌ INPUT_DIR not set")

input_path = Path(INPUT_DIR)
if not input_path.is_dir():
    sys.exit(f"❌ INPUT_DIR not found: {INPUT_DIR}")

if not OUTPUT_DIR:
    OUTPUT_DIR = str(input_path.parent / "image_jpg")

out_path = Path(OUTPUT_DIR)
out_path.mkdir(parents=True, exist_ok=True)

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    print("📦 installing pillow-heif...")
    import subprocess
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "-i", "https://pypi.tuna.tsinghua.edu.cn/simple",
        "--find-links", "/mnt/d/wheel",
        "pillow-heif",
    ])
    from pillow_heif import register_heif_opener
    register_heif_opener()

from PIL import Image

heic_files = sorted(
    list(input_path.glob("*.heic")) +
    list(input_path.glob("*.HEIC")) +
    list(input_path.glob("*.heif")) +
    list(input_path.glob("*.HEIF"))
)

if not heic_files:
    sys.exit(f"❌ no .heic/.heif files found in {INPUT_DIR}")

print(f"🚀 converting {len(heic_files)} HEIC -> JPG")
print(f"  📁 输入: {INPUT_DIR}")
print(f"  💾 输出: {OUTPUT_DIR}")
print(f"  🖼️  found: {len(heic_files)} files")

t0 = time.time()
ok, skip, fail = 0, 0, 0

for i, f in enumerate(heic_files):
    out_file = out_path / (f.stem + ".jpg")
    if out_file.exists():
        skip += 1
        continue
    try:
        img = Image.open(f)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(out_file, "JPEG", quality=95)
        ok += 1
    except Exception as e:
        print(f"  ⚠️ failed: {f.name}: {e}")
        fail += 1

    if (i + 1) % 20 == 0 or i + 1 == len(heic_files):
        print(f"  [{i+1}/{len(heic_files)}] ok={ok} skip={skip} fail={fail}")

dt = time.time() - t0
print(f"\n✅ converted: {ok}  ⏭️  skipped: {skip}  ❌ failed: {fail}")
print(f"⏱️ {dt:.1f}s  ({dt/max(ok,1):.2f}s/image)")
print(f"🎉 Done. Output: {OUTPUT_DIR}")
