#!/usr/bin/env python3
"""Batch RetouchFormer inference over a folder of face images, with timing.

A drop-in replacement for the official `img_retouching.py` that:
  - loads the model ONCE (timed), then loops over every image in INPUT_DIR
    (walked recursively) — instead of re-reading args each run;
  - prints per-image inference time + a summary (avg/min/max/total);
  - preserves the input's relative directory structure in the output
    (OUTPUT_DIR/result/<rel>.png) instead of a flat `<name>_out.png` dump;
  - reads params from env (set by 02_run_inference.sh) — no argparse needed;
  - lets the shell choose the GPU (no hardcoded CUDA_VISIBLE_DEVICES="0").

Model interface mirrors the official path exactly:
    net = importlib.import_module('model.RetouchFormer')
    model = net.InpaintGenerator().to(device)
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
    model.eval()
    pred_img, _ = model(source_tensor.to(device))
    save_image(pred_img, path, normalize=True, value_range=(-1, 1))

Env vars (set by 02_run_inference.sh):
  RETOUCH_DIR, WEIGHT_PATH, MODEL_NAME, INPUT_DIR, OUTPUT_DIR,
  RESIZE_MODE, SIZE, DEVICE
"""
import os
import sys
import time
import importlib
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

RETOUCH_DIR = os.environ.get("RETOUCH_DIR", "../RetouchFormer")
# `model` and `core` are top-level packages relative to the repo root — put it
# on sys.path so `import model.RetouchFormer` resolves (official img_retouching.py
# relies on CWD == repo root; we make it path-independent instead).
sys.path.insert(0, RETOUCH_DIR)

WEIGHT_PATH = os.environ.get("WEIGHT_PATH")
MODEL_NAME = os.environ.get("MODEL_NAME", "RetouchFormer")
INPUT_DIR = os.environ.get("INPUT_DIR")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR")
RESIZE_MODE = os.environ.get("RESIZE_MODE", "square")   # square | smallest
SIZE = int(os.environ.get("SIZE", "512"))               # model is fixed to 512
DEVICE = os.environ.get("DEVICE", "cuda")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif", ".ppm"}

# Match the official wildDataset transform: Resize(SIZE) -> ToTensor ->
# Normalize(mean=std=0.5) -> output in [-1, 1] (what save_image value_range=(-1,1)
# expects). RESIZE_MODE=square adds CenterCrop(SIZE) so non-square wild images
# don't crash the model (its VRT Stage hardcodes input_resolution (6,64,64) =>
# 512x512). On the official FFHQR test set (already 512x512) both modes are
# identical. Set RESIZE_MODE=smallest to reproduce wildDataset exactly.
def build_transform():
    steps = [transforms.Resize(SIZE)]
    if RESIZE_MODE == "square":
        steps.append(transforms.CenterCrop(SIZE))
    elif RESIZE_MODE == "smallest":
        pass  # official wildDataset behaviour (smallest edge -> SIZE)
    else:
        sys.exit(f"ERROR: unknown RESIZE_MODE='{RESIZE_MODE}' (use square|smallest).")
    steps.append(transforms.ToTensor())
    steps.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return transforms.Compose(steps)


def main():
    if not WEIGHT_PATH:
        sys.exit("ERROR: WEIGHT_PATH not set.")
    if not INPUT_DIR:
        sys.exit("ERROR: INPUT_DIR not set.")
    if not OUTPUT_DIR:
        sys.exit("ERROR: OUTPUT_DIR not set.")
    if SIZE != 512:
        print(f"WARNING: SIZE={SIZE} — the released RetouchFormer hardcodes size=512 "
              f"in Encoder/Decoder and VRT input_resolution=(6,64,64). Non-512 sizes "
              f"will most likely crash the model.", file=sys.stderr)

    device = torch.device(DEVICE if (DEVICE == "cuda" and torch.cuda.is_available()) else "cpu")
    if DEVICE == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available — falling back to CPU (very slow).", file=sys.stderr)

    # --- load model ONCE (timed) ---
    print(f"[*] building model '{MODEL_NAME}' (device={device}) ...")
    t_build0 = time.time()
    net = importlib.import_module("model." + MODEL_NAME)
    model = net.InpaintGenerator().to(device)
    print(f"[*] loading checkpoint: {WEIGHT_PATH}")
    # weights_only=False matches the official img_retouching.py (no weights_only arg;
    # the old default). Kept explicit so torch>=2.6 (default True) doesn't reject a
    # state_dict that may carry non-tensor pickled objects.
    state = torch.load(WEIGHT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    load_time = time.time() - t_build0
    print(f"[*] 模型加载耗时: {load_time:.2f}s")

    tfm = build_transform()
    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    result_dir = output_dir / "result"
    result_dir.mkdir(parents=True, exist_ok=True)

    images = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                images.append(Path(root) / f)
    images.sort(key=lambda x: str(x.relative_to(input_dir)))
    if not images:
        sys.exit(f"ERROR: no images in {input_dir}")
    print(f"[*] {len(images)} image(s): {input_dir} -> {result_dir}  "
          f"(resize={RESIZE_MODE} size={SIZE} device={device})")

    infer_times = []
    ok = 0
    t_loop0 = time.time()
    with torch.no_grad():
        for i, fp in enumerate(images, 1):
            rel = fp.relative_to(input_dir)
            result_path = result_dir / rel.with_suffix(".png")
            result_path.parent.mkdir(parents=True, exist_ok=True)

            img = Image.open(fp).convert("RGB")
            w0, h0 = img.size
            src = tfm(img).unsqueeze(0).to(device)

            t1 = time.time()
            try:
                pred, _ = model(src)            # [1,3,SIZE,SIZE] in [-1,1]
                dt = time.time() - t1
                save_image(pred, str(result_path), normalize=True, value_range=(-1, 1))
                _, _, H, W = pred.shape
                infer_times.append(dt)
                ok += 1
                print(f"[{i}/{len(images)}] {fp.name}  ->  {rel.with_suffix('.png').as_posix()}  "
                      f"| 分辨率 {w0}x{h0} -> {W}x{H} | 推理 {dt:.2f}s")
            except Exception as e:
                print(f"[{i}/{len(images)}] {fp.name}  ! failed: {e}", file=sys.stderr)

    loop_time = time.time() - t_loop0
    pure = sum(infer_times)
    print(f"[*] done. {ok}/{len(images)} succeeded. "
          f"模型加载 {load_time:.2f}s + 循环 {loop_time:.2f}s (其中纯推理 {pure:.2f}s)")
    if infer_times:
        avg = pure / len(infer_times)
        print(f"[*] 单图推理耗时: avg {avg:.2f}s, min {min(infer_times):.2f}s, "
              f"max {max(infer_times):.2f}s, 共 {len(infer_times)} 张")


if __name__ == "__main__":
    main()
