#!/usr/bin/env python3
"""Build a PAIRED beautification dataset for HYPIR using RetouchFormer.

For each face image in INPUT_DIR (walked recursively), this runs RetouchFormer
once and saves TWO perfectly-aligned 512x512 PNGs:
  - $OUTPUT_DIR/hq/<rel>.png  — RetouchFormer's beautified output (the TARGET)
  - $OUTPUT_DIR/lq/<rel>.png  — the same Resize(512)+CenterCrop(512) the model
                               saw, saved as the INPUT (original, un-beautified)

Because HQ and LQ both come from the SAME source tensor (the model input), they
are pixel-aligned by construction — safe for any input size/aspect (the model's
VRT hardcodes input_resolution=(6,64,64) => 512x512, so non-square inputs are
center-cropped). This is the key difference vs. naively copying the raw original
as LQ, which only lines up when the input is already 512x512 square.

The resulting hq/ + lq/ folders (identical filenames) feed straight into
03b_build_paired_dataset.sh -> 04b_train_paired.sh: HYPIR learns
  LQ (original face) -> HQ (beautified face),
i.e. it distills RetouchFormer's retouching (blemish removal + skin smoothing)
into HYPIR's one-step diffusion — face enhancement + a touch of beauty /
skin-smoothing at HYPIR's inference speed and tiled resolution.

Model loading mirrors the official img_retouching.py exactly:
    net = importlib.import_module('model.RetouchFormer')
    model = net.InpaintGenerator().to(device)
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
    model.eval()
    pred, _ = model(src)   # src in [-1,1], pred in [-1,1]

Env (set by 03d_build_beauty_dataset.sh):
  RETOUCH_DIR, WEIGHT_PATH, MODEL_NAME, INPUT_DIR, OUTPUT_DIR,
  RESIZE_MODE, SIZE, DEVICE, SAVE_COMPARE
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
SAVE_COMPARE = os.environ.get("SAVE_COMPARE", "0") == "1"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif", ".ppm"}

# Match the official wildDataset transform: Resize(SIZE) -> ToTensor ->
# Normalize(mean=std=0.5) -> output in [-1, 1] (what save_image value_range=(-1,1)
# expects). RESIZE_MODE=square adds CenterCrop(SIZE) so non-square wild images
# don't crash the model (its VRT Stage hardcodes input_resolution (6,64,64) =>
# 512x512). On the official FFHQR test set (already 512x512) both modes are
# identical. Set RESIZE_MODE=smallest to reproduce wildDataset exactly (only
# safe for square inputs). Identical to retouchformer/run_inference.py so the
# beautified HQ is byte-for-byte the official inference output.
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
    for name, val in (("WEIGHT_PATH", WEIGHT_PATH), ("INPUT_DIR", INPUT_DIR), ("OUTPUT_DIR", OUTPUT_DIR)):
        if not val:
            sys.exit(f"ERROR: set {name}.")
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
    out_dir = Path(OUTPUT_DIR)
    hq_dir = out_dir / "hq"
    lq_dir = out_dir / "lq"
    cmp_dir = out_dir / "compare"
    hq_dir.mkdir(parents=True, exist_ok=True)
    lq_dir.mkdir(parents=True, exist_ok=True)
    if SAVE_COMPARE:
        cmp_dir.mkdir(parents=True, exist_ok=True)

    images = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                images.append(Path(root) / f)
    images.sort(key=lambda x: str(x.relative_to(input_dir)))
    if not images:
        sys.exit(f"ERROR: no images in {input_dir}")
    print(f"[*] {len(images)} image(s): {input_dir} -> hq/+lq/ under {out_dir}  "
          f"(resize={RESIZE_MODE} size={SIZE} device={device} save_compare={SAVE_COMPARE})")
    print(f"[*] HQ(美颜/目标) -> {hq_dir}")
    print(f"[*] LQ(原图预处理/输入) -> {lq_dir}")

    infer_times = []
    ok = 0
    t_loop0 = time.time()
    with torch.no_grad():
        for i, fp in enumerate(images, 1):
            rel = fp.relative_to(input_dir)
            stem = rel.with_suffix(".png")
            hq_path = hq_dir / stem
            lq_path = lq_dir / stem
            hq_path.parent.mkdir(parents=True, exist_ok=True)
            lq_path.parent.mkdir(parents=True, exist_ok=True)

            img = Image.open(fp).convert("RGB")
            w0, h0 = img.size
            src = tfm(img).unsqueeze(0).to(device)   # [1,3,512,512] in [-1,1] — model input

            t1 = time.time()
            try:
                pred, _ = model(src)            # [1,3,512,512] in [-1,1] — beautified
                dt = time.time() - t1
                # HQ = beautified output; LQ = the exact tensor the model saw
                # (Resize+CenterCrop+normalize of the original). Both [-1,1] ->
                # [0,1] PNG via save_image normalize=True, value_range=(-1,1).
                save_image(pred, str(hq_path), normalize=True, value_range=(-1, 1))
                save_image(src, str(lq_path), normalize=True, value_range=(-1, 1))
                if SAVE_COMPARE:
                    # horizontal concat: [LQ | HQ] along width -> [1,3,512,1024]
                    cmp = torch.cat([src, pred], dim=3)
                    cmp_path = cmp_dir / stem
                    cmp_path.parent.mkdir(parents=True, exist_ok=True)
                    save_image(cmp, str(cmp_path), normalize=True, value_range=(-1, 1))
                _, _, H, W = pred.shape
                infer_times.append(dt)
                ok += 1
                print(f"[{i}/{len(images)}] {fp.name}  ->  hq/{stem.as_posix()} + lq/{stem.as_posix()}  "
                      f"| 分辨率 {w0}x{h0} -> {W}x{H} | 美颜 {dt:.2f}s")
            except Exception as e:
                print(f"[{i}/{len(images)}] {fp.name}  ! failed: {e}", file=sys.stderr)

    loop_time = time.time() - t_loop0
    pure = sum(infer_times)
    print(f"[*] done. {ok}/{len(images)} succeeded. "
          f"模型加载 {load_time:.2f}s + 循环 {loop_time:.2f}s (其中纯推理 {pure:.2f}s)")
    if infer_times:
        avg = pure / len(infer_times)
        print(f"[*] 单图美颜耗时: avg {avg:.2f}s, min {min(infer_times):.2f}s, "
              f"max {max(infer_times):.2f}s, 共 {len(infer_times)} 张")
    print(f"[*] HQ(美颜/目标): {hq_dir}")
    print(f"[*] LQ(原图预处理/输入): {lq_dir}")
    if SAVE_COMPARE:
        print(f"[*] 对齐核对图(LQ|HQ): {cmp_dir}")
    print(f"[*] next: HQ_DIR={hq_dir} LQ_DIR={lq_dir} "
          f"bash {os.path.dirname(os.path.abspath(__file__))}/03b_build_paired_dataset.sh")


if __name__ == "__main__":
    main()
