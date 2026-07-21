#!/usr/bin/env python3
"""Build a paired restoration-vs-beauty comparison dataset for HYPIR.

For each face image in INPUT_DIR (walked recursively), this runs RetouchFormer
once and saves THREE perfectly-aligned 512x512 PNGs under OUTPUT_DIR:
  - hq_orig/<rel>.png   — the aligned Resize(512)+CenterCrop(512) of the
                          ORIGINAL face (HQ target for the *restoration* run)
  - hq_beauty/<rel>.png — RetouchFormer's beautified output of that same crop
                          (HQ target for the *restoration+beauty* run)
  - lq_gauss/<rel>.png  — gaussian-blurred degradation of the same crop
                          (the LQ input for BOTH runs)

All three derive from the SAME source tensor (the model input src), so they are
pixel-aligned by construction — safe for any input size/aspect (the model's VRT
hardcodes input_resolution=(6,64,64) => 512x512, so non-square inputs are
center-cropped; hq_orig/lq_gauss save exactly that crop).

Why three folders (two datasets) instead of one:
  The current 03c/04c synthetic-degradation path trains HYPIR with
  LQ=gaussian-blur, HQ=original (USM->GT). It restores blur but tends to
  *hallucinate blemishes* ("长痘变丑") — the model over-enhances and invents
  skin flaws. By swapping the HQ target to RetouchFormer's beautified version
  (blemish-free, skin smoothed) while keeping the SAME blurred LQ, the model
  still learns deblur (enhancement preserved) but its target is clean smooth
  skin, so it stops inventing blemishes. Building BOTH lets you A/B compare:
    rest.parquet      : lq_gauss -> hq_orig   (baseline = current 03c-style)
    rest_beauty.parquet : lq_gauss -> hq_beauty (restoration + beauty, the fix)
  Train 04b on each (separate OUTPUT_DIR), then eval with 05/02 and compare.

The gaussian blur replicates the SIMPLIFIED batch_transform.py shipped in this
repo's HYPIR clone (random kernel 3/5/7/9/11, sigma 1-2, repeated 1-5 times),
one FIXED seeded realization per image (offline, not re-randomized per epoch).
NB: applied to the raw aligned crop (not USM(orig)) — a minor deviation from
03c's LQ=blur(USM(orig)); the A-vs-B comparison stays single-variable since both
share the identical lq_gauss. Pass BLUR_SEED to re-randomize reproducibly.

Model loading mirrors the official img_retouching.py exactly:
    net = importlib.import_module('model.RetouchFormer')
    model = net.InpaintGenerator().to(device)
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
    model.eval()
    pred, _ = model(src)   # src in [-1,1], pred in [-1,1]

Env (set by 03d_build_beauty_dataset.sh):
  RETOUCH_DIR, WEIGHT_PATH, MODEL_NAME, INPUT_DIR, OUTPUT_DIR,
  RESIZE_MODE, SIZE, DEVICE, SAVE_COMPARE, BLUR_SEED, SKIP_BLUR
"""
import os
import sys
import time
import random
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
BLUR_SEED = int(os.environ.get("BLUR_SEED", "231"))
SKIP_BLUR = os.environ.get("SKIP_BLUR", "0") == "1"     # 1 = don't build lq_gauss

# Multi-GPU sharding via torchrun: each process is independent (no comms).
# torchrun sets LOCAL_RANK/WORLD_SIZE; standalone (plain python) defaults to 0/1.
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif", ".ppm"}

# Match the official wildDataset transform: Resize(SIZE) -> ToTensor ->
# Normalize(mean=std=0.5) -> output in [-1, 1] (what save_image value_range=(-1,1)
# expects). RESIZE_MODE=square adds CenterCrop(SIZE) so non-square wild images
# don't crash the model (its VRT Stage hardcodes input_resolution (6,64,64) =>
# 512x512). On the official FFHQR test set (already 512x512) both modes are
# identical. Set RESIZE_MODE=smallest to reproduce wildDataset exactly (only
# safe for square inputs). Identical to retouchformer/run_inference.py so the
# beautified hq_beauty is byte-for-byte the official inference output.
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


def make_blur_fn():
    """Replicate the simplified batch_transform.py blur: random kernel
    3/5/7/9/11, sigma sampled in [1,2], repeated 1-5 times. One FIXED seeded
    realization per image (offline). Operates on a [1,3,H,W] tensor in [-1,1]
    (squeezes to [3,H,W] for torchvision.transforms.GaussianBlur, which is
    range-agnostic — a linear conv)."""
    rng = random.Random(BLUR_SEED)
    torch.manual_seed(BLUR_SEED)   # so GaussianBlur's sigma sampling is reproducible

    def blur(src):
        k = rng.randint(1, 5) * 2 + 1          # 3/5/7/9/11
        n = rng.randint(1, 5)                  # 1-5 repeats
        t = src.squeeze(0).cpu()               # [3,H,W] — cpu: GaussianBlur slower on tiny gpu?
        for _ in range(n):
            # new GaussianBlur each call -> re-samples sigma in [1,2] (matches the
            # modified batch_transform which constructs a fresh transform per iter)
            t = transforms.GaussianBlur(kernel_size=k, sigma=(1.0, 2.0))(t)
        return t.unsqueeze(0).to(src.device)   # [1,3,H,W]

    return blur


def main():
    for name, val in (("WEIGHT_PATH", WEIGHT_PATH), ("INPUT_DIR", INPUT_DIR), ("OUTPUT_DIR", OUTPUT_DIR)):
        if not val:
            sys.exit(f"ERROR: set {name}.")
    if SIZE != 512:
        print(f"WARNING: SIZE={SIZE} — the released RetouchFormer hardcodes size=512 "
              f"in Encoder/Decoder and VRT input_resolution=(6,64,64). Non-512 sizes "
              f"will most likely crash the model.", file=sys.stderr)

    device = torch.device(f"cuda:{LOCAL_RANK}" if (DEVICE == "cuda" and torch.cuda.is_available()) else "cpu")
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
    blur_fn = None if SKIP_BLUR else make_blur_fn()

    input_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    dirs = {
        "hq_orig":   out_dir / "hq_orig",     # aligned original crop (restoration HQ)
        "hq_beauty": out_dir / "hq_beauty",   # RetouchFormer beautified (beauty HQ)
    }
    if not SKIP_BLUR:
        dirs["lq_gauss"] = out_dir / "lq_gauss"  # gaussian-blurred (LQ for both)
    if SAVE_COMPARE:
        dirs["compare"] = out_dir / "compare"
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    images = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                images.append(Path(root) / f)
    images.sort(key=lambda x: str(x.relative_to(input_dir)))
    if not images:
        sys.exit(f"ERROR: no images in {input_dir}")
    if LOCAL_RANK == 0:
        print(f"[*] {len(images)} image(s): {input_dir} -> {out_dir}")
        print(f"[*]   hq_orig(原图对齐crop) -> {dirs['hq_orig']}")
        print(f"[*]   hq_beauty(美颜)        -> {dirs['hq_beauty']}")
        if not SKIP_BLUR:
            print(f"[*]   lq_gauss(高斯模糊LQ)   -> {dirs['lq_gauss']}  (blur_seed={BLUR_SEED})")
        print(f"[*] params: resize={RESIZE_MODE} size={SIZE} device={device} "
              f"save_compare={SAVE_COMPARE} skip_blur={SKIP_BLUR}")
    # shard: each rank takes a strided subset so no two ranks touch the same image
    my_images = images[LOCAL_RANK::WORLD_SIZE]
    if WORLD_SIZE > 1:
        print(f"[*]   [r{LOCAL_RANK}/{WORLD_SIZE}] {device}: handling {len(my_images)}/{len(images)} images")

    infer_times = []
    ok = 0
    t_loop0 = time.time()
    rank_pfx = f"[r{LOCAL_RANK}/{WORLD_SIZE}] " if WORLD_SIZE > 1 else ""
    with torch.no_grad():
        for i, fp in enumerate(my_images, 1):
            gidx = LOCAL_RANK + (i - 1) * WORLD_SIZE + 1   # global 1-indexed position
            rel = fp.relative_to(input_dir)
            stem = rel.with_suffix(".png")
            orig_path = dirs["hq_orig"] / stem
            beauty_path = dirs["hq_beauty"] / stem
            tag = "" if not SKIP_BLUR else " [no-blur]"
            try:
                orig_path.parent.mkdir(parents=True, exist_ok=True)
                beauty_path.parent.mkdir(parents=True, exist_ok=True)
                img = Image.open(fp).convert("RGB")
                w0, h0 = img.size
                src = tfm(img).unsqueeze(0).to(device)   # [1,3,512,512] in [-1,1] — model input

                # hq_orig = the exact aligned crop the model sees (un-beautified original)
                save_image(src, str(orig_path), normalize=True, value_range=(-1, 1))

                t1 = time.time()
                pred, _ = model(src)            # [1,3,512,512] in [-1,1] — beautified
                dt = time.time() - t1
                # hq_beauty = RetouchFormer output. Both [-1,1] -> [0,1] PNG via
                # save_image normalize=True, value_range=(-1,1) (matches run_inference.py).
                save_image(pred, str(beauty_path), normalize=True, value_range=(-1, 1))

                lq = None
                if not SKIP_BLUR:
                    lq = blur_fn(src)          # [1,3,512,512] in [-1,1] — one fixed realization
                    lq_path = dirs["lq_gauss"] / stem
                    lq_path.parent.mkdir(parents=True, exist_ok=True)
                    save_image(lq, str(lq_path), normalize=True, value_range=(-1, 1))

                if SAVE_COMPARE:
                    # [LQ(gauss) | HQ(orig) | HQ(beauty)] horizontal concat.
                    # Reuse the SAME `lq` tensor saved above (re-calling blur_fn
                    # would draw a different kernel/sigma — compare must match
                    # the saved lq_gauss).
                    panels = []
                    if not SKIP_BLUR:
                        panels.append(lq)
                    panels.append(src)
                    panels.append(pred)
                    cmp = torch.cat(panels, dim=3)
                    cmp_path = dirs["compare"] / stem
                    cmp_path.parent.mkdir(parents=True, exist_ok=True)
                    save_image(cmp, str(cmp_path), normalize=True, value_range=(-1, 1))

                _, _, H, W = pred.shape
                infer_times.append(dt)
                ok += 1
                if WORLD_SIZE == 1 or i <= 3 or i % 50 == 0:
                    print(f"{rank_pfx}♻️[{gidx}/{len(images)}] {fp.name}  ->  hq_orig + hq_beauty"
                          f"{' + lq_gauss' if not SKIP_BLUR else ''} | "
                          f"{w0}x{h0} -> {W}x{H} | 美颜 {dt:.2f}s{tag}")
            except Exception as e:
                # 损坏/截断图(OSError: image file is truncated)或推理失败都跳过、不中断；
                # 删掉本图已写的半成品(避免半对进 parquet 破坏同名配对)。
                print(f"{rank_pfx}[{gidx}/{len(images)}] {fp.name}  ! failed (skipped): {e}", file=sys.stderr)
                for d in dirs.values():
                    p = d / stem
                    try:
                        if p.exists():
                            p.unlink()
                    except Exception:
                        pass

    loop_time = time.time() - t_loop0
    pure = sum(infer_times)
    skipped = len(my_images) - ok
    skip_note = f", {skipped} skipped (损坏/截断图，见上方 ! failed 行)" if skipped else ""
    print(f"{rank_pfx}[*] done. {ok}/{len(my_images)} succeeded{skip_note}. "
          f"模型加载 {load_time:.2f}s + 循环 {loop_time:.2f}s (其中纯推理 {pure:.2f}s)")
    if infer_times:
        avg = pure / len(infer_times)
        print(f"{rank_pfx}[*] 单图美颜耗时: avg {avg:.2f}s, min {min(infer_times):.2f}s, "
              f"max {max(infer_times):.2f}s, 共 {len(infer_times)} 张")
    if LOCAL_RANK == 0:
        print(f"[*] hq_orig(原图对齐crop):    {dirs['hq_orig']}")
        print(f"[*] hq_beauty(美颜):         {dirs['hq_beauty']}")
        if not SKIP_BLUR:
            print(f"[*] lq_gauss(高斯模糊LQ):    {dirs['lq_gauss']}")
        if SAVE_COMPARE:
            print(f"[*] 对齐核对图(LQ|orig|beauty): {dirs['compare']}")
        if not SKIP_BLUR:
            print(f"[*] next: bash {os.path.dirname(os.path.abspath(__file__))}"
                  f"/03d_build_beauty_dataset.sh 会用 03b 产两张 parquet(rest/rest_beauty)")


if __name__ == "__main__":
    main()
