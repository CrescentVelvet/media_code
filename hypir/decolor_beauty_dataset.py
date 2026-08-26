#!/usr/bin/env python3
"""Build a de-colored beauty dataset INDEPENDENT of 03d (parallel, not sequential).

For each face image in INPUT_DIR (walked recursively), this runs RetouchFormer
once to get a beautified (reddish) image, then applies
``wavelet_reconstruction(content=beauty, style=src)`` = beauty's HIGH frequency
(skin-smoothing texture) blended with the ORIGINAL's LOW frequency (true skin
tone) → a de-colored beauty. Saves pixel-aligned 512×512 PNGs (all derived from
the SAME src tensor, so they're aligned by construction — safe for any input
size/aspect; the model's VRT hardcodes input_resolution=(6,64,64) => 512x512,
non-square inputs are CenterCrop'd):
  - ``hq_beauty_decolor/<name>.png`` = de-colored beauty (D-group HQ target)
  - ``lq_gauss/<name>.png``         = gaussian-blurred orig (LQ for all groups)
  - ``compare/<name>.png``          = [LQ|orig|beauty|decolor] (SAVE_COMPARE=1)

``hq_orig``/``hq_beauty`` are in-memory intermediates (NOT saved) — 03e only
produces the D group. 03b then builds ``rest_beauty_decolor.parquet``
(``lq_gauss`` -> ``hq_beauty_decolor``) for 04b training.

Why de-color: RetouchFormer's output skews reddish (肤色红润) — a low-frequency
color bias baked into the released weights (the VRT selective self-attention
pulls blemish regions toward the surrounding skin statistics, which skew warm).
Smoothing/blemish-removal, however, is a HIGH-frequency change. So:
    result = beauty_high + orig_low
keeps the smoothing (high-freq from beauty) while restoring the true skin color
(low-freq from orig) — "去红润、保留磨皮". Because the HQ *training target* is
de-colored, a LoRA trained on the parquet (via the UNCHANGED
``04b_train_paired.sh``) learns to restore toward de-colored beauty; inference
with that LoRA (via the UNCHANGED ``02_run_inference.sh``) thus produces
"去红润 + 磨皮" results — neither training nor inference scripts need any
modification; only the HQ target folder / parquet path differs.

This script is PARALLEL to ``03d_build_beauty_dataset`` (not sequential): it
runs RetouchFormer itself, so NO need to run 03d first. The RetouchFormer
loading / transform / blur logic is COPIED from ``build_beauty_dataset.py`` —
duplication is intentional (parallel scripts, each self-contained for pipeline
drop-in). The ``wavelet_*`` functions are copied verbatim from
``HYPIR/utils/common.py:32-80`` (torch only, no HYPIR import).

A/B/C vs D single-variable comparison: 03d and 03e each produce their own
``lq_gauss``, but under the SAME ``BLUR_SEED`` (default 231) + same image order
they are pixel-identical (``blur_fn`` is per-image FIXED seeded), so A/B/C (from
03d) and D (from 03e) still share the same LQ — the comparison stays
single-variable.

Model loading mirrors the official ``img_retouching.py`` exactly (same as
``build_beauty_dataset.py``):
    net = importlib.import_module('model.RetouchFormer')
    model = net.InpaintGenerator().to(device)
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
    model.eval()
    beauty, _ = model(src)                          # src in [-1,1], beauty in [-1,1]
    decolor = decolor_image(beauty, src, DECOLOR_MODE)  # wavelet | high_freq_dc

Env (set by 03e_decolor_beauty_dataset.sh):
  RETOUCH_DIR, WEIGHT_PATH, MODEL_NAME, INPUT_DIR, OUTPUT_DIR,
  RESIZE_MODE, SIZE, DEVICE, SAVE_COMPARE, BLUR_SEED, SKIP_BLUR, DECOLOR_MODE
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


# ── BEGIN verbatim copy from HYPIR/utils/common.py:32-80 ──────────────────────
# (self-contained — only torch.nn.functional; kept inline so this script has no
#  HYPIR import dependency and runs in any torch env. Do not "fix" the kernel
#  layout / dilation — it implements the multi-scale Gaussian wavelet used by
#  HYPIR/SUPIR/CCSR; matching it byte-for-byte guarantees the same fusion as
#  HYPIR's own inference-time wavelet_reconstruction.)
def wavelet_blur(image, radius):
    """
    Apply wavelet blur to the input tensor.
    """
    # input shape: (1, 3, H, W)
    # convolution kernel
    kernel_vals = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device)
    # add channel dimensions to the kernel to make it a 4D tensor
    kernel = kernel[None, None]
    # repeat the kernel across all input channels
    kernel = kernel.repeat(3, 1, 1, 1)
    image = torch.nn.functional.pad(image, (radius, radius, radius, radius), mode='replicate')
    # apply convolution
    output = torch.nn.functional.conv2d(image, kernel, groups=3, dilation=radius)
    return output


def wavelet_decomposition(image, levels=5):
    """
    Apply wavelet decomposition to the input tensor.
    This function only returns the low frequency & the high frequency.
    """
    high_freq = torch.zeros_like(image)
    for i in range(levels):
        radius = 2 ** i
        low_freq = wavelet_blur(image, radius)
        high_freq += (image - low_freq)
        image = low_freq

    return high_freq, low_freq


def wavelet_reconstruction(content_feat, style_feat):
    """
    Apply wavelet decomposition, so that the content will have the same color as the style.
    """
    # calculate the wavelet decomposition of the content feature
    content_high_freq, content_low_freq = wavelet_decomposition(content_feat)
    del content_low_freq
    # calculate the wavelet decomposition of the style feature
    style_high_freq, style_low_freq = wavelet_decomposition(style_feat)
    del style_high_freq
    # reconstruct the content feature with the style's high frequency
    return content_high_freq + style_low_freq
# ── END verbatim copy (HYPIR/utils/common.py) ─────────────────────────────────


RETOUCH_DIR = os.environ.get("RETOUCH_DIR", "../RetouchFormer")
# `model` is a top-level package relative to the repo root — put it on sys.path
# so `import model.RetouchFormer` resolves (official img_retouching.py relies on
# CWD == repo root; we make it path-independent instead). Copied from
# build_beauty_dataset.py (parallel script).
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
DECOLOR_MODE = os.environ.get("DECOLOR_MODE", "wavelet")  # wavelet | high_freq_dc (see decolor_image())

# Multi-GPU sharding via torchrun: each process is independent (no comms).
# torchrun sets LOCAL_RANK/WORLD_SIZE; standalone (plain python) defaults to 0/1.
# Same scheme as build_beauty_dataset.py (parallel script).
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif", ".ppm"}


# ── BEGIN copy from build_beauty_dataset.py (parallel script, intentional dup) ─
# Match the official wildDataset transform: Resize(SIZE) -> ToTensor ->
# Normalize(mean=std=0.5) -> output in [-1, 1] (what save_image value_range=(-1,1)
# expects). RESIZE_MODE=square adds CenterCrop(SIZE) so non-square wild images
# don't crash the model (its VRT Stage hardcodes input_resolution (6,64,64) =>
# 512x512). On the official FFHQR test set (already 512x512) both modes are
# identical. Identical to build_beauty_dataset.py so the de-colored output is
# byte-for-byte aligned with 03d's hq_beauty (same src tensor => same crop).
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
    realization per image (offline). Same BLUR_SEED + same image order as 03d
    => pixel-identical lq_gauss, so A/B/C (03d) and D (03e) share the same LQ
    (single-variable comparison). Operates on a [1,3,H,W] tensor in [-1,1]
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
# ── END copy from build_beauty_dataset.py ─────────────────────────────────────


def decolor_image(beauty, orig, mode):
    """De-color RetouchFormer's reddish bias from the beautified image.

    Both modes take the beautified image's HIGH-freq (smoothing/blemish-removal,
    what we want to KEEP) and the original's LOW-freq (true skin tone, what we
    want to RESTORE). They differ in how they handle the reddish tint that lives
    in beauty's HIGH-freq — RetouchFormer pushes skin toward a "healthy warm"
    statistics, which leaks into the high freq as a per-channel DC offset:

    - ``wavelet`` (default, original behaviour): plain
      ``wavelet_reconstruction(beauty, orig)`` = ``beauty_high + orig_low``.
      Only swaps the LOW-freq color. CANNOT remove redness that lives in
      ``beauty_high`` (the typical case — this is why the original 03e fails to
      de-color). Kept for backward-compat / A-B comparison.

    - ``high_freq_dc``: subtract ``beauty_high``'s per-channel spatial DC (the
      global reddish tint baked into the high freq) before adding ``orig_low``:
      ``(beauty_high - mean(beauty_high, per-channel)) + orig_low``.

      Why this preserves smoothing: smoothing lives in the high-freq STRUCTURE
      (the texture pattern — which frequencies, what amplitude — that does the
      skin-smoothing), while the reddish tint is the high-freq per-channel DC
      (a constant tint layer). Subtracting the DC is a constant per-channel
      shift; it does NOT touch the structure, so blemish-removal / skin
      smoothing is preserved — only the reddish tint is stripped.

      Limit: only fixes redness that is a HIGH-freq GLOBAL DC offset. If redness
      is LOCAL (skin region only, global DC ≈ 0), this won't fully remove it —
      escalate to a global tone-match (Reinhard per-channel mean/std → orig) or
      a skin-masked local de-color.

    ``wavelet_*`` used here are the verbatim HYPIR copy above.
    """
    if mode == "wavelet":
        return wavelet_reconstruction(beauty, orig)
    elif mode == "high_freq_dc":
        beauty_high, _ = wavelet_decomposition(beauty)
        # per-channel spatial DC of the high freq — the global reddish tint.
        # keepdim so the broadcast aligns with [B,C,H,W].
        beauty_high = beauty_high - beauty_high.mean(dim=[2, 3], keepdim=True)
        _, orig_low = wavelet_decomposition(orig)
        return beauty_high + orig_low
    else:
        sys.exit(f"ERROR: unknown DECOLOR_MODE='{mode}' (use wavelet|high_freq_dc).")


def main():
    for name, val in (("WEIGHT_PATH", WEIGHT_PATH), ("INPUT_DIR", INPUT_DIR), ("OUTPUT_DIR", OUTPUT_DIR)):
        if not val:
            sys.exit(f"ERROR: set {name}.")
    if SIZE != 512:
        print(f"WARNING: SIZE={SIZE} — the released RetouchFormer hardcodes size=512 "
              f"in Encoder/Decoder and VRT input_resolution=(6,64,64). Non-512 sizes "
              f"will most likely crash the model.", file=sys.stderr)

    # Pin this process's current CUDA device to its rank's GPU. Custom CUDA ops
    # (e.g. RetouchFormer's stylegan2 kernels) use torch.cuda.current_device();
    # without set_device it defaults to 0, so on non-zero ranks they allocate on
    # cuda:0 while the model lives on cuda:LOCAL_RANK -> "illegal memory access".
    if DEVICE == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(LOCAL_RANK)
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
        "hq_beauty_decolor": out_dir / "hq_beauty_decolor",   # D-group HQ (de-colored beauty)
    }
    if not SKIP_BLUR:
        dirs["lq_gauss"] = out_dir / "lq_gauss"               # gaussian-blurred (LQ for all)
    if SAVE_COMPARE:
        dirs["compare"] = out_dir / "compare"                  # [LQ|orig|beauty|decolor]
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
        print(f"[*]   hq_beauty_decolor(去红润美颜) -> {dirs['hq_beauty_decolor']}")
        if not SKIP_BLUR:
            print(f"[*]   lq_gauss(高斯模糊LQ)        -> {dirs['lq_gauss']}  (blur_seed={BLUR_SEED})")
        if SAVE_COMPARE:
            print(f"[*]   compare(LQ|orig|beauty|decolor) -> {dirs['compare']}")
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
            decolor_path = dirs["hq_beauty_decolor"] / stem
            tag = "" if not SKIP_BLUR else " [no-blur]"
            try:
                decolor_path.parent.mkdir(parents=True, exist_ok=True)
                img = Image.open(fp).convert("RGB")
                w0, h0 = img.size
                src = tfm(img).unsqueeze(0).to(device)   # [1,3,512,512] in [-1,1] — model input

                t1 = time.time()
                # RetouchFormer 1 pass — beautified (reddish); in-memory intermediate, NOT saved
                # (03e only produces the D group; hq_orig/hq_beauty stay in memory).
                beauty, _ = model(src)                    # [1,3,512,512] in [-1,1]
                # De-color: beauty's HIGH-freq (smoothing/blemish-removal) + src's LOW-freq
                # (true skin tone). content=beauty, style=src(=hq_orig). Linear ops =>
                # range-agnostic, but clamp below to guard tiny out-of-range drift.
                # DECOLOR_MODE: wavelet(只换低频,去不掉高频红) | high_freq_dc(高频减DC去红).
                decolor = decolor_image(beauty, src, DECOLOR_MODE)
                dt = time.time() - t1
                decolor = decolor.clamp(-1, 1)
                save_image(decolor, str(decolor_path), normalize=True, value_range=(-1, 1))

                lq = None
                if not SKIP_BLUR:
                    lq = blur_fn(src)          # [1,3,512,512] in [-1,1] — one fixed realization
                    lq_path = dirs["lq_gauss"] / stem
                    lq_path.parent.mkdir(parents=True, exist_ok=True)
                    save_image(lq, str(lq_path), normalize=True, value_range=(-1, 1))

                if SAVE_COMPARE:
                    # [LQ(gauss) | HQ(orig) | HQ(beauty,reddish) | HQ(decolor)]
                    # Reuse the SAME lq/src/beauty/decolor tensors saved above (re-calling
                    # blur_fn / the model would draw different results — compare must match
                    # the saved files).
                    panels = []
                    if not SKIP_BLUR:
                        panels.append(lq)
                    panels.append(src)                                  # hq_orig
                    panels.append(beauty)                              # hq_beauty (reddish)
                    panels.append(decolor)                             # hq_beauty_decolor
                    cmp = torch.cat(panels, dim=3)
                    cmp_path = dirs["compare"] / stem
                    cmp_path.parent.mkdir(parents=True, exist_ok=True)
                    save_image(cmp, str(cmp_path), normalize=True, value_range=(-1, 1))

                _, _, H, W = decolor.shape
                infer_times.append(dt)
                ok += 1
                if WORLD_SIZE == 1 or i <= 3 or i % 50 == 0:
                    print(f"{rank_pfx}♻️[{gidx}/{len(images)}] {fp.name}  ->  hq_beauty_decolor"
                          f"{' + lq_gauss' if not SKIP_BLUR else ''} | "
                          f"{w0}x{h0} -> {W}x{H} | 美颜+去红润 {dt:.2f}s{tag}")
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
          f"模型加载 {load_time:.2f}s + 循环 {loop_time:.2f}s (其中纯推理+融合 {pure:.2f}s)")
    if infer_times:
        avg = pure / len(infer_times)
        print(f"{rank_pfx}[*] 单图耗时: avg {avg:.2f}s, min {min(infer_times):.2f}s, "
              f"max {max(infer_times):.2f}s, 共 {len(infer_times)} 张")
    if LOCAL_RANK == 0:
        print(f"[*] hq_beauty_decolor(去红润美颜): {dirs['hq_beauty_decolor']}")
        if not SKIP_BLUR:
            print(f"[*] lq_gauss(高斯模糊LQ):      {dirs['lq_gauss']}")
        if SAVE_COMPARE:
            print(f"[*] compare(LQ|orig|beauty|decolor): {dirs['compare']}")
        if not SKIP_BLUR:
            print(f"[*] next: 03e 末尾会用 03b 产 parquet(rest_beauty_decolor)")


if __name__ == "__main__":
    main()
