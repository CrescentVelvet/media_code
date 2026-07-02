#!/usr/bin/env python3
"""Batch HYPIR inference over a folder of LQ images, with timing.

Loads the SD2 + LoRA pipeline ONCE (timed), then loops over every image in
LQ_DIR (walked recursively), restoring each and saving to
OUTPUT_DIR/result/<rel>.png (+ OUTPUT_DIR/prompt/<rel>.txt). Prints per-image
inference time and a summary (avg/min/max/total).

Mirrors the official test.py's logic (recursive walk, relative-path output,
prompt handling) but adds timing and reads params from env (set by
02_run_inference.sh) — so it neither depends on test.py's CLI args nor needs
a (nonexistent) --config flag.

Env vars (set by 02_run_inference.sh):
  HYPIR_DIR, BASE_MODEL_PATH, WEIGHT_PATH, LORA_RANK, LORA_MODULES,
  MODEL_T, COEFF_T, LQ_DIR, TXT_DIR, CAPTIONER, FIXED_CAPTION,
  OUTPUT_DIR, SCALE_BY, UPSCALE, TARGET_LONGEST_SIDE,
  PATCH_SIZE, STRIDE, SEED, DEVICE
"""
import os
import sys
import time
from pathlib import Path

HYPIR_DIR = os.environ.get("HYPIR_DIR", "../HYPIR")
sys.path.insert(0, HYPIR_DIR)  # so `from HYPIR import ...` resolves

from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms

from HYPIR.enhancer.sd2 import SD2Enhancer
from HYPIR.utils.captioner import EmptyCaptioner, FixedCaptioner

BASE_MODEL_PATH = os.environ.get("BASE_MODEL_PATH")
WEIGHT_PATH = os.environ.get("WEIGHT_PATH")
LORA_RANK = int(os.environ.get("LORA_RANK", "256"))
LORA_MODULES = os.environ.get(
    "LORA_MODULES",
    "to_k,to_q,to_v,to_out.0,conv,conv1,conv2,conv_shortcut,conv_out,proj_in,proj_out,ff.net.2,ff.net.0.proj",
).split(",")
MODEL_T = int(os.environ.get("MODEL_T", "200"))
COEFF_T = int(os.environ.get("COEFF_T", "200"))
LQ_DIR = os.environ.get("LQ_DIR")
TXT_DIR = os.environ.get("TXT_DIR") or None
CAPTIONER = os.environ.get("CAPTIONER") or None     # None | "empty" | "fixed"
FIXED_CAPTION = os.environ.get("FIXED_CAPTION")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR")
SCALE_BY = os.environ.get("SCALE_BY", "factor")
UPSCALE = int(os.environ.get("UPSCALE", "4"))
_TLS = os.environ.get("TARGET_LONGEST_SIDE")
TARGET_LONGEST_SIDE = int(_TLS) if _TLS else None
PATCH_SIZE = int(os.environ.get("PATCH_SIZE", "512"))
STRIDE = int(os.environ.get("STRIDE", "256"))
SEED = int(os.environ.get("SEED", "231"))
DEVICE = os.environ.get("DEVICE", "cuda")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def main():
    if not BASE_MODEL_PATH:
        sys.exit("ERROR: BASE_MODEL_PATH not set.")
    if not WEIGHT_PATH:
        sys.exit("ERROR: WEIGHT_PATH not set.")
    if not LQ_DIR:
        sys.exit("ERROR: LQ_DIR not set.")
    if not OUTPUT_DIR:
        sys.exit("ERROR: OUTPUT_DIR not set.")

    set_seed(SEED)

    model = SD2Enhancer(
        base_model_path=BASE_MODEL_PATH,
        weight_path=WEIGHT_PATH,
        lora_modules=LORA_MODULES,
        lora_rank=LORA_RANK,
        model_t=MODEL_T,
        coeff_t=COEFF_T,
        device=DEVICE,
    )
    print(f"[*] loading models (device={DEVICE}) ...")
    t_load0 = time.time()
    model.init_models()
    load_time = time.time() - t_load0
    print(f"[*] 模型加载耗时: {load_time:.2f}s")

    input_dir = Path(LQ_DIR)
    output_dir = Path(OUTPUT_DIR)
    result_dir = output_dir / "result"
    prompt_dir = output_dir / "prompt"

    images = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                images.append(Path(root) / f)
    images.sort(key=lambda x: str(x.relative_to(input_dir)))
    if not images:
        sys.exit(f"ERROR: no images in {input_dir}")
    print(f"[*] {len(images)} image(s): {input_dir} -> {result_dir}  "
          f"(scale_by={SCALE_BY} upscale={UPSCALE} patch={PATCH_SIZE} stride={STRIDE} seed={SEED})")

    # Captioner is only used when TXT_DIR is None (mirrors test.py).
    captioner = None
    if TXT_DIR is None:
        if CAPTIONER == "fixed":
            if not FIXED_CAPTION:
                sys.exit("ERROR: CAPTIONER=fixed requires FIXED_CAPTION.")
            captioner = FixedCaptioner(DEVICE, FIXED_CAPTION)
        else:
            captioner = EmptyCaptioner(DEVICE)

    to_tensor = transforms.ToTensor()
    infer_times = []
    ok = 0
    t_loop0 = time.time()
    for i, fp in enumerate(images, 1):
        rel = fp.relative_to(input_dir)
        result_path = result_dir / rel.with_suffix(".png")
        prompt_path = prompt_dir / rel.with_suffix(".txt")
        result_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.parent.mkdir(parents=True, exist_ok=True)

        lq_pil = Image.open(fp).convert("RGB")
        w0, h0 = lq_pil.size            # 超分前分辨率 (WxH)
        lq_tensor = to_tensor(lq_pil).unsqueeze(0)

        if TXT_DIR is not None:
            with open(Path(TXT_DIR) / rel.with_suffix(".txt"), "r", encoding="utf-8") as fpp:
                prompt = fpp.read().strip()
        else:
            prompt = captioner(lq_pil)
        with open(prompt_path, "w", encoding="utf-8") as fpp:
            fpp.write(prompt)

        t1 = time.time()
        try:
            result = model.enhance(
                lq=lq_tensor,
                prompt=prompt,
                scale_by=SCALE_BY,
                upscale=UPSCALE,
                target_longest_side=TARGET_LONGEST_SIDE,
                patch_size=PATCH_SIZE,
                stride=STRIDE,
                return_type="pil",
            )[0]
            dt = time.time() - t1
            w1, h1 = result.size          # 超分后分辨率 (WxH)
            result.save(result_path)
            infer_times.append(dt)
            ok += 1
            pshow = (prompt[:40] + ("…" if len(prompt) > 40 else "")) if prompt else "<empty>"
            ratio = f"×{w1 / w0:.1f}" if w0 else ""
            print(f"[{i}/{len(images)}] {fp.name}  ->  {rel.with_suffix('.png').as_posix()}  "
                  f"| 分辨率 {w0}x{h0} -> {w1}x{h1} {ratio} | 推理 {dt:.2f}s | prompt: {pshow}")
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
