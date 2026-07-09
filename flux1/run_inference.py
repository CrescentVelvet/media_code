#!/usr/bin/env python3
"""Batch FLUX.1 text-to-image inference over a list of prompts, with timing.

Loads the diffusers FluxPipeline ONCE (timed), then loops over every prompt
(from PROMPTS_FILE, one per line; or a single PROMPT), generating a PNG per
prompt (or NUM_IMAGES_PER_PROMPT per prompt) and saving to
OUTPUT_DIR/result/<idx>_<slug>.png (+ OUTPUT_DIR/prompt/<idx>_<slug>.txt = the
prompt used). Prints per-prompt generation time and a summary (avg/min/max/total).

Defaults target FLUX.1-schnell (4-step, guidance 0.0, max_seq_len 256). For
FLUX.1-dev pass NUM_INFERENCE_STEPS=28 GUIDANCE_SCALE=3.5 MAX_SEQUENCE_LENGTH=512.

Env vars (set by 02_run_inference.sh):
  MODEL_PATH, PROMPT, PROMPTS_FILE, OUTPUT_DIR,
  NUM_INFERENCE_STEPS, GUIDANCE_SCALE, HEIGHT, WIDTH, MAX_SEQUENCE_LENGTH,
  SEED, NUM_IMAGES_PER_PROMPT, DTYPE, OFFLOAD, VAE_SLICING, VAE_TILING, ATTN_IMPL
"""
import os
import re
import sys
import time
from pathlib import Path

import torch

MODEL_PATH = os.environ.get("MODEL_PATH")
PROMPT = os.environ.get(
    "PROMPT",
    "A cinematic shot of a panda eating bamboo in a misty forest, soft morning light, highly detailed.",
)
PROMPTS_FILE = os.environ.get("PROMPTS_FILE") or None
OUTPUT_DIR = os.environ.get("OUTPUT_DIR")
NUM_INFERENCE_STEPS = int(os.environ.get("NUM_INFERENCE_STEPS", "4"))
GUIDANCE_SCALE = float(os.environ.get("GUIDANCE_SCALE", "0.0"))
HEIGHT = int(os.environ.get("HEIGHT", "1024"))
WIDTH = int(os.environ.get("WIDTH", "1024"))
MAX_SEQUENCE_LENGTH = int(os.environ.get("MAX_SEQUENCE_LENGTH", "256"))
SEED = int(os.environ.get("SEED", "231"))
NUM_IMAGES_PER_PROMPT = int(os.environ.get("NUM_IMAGES_PER_PROMPT", "1"))
DTYPE = os.environ.get("DTYPE", "bf16")
OFFLOAD = os.environ.get("OFFLOAD", "model")          # model | sequential | none
VAE_SLICING = os.environ.get("VAE_SLICING", "1") == "1"
VAE_TILING = os.environ.get("VAE_TILING", "0") == "1"
ATTN_IMPL = os.environ.get("ATTN_IMPL", "")           # "" = diffusers default(sdpa); flash_attention_2 | eager

DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def load_pipeline():
    from diffusers import FluxPipeline

    torch_dtype = DTYPE_MAP.get(DTYPE, torch.bfloat16)
    kwargs = {"torch_dtype": torch_dtype}

    # Optional: load the transformer with a chosen attention impl, then pass it
    # into the pipeline (FluxPipeline otherwise uses sdpa). Advanced; falls back
    # to the default load on any error so a missing flash-attn never blocks runs.
    if ATTN_IMPL:
        try:
            from diffusers import FluxTransformer2DModel
            transformer = FluxTransformer2DModel.from_pretrained(
                MODEL_PATH, subfolder="transformer",
                torch_dtype=torch_dtype, attn_implementation=ATTN_IMPL,
            )
            kwargs["transformer"] = transformer
            print(f"[*] transformer loaded with attn_implementation={ATTN_IMPL}")
        except Exception as e:  # noqa: BLE001
            print(f"[!] could not load transformer with attn_implementation={ATTN_IMPL}: {e}",
                  file=sys.stderr)
            print("    falling back to default (sdpa) attention.", file=sys.stderr)

    pipe = FluxPipeline.from_pretrained(MODEL_PATH, **kwargs)
    if VAE_SLICING:
        pipe.enable_vae_slicing()
    if VAE_TILING:
        pipe.enable_vae_tiling()

    # VRAM strategy (mutually exclusive). model offload fits ~12GB; none needs
    # the whole pipeline on one GPU (FLUX.1 is ~12B -> ~24GB bf16 for dev).
    if OFFLOAD == "sequential":
        pipe.enable_sequential_cpu_offload()
    elif OFFLOAD == "none":
        pipe.to("cuda")
    else:  # "model" or unknown -> safest default
        pipe.enable_model_cpu_offload()
    return pipe


def read_prompts():
    if PROMPTS_FILE:
        out = []
        with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                p = line.strip()
                if p and not p.startswith("#"):
                    out.append(p)
        if not out:
            sys.exit(f"ERROR: no prompts (non-empty, non-#) found in {PROMPTS_FILE}")
        return out
    return [PROMPT]


def slugify(s, n=50):
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE).strip().lower()
    s = re.sub(r"[\s_-]+", "_", s).strip("_")
    return s[:n] or "prompt"


def main():
    if not MODEL_PATH:
        sys.exit("ERROR: MODEL_PATH not set.")
    if not OUTPUT_DIR:
        sys.exit("ERROR: OUTPUT_DIR not set.")

    torch.manual_seed(SEED)

    print(f"[*] loading pipeline (dtype={DTYPE} offload={OFFLOAD} "
          f"vae_slicing={VAE_SLICING} vae_tiling={VAE_TILING} attn={ATTN_IMPL or 'default'}) ...")
    t_load0 = time.time()
    pipe = load_pipeline()
    load_time = time.time() - t_load0
    print(f"[*] 模型加载耗时: {load_time:.2f}s")

    prompts = read_prompts()
    output_dir = Path(OUTPUT_DIR)
    result_dir = output_dir / "result"
    prompt_dir = output_dir / "prompt"
    result_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)

    total = len(prompts) * NUM_IMAGES_PER_PROMPT
    print(f"[*] {len(prompts)} prompt(s) x {NUM_IMAGES_PER_PROMPT} img = {total} image(s) -> {result_dir}  "
          f"(steps={NUM_INFERENCE_STEPS} cfg={GUIDANCE_SCALE} {WIDTH}x{HEIGHT} max_seq_len={MAX_SEQUENCE_LENGTH} seed={SEED})")

    infer_times = []
    ok = 0
    idx = 0
    t_loop0 = time.time()
    with torch.inference_mode():
        for i, prompt in enumerate(prompts, 1):
            # Deterministic per-prompt seed (distinct across prompts, reproducible).
            generator = torch.Generator("cpu").manual_seed(SEED + i)
            slug = slugify(prompt)
            t1 = time.time()
            try:
                images = pipe(
                    prompt=prompt,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    height=HEIGHT,
                    width=WIDTH,
                    max_sequence_length=MAX_SEQUENCE_LENGTH,
                    num_images_per_prompt=NUM_IMAGES_PER_PROMPT,
                    generator=generator,
                ).images
                dt = time.time() - t1
                for j, img in enumerate(images, 1):
                    idx += 1
                    suffix = f"_v{j}" if NUM_IMAGES_PER_PROMPT > 1 else ""
                    name = f"{i:04d}_{slug}{suffix}"
                    img.save(result_dir / f"{name}.png")
                    with open(prompt_dir / f"{name}.txt", "w", encoding="utf-8") as fp:
                        fp.write(prompt)
                infer_times.append(dt)
                ok += len(images)
                pshow = prompt[:48] + ("…" if len(prompt) > 48 else "")
                print(f"[{i}/{len(prompts)}] {pshow}  ->  {WIDTH}x{HEIGHT} x{len(images)}  | 推理 {dt:.2f}s")
            except Exception as e:  # noqa: BLE001
                print(f"[{i}/{len(prompts)}] {prompt[:48]}  ! failed: {e}", file=sys.stderr)

    loop_time = time.time() - t_loop0
    pure = sum(infer_times)
    print(f"[*] done. {ok}/{total} image(s) succeeded. "
          f"模型加载 {load_time:.2f}s + 循环 {loop_time:.2f}s (其中纯推理 {pure:.2f}s)")
    if infer_times:
        avg = pure / len(infer_times)
        print(f"[*] 单提示词推理耗时: avg {avg:.2f}s, min {min(infer_times):.2f}s, "
              f"max {max(infer_times):.2f}s, 共 {len(infer_times)} 条")


if __name__ == "__main__":
    main()
