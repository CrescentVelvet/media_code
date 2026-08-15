#!/usr/bin/env python3
"""Batch FLUX.2 text-to-image AND image-editing inference, with timing.

Loads a Flux2 pipeline ONCE (timed), then loops over every prompt (from
PROMPTS_FILE, one per line; or a single PROMPT), generating a PNG per prompt
(or NUM_IMAGES_PER_PROMPT per prompt) and saving to
OUTPUT_DIR/result/<idx>_<slug>.png (+ OUTPUT_DIR/prompt/<idx>_<slug>.txt = the
prompt used). Prints per-prompt generation time and a summary.

Two models, one script (MODEL_TYPE selects the diffusers pipeline + defaults):
  - dev   : Flux2Pipeline           — 32B MMDiT + Mistral3 (24B) text encoder.
            guidance-distilled; defaults steps=50 guidance=4.0. Needs H100-class
            (OFFLOAD=model) or quantization (QUANT=4bit) on consumer GPUs.
  - klein : Flux2KleinPipeline      — 9B MMDiT + Qwen3 (8B) text encoder.
            step- AND guidance-distilled; defaults steps=4 guidance=1.0
            (guidance baked in / ignored). Sub-second; fits consumer GPUs.

Both pipelines also do image EDITING / multi-reference: pass INPUT_IMAGES
(comma-separated paths) and they are fed as reference images alongside the
prompt (single-ref edit or multi-ref compose). No input images -> pure T2I.

Env vars (set by 02_run_inference.sh):
  MODEL_PATH, MODEL_TYPE, PROMPT, PROMPTS_FILE, INPUT_IMAGES, OUTPUT_DIR,
  NUM_INFERENCE_STEPS, GUIDANCE_SCALE, HEIGHT, WIDTH, MAX_SEQUENCE_LENGTH,
  SEED, NUM_IMAGES_PER_PROMPT, DTYPE, OFFLOAD, QUANT
"""
import os
import re
import sys
import time
from pathlib import Path

import torch

MODEL_PATH = os.environ.get("MODEL_PATH")
MODEL_TYPE = os.environ.get("MODEL_TYPE", "klein").lower()
PROMPT = os.environ.get(
    "PROMPT",
    "A cinematic shot of a panda eating bamboo in a misty forest, soft morning light, highly detailed.",
)
PROMPTS_FILE = os.environ.get("PROMPTS_FILE") or None
INPUT_IMAGES = os.environ.get("INPUT_IMAGES") or ""   # comma-separated paths (editing/multi-ref)
OUTPUT_DIR = os.environ.get("OUTPUT_DIR")

# Per-model defaults (match official FLUX2_MODEL_INFO). Overridable via env.
IS_DEV = MODEL_TYPE == "dev"
DEF_STEPS = 50 if IS_DEV else 4
DEF_GUIDANCE = 4.0 if IS_DEV else 1.0   # klein: guidance distilled (ignored), 1.0 = no-op
NUM_INFERENCE_STEPS = int(os.environ.get("NUM_INFERENCE_STEPS", str(DEF_STEPS)))
GUIDANCE_SCALE = float(os.environ.get("GUIDANCE_SCALE", str(DEF_GUIDANCE)))
HEIGHT = int(os.environ.get("HEIGHT", "1024"))
WIDTH = int(os.environ.get("WIDTH", "1024"))
MAX_SEQUENCE_LENGTH = int(os.environ.get("MAX_SEQUENCE_LENGTH", "512"))
SEED = int(os.environ.get("SEED", "231"))
NUM_IMAGES_PER_PROMPT = int(os.environ.get("NUM_IMAGES_PER_PROMPT", "1"))
DTYPE = os.environ.get("DTYPE", "bf16")
OFFLOAD = os.environ.get("OFFLOAD", "model")          # model | sequential | none
QUANT = os.environ.get("QUANT", "").lower()           # "" | 4bit | 8bit (needs bitsandbytes)

DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def load_pipeline():
    from diffusers import Flux2Pipeline, Flux2KleinPipeline

    pipe_cls = Flux2Pipeline if IS_DEV else Flux2KleinPipeline
    torch_dtype = DTYPE_MAP.get(DTYPE, torch.bfloat16)
    print(f"🤖 {MODEL_TYPE}: {pipe_cls.__name__}  dtype={DTYPE} offload={OFFLOAD} "
          f"quant={QUANT or 'none'}")
    print(f"🏋️ weights: {MODEL_PATH}")

    kwargs = {"torch_dtype": torch_dtype}

    # Optional on-the-fly quantization (dev on consumer GPUs). Mirrors the
    # diffusers FLUX.2-dev 4-bit guide: load transformer + text_encoder with a
    # BitsAndBytesConfig, pass them into the pipeline, then enable_model_cpu_offload.
    # For the EASIEST 4-bit path, instead point MODEL_PATH at the pre-quantized
    # 'diffusers/FLUX.2-dev-bnb-4bit' repo (no QUANT needed).
    if QUANT in ("4bit", "8bit"):
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            sys.exit(
                f"❌ QUANT={QUANT} needs bitsandbytes. Install: pip install bitsandbytes\n"
                f"   (or use the pre-quantized repo: MODEL_PATH=.../FLUX.2-dev-bnb-4bit QUANT= )"
            )
        from transformers import BitsAndBytesConfig
        from diffusers import Flux2Transformer2DModel
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=(QUANT == "4bit"),
            load_in_8bit=(QUANT == "8bit"),
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ) if QUANT == "4bit" else BitsAndBytesConfig(load_in_8bit=True)
        print(f"📦 loading quantized transformer ({QUANT}) ...")
        transformer = Flux2Transformer2DModel.from_pretrained(
            MODEL_PATH, subfolder="transformer",
            quantization_config=bnb_config, torch_dtype=torch_dtype,
        )
        # Text encoder class differs per model (Mistral3 for dev, Qwen3 for klein).
        if IS_DEV:
            from transformers import Mistral3ForConditionalGeneration as TE
        else:
            from transformers import Qwen3ForCausalLM as TE
        print(f"📦 loading quantized text_encoder ({QUANT}) ...")
        text_encoder = TE.from_pretrained(
            MODEL_PATH, subfolder="text_encoder",
            quantization_config=bnb_config, torch_dtype=torch_dtype, device_map="cpu",
        )
        kwargs["transformer"] = transformer
        kwargs["text_encoder"] = text_encoder

    pipe = pipe_cls.from_pretrained(MODEL_PATH, **kwargs)

    # VRAM strategy (mutually exclusive). Flux2 pipelines do NOT multi-GPU
    # shard — use offload for VRAM. Quantized components: keep model offload.
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
            sys.exit(f"❌ no prompts (non-empty, non-#) found in {PROMPTS_FILE}")
        return out
    return [PROMPT]


def load_ref_images():
    """Load INPUT_IMAGES (comma-separated paths) as a list of PIL images."""
    if not INPUT_IMAGES:
        return None
    from PIL import Image
    paths = [p.strip() for p in INPUT_IMAGES.split(",") if p.strip()]
    imgs = []
    for p in paths:
        if not os.path.isfile(p):
            sys.exit(f"❌ input image not found: {p}")
        imgs.append(Image.open(p).convert("RGB"))
        print(f"🖼️ ref image: {p}")
    return imgs  # None (T2I) or list[PIL.Image] (editing/multi-ref)


def slugify(s, n=50):
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE).strip().lower()
    s = re.sub(r"[\s_-]+", "_", s).strip("_")
    return s[:n] or "prompt"


def main():
    if not MODEL_PATH:
        sys.exit("❌ MODEL_PATH not set.")
    if not OUTPUT_DIR:
        sys.exit("❌ OUTPUT_DIR not set.")
    if MODEL_TYPE not in ("dev", "klein"):
        sys.exit(f"❌ MODEL_TYPE must be 'dev' or 'klein' (got {MODEL_TYPE!r}).")

    torch.manual_seed(SEED)

    print(f"🚀 loading pipeline ...")
    t_load0 = time.time()
    pipe = load_pipeline()
    load_time = time.time() - t_load0
    print(f"⏱️ 模型加载耗时: {load_time:.2f}s")

    prompts = read_prompts()
    ref_images = load_ref_images()
    output_dir = Path(OUTPUT_DIR)
    result_dir = output_dir / "result"
    prompt_dir = output_dir / "prompt"
    result_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)

    total = len(prompts) * NUM_IMAGES_PER_PROMPT
    mode = "editing" if ref_images else "T2I"
    print(f"[*] {len(prompts)} prompt(s) x {NUM_IMAGES_PER_PROMPT} img = {total} image(s) "
          f"[{mode}] -> {result_dir}  "
          f"(steps={NUM_INFERENCE_STEPS} cfg={GUIDANCE_SCALE} {WIDTH}x{HEIGHT} "
          f"max_seq_len={MAX_SEQUENCE_LENGTH} seed={SEED})")

    infer_times = []
    ok = 0
    idx = 0
    t_loop0 = time.time()
    with torch.inference_mode():
        for i, prompt in enumerate(prompts, 1):
            generator = torch.Generator("cuda").manual_seed(SEED + i)
            slug = slugify(prompt)
            t1 = time.time()
            try:
                out = pipe(
                    prompt=prompt,
                    image=ref_images,                  # None (T2I) or list[PIL] (editing)
                    height=HEIGHT,
                    width=WIDTH,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    num_images_per_prompt=NUM_IMAGES_PER_PROMPT,
                    max_sequence_length=MAX_SEQUENCE_LENGTH,
                    generator=generator,
                )
                images = out.images
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
                print(f"[{i}/{len(prompts)}] {prompt[:48]}  ❌ failed: {e}", file=sys.stderr)

    loop_time = time.time() - t_loop0
    pure = sum(infer_times)
    print(f"✅ done. {ok}/{total} image(s) succeeded. "
          f"模型加载 {load_time:.2f}s + 循环 {loop_time:.2f}s (其中纯推理 {pure:.2f}s)")
    if infer_times:
        avg = pure / len(infer_times)
        print(f"⏱️ 单提示词推理耗时: avg {avg:.2f}s, min {min(infer_times):.2f}s, "
              f"max {max(infer_times):.2f}s, 共 {len(infer_times)} 条")


if __name__ == "__main__":
    main()
