#!/usr/bin/env python3
"""Batch Qwen3-VL image-to-text inference over a folder of images, with timing.

Loads the Qwen3-VL model + processor ONCE (timed), then loops over every image
in IMAGE_DIR (walked recursively), asking PROMPT (or a per-image question from
TXT_DIR) and saving the generated text to
OUTPUT_DIR/result/<rel>.txt (+ OUTPUT_DIR/prompt/<rel>.txt = the question used).
Prints per-image generation time and a summary (avg/min/max/total).

Uses the Auto* classes (AutoProcessor / AutoModelForImageTextToText) so it works
for any Qwen3-VL checkpoint (7B / 30B-A3B / instruct / thinking) without hard-
coding a model class name. process_vision_info (qwen_vl_utils) extracts the
image from the chat message.

Env vars (set by 02_run_inference.sh):
  MODEL_PATH, IMAGE_DIR, TXT_DIR, OUTPUT_DIR,
  PROMPT, MAX_NEW_TOKENS, TEMPERATURE, TOP_P, TOP_K, REPETITION_PENALTY,
  DO_SAMPLE, SEED, DTYPE, DEVICE_MAP, LOAD_IN_4BIT, LOAD_IN_8BIT,
  ATTN_IMPL, THINKING, STRIP_THINKING
"""
import os
import re
import sys
import time
from pathlib import Path

import torch

MODEL_PATH = os.environ.get("MODEL_PATH")
IMAGE_DIR = os.environ.get("IMAGE_DIR")
TXT_DIR = os.environ.get("TXT_DIR") or None
OUTPUT_DIR = os.environ.get("OUTPUT_DIR")
PROMPT = os.environ.get(
    "PROMPT",
    "Describe this image in detail, including objects, scene, colors, text, and any notable details.",
)
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "512"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
TOP_P = float(os.environ.get("TOP_P", "0.8"))
TOP_K = int(os.environ.get("TOP_K", "20"))
REPETITION_PENALTY = float(os.environ.get("REPETITION_PENALTY", "1.05"))
DO_SAMPLE = os.environ.get("DO_SAMPLE", "true").lower() in ("1", "true", "yes")
SEED = int(os.environ.get("SEED", "231"))
DTYPE = os.environ.get("DTYPE", "bf16")
DEVICE_MAP = os.environ.get("DEVICE_MAP", "auto")
LOAD_IN_4BIT = os.environ.get("LOAD_IN_4BIT", "0") == "1"
LOAD_IN_8BIT = os.environ.get("LOAD_IN_8BIT", "0") == "1"
ATTN_IMPL = os.environ.get("ATTN_IMPL", "")          # "" = config default; "sdpa" | "flash_attention_2" | "eager"
THINKING = os.environ.get("THINKING", "0") == "1"
STRIP_THINKING = os.environ.get("STRIP_THINKING", "0") == "1"

DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif"}


def load_model():
    from transformers import AutoModelForImageTextToText, AutoProcessor

    torch_dtype = DTYPE_MAP.get(DTYPE, torch.bfloat16)
    kwargs = {"torch_dtype": torch_dtype, "device_map": DEVICE_MAP or None}
    if ATTN_IMPL:
        kwargs["attn_implementation"] = ATTN_IMPL

    # Optional bitsandbytes quantization (needs `pip install bitsandbytes`).
    if LOAD_IN_4BIT or LOAD_IN_8BIT:
        from transformers import BitsAndBytesConfig
        bnb_kwargs = {"bnb_4bit_compute_dtype": torch_dtype} if LOAD_IN_4BIT else {}
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=LOAD_IN_4BIT, load_in_8bit=LOAD_IN_8BIT, **bnb_kwargs
        )

    model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, **kwargs)
    model.eval()

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    # process_vision_info extracts images/videos from the chat message content.
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        sys.exit(
            "ERROR: qwen_vl_utils not installed. pip install qwen-vl-utils  "
            "(or run: INSTALL_DEPS=1 bash qwen3vl/00_setup_env.sh)"
        )
    return model, processor, process_vision_info


def build_inputs(processor, process_vision_info, image_path, prompt):
    """Build one chat message, apply the chat template, and run the processor."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": prompt},
        ],
    }]
    # Thinking variants: Qwen3's chat template accepts enable_thinking=True.
    # Older transformers / non-thinking variants lack the kwarg -> fall back.
    try:
        if THINKING:
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
            )
        else:
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    except TypeError:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # process_vision_info signature differs across qwen-vl-utils versions:
    # newer returns (images, videos, video_kwargs); older returns (images, videos).
    try:
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )
    except TypeError:
        image_inputs, video_inputs = process_vision_info(messages)
        video_kwargs = {}

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs or None,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    return inputs


def move_inputs_to_device(inputs, model):
    device = getattr(model, "device", None)
    if device is None or str(device) == "meta":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    return inputs


def gen_kwargs():
    if DO_SAMPLE:
        return {
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": True,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "repetition_penalty": REPETITION_PENALTY,
        }
    return {"max_new_tokens": MAX_NEW_TOKENS, "do_sample": False}


def strip_thinking(text):
    # Qwen3 thinking models wrap reasoning in <think>...</think>. Remove the
    # block (and any leading whitespace) to keep only the final answer.
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def main():
    if not MODEL_PATH:
        sys.exit("ERROR: MODEL_PATH not set.")
    if not IMAGE_DIR:
        sys.exit("ERROR: IMAGE_DIR not set.")
    if not OUTPUT_DIR:
        sys.exit("ERROR: OUTPUT_DIR not set.")

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print(f"[*] loading model (dtype={DTYPE} device_map={DEVICE_MAP or '<default>'} "
          f"4bit={LOAD_IN_4BIT} 8bit={LOAD_IN_8BIT} attn={ATTN_IMPL or 'default'}) ...")
    t_load0 = time.time()
    model, processor, process_vision_info = load_model()
    load_time = time.time() - t_load0
    print(f"[*] 模型加载耗时: {load_time:.2f}s")

    input_dir = Path(IMAGE_DIR)
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
          f"(max_new_tokens={MAX_NEW_TOKENS} do_sample={DO_SAMPLE} seed={SEED})")

    gk = gen_kwargs()
    infer_times = []
    ok = 0
    t_loop0 = time.time()
    with torch.inference_mode():
        for i, fp in enumerate(images, 1):
            rel = fp.relative_to(input_dir)
            result_path = result_dir / rel.with_suffix(".txt")
            prompt_path = prompt_dir / rel.with_suffix(".txt")
            result_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path.parent.mkdir(parents=True, exist_ok=True)

            if TXT_DIR is not None:
                qpath = Path(TXT_DIR) / rel.with_suffix(".txt")
                with open(qpath, "r", encoding="utf-8") as fpp:
                    question = fpp.read().strip()
            else:
                question = PROMPT
            with open(prompt_path, "w", encoding="utf-8") as fpp:
                fpp.write(question)

            t1 = time.time()
            try:
                inputs = build_inputs(processor, process_vision_info, fp, question)
                inputs = move_inputs_to_device(inputs, model)
                generated_ids = model.generate(**inputs, **gk)
                # Strip the prompt tokens, decode only the new ones.
                in_len = inputs["input_ids"].shape[1]
                out_ids = generated_ids[:, in_len:]
                text_out = processor.batch_decode(
                    out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                if STRIP_THINKING:
                    text_out = strip_thinking(text_out)
                dt = time.time() - t1
                with open(result_path, "w", encoding="utf-8") as fpp:
                    fpp.write(text_out)
                infer_times.append(dt)
                ok += 1
                qshow = (question[:40] + ("…" if len(question) > 40 else "")) if question else "<none>"
                oshow = (text_out[:40].replace("\n", " ") + ("…" if len(text_out) > 40 else "")) or "<empty>"
                print(f"[{i}/{len(images)}] {fp.name}  ->  {rel.with_suffix('.txt').as_posix()}  "
                      f"| 推理 {dt:.2f}s | Q: {qshow} | A: {oshow}")
            except Exception as e:  # noqa: BLE001
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
