#!/usr/bin/env python3
"""MiniMax-H3 Turbo LoRA 加速推理（无需起服务，单卡 + auto CPU offload）。

06a 的 Turbo 变体：用 lightx2v/Minimax-h3-Turbo 蒸馏的 LoRA checkpoint 把
官方 50 步去噪压到 4/8 步（NFE=4 或 8），推理快 ~10×，效果接近原版。
- 不做 int8 量化（LoRA 蒸馏已省算力，bf16 即可）；用 ComponentsManager 的
  auto CPU offload 自动在 GPU/CPU 间搬运，单卡 80GB 可跑。
- LoRA 用 PEFT 手动注入（MiniMax-H3 是 ModularPipeline，标准
  pipe.load_lora_weights 不好用），照 Turbo 仓库 inference_minimax_h3.py。
- scheduler grid points = NFE + 1（MiniMaxH3Scheduler 把 num_inference_steps
  解释为含末尾零点的 sigma grid，N 次去噪要传 N+1）。
- 768p 4-step v1.0 checkpoint 必须传 VIDEO_SHIFT=6（训练 shift=6，非 12）。

Env vars:
  MODEL_PATH, LORA_PATH, TASK, PROMPT/PROMPT_FILE, FIRST_FRAME/LAST_FRAME,
  NUM_FRAMES/DURATION, WIDTH/HEIGHT/MAX_PIXELS, FPS, SEED, DEVICE,
  NUM_INFERENCE_STEPS, VIDEO_SHIFT, AUDIO_SHIFT, LORA_ALPHA, LORA_SCALE, FUSE_LORA
"""
import os, sys, time, gc
from pathlib import Path
from collections.abc import Mapping

import torch

try:
    from diffusers import ModularPipeline, ComponentsManager
    from diffusers.utils import load_image
    from diffusers.utils.export_utils import encode_video
    from peft import LoraConfig
    from safetensors.torch import load_file as load_safetensors_file
except ImportError as e:
    sys.exit(f"❌ {e}. Install: pip install -U diffusers peft safetensors")

from _turbo_lora import load_lora_adapter


def main():
    MODEL_PATH = os.path.abspath(os.environ.get("MODEL_PATH", "../../model/MiniMax-H3"))
    LORA_PATH = os.environ.get("LORA_PATH", "")
    TASK = os.environ.get("TASK", "")
    PROMPT = os.environ.get("PROMPT", "")
    PROMPT_FILE = os.environ.get("PROMPT_FILE", "")
    FIRST_FRAME = os.environ.get("FIRST_FRAME", "")
    LAST_FRAME = os.environ.get("LAST_FRAME", "")
    OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "../MiniMax-H3/results/turbo")
    if not TASK:
        TASK = "fl2va" if FIRST_FRAME else "t2va"
    OUTPUT_NAME = os.environ.get("OUTPUT_NAME") or f"{TASK}_turbo_seed{os.environ.get('SEED','0')}.mp4"
    DEVICE = os.environ.get("DEVICE", "cuda:0")
    NUM_FRAMES = int(os.environ.get("NUM_FRAMES", "124"))
    WIDTH = int(os.environ.get("WIDTH", "0"))
    HEIGHT = int(os.environ.get("HEIGHT", "0"))
    MAX_PIXELS = int(os.environ.get("MAX_PIXELS", str(1344*768)))
    FPS = int(os.environ.get("FPS", "24"))
    SEED = int(os.environ.get("SEED", "0"))
    # Turbo 蒸馏参数（默认 4-step v1.0 768p 配方）
    NUM_INFERENCE_STEPS = int(os.environ.get("NUM_INFERENCE_STEPS", "4"))
    VIDEO_SHIFT = float(os.environ.get("VIDEO_SHIFT", "6.0"))   # 768p 4-step 用 6；544p 用 12
    AUDIO_SHIFT = float(os.environ.get("AUDIO_SHIFT", "3.0"))
    LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "128"))       # 768p 4-step v1.0 训练用 128；544p 用 8
    LORA_SCALE = float(os.environ.get("LORA_SCALE", "1.0"))
    FUSE_LORA = os.environ.get("FUSE_LORA", "0") == "1"

    if not LORA_PATH:
        sys.exit("❌ LORA_PATH not set (Turbo LoRA checkpoint, e.g. minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors)")
    if not PROMPT and PROMPT_FILE:
        with open(PROMPT_FILE, encoding="utf-8") as f:
            PROMPT = f.read().strip()
    if not PROMPT:
        sys.exit("❌ PROMPT (or PROMPT_FILE) not set")

    # auto resolution（同 06a）
    if WIDTH == 0 or HEIGHT == 0:
        import math
        if FIRST_FRAME:
            from diffusers.utils import load_image as _li
            _img = _li(FIRST_FRAME)
            iw, ih = _img.size
            ratio = iw / ih
            print(f"  📐 auto ratio from image: {iw}x{ih} -> {ratio:.2f}")
        else:
            ratio = 16 / 9
            print(f"  📐 auto ratio: 16:9 (no image)")
        if ratio >= 1:
            H = int(math.sqrt(MAX_PIXELS / ratio))
            W = int(H * ratio)
        else:
            r2 = 1 / ratio
            W = int(math.sqrt(MAX_PIXELS / r2))
            H = int(W * r2)
        WIDTH = max(32, W - (W % 32))
        HEIGHT = max(32, H - (H % 32))
        print(f"  📐 auto resolution: {WIDTH}x{HEIGHT} ({WIDTH*HEIGHT} px, max {MAX_PIXELS})")

    workflow = "ref2va" if TASK == "ref2va" else "fl2va"
    print(f"🚀 MiniMax-H3 Turbo LoRA inference (no server, single card + auto offload)")
    print(f"  🤖 model: {MODEL_PATH}")
    print(f"  🏋️ lora: {LORA_PATH}")
    print(f"  🎯 task: {TASK} (workflow={workflow})")
    print(f"  🖼️ first_frame: {FIRST_FRAME or '(none)'}")
    print(f"  📐 num_frames: {NUM_FRAMES}")
    print(f"  📐 resolution: {WIDTH}x{HEIGHT}")
    print(f"  🎬 fps: {FPS}")
    print(f"  ⏱️ NFE: {NUM_INFERENCE_STEPS} (scheduler grid={NUM_INFERENCE_STEPS+1})")
    print(f"  🎮 device: {DEVICE} (bf16, auto CPU offload)")
    print(f"  📐 shifts: video={VIDEO_SHIFT} audio={AUDIO_SHIFT}")

    # 加载 pipeline（bf16，不量化；用 ComponentsManager auto CPU offload）
    print("📦 loading pipeline (this takes minutes)...")
    from _ensure_modular_index import ensure_modular_model_index
    print(f"  📦 {ensure_modular_model_index(MODEL_PATH)}")
    manager = ComponentsManager()
    manager.enable_auto_cpu_offload(device=DEVICE, memory_reserve_margin="12GB")
    pipe = ModularPipeline.from_pretrained(MODEL_PATH, components_manager=manager)
    pipe.load_components(workflow=workflow, dtype=torch.bfloat16,
                         pretrained_model_name_or_path=MODEL_PATH)

    # 选 active transformer（ref2va 用 transformer_ref，其他用 transformer）
    transformer_name = "transformer_ref" if workflow == "ref2va" else "transformer"
    active_transformer = getattr(pipe, transformer_name, None)
    if active_transformer is None:
        sys.exit(f"❌ workflow {workflow!r} requires {transformer_name!r}, not loaded")

    # 设 shift（Turbo 蒸馏配方：768p 4-step 用 video_shift=6）
    pipe.scheduler.set_shift(VIDEO_SHIFT)
    pipe.audio_scheduler.set_shift(AUDIO_SHIFT)

    # 注入 LoRA
    load_lora_adapter(active_transformer, LORA_PATH, LORA_ALPHA, LORA_SCALE, FUSE_LORA)

    generator = torch.Generator(DEVICE).manual_seed(SEED)
    outputs = ["videos", "audio", "sampling_rate"]

    t0 = time.time()
    kwargs = dict(
        prompt=PROMPT, num_frames=NUM_FRAMES, generator=generator, output=outputs,
        # MiniMaxH3Scheduler 把 num_inference_steps 解释为含末尾零点的 sigma grid，
        # N 次去噪要传 N+1（照 Turbo 仓库 inference_minimax_h3.py）
        num_inference_steps=NUM_INFERENCE_STEPS + 1,
        width=WIDTH, height=HEIGHT,
    )
    if FIRST_FRAME:
        kwargs["image"] = load_image(FIRST_FRAME)
    if LAST_FRAME:
        kwargs["last_image"] = load_image(LAST_FRAME)
    print("🎬 generating (Turbo {0}-step)...".format(NUM_INFERENCE_STEPS))
    with torch.inference_mode():
        results = pipe(**kwargs)
    print(f"  ⏱️ generation: {time.time()-t0:.0f}s")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    encode_video(
        results["videos"][0], fps=FPS, output_path=out_path,
        audio=results["audio"][0], audio_sample_rate=results["sampling_rate"],
    )
    sz = os.path.getsize(out_path)
    print(f"✅ saved: {out_path} ({sz/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
