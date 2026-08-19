#!/usr/bin/env python3
"""MiniMax-H3 diffusers int8 量化推理（无需起服务，单卡常驻）。

06 的量化变体：用 int8 weight-only 量化（torchao Int8WeightOnlyConfig version=2）：
- 权重 int8（显存减半：transformer 61.7→~31GB，text_encoder 62.1→~31GB）
- 激活 bf16（精度损失小）+ 关键模块不量化（proj_in/out, AdaLN, time_embedder 等保留 bf16）
- 量化后总 ~65GB，单卡 80GB 常驻（不走两卡分拆/offload，无搬运开销，比 06 快）

Env vars: 同 06；DEVICE 默认 cuda:0（单卡，不分拆）
  MODEL_PATH, TASK, PROMPT/PROMPT_FILE, FIRST_FRAME/LAST_FRAME,
  NUM_FRAMES/DURATION, WIDTH/HEIGHT/MAX_PIXELS, FPS, SEED, DEVICE
"""
import os, sys, time

try:
    from diffusers import ModularPipeline, MiniMaxH3Transformer3DModel, TorchAoConfig
    from diffusers.hooks import apply_group_offloading
    from torchao.quantization import Int8WeightOnlyConfig
    from transformers import Qwen3VLForConditionalGeneration
    from transformers import TorchAoConfig as TransformersTorchAoConfig
except ImportError as e:
    sys.exit(f"❌ {e}. Install: pip install -U diffusers torchao transformers")


def main():
    MODEL_PATH = os.path.abspath(os.environ.get("MODEL_PATH", "../../model/MiniMax-H3"))
    TASK = os.environ.get("TASK", "")
    PROMPT = os.environ.get("PROMPT", "")
    PROMPT_FILE = os.environ.get("PROMPT_FILE", "")
    FIRST_FRAME = os.environ.get("FIRST_FRAME", "")
    LAST_FRAME = os.environ.get("LAST_FRAME", "")
    OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "../MiniMax-H3/results/int8")
    if not TASK:
        TASK = "fl2va" if FIRST_FRAME else "t2va"
    OUTPUT_NAME = os.environ.get("OUTPUT_NAME") or f"{TASK}_int8_seed{os.environ.get('SEED','0')}.mp4"
    DEVICE = os.environ.get("DEVICE", "cuda:0")  # 单卡常驻（int8 后 ~65GB，80GB 够放）
    NUM_FRAMES = int(os.environ.get("NUM_FRAMES", "124"))  # 帧数（需满足 17*n+5，124=17*4+5 ~5s@24fps）
    WIDTH = int(os.environ.get("WIDTH", "0"))
    HEIGHT = int(os.environ.get("HEIGHT", "0"))
    MAX_PIXELS = int(os.environ.get("MAX_PIXELS", str(512*768)))
    FPS = int(os.environ.get("FPS", "24"))
    SEED = int(os.environ.get("SEED", "0"))

    if not PROMPT and PROMPT_FILE:
        with open(PROMPT_FILE, encoding="utf-8") as f:
            PROMPT = f.read().strip()
    if not PROMPT:
        sys.exit("❌ PROMPT (or PROMPT_FILE) not set")

    # auto resolution（同 06）
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

    print(f"🚀 MiniMax-H3 int8 quantized inference (no server, single card)")
    print(f"  🤖 model: {MODEL_PATH}")
    print(f"  🎯 task: {TASK}")
    print(f"  🖼️ first_frame: {FIRST_FRAME or '(none)'}")
    print(f"  📐 num_frames: {NUM_FRAMES}")
    print(f"  🖼️ resolution: {WIDTH}x{HEIGHT}")
    print(f"  🎬 fps: {FPS}")
    print(f"  🎮 device: {DEVICE} (int8 ~65GB, single card resident)")

    import torch
    from diffusers.utils import load_image
    from diffusers.utils.export_utils import encode_video

    # int8 量化加载：用顶层 MODEL_PATH（diffusers 格式权重），不是 FL2VA/（原始 MiniMax 格式）。
    # 顶层 transformer/ 是 diffusers 转换后的格式（transformer_blocks.*, diffusion_pytorch_model-*.safetensors），
    # FL2VA/transformer/ 是原始格式（blocks.*, model-*.safetensors），MiniMaxH3Transformer3DModel 不认。
    # 关键模块（proj_in/out, AdaLN, time_embedder, visual, embed_tokens 等）不量化
    print("📦 loading int8 quantized pipeline (this takes minutes)...")
    pipe = ModularPipeline.from_pretrained(MODEL_PATH)
    pipe.update_components(
        transformer=MiniMaxH3Transformer3DModel.from_pretrained(
            MODEL_PATH, subfolder="transformer", dtype=torch.bfloat16,
            quantization_config=TorchAoConfig(
                Int8WeightOnlyConfig(version=2),
                modules_to_not_convert=[
                    "proj_in", "audio_proj_in", "context_embedder", "time_embedder", "time_proj",
                    "token_refiner", "norm_out", "proj_out", "audio_proj_out",
                ],
            ),
            low_cpu_mem_usage=False,
        ),
        text_encoder=Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, subfolder="text_encoder", dtype=torch.bfloat16,
            quantization_config=TransformersTorchAoConfig(
                Int8WeightOnlyConfig(version=2),
                modules_to_not_convert=[
                    "model.visual", "model.language_model.embed_tokens",
                    "model.language_model.norm", "lm_head",
                ],
            ),
            low_cpu_mem_usage=False,
        ),
    )
    pipe.load_components(workflow="fl2va", dtype=torch.bfloat16, pretrained_model_name_or_path=MODEL_PATH)
    pipe.transformer.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    # group_offload：transformer 按 block offload 到 CPU，去噪时只加载当前 block 到 GPU
    # （int8 version=2 的 tensor 是 pinnable，use_stream=True 异步搬运，开销小）
    # 显存只占当前 block（~2-3GB）+ 激活 + VAE（~3GB），单卡 80GB 绰绰有余
    offload = dict(onload_device=torch.device(DEVICE), offload_device=torch.device("cpu"), use_stream=True)
    pipe.transformer.enable_group_offload(offload_type="block_level", num_blocks_per_group=1, **offload)
    apply_group_offloading(pipe.text_encoder.model, offload_type="leaf_level", **offload)
    pipe.vae.to(DEVICE)       # VAE 常驻（解码用）
    pipe.audio_vae.to(DEVICE)
    print(f"  ✅ int8 pipeline loaded on {DEVICE} (transformer+text_encoder block-level offloaded)")

    generator = torch.Generator(DEVICE).manual_seed(SEED)
    outputs = ["videos", "audio", "sampling_rate"]

    t0 = time.time()
    kwargs = dict(prompt=PROMPT, num_frames=NUM_FRAMES, generator=generator, output=outputs)
    if WIDTH > 0:
        kwargs["width"] = WIDTH
    if HEIGHT > 0:
        kwargs["height"] = HEIGHT
    if FIRST_FRAME:
        kwargs["image"] = load_image(FIRST_FRAME)
    if LAST_FRAME:
        kwargs["last_image"] = load_image(LAST_FRAME)
    print("🎬 generating...")
    results = pipe(**kwargs)
    print(f"  ⏱️ generation: {time.time()-t0:.0f}s")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    encode_video(
        results["videos"][0],
        fps=FPS,
        output_path=out_path,
        audio=results["audio"][0],
        audio_sample_rate=results["sampling_rate"],
    )
    sz = os.path.getsize(out_path)
    print(f"✅ saved: {out_path} ({sz/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
