#!/usr/bin/env python3
"""MiniMax-H3 diffusers 直接推理（无需起服务）。

用 diffusers 的 ModularPipeline 直接跑，输入图像+文字生成视频+音频。
不需要 SGLang serve，适合单次生成或不想起服务的场景。

显存：transformer 61.7GB + Qwen3-VL conditioner 62.1GB，单卡 80GB 放不下，
用 ComponentsManager auto offload（CPU↔GPU 搬运）。比 SGLang 慢（没有 Ulysses/TP
多卡并行），但简单直接。8 卡 A100 可参考 diffusers 文档两卡方案（text_encoder
cuda:1, rest cuda:0，全 bfloat16 不用 offload）。

Env vars:
  MODEL_PATH:    模型路径（默认 MiniMaxAI/MiniMax-H3）
  TASK:          t2va | fl2va（默认 t2va）
  PROMPT:        提示词（必填）
  FIRST_FRAME:   首帧图像路径（fl2va，本地路径或 URL）
  LAST_FRAME:    末帧图像路径（fl2va，可选）
  OUTPUT_DIR:    输出目录（默认 ../MiniMax-H3/results/diffusers）
  OUTPUT_NAME:   输出文件名（默认 <task>_seed<seed>.mp4）
  NUM_FRAMES:    帧数（默认 124，对应 ~5s@24fps）
  SEED:          种子（默认 0）
  DEVICE:        设备（默认 cuda；两卡设 cuda:0，参考文档分拆 text_encoder 到 cuda:1）
"""
import os, sys, time

try:
    from diffusers import ComponentsManager, ModularPipeline
except ImportError:
    sys.exit("❌ diffusers not installed or too old (need ModularPipeline). Install: pip install -U diffusers")


def main():
    MODEL_PATH = os.environ.get("MODEL_PATH", "MiniMaxAI/MiniMax-H3")
    TASK = os.environ.get("TASK", "t2va")
    PROMPT = os.environ.get("PROMPT", "")
    FIRST_FRAME = os.environ.get("FIRST_FRAME", "")
    LAST_FRAME = os.environ.get("LAST_FRAME", "")
    OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "../MiniMax-H3/results/diffusers")
    OUTPUT_NAME = os.environ.get("OUTPUT_NAME") or f"{TASK}_seed{os.environ.get('SEED','0')}.mp4"
    DEVICE = os.environ.get("DEVICE", "cuda")
    NUM_FRAMES = int(os.environ.get("NUM_FRAMES", "124"))
    SEED = int(os.environ.get("SEED", "0"))

    if not PROMPT:
        sys.exit("❌ PROMPT not set")
    if TASK not in ("t2va", "fl2va"):
        sys.exit(f"❌ TASK={TASK} not supported (this script does t2va/fl2va; use 02_serve+03_generate for ref2va)")

    print(f"🚀 MiniMax-H3 diffusers inference (no server)")
    print(f"  🤖 model: {MODEL_PATH}")
    print(f"  🎯 task: {TASK}")
    print(f"  🖼️ first_frame: {FIRST_FRAME or '(none)'}")
    print(f"  🖼️ last_frame: {LAST_FRAME or '(none)'}")
    print(f"  📐 num_frames: {NUM_FRAMES}")
    print(f"  🎮 device: {DEVICE}")

    import torch
    from diffusers.utils import load_image
    from diffusers.utils.export_utils import encode_video

    # 加载 pipeline（ComponentsManager auto offload：单卡 80GB 放不下 61.7+62.1GB）
    print("📦 loading pipeline (this takes minutes)...")
    manager = ComponentsManager()
    manager.enable_auto_cpu_offload(device=DEVICE)

    # fl2va workflow 覆盖 t2va（t2va 是 fl2va 无 keyframe）
    pipe = ModularPipeline.from_pretrained(MODEL_PATH, workflow="fl2va", components_manager=manager)
    pipe.load_components(dtype=torch.bfloat16)
    print("  ✅ pipeline loaded")

    generator = torch.Generator().manual_seed(SEED)
    outputs = ["videos", "audio", "sampling_rate"]

    t0 = time.time()
    kwargs = dict(prompt=PROMPT, num_frames=NUM_FRAMES, generator=generator, output=outputs)
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
        fps=24,
        output_path=out_path,
        audio=results["audio"][0],
        audio_sample_rate=results["sampling_rate"],
    )
    sz = os.path.getsize(out_path)
    print(f"✅ saved: {out_path} ({sz/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
