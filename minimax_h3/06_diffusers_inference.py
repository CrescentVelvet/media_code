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
  TRANSFORMER_DEVICE:        设备（默认 cuda；两卡设 cuda:0，参考文档分拆 text_encoder 到 cuda:1）
"""
import os, sys, time

try:
    from diffusers import ComponentsManager, ModularPipeline
except ImportError:
    sys.exit("❌ diffusers not installed or too old (need ModularPipeline). Install: pip install -U diffusers")


def main():
    MODEL_PATH = os.path.abspath(os.environ.get("MODEL_PATH", "../../model/MiniMax-H3"))
    # modular_model_index.json 里组件的 pretrained_model_name_or_path 写的是
    # "MiniMaxAI/MiniMax-H3"（HF Hub ID），load_components 会从 HF 下载。
    # 传 pretrained_model_name_or_path=MODEL_PATH 覆盖，强制从本地路径加载。
    TASK = os.environ.get("TASK", "")
    PROMPT = os.environ.get("PROMPT", "")
    PROMPT_FILE = os.environ.get("PROMPT_FILE", "")  # 读 prompt 文件（优先级低于 PROMPT）
    FIRST_FRAME = os.environ.get("FIRST_FRAME", "")
    LAST_FRAME = os.environ.get("LAST_FRAME", "")
    OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "../MiniMax-H3/results/diffusers")
    # TASK 不传时根据 FIRST_FRAME 自动决定：有首帧就 fl2va，没有就 t2va
    if not TASK:
        TASK = "fl2va" if FIRST_FRAME else "t2va"
    OUTPUT_NAME = os.environ.get("OUTPUT_NAME") or f"{TASK}_seed{os.environ.get('SEED','0')}.mp4"
    TRANSFORMER_DEVICE = os.environ.get("TRANSFORMER_DEVICE", "cuda:0")
    TEXT_ENCODER_DEVICE = os.environ.get("TEXT_ENCODER_DEVICE", "cuda:1")
    NUM_FRAMES = int(os.environ.get("NUM_FRAMES", "124"))  # 帧数（需满足 17*n+5，124=17*4+5 ~5s@24fps）
    WIDTH = int(os.environ.get("WIDTH", "0"))   # 宽（0=auto 按图比例算，必须 32 倍数）
    HEIGHT = int(os.environ.get("HEIGHT", "0"))  # 高（0=auto 按图比例算，必须 32 倍数）
    MAX_PIXELS = int(os.environ.get("MAX_PIXELS", str(512*768)))  # auto 时总像素上限（默认 ~0.39M）
    FPS = int(os.environ.get("FPS", "24"))       # 输出 mp4 帧率（模型固定 24fps，改只影响 mp4 容器）
    SEED = int(os.environ.get("SEED", "0"))

    # WIDTH/HEIGHT 不传时自动算：有首帧图按图比例，无图默认 16:9；总像素不超过 MAX_PIXELS
    if WIDTH == 0 or HEIGHT == 0:
        import math
        if FIRST_FRAME:
            from diffusers.utils import load_image as _li
            _img = _li(FIRST_FRAME)
            iw, ih = _img.size
            ratio = iw / ih  # 宽/高
            print(f"  📐 auto ratio from image: {iw}x{ih} -> {ratio:.2f}")
        else:
            ratio = 16/9
            print(f"  📐 auto ratio: 16:9 (no image)")
        # 总像素 W*H < MAX_PIXELS，W=ratio*H（横）或 H=(1/ratio)*W（竖）
        if ratio >= 1:  # 横屏 W>=H
            H = int(math.sqrt(MAX_PIXELS / ratio))
            W = int(H * ratio)
        else:  # 竖屏 H>W
            r2 = 1 / ratio
            W = int(math.sqrt(MAX_PIXELS / r2))
            H = int(W * r2)
        # 32 倍数对齐（向下取整，保证不超阈值）
        WIDTH = max(32, W - (W % 32))
        HEIGHT = max(32, H - (H % 32))
        print(f"  📐 auto resolution: {WIDTH}x{HEIGHT} ({WIDTH*HEIGHT} px, max {MAX_PIXELS})")

    if not PROMPT and PROMPT_FILE:
        with open(PROMPT_FILE, encoding="utf-8") as f:
            PROMPT = f.read().strip()
    if not PROMPT:
        sys.exit("❌ PROMPT (or PROMPT_FILE) not set")
    if TASK not in ("t2va", "fl2va"):
        sys.exit(f"❌ TASK={TASK} not supported (this script does t2va/fl2va; use 02_serve+03_generate for ref2va)")

    print(f"🚀 MiniMax-H3 diffusers inference (no server)")
    print(f"  🤖 model: {MODEL_PATH}")
    print(f"  🎯 task: {TASK}")
    print(f"  🖼️ first_frame: {FIRST_FRAME or '(none)'}")
    print(f"  🖼️ last_frame: {LAST_FRAME or '(none)'}")
    print(f"  📐 num_frames: {NUM_FRAMES}")
    print(f"  🖼️ resolution: {WIDTH}x{HEIGHT}")
    print(f"  🎬 fps: {FPS}")
    print(f"  🎮 device: {TRANSFORMER_DEVICE} (rest) + {TEXT_ENCODER_DEVICE} (text_encoder)")

    import torch
    from diffusers.utils import load_image
    from diffusers.utils.export_utils import encode_video

    # 两卡分拆（单卡 80GB 放不下 transformer 61.7GB + text_encoder 62.1GB）：
    # text_encoder 放 TEXT_ENCODER_DEVICE，rest（transformer/vae/...）放 TRANSFORMER_DEVICE。
    # pretrained_model_name_or_path 覆盖 modular_model_index.json 里的 HF Hub ID，强制本地加载。
    print("📦 loading pipeline (two-card split, this takes minutes)...")
    from _ensure_modular_index import ensure_modular_model_index
    print(f"  📦 {ensure_modular_model_index(MODEL_PATH)}")
    workflow = ModularPipeline.from_pretrained(MODEL_PATH).blocks.get_workflow("fl2va")

    # 1) text_encoder 拆出来放 TEXT_ENCODER_DEVICE
    text_manager = ComponentsManager()
    text_manager.enable_auto_cpu_offload(device=TEXT_ENCODER_DEVICE)
    conditioner = workflow.sub_blocks.pop("text_encoder").init_pipeline(
        MODEL_PATH, components_manager=text_manager
    )
    conditioner.load_components(dtype=torch.bfloat16, pretrained_model_name_or_path=MODEL_PATH)
    print(f"  ✅ text_encoder loaded on {TEXT_ENCODER_DEVICE}")

    # 2) rest（transformer/vae/scheduler/...）放 TRANSFORMER_DEVICE
    manager = ComponentsManager()
    manager.enable_auto_cpu_offload(device=TRANSFORMER_DEVICE)
    rest = workflow.init_pipeline(MODEL_PATH, components_manager=manager)
    rest.load_components(dtype=torch.bfloat16, pretrained_model_name_or_path=MODEL_PATH)
    print(f"  ✅ rest loaded on {TRANSFORMER_DEVICE}")

    generator = torch.Generator().manual_seed(SEED)
    outputs = ["videos", "audio", "sampling_rate"]

    t0 = time.time()
    # conditioner 编码 prompt（+ image/last_image keyframe），rest 用 state 生成
    print("🎬 encoding prompt + generating...")
    state = conditioner(prompt=PROMPT)
    rest_kwargs = dict(state=state, num_frames=NUM_FRAMES, generator=generator, output=outputs)
    if WIDTH > 0:
        rest_kwargs["width"] = WIDTH
    if HEIGHT > 0:
        rest_kwargs["height"] = HEIGHT
    if FIRST_FRAME:
        rest_kwargs["image"] = load_image(FIRST_FRAME)
    if LAST_FRAME:
        rest_kwargs["last_image"] = load_image(LAST_FRAME)
    results = rest(**rest_kwargs)
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
