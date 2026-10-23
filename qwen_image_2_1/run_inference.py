#!/usr/bin/env python3
"""Qwen-Image-2.1 推理（t2i / edit / rgba 三模式）。

Env vars:
  MODEL_PATH     模型目录（diffusers 结构，含 model_index.json）
  MODE           t2i | edit | rgba（rgba 仅约定 prompt 写法，调用同 t2i）
  PROMPT         提示词
  IMAGES         edit 模式输入图，逗号分隔（多图=多参考图编辑）
  ASPECT         1:1|4:3|3:4|3:2|2:3|16:9|9:16，设置后覆盖 WIDTH/HEIGHT
  WIDTH/HEIGHT   输出尺寸（默认 2048x2048）
  STEPS          采样步数（默认 40，官方示例值）
  SEED           随机种子（默认 42）
  GUIDANCE_SCALE 可选；不设则用 pipeline 默认值
  CPU_OFFLOAD    1=enable_model_cpu_offload（默认；24G 显存装不下 33G bf16 必须开）
                 0=整模 .to(cuda)（需 >=40G 显存）
  OUTPUT         输出 png 路径
"""
import os
import sys
import time

MODEL_PATH = os.environ.get("MODEL_PATH", "")
MODE = os.environ.get("MODE", "t2i").lower()
PROMPT = os.environ.get("PROMPT", "")
IMAGES = os.environ.get("IMAGES", "")
ASPECT = os.environ.get("ASPECT", "")
WIDTH = int(os.environ.get("WIDTH", "2048"))
HEIGHT = int(os.environ.get("HEIGHT", "2048"))
STEPS = int(os.environ.get("STEPS", "40"))
SEED = int(os.environ.get("SEED", "42"))
GUIDANCE_SCALE = os.environ.get("GUIDANCE_SCALE", "")
CPU_OFFLOAD = os.environ.get("CPU_OFFLOAD", "1") == "1"
OUTPUT = os.environ.get("OUTPUT", "")

# 官方 README 给出的支持分辨率表
ASPECT_RATIOS = {
    "1:1": (2048, 2048),
    "4:3": (2400, 1792),
    "3:4": (1792, 2400),
    "3:2": (2528, 1696),
    "2:3": (1696, 2528),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}


def main():
    if not MODEL_PATH or not os.path.isfile(os.path.join(MODEL_PATH, "model_index.json")):
        sys.exit(f"❌ MODEL_PATH 无效: {MODEL_PATH!r}（缺 model_index.json）")
    if MODE not in ("t2i", "edit", "rgba"):
        sys.exit(f"❌ MODE 必须是 t2i/edit/rgba，当前: {MODE!r}")
    if not PROMPT:
        sys.exit("❌ PROMPT 为空")
    if not OUTPUT:
        sys.exit("❌ OUTPUT 为空")

    width, height = WIDTH, HEIGHT
    if ASPECT:
        if ASPECT not in ASPECT_RATIOS:
            sys.exit(f"❌ ASPECT 不支持: {ASPECT!r}，可选 {sorted(ASPECT_RATIOS)}")
        width, height = ASPECT_RATIOS[ASPECT]

    import torch
    from diffusers import QwenImage21Pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 [{MODE}] 加载模型: {MODEL_PATH}")
    print(f"  🎮 device: {device}  cpu_offload: {CPU_OFFLOAD}")
    print(f"  📐 {width}x{height}  steps={STEPS}  seed={SEED}")

    pipe = QwenImage21Pipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
    if CPU_OFFLOAD:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    kwargs = dict(
        prompt=PROMPT,
        num_inference_steps=STEPS,
        generator=torch.Generator(device).manual_seed(SEED),
    )
    # edit 模式尺寸由输入图决定，不传 width/height（与官方示例一致）
    if MODE in ("t2i", "rgba"):
        kwargs["width"] = width
        kwargs["height"] = height
    if GUIDANCE_SCALE:
        kwargs["guidance_scale"] = float(GUIDANCE_SCALE)
    if MODE == "edit":
        from PIL import Image

        paths = [p.strip() for p in IMAGES.split(",") if p.strip()]
        if not paths:
            sys.exit("❌ edit 模式需要 IMAGES（逗号分隔）")
        imgs = []
        for p in paths:
            if not os.path.isfile(p):
                sys.exit(f"❌ 输入图不存在: {p}")
            imgs.append(Image.open(p).convert("RGB"))
        # 多图传 list（官方宣传的 10 参考图能力）；单图传 Image，与官方示例一致
        kwargs["image"] = imgs[0] if len(imgs) == 1 else imgs
        print(f"  🖼️ 参考图 {len(imgs)} 张")

    t0 = time.time()
    image = pipe(**kwargs).images[0]
    dt = time.time() - t0

    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT)), exist_ok=True)
    image.save(OUTPUT)
    print(f"⏱️ 采样耗时 {dt:.1f}s")
    print(f"✅ saved: {OUTPUT}")


if __name__ == "__main__":
    main()
