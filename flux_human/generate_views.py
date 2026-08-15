#!/usr/bin/env python3
"""Flux1-dev + ControlNet(depth) + IP-Adapter(可选) 多视角静止人体生成。

对 03 渲染的每张骨骼深度图, 用 FluxControlNetPipeline 生成对应视角的人体
静止图像 (同 seed 锁随机性, depth ControlNet 锁几何, IP-Adapter 锁外观)。
输出 view_<view>.png。

⚠️ Flux ControlNet / IP-Adapter 的 API 随 diffusers / 模型 repo 变化较快:
- ControlNet: 用 diffusers FluxControlNetPipeline + FluxControlNetModel。
  InstantX Union controlnet 可能需要 control_mode / union flag — 首次跑时
  print(dir(controlnet)) introspect, 确认后再补 mode 参数。
- IP-Adapter (xlabs): 加载方式与 diffusers 原生不同, 标注 TODO; 默认
  USE_IPADAPTER=0 跳过, 跑通 ControlNet 后再开。

Env vars (set by 04_generate_views.sh):
  MODEL_PATH, CONTROLNET_PATH, IPADAPTER_PATH, DEPTH_DIR, VIEWS_DIR, REF_IMAGE,
  PROMPT, NUM_VIEWS, SEED, HEIGHT, WIDTH, NUM_INFERENCE_STEPS, GUIDANCE_SCALE,
  CONTROLNET_SCALE, DTYPE, OFFLOAD, USE_IPADAPTER
"""
import os
import sys
import glob
import time
import numpy as np
import torch
from PIL import Image

MODEL_PATH = os.environ.get("MODEL_PATH")
CONTROLNET_PATH = os.environ.get("CONTROLNET_PATH", "")
IPADAPTER_PATH = os.environ.get("IPADAPTER_PATH", "")
DEPTH_DIR = os.environ.get("DEPTH_DIR")
VIEWS_DIR = os.environ.get("VIEWS_DIR")
REF_IMAGE = os.environ.get("REF_IMAGE", "")
PROMPT = os.environ.get(
    "PROMPT",
    "a person standing still, full body, detailed clothing, "
    "studio lighting, neutral background, photorealistic, high detail",
)
NUM_VIEWS = int(os.environ.get("NUM_VIEWS", "24"))
SEED = int(os.environ.get("SEED", "231"))
HEIGHT = int(os.environ.get("HEIGHT", "1024"))
WIDTH = int(os.environ.get("WIDTH", "1024"))
NUM_INFERENCE_STEPS = int(os.environ.get("NUM_INFERENCE_STEPS", "28"))
GUIDANCE_SCALE = float(os.environ.get("GUIDANCE_SCALE", "3.5"))
CONTROLNET_SCALE = float(os.environ.get("CONTROLNET_SCALE", "0.7"))
DTYPE = os.environ.get("DTYPE", "bf16")
OFFLOAD = os.environ.get("OFFLOAD", "model")
USE_IPADAPTER = os.environ.get("USE_IPADAPTER", "0") == "1"

DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def load_pipeline():
    from diffusers import FluxControlNetPipeline, FluxControlNetModel

    torch_dtype = DTYPE_MAP.get(DTYPE, torch.bfloat16)
    print(f"🤖 Flux1-dev + ControlNet(depth)  dtype={DTYPE} offload={OFFLOAD}")
    print(f"🏋️ base: {MODEL_PATH}")
    print(f"🏋️ controlnet: {CONTROLNET_PATH}")

    controlnet = None
    if CONTROLNET_PATH:
        controlnet = FluxControlNetModel.from_pretrained(
            CONTROLNET_PATH, torch_dtype=torch_dtype
        )
        # ⚠️ introspect: InstantX Union controlnet 可能需 control_mode/union flag.
        # 首次跑看输出, 若生成的视角和 depth 不符, 补 union mode 参数。
        hints = [a for a in dir(controlnet) if "mode" in a.lower() or "union" in a.lower()]
        if hints:
            print(f"🔍 controlnet mode/union attrs: {hints}  (Union 模型可能要指定 depth mode)")

    pipe = FluxControlNetPipeline.from_pretrained(
        MODEL_PATH, controlnet=controlnet, torch_dtype=torch_dtype
    )

    if OFFLOAD == "sequential":
        pipe.enable_sequential_cpu_offload()
    elif OFFLOAD == "none":
        pipe.to("cuda")
    else:
        pipe.enable_model_cpu_offload()

    if USE_IPADAPTER and IPADAPTER_PATH:
        print(f"🖼️ ip-adapter: {IPADAPTER_PATH}  (⚠️ xlabs 加载 API 待验证)")
        try:
            # TODO: xlabs flux-ip-adapter 加载方式 (pipe.load_ip_adapter / 自定义);
            # diffusers 原生 Flux IP-Adapter 支持待确认。先 introspect:
            print(f"🔍 pipe ip-adapter methods: {[m for m in dir(pipe) if 'ip' in m.lower()]}")
            # pipe.load_ip_adapter(IPADAPTER_PATH, ...)  # uncomment after verifying API
            print("⚠️ IP-Adapter 未加载 (API 待验证); 仅用 ControlNet depth 生成")
        except Exception as e:
            print(f"⚠️ IP-Adapter 加载失败 (跳过): {e}")

    return pipe


def load_depth_as_control_image(path):
    """Load depth png -> RGB PIL (FluxControlNet expects 3-channel control_image)."""
    d = Image.open(path).convert("L").resize((WIDTH, HEIGHT), Image.BILINEAR)
    return Image.merge("RGB", [d, d, d])


def main():
    if not MODEL_PATH:
        sys.exit("❌ MODEL_PATH not set")
    if not DEPTH_DIR or not VIEWS_DIR:
        sys.exit("❌ DEPTH_DIR / VIEWS_DIR not set")

    depth_files = sorted(glob.glob(os.path.join(DEPTH_DIR, "depth_*.png")))
    if not depth_files:
        sys.exit(f"❌ no depth_*.png in {DEPTH_DIR} — run 03 first")
    if NUM_VIEWS < len(depth_files):
        depth_files = depth_files[:NUM_VIEWS]
    print(f"🖼️ {len(depth_files)} 张深度图 -> 生成 {len(depth_files)} 视角")

    os.makedirs(VIEWS_DIR, exist_ok=True)
    torch.manual_seed(SEED)

    print("🚀 loading pipeline ...")
    t0 = time.time()
    pipe = load_pipeline()
    print(f"⏱️ 模型加载: {time.time()-t0:.2f}s")

    times = []
    ok = 0
    with torch.inference_mode():
        for i, dp in enumerate(depth_files, 1):
            ctrl = load_depth_as_control_image(dp)
            gen = torch.Generator("cuda").manual_seed(SEED)  # same seed -> lock noise
            t1 = time.time()
            try:
                out = pipe(
                    prompt=PROMPT,
                    control_image=ctrl,
                    controlnet_conditioning_scale=CONTROLNET_SCALE,
                    height=HEIGHT,
                    width=WIDTH,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    generator=gen,
                )
                img = out.images[0]
                name = f"view_{i-1:03d}.png"
                img.save(os.path.join(VIEWS_DIR, name))
                dt = time.time() - t1
                times.append(dt)
                ok += 1
                print(f"[{i}/{len(depth_files)}] {os.path.basename(dp)} -> {name}  | 推理 {dt:.2f}s")
            except Exception as e:
                print(f"[{i}/{len(depth_files)}] {os.path.basename(dp)} ❌ failed: {e}", file=sys.stderr)

    print(f"✅ done. {ok}/{len(depth_files)} 视角生成成功.")
    if times:
        print(f"⏱️ 单视角推理: avg {sum(times)/len(times):.2f}s, 共 {len(times)} 视角")
    print(f"   输出: {VIEWS_DIR}/view_*.png")
    print(f"   下一步: 05_reconstruct (3DGS/NeuS 重建) — 见 README 待办")


if __name__ == "__main__":
    main()
