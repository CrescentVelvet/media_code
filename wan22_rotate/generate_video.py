#!/usr/bin/env python3
"""Generate a 360-degree rotation video with Wan2.2-TI2V-5B + LoRA.

Uses ModelConfig(path=...) to load model files directly.
Height/width auto-calculated from input image aspect ratio (aligned to mod_value=16).

Env vars (set by 02_generate_video.sh):
  WAN_MODEL_PATH, WEIGHT_PATH, PROMPT, NEGATIVE_PROMPT, INPUT_IMAGE,
  OUTPUT_DIR, OUTPUT_NAME, NUM_FRAMES, SEED, TILED, FPS, QUALITY,
  DEVICE, MAX_AREA, NUM_INFERENCE_STEPS, CFG_SCALE
"""
import os
import sys
import time
import glob

import numpy as np
import torch
from PIL import Image
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

prompt = os.environ["PROMPT"]
negative_prompt = os.environ.get("NEGATIVE_PROMPT", "")
input_image_path = os.environ.get("INPUT_IMAGE", "")
weight_path = os.environ.get("WEIGHT_PATH", "")
output_dir = os.environ["OUTPUT_DIR"]
output_name = os.environ["OUTPUT_NAME"]
num_frames = int(os.environ["NUM_FRAMES"])
seed = int(os.environ.get("SEED", "42"))
tiled = os.environ.get("TILED", "1") == "1"
fps = int(os.environ.get("FPS", "15"))
quality = int(os.environ.get("QUALITY", "5"))
device = os.environ.get("DEVICE", "cuda")
model_path = os.environ.get("WAN_MODEL_PATH", "")
max_area = int(os.environ.get("MAX_AREA", "399360"))  # 480*832
num_inference_steps = int(os.environ.get("NUM_INFERENCE_STEPS", "15"))
cfg_scale = float(os.environ.get("CFG_SCALE", "1"))

if not model_path:
    wan_model_dir = os.environ.get("WAN_MODEL_DIR", "../../model")
    model_path = os.path.join(wan_model_dir, "Wan2.2-TI2V-5B")
if not model_path.endswith("/"):
    model_path += "/"

# --- load pipeline ---
t0 = time.time()
print(f"🚀 loading Wan2.2-TI2V-5B from {model_path} ...")

dit_files = sorted(glob.glob(model_path + "diffusion_pytorch_model*.safetensors"))
if not dit_files:
    sys.exit(f"❌ no diffusion_pytorch_model*.safetensors found in {model_path}")
print(f"   DiT files: {dit_files}")

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device=device,
    model_configs=[
        ModelConfig(path=dit_files, offload_device=device),
        ModelConfig(path=model_path + "models_t5_umt5-xxl-enc-bf16.pth", offload_device=device),
        ModelConfig(path=model_path + "Wan2.2_VAE.pth", offload_device=device),
    ],
    tokenizer_config=ModelConfig(path=model_path + "google/umt5-xxl", offload_device=device),
)
print(f"✅ pipeline loaded in {time.time()-t0:.1f}s")

# --- load LoRA ---
if weight_path:
    print(f"🏋️ loading LoRA: {weight_path} (alpha=1)")
    pipe.load_lora(pipe.dit, weight_path, alpha=1)
    print("✅ LoRA loaded")

# --- prepare input image + auto height/width ---
input_image = None
if input_image_path:
    print(f"🖼️ I2V mode: {input_image_path}")
    input_image = Image.open(input_image_path)
    aspect_ratio = input_image.height / input_image.width
    mod_value = 8 * 2  # vae_scale_factor_spatial * patch_size
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    input_image = input_image.resize((width, height))
    print(f"📐 aspect={aspect_ratio:.3f} -> {width}x{height} (max_area={max_area})")
else:
    print("📝 T2V mode")
    height = int(os.environ.get("HEIGHT", "480"))
    width = int(os.environ.get("WIDTH", "832"))

# --- generate ---
t1 = time.time()
pipe_kwargs = dict(
    prompt=prompt,
    negative_prompt=negative_prompt,
    seed=seed,
    tiled=tiled,
    height=height,
    width=width,
    num_frames=num_frames,
    num_inference_steps=num_inference_steps,
    cfg_scale=cfg_scale,
)
if input_image is not None:
    pipe_kwargs["input_image"] = input_image

video = pipe(**pipe_kwargs)
gen_time = time.time() - t1
print(f"🎬 generation done in {gen_time:.1f}s ({num_frames} frames, {gen_time/num_frames:.2f}s/frame)")

# --- save ---
out_path = os.path.join(output_dir, f"{output_name}.mp4")
save_video(video, out_path, fps=fps, quality=quality)
print(f"💾 saved: {out_path}")
