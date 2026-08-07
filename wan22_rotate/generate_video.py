#!/usr/bin/env python3
"""Generate a 360-degree rotation video with Wan2.2-TI2V-5B + LoRA.

Uses ModelConfig(path=...) to load model files directly, avoiding the
model_id / DIFFSYNTH_MODEL_BASE_PATH / Wan-AI symlink lookup.

Env vars (set by 02_generate_video.sh):
  WAN_MODEL_PATH, WEIGHT_PATH, PROMPT, NEGATIVE_PROMPT, INPUT_IMAGE,
  OUTPUT_DIR, OUTPUT_NAME, HEIGHT, WIDTH, NUM_FRAMES, SEED, TILED,
  FPS, QUALITY, DEVICE
"""
import os
import sys
import time
import glob

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
height = int(os.environ["HEIGHT"])
width = int(os.environ["WIDTH"])
num_frames = int(os.environ["NUM_FRAMES"])
seed = int(os.environ.get("SEED", "0"))
tiled = os.environ.get("TILED", "1") == "1"
fps = int(os.environ.get("FPS", "15"))
quality = int(os.environ.get("QUALITY", "5"))
device = os.environ.get("DEVICE", "cuda")
model_path = os.environ.get("WAN_MODEL_PATH", "")

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

# --- generate ---
input_image = None
if input_image_path:
    print(f"🖼️ I2V mode: {input_image_path}")
    input_image = Image.open(input_image_path).resize((width, height))
else:
    print("📝 T2V mode")

t1 = time.time()
pipe_kwargs = dict(
    prompt=prompt,
    negative_prompt=negative_prompt,
    seed=seed,
    tiled=tiled,
    height=height,
    width=width,
    num_frames=num_frames,
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
