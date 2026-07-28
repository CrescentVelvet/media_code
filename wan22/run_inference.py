import os, time, torch
from PIL import Image
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

# --- read params from env (set by 02_run_inference.sh) ---
prompt = os.environ["PROMPT"]
negative_prompt = os.environ["NEGATIVE_PROMPT"]
input_image_path = os.environ.get("INPUT_IMAGE", "")
weight_path = os.environ.get("WEIGHT_PATH", "")
output_dir = os.environ["OUTPUT_DIR"]
output_name = os.environ["OUTPUT_NAME"]
height = int(os.environ["HEIGHT"])
width = int(os.environ["WIDTH"])
num_frames = int(os.environ["NUM_FRAMES"])
seed = int(os.environ["SEED"])
tiled = os.environ["TILED"] == "1"
fps = int(os.environ["FPS"])
quality = int(os.environ["QUALITY"])
low_vram = os.environ["LOW_VRAM"] == "1"
vram_limit_env = os.environ.get("VRAM_LIMIT", "")

# --- build model configs ---
# DIFFSYNTH_MODEL_BASE_PATH + DIFFSYNTH_SKIP_DOWNLOAD=True are set in _env.sh,
# so ModelConfig(model_id=..., origin_file_pattern=...) loads from the local
# store without downloading. This keeps the code identical to the official
# example (examples/wanvideo/model_inference/Wan2.2-TI2V-5B.py).
vram_config = {}
if low_vram:
    vram_config = {
        "offload_dtype": "disk",
        "offload_device": "disk",
        "onload_dtype": torch.bfloat16,
        "onload_device": "cpu",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    }

model_configs = [
    ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", **vram_config),
    ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors", **vram_config),
    ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth", **vram_config),
]
tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/")

kwargs = dict(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=model_configs,
    tokenizer_config=tokenizer_config,
)
if low_vram:
    if vram_limit_env:
        vram_limit = float(vram_limit_env)
    else:
        vram_limit = torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2
    kwargs["vram_limit"] = vram_limit
    print(f"[*] low-VRAM mode: vram_limit={vram_limit:.1f} GB, disk offload enabled")

# --- load pipeline ---
t0 = time.time()
print("[*] loading Wan2.2-TI2V-5B pipeline ...")
pipe = WanVideoPipeline.from_pretrained(**kwargs)
print(f"[*] pipeline loaded in {time.time()-t0:.1f}s")

# --- load LoRA if provided ---
if weight_path:
    from diffsynth.core import load_state_dict
    print(f"[*] loading LoRA: {weight_path} (alpha=1)")
    pipe.load_lora(pipe.dit, weight_path, alpha=1)
    print("[*] LoRA loaded")

# --- generate ---
input_image = None
if input_image_path:
    print(f"[*] I2V mode: input image = {input_image_path}")
    input_image = Image.open(input_image_path).resize((width, height))
else:
    print("[*] T2V mode")

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
print(f"[*] generation done in {gen_time:.1f}s ({num_frames} frames, {gen_time/num_frames:.2f}s/frame)")

# --- save ---
out_path = os.path.join(output_dir, f"{output_name}.mp4")
save_video(video, out_path, fps=fps, quality=quality)
print(f"[*] saved: {out_path}")
