#!/usr/bin/env python3
"""10_orbit_datagen.py — 人物静止 + 相机 360° 环绕视频的批量数据生成器。

为什么（2026-10-08）：vggt_human 的输入是实拍环绕视频，但实拍里人物总会动
（呼吸/重心晃动/走步），拖垮重建。本脚本用 MiniMax-H3 的 FL2VA 合成
「人物绝对静止、相机绕一圈」的干净环绕视频，作为 vggt_human 的输入数据。

核心设计：
1. FL2VA 首尾帧同图（loop closure）：first_frame = last_frame = 输入图。
   尾帧约束强制相机在结尾回到起始视角 —— 正好闭合 360° 环绕，
   且 fl2v_turbo LoRA 的原生任务就是首尾帧插值。
2. Prompt 双锚定（见 ORBIT_PROMPT）：
   - 人物侧：枚举式否定一切微动（no swaying/blinking/hair movement/...），
     因为 H3 是 guidance-distilled、没有 negative prompt，约束必须写进正向。
   - 相机侧：把运镜描述成「环形轨道 dolly」（constant angular speed /
     fixed height / no pitch/roll/zoom），抑制模型自由发挥。
   结构照官方 examples/fl2va_prompt.txt：integrated_multimodal_description
   [Shot N] + overall_soundscape + non_diegetic_music。
3. 批量 + QC：每张图 × VARIANTS 个 seed 各出一条；每条算两个量化指标
   写进 manifest.jsonl，方便挑最稳的一条进 vggt_human：
   - motion_index：中心裁剪区逐帧灰度差均值（人越不动越接近 0）
   - loop_err：首帧 vs 尾帧灰度差（环绕闭合误差，越小越好）

多图输入说明：本地权重没有 transformer_ref（Ref2VA 多参考组件未下载），
所以多图 = 每张图各出一条视频（数据工厂语义），不是多参考合成。
要真 Ref2VA 需另下 transformer_ref 权重（见 README）。

速度档（LORA_PATH 控制）：
  默认 turbo 4-step（768p 配方，~5min/条，加载 20min 摊薄后适合批量）；
  LORA_PATH="" 回退基座 50-step（shift=12，画质上限更高，~70min/条）。

用法（WSL，minimax_h3 env）：
  cd /mnt/c/code/media_code/minimax_h3
  IMAGES=/path/person_a.jpg VARIANTS=2 \
    /home/velvet/miniconda3/envs/minimax_h3/bin/python -u 10_orbit_datagen.py
  # 或整个目录批量：
  IMAGE_DIR=/path/people VARIANTS=1 ... python -u 10_orbit_datagen.py
  # 先不加载模型、只打印生成计划（检查路径/分辨率/prompt）：
  DRY_RUN=1 IMAGES=... python -u 10_orbit_datagen.py

Env vars:
  MODEL_PATH, DEVICE, OUTPUT_DIR, MAX_PIXELS, FPS, NUM_FRAMES(17n+5),
  IMAGES(逗号分隔) / IMAGE_DIR, SEED, VARIANTS,
  LORA_PATH(空=基座50步), NUM_INFERENCE_STEPS, VIDEO_SHIFT, AUDIO_SHIFT,
  LORA_ALPHA, PROMPT(整段覆盖) / PROMPT_FILE, LOOP_CLOSE(默认1),
  DUMP_FRAMES(默认1), DRY_RUN
"""
import json
import math
import os
import sys
import time

# ── 1. 环境变量 ──────────────────────────────────────────────────────────
MODEL_PATH = os.path.abspath(os.environ.get("MODEL_PATH", "/mnt/d/wheel/minimaxh3_ms"))
DEVICE = os.environ.get("DEVICE", "cuda:0")
OUTPUT_DIR = os.path.abspath(os.environ.get(
    "OUTPUT_DIR", "/mnt/d/output/minimaxh3_rotate_results/orbit_datagen"))
MAX_PIXELS = int(os.environ.get("MAX_PIXELS", str(1344 * 768)))  # 768p 预算
FPS = int(os.environ.get("FPS", "24"))
NUM_FRAMES = int(os.environ.get("NUM_FRAMES", "124"))            # 5.2s @24fps
SEED = int(os.environ.get("SEED", "42"))
VARIANTS = int(os.environ.get("VARIANTS", "1"))

LORA_PATH = os.environ.get("LORA_PATH", os.path.join(
    MODEL_PATH, "minimax_h3_turbo",
    "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"))
# turbo 4-step 768p 配方：NFE=4 shift=6；LORA_PATH="" → 基座 50 步 shift=12
if LORA_PATH:
    NFE = int(os.environ.get("NUM_INFERENCE_STEPS", "4"))
    VIDEO_SHIFT = float(os.environ.get("VIDEO_SHIFT", "6.0"))
else:
    NFE = int(os.environ.get("NUM_INFERENCE_STEPS", "50"))
    VIDEO_SHIFT = float(os.environ.get("VIDEO_SHIFT", "12.0"))
AUDIO_SHIFT = float(os.environ.get("AUDIO_SHIFT", "3.0"))
LORA_ALPHA = os.environ.get("LORA_ALPHA", "auto").strip() or "auto"

IMAGES = [s for s in os.environ.get("IMAGES", "").split(",") if s.strip()]
IMAGE_DIR = os.environ.get("IMAGE_DIR", "")
LOOP_CLOSE = os.environ.get("LOOP_CLOSE", "1") == "1"
DUMP_FRAMES = os.environ.get("DUMP_FRAMES", "1") == "1"
DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"

# ── 2. 环绕 prompt（设计说明见文件头第 2 点）─────────────────────────────
ORBIT_PROMPT = (
    "integrated_multimodal_description: [Shot 1] Photorealistic full-body "
    "studio shot. The person from the first frame stands perfectly centered "
    "in frame, frozen like a statue in their exact original pose — arms, "
    "hands, legs, head and face remain in the identical position for the "
    "entire clip: no walking, no stepping, no swaying, no weight shifting, "
    "no gesturing, no head turning, no nodding, no blinking, no hair or "
    "clothing movement, no change of expression. The person is completely "
    "motionless; only the camera moves. The camera is mounted on a precision "
    "circular dolly track at a fixed height and a fixed distance from the "
    "person, and glides at a slow, constant angular speed through one "
    "smooth, perfectly level 360-degree orbit: no vertical bobbing, no "
    "pitch, no tilt, no roll, no zoom, no focus breathing, no shake. As "
    "the camera travels around the motionless person, the front, "
    "three-quarter, profile, back, and opposite three-quarter views are "
    "revealed in one continuous unbroken take, while the background slides "
    "past in steady lateral parallax behind them. The setting is a clean "
    "photography studio: seamless light-gray backdrop, a few static pieces "
    "of studio equipment far in the background for spatial reference, soft "
    "even diffuse lighting from large softboxes, gentle contact shadow "
    "under the feet, no flicker, no moving shadows. [Shot 2] The camera "
    "completes exactly one full revolution and settles precisely back at "
    "the starting front view, ending on the identical framing and the "
    "identical frozen pose as the first frame.\n"
    "overall_soundscape: A quiet, constant studio room tone. No footsteps, "
    "no cloth rustling, no movement sounds of any kind.\n"
    "non_diegetic_music: None. Near silence, matching the completely "
    "static subject."
)
PROMPT = os.environ.get("PROMPT", "").strip()
PROMPT_FILE = os.environ.get("PROMPT_FILE", "")
if not PROMPT and PROMPT_FILE:
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        PROMPT = f.read().strip()
if not PROMPT:
    PROMPT = ORBIT_PROMPT


# ── 3. 工具函数 ──────────────────────────────────────────────────────────
def collect_images():
    """汇总输入图：IMAGES 列表优先，否则 IMAGE_DIR 扫目录。"""
    imgs = list(IMAGES)
    if not imgs and IMAGE_DIR:
        exts = (".jpg", ".jpeg", ".png", ".webp")
        imgs = sorted(os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)
                      if f.lower().endswith(exts))
    imgs = [os.path.abspath(os.path.expanduser(p)) for p in imgs]
    bad = [p for p in imgs if not os.path.isfile(p)]
    if bad:
        sys.exit("❌ 图片不存在: " + ", ".join(bad))
    if not imgs:
        sys.exit("❌ 没给输入图：IMAGES=a.jpg,b.jpg 或 IMAGE_DIR=目录")
    return imgs


def auto_resolution(img_path, max_pixels):
    """按输入图宽高比在像素预算内取整到 32 的倍数（同 06d）。"""
    from diffusers.utils import load_image
    img = load_image(img_path)
    iw, ih = img.size
    ratio = iw / ih
    if ratio >= 1:
        h = int(math.sqrt(max_pixels / ratio))
        w = int(h * ratio)
    else:
        w = int(math.sqrt(max_pixels * ratio))
        h = int(w / ratio)
    w = max(32, w - (w % 32))
    h = max(32, h - (h % 32))
    return w, h


def valid_frames(n):
    """H3 帧数规则 17n+5（120-360）：不合法就向上取最近合法值。"""
    if (n - 5) % 17 == 0 and 120 <= n <= 360:
        return n
    k = max(7, math.ceil((n - 5) / 17))
    fixed = 17 * k + 5
    print(f"  ⚠️ NUM_FRAMES={n} 不满足 17n+5，调整为 {fixed}")
    return fixed


def qc_metrics(frames_np):
    """静止度 + 环绕闭合误差。frames_np: (T,H,W,3) uint8。"""
    import numpy as np
    g = frames_np.astype(np.float32).mean(axis=3)          # (T,H,W)
    w = g.shape[2]
    center = g[:, :, w // 4: 3 * w // 4]                    # 人物居中假设
    motion = float(np.abs(np.diff(center, axis=0)).mean())
    loop = float(np.abs(g[-1] - g[0]).mean())
    return motion, loop


# ── 4. 加载（严格照 06d：int8 + turbo LoRA + block offload）───────────────
def load_model():
    from diffusers import ModularPipeline, MiniMaxH3Transformer3DModel, TorchAoConfig
    from diffusers.hooks import apply_group_offloading
    from torchao.quantization import Int8WeightOnlyConfig
    from transformers import Qwen3VLForConditionalGeneration
    from transformers import TorchAoConfig as TransformersTorchAoConfig
    import torch

    from _ensure_modular_index import ensure_modular_model_index
    print(f"  📦 {ensure_modular_model_index(MODEL_PATH)}")

    t0 = time.time()
    p = ModularPipeline.from_pretrained(MODEL_PATH)
    p.update_components(
        transformer=MiniMaxH3Transformer3DModel.from_pretrained(
            MODEL_PATH, subfolder="transformer", dtype=torch.bfloat16,
            quantization_config=TorchAoConfig(
                Int8WeightOnlyConfig(version=2),
                modules_to_not_convert=[
                    "proj_in", "audio_proj_in", "context_embedder",
                    "time_embedder", "time_proj", "token_refiner",
                    "norm_out", "proj_out", "audio_proj_out",
                ],
            ),
            low_cpu_mem_usage=True,
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
            low_cpu_mem_usage=True,
        ),
    )
    p.load_components(workflow="fl2va", dtype=torch.bfloat16,
                      pretrained_model_name_or_path=MODEL_PATH)
    p.transformer.requires_grad_(False)
    p.text_encoder.requires_grad_(False)

    p.scheduler.set_shift(VIDEO_SHIFT)
    p.audio_scheduler.set_shift(AUDIO_SHIFT)

    if LORA_PATH:
        # 必须在 enable_group_offload 之前注入（06d 注释：否则 LoRA 子模块
        # 不被 offload hook 覆盖 → 设备不匹配）
        from _turbo_lora import load_lora_adapter
        load_lora_adapter(p.transformer, LORA_PATH, LORA_ALPHA, 1.0, False)

    offload = dict(onload_device=torch.device(DEVICE),
                   offload_device=torch.device("cpu"))
    p.transformer.enable_group_offload(
        offload_type="block_level", num_blocks_per_group=1,
        low_cpu_mem_usage=True, **offload)
    apply_group_offloading(
        p.text_encoder.model, offload_type="leaf_level",
        low_cpu_mem_usage=True, **offload)
    p.vae.to(DEVICE)
    p.audio_vae.to(DEVICE)
    print(f"  ✅ loaded in {(time.time() - t0) / 60:.1f} min "
          f"(int8{'+turbo' if LORA_PATH else ' base'} NFE={NFE} shift={VIDEO_SHIFT})")
    return p


# ── 5. 主流程 ────────────────────────────────────────────────────────────
def main():
    import numpy as np
    import torch
    from diffusers.utils import load_image
    from diffusers.utils.export_utils import encode_video

    images = collect_images()
    frames_n = valid_frames(NUM_FRAMES)
    jobs = [(img, SEED + i) for img in images for i in range(VARIANTS)]

    print("=" * 70)
    print("🛰️  10_orbit_datagen — 静止人物 360° 环绕视频批量生成")
    print(f"  图片 {len(images)} 张 × VARIANTS={VARIANTS} = {len(jobs)} 条")
    print(f"  frames={frames_n} fps={FPS} NFE={NFE} shift={VIDEO_SHIFT} "
          f"lora={'turbo' if LORA_PATH else '无(基座)'}")
    print(f"  loop_close={'ON(首尾同图)' if LOOP_CLOSE else 'OFF'} "
          f"dump_frames={'ON' if DUMP_FRAMES else 'OFF'}")
    print(f"  output: {OUTPUT_DIR}")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    prompt_path = os.path.join(OUTPUT_DIR, "prompt_used.txt")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(PROMPT)
    print(f"  📝 prompt 已存: {prompt_path}")

    # 生成计划（含分辨率），DRY_RUN 到此为止
    plan = []
    for img, seed in jobs:
        w, h = auto_resolution(img, MAX_PIXELS)
        plan.append((img, seed, w, h))
        print(f"  🎯 {os.path.basename(img)}  seed={seed}  res={w}x{h}")
    if DRY_RUN:
        print("\n🏁 DRY_RUN=1，不加载模型，计划如上。")
        return

    pipe = load_model()
    manifest_path = os.path.join(OUTPUT_DIR, "manifest.jsonl")

    for idx, (img, seed, w, h) in enumerate(plan, 1):
        stem = os.path.splitext(os.path.basename(img))[0]
        name = f"{stem}_orbit_seed{seed}"
        out_mp4 = os.path.join(OUTPUT_DIR, name + ".mp4")
        print(f"\n[{idx}/{len(plan)}] 🎬 {name} ({w}x{h}, {frames_n}f, NFE={NFE})")

        gen = torch.Generator(DEVICE).manual_seed(seed)
        kwargs = dict(
            prompt=PROMPT, num_frames=frames_n, generator=gen,
            num_inference_steps=NFE + 1,   # grid 含末端零点 → N+1（06d 注释）
            width=w, height=h,
            image=load_image(img),
            output=["videos", "audio", "sampling_rate"],
        )
        if LOOP_CLOSE:
            kwargs["last_image"] = load_image(img)

        t0 = time.time()
        # 用 no_grad：inference_mode + int8 offload 回搬会触发 storage 冲突
        # （torch2.6 + torchao0.16，09g/06d 同款教训）
        with torch.no_grad():
            results = pipe(**kwargs)
        dt = time.time() - t0

        encode_video(results["videos"][0], fps=FPS, output_path=out_mp4,
                     audio=results["audio"][0],
                     audio_sample_rate=results["sampling_rate"])

        vid = results["videos"][0]
        arr = (vid if isinstance(vid, np.ndarray)
               else np.stack([np.asarray(fr) for fr in vid]))
        motion, loop = qc_metrics(arr)

        frames_dir = ""
        if DUMP_FRAMES:
            from PIL import Image
            frames_dir = os.path.join(OUTPUT_DIR, name)
            os.makedirs(frames_dir, exist_ok=True)
            for t in range(arr.shape[0]):
                Image.fromarray(arr[t]).save(
                    os.path.join(frames_dir, f"{t:05d}.jpg"), quality=95)

        rec = {
            "image": img, "seed": seed, "mp4": out_mp4,
            "frames_dir": frames_dir, "width": w, "height": h,
            "num_frames": frames_n, "nfe": NFE, "video_shift": VIDEO_SHIFT,
            "lora": LORA_PATH or None, "loop_close": LOOP_CLOSE,
            "motion_index": round(motion, 3), "loop_err": round(loop, 3),
            "time_s": round(dt, 1),
        }
        with open(manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  ✅ {out_mp4} ({os.path.getsize(out_mp4) / 1e6:.1f} MB, {dt:.0f}s)")
        print(f"  📊 motion_index={motion:.2f} loop_err={loop:.2f} "
              f"(越小越静止/越闭合) → manifest.jsonl")

    print(f"\n🏁 全部完成：{len(plan)} 条 → {OUTPUT_DIR}")
    print(f"   manifest: {manifest_path}")
    print("   挑数建议：按 motion_index 升序挑最稳的进 vggt_human")


if __name__ == "__main__":
    main()
