#!/usr/bin/env python3
"""09g — 真权重小画布探针：加载一次，在同进程内做子系统二分。

为什么（2026-09-29）：
  已排除：int8（06b bf16 也坏）、LoRA（06c/09b 无 LoRA 也坏）、VAE（09a roundtrip 干净）、
  offload 机制（tiny 四配方 bit-exact）、配置（26 json 与上游同字节）、text encoder
  （09f：确定性 0 误差、massive 通道同 3 个、剔除后 prompt 可分）。
  diffusers 移植在 M5 Max 上跑通过真权重（issue #14639）→ 嫌疑 = 本机环境
  （torch 2.6 内核 / sdpa 长序列）或真机专属路径。
  tiny fixture 验证不了真机数值；09b 一次要 40 分钟 → 太慢，无法迭代。
  本脚本把画布压到 64×64（~150 视频 token + ~600 音频 token），50 步去噪在
  block offload 下每步 ~2-4s（搬运主导，与 token 数无关），一轮 ~3 分钟。
  加载一次后同进程跑多组配置，把「每试一次 40min」变成「每试一次 3min」。

判据（09d 同款，印在每组后面）：
  · 相邻帧余弦相关：自然视频 latent >0.9；09b 的坏 latent = 0.038
  · z 整体 std vs RMS(latents_std)=1.925
  · 逐层打印每组配置，方便横向对比

用法：
  MODEL_PATH=... python minimax_h3/09g_realscale_probe.py
  只跑一组：ARMS=sdpa NIS=51 W=64 H=64 python ...
Env vars:
  MODEL_PATH, DEVICE, OUTPUT_DIR, ARMS（逗号分隔，默认 sdpa,eager）,
  W, H, NUM_FRAMES, NIS, SEED, VIDEO_SHIFT, PROMPT
"""
import os
import sys
import time

import torch

MODEL_PATH = os.environ.get("MODEL_PATH", "/mnt/d/wheel/minimaxh3_ms")
DEVICE = os.environ.get("DEVICE", "cuda:0")
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR", "/mnt/d/output/minimaxh3_rotate_results/diag_probe")
W = int(os.environ.get("W", "64"))
H = int(os.environ.get("H", "64"))
NUM_FRAMES = int(os.environ.get("NUM_FRAMES", "124"))
NIS = int(os.environ.get("NIS", "51"))          # 50 步去噪（grid 含端点 0）
SEED = int(os.environ.get("SEED", "42"))
VIDEO_SHIFT = float(os.environ.get("VIDEO_SHIFT", "12.0"))   # 原版基座配方（非 Turbo）
ARMS = os.environ.get("ARMS", "sdpa,eager").split(",")
PROMPT = os.environ.get(
    "PROMPT", "A red fox trotting through a snowy pine forest, "
              "snow crunching underfoot")

NOT_CONVERT = [
    "proj_in", "audio_proj_in", "context_embedder", "time_embedder", "time_proj",
    "token_refiner", "norm_out", "proj_out", "audio_proj_out",
]
TE_NOT_CONVERT = [
    "model.visual", "model.language_model.embed_tokens",
    "model.language_model.norm", "lm_head",
]


def latent_metrics(z):
    """09d 同款核心判据。z: (1, C, T, H, W) raw latent。"""
    zf = z.detach().float()
    T = zf.shape[2]
    per_frame = zf[0].permute(1, 0, 2, 3).flatten(1)     # (T, C*H*W)
    if T > 1:
        a = per_frame[:-1]
        b = per_frame[1:]
        corr = torch.nn.functional.cosine_similarity(a, b, dim=1).mean().item()
    else:
        corr = float("nan")
    return zf.std().item(), corr


def main():
    from diffusers import (
        MiniMaxH3Transformer3DModel, ModularPipeline, TorchAoConfig)
    from diffusers.hooks import apply_group_offloading
    from torchao.quantization import Int8WeightOnlyConfig
    from transformers import Qwen3VLForConditionalGeneration
    from transformers import TorchAoConfig as TransformersTorchAoConfig

    print(f"{'=' * 70}")
    print("🎯 09g — 真权重小画布探针")
    print(f"  canvas={W}x{H} frames={NUM_FRAMES} NIS={NIS}({NIS - 1} 步) "
          f"shift={VIDEO_SHIFT} seed={SEED}")
    print(f"  arms={ARMS}")
    print(f"{'=' * 70}")

    from _ensure_modular_index import ensure_modular_model_index
    print(f"  📦 {ensure_modular_model_index(MODEL_PATH)}")

    print("\n📦 loading (int8 + block offload, 约 20-40 min，只加载这一次)...")
    t0 = time.time()
    pipe = ModularPipeline.from_pretrained(MODEL_PATH)
    pipe.update_components(
        transformer=MiniMaxH3Transformer3DModel.from_pretrained(
            MODEL_PATH, subfolder="transformer", dtype=torch.bfloat16,
            quantization_config=TorchAoConfig(
                Int8WeightOnlyConfig(version=2),
                modules_to_not_convert=NOT_CONVERT,
            ),
            low_cpu_mem_usage=True,
        ),
        text_encoder=Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, subfolder="text_encoder", dtype=torch.bfloat16,
            quantization_config=TransformersTorchAoConfig(
                Int8WeightOnlyConfig(version=2),
                modules_to_not_convert=TE_NOT_CONVERT,
            ),
            low_cpu_mem_usage=True,
        ),
    )
    pipe.load_components(workflow="t2va", dtype=torch.bfloat16,
                         pretrained_model_name_or_path=MODEL_PATH)
    pipe.transformer.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    offload = dict(onload_device=torch.device(DEVICE),
                   offload_device=torch.device("cpu"),
                   low_cpu_mem_usage=True)
    pipe.transformer.enable_group_offload(
        offload_type="block_level", num_blocks_per_group=1, **offload)
    apply_group_offloading(
        pipe.text_encoder.model, offload_type="leaf_level", **offload)
    pipe.vae.enable_group_offload(onload_device=torch.device(DEVICE),
                                  offload_device=torch.device("cpu"),
                                  offload_type="leaf_level",
                                  low_cpu_mem_usage=True)
    pipe.audio_vae.to(DEVICE)
    print(f"  ✅ loaded in {(time.time() - t0) / 60:.1f} min")

    # sigma grid 核对：打印实际用的前几个 sigma
    pipe.scheduler.set_shift(VIDEO_SHIFT)
    pipe.audio_scheduler.set_shift(3.0)

    # 挂 decode 钩子收集每组 latent
    captured = {}
    orig_decode = pipe.vae.decode

    def logged_decode(z, *a, **kw):
        captured["z"] = z.detach().to("cpu", torch.float32)
        return orig_decode(z, *a, **kw)

    pipe.vae.decode = logged_decode

    rms_std = torch.tensor(pipe.vae.config.latents_std).pow(2).mean().sqrt().item()

    for arm in ARMS:
        arm = arm.strip()
        if not arm:
            continue
        if arm != "default":
            try:
                pipe.transformer.set_attention_backend(arm)
                print(f"\n🧪 arm={arm}（set_attention_backend 成功）")
            except Exception as e:
                print(f"\n⚠️ arm={arm} set_attention_backend 失败: "
                      f"{type(e).__name__}: {e} —— 跳过")
                continue
        gen = torch.Generator("cpu").manual_seed(SEED)
        captured.clear()
        t1 = time.time()
        # 2026-10-08：inference_mode 在本机 torch2.6+torchao0.16 下会触发
        # int8 叶子 offload 回搬时的 storage set_ 设备冲突（09b 无上下文、
        # 09f 用 no_grad 均正常）——改用 no_grad。
        with torch.no_grad():
            results = pipe(prompt=PROMPT, num_frames=NUM_FRAMES, generator=gen,
                           num_inference_steps=NIS, width=W, height=H,
                           output=["videos", "audio", "sampling_rate"])
        dt = time.time() - t1
        z = captured.get("z")
        if z is None:
            print(f"  ❌ arm={arm}: decode 钩子没抓到 latent")
            continue
        zstd, corr = latent_metrics(z)
        px = results["videos"][0]
        print(f"  ── arm={arm}  去噪+解码 {dt:.0f}s ──")
        print(f"     z shape={tuple(z.shape)} std={zstd:.3f} "
              f"(基准 RMS={rms_std:.3f}) 相邻帧相关={corr:.4f}")
        try:
            import numpy as np
            arr = np.asarray(px[0]) if not isinstance(px, np.ndarray) else px[0]
            print(f"     首帧像素 mean={arr.mean() / 255:.3f} std={arr.std() / 255:.3f}")
        except Exception as e:
            print(f"     (像素统计跳过: {e})")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        torch.save({"z": z, "arm": arm, "canvas": (W, H), "nis": NIS,
                    "seed": SEED, "shift": VIDEO_SHIFT},
                   os.path.join(OUTPUT_DIR, f"probe_{arm}_{W}x{H}.pt"))
        verdict = "✅ 有结构（>0.9）" if corr > 0.9 else (
            "⚠️ 弱结构" if corr > 0.5 else "❌ 零结构噪声")
        print(f"     判定: {verdict}")

    print(f"\n{'=' * 70}")
    print("👉 判读：")
    print("  · 64×64 也零结构（<0.1）→ 真机 transformer 侧坏，与分辨率/tiling 无关；")
    print("    对比各 arm：若 eager 正常而 sdpa 坏 → sdpa 内核实锤")
    print("  · 64×64 有结构（>0.9）→ 系统在小画布健康，问题与 token 数/分辨率相关")
    print("    （长序列 attention / VAE tiling），下一步放大画布找临界点")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
