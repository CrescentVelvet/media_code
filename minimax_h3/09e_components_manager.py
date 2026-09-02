#!/usr/bin/env python3
"""09e — 官方 ComponentsManager.auto_cpu_offload 配方（与 09b 对照实验）。

起因（2026-09-02）：09b 跑完 50/50 步，出片是纯噪声场（latent 帧间相关
0.04、空间剖面完全平坦，100% transformer 零结构生成）。回顾 09b 用的内存
管理路径：

    pipe = ModularPipeline.from_pretrained(MODEL_PATH)            # 无 manager
    pipe.transformer.enable_group_offload(...)                    # 手动
    apply_group_offloading(pipe.text_encoder.model, ...)          # 手动
    pipe.vae.enable_group_offload(...)                            # 手动

而 diffusers 0.40.0 官方 MiniMax-H3 doc 明确说：
    "Pair it with a ComponentsManager and auto offloading"
（见 huggingface.co/docs/diffusers/api/pipelines/minimax_h3 Memory 一节）

查源码验证：
  · ModularPipeline.from_pretrained 默认 components_manager=None
  · 组件只有显式注册到 ComponentsManager 才能被 enable_auto_cpu_offload 管到
  · modular pipeline 的 block 调度器依赖 ComponentsManager 维护"激活组件"
    状态；手动 offload 不参与这个调度，可能在 packed sequence 路由的
    某个 block 让模型读到"半激活"状态 → 静默生成零结构噪声

本脚本保持 09b 的全部量化配方（int8 + NOT_CONVERT），**只换 offload 路径**：
  · ComponentsManager()
  · ModularPipeline.from_pretrained(MODEL_PATH, components_manager=cm)
  · pipe.update_components(..., components_manager=cm)
  · pipe.load_components(workflow="t2va", components_manager=cm)
  · cm.enable_auto_cpu_offload(device="cuda:0")
  · transformer/text_encoder/VAE 的 group_offload 保留叠加（官方明确说
    "Either order works"，但要传 offload_strategy 给 AutoOffloadStrategy
    让它理解 group-offload 模型的真实显存占用）
  · audio_vae 显式搬到 cuda:0（保持 09b 一致）

预期：~45 分钟一次（加载 20-40min + 去噪 ~45min）。
若出片干净 → 09b 的 manual offload 路径与 ComponentsManager 状态机冲突，
            改回官方路径即可（06c/06d 也是同理）；
若仍马赛克 → 100% 锁定量化损失，下一步 09f：transformer bf16 对照。

Env vars: 与 09b 一致。
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 必须在 torch/CUDA 初始化之前设置（沿用 09b 默认）
import torch  # noqa: E402

MODEL_PATH = os.environ.get("MODEL_PATH", "/mnt/d/wheel/minimaxh3_ms")
DEVICE = os.environ.get("DEVICE", "cuda:0")
WIDTH = int(os.environ.get("WIDTH", "960"))
HEIGHT = int(os.environ.get("HEIGHT", "544"))
NUM_FRAMES = int(os.environ.get("NUM_FRAMES", "124"))
FPS = int(os.environ.get("FPS", "24"))
SEED = int(os.environ.get("SEED", "42"))
NIS = int(os.environ.get("NUM_INFERENCE_STEPS", "51"))
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR", "/mnt/d/output/minimaxh3_rotate_results/diag_baseline")
PROMPT = os.environ.get("PROMPT", "A red fox trotting through a snowy pine forest, "
                                  "snow crunching underfoot")
FIRST_FRAME = os.environ.get("FIRST_FRAME", "")

NOT_CONVERT = [
    "proj_in", "audio_proj_in", "context_embedder", "time_embedder", "time_proj",
    "token_refiner", "norm_out", "proj_out", "audio_proj_out",
]
TE_NOT_CONVERT = [
    "model.visual", "model.language_model.embed_tokens",
    "model.language_model.norm", "lm_head",
]


def stats(name, t):
    t = t.detach().float()
    finite = torch.isfinite(t)
    if int(finite.sum()) == 0:
        print(f"  {name:<28} 全部 NaN/Inf !!")
        return
    print(
        f"  {name:<28} shape={tuple(t.shape)} "
        f"mean={t[finite].mean():+.4f} std={t[finite].std():.4f} "
        f"min={t[finite].min():+.3f} max={t[finite].max():+.3f} "
        f"nan/inf={int((~finite).sum())}"
    )


def _host_mem_gb():
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                info[k] = int(v.split()[0]) / 1024 / 1024
        return (info["MemTotal"], info.get("MemAvailable", 0.0),
                info["SwapTotal"] - info.get("SwapFree", info["SwapTotal"]))
    except Exception:
        return None


def preflight(torch):
    print("\n🩺 pre-flight GPU 自检（约 5 秒）...")
    try:
        a = torch.randn(2048, 2048, device=DEVICE, dtype=torch.float16)
        b = torch.randn(2048, 2048, device=DEVICE, dtype=torch.float16)
        s = (a @ b).sum().item()
        torch.cuda.synchronize()
        del a, b
        free, total = torch.cuda.mem_get_info()
        print(f"  ✅ GPU 响应正常（matmul={s:.3e}）  "
              f"VRAM free={free / 2 ** 30:.1f}GB / total={total / 2 ** 30:.1f}GB")
        if free / 2 ** 30 < 20:
            print("  ⚠️  显存不足 20GB 空闲：可能有残留进程占着，先 `nvidia-smi` 看一眼")
    except Exception as e:
        print(f"\n  ❌ GPU 自检失败：{type(e).__name__}: {e}")
        print("  → 上次 OOM 后 WSL2 的 GPU 分区很可能没释放干净。")
        print("    Windows PowerShell 执行： wsl --shutdown")
        print("    等 10 秒重开终端，`nvidia-smi` 确认正常后再重跑。")
        raise SystemExit(1)

    m = _host_mem_gb()
    if m:
        tot, avail, swap_used = m
        print(f"  🖥️  主机内存：总 {tot:.1f}GB，可用 {avail:.1F}GB，"
              f"swap 已用 {swap_used:.1f}GB")


def main():
    import torch
    from diffusers import (
        MiniMaxH3Transformer3DModel,
        ModularPipeline,
        TorchAoConfig,
    )
    from diffusers.modular_pipelines import ComponentsManager
    from diffusers.hooks import apply_group_offloading
    from diffusers.utils.export_utils import encode_video
    from diffusers.modular_pipelines.components_manager import AutoOffloadStrategy
    from torchao.quantization import Int8WeightOnlyConfig
    from transformers import Qwen3VLForConditionalGeneration
    from transformers import TorchAoConfig as TransformersTorchAoConfig

    import diffusers
    print(f"{'=' * 68}")
    print("🎯 09e — 官方 ComponentsManager.auto_cpu_offload 配方（与 09b 对照）")
    print(f"  diffusers : {diffusers.__version__}  ({os.path.dirname(diffusers.__file__)})")
    print(f"  torch     : {torch.__version__}")
    print(f"  model     : {MODEL_PATH}")
    print(f"  device    : {DEVICE}  {torch.cuda.get_device_name(0)}")
    print(f"  video     : {NUM_FRAMES}f {WIDTH}x{HEIGHT} @ {FPS}fps")
    print(f"  steps     : {NIS} (={NIS - 1} 次去噪)  seed={SEED}")
    print(f"  唯一改动  : offload 走 ComponentsManager.auto_cpu_offload + group_offload 叠加")
    print(f"{'=' * 68}")

    preflight(torch)

    lcm_flag = True
    from _ensure_modular_index import ensure_modular_model_index
    print(f"\n  📦 {ensure_modular_model_index(MODEL_PATH)}")

    # ── 核心差异：创建 ComponentsManager 并注入 ModularPipeline ─────────
    print("\n🧩 创建 ComponentsManager（官方 doc 推荐路径）...")
    cm = ComponentsManager()

    print("\n📦 loading（int8 量化 + ComponentsManager 注册，约 20-40 min）...")
    t0 = time.time()
    pipe = ModularPipeline.from_pretrained(MODEL_PATH, components_manager=cm)
    pipe.update_components(
        transformer=MiniMaxH3Transformer3DModel.from_pretrained(
            MODEL_PATH, subfolder="transformer", dtype=torch.bfloat16,
            quantization_config=TorchAoConfig(
                Int8WeightOnlyConfig(version=2),
                modules_to_not_convert=NOT_CONVERT,
            ),
            low_cpu_mem_usage=lcm_flag,
        ),
        text_encoder=Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, subfolder="text_encoder", dtype=torch.bfloat16,
            quantization_config=TransformersTorchAoConfig(
                Int8WeightOnlyConfig(version=2),
                modules_to_not_convert=TE_NOT_CONVERT,
            ),
            low_cpu_mem_usage=True,
        ),
        components_manager=cm,
    )
    workflow = "fl2va" if FIRST_FRAME else "t2va"
    pipe.load_components(workflow=workflow, dtype=torch.bfloat16,
                         pretrained_model_name_or_path=MODEL_PATH,
                         components_manager=cm)
    pipe.transformer.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    # ── 1. 打开 ComponentsManager 自动 CPU offload（关键改动）────────
    # Group offload 在 enable_auto_cpu_offload 之前或之后都可以（"Either order
    # works"，见 components_manager.py:708 注释）。我们按"先 group_offload 再
    # auto_offload"的顺序：先告诉分组机制哪些层是 group offloaded 的，再让
    # AutoOffloadStrategy 看到一个完整的组件图。
    print("\n🧩 1) group offload（沿用 09b 的 block/leaf 配置）...")
    offload = dict(onload_device=torch.device(DEVICE),
                   offload_device=torch.device("cpu"),
                   low_cpu_mem_usage=True)
    try:
        pipe.transformer.enable_group_offload(
            offload_type="block_level", num_blocks_per_group=1, **offload)
        apply_group_offloading(
            pipe.text_encoder.model, offload_type="leaf_level", **offload)
    except TypeError as e:
        print(f"  ⚠️ use_stream 不被当前版本支持（{e}），回退到无流式 offload")
        offload.pop("use_stream", None)
        pipe.transformer.enable_group_offload(
            offload_type="block_level", num_blocks_per_group=1, **offload)
        apply_group_offloading(
            pipe.text_encoder.model, offload_type="leaf_level", **offload)
    pipe.vae.enable_group_offload(onload_device=torch.device(DEVICE),
                                  offload_device=torch.device("cpu"),
                                  offload_type="leaf_level",
                                  low_cpu_mem_usage=True)
    pipe.audio_vae.to(DEVICE)

    # ── 2. 打开 ComponentsManager auto_cpu_offload ────────────────────
    # group_offloaded 模型 的"显存占用"对 AutoOffloadStrategy 不可见
    # （它一次只持一组，不是整模），需要传一个 workflow-aware 的 strategy。
    # 这里用默认 AutoOffloadStrategy 但保留 3GB 边缘（保留 09b 的数值）。
    print("\n🧩 2) enable_auto_cpu_offload（ComponentsManager 官方推荐）...")
    cm.enable_auto_cpu_offload(device=DEVICE, memory_reserve_margin="3GB")

    # 确认组件已注册
    print(f"  ✅ ComponentsManager 已注册 {len(cm.components)} 个组件: "
          f"{list(cm.components.keys())}")
    print(f"  ✅ loaded in {(time.time() - t0) / 60:.1f} min")

    # ── latent 统计钩子（与 09b 完全一致，便于对照）──────────────────
    n = {"calls": 0}
    orig_decode = pipe.vae.decode

    def logged_decode(z, *a, **kw):
        n["calls"] += 1
        print(f"\n  🔍 [VAE decode #{n['calls']}]")
        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            lat_path = os.path.join(OUTPUT_DIR, f"latent_e{n['calls']}.pt")
            torch.save(
                {"z": z.detach().to("cpu", torch.float32),
                 "latents_mean": list(pipe.vae.config.latents_mean),
                 "latents_std": list(pipe.vae.config.latents_std),
                 "meta": {"width": WIDTH, "height": HEIGHT,
                          "num_frames": NUM_FRAMES, "fps": FPS,
                          "seed": SEED, "steps": NIS,
                          "via": "09e_components_manager"}},
                lat_path)
            print(f"  💾 latent 已存盘: {lat_path}")
        except Exception as e:
            print(f"  ⚠️ 存盘失败（不影响继续）: {type(e).__name__}: {e}")
        try:
            stats("z (raw, decode 输入)", z)
        except Exception as e:
            print(f"  ⚠️ 统计失败: {e}")
        out = orig_decode(z, *a, **kw)
        try:
            sample = out.sample if hasattr(out, "sample") else out[0]
            stats("decoded sample", sample)
        except Exception as e:
            print(f"  ⚠️ 输出统计失败: {e}")
        return out

    pipe.vae.decode = logged_decode

    # ── 生成 ────────────────────────────────────────────────────────
    gen = torch.Generator("cpu").manual_seed(SEED)  # 与 09b 完全相同
    kwargs = dict(prompt=PROMPT, num_frames=NUM_FRAMES, generator=gen,
                  num_inference_steps=NIS, width=WIDTH, height=HEIGHT,
                  output=["videos", "audio", "sampling_rate"])
    if FIRST_FRAME:
        from diffusers.utils import load_image
        kwargs["image"] = load_image(FIRST_FRAME)

    print(f"\n🎬 generating (workflow={workflow})...")
    t1 = time.time()
    results = pipe(**kwargs)
    print(f"  ⏱️  generation: {(time.time() - t1) / 60:.1f} min   "
          f"peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"components_e_{workflow}_seed{SEED}.mp4")
    encode_video(results["videos"][0], fps=FPS, output_path=out_path,
                 audio=results["audio"][0],
                 audio_sample_rate=results["sampling_rate"])
    print(f"  💾 {out_path}  ({os.path.getsize(out_path) / 1e6:.1f} MB)")

    print(f"\n{'=' * 68}")
    print("👉 判读：")
    print("  · components_e_*.mp4 不再是彩色噪声（结构、帧间连贯）")
    print("      → 09b 的 manual offload 与 ComponentsManager 状态机冲突是根因。")
    print("        修法：把 06c/06d 的加载改用 ComponentsManager + enable_auto_cpu_offload。")
    print("  · 仍是彩色噪声场（latent 帧间相关 ≈0、空间剖面平坦）")
    print("      → 100% 锁定量化损失。下一步 09f：transformer bf16 + offload 对照")
    print("        （66GB bf16 权重，WSL 56GB RAM + 128GB swap，leaf/block offload，" )
    print("        单卡 24GB，预计 1.5-3 小时一次）")
    print(f"{'=' * 68}")


if __name__ == "__main__":
    main()