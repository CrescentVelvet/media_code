#!/usr/bin/env python3
"""09b — 官方 24-32GB 消费卡配方 baseline（逐字照抄 + latent 统计钩子）。

这份脚本的价值在于「干净」：不加任何魔改，完全按 diffusers 官方
docs/source/en/api/pipelines/minimax_h3.md 的 consumer card (24-32GB) 配方加载。
它是唯一能一次切断疑点树的动作——如果这条 baseline 出片干净，说明模型、
pipeline、权重都没问题，锅全在你的 06c/06d 魔改上；如果它也是马赛克，
问题就在模型/pipeline 这一层，跟你的优化无关。

与官方 doc 配方的关系（实测勘误，2026-09-01）：
  · 官方 doc 写 low_cpu_mem_usage=False，但实测 diffusers 0.40.0（含 main）
    在 from_pretrained 里硬性断言「量化加载必须 low_cpu_mem_usage=True」，
    False 直接 ValueError。doc 与代码矛盾，以代码为准 → 三处全部 True。
    06c/06d 原本写 True 反而是对的，此前对它的怀疑撤回。
  · offload kwargs      use_stream：官方 doc 推荐 ON，但本脚本默认 OFF，原因见下
  · generator           torch.Generator("cpu")    ← 06c/06d 用的是 cuda:0

⚠️ 为什么 use_stream 默认关（2026-09-02 实测）：
  读 diffusers/hooks/group_offloading.py 确认，use_stream=True 会走两个额外机制：
    1. ModuleGroup._onload_from_memory() 在每次 onload 时对该组全部张量做
       _pinned_memory_tensors()（临时 pin_memory）。
    2. _apply_group_offloading_leaf_level() / block_level 会注册
       LazyPrefetchGroupOffloadingHook —— 它靠 trace 首帧的执行顺序来预取下一组。
  机制 2 在本模型上是脆的：t2va 工作流下 Qwen3-VL 的 ~150 个 visual.* 层根本不执行，
  diffusers 自己就会打出「layers were not executed ... lead to device-mismatch related
  errors」警告，trace 出来的顺序必然是残缺的。
  关掉 stream 后：不 pin、不预取、同步 H2D，是最保守也最不容易出驱动级错误的路径。
  代价只是权重搬运不与计算重叠（本场景早已是 swap/IO 瓶颈，损失有限）。

A/B 开关（默认全关 = 最保守路径）：
  DEV_STREAM=1        打开 use_stream=True（官方 doc 推荐，但会启用 lazy prefetch）
  DEV_CUDA_GEN=1      用 torch.Generator("cuda:0") 代替 CPU generator
  DEV_EXP_SEG=1       设置 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
                      （默认不开：run#3 的 OOM 真因是 VAE 常驻占 11GB，不是碎片）

Env vars:
  MODEL_PATH, DEVICE, WIDTH, HEIGHT, NUM_FRAMES, FPS, SEED,
  NUM_INFERENCE_STEPS（默认 51 = 50 步去噪，见下方说明）, PROMPT, FIRST_FRAME, OUTPUT_DIR
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 必须在 torch/CUDA 初始化之前设置。默认不开：WSL2 下 CUDA VMM(虚拟内存)接口
# 不够稳，而 run#3 的 OOM 真因是 VAE 常驻占 11GB，并非碎片。
if os.environ.get("DEV_EXP_SEG", "0") == "1":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 模块级 import：vae.decode 钩子（logged_decode/stats）在 main() 之外被调用，
# torch 必须在模块作用域可用。run#5 教训：钩子里 NameError 杀掉了 37 分钟的
# 去噪成果 —— 诊断代码绝不允许弄丢生成结果。
import torch  # noqa: E402  (必须在上面 env 设置之后)

MODEL_PATH = os.environ.get("MODEL_PATH", "/mnt/d/wheel/minimaxh3_ms")
DEVICE = os.environ.get("DEVICE", "cuda:0")
WIDTH = int(os.environ.get("WIDTH", "960"))
HEIGHT = int(os.environ.get("HEIGHT", "544"))
NUM_FRAMES = int(os.environ.get("NUM_FRAMES", "124"))
FPS = int(os.environ.get("FPS", "24"))
SEED = int(os.environ.get("SEED", "42"))
# MiniMaxH3Scheduler 的 num_inference_steps 计 sigma 格点（含末尾 0），
# N 次去噪要传 N+1 —— 官方 doc 原文确认，所以 50 步传 51。
NIS = int(os.environ.get("NUM_INFERENCE_STEPS", "51"))
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR", "/mnt/d/output/minimaxh3_rotate_results/diag_baseline")
PROMPT = os.environ.get("PROMPT", "A red fox trotting through a snowy pine forest, "
                                  "snow crunching underfoot")
FIRST_FRAME = os.environ.get("FIRST_FRAME", "")

DEV_STREAM = os.environ.get("DEV_STREAM", "0") == "1"
DEV_CUDA_GEN = os.environ.get("DEV_CUDA_GEN", "0") == "1"
DEV_EXP_SEG = os.environ.get("DEV_EXP_SEG", "0") == "1"

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
    """(总GB, 可用GB, swap已用GB)；非 Linux 返回 None。"""
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
    """5 秒自检，把「GPU 状态坏」从 20 分钟后的失败提前成 5 秒后的失败。

    起因：run#3 以 torch.OutOfMemoryError 硬崩（PyTorch 自报在 24GB 卡上
    分配了 65.68GiB，明显异常），run#4 紧接着在第一次 H2D 传输就报
    CUDA driver error: device not ready —— 这是驱动级错误，不是内存错误，
    典型原因是 WSL2 的 GPU 分区在上次崩溃后没释放干净。
    """
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
        print("  → 上一次 OOM 后 WSL2 的 GPU 分区很可能没释放干净。")
        print("    Windows PowerShell 执行： wsl --shutdown")
        print("    等 10 秒重开终端，`nvidia-smi` 确认正常后再重跑本脚本。")
        raise SystemExit(1)

    m = _host_mem_gb()
    if m:
        tot, avail, swap_used = m
        print(f"  🖥️  主机内存：总 {tot:.1f}GB，可用 {avail:.1f}GB，"
              f"swap 已用 {swap_used:.1f}GB")
        if avail < 8:
            print("  ⚠️  可用主机内存偏低。int8 权重工作集约 76GB"
                  "（transformer 33 + text_encoder 32 + VAE 10.4 + audio_vae 0.6），"
                  "超出部分走 swap：慢，且极端时会触发驱动超时。")


def main():
    import torch
    from diffusers import (
        MiniMaxH3Transformer3DModel,
        ModularPipeline,
        TorchAoConfig,
    )
    from diffusers.hooks import apply_group_offloading
    from diffusers.utils.export_utils import encode_video
    from torchao.quantization import Int8WeightOnlyConfig
    from transformers import Qwen3VLForConditionalGeneration
    from transformers import TorchAoConfig as TransformersTorchAoConfig

    import diffusers
    print(f"{'=' * 68}")
    print("🎯 09b — MiniMax-H3 官方消费卡 baseline（无魔改）")
    print(f"  diffusers : {diffusers.__version__}  ({os.path.dirname(diffusers.__file__)})")
    print(f"  torch     : {torch.__version__}")
    print(f"  model     : {MODEL_PATH}")
    print(f"  device    : {DEVICE}  {torch.cuda.get_device_name(0)}")
    print(f"  video     : {NUM_FRAMES}f {WIDTH}x{HEIGHT} @ {FPS}fps")
    print(f"  steps     : {NIS} (={NIS - 1} 次去噪)  seed={SEED}")
    print(f"  A/B 开关  : stream={DEV_STREAM}  cuda_gen={DEV_CUDA_GEN}"
          f"  exp_seg={DEV_EXP_SEG}  (low_cpu_mem 恒为 True：量化加载硬性要求)")
    print(f"{'=' * 68}")

    preflight(torch)

    lcm_flag = True  # 量化加载在 diffusers（0.40.0 起，main 同）强制要求 True
    from _ensure_modular_index import ensure_modular_model_index
    print(f"\n  📦 {ensure_modular_model_index(MODEL_PATH)}")

    print("\n📦 loading（int8 + block offload，从 D 盘约 20-40 min）...")
    t0 = time.time()
    pipe = ModularPipeline.from_pretrained(MODEL_PATH)
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
    )
    workflow = "fl2va" if FIRST_FRAME else "t2va"
    pipe.load_components(workflow=workflow, dtype=torch.bfloat16,
                         pretrained_model_name_or_path=MODEL_PATH)
    pipe.transformer.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    # low_cpu_mem_usage=True 是 64GB 内存的硬性要求：默认 False 时 offload 会把
    # 全部权重 pin 成锁页内存（transformer 33GB + text_encoder 32GB ≈ 65GB > 物理内存
    # → pin_memory() 直接 CUDA OOM）。True 时不复制不 pin，页可换出 swap。
    offload = dict(onload_device=torch.device(DEVICE),
                   offload_device=torch.device("cpu"),
                   low_cpu_mem_usage=True)
    used_stream = False
    if DEV_STREAM:
        offload["use_stream"] = True
    try:
        pipe.transformer.enable_group_offload(
            offload_type="block_level", num_blocks_per_group=1, **offload)
        apply_group_offloading(
            pipe.text_encoder.model, offload_type="leaf_level", **offload)
        used_stream = DEV_STREAM
    except TypeError as e:
        # 旧版 enable_group_offload 可能没有 use_stream 参数
        print(f"  ⚠️ use_stream 不被当前版本支持（{e}），回退到无流式 offload")
        offload.pop("use_stream", None)
        pipe.transformer.enable_group_offload(
            offload_type="block_level", num_blocks_per_group=1, **offload)
        apply_group_offloading(
            pipe.text_encoder.model, offload_type="leaf_level", **offload)
    # VAE 一律不走 stream：避免再挂一份 LazyPrefetch 到 VAE 上
    print(f"  ℹ️ streamed offload: {'ON（含 lazy prefetch）' if used_stream else 'OFF'}")
    # VAE fp32 10.4GB 不常驻 GPU（去噪阶段用不到，省出 11GB 给 transformer 激活）；
    # 解码时 leaf 级按需搬入。09a 已验证 tile 分块解码单卡放得下。
    pipe.vae.enable_group_offload(onload_device=torch.device(DEVICE),
                                  offload_device=torch.device("cpu"),
                                  offload_type="leaf_level",
                                  low_cpu_mem_usage=True)
    pipe.audio_vae.to(DEVICE)
    print(f"  ✅ loaded in {(time.time() - t0) / 60:.1f} min")

    # ── latent 统计钩子：这是整份脚本的核心判据 ─────────────────────────
    # 去噪结束后喂给 VAE 的那个 latent，直接区分「transformer 侧的锅」和
    # 「VAE 解码侧的锅」。若它幅值正常而输出是马赛克 → 查 VAE；若它本身
    # 就爆掉 → 查 transformer / 量化 / offload / scheduler。
    n = {"calls": 0}
    orig_decode = pipe.vae.decode

    def logged_decode(z, *a, **kw):
        n["calls"] += 1
        print(f"\n  🔍 [VAE decode #{n['calls']}]")
        # ① 先落盘再干别的：去噪 37 分钟的成果绝不能再丢。
        #    存的就是喂给 vae.decode 的原始输入，09c 可离线重解码。
        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            lat_path = os.path.join(OUTPUT_DIR, f"latent_decode{n['calls']}.pt")
            torch.save(
                {"z": z.detach().to("cpu", torch.float32),
                 "latents_mean": list(pipe.vae.config.latents_mean),
                 "latents_std": list(pipe.vae.config.latents_std),
                 "meta": {"width": WIDTH, "height": HEIGHT,
                          "num_frames": NUM_FRAMES, "fps": FPS,
                          "seed": SEED, "steps": NIS}},
                lat_path)
            print(f"  💾 latent 已存盘: {lat_path}")
        except Exception as e:
            print(f"  ⚠️ latent 存盘失败（不影响继续）: {type(e).__name__}: {e}")
        # ② 统计打印：任何失败都不允许杀掉生成（run#5 的教训）
        try:
            print(f"  🔍 [VAE decode #{n['calls']}] 输入 latent:")
            stats("z (video latent)", z)
            lm = torch.tensor(pipe.vae.config.latents_mean,
                              device=z.device).view(1, -1, 1, 1, 1)
            ls = torch.tensor(pipe.vae.config.latents_std,
                              device=z.device).view(1, -1, 1, 1, 1)
            stats("z * std + mean (反归一化)", z * ls + lm)
        except Exception as e:
            print(f"  ⚠️ latent 统计失败（不影响继续）: {type(e).__name__}: {e}")
        out = orig_decode(z, *a, **kw)
        try:
            sample = out.sample if hasattr(out, "sample") else out[0]
            print(f"  🔍 [VAE decode #{n['calls']}] 输出像素:")
            stats("decoded sample", sample)
        except Exception as e:
            print(f"  ⚠️ 输出统计失败（不影响继续）: {type(e).__name__}: {e}")
        return out

    pipe.vae.decode = logged_decode

    # ── 生成 ────────────────────────────────────────────────────────────
    gen = (torch.Generator(DEVICE) if DEV_CUDA_GEN else torch.Generator("cpu"))
    gen = gen.manual_seed(SEED)
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
    out_path = os.path.join(OUTPUT_DIR, f"baseline_{workflow}_seed{SEED}.mp4")
    encode_video(results["videos"][0], fps=FPS, output_path=out_path,
                 audio=results["audio"][0],
                 audio_sample_rate=results["sampling_rate"])
    print(f"  💾 {out_path}  ({os.path.getsize(out_path) / 1e6:.1f} MB)")

    print(f"\n{'=' * 68}")
    print("👉 判读：")
    print("  · latent std 在 1 附近、无 NaN、|max| < 10，但出片是马赛克")
    print("      → transformer 正常，锅在 VAE 解码 / 后处理，去跑 09a")
    print("  · latent 幅值爆炸（std > 10 或 |max| > 100）或含 NaN")
    print("      → transformer 侧坏了：按 A/B 开关逐个复现魔改，定位到具体哪一项")
    print("  · 本脚本官方原样就出片干净")
    print("      → 06c/06d 的 4 处偏离就是根因，照这份脚本改回去即可")
    print(f"{'=' * 68}")


if __name__ == "__main__":
    main()
