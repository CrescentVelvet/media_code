#!/usr/bin/env python3
"""09f — text encoder（Qwen3-VL-32B）单独体检：它给 transformer 供的条件是不是好的。

为什么有这个脚本（2026-09-29）：
  历史产物对照（A=06c / B,C=06d / D=06b / E=09b 全部出噪点）已逐一排除：
    int8（D 是 bf16 也坏）、LoRA（A/E 无 LoRA 也坏）、VAE offload（常驻也坏）、
    offload 路径（tiny fixture 四配方 latent 逐字节一致）、配置（26 个 json 与上游同字节）。
  所有失败路径共用的、又从未单独验证过的组件只剩 text encoder：
  真实 Qwen3-VL-32B + transformers 5.15.1 + int8/leaf-offload 的输出从未被单独看过。
  tiny fixture 的 text encoder 是随机小模型，验证不了它。

检查项（逐层 + 三 prompt）：
  1. hidden_states 各层的 NaN / mean / std / |max|（层 0 / 25 / 50 / 最后）
  2. token 区分度：同一 prompt 内 token 两两余弦均值（健康应 << 1，趋 1 = 输出塌缩）
  3. 确定性：同一 prompt 跑两遍，prefix 相同部分 max|Δ| 应为 0
  4. prompt 区分度：不同 prompt 的 pooled 余弦（健康应 < 0.99，趋 1 = 无视输入）
  5. 额外构造共享前缀的 prompt 对，验证因果前缀不变性

与 09b 同款加载：int8(version=2) + TE_NOT_CONVERT + leaf_level offload（低 RAM 配方）。
加载 ~10-15 min（int8 32GB，从 /mnt/d 读）。

Env vars:
  MODEL_PATH, DEVICE, OUTPUT_DIR
"""
import os
import sys
import time

import torch

MODEL_PATH = os.environ.get("MODEL_PATH", "/mnt/d/wheel/minimaxh3_ms")
DEVICE = os.environ.get("DEVICE", "cuda:0")
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR", "/mnt/d/output/minimaxh3_rotate_results/diag_textenc")

TE_NOT_CONVERT = [
    "model.visual", "model.language_model.embed_tokens",
    "model.language_model.norm", "lm_head",
]

FOX = ("A red fox trotting through a snowy pine forest, "
       "snow crunching underfoot")
ROTATE = ("integrated_multimodal_description: [Shot 1] Cinematic medium-wide "
          "shot. The subject stands perfectly centered in frame, stock-still. "
          "The camera glides along a horizontal circular ring track in a "
          "smooth 360-degree orbit around the subject.")
# 与 FOX 共享前缀，用于因果前缀不变性校验
FOX_LONG = FOX + " The camera slowly pulls back to reveal the whole valley."


def stats(name, t):
    t = t.detach().float()
    finite = torch.isfinite(t)
    if int(finite.sum()) == 0:
        print(f"  {name:<34} 全部 NaN/Inf !!")
        return
    print(
        f"  {name:<34} shape={tuple(t.shape)} "
        f"mean={t[finite].mean():+.4f} std={t[finite].std():.4f} "
        f"min={t[finite].min():+.3f} max={t[finite].max():+.3f} "
        f"nan/inf={int((~finite).sum())}"
    )


def cos(a, b):
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    n = min(a.numel(), b.numel())
    a, b = a[:n], b[:n]
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def encode(te, processor, token_ids, layers_want):
    """复刻 get_qwen3vl_prompt_embeds 的内部，但返回全部 hidden_states。"""
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)
    mm_token_type_ids = torch.tensor(
        processor.create_mm_token_type_ids([token_ids]),
        dtype=torch.long, device=DEVICE)
    hook = getattr(te, "_hf_hook", None)
    if hook is not None and hasattr(hook, "pre_forward"):
        hook.pre_forward(te)
    with torch.no_grad():
        outputs = te.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            mm_token_type_ids=mm_token_type_ids,
            use_cache=False,
            output_hidden_states=True,
        )
    hs = outputs.hidden_states
    out = {}
    for l in layers_want:
        idx = l if l >= 0 else len(hs) + l
        out[l] = hs[idx].detach().to("cpu", torch.float32)
    return out, len(hs), len(token_ids)


def main():
    from transformers import (
        Qwen2TokenizerFast, Qwen3VLProcessor, Qwen3VLForConditionalGeneration)
    from transformers import TorchAoConfig as TransformersTorchAoConfig
    from torchao.quantization import Int8WeightOnlyConfig
    from diffusers.hooks import apply_group_offloading

    print(f"{'=' * 70}")
    print("🎯 09f — text encoder 单独体检（t2va 编码路径复刻）")
    print(f"  model : {MODEL_PATH}")
    print(f"  device: {DEVICE}")
    print(f"{'=' * 70}")

    tokenizer = Qwen2TokenizerFast.from_pretrained(
        MODEL_PATH, subfolder="tokenizer")
    processor = Qwen3VLProcessor.from_pretrained(
        MODEL_PATH, subfolder="processor")

    print("\n📦 loading text_encoder (int8 + leaf offload, ~10-15 min)...")
    t0 = time.time()
    te = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, subfolder="text_encoder", dtype=torch.bfloat16,
        quantization_config=TransformersTorchAoConfig(
            Int8WeightOnlyConfig(version=2),
            modules_to_not_convert=TE_NOT_CONVERT,
        ),
        low_cpu_mem_usage=True,
    )
    te.requires_grad_(False)
    apply_group_offloading(
        te.model, offload_type="leaf_level",
        onload_device=torch.device(DEVICE),
        offload_device=torch.device("cpu"),
        low_cpu_mem_usage=True)
    print(f"  ✅ loaded in {(time.time() - t0) / 60:.1f} min")
    n_layers = te.config.text_config.num_hidden_layers
    print(f"  layers={n_layers} hidden={te.config.text_config.hidden_size}")

    LAYERS = [0, 25, 50, -1]
    prompts = {"fox": FOX, "rotate": ROTATE, "a": "a"}
    results = {}
    n_tok = {}

    for name, p in prompts.items():
        token_ids = tokenizer(p, add_special_tokens=False)["input_ids"]
        n_tok[name] = len(token_ids)
        t1 = time.time()
        hs, total, ntok = encode(te, processor, token_ids, LAYERS)
        dt = time.time() - t1
        results[name] = hs
        print(f"\n── prompt[{name}] tokens={ntok} hidden_states={total} "
              f"encode={dt:.1f}s ──")
        for l in LAYERS:
            stats(f"hs[{l}] {name}", hs[l])
        # token 区分度：token 两两余弦均值（用 50 层）
        e = hs[50][0]                                  # (N, 5120)
        if e.shape[0] > 1:
            sim = torch.nn.functional.cosine_similarity(
                e.unsqueeze(0), e.unsqueeze(1), dim=-1)
            m = sim.shape[0]
            off = sim[~torch.eye(m, dtype=bool)]
            print(f"  token 两两余弦[{name}] mean={off.mean():+.4f} "
                  f"max={off.max():.4f} min={off.min():+.4f}  "
                  f"(趋 1 = 输出塌缩)")

    # 确定性：fox 再跑一遍，比较 50 层
    print("\n── 确定性：fox 重跑 ──")
    hs2, _, _ = encode(
        te, processor,
        tokenizer(FOX, add_special_tokens=False)["input_ids"], [50])
    d = (hs2[50] - results["fox"][50]).abs()
    print(f"  fox 两次 max|Δ|={d.max():.3e}  mean|Δ|={d.mean():.3e}  "
          f"(应严格为 0；非 0 = offload 搬运在改权重)")

    # 因果前缀不变性：FOX vs FOX_LONG 的前 n 个 token
    print("\n── 前缀不变性：fox vs fox+长后缀 ──")
    ids_long = tokenizer(FOX_LONG, add_special_tokens=False)["input_ids"]
    hs3, _, _ = encode(te, processor, ids_long, [50])
    n = results["fox"][50].shape[1]
    d = (hs3[50][:, :n] - results["fox"][50]).abs()
    print(f"  共享前缀 {n} tokens max|Δ|={d.max():.3e}  "
          f"(应严格为 0；非 0 = 非因果/状态污染)")

    # prompt 区分度
    print("\n── prompt 区分度（50 层 mean-pool 余弦）──")
    pooled = {k: v[50].mean(dim=1)[0] for k, v in results.items()}
    for i, a in enumerate(pooled):
        for b in list(pooled)[i + 1:]:
            print(f"  cos({a},{b}) = {cos(pooled[a], pooled[b]):+.4f}  "
                  f"(趋 1 = 无视 prompt)")

    # 落盘，供后续离线比对
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "textenc_embeds.pt")
    torch.save({k: {l: v for l, v in hs.items()}
                for k, hs in results.items()}, out_path)
    print(f"\n💾 embeds 已存: {out_path}")

    print(f"\n{'=' * 70}")
    print("👉 判读：")
    print("  · 任一层 NaN / std 爆（>1e3）/ token 余弦趋 1 / 两次跑 Δ≠0")
    print("      → text encoder 侧实锤（int8 量化或 transformers 实现）")
    print("  · 全部正常（无 NaN、token 有区分、Δ=0、prompt 可分）")
    print("      → text encoder 洗清，嫌疑收敛到 transformer 权重本体")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
