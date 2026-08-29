#!/usr/bin/env python3
"""Turbo LoRA 注入工具（06b / 06d 共用）。

手动把 lightx2v/Minimax-h3-Turbo 蒸馏的 LoRA checkpoint 注入 MiniMax-H3 的
active transformer。照 Turbo 仓库 inference_minimax_h3.py：MiniMax-H3 是
ModularPipeline，标准 pipe.load_lora_weights 不认 active transformer，故手动
add_adapter + load_state_dict + set_adapters。

06d（int8 + Turbo）复用同一函数：LoRA 的 A/B 矩阵是新增 bf16 参数，与 int8
weight-only 量化兼容（量化的是原权重，LoRA 走独立计算支路）。注意：
- 注入 LoRA 必须在 enable_group_offload 之前，否则新增的 lora_A/B 子模块没被
  offload hook 覆盖，前向时 block 搬到 GPU 而 LoRA 留在 CPU → device 不匹配。
- FUSE_LORA 对 int8 不可用（fuse 要把 bf16 LoRA delta 加进 Int8Tensor 权重，
  safe_fusing=True 会直接报错）；06d 默认 FUSE_LORA=0，靠 set_adapters 运行时合并。
"""
import gc
import sys
from collections.abc import Mapping
from pathlib import Path

import torch
from peft import LoraConfig
from safetensors.torch import load_file as load_safetensors_file

# Turbo LoRA target modules（照 lightx2v/Minimax-h3-Turbo 训练配置）
LORA_TARGET_MODULES = ("to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2")
LORA_A_SUFFIX = ".lora_A.default.weight"
LORA_B_SUFFIX = ".lora_B.default.weight"


def load_lora_adapter(transformer, lora_path, lora_alpha, lora_scale, fuse_lora):
    """手动注入 PEFT LoRA（照 Turbo 仓库 inference_minimax_h3.py）。

    MiniMax-H3 是 ModularPipeline，标准 pipe.load_lora_weights 不认 active
    transformer；改成手动 add_adapter + load_state_dict。
    """
    lora_path = Path(lora_path)
    if not lora_path.is_file():
        sys.exit(f"❌ LoRA checkpoint not found: {lora_path}")
    if lora_path.suffix.lower() == ".safetensors":
        state_dict = load_safetensors_file(lora_path, device="cpu")
    else:
        try:
            state_dict = torch.load(lora_path, map_location="cpu", weights_only=True, mmap=True)
        except TypeError:
            state_dict = torch.load(lora_path, map_location="cpu", weights_only=True)
    if isinstance(state_dict, Mapping) and isinstance(state_dict.get("state_dict"), Mapping):
        state_dict = state_dict["state_dict"]

    # 校验 + 算 rank
    lora_a, lora_b, bad = {}, {}, []
    for k, v in state_dict.items():
        if k.endswith(LORA_A_SUFFIX):
            lora_a[k[:-len(LORA_A_SUFFIX)]] = v
        elif k.endswith(LORA_B_SUFFIX):
            lora_b[k[:-len(LORA_B_SUFFIX)]] = v
        else:
            bad.append(k)
    if bad:
        sys.exit(f"❌ {lora_path} not a pure PEFT LoRA state dict; bad keys: {bad[:3]}")
    if not lora_a:
        sys.exit(f"❌ no {LORA_A_SUFFIX} tensors in {lora_path}")
    missing_a = sorted(lora_b.keys() - lora_a.keys())
    missing_b = sorted(lora_a.keys() - lora_b.keys())
    if missing_a or missing_b:
        sys.exit(f"❌ unpaired LoRA tensors: missing A={missing_a[:3]}, missing B={missing_b[:3]}")
    ranks = set()
    for name in lora_a:
        a, b = lora_a[name], lora_b[name]
        if a.shape[0] != b.shape[1]:
            sys.exit(f"❌ LoRA rank mismatch for {name}: A{tuple(a.shape)} B{tuple(b.shape)}")
        ranks.add(a.shape[0])
    if len(ranks) != 1:
        sys.exit(f"❌ mixed LoRA ranks unsupported: {sorted(ranks)}")
    rank = ranks.pop()

    transformer.add_adapter(LoraConfig(
        r=rank, lora_alpha=lora_alpha, init_lora_weights=False,
        target_modules=list(LORA_TARGET_MODULES), use_rslora=False,
    ))
    adapter_params = {n: p for n, p in transformer.named_parameters()
                      if ".lora_A." in n or ".lora_B." in n}
    missing = sorted(adapter_params.keys() - state_dict.keys())
    unexpected = sorted(state_dict.keys() - adapter_params.keys())
    if missing or unexpected:
        sys.exit(f"❌ LoRA incompatible: missing={missing[:3]}, unexpected={unexpected[:3]}")
    transformer.load_state_dict(state_dict, strict=False)
    transformer.set_adapters("default", weights=lora_scale)
    if fuse_lora:
        # safe_fusing=True：融不进就报错而非静默损坏（int8 权重融不进，会在这里炸）
        transformer.fuse_lora(lora_scale=1.0, safe_fusing=True, adapter_names=["default"])
        transformer.unload_lora()
    transformer.requires_grad_(False)
    transformer.eval()
    del state_dict; gc.collect()
    print(f"  🏋️ LoRA loaded: {lora_path.name} rank={rank} alpha={lora_alpha} "
          f"scale={lora_scale} fused={fuse_lora}")
