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

⚠️ lora_alpha 的取值规则（2026-09-29 依官方 Turbo 仓库源码定版）：
  PEFT 的实际缩放是 `lora_alpha / r`，不是 lora_alpha 本身。lightx2v 各 checkpoint
  **同一个 rank=128，但训练用 alpha 不同**。权威依据是官方推理脚本
  `ModelTC/Minimax-H3-Turbo/inference_minimax_h3.py`：
    · `DEFAULT_LORA_ALPHA = 8`（脚本默认，544p 两支与 v1.2/8step_768p 都按它训练）
    · 只有 `4step_v1.0_768p` 的官方示例显式 `--lora-alpha 128`
  与本机实测的 safetensors `__metadata__['alpha']`（rank 全部 =128）完全互洽：
    · 4step_v1.0_768p / v1.1_768p : alpha=128 → scale 1.0    （官方示例确认）
    · 4step_v1.2_768p             : alpha=8   → scale 0.0625 （= 官方默认）
    · 8step_v1.0_768p             : alpha=8   → scale 0.0625 （= 官方默认；Studio 在用的就是它）
    · 4step_v0.1 / 8step_v1.0(544p): 文件无 alpha 字段 → **回退官方默认 8** → scale 0.0625
  故 `lora_alpha` 默认走 "auto"：文件有记录用记录，没记录回退 **8**（不是 rank）。
  注（2026-09-29 勘误）：此前文档把「v1.1=128 vs v1.2=8」误判为「metadata 自相矛盾、
  至少一个记录是错的」并要求 A/B —— 官方 DEFAULT_LORA_ALPHA=8 证明 alpha=8 是正常
  训练值，不存在矛盾，无需 A/B。
  历史 bug：06b/06d 默认写死 128 → 8-step / v1.2 / v0.1 文件被放大 16×。
  ⚠️ 但注意：这个 bug 解释不了本机马赛克 —— 历史出噪点的运行（06c/06d/06b/09b）
  用的 checkpoint 其 alpha 恰好都是对的（v1.0_768p=128、v0.1=8、无 LoRA），根因另查。
"""
import gc
import sys
from collections.abc import Mapping
from pathlib import Path

import torch
from peft import LoraConfig
from safetensors import safe_open
from safetensors.torch import load_file as load_safetensors_file

# Turbo LoRA target modules（照 lightx2v/Minimax-h3-Turbo 训练配置）
LORA_TARGET_MODULES = ("to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2")
LORA_A_SUFFIX = ".lora_A.default.weight"
LORA_B_SUFFIX = ".lora_B.default.weight"

#: 这些值表示"不显式指定，按文件 __metadata__ 推导"
AUTO_ALPHA = (None, "", "auto", "AUTO")


def _read_file_alpha(lora_path):
    """读 safetensors 头的 `__metadata__['alpha']`，失败/缺失返回 None。"""
    try:
        with safe_open(str(lora_path), framework="pt") as f:
            raw = (f.metadata() or {}).get("alpha")
    except Exception as e:  # 读头失败不该中断推理
        print(f"  ⚠️ 读 {Path(lora_path).name} 的 __metadata__ 失败（{type(e).__name__}: {e}）")
        return None
    if raw is None:
        return None
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        print(f"  ⚠️ __metadata__['alpha']={raw!r} 不是数字，忽略（diffusers 同此处理）")
        return None


def resolve_lora_alpha(lora_path, rank, explicit=None):
    """决定 PEFT 的 `lora_alpha`，返回 (alpha, 来源说明)。

    优先级：显式传值 > 文件 `__metadata__['alpha']` > 官方默认 8。
    官方依据：`ModelTC/Minimax-H3-Turbo/inference_minimax_h3.py` 的
    `DEFAULT_LORA_ALPHA = 8`（544p 两支官方示例不传 --lora-alpha 即走 8）。

    显式值与文件记录值不一致时会打印告警（历史上写死 128 出 16× 偏差）。

    ⚠️ 回退值是 8 不是 rank：这些 Turbo LoRA 全部训练于 alpha=8（768p v1.x
    除外，它们 metadata 记 128）。若按 diffusers 惯例回退 rank(=128, scale 1.0)，
    会把 544p checkpoint 放大 16×。
    """
    file_alpha = _read_file_alpha(lora_path)

    if explicit not in AUTO_ALPHA and explicit != 0:
        alpha = int(explicit)
        if file_alpha is not None and file_alpha != alpha:
            ratio = alpha / file_alpha
            print(f"  ⚠️ 显式 LORA_ALPHA={alpha} 与文件 __metadata__['alpha']={file_alpha} 不一致"
                  f"（相差 {ratio:.4g}×）；按显式值 {alpha} 走 → scale={alpha / rank:.4f}")
        return alpha, "显式指定"

    if file_alpha is not None:
        return file_alpha, f"文件 __metadata__['alpha']={file_alpha}"

    return 8, "文件未记录 alpha，回退官方默认 8（DEFAULT_LORA_ALPHA）"


def load_lora_adapter(transformer, lora_path, lora_alpha=None, lora_scale=1.0, fuse_lora=False):
    """手动注入 PEFT LoRA（照 Turbo 仓库 inference_minimax_h3.py）。

    MiniMax-H3 是 ModularPipeline，标准 pipe.load_lora_weights 不认 active
    transformer；改成手动 add_adapter + load_state_dict。

    lora_alpha: None/""/"auto" 时按文件 __metadata__['alpha'] 推导（回退 rank），
                显式给数值则覆盖。规则见模块 docstring。
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

    # ⚠️ 关键：alpha 从文件读，别套常数（见模块 docstring 的取值规则）
    lora_alpha, alpha_src = resolve_lora_alpha(lora_path, rank, lora_alpha)
    # PEFT 的实际缩放 = alpha / r；打印出来便于和 checkpoint 配方表对照
    print(f"  🏋️ LoRA alpha = {lora_alpha} （{alpha_src}）→ PEFT scale = {lora_alpha}/{rank} "
          f"= {lora_alpha / rank:.4f}")

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
