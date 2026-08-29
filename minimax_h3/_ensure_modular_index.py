#!/usr/bin/env python3
"""确保 diffusers 0.40 能加载旧版（0.36）保存的模块化 pipeline。

坑：modelscope/HF 上的 MiniMax-H3 快照由 diffusers 0.36.0.dev0 保存，模块化配置
（3 元组 + _blocks_class_name）写在 model_index.json 里，没有 modular_model_index.json。
diffusers 0.40 的 _load_pipeline_config 先找 modular_model_index.json（找不到）→
回退 model_index.json 路径 → _get_model("MiniMaxH3ModularPipeline") 返回 None（这函数
只认标准 pipeline 类名）→ 误用基类 ModularPipeline（default_blocks_name=None）→
__init__ 里 blocks_class_name 为 None、blocks_class 从未赋值 → UnboundLocalError。

修法：若 modular_model_index.json 缺失而 model_index.json 是模块化格式（含
_blocks_class_name），复制一份。复制后 0.40 走模块化加载路径，_class_name 直接
解析成 MiniMaxH3ModularPipeline，正常工作。幂等，已存在则跳过。
"""
import os
import json
import shutil


def ensure_modular_model_index(model_path):
    """若缺 modular_model_index.json 且 model_index.json 是模块化格式，复制一份。

    返回操作描述字符串（供调用方 print）。
    """
    modular_idx = os.path.join(model_path, "modular_model_index.json")
    standard_idx = os.path.join(model_path, "model_index.json")

    if os.path.isfile(modular_idx):
        return "modular_model_index.json already exists — skip"

    if not os.path.isfile(standard_idx):
        return "model_index.json not found — let from_pretrained report the error"

    with open(standard_idx, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # 模块化格式标志：_blocks_class_name（标准 model_index.json 没有这个字段）
    if "_blocks_class_name" not in cfg:
        return "model_index.json is standard (non-modular) format — no action needed"

    shutil.copy2(standard_idx, modular_idx)
    return (
        f"created modular_model_index.json (copy of model_index.json, "
        f"_blocks_class_name={cfg['_blocks_class_name']}) — fixes diffusers 0.40 "
        f"loading of 0.36-saved model"
    )
