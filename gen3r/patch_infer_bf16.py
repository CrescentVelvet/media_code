#!/usr/bin/env python3
"""[media_code patch] Gen3R infer.py 改为 bf16 直接加载。

原版:
    pipeline = Gen3RPipeline.from_pretrained(path)
    pipeline.to(device).to(torch.bfloat16)
先把 37.7GB fp32 权重推上显存再转 bf16 -> 24GB 卡溢出主机内存，
且 fp32 下 SDPA 无法走 flash/efficient 后端（仅支持 fp16/bf16），单步 >20min。

用法: python3 patch_infer_bf16.py [--revert]
"""
import sys

PATH = "/home/velvet/repos/Gen3R/infer.py"

ORIG = """    pipeline = Gen3RPipeline.from_pretrained(args.pretrained_model_name_or_path)
    pipeline.to(device).to(torch.bfloat16)"""

NEW = """    # [media_code patch] 原版不带 dtype = 先按 fp32 把 37.7GB 权重塞进显存再转 bf16：
    # 24GB 卡上溢出到主机内存（实测 shared GPU mem 21.6GB，单步 >20min），
    # 且 fp32 下 SDPA 走不了 flash/efficient 后端（仅支持 fp16/bf16）。改为直接 bf16 加载。
    # ⚠️ 光传 torch_dtype 不够：VAE 等子模块仍可能是 fp32（会报
    # "Input type (c10::BFloat16) and bias type (float) should be the same"），
    # 所以再 .to(dtype=) 统一一遍（此时权重已在 CPU 上，不会占显存）。
    # GEN3R_DTYPE=fp32 可回退原行为。
    _dtype = torch.float32 if os.environ.get("GEN3R_DTYPE", "bf16") == "fp32" else torch.bfloat16
    pipeline = Gen3RPipeline.from_pretrained(
        args.pretrained_model_name_or_path, torch_dtype=_dtype)
    pipeline.to(device=device, dtype=_dtype)"""


def main():
    revert = "--revert" in sys.argv
    with open(PATH, "r", encoding="utf-8") as f:
        src = f.read()
    if revert:
        old, new = NEW, ORIG
    else:
        old, new = ORIG, NEW
    if old not in src:
        print("⚠️ 目标片段不存在（可能已打过补丁）")
        sys.exit(1)
    src = src.replace(old, new, 1)
    with open(PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(src)
    print("✅ 已{}补丁".format("回退" if revert else "应用"))


if __name__ == "__main__":
    main()
