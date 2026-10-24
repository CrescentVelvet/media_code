#!/usr/bin/env python3
"""Qwen-Image-2.1 批量图像编辑（去反光等逐图编辑任务的通用批处理）。

模型只加载一次，逐图 edit，支持断点续跑（输出已存在则跳过）。

Env vars:
  MODEL_PATH     模型目录（diffusers 结构）
  INPUT_DIR      输入图像目录（必填）
  OUTPUT_DIR     输出目录（必填）；输出文件名与输入同名
  GLOB           匹配模式（默认 "*.jpg"，多个用逗号分隔，如 "*.jpg,*.png"）
  PROMPT         编辑指令（必填）
  STEPS          采样步数（默认 40；2MP 图在 3090 上约 3s/步）
  SEED           基础种子（默认 42）；每张图实际种子 = SEED + 序号，保证可复现且图间不雷同
  GUIDANCE_SCALE 可选
  CPU_OFFLOAD    1=enable_model_cpu_offload（默认，3090 必需）
  START          起始序号（默认 0，含）
  LIMIT          最多处理张数（默认 0=全部）；配合 START 分批
  RESIZE_BACK    1（默认）= 输出尺寸与输入不一致时 LANCZOS 缩回原尺寸
                 （重建下游要求内参一致，尺寸必须保持）
  JPG_QUALITY    jpg 输出质量（默认 98，subsampling=0）
  FAIL_LOG       失败清单文件名（默认 OUTPUT_DIR/_failed.txt）
"""
import os
import sys
import time
import glob as globmod

MODEL_PATH = os.environ.get("MODEL_PATH", "")
INPUT_DIR = os.environ.get("INPUT_DIR", "")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "")
GLOB = os.environ.get("GLOB", "*.jpg")
PROMPT = os.environ.get("PROMPT", "")
STEPS = int(os.environ.get("STEPS", "40"))
SEED = int(os.environ.get("SEED", "42"))
GUIDANCE_SCALE = os.environ.get("GUIDANCE_SCALE", "")
CPU_OFFLOAD = os.environ.get("CPU_OFFLOAD", "1") == "1"
START = int(os.environ.get("START", "0"))
LIMIT = int(os.environ.get("LIMIT", "0"))
RESIZE_BACK = os.environ.get("RESIZE_BACK", "1") == "1"
JPG_QUALITY = int(os.environ.get("JPG_QUALITY", "98"))
FAIL_LOG = os.environ.get("FAIL_LOG", "")


def collect_inputs():
    paths = []
    for pat in GLOB.split(","):
        pat = pat.strip()
        if pat:
            paths.extend(globmod.glob(os.path.join(INPUT_DIR, pat)))
    # 去重 + 按文件名排序（数值序文件名即拍摄顺序）
    paths = sorted(set(paths), key=lambda p: os.path.basename(p))
    return paths


def main():
    if not MODEL_PATH or not os.path.isfile(os.path.join(MODEL_PATH, "model_index.json")):
        sys.exit("❌ MODEL_PATH 无效: " + repr(MODEL_PATH))
    if not INPUT_DIR or not os.path.isdir(INPUT_DIR):
        sys.exit("❌ INPUT_DIR 无效: " + repr(INPUT_DIR))
    if not OUTPUT_DIR:
        sys.exit("❌ OUTPUT_DIR 为空")
    if not PROMPT:
        sys.exit("❌ PROMPT 为空")

    files = collect_inputs()
    if not files:
        sys.exit("❌ 未匹配到输入图: " + INPUT_DIR + " / " + GLOB)
    total_all = len(files)
    files = files[START:]
    if LIMIT > 0:
        files = files[:LIMIT]
    if not files:
        sys.exit("❌ START 超出范围，无图可处理")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fail_log = FAIL_LOG or os.path.join(OUTPUT_DIR, "_failed.txt")

    # 断点续跑：输出已存在且非空则跳过
    todo = []
    for p in files:
        out = os.path.join(OUTPUT_DIR, os.path.basename(p))
        if os.path.isfile(out) and os.path.getsize(out) > 0:
            continue
        todo.append((p, out))

    print("🚀 [batch] Qwen-Image-2.1 批量编辑")
    print("  📂 输入: " + INPUT_DIR + "（共 " + str(total_all) + " 张，"
          + "本次区间 " + str(len(files)) + " 张，待处理 " + str(len(todo)) + " 张）")
    print("  📂 输出: " + OUTPUT_DIR)
    print("  📝 prompt: " + PROMPT)
    print("  ⚙️  steps=" + str(STEPS) + " seed_base=" + str(SEED)
          + " offload=" + str(CPU_OFFLOAD) + " resize_back=" + str(RESIZE_BACK))
    if not todo:
        print("✅ 全部已完成，无需处理")
        return

    import torch
    from PIL import Image
    from diffusers import QwenImage21Pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("  🎮 device: " + device)
    t_load = time.time()
    pipe = QwenImage21Pipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
    if CPU_OFFLOAD:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)
    print("  ⏱️ 模型加载 %.1fs" % (time.time() - t_load))

    n_done, n_fail = 0, 0
    t_start = time.time()
    for i, (src, out) in enumerate(todo):
        seed = SEED + i
        tag = "[" + str(i + 1) + "/" + str(len(todo)) + "] " + os.path.basename(src)
        try:
            img = Image.open(src).convert("RGB")
            in_size = img.size
            kwargs = dict(
                prompt=PROMPT,
                image=img,
                num_inference_steps=STEPS,
                generator=torch.Generator(device).manual_seed(seed),
            )
            if GUIDANCE_SCALE:
                kwargs["guidance_scale"] = float(GUIDANCE_SCALE)
            t0 = time.time()
            result = pipe(**kwargs).images[0]
            dt = time.time() - t0
            if RESIZE_BACK and result.size != in_size:
                print("  ⚠️ 输出尺寸 " + str(result.size) + " != 输入 " + str(in_size)
                      + "，LANCZOS 缩回")
                result = result.resize(in_size, Image.LANCZOS)
            if out.lower().endswith((".jpg", ".jpeg")):
                result.save(out, quality=JPG_QUALITY, subsampling=0)
            else:
                result.save(out)
            n_done += 1
            # ETA：已完成张数的平均耗时
            avg = (time.time() - t_start) / (i + 1)
            eta_min = avg * (len(todo) - i - 1) / 60.0
            print("✅ " + tag + "  %.1fs  (ETA %.1f min)" % (dt, eta_min), flush=True)
        except Exception as e:
            n_fail += 1
            print("❌ " + tag + "  " + repr(e), flush=True)
            with open(fail_log, "a", encoding="utf-8", newline="\n") as f:
                f.write(src + "\t" + repr(e) + "\n")

    dt_all = time.time() - t_start
    print("🎉 Done. 成功 " + str(n_done) + " / 失败 " + str(n_fail)
          + "，总耗时 %.1f min" % (dt_all / 60.0))
    if n_fail > 0:
        print("⚠️ 失败清单: " + fail_log)


if __name__ == "__main__":
    main()
