#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
structsplat/00e_probe_vram.py — 前向显存/耗时探测（逐帧数扫描，找 3090 24GB 的安全上限）

目的：VGGT-1B 有 24 层 frame + 24 层 global attention，global 注意力是对
「所有帧的 patch token 拉平」做的，token 数 = S × (H/14)×(W/14) + 寄存器。
294×518 时每帧 21×37=777 patch + 4 reg + 1 cls ≈ 782，fp32 下 82 帧 ≈ 6.4 万 token，
显存随 S 近似线性甚至超线性增长。官方在 H100 80GB 上跑，单卡 3090 必须先探边界。

本脚本只做「建模型 + 前向」，不渲染不落盘，用来：
  1) 找到不 OOM 的最大帧数（二分/递增扫描）；
  2) 记录每个帧数的峰值显存与耗时，给 02_run_inference.sh 定默认 max-images。

用法:
  python 00e_probe_vram.py --images <目录> [--frames 4,8,16,24,32] [--resize 294 518]
"""

import argparse
import gc
import importlib.util
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("STRUCTSPLAT_REPO") or os.path.expanduser("~/repos/StructSplat")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "third_party", "vggt"))


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--frames", default="4,8,12,16,24,32",
                    help="逗号分隔的帧数序列，逐档试")
    ap.add_argument("--resize", nargs=2, type=int, default=[294, 518])
    ap.add_argument("--config", default=os.path.join(REPO, "config", "dl3dv.yaml"))
    args = ap.parse_args()

    inf = load_mod("inf_probe", os.path.join(HERE, "infer.py"))
    new_h, new_w = args.resize
    frames_list = [int(x) for x in args.frames.split(",") if x.strip()]

    if not torch.cuda.is_available():
        print("❌ 无 CUDA")
        return 2
    dev = torch.device("cuda")
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {torch.cuda.get_device_name(0)}  {total_vram:.1f} GiB")

    # 一次性把最长的图片列表准备好，后面按子集切
    all_paths = inf.collect_images(args.images, max_images=max(frames_list))
    print(f"可用图片 {len(all_paths)} 张")

    from src.config import load_configs
    from src.model.gaussian_wrapper import build_gaussian_wrapper

    cfg = load_configs(args.config, [])
    ckpt_path = cfg.gaussian_evaluation_stage.ckpt
    if not os.path.exists(ckpt_path):
        print(f"❌ ckpt 不存在: {ckpt_path}")
        return 2

    # VGGT 必须给 model.safetensors（不是 model.pt）：huggingface_hub 的本地目录
    # 分支硬编码只读 safetensors，HF 上 VGGT-1B 却只发 model.pt → 目录存在也报
    # FileNotFoundError: ckpts/vggt/model.safetensors（详见 README §4.1c）
    for _key, _p in (("geo_encoder", cfg.module.geo_encoder.path),
                     ("sem_encoder", getattr(cfg.module.sem_encoder, "path", None))):
        if not _p:
            continue
        _abs = _p if os.path.isabs(_p) else os.path.join(inf.REPO, _p)
        if not os.path.exists(os.path.join(_abs, "model.safetensors")):
            print(f"❌ {_key} 缺 model.safetensors: {_abs}")
            print("   先跑 bash structsplat/01_download_models.sh all")
            return 2

    print(f"\n{'frames':>7} {'峰值显存':>10} {'剩余':>8} {'前向':>8}  状态")
    print("-" * 62)

    # ★ 模型只建一次。早期版本把 torch.load(3.5GB) + build 放在循环里，
    #   扫 6 档就重复读盘/建模型 6 次，光这一项就浪费好几分钟。
    param_dict = torch.load(ckpt_path, map_location="cpu")
    module_dict = {"gaussian_predictor": {}}
    for k, v in param_dict.items():
        if k.startswith("gaussian_predictor."):
            module_dict["gaussian_predictor"][k[len("gaussian_predictor."):]] = v
    del param_dict
    gc.collect()

    wrapper = build_gaussian_wrapper(module_dict, cfg).to(dev).eval()
    for p in wrapper.parameters():
        p.requires_grad = False
    print(f"模型已就位（{sum(p.numel() for p in wrapper.parameters())/1e6:.1f} M 参数）\n")

    results = []
    for S in frames_list:
        if S > len(all_paths):
            print(f"{S:>7}  ——  跳过（只有 {len(all_paths)} 张图）")
            continue
        try:
            torch.cuda.empty_cache()
            gc.collect()

            imgs = []
            for p in all_paths[:S]:
                t, _ = inf.load_image_tensor(p, new_h, new_w)
                imgs.append(t)
            src = torch.stack(imgs).unsqueeze(0).to(dev)

            torch.cuda.reset_peak_memory_stats()

            t0 = time.time()
            with torch.no_grad():
                preds = wrapper(src)
            torch.cuda.synchronize()
            dt = time.time() - t0

            peak = torch.cuda.max_memory_allocated() / 1024**3
            n_gauss = preds["coordinate"].shape[1]
            free = total_vram - peak
            print(f"{S:>7} {peak:>9.2f}G {free:>7.2f}G {dt:>7.2f}s  ✅ {n_gauss} 高斯")
            results.append((S, peak, dt, n_gauss))

            del preds, src, imgs
        except torch.cuda.OutOfMemoryError as e:
            print(f"{S:>7} {'OOM':>10} {'--':>8} {'--':>8}  ❌ 显存不足")
            print(f"        {str(e)[:100]}")
            torch.cuda.empty_cache()
            gc.collect()
            # 一旦 OOM，更大的帧数只会更差，停止扫描
            print("\n（已 OOM，停止向上扫描）")
            break
        except Exception as e:
            print(f"{S:>7} {'ERR':>10} {'--':>8} {'--':>8}  ❌ {type(e).__name__}")
            print(f"        {str(e)[:200]}")
            break
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    print("-" * 62)
    if results:
        ok = [r for r in results]
        best = max(ok, key=lambda r: r[0])
        print(f"安全上限（本扫描内）：{best[0]} 帧，峰值 {best[1]:.2f} GiB，{best[2]:.2f}s")
        print(f"建议 02_run_inference.sh 默认 --max-images ≤ {best[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
