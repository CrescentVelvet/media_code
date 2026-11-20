#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
structsplat/00ac_bench.py — 分阶段耗时 / 峰值显存实测（PPT 数据源）

为什么要单独测：meta.json 只有 inference_ms（前馈），没有峰值显存；而 PPT 要讲
「2 图 2.7s / 24GB 卡」这类结论，必须给出可复核的数字与测量口径。

口径（写进 PPT 脚注，别混谈）：
  - fb_ms      : 网络前向（VGGT-1B 聚合 + DINOv3 + 三解码器），不含建模型与读盘
  - build_ms   : 从 .bin 构 wrapper（torch.load 3.5GB + to(cuda)），一次性成本
  - raster_ms  : gsplat 光栅化 N 张视角的总耗时
  - peak_*     : torch.cuda.max_memory_allocated()，**分配器侧**而非 nvidia-smi
  - WSL2 统一内存下分配器峰值可超物理显存，故同时打印 nvidia-smi 的 used

用法:
  python 00ac_bench.py --images <dir> --frames 2,4 --resize 392 518
"""

import argparse
import gc
import importlib.util
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("STRUCTSPLAT_REPO") or os.path.expanduser("~/repos/StructSplat")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "third_party", "vggt"))
sys.path.insert(0, HERE)

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def smi_used():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL).decode()
        return float(out.strip().splitlines()[0]) / 1024.0
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--frames", default="2,4")
    ap.add_argument("--resize", nargs=2, type=int, default=[392, 518])
    ap.add_argument("--config", default=os.path.join(REPO, "config", "dl3dv.yaml"))
    ap.add_argument("--render-size", type=int, default=512)
    ap.add_argument("--out", default=None, help="结果 json 落点")
    args = ap.parse_args()

    os.chdir(REPO)          # config 内相对路径按 cwd 解析，见 README §4.1c
    inf = load_mod("inf_bench", os.path.join(HERE, "infer.py"))
    new_h, new_w = args.resize
    frames_list = [int(x) for x in args.frames.split(",") if x.strip()]

    dev = torch.device("cuda")
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {torch.cuda.get_device_name(0)}  {total:.1f} GiB  驱动可见")

    from src.config import load_configs
    from src.model.gaussian_wrapper import build_gaussian_wrapper

    cfg = load_configs(args.config, [])
    ckpt = cfg.gaussian_evaluation_stage.ckpt
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(REPO, ckpt)
    if not os.path.exists(ckpt):
        raise SystemExit(f"❌ ckpt 不存在: {ckpt}")

    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    param_dict = torch.load(ckpt, map_location="cpu")
    module_dict = {"gaussian_predictor": {}}
    for k, v in param_dict.items():
        if k.startswith("gaussian_predictor."):
            module_dict["gaussian_predictor"][k[len("gaussian_predictor."):]] = v
    del param_dict
    gc.collect()
    t_load = time.time() - t

    t = time.time()
    wrapper = build_gaussian_wrapper(module_dict, cfg).to(dev).eval()
    for p in wrapper.parameters():
        p.requires_grad = False
    torch.cuda.synchronize()
    t_build = time.time() - t
    n_param = sum(p.numel() for p in wrapper.parameters())
    print(f"模型就位 {n_param/1e6:.1f} M 参数  读盘 {t_load:.1f}s  构建 {t_build:.1f}s")
    print(f"权重常驻显存 {torch.cuda.memory_allocated()/1024**3:.2f} GiB")

    from gsplat import rasterization
    from src.utils.utils_3d import pose_encoding_to_extri, pose_encoding_to_intri

    all_paths = inf.collect_images(args.images, max_images=max(frames_list))
    rows = []
    print(f"\n{'图数':>4} {'前馈':>8} {'峰值':>8} {'smi':>7} {'光栅':>7} {'高斯数':>10}")
    print("-" * 56)
    for S in frames_list:
        if S > len(all_paths):
            print(f"{S:>4}  跳过（仅 {len(all_paths)} 张）")
            continue
        torch.cuda.empty_cache()
        gc.collect()
        imgs = [inf.load_image_tensor(p, new_h, new_w)[0] for p in all_paths[:S]]
        src = torch.stack(imgs).unsqueeze(0).to(dev)

        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            t = time.time()
            preds = wrapper(src)
            torch.cuda.synchronize()
            fb = time.time() - t
        peak_fb = torch.cuda.max_memory_allocated() / 1024**3
        used_fb = smi_used()

        means = preds["coordinate"][0].float()
        quats = preds["rotation"][0].float()
        scales = preds["scale"][0].float()
        opac = preds["opacity"][0].float()
        colors = preds["color"][0].float()
        N = int(means.shape[0])
        w2c = pose_encoding_to_extri(preds["camera"])[0].to(dev).float()
        K = pose_encoding_to_intri(preds["camera"], (new_h, new_w))[0].to(dev).float()

        s = args.render_size / new_w
        rw = args.render_size
        rh = int(round(rw * new_h / new_w))
        K = K.clone()
        K[:, 0, :] *= s
        K[:, 1, :] *= s

        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            t = time.time()
            rasterization(means=means, quats=quats, scales=scales, opacities=opac,
                          colors=colors, viewmats=w2c, Ks=K,
                          width=rw, height=rh, render_mode="RGB", packed=False,
                          backgrounds=torch.zeros((S, 3), dtype=torch.float32, device=dev))
            torch.cuda.synchronize()
            ras = time.time() - t
        peak_ras = torch.cuda.max_memory_allocated() / 1024**3

        print(f"{S:>4} {fb:>7.2f}s {peak_fb:>7.2f}G {used_fb:>6.2f}G "
              f"{ras:>6.2f}s {N:>10d}")
        rows.append({
            "images": S, "forward_ms": round(fb * 1000, 1),
            "peak_forward_alloc_gib": round(peak_fb, 3),
            "smi_used_gib": round(float(used_fb), 3),
            "raster_ms": round(ras * 1000, 1),
            "peak_raster_alloc_gib": round(peak_ras, 3),
            "n_gaussians": N,
        })
        del preds, src, imgs, means, quats, scales, opac, colors
        gc.collect()
        torch.cuda.empty_cache()

    print("-" * 56)
    print(f"权重 {n_param/1e6:.1f}M  读盘 {t_load:.1f}s  构建 {t_build:.1f}s  "
          f"卡 {total:.1f}GiB")
    res = {
        "gpu": torch.cuda.get_device_name(0),
        "vram_total_gib": round(total, 2),
        "params_m": round(n_param / 1e6, 1),
        "load_s": round(t_load, 2),
        "build_s": round(t_build, 2),
        "resize": [new_h, new_w],
        "render": [int(round(args.render_size * new_h / new_w)), args.render_size],
        "rows": rows,
        "note": "peak_* = torch.cuda.max_memory_allocated（分配器侧）；"
                "smi_used = nvidia-smi 实测占用（含上下文与其它进程）",
    }
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(res, open(args.out, "w", encoding="utf-8", newline="\n"),
                  indent=2, ensure_ascii=False)
        print(f"💾 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
