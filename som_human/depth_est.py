#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""depth_est.py — 单目深度推理后端（方案 E 的离线那一半）

模型：UniDepth **V1 ViT-L14**（本机已有权重，走 torch.hub 离线缓存，零网络）：
  - 权重：~/.cache/huggingface/hub/models--lpiccinelli--unidepth-v1-vitl14/snapshots/local/
  - 代码：~/.cache/torch/hub/lpiccinelli-eth_UniDepth_main（00a 里种过）
  - 运行环境：conda env `gavs`（torch 2.2.2+cu121 + xformers 0.0.25.post1，
    UniDepth V1 decoder 的 NystromBlock 硬依赖 xformers）
  → **不要在 deformable_human env 里跑**（那是 torch 2.5.1，没有钉版 xformers）。

选 UniDepth 而非 MoGe 的原因：本机已有权重（MoGe/DA-3 要等迅雷下载），
且它输出**米制**深度 + 置信度；训练侧损失用尺度/平移不变（SSI）形式，
米制与否不影响监督，但置信度可用于剔离群点。

输出：`<scene>/depths/<stem>.npz`，键：
  depth  float16 [h,w] 米制深度（模型原生分辨率，**不缩放**——训练侧按渲染分辨率重采样）
  conf   float16 [h,w] 置信度（v1 可能没有该键，则省略）
  meta   json 字符串（模型版本/backbone/原图尺寸/耗时）
`<scene>/depths/_qc/<stem>.jpg`：前 --qc 帧的伪彩可视化（供目视检查用的抽检图）

用法（由 01f_depth_maps.sh 调用；也可单独跑）：
  python depth_est.py --images <scene>/images --out <scene>/depths [--limit 4] [--force]
"""
import argparse
import json
import os
import sys
import time

# HF 离线：必须在 import huggingface_hub 之前设，否则 hf_hub_download 会去连网
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
from PIL import Image


def _hub_repo_dir():
    """torch.hub 的 UniDepth 仓代码缓存目录（00a_setup_env 里种过）。"""
    return os.path.expanduser("~/.cache/torch/hub/lpiccinelli-eth_UniDepth_main")


def _load_model(version, backbone, device):
    import torch
    t0 = time.time()
    repo_dir = _hub_repo_dir()
    if os.path.isdir(repo_dir):
        # ⚠️ 必须 source="local"：即使仓已在缓存里，torch.hub.load("owner/repo")
        # 也会先 urlopen github 的 /tree/main/ 去解析 ref（_parse_repo_info）——
        # 本机 github API/raw 不可达 → RemoteDisconnected。
        # 2026-10-16 踩过：01f 因此静默失败（脚本当时没查退出码，仍打印 Done）。
        model = torch.hub.load(repo_dir, "UniDepth", source="local",
                               version=version, backbone=backbone, pretrained=True,
                               trust_repo=True)
    else:
        model = torch.hub.load("lpiccinelli-eth/UniDepth", "UniDepth",
                               version=version, backbone=backbone, pretrained=True,
                               trust_repo=True, force_reload=False)
    model = model.to(device).eval()
    print("[depth_est] UniDepth {}/{} loaded in {:.1f}s (device={})".format(
        version, backbone, time.time() - t0, device), flush=True)
    return model


def _colorize(depth, valid=None):
    """深度伪彩（近=暖色）。只为目视抽检，不参与训练。"""
    d = depth.astype(np.float32)
    if valid is not None:
        d = np.where(valid, d, np.nan)
    lo, hi = np.nanpercentile(d, 2), np.nanpercentile(d, 98)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(d)), float(np.nanmax(d)) or 1.0
    x = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    x = np.nan_to_num(x, nan=0.0)
    # 近处亮（1 蓝 → 0 红），用简单三通道映射避免额外依赖
    r = np.clip(1.5 - 2.0 * x, 0, 1)
    g = np.clip(1.5 - np.abs(x - 0.5) * 2.6, 0, 1)
    b = np.clip(2.0 * x - 0.7, 0, 1)
    rgb = (np.stack([r, g, b], -1) * 255).astype(np.uint8)
    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="图像目录（*.png/jpg/jpeg/heic→已转 png）")
    ap.add_argument("--out", required=True, help="输出目录（.npz）")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--backbone", default="vitl14")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 张（调试用，0=全部）")
    ap.add_argument("--qc", type=int, default=6, help="输出多少张伪彩抽检图")
    ap.add_argument("--force", action="store_true", help="已存在也重算")
    args = ap.parse_args()

    import torch
    from torchvision import transforms as T

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exts = (".png", ".jpg", ".jpeg", ".webp")
    names = sorted(n for n in os.listdir(args.images) if n.lower().endswith(exts))
    if args.limit:
        names = names[:args.limit]
    if not names:
        sys.exit("❌ {} 下没有图像".format(args.images))

    os.makedirs(args.out, exist_ok=True)
    qc_dir = os.path.join(args.out, "_qc")
    os.makedirs(qc_dir, exist_ok=True)

    todo = [n for n in names
            if args.force or not os.path.exists(os.path.join(args.out, os.path.splitext(n)[0] + ".npz"))]
    print("[depth_est] {} 张图，待算 {} 张（已存在自动跳过）".format(len(names), len(todo)), flush=True)
    if not todo:
        return

    model = _load_model(args.version, args.backbone, device)

    times = []
    for i, name in enumerate(todo):
        stem = os.path.splitext(name)[0]
        path = os.path.join(args.images, name)
        with Image.open(path) as im:
            im = im.convert("RGB")
            ours = im.size  # (w,h)
            img = T.ToTensor()(im).unsqueeze(0).to(device)  # [1,3,H,W]
        t0 = time.time()
        with torch.no_grad():
            outs = model.infer(img)
        dt = time.time() - t0
        times.append(dt)

        depth = outs["depth"][0].float().cpu().numpy()  # [1,H,W] → squeeze
        depth = np.squeeze(depth)
        payload = {
            "depth": depth.astype(np.float16),
            "meta": json.dumps({"model": "unidepth-{}-{}".format(args.version, args.backbone),
                                "src_wh": list(ours), "infer_sec": round(dt, 3)}),
        }
        conf = outs.get("confidence", None)
        if conf is not None:
            payload["conf"] = np.squeeze(conf[0].float().cpu().numpy()).astype(np.float16)
        np.savez_compressed(os.path.join(args.out, stem + ".npz"), **payload)

        if i < args.qc:
            Image.fromarray(_colorize(depth)).save(os.path.join(qc_dir, stem + ".jpg"), quality=88)
        if (i + 1) % 10 == 0 or i == len(todo) - 1:
            print("  [{}/{}] {}  {:.2f}s/帧  平均 {:.2f}s".format(
                i + 1, len(todo), stem, dt, sum(times) / len(times)), flush=True)

    print("[depth_est] ✅ 完成 {} 张，平均 {:.2f}s/帧（首帧含加载除外）".format(
        len(times), sum(times) / max(len(times), 1)), flush=True)


if __name__ == "__main__":
    main()
