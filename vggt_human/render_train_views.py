#!/usr/bin/env python3
"""render_train_views.py — 渲染训练视角，按原始文件名保存（供 face_metrics 对比）。

官方 render.py 用 {0:05d}.png 编号，无法按 stem 与原图/mask 对应。
本脚本用 view.image_name 保存，确保 face_metrics.py 能按文件名匹配。

用法:
  GS_DIR=~/repos/gaussian-splatting python render_train_views.py \
      -s $SOURCE_DIR -m $MODEL_DIR --iteration 30000 --out_dir $OUT_DIR
"""
import os
import sys

_GS_DIR = os.environ.get("GS_DIR") or os.path.expanduser("~/repos/gaussian-splatting")
if _GS_DIR not in sys.path:
    sys.path.insert(0, _GS_DIR)

import torch
import torchvision
from argparse import ArgumentParser
from PIL import Image as PILImage
import numpy as np

from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams
from utils.general_utils import safe_state

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False


def main():
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=False)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--out_dir", default="", help="output dir (default: $model_path/train_renders)")
    parser.add_argument("--quiet", action="store_true")
    args, _ = parser.parse_known_args()
    print(f"Rendering train views: {args.model_path} @ iter={args.iteration}")

    safe_state(args.quiet)

    gaussians = GaussianModel(model.extract(args).sh_degree)
    scene = Scene(model.extract(args), gaussians, load_iteration=args.iteration, shuffle=False)

    bg = [1, 1, 1] if model.extract(args).white_background else [0, 0, 0]
    background = torch.tensor(bg, dtype=torch.float32, device="cuda")

    out_dir = args.out_dir or os.path.join(args.model_path, "train_renders")
    os.makedirs(out_dir, exist_ok=True)

    cams = scene.getTrainCameras()
    print(f"  {len(cams)} train cameras, saving to {out_dir}")
    with torch.no_grad():
        for i, view in enumerate(cams):
            rendering = render(view, gaussians, pipeline.extract(args), background,
                               use_trained_exp=model.extract(args).train_test_exp,
                               separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
            # rendering: (3, H, W) float [0,1]
            arr = (rendering.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            name = view.image_name
            # 确保扩展名是 .png（face_metrics 按 stem 匹配）
            stem = os.path.splitext(name)[0]
            PILImage.fromarray(arr).save(os.path.join(out_dir, stem + ".png"))
            if (i + 1) % 20 == 0 or i == 0:
                print(f"  [{i+1}/{len(cams)}] {stem}.png  shape={arr.shape}")

    print(f"  done → {out_dir}")


if __name__ == "__main__":
    main()
