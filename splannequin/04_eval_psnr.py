#!/usr/bin/env python3
"""04_eval_psnr.py — 复刻 train.py 训练内 test 评测（PSNR/L1），独立于训练运行。

用途：02_train.sh 可用 --test_iterations 999999 关掉训练内周期评测省时间，
但训练完就没有 test PSNR。本脚本加载最终模型 + 同一场景，对全部测试相机
渲染并计算 PSNR/L1，写 JSON 到模型目录。

Env vars:
  MODEL_DIR_PATH  模型目录（含 point_cloud/iteration_*/ 与 cfg_args.json）
  RESOLUTION      覆盖评测加载分辨率（默认沿用 cfg_args.json 的 resolution，
                  即"训练分辨率口径"；=1 强制全分辨率渲染，用于跨分辨率公平对比）
  ITERATION       加载的迭代号（默认 -1 = 最新）
  CONFIGS         超参配置（默认 ./arguments/splannequin.py，相对官方仓根）
输出: <MODEL_DIR_PATH>/eval/test_psnr_r<RESOLUTION>.json
"""
import json
import os
import sys
from argparse import ArgumentParser

SPLANNEQUIN_DIR = os.environ.get("SPLANNEQUIN_DIR", os.getcwd())
sys.path.insert(0, SPLANNEQUIN_DIR)

import torch  # noqa: E402
from scene import Scene  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from arguments import ModelParams, PipelineParams, ModelHiddenParams, get_combined_args  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402
from utils.loss_utils import l1_loss  # noqa: E402
from utils.image_utils import psnr  # noqa: E402


def main():
    parser = ArgumentParser(description="eval test PSNR/L1")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=int(os.environ.get("ITERATION", "-1")), type=int)
    parser.add_argument("--configs", default=os.environ.get("CONFIGS", "./arguments/splannequin.py"), type=str)
    parser.add_argument("--quiet", action="store_true")
    # ModelParams 的 --model_path 从环境注入（get_combined_args 要靠它找 cfg_args.json）
    if os.environ.get("MODEL_DIR_PATH") and "--model_path" not in sys.argv and "-m" not in sys.argv:
        sys.argv += ["--model_path", os.environ["MODEL_DIR_PATH"]]
    args = get_combined_args(parser)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        args = merge_hparams(args, mmcv.Config.fromfile(args.configs))

    # 分辨率覆盖：默认沿用模型训练时的 resolution；显式 RESOLUTION 强制口径
    res_env = os.environ.get("RESOLUTION", "")
    if res_env:
        args.resolution = int(res_env)
    eval_res = getattr(args, "resolution", 1)
    print("🚀 [04] eval test PSNR | model={} | resolution={}".format(args.model_path, eval_res))

    safe_state(args.quiet)
    dataset = model.extract(args)
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam.extract(args))
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
        cam_type = scene.dataset_type
        bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                          dtype=torch.float32, device="cuda")
        pipe = pipeline.extract(args)

        test_cams = scene.getTestCameras()
        n = len(test_cams)
        results = []
        l1_sum, psnr_sum = 0.0, 0.0
        for idx in range(n):
            viewpoint = test_cams[idx]
            image = torch.clamp(
                render(viewpoint, gaussians, pipe, bg, stage="fine", cam_type=cam_type)["render"],
                0.0, 1.0)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            if image.shape != gt_image.shape:
                sys.exit("❌ render {} 与 GT {} 尺寸不一致".format(tuple(image.shape), tuple(gt_image.shape)))
            l1 = l1_loss(image, gt_image).mean().double().item()
            p = psnr(image, gt_image, mask=None).mean().double().item()
            l1_sum += l1
            psnr_sum += p
            results.append({"idx": idx, "name": viewpoint.image_name, "l1": l1, "psnr": p})
            print("  📐 [{}/{}] {} PSNR={:.3f}".format(idx + 1, n, viewpoint.image_name, p))

        out = {
            "model_path": args.model_path,
            "iteration": scene.loaded_iter,
            "resolution": eval_res,
            "num_test_cams": n,
            "mean_psnr": psnr_sum / n,
            "mean_l1": l1_sum / n,
            "min_psnr": min(r["psnr"] for r in results),
            "max_psnr": max(r["psnr"] for r in results),
            "per_cam": results,
        }
        out_dir = os.path.join(args.model_path, "eval")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "test_psnr_r{}.json".format(eval_res))
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print("✅ mean PSNR={:.3f} (min {:.3f} / max {:.3f}), L1={:.5f}, {} cams".format(
            out["mean_psnr"], out["min_psnr"], out["max_psnr"], out["mean_l1"], n))
        print("💾 saved:", out_path)


if __name__ == "__main__":
    main()
