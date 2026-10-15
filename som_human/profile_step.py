"""profile_step.py — 训练单步时间分解（决定 C/G/稀疏 Adam 优先级用）。

加载现成 ckpt + som 权重，模拟 train_som.py 的单个训练步
（deform step → render → loss → backward → optimizer），
用 cuda sync + perf_counter 分解各段耗时，输出 mean ms / 占比 / 外推 it/s。

用法（WSL, deformable_human env）：
    cd ~/repos/Deformable-3D-Gaussians
    python /mnt/c/code/media_code/som_human/profile_step.py \
        -s <colmap_refined> --model_path <model_colmap_exp> \
        --load_iteration 20000 --resolution 2 --iters 200

不改训练循环、不碰官方仓；ckpt 只读。
"""
import os
import sys
import time
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # som_human 自身（som_motion）
sys.path.append(os.environ.get("DG_DIR", os.path.expanduser("~/repos/Deformable-3D-Gaussians")))

import torch
import torch.nn.functional as F

from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim
from som_motion import SoMMotionModel  # som_human 侧需 cwd 或 PYTHONPATH 含 som_human

SECTIONS = ["deform_step", "render", "loss", "backward", "opt_step", "zero_grad"]


def build_args():
    parser = argparse.ArgumentParser(description="profile one training step")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--load_iteration", type=int, required=True)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--num_bases", type=int, default=int(os.environ.get("SOM_NUM_BASES", "10")))
    args = parser.parse_args()
    return args, lp, pp, op


def main():
    args, lp, pp, op = build_args()
    dataset = lp.extract(args)
    pipe = pp.extract(args)
    opt = op.extract(args)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=None, shuffle=False)

    ply = os.path.join(dataset.model_path,
                       "point_cloud/iteration_{}/point_cloud.ply".format(args.load_iteration))
    gaussians.load_ply(ply)
    gaussians.training_setup(opt)

    train_views = scene.getTrainCameras()
    test_views = scene.getTestCameras()
    num_frames = len(train_views) + len(test_views)
    deform = SoMMotionModel(num_frames=num_frames, num_bases=args.num_bases)
    deform.load_weights(dataset.model_path)
    deform.train_setting(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    n_pts = gaussians.get_xyz.shape[0]
    print("[PROFILE] N_gaussians={}  frames={}  ckpt=iteration_{}"
          .format(n_pts, num_frames, args.load_iteration))

    accum = {k: 0.0 for k in SECTIONS}
    total_accum = 0.0
    n_timed = 0

    for it in range(args.warmup + args.iters):
        viewpoint = train_views[it % len(train_views)]
        N = gaussians.get_xyz.shape[0]
        fid = torch.full((N, 1), float(viewpoint.fid),
                         device="cuda")  # step 接口要求 [N,1]

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        d_xyz, d_rotation, d_scaling = deform.step(
            gaussians.get_xyz, fid, gaussians.get_rotation)

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        image = render(viewpoint, gaussians, pipe, background,
                       d_xyz, d_rotation, d_scaling, dataset.is_6dof)["render"]

        torch.cuda.synchronize()
        t2 = time.perf_counter()

        gt_image = viewpoint.original_image.to("cuda")
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        torch.cuda.synchronize()
        t3 = time.perf_counter()

        loss.backward()

        torch.cuda.synchronize()
        t4 = time.perf_counter()

        gaussians.optimizer.step()
        deform.optimizer.step()

        torch.cuda.synchronize()
        t5 = time.perf_counter()

        gaussians.optimizer.zero_grad(set_to_none=True)
        deform.optimizer.zero_grad()

        torch.cuda.synchronize()
        t6 = time.perf_counter()

        if it >= args.warmup:
            accum["deform_step"] += t1 - t0
            accum["render"] += t2 - t1
            accum["loss"] += t3 - t2
            accum["backward"] += t4 - t3
            accum["opt_step"] += t5 - t4
            accum["zero_grad"] += t6 - t5
            total_accum += t6 - t0
            n_timed += 1

    print("\n[PROFILE] iters_timed={}  (ms/step, % of total)".format(n_timed))
    print("{:<12} {:>10} {:>8}".format("section", "mean_ms", "pct"))
    for k in SECTIONS:
        mean_ms = accum[k] / n_timed * 1000.0
        pct = accum[k] / total_accum * 100.0
        print("{:<12} {:>10.2f} {:>7.1f}%".format(k, mean_ms, pct))
    total_ms = total_accum / n_timed * 1000.0
    print("{:<12} {:>10.2f} {:>7.1f}%".format("TOTAL", total_ms, 100.0))
    print("[PROFILE] it/s = {:.2f}  (20k steps ~ {:.1f} min)".format(
        1000.0 / total_ms, 20000.0 * total_ms / 1000.0 / 60.0))
    print("[PROFILE] peak_mem = {:.1f} GB".format(torch.cuda.max_memory_allocated() / 1e9))


if __name__ == "__main__":
    main()
