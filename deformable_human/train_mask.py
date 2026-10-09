#!/usr/bin/env python3
"""train_mask.py — Deformable-GS 训练 + 前景 mask 加权 loss（fork 官方 train.py）。

动机（2026-10-09 设计，峰哥拍板）：
  人物只占画面一部分，L1/DSSIM 全图平均 → 大部分梯度被「本来就很好拟合」的
  静态背景拿走，手部等快速运动区域监督被稀释（hand_motion 手部残影的直接成因）。
  用 SAM3 前景 mask（人+持有物，01d_fg_masks.sh 生成）把像素级 L1 权重
  向 fg 倾斜；稠密化是梯度驱动的，loss 偏向 fg 后新高斯自然往人身上长，
  不需要改点云初始化。

与官方 train.py 的差异（全部用「MASK-LOSS」注释标注）：
  1. 新增 --mask_dir/--fg_weight/--bg_weight 三个参数。
  2. MaskWeightCache：按 viewpoint_cam.image_name 懒加载 {mask_dir}/{stem}.png，
     双线性 resize 到 gt 尺寸（mask 按原图分辨率存，相机侧 -r 降采样与它无关）。
     缺失的帧回退全 1 权重并只警告一次。
  3. L1 改为加权平均：sum(W*|d|)/（W 的和按通道广播），尺度与普通 L1 相当，
     不改变优化器的 LR 语义。DSSIM 保持全图不加权——窗口化指标加权后
     标定会变，mask 先验由 L1 承载即可。
  4. 评估（training_report）口径不变：test PSNR 仍是全图无加权，
     与 vanilla 基线直接可比。

与 train_pose_refine.py 的关系：两个独立 fork，02_train.sh 里互斥
（USE_MASK_LOSS 与 USE_POSE_REFINE 同时开会直接报错，需要时再合并）。

用法（由 02_train.sh USE_MASK_LOSS=1 调用，其余参数与官方 train.py 全兼容）：
  cd $DG_DIR && python train_mask.py -s <scene> -m <out> \
      --mask_dir <scene>/masks --fg_weight 1.0 --bg_weight 0.2 [官方参数...]
"""
import os
import sys

# 脚本在 media_code/deformable_human/，而 scene/utils/arguments 在官方仓。
# cwd 是 $DG_DIR（02 脚本保证），但 sys.path[0] 是脚本目录 → 显式补 $DG_DIR。
_DG_DIR = os.environ.get("DG_DIR")
if not _DG_DIR or not os.path.isfile(os.path.join(_DG_DIR, "train.py")):
    sys.exit("❌ DG_DIR 未指向 Deformable-3D-Gaussians 仓（由 02_train.sh 注入）")
sys.path.insert(0, _DG_DIR)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import network_gui, render
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene, GaussianModel, DeformModel
from utils.general_utils import safe_state, get_linear_noise_func

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


# ─── MASK-LOSS ───────────────────────────────────────────────────────────────
class MaskWeightCache:
    """按 image_name 懒加载前景软 mask → 逐像素 loss 权重图（在原生分辨率缓存，
    每次取用双线性 resize 到 gt 尺寸；weight = bg + (fg-bg)*alpha）。"""

    def __init__(self, mask_dir, fg_weight, bg_weight):
        self.mask_dir = mask_dir
        self.fg = float(fg_weight)
        self.bg = float(bg_weight)
        self.cache = {}
        self.missing_warned = False
        n = len([f for f in os.listdir(mask_dir) if f.endswith(".png")]) \
            if os.path.isdir(mask_dir) else 0
        print("[MASK-LOSS] mask_dir=%s (%d png), fg=%.2f bg=%.2f"
              % (mask_dir, n, self.fg, self.bg))
        if n == 0:
            sys.exit("❌ mask_dir 里没有 png（先跑 01d_fg_masks.sh）: " + mask_dir)

    def get(self, image_name, ref_h, ref_w, device):
        if image_name not in self.cache:
            path = os.path.join(self.mask_dir, image_name + ".png")
            if os.path.isfile(path):
                m = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
            else:
                if not self.missing_warned:
                    print("[MASK-LOSS] WARN: 缺 mask（回退全 1 权重），"
                          "例如 %s；后续缺失不再重复警告" % path)
                    self.missing_warned = True
                m = np.ones((ref_h, ref_w), dtype=np.float32)
            self.cache[image_name] = torch.from_numpy(m)[None, None]  # (1,1,Hm,Wm)
        t = self.cache[image_name]
        if t.shape[-2] != ref_h or t.shape[-1] != ref_w:
            t = F.interpolate(t, size=(ref_h, ref_w), mode="bilinear",
                              align_corners=False)
        w = self.bg + (self.fg - self.bg) * t[0, 0]  # (H,W)
        return w.to(device)


def training(dataset, opt, pipe, testing_iterations, saving_iterations, args):
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    deform = DeformModel(dataset.is_blender, dataset.is_6dof)
    deform.train_setting(opt)

    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    # MASK-LOSS：前景权重缓存（mask 在 01d 生成，soft alpha 直接进权重）
    mask_weighter = MaskWeightCache(args.mask_dir, args.fg_weight, args.bg_weight)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")
    smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15,
                                        lr_delay_mult=0.01, max_steps=20000)
    for iteration in range(1, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.do_shs_python, pipe.do_cov_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background,
                                       0.0, 0.0, 0.0, dataset.is_6dof,
                                       scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                               0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        total_frame = len(viewpoint_stack)
        time_interval = 1 / total_frame

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
        fid = viewpoint_cam.fid

        if iteration < opt.warm_up:
            d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
        else:
            N = gaussians.get_xyz.shape[0]
            time_input = fid.unsqueeze(0).expand(N, -1)

            ast_noise = 0 if dataset.is_blender else torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * smooth_term(iteration)
            d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), time_input + ast_noise)

        # Render
        render_pkg_re = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re[
            "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        # MASK-LOSS：逐像素加权 L1（fg 权重大），DSSIM 保持全图口径
        W = mask_weighter.get(viewpoint_cam.image_name,
                              gt_image.shape[1], gt_image.shape[2], gt_image.device)
        Ll1 = (W.unsqueeze(0) * (image - gt_image).abs()).sum() / (3.0 * W.sum() + 1e-8)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()

        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                 radii[visibility_filter])

            # Log and save
            cur_psnr = training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                       testing_iterations, scene, render, (pipe, background), deform,
                                       dataset.load2gpu_on_the_fly, dataset.is_6dof)
            if iteration in testing_iterations:
                if cur_psnr.item() > best_psnr:
                    best_psnr = cur_psnr.item()
                    best_iteration = iteration

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                # 官方这里引用全局 args（bug 习惯），fork 改用 dataset.model_path
                deform.save_weights(dataset.model_path, iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                viewspace_point_tensor_densify = render_pkg_re["viewspace_points_densify"]
                gaussians.add_densification_stats(viewspace_point_tensor_densify, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration)
                deform.optimizer.step()
                deform.update_learning_rate(iteration)
                gaussians.optimizer.zero_grad(set_to_none=True)
                deform.optimizer.zero_grad()

    print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, deform, load2gpu_on_the_fly, is_6dof=False):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    test_psnr = 0.0
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                images = torch.tensor([], device="cuda")
                gts = torch.tensor([], device="cuda")
                for idx, viewpoint in enumerate(config['cameras']):
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()
                    fid = viewpoint.fid
                    xyz = scene.gaussians.get_xyz
                    time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                    d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs, d_xyz, d_rotation, d_scaling, is_6dof)["render"],
                        0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    images = torch.cat((images, image.unsqueeze(0)), dim=0)
                    gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)

                    if load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)

                l1_test = l1_loss(images, gts)
                psnr_test = psnr(images, gts).mean()
                if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                    test_psnr = psnr_test
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    return test_psnr


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters (+ mask weighted loss)")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    # MASK-LOSS
    parser.add_argument("--mask_dir", type=str, required=True,
                        help="01d 生成的前景软 mask 目录（{stem}.png，0-255）")
    parser.add_argument("--fg_weight", type=float, default=1.0)
    parser.add_argument("--bg_weight", type=float, default=0.2)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[5000, 6000, 7_000] + list(range(10000, 40001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000, 20_000, 30_000, 40000])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args)

    print("\nTraining complete.")
