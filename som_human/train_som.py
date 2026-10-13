#!/usr/bin/env python3
"""train_som.py — D-3DGS 训练，形变 MLP 替换为 SoM 式低秩 SE(3) 运动基（fork 官方 train.py）。

与官方 train.py 的差异（全部用「SOM」注释标注）：
  1. DeformModel -> SoMMotionModel（som_motion.py）：B 条逐帧 SE(3) 基凸组合 +
     空间 MLP 系数场。持久身份由 canonical 高斯保证，跨帧一致性由共享基保证
     —— 这是治手部重影/撕裂（成因③）的核心改动。
  2. deform.step 多传 rotations：刚体 warp 同时旋转高斯朝向（q' = q_R ⊗ q），
     官方只学加性 d_rotation 位移。
  3. loss 增加基加速度平滑（权重 --basis_smooth_weight，默认 0.1，对应 SoM 0.1）。
  4. 帧数 T = train+test 相机数（COLMAP 场景 fid = image_idx/(num_frames-1)），
     基是 [T, B] 的逐帧参数表。
  5. 去掉 ast_noise 时间抖动：它对连续时间 MLP 是正则，对查表式基无意义
     （round 后落回同一帧索引）。
  6. 权重存 model_path/som_motion/iteration_N/som.pth（不复用 deform/ 目录，
     避免与 vanilla 产物混淆）。
  7. SOM-RES：分阶段分辨率调度（res_scheduler.py，默认 "0.3:4,0.7:2"）——
     前段 1/4 → 中段 1/2 → 末段全分辨率，省 20~40% 训练时间。
     置 SOM_RES_SCHEDULE="" 关闭。

用法（由 02_train.sh 调用，CLI 参数与官方 train.py 全兼容；SOM 超参走 env，
与仓内 train_pose_refine.py 的 POSE_REFINE_* 同一惯例）：
  cd $DG_DIR && SOM_NUM_BASES=10 SOM_SMOOTH_WEIGHT=0.1 \
      python train_som.py -s <scene> -m <out> [官方参数...]
"""
import os
import sys

# 脚本在 media_code/som_human/，而 scene/utils/arguments 在官方仓。
# cwd 是 $DG_DIR（02 脚本保证），但 sys.path[0] 是脚本目录 → 显式补 $DG_DIR。
_DG_DIR = os.environ.get("DG_DIR")
if not _DG_DIR or not os.path.isfile(os.path.join(_DG_DIR, "train.py")):
    sys.exit("❌ DG_DIR 未指向 Deformable-3D-Gaussians 仓（由 02_train.sh 注入）")
sys.path.insert(0, _DG_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # som_motion 同目录

import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene, GaussianModel
from utils.general_utils import safe_state

# SOM：运动基模型替换形变 MLP
from som_motion import SoMMotionModel
# SOM-RES：分阶段分辨率调度
from res_scheduler import ResolutionScheduler

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# SOM：新增超参（由命令行注入，见 main）
SOM_NUM_BASES = int(os.environ.get("SOM_NUM_BASES", "10"))
SOM_SMOOTH_WEIGHT = float(os.environ.get("SOM_SMOOTH_WEIGHT", "0.1"))
SOM_BASIS_LR = float(os.environ.get("SOM_BASIS_LR", "1.6e-4"))
SOM_COEF_LR = float(os.environ.get("SOM_COEF_LR", "1e-2"))
# SOM-RES：分阶段分辨率（空串=关闭）；稠密化阈值补偿默认开（量纲推导，见 res_scheduler.py）
SOM_RES_SCHEDULE = os.environ.get("SOM_RES_SCHEDULE", "0.3:4,0.7:2")
SOM_RES_COMPENSATE_DENSIFY = os.environ.get("SOM_RES_COMPENSATE_DENSIFY", "1") == "1"


def training(dataset, opt, pipe, testing_iterations, saving_iterations):
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)

    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    # SOM：帧数 = 全部注册相机数（COLMAP 场景 fid = int(name)/(num_frames-1)，
    # num_frames 就是 len(cam_extrinsics)，即 train+test 之和）
    num_frames = len(scene.getTrainCameras()) + len(scene.getTestCameras())
    deform = SoMMotionModel(num_frames=num_frames, num_bases=SOM_NUM_BASES)
    deform.train_setting(opt, basis_lr=SOM_BASIS_LR, coef_lr=SOM_COEF_LR)
    print("[SOM] num_frames={}, num_bases={}, basis_lr={}, coef_lr={}, smooth_w={}".format(
        num_frames, SOM_NUM_BASES, SOM_BASIS_LR, SOM_COEF_LR, SOM_SMOOTH_WEIGHT))

    # SOM-RES：分阶段分辨率调度（就地改相机 image_height/width + 预缓存低分辨率 GT；
    # 评测走 original_image 全分辨率，指标与基线可比）
    res_sched = ResolutionScheduler(SOM_RES_SCHEDULE, opt.iterations)
    res_sched.prepare(scene.getTrainCameras())
    if res_sched.enabled:
        print("[RES] schedule={} compensate_densify={}".format(
            SOM_RES_SCHEDULE, SOM_RES_COMPENSATE_DENSIFY))

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")
    for iteration in range(1, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.do_shs_python, pipe.do_cov_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
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

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
        fid = viewpoint_cam.fid

        # SOM-RES：按 iteration 门控渲染分辨率（rasterizer 每步读相机
        # image_height/width；投影矩阵与分辨率无关，故就地改属性是安全的）
        cur_scale = res_sched.scale(iteration) if res_sched.enabled else 1
        if res_sched.enabled:
            res_sched.log_transition(iteration, cur_scale,
                                     viewpoint_cam.image_height, viewpoint_cam.image_width)
            res_sched.apply(viewpoint_cam, cur_scale)

        if iteration < opt.warm_up:
            d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
        else:
            N = gaussians.get_xyz.shape[0]
            time_input = fid.unsqueeze(0).expand(N, -1)
            # SOM：刚体 warp 需要原始朝向以计算 d_rotation = q_R⊗q - q；
            # 不加 ast_noise（查表式基 round 后噪声无效，见 docstring 第 5 条）
            d_xyz, d_rotation, d_scaling = deform.step(
                gaussians.get_xyz.detach(), time_input,
                rotations=gaussians.get_rotation.detach())

        # Render
        render_pkg_re = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re[
            "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]
        # SOM-RES：渲染完立即还原全分辨率（training_report / cameras 导出都拿全分辨率）
        if res_sched.enabled:
            res_sched.restore(viewpoint_cam)

        # Loss
        # SOM-RES：GT 从对应分辨率的缓存取（全分辨率时就是 original_image 本身）
        if res_sched.enabled:
            gt_image = res_sched.gt_image(viewpoint_cam, cur_scale).cuda()
        else:
            gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        # SOM：基加速度时间平滑（warm_up 后基才开始动，故只在之后计）
        if iteration >= opt.warm_up and SOM_SMOOTH_WEIGHT > 0:
            loss = loss + SOM_SMOOTH_WEIGHT * deform.smooth_loss()
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
                # SOM：存 som_motion/ 而不是 deform/（官方这里引用全局 args 是 bug 习惯，
                # 沿用 train_pose_refine.py 的修正：用 dataset.model_path）
                deform.save_weights(dataset.model_path, iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                viewspace_point_tensor_densify = render_pkg_re["viewspace_points_densify"]
                gaussians.add_densification_stats(viewspace_point_tensor_densify, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    # SOM-RES：低分辨率下 screenspace 梯度模长 ≈1/scale，阈值同步除 scale
                    # 保持稠密化判定不变（量纲推导启发式，见 res_scheduler.py）
                    dens_thr = opt.densify_grad_threshold
                    if res_sched.enabled and SOM_RES_COMPENSATE_DENSIFY:
                        dens_thr = res_sched.densify_threshold(dens_thr, cur_scale)
                    gaussians.densify_and_prune(dens_thr, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration)
                deform.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                deform.optimizer.zero_grad()
                deform.update_learning_rate(iteration)

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
                    # SOM：评估同样带 rotations（与训练渲染口径一致）
                    d_xyz, d_rotation, d_scaling = deform.step(
                        xyz.detach(), time_input,
                        rotations=scene.gaussians.get_rotation.detach())
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
    parser = ArgumentParser(description="Training script parameters (+ SoM motion bases)")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[5000, 6000, 7_000] + list(range(10000, 40001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000, 20_000, 30_000, 40000])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations)

    print("\nTraining complete.")
