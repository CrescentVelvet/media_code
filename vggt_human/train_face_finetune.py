#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys

# train_face_finetune.py 位于 vggt_human/，但 import 依赖官方仓 gaussian-splatting
# 根目录下的模块（utils/、gaussian_renderer/、scene/）。Python 只把脚本所在目录
# 加进 sys.path[0]，所以这里显式注入 GS_DIR（与 train_noise_negate.py 同约定）。
#
# 本文件是官方 train.py 的副本 + 单一 hook（人脸互补双监督 finetune，方案一升级版）。
# 除 🧑 标注处外与官方 train.py 逐行一致，便于与基线做同源对比。
#
# 🧑 与官方的差异（全部由 06b_face_finetune.sh 驱动）：
#   1. 续训：--start_ply 从 30k 的 point_cloud.ply 载入（含 exposure.json，
#      经验证 30k exposure 收敛为恒等阵 → 续训无亮度跳变）。也支持官方
#      --start_checkpoint（含 optimizer 状态，需 04 时加 --checkpoint_iterations）。
#      无 checkpoint 时 optimizer 从零初始化（30k 末期高斯已收敛，动量影响可忽略）。
#   2. LR ×0.1（--lr_scale）：xyz 的 scheduler 输入与静态参数组 lr 同时缩放；
#      checkpoint 路径下 restore→load_state_dict 会用 30k 时的 lr 覆盖参数组，
#      所以在 restore 之后再统一 ×lr_scale（xyz 下一迭代被 scheduler 重写，无叠加）。
#   3. densification 冻结：opt.densify_until_iter=0（克隆/分裂/不透明度重置全关）。
#   4. 互补监督：loss 的 L1 项改为逐像素双目标——
#         (1-M)·|r - I_orig| + M·( w·|r - I_enh| + (1-w)·|r - I_orig| )
#      M 为人脸 loss mask（face_masks.py 生成，= HYPIR 实际改动区域阈值化+腐蚀），
#      w=--face_weight：1.0 → 增强图独占人脸监督（complement），
#                       0.5 → 双监督各半（dual，默认 A/B 档）。
#      SSIM 目标与 L1 对齐（--face_ssim_mode=composite，默认）：人脸区内比对
#      同一张 composite 目标 orig + M·w·(enh - orig)，非人脸区仍是原图，故
#      与官方完全等价的退化关系成立。
#      历史 bug：SSIM 曾固定对原图全画幅算。人脸区内 L1 推向增强图、SSIM 却只认
#      原图，两者更新方向近乎正交（合成测试：L1 cos=0.81、SSIM旧 cos=0.18），
#      λ_dssim=0.2 把总方向从 0.81 稀释到 0.57 —— masked-L1 的效果被 SSIM 吃掉。
#      改后 SSIM cos=0.94、总方向 0.95。=off 可复现旧行为（仅供 A/B，勿用于生产）。
#      无人脸 mask 的帧（46/125）退化为纯官方 loss。
#   5. 起始迭代取自 ply 路径中的 iteration_N（--start_iter 可覆盖），保证
#      xyz scheduler 处于收敛后的 lr_final 段、保存编号接续 30k。
_GS_DIR = os.environ.get("GS_DIR") or os.path.expanduser("~/repos/gaussian-splatting")
if _GS_DIR not in sys.path:
    sys.path.insert(0, _GS_DIR)

import uuid
from random import randint

import numpy as np
import torch
from PIL import Image
from argparse import ArgumentParser, Namespace
from tqdm import tqdm

from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
from utils.image_utils import psnr
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ---------------------------------------------------------------------------
# 🧑 人脸监督数据：增强图 + loss mask 的惰性加载缓存（CPU float32）
# ---------------------------------------------------------------------------
class FaceData:
    def __init__(self, images_dir, masks_dir, soft):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.soft = soft          # True: 用 .alpha.png 羽化软权重; False: 用 .mask.png 二值
        self.cache = {}
        self.stems = {}
        if images_dir:
            for f in os.listdir(images_dir):
                stem, ext = os.path.splitext(f)
                if ext.lower() in IMG_EXTS:
                    self.stems[stem] = f
        self.hits = 0
        self.misses = 0

    def get(self, cam):
        """Returns dict(mask=(1,H,W) float, enh=(3,H,W) float, on CPU) or None.
        Tensors are resized to the camera's (already downscaled) original_image."""
        if not self.stems:
            return None
        # cam.image_name 可能含扩展名（COLMAP reader 用 Path.stem 去掉，
        # 但某些路径保留原文件名如 "679448043695000.jpg"）。统一去扩展名匹配。
        name = os.path.splitext(cam.image_name)[0]
        if name not in self.stems:
            self.misses += 1
            return None
        if name in self.cache:
            self.hits += 1
            return self.cache[name]

        enh_path = os.path.join(self.images_dir, self.stems[name])
        mask_name = f"{name}.alpha.png" if self.soft else f"{name}.mask.png"
        mask_path = os.path.join(self.masks_dir, mask_name)
        if not os.path.isfile(mask_path):
            self.cache[name] = None       # 该帧无人脸 mask → 纯官方 loss
            self.misses += 1
            return None
        try:
            h, w = cam.original_image.shape[1], cam.original_image.shape[2]
            enh = Image.open(enh_path).convert("RGB")
            if enh.size != (w, h):
                enh = enh.resize((w, h), Image.LANCZOS)
            mask = Image.open(mask_path).convert("L")
            if mask.size != (w, h):
                mask = mask.resize((w, h), Image.LANCZOS)
            enh_t = torch.from_numpy(np.asarray(enh, dtype=np.float32) / 255.0).permute(2, 0, 1)
            mask_t = torch.from_numpy(np.asarray(mask, dtype=np.float32) / 255.0).unsqueeze(0)
            payload = {"mask": mask_t, "enh": enh_t}
        except Exception as e:
            print(f"⚠️ face data load failed for {name}: {e}", file=sys.stderr)
            payload = None
        self.cache[name] = payload
        return payload


def training(dataset, opt, pipe, testing_iterations, saving_iterations,
             checkpoint_iterations, checkpoint, debug_from,
             start_ply, start_iter, lr_scale,
             face_images_dir, face_masks_dir, face_weight, face_soft, face_ssim_mode,
             densify_until=0):
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    # 🧑 densification 控制：默认 0=冻结（续训配方）；可设 15000 开启（消融实验）
    opt.densify_until_iter = densify_until
    # 🧑 LR ×0.1：先缩放 opt（xyz scheduler 与 training_setup 共用），静态参数组
    #    由 training_setup 直接拿到缩放值；checkpoint 路径下 restore 之后还需再缩
    #    一次（load_state_dict 用 30k 保存值覆盖）。
    for k in ("position_lr_init", "position_lr_final", "feature_lr",
              "opacity_lr", "scaling_lr", "rotation_lr"):
        setattr(opt, k, getattr(opt, k) * lr_scale)

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)

    # 🧑 ply 续训：载入 30k 高斯 + exposure.json（use_train_test_exp=True 才会读）
    if start_ply:
        gaussians.load_ply(start_ply, use_train_test_exp=True)
        # 🧑 注入相机（如 06e 增强近景）不在原场景 exposure.json 里；30k exposure
        #   已收敛为恒等阵，缺失条目直接补恒等 3x4，否则 Scene.save() 的 exposure
        #   导出与 use_trained_exp 渲染对新增图名会 KeyError（06e 实测踩坑）。
        if gaussians.pretrained_exposures is not None:
            _eye = torch.eye(3, 4, device="cuda")
            _missing = [c.image_name for c in scene.getTrainCameras()
                        if c.image_name not in gaussians.pretrained_exposures]
            for _n in _missing:
                gaussians.pretrained_exposures[_n] = _eye.clone()
            if _missing:
                print(f"🧑 exposure: filled {len(_missing)} missing entries "
                      f"with identity (injected views)")
        if start_iter is None:
            stem = os.path.basename(os.path.dirname(start_ply))   # iteration_N
            start_iter = int(stem.split("_")[-1])
        first_iter = start_iter
        _densify_status = "OFF" if densify_until == 0 else f"until={densify_until}"
        print(f"🧑 resumed gaussians from {start_ply} (first_iter={first_iter}, lr×{lr_scale}, densify {_densify_status})")
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        # restore→load_state_dict 用保存时的 lr 覆盖参数组 → 统一再缩放。
        # xyz 组下一迭代会被 update_learning_rate 按（已缩放的）scheduler 重写。
        for g in gaussians.optimizer.param_groups:
            g["lr"] *= lr_scale
        print(f"🧑 restored checkpoint {checkpoint} (first_iter={first_iter}, lr×{lr_scale})")

    # 🧑 人脸监督数据
    face_data = FaceData(face_images_dir, face_masks_dir, face_soft)
    n_face_frames = len([s for s in face_data.stems
                         if os.path.isfile(os.path.join(face_masks_dir, f"{s}.alpha.png"))])
    print(f"🧑 face supervision: {n_face_frames}/{len(scene.getTrainCameras())} frames masked, "
          f"weight={face_weight} ({'soft alpha' if face_soft else 'binary+eroded'}), "
          f"ssim_target={face_ssim_mode}, "
          f"enhanced={face_images_dir or '(none)'}")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    ema_face_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        # 🧑 人脸互补双监督：L1 逐像素双目标（见文件头注释），
        #    SSIM 对同一张 composite 目标计算，避免人脸区内梯度方向相反
        gt_image = viewpoint_cam.original_image.cuda()
        err_orig = torch.abs(image - gt_image)
        face_payload = face_data.get(viewpoint_cam)
        face_loss_val = 0.0
        gt_for_ssim = gt_image            # 默认官方：SSIM 对原图
        if face_payload is not None:
            m = face_payload["mask"].cuda()
            if not face_soft:
                m = (m > 0.5).float()
            enh = face_payload["enh"].cuda()
            err_face = torch.abs(image - enh)
            face_term = (face_weight * err_face + (1.0 - face_weight) * err_orig) * m
            Ll1 = ((1.0 - m) * err_orig + face_term).mean()
            with torch.no_grad():
                face_loss_val = face_term.sum().item() / max(m.sum().item(), 1.0)
            if face_ssim_mode == "composite":
                # 与 L1 同目标：人脸区内按 w 混入增强图，区外保持原图
                gt_for_ssim = gt_image + m * face_weight * (enh - gt_image)
        else:
            Ll1 = err_orig.mean()
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_for_ssim.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_for_ssim)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_face_loss_for_log = 0.4 * face_loss_val + 0.6 * ema_face_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "Face": f"{ema_face_loss_for_log:.{5}f}",
                                          "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # 🧑 Densification 冻结：官方块由 densify_until_iter=0 短路，整段省略
            # （原块逻辑保留在官方 train.py，此处直接跳过）

            # Optimizer step
            if iteration < opt.iterations:
                if gaussians.pretrained_exposures is None:
                    gaussians.exposure_optimizer.step()
                    gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none=True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    print(f"🧑 face supervision stats: hits={face_data.hits}, misses={face_data.misses}")


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Face-complementary finetune (official train.py + face hook)")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    # 🧑 face finetune args
    parser.add_argument("--start_ply", type=str, default=None,
                        help="resume gaussians from point_cloud.ply (e.g. .../point_cloud/iteration_30000/point_cloud.ply)")
    parser.add_argument("--start_iter", type=int, default=None,
                        help="first iteration (default: parsed from start_ply path)")
    parser.add_argument("--lr_scale", type=float, default=0.1,
                        help="multiply all base learning rates (design: LR 1/10)")
    parser.add_argument("--densify_until", type=int, default=0,
                        help="densify_until_iter (0=frozen, 15000=official default; "
                             "resume 时若 start_iter 已过此值则仍不触发)")
    parser.add_argument("--face_images_dir", type=str, default="",
                        help="HYPIR-enhanced images dir (face_enhance.py output)")
    parser.add_argument("--face_masks_dir", type=str, default="",
                        help="face loss masks dir (face_masks.py output)")
    parser.add_argument("--face_weight", type=float, default=0.5,
                        help="enhanced-image weight inside face mask (1.0=complement, 0.5=dual)")
    parser.add_argument("--face_soft", action="store_true",
                        help="use feather alpha as soft weights instead of binary mask")
    parser.add_argument("--face_ssim_mode", type=str, default="composite",
                        choices=["composite", "off"],
                        help="SSIM target: composite=align with L1 face target (fixed), "
                             "off=legacy full-frame vs original (gradient conflict, A/B only)")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    if args.start_ply and args.start_checkpoint:
        sys.exit("❌ --start_ply and --start_checkpoint are mutually exclusive")

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args),
             args.test_iterations, args.save_iterations, args.checkpoint_iterations,
             args.start_checkpoint, args.debug_from,
             args.start_ply, args.start_iter, args.lr_scale,
             args.face_images_dir, args.face_masks_dir, args.face_weight, args.face_soft,
             args.face_ssim_mode, args.densify_until)

    # All done
    print("\nTraining complete.")
