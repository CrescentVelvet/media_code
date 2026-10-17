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
  7. SOM-RES：分阶段分辨率调度（res_scheduler.py，默认关闭）——
     2026-10-13 A/B/C 实测在本链路零加速（瓶颈是随高斯数 N 增长的
     非渲染开销，不是光栅化像素），阈值补偿有害。数据见 README 负结果节。
  8. SOM-EXPOSURE：方案 A 逐相机曝光仿射（exposure.py，默认关闭）——
     门控开启瞬间 lstsq 闭式初始化，独立 lr 1e-3 + 拉回恒等正则，
     test 帧按 fid 插值。针对手机 auto-exposure 逐帧光度漂移。
  9. SOM-DENSIFY：方案 D 稠密化自动调度（densify_sched.py，默认关闭）——
     显式 cap（SOM_MAX_GAUSSIANS）为默认路径，momentum 自适应仅在 cap≤0 时启用；
     target 按 absgrad 分数 topk 截断；窗口结束后只剪不增（可选）。口径与服务器
     DensificationScheduler 对齐（详见 densify_sched.py 顶部）。
     窗口后的 prune **只按 opacity**（`max_screen_size=None`）：尺寸判据
     （max_radii2D>20 / scaling>0.1·extent）在窗口内安全（每轮 prune 后紧跟
     clone/split 补点），一旦没有补点来源就是死亡螺旋。2026-10-16 三次独立
     冒烟实测：599k→463k→…→9.6k，PSNR 20.52→5.34；对照组 21.81 正常。
 10. 硬护栏语义修正（2026-10-16）：到 SOM_MAX_GAUSSIANS 后**只 prune 不 densify**
     （旧版直接跳过整个 densify_and_prune 调用，连 prune 一起停了，与注释不符）。
     ⚠️ 同一条理由：**只剪不增的调用一律不得带尺寸判据**（max_screen_size=None）。
     护栏路径同样没有 densify 补点 → max_radii2D 不清零 → 判据退化成「历史最大
     屏幕半径」。低 cap 实测（cap 120k）：带判据时 iter 3100 一步删 86,738 点
     （162k→75k，−54%），随后每 100 步在 110k↔150k 震荡；去判据后 applied 只剩
     opa 的 70~350，点数平稳。
 11. SOM-DEPTH：方案 E 单目深度先验正则（depth_prior.py，默认关闭）——
     **点级锚定**（不是渲染深度 loss：本仓 fork 光栅化器的 depth 是前向-only
     输出，backward 没有深度梯度通道）：形变后高斯中心投影采样单目深度做
     加权 SSI-L1，warm_up 后启用；fg mask 来自 01d，缺失时退化为纯 opacity 权重。

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
from collections import OrderedDict
import numpy as np
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
# SOM-RES：分阶段分辨率（默认空串=关闭：实测零加速，见 res_scheduler.py 顶部结论）
SOM_RES_SCHEDULE = os.environ.get("SOM_RES_SCHEDULE", "")
SOM_RES_COMPENSATE_DENSIFY = os.environ.get("SOM_RES_COMPENSATE_DENSIFY", "0") == "1"
# SOM：高斯数上限护栏（v2 事故后引入）。2026-10-15 cap600k A/B 后默认值 150万→60万：
# @10k 掉 0.09dB（<0.3dB 判定线）、@20k 21.81 反超无 cap 组、2.4× 省点、~1.7× 加速。
# 语义：超过即停止 densify，但**仍按周期 prune**（2026-10-16 修正；旧版连 prune 一起停）。
# 置 0 或负数 → 不设硬护栏（此时调度器自动落入 momentum 自适应路径）。
SOM_MAX_GAUSSIANS = int(os.environ.get("SOM_MAX_GAUSSIANS", "600000"))
# SOM-DENSIFY：稠密化自动调度（方案 D 移植，默认关；开后 thr/窗口不再靠人肉调）
SOM_DENSIFY_SCHED = os.environ.get("SOM_DENSIFY_SCHED", "0") == "1"
SOM_DENSIFY_MOMENTUM_MULT = os.environ.get("SOM_DENSIFY_MOMENTUM_MULT", "auto")
SOM_DENSIFY_MAX_RATE = os.environ.get("SOM_DENSIFY_MAX_RATE", "1.0")   # 服务器同名常量
SOM_DENSIFY_CNT_CLAMP = os.environ.get("SOM_DENSIFY_CNT_CLAMP", "3.0")  # 服务器 /cnt.clamp_min(3)
# densify 窗口结束后是否继续按期 prune（服务器「只剪不增」；1=开）
SOM_DENSIFY_PRUNE_AFTER = os.environ.get("SOM_DENSIFY_PRUNE_AFTER", "1") == "1"
# SOM-EXPOSURE：方案 A 逐相机曝光仿射补偿（exposure.py，默认关）。
# 门控：iteration < FROM（默认 warm_up）前 A 恒等；门控瞬间对全部训练帧
# 渲染一遍做 lstsq 闭式初始化，再以独立小 lr 联合训练；test 帧按 fid 插值。
SOM_EXPOSURE = os.environ.get("SOM_EXPOSURE", "0") == "1"
SOM_EXPOSURE_FROM_ITER = int(os.environ.get("SOM_EXPOSURE_FROM_ITER", "-1"))
SOM_EXPOSURE_LR = float(os.environ.get("SOM_EXPOSURE_LR", "1e-3"))
SOM_EXPOSURE_REG = float(os.environ.get("SOM_EXPOSURE_REG", "0.01"))
# SOM-DEPTH：方案 E 单目深度先验正则（depth_prior.py + 01f_depth_maps.sh 产物，
# 默认关）。⚠️ 不是渲染深度 loss（本仓 fork 光栅化器的 depth 无梯度回路），
# 是**点级锚定**：形变后高斯中心投影采样单目深度，加权 SSI-L1 拉几何。
# warm_up 后启用，λ 0.05 起步；fg mask 来自 01d（缺失时退化为纯 opacity 权重）。
SOM_DEPTH_LAMBDA = float(os.environ.get("SOM_DEPTH_LAMBDA", "0"))
SOM_DEPTH_FROM_ITER = int(os.environ.get("SOM_DEPTH_FROM_ITER", "-1"))  # -1 = warm_up
SOM_DEPTH_USE_MASK = int(os.environ.get("SOM_DEPTH_USE_MASK", "1"))     # 1=优先用 01d 前景 mask
SOM_DEPTH_MIN_POINTS = int(os.environ.get("SOM_DEPTH_MIN_POINTS", "2048"))  # 帧内投影点下限
# SOM-FG：单人前景监督（per-person 独立重建，2026-10-17）——
# 每个 pid 一个场景（01h 产出），loss 只在该人的 mask 内计算、背景与其他人
# 一律不产生梯度；COLMAP 初始点按多视角 mask 投票过滤，避免整场景点云里的
# 背景/他人点占容量。关闭（SOM_FG_MASK_DIR 为空）时行为与历史完全一致。
SOM_FG_MASK_DIR = os.environ.get("SOM_FG_MASK_DIR", "")
SOM_FG_INIT_FILTER = os.environ.get("SOM_FG_INIT_FILTER", "1") == "1"
SOM_FG_INIT_MIN_VIEWS = int(os.environ.get("SOM_FG_INIT_MIN_VIEWS", "3"))
SOM_FG_INIT_MIN_RATIO = float(os.environ.get("SOM_FG_INIT_MIN_RATIO", "0.3"))
SOM_FG_INIT_MAX_CAMS = int(os.environ.get("SOM_FG_INIT_MAX_CAMS", "48"))
SOM_BG_RANDOM = os.environ.get("SOM_BG_RANDOM", "1") == "1"
# SOM-HAND：手部区域 loss 加权（2026-10-17 档 0 判决实验）——
# 目的：手部只占画面 ~2%，静止区（身体/腿/桌面）主导光度梯度，运动基被
# "大家不动"的信号压倒而塌缩到恒等（见 EXPERIMENTS.md E-01）。给手部 mask
# 内的像素额外权重，让手部运动信号进入运动基的梯度。
# 权重图 w = fg * (1 + (HAND_WEIGHT-1) * hand)：手外权重 1，手内 HAND_WEIGHT。
# 面积归一化按 w.sum()（而不是 fg.sum()），保持 densify 梯度阈值语义。
# 依赖：SOM_FG_MASK_DIR 已启用；hand mask 目录约定同 FGMaskBank（<stem>.png）。
SOM_HAND_MASK_DIR = os.environ.get("SOM_HAND_MASK_DIR", "")
SOM_HAND_WEIGHT = float(os.environ.get("SOM_HAND_WEIGHT", "1.0"))


class FGMaskBank:
    """单人前景软 alpha 的 CPU 缓存（按需上 GPU）。

    目录约定：<mask_dir>/<image_name>.png，灰度 0-255 → 0-1。
    缺失文件返回 None（调用方退化为全图监督并告警一次）。
    """

    def __init__(self, mask_dir, cache_size=24):
        self.dir = mask_dir
        self._cache = OrderedDict()
        self._cache_size = int(cache_size)
        self._missing_reported = False
        self.available = False
        if os.path.isdir(mask_dir):
            self.available = any(f.endswith(".png") for f in os.listdir(mask_dir))

    def get(self, stem, hw):
        """返回 [1,1,h,w] cuda float32 软 mask；缺失返回 None。"""
        key = (stem, int(hw[0]), int(hw[1]))
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        path = os.path.join(self.dir, str(stem) + ".png")
        if not os.path.exists(path):
            if not self._missing_reported:
                print("[FG] WARN: 缺 mask {}（该帧退化为全图监督）".format(path), flush=True)
                self._missing_reported = True
            self._cache[key] = None
            return None
        from PIL import Image
        m = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        t = torch.from_numpy(m)[None, None].to("cuda")
        if tuple(t.shape[-2:]) != (int(hw[0]), int(hw[1])):
            t = torch.nn.functional.interpolate(
                t, size=(int(hw[0]), int(hw[1])), mode="bilinear", align_corners=False)
        self._cache[key] = t
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return t


def filter_init_points_by_fg(gaussians, cams, fg_bank, min_views, min_ratio, max_cams):
    """按多视角 mask 投票过滤 COLMAP 初始点（per-person 重建必需）。

    每个点投影到抽样帧：可见（z>0.2、落画幅内）且落在 fg 的帧数 ≥ min_views，
    且 落入比例 ≥ min_ratio → 保留。不过滤的话，整场景点云里的背景/他人点
    会一直留在模型里（loss 只算 fg → 它们没有梯度、不会被 prune 掉），既占
    容量也会在合成时浮在人体周围。

    必须在 gaussians.training_setup() **之前**调用（优化器要按新点数建）。
    """
    if not cams:
        return
    dev = gaussians.get_xyz.device
    xyz = gaussians.get_xyz.detach()
    n0 = xyz.shape[0]
    votes = torch.zeros(n0, device=dev)
    vis = torch.zeros(n0, device=dev)
    step = max(1, len(cams) // max(1, int(max_cams)))
    used = cams[::step]
    n_used = 0
    for cam in used:
        fg = fg_bank.get(cam.image_name, (cam.image_height, cam.image_width))
        if fg is None:
            continue
        n_used += 1
        ones = torch.ones(n0, 1, device=dev, dtype=xyz.dtype)
        pts_h = torch.cat([xyz, ones], dim=1)
        p_view = pts_h @ cam.world_view_transform
        z_view = p_view[:, 2]
        clip = pts_h @ cam.full_proj_transform
        w_clip = clip[:, 3]
        ndc = clip[:, :3] / w_clip.clamp_min(1e-6).unsqueeze(1)
        # 与 depth_prior 相同的投影约定（ndc_y>0 = 画面下半部，grid_sample 同向）
        in_f = ((w_clip > 0) & (ndc[:, 0].abs() <= 1.0) & (ndc[:, 1].abs() <= 1.0)
                & (z_view > 0.2))
        grid = torch.stack([ndc[:, 0], ndc[:, 1]], dim=1).view(1, 1, n0, 2)
        # fg 本身已是 [1,1,h,w]（4D），不能再套 [None]——那会变 5D，grid_sample
        # 会按 3D 采样解释并要求 grid 末维=3（2026-10-17 冒烟实踩）。
        a = torch.nn.functional.grid_sample(
            fg, grid, mode="bilinear", padding_mode="zeros",
            align_corners=False).view(n0)
        vis += in_f.to(torch.float32)
        votes += (in_f & (a > 0.5)).to(torch.float32)

    if n_used == 0:
        print("[FG] WARN: 没有可用 mask，跳过初始点过滤", flush=True)
        return
    ratio = votes / vis.clamp_min(1.0)
    keep = (votes >= float(min_views)) & (ratio >= float(min_ratio))
    n_keep = int(keep.sum())
    print("[FG] init filter: {} -> {} 点（{:.1f}%，{} 帧投票, min_views={}, ratio≥{:.2f}）".format(
        n0, n_keep, 100.0 * n_keep / max(n0, 1), n_used, min_views, min_ratio), flush=True)
    if n_keep < 100:
        print("[FG] WARN: 保留点 <100，判定 mask 与图像未对齐，跳过过滤（请核对 01h 场景）",
              flush=True)
        return
    for attr in ("_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation",
                 "_opacity"):
        t = getattr(gaussians, attr, None)
        if t is None or t.numel() == 0:
            continue
        setattr(gaussians, attr,
                torch.nn.Parameter(t.data[keep].clone().requires_grad_(True)))
    # ⚠️ max_radii2D 在 create_from_pcd 末尾按**原点数**建的，training_setup
    # 不会重建它 → 不同步会让第一步 `max_radii2D[visibility_filter]` 直接
    # IndexError（2026-10-17 冒烟实踩：mask[13925] vs tensor[53676]）。
    gaussians.max_radii2D = torch.zeros(int(n_keep), device=dev)


def training(dataset, opt, pipe, testing_iterations, saving_iterations):
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)

    scene = Scene(dataset, gaussians)

    # SOM-FG：单人前景监督（per-person 重建）。必须在 training_setup 之前过滤
    # 初始点——优化器是按当时的点数建的。
    fg_bank = None
    if SOM_FG_MASK_DIR:
        fg_bank = FGMaskBank(SOM_FG_MASK_DIR)
        if not fg_bank.available:
            print("[FG] WARN: {} 下没有 *.png，退化全图监督".format(SOM_FG_MASK_DIR),
                  flush=True)
            fg_bank = None
        else:
            print("[FG] 单人前景监督: dir={} bg_random={} init_filter={}".format(
                SOM_FG_MASK_DIR, SOM_BG_RANDOM, SOM_FG_INIT_FILTER), flush=True)
    # SOM-HAND：手部区域加权（仅在 FG 监督基础上有意义）
    hand_bank = None
    if SOM_HAND_MASK_DIR and SOM_HAND_WEIGHT != 1.0:
        hand_bank = FGMaskBank(SOM_HAND_MASK_DIR)
        if not hand_bank.available:
            print("[HAND] WARN: {} 下没有 *.png，不加权".format(SOM_HAND_MASK_DIR),
                  flush=True)
            hand_bank = None
        else:
            print("[HAND] 手部区域加权: dir={} weight={}x".format(
                SOM_HAND_MASK_DIR, SOM_HAND_WEIGHT), flush=True)
            if SOM_FG_INIT_FILTER:
                filter_init_points_by_fg(
                    gaussians, scene.getTrainCameras(), fg_bank,
                    SOM_FG_INIT_MIN_VIEWS, SOM_FG_INIT_MIN_RATIO,
                    SOM_FG_INIT_MAX_CAMS)

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

    # SOM-DENSIFY：稠密化自动调度器（方案 D）。init_n 在 Scene 加载 pcd 后确定。
    # 口径（2026-10-16 对齐服务器）：cap>0 = 显式上限路径（momentum 不参与，默认）；
    # cap<=0 = momentum 自适应路径。硬护栏 SOM_MAX_GAUSSIANS 与 cap 同值，互补。
    densify_sched = None
    if SOM_DENSIFY_SCHED:
        from densify_sched import DensifyScheduler
        _mult = SOM_DENSIFY_MOMENTUM_MULT
        densify_sched = DensifyScheduler(
            init_n=gaussians.get_xyz.shape[0],
            mult=(float(_mult) if _mult != "auto" else "auto"),
            cap=SOM_MAX_GAUSSIANS,
            max_rate=float(SOM_DENSIFY_MAX_RATE),
            cnt_clamp=float(SOM_DENSIFY_CNT_CLAMP))
        print("[DENSIFY-SCHED] init_n={} mult={} prune_after={} {}".format(
            densify_sched.init_n, densify_sched.mult, SOM_DENSIFY_PRUNE_AFTER,
            densify_sched.state_str()), flush=True)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    # SOM-FG：评测/存档用固定背景（训练时若开 SOM_BG_RANDOM 则每步随机，
    # 防止模型把「背景色」也学进 SH——per-person 模型输出应是纯人，不该带底）
    eval_background = background.clone()

    # SOM-EXPOSURE：方案 A 逐相机曝光仿射（exposure.py）。FROM=-1 → warm_up
    exposure = None
    exposure_from = -1
    if SOM_EXPOSURE:
        from exposure import ExposureBank
        _train_cams = scene.getTrainCameras()
        exposure = ExposureBank([c.image_name for c in _train_cams],
                                [float(c.fid) for c in _train_cams],
                                lr=SOM_EXPOSURE_LR, reg_weight=SOM_EXPOSURE_REG)
        exposure_from = (SOM_EXPOSURE_FROM_ITER if SOM_EXPOSURE_FROM_ITER > 0
                         else opt.warm_up)
        print("[EXPOSURE] enabled: {} train cams, from_iter={}, lr={}, reg={}".format(
            len(exposure.names), exposure_from, SOM_EXPOSURE_LR, SOM_EXPOSURE_REG),
            flush=True)

    # SOM-DEPTH：方案 E 单目深度先验（depth_prior.py，点级锚定）。FROM=-1 → warm_up
    depth_prior = None
    depth_from = -1
    if SOM_DEPTH_LAMBDA > 0:
        from depth_prior import DepthPrior
        depth_prior = DepthPrior(
            dataset.source_path, use_mask=SOM_DEPTH_USE_MASK,
            min_points=SOM_DEPTH_MIN_POINTS)
        # 点级锚定需要形变基已成形（warm_up 前 d_xyz=0 无梯度意义）→ 不低于 warm_up
        depth_from = max(
            SOM_DEPTH_FROM_ITER if SOM_DEPTH_FROM_ITER > 0 else opt.warm_up,
            opt.warm_up)
        print("[DEPTH] enabled: lambda={} from_iter={} mask={} min_points={} available={}".format(
            SOM_DEPTH_LAMBDA, depth_from, SOM_DEPTH_USE_MASK, SOM_DEPTH_MIN_POINTS,
            depth_prior.available), flush=True)
        if not depth_prior.available:
            print("[DEPTH] WARN: {} 下没有 depths/*.npz（先跑 01f_depth_maps.sh），"
                  "本轮深度损失恒为 0".format(dataset.source_path), flush=True)

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

        # SOM-EXPOSURE：门控开启瞬间做闭式初始化（全部训练帧渲染一遍解 lstsq）
        if exposure is not None and iteration == exposure_from and not exposure.initialized:
            exposure.closed_form_init(_exposure_render_pass(
                scene, gaussians, deform, render, pipe, background, dataset))

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
        # SOM-FG：训练时随机背景色（每步一次），让背景区域的渲染值不进入任何
        # 稳定拟合；评测用固定色（见 training_report 调用）。
        if fg_bank is not None and SOM_BG_RANDOM:
            background = torch.rand(3, device="cuda")
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
        # SOM-EXPOSURE：门控后对渲染图做逐相机仿射校正；L1 与 DSSIM 用同一张
        # 校正图（口径一致），test 帧评测时按 fid 插值（见 training_report）
        exposure_on = exposure is not None and iteration >= exposure_from
        if exposure_on:
            image = exposure.apply(viewpoint_cam.image_name, image)
        # SOM-FG：只在该人的 mask 内计算光度损失（背景 + 其他人在梯度上不存在，
        # 于是 densify 也只会往人身上加点）。
        fg = None
        if fg_bank is not None:
            fg = fg_bank.get(viewpoint_cam.image_name, gt_image.shape[-2:])
        if fg is not None:
            # SOM-HAND：手部区域加权。w = fg * (1 + (W-1)*hand)，
            # 面积归一化按 w.sum()（权重和），保持 densify 阈值语义。
            w = fg
            if hand_bank is not None:
                hand = hand_bank.get(viewpoint_cam.image_name, gt_image.shape[-2:])
                if hand is not None:
                    w = fg * (1.0 + (SOM_HAND_WEIGHT - 1.0) * hand)
            # 面积归一化：只算 fg 会让 loss 量级按人体占比缩小，而 densify 的
            # 梯度阈值是固定值 → 归一化回「全图同量级」，阈值语义不变。
            area = w.sum().clamp_min(1.0)
            inv_area = fg.numel() / area
            # 只加权 L1（像素级监督）；SSIM 的 C1/C2 常数按 [0,1] 值域设定，
            # 乘权重会破坏其口径 → SSIM 保持 fg 不加权。
            Ll1 = l1_loss(image * w, gt_image * w) * inv_area
            loss = ((1.0 - opt.lambda_dssim) * Ll1
                    + opt.lambda_dssim * (1.0 - ssim(image * fg, gt_image * fg)))
        else:
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        # SOM-EXPOSURE：拉回恒等正则（防与 SH 颜色互相吸收震荡）
        if exposure_on and SOM_EXPOSURE_REG > 0:
            loss = loss + SOM_EXPOSURE_REG * exposure.reg_loss()
        # SOM-DEPTH：方案 E 单目深度先验（点级锚定：把形变后高斯中心投影到
        # 当前相机、采样单目深度做加权 SSI-L1；只监督几何位置，opacity 取 detach）
        if depth_prior is not None and iteration >= depth_from:
            d_loss, d_info = depth_prior.loss(
                viewpoint_cam, gaussians.get_xyz + d_xyz, gaussians.get_opacity)
            if d_loss is not None:
                loss = loss + SOM_DEPTH_LAMBDA * d_loss
                if iteration % 200 == 0:
                    print("[DEPTH] iter {} {}".format(iteration, d_info), flush=True)
            elif iteration % 200 == 0:
                # ⚠️ 失败也要可见（2026-10-16 教训：gy 镜像 bug 让 a<0 门限把损失
                # 静默跳过 10k 步，只在成功时打印根本发现不了）
                print("[DEPTH] iter {} SKIP ({})".format(iteration, d_info), flush=True)
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
                                       testing_iterations, scene, render, (pipe, eval_background), deform,
                                       dataset.load2gpu_on_the_fly, dataset.is_6dof,
                                       exposure=exposure if exposure_on else None)
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
                # SOM-EXPOSURE：随 ckpt 存曝光仿射（render_som/评测复用）
                if exposure is not None:
                    exposure.save_weights(dataset.model_path, iteration)

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
                    n_before = gaussians.get_xyz.shape[0]
                    if densify_sched is not None:
                        # SOM-DENSIFY：方案 D——target 由调度器给（显式 cap 路径下
                        # 就是「补到 cap」，momentum 路径下随需求自适应），超出 target
                        # 的候选按 absgrad 分数 topk 截断；截断前候选数回灌 momentum
                        from densify_sched import sched_densify_and_prune
                        target = densify_sched.target_add(n_before, cur_scale)
                        want_report = iteration % (opt.densification_interval * 10) == 0
                        ret = sched_densify_and_prune(
                            gaussians, dens_thr, 0.005, scene.cameras_extent,
                            size_threshold, target, float(SOM_DENSIFY_CNT_CLAMP),
                            report=want_report)
                        n_cand, pparts = ret if want_report else (ret, None)
                        densify_sched.update(n_cand)
                        print("[DENSIFY] iter {} thr={:.5f} scale={} points {} -> {} "
                              "(cand={} target={} {})".format(
                                  iteration, dens_thr, cur_scale, n_before,
                                  gaussians.get_xyz.shape[0], n_cand, target,
                                  densify_sched.state_str()), flush=True)
                        if want_report:
                            # 三路判据分量（诊断：一次大删由哪路造成）
                            print("[DENSIFY-PRUNE] iter {} opa={} vs={} ws={} "
                                  "applied={} max_screen_size={}".format(
                                      iteration, pparts["opa"], pparts["vs"],
                                      pparts["ws"], pparts["applied"],
                                      size_threshold), flush=True)
                    elif n_before >= SOM_MAX_GAUSSIANS:
                        # SOM：点数上限护栏（v2 事故：thr 0.0002+窗口 10000 导致点数
                        # 爆炸、步速掉 13 倍）。2026-10-16 修正：超帽后**只 prune 不
                        # densify**（旧版直接跳过整个调用，连 prune 都停了，与注释不符）。
                        # ⚠️ max_screen_size 必须传 None（与窗口后同一条理由，2026-10-16
                        # 低 cap 实测实锤）：这条路径没有 densify 补点 → densification_postfix
                        # 不清零 max_radii2D → 尺寸判据退化成「历史最大屏幕半径」。
                        # 带判据时 iter 3100 一步删 86,738 点（162k→75k，−54%），随后
                        # 与 densify 交替震荡（110k↔150k 每 100 步删 6k~26k）。
                        from densify_sched import sched_prune_only
                        pparts = sched_prune_only(gaussians, 0.005, scene.cameras_extent,
                                                  None, report=True)
                        # 分量埋点：这条路径同样「只剪不增」，必须盯 vs（max_radii2D
                        # 无 densify 补点清零 → 会单调累积），否则与窗口后同类死亡螺旋
                        print("[DENSIFY-PRUNE] iter {} (cap) opa={} vs={} ws={} "
                              "applied={} max_screen_size=None".format(
                                  iteration, pparts["opa"], pparts["vs"], pparts["ws"],
                                  pparts["applied"]), flush=True)
                        if iteration % (opt.densification_interval * 10) == 0:
                            print("[DENSIFY] iter {} CAP reached ({} >= {}), prune-only "
                                  "{} -> {}".format(
                                      iteration, n_before, SOM_MAX_GAUSSIANS, n_before,
                                      gaussians.get_xyz.shape[0]), flush=True)
                    else:
                        gaussians.densify_and_prune(dens_thr, 0.005, scene.cameras_extent, size_threshold)
                        # SOM：稠密化记账（定位显存爆炸/点数膨胀的一手数据）
                        # flush=True：stdout 重定向到日志文件时是块缓冲，不 flush 会
                        # 攒几小时才落盘（v2 排查时踩过）
                        print("[DENSIFY] iter {} thr={:.5f} scale={} points {} -> {}".format(
                            iteration, dens_thr, cur_scale, n_before, gaussians.get_xyz.shape[0]),
                            flush=True)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            elif (densify_sched is not None and SOM_DENSIFY_PRUNE_AFTER
                    and iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0):
                # SOM-DENSIFY：densify 窗口结束后「只剪不增」（服务器口径：next_n=max_n，
                # 仍按期 prune）。官方在此把 prune 一起停掉，低不透明度点会一直留着。
                from densify_sched import sched_prune_only
                densify_sched.target_add(0, window_closed=True)
                n_before = gaussians.get_xyz.shape[0]
                # ⚠️ max_screen_size=None：窗口后没有 densify 补点，尺寸判据会触发
                # 死亡螺旋（2026-10-16 冒烟：599k→463k→…→9.6k，PSNR→5.34；
                # 三次独立重跑完全一致）。见 densify_sched 模块 docstring 第 6 条。
                pparts = sched_prune_only(gaussians, 0.005, scene.cameras_extent,
                                          None, report=True)
                print("[DENSIFY-PRUNE] iter {} (window closed) opa={} vs={} ws={} "
                      "applied={}".format(iteration, pparts["opa"], pparts["vs"],
                                          pparts["ws"], pparts["applied"]), flush=True)
                if iteration % (opt.densification_interval * 10) == 0:
                    print("[DENSIFY] iter {} window closed, prune-only {} -> {}".format(
                        iteration, n_before, gaussians.get_xyz.shape[0]), flush=True)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration)
                deform.optimizer.step()
                if exposure_on:
                    exposure.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                deform.optimizer.zero_grad()
                if exposure_on:
                    exposure.optimizer.zero_grad()
                deform.update_learning_rate(iteration)

    print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))


def _exposure_render_pass(scene, gaussians, deform, render_func, pipe, background, dataset):
    """SOM-EXPOSURE：对全部训练相机渲染一遍（no_grad），供闭式初始化解 lstsq。
    渲染口径与 training_report 的评测完全一致（deform.step 带 rotations）。"""
    outs = {}
    with torch.no_grad():
        for cam in scene.getTrainCameras():
            if dataset.load2gpu_on_the_fly:
                cam.load2device()
            fid = cam.fid
            xyz = gaussians.get_xyz
            time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
            d_xyz, d_rotation, d_scaling = deform.step(
                xyz.detach(), time_input,
                rotations=gaussians.get_rotation.detach())
            pkg = render_func(cam, gaussians, pipe, background,
                              d_xyz, d_rotation, d_scaling, dataset.is_6dof)
            outs[cam.image_name] = (
                pkg["render"].clamp(0, 1).cpu(),
                cam.original_image.clamp(0, 1).cpu(),
                None)  # 渲染器无 alpha 输出 → 全像素 + 抗外点重解
            if dataset.load2gpu_on_the_fly:
                cam.load2device('cpu')
    return outs


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
                    renderArgs, deform, load2gpu_on_the_fly, is_6dof=False, exposure=None):
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
                    # SOM-EXPOSURE：评测同口径 apply（train 查表 / test 按 fid 插值）
                    if exposure is not None:
                        image = torch.clamp(exposure.apply_eval(
                            viewpoint.image_name, viewpoint.fid, image), 0.0, 1.0)
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
