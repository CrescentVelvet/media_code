#!/usr/bin/env python3
"""render_som.py — SoM 运动基模型的渲染/评测（fork 官方 render.py）。

与官方 render.py 的差异：
  1. DeformModel -> SoMMotionModel（som_motion.py），从 model_path/som_motion/ 加载；
  2. deform.step 统一走 _step() 并传 rotations（与 train_som.py 的渲染口径一致）；
  3. 只保留三个实用 mode：render（测试视角+指标）/ time（固定视角扫时间）/
     original（时间×视角组合——目视检查重影的标准工具，同 deformable_human）。
     其余 mode（view/pose/all）用 pose_spherical 等预设轨迹，对本场景价值低，未移植。

用法（由 03_render.sh 调用）：
  cd $DG_DIR && SOM_NUM_BASES=10 python render_som.py -m <model_path> --mode render
"""
import os
import sys
import math

_DG_DIR = os.environ.get("DG_DIR")
if not _DG_DIR or not os.path.isfile(os.path.join(_DG_DIR, "render.py")):
    sys.exit("❌ DG_DIR 未指向 Deformable-3D-Gaussians 仓（由 03_render.sh 注入）")
sys.path.insert(0, _DG_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # som_motion 同目录

import torch
from scene import Scene
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import imageio
import numpy as np
import time
from tqdm import tqdm

from som_motion import SoMMotionModel

# SOM：渲染时必须与训练一致的 B（ckpt 里有断言核对）
SOM_NUM_BASES = int(os.environ.get("SOM_NUM_BASES", "10"))


def _step(deform, gaussians, fid):
    """统一的形变求值：warp 位置 + 旋转朝向（口径与 train_som.py 一致）。"""
    xyz = gaussians.get_xyz
    time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
    return deform.step(xyz.detach(), time_input,
                       rotations=gaussians.get_rotation.detach())


def render_set(model_path, load2gpu_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background, deform):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    t_list = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if load2gpu_on_the_fly:
            view.load2device()
        d_xyz, d_rotation, d_scaling = _step(deform, gaussians, view.fid)
        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
        rendering = results["render"]
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))

    for idx, view in enumerate(tqdm(views, desc="FPS measuring")):
        _step(deform, gaussians, view.fid)
        torch.cuda.synchronize()
        t_start = time.time()
        d_xyz, d_rotation, d_scaling = _step(deform, gaussians, view.fid)
        render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
        torch.cuda.synchronize()
        t_end = time.time()
        t_list.append(t_end - t_start)

    t = np.array(t_list[5:])
    fps = 1.0 / t.mean()
    print('Test FPS: \033[1;35m{:.5f}\033[0m, Num. of GS: {}'.format(fps, gaussians.get_xyz.shape[0]))


def interpolate_time(model_path, load2gpt_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background, deform):
    render_path = os.path.join(model_path, name, "interpolate_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_{}".format(iteration), "depth")
    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
    frame = 150
    idx = torch.randint(0, len(views), (1,)).item()
    view = views[idx]
    renderings = []
    for t in tqdm(range(0, frame, 1), desc="Rendering progress"):
        fid = torch.Tensor([t / (frame - 1)]).cuda()
        d_xyz, d_rotation, d_scaling = _step(deform, gaussians, fid)
        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(t) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(t) + ".png"))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)


def _mat_to_quat(R):
    """numpy (3,3) 旋转矩阵 -> (4,) 四元数 (w,x,y,z)。Shepperd 方法，数值稳定。"""
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        return np.array([0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s])
    if m00 > m11 and m00 > m22:
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        return np.array([(m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s])
    if m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        return np.array([(m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s])
    s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
    return np.array([(m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s])


def _quat_to_mat(q):
    """(4,) 四元数 (w,x,y,z) -> (3,3) 旋转矩阵。"""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _slerp_mat(R0, R1, t):
    """SOM 修复：旋转矩阵四元数球面插值（替代官方矩阵 lerp）。

    官方 D-3DGS 对 R 直接线性插值，相邻相机角度差大时（如环绕采集）
    插值矩阵失去正交性甚至奇异（double_wave 在 ~i=6xx 触发 LinAlgError）。
    """
    q0, q1 = _mat_to_quat(R0), _mat_to_quat(R1)
    if float(np.dot(q0, q1)) < 0.0:
        q1 = -q1  # 双重覆盖取短路径
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
    else:
        theta = math.acos(dot)
        q = (np.sin((1.0 - t) * theta) * q0 + np.sin(t * theta) * q1) / np.sin(theta)
    return _quat_to_mat(q)


def interpolate_view_original(model_path, load2gpt_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline,
                              background, deform):
    """时间×视角组合渲染（目视检查重影/撕裂的标准工具）。"""
    render_path = os.path.join(model_path, name, "interpolate_hyper_view_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_hyper_view_{}".format(iteration), "depth")
    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    frame = 1000
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    view = views[0]
    renderings = []
    for i in tqdm(range(frame), desc="Rendering progress"):
        fid = torch.Tensor([i / (frame - 1)]).cuda()
        query_idx = i / frame * len(views)
        begin_idx = int(np.floor(query_idx))
        end_idx = int(np.ceil(query_idx))
        if end_idx == len(views):
            break
        R_begin, R_end = views[begin_idx].R, views[end_idx].R
        t_begin, t_end = views[begin_idx].T, views[end_idx].T
        ratio = query_idx - begin_idx
        R_cur = _slerp_mat(R_begin, R_end, ratio)  # SOM 修复：SLERP 替代矩阵 lerp
        T_cur = (1 - ratio) * t_begin + ratio * t_end
        view.reset_extrinsic(R_cur, T_cur)

        d_xyz, d_rotation, d_scaling = _step(deform, gaussians, fid)
        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(i) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(i) + ".png"))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=60, quality=8)


def render_sets(dataset, iteration, pipeline, skip_train, skip_test, mode):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        num_frames = len(scene.getTrainCameras()) + len(scene.getTestCameras())
        deform = SoMMotionModel(num_frames=num_frames, num_bases=SOM_NUM_BASES)
        deform.load_weights(dataset.model_path)
        print("[SOM] num_frames={}, num_bases={}".format(num_frames, SOM_NUM_BASES))

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if mode == "render":
            render_func = render_set
        elif mode == "time":
            render_func = interpolate_time
        elif mode == "original":
            render_func = interpolate_view_original
        else:
            raise ValueError("render_som.py 只支持 render/time/original，收到: {}".format(mode))

        if not skip_train:
            render_func(dataset.model_path, dataset.load2gpu_on_the_fly, dataset.is_6dof, "train", scene.loaded_iter,
                        scene.getTrainCameras(), gaussians, pipeline, background, deform)
        if not skip_test:
            render_func(dataset.model_path, dataset.load2gpu_on_the_fly, dataset.is_6dof, "test", scene.loaded_iter,
                        scene.getTestCameras(), gaussians, pipeline, background, deform)


if __name__ == "__main__":
    parser = ArgumentParser(description="Rendering script parameters (+ SoM motion bases)")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mode", default='render', choices=['render', 'time', 'original'])
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.mode)
