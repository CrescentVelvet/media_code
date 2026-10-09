#!/usr/bin/env python3
"""pose_delta.py — 训练中联合精炼相机位姿（可学四元数 + 平移），场景侧 delta 注入。

为什么不用 vggt_human 的相机侧 PoseRefinedCamera 方案：
  stock diff_gaussian_rasterization 的 CUDA backward **不对 viewmatrix/projmatrix
  回传梯度**（vggt_human/04_train_3dgs.sh 已标注 train_pose_refine.py 因此不可用，
  位姿参数拿不到 grad）。本实现把问题反过来：相机矩阵保持冻结喂给光栅化器，
  把「相机精炼」等价为「形变后高斯的刚体变换」——数学上严格等价：

    精炼相机渲染原场景  ≡  原相机渲染变换后场景
    A_w = R0^T @ R'         （R0/t0=冻结的 COLMAP w2c，R'/t'=可学精炼 w2c）
    b_w = R0^T @ (t' - t0)
    means3D'  = means3D @ A_w^T + b_w
    rotation' = q(A_w) ⊗ rotation
    campos'   = A_w @ c' + b_w   （c' = -R'^T t'，精炼后的相机中心）

  梯度经 means3D'/rotation' 流回 q/t 参数，绕开光栅化器的无梯度矩阵。

已知近似（可接受）：SH 颜色在 CUDA 内用 (means3D - campos) 求方向，
  变换后方向差一个旋转 A_w（SH 系数不做 Wigner-D 旋转）。delta 被正则压在
  小角度（通常 <1°），二阶小量，忽略。内参不可学：焦距只出现在 projmatrix/
  tanfov，同样拿不到梯度（如需内参精炼，在 01 阶段用 colmap bundle_adjuster）。

环境变量（由 02_train.sh 透传）：
  POSE_REFINE_WEIGHT (0.01)  位姿正则权重（L2 拉回初始 COLMAP 位姿）
  POSE_REFINE_LR_Q   (1e-3)  四元数学习率
  POSE_REFINE_LR_T   (1e-3)  平移学习率
"""
import os
import json
import math
import torch
import torch.nn as nn

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.sh_utils import eval_sh
from utils.rigid_utils import from_homogenous, to_homogenous

POSE_REFINE_WEIGHT = float(os.environ.get("POSE_REFINE_WEIGHT", "0.01"))
POSE_REFINE_LR_Q = float(os.environ.get("POSE_REFINE_LR_Q", "1e-3"))
POSE_REFINE_LR_T = float(os.environ.get("POSE_REFINE_LR_T", "1e-3"))


# ---------------------------------------------------------------------------
# 四元数工具（[w,x,y,z] 约定，与 COLMAP/pose_adjuster.py 一致）
# ---------------------------------------------------------------------------
def quat_to_rotmat(q):
    """单位四元数 -> 3x3 旋转矩阵，可微。q: (..., 4)"""
    q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-9)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)


def rotmat_to_quat(R):
    """3x3 旋转矩阵 -> [w,x,y,z]，Shepperd 全分支（非可微路径用，float64 安全）。"""
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = torch.sqrt(tr + 1.0) * 2
        q = torch.stack([0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s])
    elif m00 > m11 and m00 > m22:
        s = torch.sqrt(1.0 + m00 - m11 - m22) * 2
        q = torch.stack([(m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s])
    elif m11 > m22:
        s = torch.sqrt(1.0 + m11 - m00 - m22) * 2
        q = torch.stack([(m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s])
    else:
        s = torch.sqrt(1.0 + m22 - m00 - m11) * 2
        q = torch.stack([(m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s])
    return q / (torch.norm(q) + 1e-9)


def quat_multiply(q1, q2):
    """Hamilton 积 q1 ⊗ q2（先施加 q2 再 q1；旋转矩阵对应 R(q1) @ R(q2)）。"""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return torch.stack((
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ), dim=-1)


# ---------------------------------------------------------------------------
# PoseDeltaStore — 每个训练相机一组可学 w2c（四元数 + 平移）
# ---------------------------------------------------------------------------
class PoseDeltaStore(nn.Module):
    """保存全部训练相机的精炼位姿，输出场景侧等价变换。"""

    def __init__(self, train_cameras, device="cuda"):
        super().__init__()
        self.uids = [cam.uid for cam in train_cameras]
        self.names = [cam.image_name for cam in train_cameras]
        self.uid2idx = {u: i for i, u in enumerate(self.uids)}
        n = len(self.uids)

        r0 = torch.zeros(n, 3, 3, dtype=torch.float64)
        t0 = torch.zeros(n, 3, dtype=torch.float64)
        for i, cam in enumerate(train_cameras):
            # 官方 Camera：R 存的是 w2c 的转置（dataset_readers.py: R = qvec2rotmat(q).T），
            # T 是 w2c 平移 → w2c 旋转 = cam.R.T
            r0[i] = torch.tensor(cam.R, dtype=torch.float64).T
            t0[i] = torch.tensor(cam.T, dtype=torch.float64)
        q0 = torch.stack([rotmat_to_quat(r0[i]) for i in range(n)])

        # 初值用 float64 计算保精度，存 float32——训练期与 delta/float32 计算图对齐，
        # 否则 r0.T @ r_ref 报 double != float（烟雾测试实测踩过）
        self.register_buffer("r0", r0.to(device).float())
        self.register_buffer("t0", t0.to(device).float())
        self.register_buffer("q0", q0.to(device).float())
        self.delta_q = nn.Parameter(torch.zeros(n, 4, device=device))   # 加在 q0 上
        self.delta_t = nn.Parameter(torch.zeros(n, 3, device=device))

    # ── 当前精炼位姿 ────────────────────────────────────────────────────────
    def refined_rt(self, idx):
        q = torch.nn.functional.normalize(self.q0[idx] + self.delta_q[idx], dim=0)
        r = quat_to_rotmat(q)
        t = self.t0[idx] + self.delta_t[idx]
        return q, r, t

    def scene_transform(self, idx):
        """返回 (A_w, b_w, q_A, campos')：把形变后高斯做等价刚体变换。"""
        q, r_ref, t_ref = self.refined_rt(idx)
        r0, t0 = self.r0[idx], self.t0[idx]
        A_w = r0.T @ r_ref                       # 世界系等价旋转
        b_w = r0.T @ (t_ref - t0)                # 世界系等价平移
        q_A = rotmat_to_quat(A_w.double()).float()
        cam_center_ref = -r_ref.T @ t_ref        # 精炼相机中心（原世界系）
        campos = A_w @ cam_center_ref + b_w      # 变换后场景里的相机中心
        return A_w.float(), b_w.float(), q_A, campos.float()

    # ── 正则与优化器 ────────────────────────────────────────────────────────
    def reg_loss(self, idx):
        return (self.delta_q[idx] ** 2).sum() + (self.delta_t[idx] ** 2).sum()

    def get_optimizer(self, lr_q=POSE_REFINE_LR_Q, lr_t=POSE_REFINE_LR_T):
        return torch.optim.Adam([
            {"params": [self.delta_q], "lr": lr_q},
            {"params": [self.delta_t], "lr": lr_t},
        ])

    # ── 记录 ────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def drift_stats(self):
        # 正确口径：精炼位姿与初始位姿两个单位四元数的夹角 2·acos(|q_ref·q0|)。
        # 不能对 delta_q 本身归一化算角——delta 是加性偏移不是旋转，
        # 归一化后方向任意，曾算出假 115°（烟雾测试实测）
        q_ref = torch.nn.functional.normalize(self.q0 + self.delta_q, dim=-1)
        cos = (q_ref * self.q0).sum(-1).abs().clamp(max=1.0)
        ang = torch.rad2deg(2 * torch.acos(cos))
        dt = self.delta_t.norm(dim=-1)
        return (f"rot: mean={ang.mean():.3f}° max={ang.max():.3f}° | "
                f"trans: mean={dt.mean():.4f} max={dt.max():.4f}")

    @torch.no_grad()
    def save(self, path):
        records = {}
        for i, name in enumerate(self.names):
            q, r, t = self.refined_rt(i)
            records[name] = {
                "qvec_w2c": [float(x) for x in q.cpu()],
                "tvec_w2c": [float(x) for x in t.cpu()],
                "delta_q": [float(x) for x in self.delta_q[i].cpu()],
                "delta_t": [float(x) for x in self.delta_t[i].cpu()],
            }
        with open(path, "w") as f:
            json.dump({"convention": "COLMAP w2c qvec/tvec, 含 COLMAP 初值+delta",
                       "cameras": records}, f, indent=2)
        print(f"[PoseDeltaStore] saved -> {path}")


# ---------------------------------------------------------------------------
# render 的 pose-delta 变体（fork 自官方 gaussian_renderer/__init__.py，
# MIT license；改动点用「POSE-DELTA」标注）
# ---------------------------------------------------------------------------
def render_pose_delta(viewpoint_camera, pc, pipe, bg_color, d_xyz, d_rotation, d_scaling,
                      is_6dof=False, scaling_modifier=1.0, override_color=None,
                      store: PoseDeltaStore = None):
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype,
                                          requires_grad=True, device="cuda") + 0
    screenspace_points_densify = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype,
                                                  requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
        screenspace_points_densify.retain_grad()
    except Exception:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # POSE-DELTA：有精炼位姿时覆盖 campos（见模块 docstring 的近似说明）
    campos = viewpoint_camera.camera_center
    A_w = b_w = q_A = None
    if store is not None and viewpoint_camera.uid in store.uid2idx:
        idx = store.uid2idx[viewpoint_camera.uid]
        A_w, b_w, q_A, campos = store.scene_transform(idx)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx, tanfovy=tanfovy,
        bg=bg_color, scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,     # 冻结的原位姿
        projmatrix=viewpoint_camera.full_proj_transform,      # 冻结
        sh_degree=pc.active_sh_degree,
        campos=campos,
        prefiltered=False, debug=pipe.debug)
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    if is_6dof:
        if torch.is_tensor(d_xyz) is False:
            means3D = pc.get_xyz
        else:
            means3D = from_homogenous(
                torch.bmm(d_xyz, to_homogenous(pc.get_xyz).unsqueeze(-1)).squeeze(-1))
    else:
        means3D = pc.get_xyz + d_xyz
    opacity = pc.get_opacity

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling + d_scaling
        rotations = pc.get_rotation + d_rotation

    # POSE-DELTA：场景侧等价刚体变换（梯度经此流回位姿参数）
    if A_w is not None:
        means3D = means3D @ A_w.T + b_w
        if rotations is not None:
            rotations = quat_multiply(q_A.unsqueeze(0), rotations)
        if cov3D_precomp is not None:
            # cov' = A @ cov @ A^T
            cov3D_precomp = A_w.unsqueeze(0) @ cov3D_precomp @ A_w.T.unsqueeze(0)

    shs = None
    colors_precomp = None
    if colors_precomp is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            # POSE-DELTA：方向也要用变换后的点与相机中心
            dir_pp = (means3D - campos.repeat(pc.get_features.shape[0], 1)) if A_w is not None \
                else (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    rendered_image, radii, depth = rasterizer(
        means3D=means3D, means2D=screenspace_points,
        means2D_densify=screenspace_points_densify,
        shs=shs, colors_precomp=colors_precomp, opacities=opacity,
        scales=scales, rotations=rotations, cov3D_precomp=cov3D_precomp)

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "viewspace_points_densify": screenspace_points_densify,
            "visibility_filter": radii > 0,
            "radii": radii,
            "depth": depth}
