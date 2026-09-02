#!/usr/bin/env python3
"""pose_refine.py — Learnable camera pose refinement during 3DGS training.

PoseRefineModule: each camera's w2c quaternion + translation (+ optional intrinsics)
  as nn.Parameters, refined by gradient descent alongside the gaussian optimizer.

PoseRefinedCamera: wraps the original 3DGS Camera, overriding world_view_transform
  / full_proj_transform / camera_center as @property computed from the learnable
  quaternion+translation → gradients flow through the rasterizer to the pose params.

Env vars:
  POSE_REFINE_WEIGHT (float, default 0.01): regularization loss weight
  POSE_REFINE_LR_Q  (float, default 1e-3): quaternion learning rate
  POSE_REFINE_LR_T  (float, default 1e-3): translation learning rate
  POSE_REFINE_LR_I  (float, default 1e-4): intrinsic learning rate
  REFINE_INTRINSIC   (0|1, default 0): learn intrinsics or not
"""
import os
import math
import torch
import torch.nn as nn

# Import quat_to_rotmat from pose_adjuster (same directory)
from pose_adjuster import quat_to_rotmat


# ---------------------------------------------------------------------------
# NDC intrinsic conversion (pixel ↔ NDC).
# ---------------------------------------------------------------------------
def intrinsic_to_ndc(intrinsic, W, H):
    """[fx, fy, cx, cy] in pixels → [fx/W, fy/H, cx/W, cy/H] in NDC."""
    denom = torch.tensor([float(W), float(H), float(W), float(H)],
                          device=intrinsic.device, dtype=intrinsic.dtype)
    return intrinsic / denom


def intrinsic_ndc_to_pixel(ndc, W, H):
    """NDC → pixels."""
    mul = torch.tensor([float(W), float(H), float(W), float(H)],
                        device=ndc.device, dtype=ndc.dtype)
    return ndc * mul


# ---------------------------------------------------------------------------
# PoseRefineModule
# ---------------------------------------------------------------------------
class PoseRefineModule(nn.Module):
    """Holds learnable w2c quaternion, translation, and (optionally) intrinsics."""

    def __init__(self, w2c_qvec, w2c_tvec, intrinsic_ndc,
                 refine_intrinsic=False, device="cuda",
                 lr_q=1e-3, lr_t=1e-3, lr_i=1e-4):
        super().__init__()
        # Frozen initial values (for regularization)
        self.register_buffer("w2c_q_vec_ori", w2c_qvec.clone())
        self.register_buffer("w2c_t_vec_ori", w2c_tvec.clone())
        self.register_buffer("intrinsic_ndc_ori", intrinsic_ndc.clone())

        # Learnable parameters
        self.w2c_q_vec = nn.Parameter(w2c_qvec.clone(), requires_grad=True)
        self.w2c_t_vec = nn.Parameter(w2c_tvec.clone(), requires_grad=True)
        if refine_intrinsic:
            self.intrinsic_ndc = nn.Parameter(intrinsic_ndc.clone(), requires_grad=True)
        else:
            self.register_buffer("intrinsic_ndc", intrinsic_ndc.clone())

        self.lr_q = lr_q
        self.lr_t = lr_t
        self.lr_i = lr_i
        self.refine_intrinsic = refine_intrinsic

    @property
    def w2c_r_mat(self):
        """3x3 rotation matrix from learnable quaternion (differentiable)."""
        return quat_to_rotmat(self.w2c_q_vec)

    def reg_loss(self):
        """L2 regularization: keep pose close to initial."""
        q_loss = ((self.w2c_q_vec - self.w2c_q_vec_ori) ** 2).sum()
        t_loss = ((self.w2c_t_vec - self.w2c_t_vec_ori) ** 2).sum()
        return q_loss + t_loss

    def get_optimizer(self):
        params = [
            {"params": [self.w2c_q_vec], "lr": self.lr_q},
            {"params": [self.w2c_t_vec], "lr": self.lr_t},
        ]
        if self.refine_intrinsic and self.intrinsic_ndc.requires_grad:
            params.append({"params": [self.intrinsic_ndc], "lr": self.lr_i})
        return torch.optim.Adam(params)

    def update_ori(self):
        """Update frozen initial values to current (after PoseAdjuster, before training)."""
        with torch.no_grad():
            self.w2c_q_vec_ori.copy_(self.w2c_q_vec)
            self.w2c_t_vec_ori.copy_(self.w2c_t_vec)
            self.intrinsic_ndc_ori.copy_(self.intrinsic_ndc)


# ---------------------------------------------------------------------------
# PoseRefinedCamera — wraps original 3DGS Camera with dynamic w2c.
# ---------------------------------------------------------------------------
class PoseRefinedCamera:
    """Provides the same interface as 3DGS Camera, but world_view_transform
    etc. are computed dynamically from PoseRefineModule (learnable quaternion+translation).

    The base Camera provides: original_image, image_width, image_height, FoVx, FoVy,
    tanfovx, tanfovy, projection_matrix (fixed unless refining intrinsics).
    """

    def __init__(self, base_camera, w2c_qvec, w2c_tvec, intrinsic,
                 refine_intrinsic=False, device="cuda",
                 lr_q=1e-3, lr_t=1e-3, lr_i=1e-4):
        self.base = base_camera
        W = base_camera.image_width
        H = base_camera.image_height
        ndc = intrinsic_to_ndc(intrinsic, W, H)
        self.pose_module = PoseRefineModule(
            w2c_qvec, w2c_tvec, ndc,
            refine_intrinsic=refine_intrinsic, device=device,
            lr_q=lr_q, lr_t=lr_t, lr_i=lr_i)
        self.pose_optimizer = self.pose_module.get_optimizer()
        self.uid = base_camera.uid
        self.image_name = base_camera.image_name

    # ── 未定义属性委托给 base Camera（alpha_mask, depth_reliable 等）─────────
    def __getattr__(self, name):
        # __getattr__ 只在正常查找失败时触发，不会拦截已定义的属性
        return getattr(self.base, name)

    # ── Delegated to base Camera ────────────────────────────────────────────
    @property
    def original_image(self):
        return self.base.original_image

    @property
    def image_width(self):
        return self.base.image_width

    @property
    def image_height(self):
        return self.base.image_height

    @property
    def tanfovx(self):
        return self.base.tanfovx

    @property
    def tanfovy(self):
        return self.base.tanfovy

    @property
    def FoVx(self):
        return self.base.FoVx

    @property
    def FoVy(self):
        return self.base.FoVy

    @property
    def projection_matrix(self):
        if self.pose_module.refine_intrinsic:
            # Recompute projection from dynamic intrinsics
            ndc = self.pose_module.intrinsic_ndc
            W, H = self.base.image_width, self.base.image_height
            intr = intrinsic_ndc_to_pixel(ndc, W, H)
            fx, fy = intr[0], intr[1]
            fovx = 2.0 * math.atan(float(W) / (2.0 * float(fx)))
            fovy = 2.0 * math.atan(float(H) / (2.0 * float(fy)))
            from utils.graphics_utils import getProjectionMatrix
            return getProjectionMatrix(znear=0.01, zfar=100.0,
                                        FoVx=fovx, FoVy=fovy).transpose(0, 1).to(self.base.original_image.device)
        return self.base.projection_matrix

    # ── Dynamic computation from PoseRefineModule ──────────────────────────
    @property
    def world_view_transform(self):
        """4x4 w2c matrix in 3DGS convention: [[R, 0], [T, 1]] (row-major, differentiable).

        Matches original Camera: getWorld2View(R,T).transpose(0,1) = [[R, 0], [T, 1]].
        用 cat 构造而非 zeros+in-place 赋值，保证梯度可流回位姿参数。
        """
        R = self.pose_module.w2c_r_mat            # (3, 3) w2c rotation from quaternion
        T = self.pose_module.w2c_t_vec            # (3,) w2c translation
        # 顶 3 行: [R | 0]
        top = torch.cat([R, torch.zeros(3, 1, device=R.device, dtype=R.dtype)], dim=1)  # (3,4)
        # 底 1 行: [T | 1]
        bottom = torch.cat([T.unsqueeze(0), torch.ones(1, 1, device=R.device, dtype=R.dtype)], dim=1)  # (1,4)
        wvt = torch.cat([top, bottom], dim=0)     # (4,4) [[R,0],[T,1]]
        return wvt.contiguous()

    @property
    def full_proj_transform(self):
        return (self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0))).squeeze(0).contiguous()

    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[3, :3].contiguous()
