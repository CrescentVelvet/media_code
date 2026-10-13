#!/usr/bin/env python3
"""som_motion.py — SoM 式低秩 SE(3) 运动基模型（替换 D-3DGS 逐帧形变 MLP）。

设计来源：Shape of Motion（媒体论文调研库 3D-20240718-Shape-of-Motion.html §4-§7），
移植到 Deformable-3D-Gaussians 框架（canonical 高斯 + 光栅化/训练循环全部复用）。

表示（SoM 式 4 的忠实对应）：
    T_{0->t}(x) = sum_b w_b(x) * (R_b(t) x + t_b(t)),  sum_b w_b = 1
  - 运动基：逐帧参数表 rot6d [T,B,6] + transl [T,B,3]，可学习，identity 初始化
    （SoM 用 6D 连续旋转 + 逐帧表，不用 MLP —— params.py:212-243）。
  - 系数 w：SoM 是 per-gaussian 可学习表 [G,B]（params.py:136）。本实现改用
    空间 MLP 场 softmax(coef_net(fourier(x)))——原因：D-3DGS 训练期 densify/prune
    会增删高斯，per-gaussian 参数表需要 hook densify 记账（SoM 用的 gsplat 自适应
    控制与 D-3DGS 的 densify_and_prune 不同构），MLP 场对新增点天然给出合理系数，
    且空间相邻点系数相近（软刚体分组的光滑版）。代价：分组边界不如查表锐利。
    【Phase 1 简化决策，如需恢复查表式系数见 README「与 SoM 的差异」】
  - 合成：6D 加权求和后 Gram-Schmidt 正交化（SoM compute_transforms 同款；
    已知局限——两条朝向差异大的基之间过渡会「缩短」，见调研笔记 §11 批评，
    需要 screw/log 空间混合时另起炉灶）。

与 D-3DGS 的接口对齐：step(xyz, time_input, rotations=None) ->
(d_xyz, d_rotation, d_scaling)，可直接替换 DeformModel：
  - render 里 means3D = xyz + d_xyz；故 d_xyz = warp(x) - x。
  - render 里 rotations = pc.get_rotation + d_rotation（光栅化器内部归一化）；
    刚体旋转应得 q' = q_R ⊗ q，故 d_rotation = q' - q（先做符号对齐让 delta 最小）。
    不传 rotations 时返回 0（位置照常 warp）——兼容官方 render.py 调用签名。
  - 刚体不变换尺度：d_scaling 恒 0。

正则：基加速度平滑（中心差分，rot:transl 内权 1:2，对应 SoM loss_utils.py:175-194）。
"""
import os

import torch
import torch.nn as nn

from utils.general_utils import get_expon_lr_func
from utils.system_utils import searchForMaxIteration


# ───────────────────────────── 旋转工具（wxyz 约定）──────────────────────────
def rot6d_to_matrix(r6):
    """6D 连续旋转表示 -> 旋转矩阵（Gram-Schmidt 正交化）。

    r6: [..., 6]（前 3 维 a1，后 3 维 a2）-> [..., 3, 3]，列为 (b1,b2,b3)。
    """
    a1, a2 = r6[..., 0:3], r6[..., 3:6]
    b1 = a1 / (a1.norm(dim=-1, keepdim=True) + 1e-8)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = b2 / (b2.norm(dim=-1, keepdim=True) + 1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def matrix_to_quaternion(R):
    """旋转矩阵 -> 四元数（wxyz）。R: [..., 3, 3] -> [..., 4]。

    无分支实现（按迹与对角线选最大分量），全程可导。
    """
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]

    qw = 0.5 * torch.sqrt((1 + m00 + m11 + m22).clamp(min=1e-8))
    qx = 0.5 * torch.sqrt((1 + m00 - m11 - m22).clamp(min=1e-8))
    qy = 0.5 * torch.sqrt((1 - m00 + m11 - m22).clamp(min=1e-8))
    qz = 0.5 * torch.sqrt((1 - m00 - m11 + m22).clamp(min=1e-8))

    # copysign 语义；注意 torch.sign(0)=0 会把分量清零，必须用 where 实现
    qx = qx * torch.where((m21 - m12) >= 0, 1.0, -1.0)
    qy = qy * torch.where((m02 - m20) >= 0, 1.0, -1.0)
    qz = qz * torch.where((m10 - m01) >= 0, 1.0, -1.0)

    q = torch.stack((qw, qx, qy, qz), dim=-1)
    return q / (q.norm(dim=-1, keepdim=True) + 1e-8)


def quaternion_multiply(q1, q2):
    """wxyz 四元数乘法 q1 ⊗ q2（与 gaussian_renderer 内的约定一致）。"""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack((w, x, y, z), dim=-1)


class FourierEmbed(nn.Module):
    """多频 Fourier 位置编码：x -> [x, sin/cos(2^k * pi * x)]。"""

    def __init__(self, num_freqs=6):
        super().__init__()
        self.num_freqs = num_freqs
        freqs = 2.0 ** torch.arange(num_freqs, dtype=torch.float32) * torch.pi
        self.register_buffer("freqs", freqs, persistent=False)
        self.out_dim = 3 * (1 + 2 * num_freqs)

    def forward(self, x):
        # x: [N, 3]
        xb = x.unsqueeze(-1) * self.freqs  # [N, 3, F]
        return torch.cat([x, torch.sin(xb).flatten(-2), torch.cos(xb).flatten(-2)], dim=-1)


class SoMMotionModel:
    """SoM 式运动模型。接口与 DeformModel 对齐（step/train_setting/save/load）。"""

    def __init__(self, num_frames, num_bases=10, coef_hidden=128, coef_freqs=6):
        self.num_frames = int(num_frames)
        self.num_bases = int(num_bases)
        self.optimizer = None
        # deform_model.py 的同款缩放（沿用官方 lr 调度器接线方式）
        self.spatial_lr_scale = 5

        dev = "cuda"
        # 运动基：逐帧参数表，identity 初始化（6D identity = [1,0,0,0,1,0]）
        rot6d_init = torch.zeros(self.num_frames, self.num_bases, 6, device=dev)
        rot6d_init[..., 0] = 1.0
        rot6d_init[..., 4] = 1.0
        self.rot6d = nn.Parameter(rot6d_init)
        self.transl = nn.Parameter(torch.zeros(self.num_frames, self.num_bases, 3, device=dev))

        # 系数场：fourier(x) -> MLP -> B logits（softmax 在 forward 里做）
        embed = FourierEmbed(coef_freqs)
        self.coef_net = nn.Sequential(
            embed,
            nn.Linear(embed.out_dim, coef_hidden), nn.ReLU(inplace=True),
            nn.Linear(coef_hidden, coef_hidden), nn.ReLU(inplace=True),
            nn.Linear(coef_hidden, self.num_bases),
        ).to(dev)
        # 末层小初始化：开局系数近均匀 → 初始 warp ≈ 各基平均（各基 identity → 恒等）
        nn.init.uniform_(self.coef_net[-1].weight, -1e-4, 1e-4)
        nn.init.zeros_(self.coef_net[-1].bias)

    # ── DeformModel 兼容接口 ──────────────────────────────────────────────
    def step(self, xyz, time_input, rotations=None):
        """xyz: [N,3]（canonical，外部已 detach）；time_input: [N,1] ∈ [0,1]。

        返回 (d_xyz, d_rotation, d_scaling)，与官方 render 的加法接口对应。
        """
        N = xyz.shape[0]
        t_idx = (time_input.squeeze(-1) * (self.num_frames - 1)).round().long()
        t_idx = t_idx.clamp(0, self.num_frames - 1)  # [N]

        w = torch.softmax(self.coef_net(xyz), dim=-1)  # [N, B]

        r6 = (w.unsqueeze(-1) * self.rot6d[t_idx]).sum(dim=1)   # [N, 6]
        tr = (w.unsqueeze(-1) * self.transl[t_idx]).sum(dim=1)  # [N, 3]
        R = rot6d_to_matrix(r6)                                  # [N, 3, 3]

        x_warp = torch.bmm(R, xyz.unsqueeze(-1)).squeeze(-1) + tr
        d_xyz = x_warp - xyz

        if rotations is not None:
            q_R = matrix_to_quaternion(R)                        # [N, 4] wxyz
            q_new = quaternion_multiply(q_R, rotations)
            # 符号对齐：q 与 -q 同旋转，选与 q 同侧的表示让 delta 最小
            sign = torch.sign((q_new * rotations).sum(dim=-1, keepdim=True))
            sign = torch.where(sign == 0, torch.ones_like(sign), sign)
            d_rotation = q_new * sign - rotations
        else:
            d_rotation = torch.zeros(N, 4, device=xyz.device, dtype=xyz.dtype)

        return d_xyz, d_rotation, 0.0

    def smooth_loss(self):
        """基加速度平滑（中心差分）。内权 rot:transl = 1:2（SoM loss_utils.py:175-194）。"""
        if self.num_frames < 3:
            return torch.zeros((), device=self.rot6d.device)
        acc_rot = self.rot6d[2:] - 2 * self.rot6d[1:-1] + self.rot6d[:-2]
        acc_tr = self.transl[2:] - 2 * self.transl[1:-1] + self.transl[:-2]
        return acc_rot.pow(2).mean() + 2.0 * acc_tr.pow(2).mean()

    def train_setting(self, training_args, basis_lr=1.6e-4, coef_lr=1e-2):
        """分组学习率：基 rots/transls 1.6e-4、系数 1e-2（SoM configs.py:5-11,23-26）。

        ⚠️ 不乘 spatial_lr_scale——SoM 的学习率是绝对值口径，与 D-3DGS 的
        position_lr 体系无关。
        """
        l = [
            {"params": [self.rot6d, self.transl], "lr": basis_lr, "name": "motion_bases"},
            {"params": list(self.coef_net.parameters()), "lr": coef_lr, "name": "coef_net"},
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        # 恒等调度（保持 lr 不变；要衰减可换成 get_expon_lr_func）
        self.deform_scheduler_args = get_expon_lr_func(
            lr_init=1.0, lr_final=1.0, lr_delay_mult=0.0, max_steps=1)

    def save_weights(self, model_path, iteration):
        out_dir = os.path.join(model_path, "som_motion/iteration_{}".format(iteration))
        os.makedirs(out_dir, exist_ok=True)
        torch.save(
            {
                "num_frames": self.num_frames,
                "num_bases": self.num_bases,
                "rot6d": self.rot6d.detach().cpu(),
                "transl": self.transl.detach().cpu(),
                "coef_net": self.coef_net.state_dict(),
            },
            os.path.join(out_dir, "som.pth"),
        )

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "som_motion"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "som_motion/iteration_{}/som.pth".format(loaded_iter))
        ckpt = torch.load(weights_path, map_location="cuda", weights_only=True)
        assert ckpt["num_frames"] == self.num_frames, (
            "num_frames 不一致：ckpt={} vs 当前={}（场景相机数变了？）".format(
                ckpt["num_frames"], self.num_frames))
        assert ckpt["num_bases"] == self.num_bases, (
            "num_bases 不一致：ckpt={} vs 当前={}（渲染时须与训练 NUM_BASES 一致）".format(
                ckpt["num_bases"], self.num_bases))
        with torch.no_grad():
            self.rot6d.copy_(ckpt["rot6d"].cuda())
            self.transl.copy_(ckpt["transl"].cuda())
        self.coef_net.load_state_dict(ckpt["coef_net"])
        print("✅ [SoM] 加载运动基权重: {}".format(weights_path))

    def update_learning_rate(self, iteration):
        # 恒等调度：直接返回当前 lr（接口对齐用）
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "motion_bases":
                return param_group["lr"]
        return None
