# -*- coding: utf-8 -*-
"""depth_prior.py — 方案 E 训练侧：单目深度先验正则（点级锚定版）

## 关键设计转折（2026-10-16 查证）

最初设计（BACKLOG E 案）是「渲染深度 ↔ 单目深度」的图像级 SSI loss。但实测
本仓 fork 光栅化器（depth-diff-gaussian-rasterization）的 **depth 是前向-only
输出**：autograd backward 签名虽接收 grad_out_depth，却从不传给 CUDA kernel
（RasterizeGaussiansBackwardCUDA 只有 dL_dout_color 入口，backward.cu 全文无
dL_ddepth）→ 渲染深度 loss 没有任何梯度回路，是个摆设。

故改为**点级锚定**（MonoGS/Depth-GS 一族的标准做法，等价于把渲染深度监督
拆开成逐点几何约束）：
  - 把**形变后**的高斯中心投影到当前相机（投影约定：`clip = p_h @
    full_proj_transform`，**gy = +ndc_y**——本 fork 视图矩阵 y 轴朝下、投影
    矩阵不翻转，故 ndc_y>0 即画面下半部，grid_sample y=+1 也是底部，无需取负。
    ⚠️ 曾照抄 sample_fg_prob 的 gy=-ndc_y，系镜像 bug，2026-10-16 用光栅化器
    自渲染深度逐点核对后更正，见 loss() 内注释）；
  - 在单目深度图上双线性采样得 mono_d[N]，相机系 z_view[N]；
  - 尺度/平移对齐：加权最小二乘解 (a,b) 使 a·mono+b ≈ z_view（a/b detach，
    梯度只往几何位置流，不对监督信号求导）；
  - 损失 = Σ w·|a·mono+b − z_view| / Σw，w = opacity.detach() × fg_prob。
    opacity 取 detach：**只监督位置，不给「调低透明度逃惩罚」留捷径**。

为什么这个形式反而更贴「手部残余模糊」的目标：漂浮的手部高斯本身就是
位置自由度欠约束的产物，直接约束逐点位置比约束渲染深度更直指根因。

资产（离线产物，训练只读）：
  - <scene>/depths/<stem>.npz  01f_depth_maps.sh（UniDepth V1 ViT-L14，gavs env）
  - <scene>/masks/<stem>.png   可选；deformable_human 01d 的软 mask（0-255），
    采样为逐点权重（0..1 软值，软边缘天然加权）。缺失时权重全 1（只 opacity）。

BACKLOG 风险应对：深度网络对快速运动小手部误差大 → 权重小（λ 0.05 起步）+
fg mask 加权 + 对齐系数 a 必须 >0（a≤0 说明该帧深度结构性不可信，跳过）。
"""

import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F


class DepthPrior:
    def __init__(self, scene_path, use_mask=1, min_points=2048, cache_size=16):
        self.depths_dir = os.path.join(scene_path, "depths")
        self.masks_dir = os.path.join(scene_path, "masks")
        self.use_mask = bool(use_mask)
        self.min_points = int(min_points)
        self._cache = OrderedDict()   # stem -> (mono[1,1,H,W] float32 cpu, fg[1,1,H,W] or None)
        self._cache_size = int(cache_size)
        self.available = os.path.isdir(self.depths_dir) and any(
            f.endswith(".npz") for f in os.listdir(self.depths_dir))

    # ---- 资产读取（cpu 缓存，使用时再上 GPU；npz 里 depth 是 float16）------

    def _load(self, stem):
        if stem in self._cache:
            self._cache.move_to_end(stem)
            return self._cache[stem]
        path = os.path.join(self.depths_dir, stem + ".npz")
        if not os.path.exists(path):
            return None
        with np.load(path, allow_pickle=False) as z:
            d = z["depth"].astype(np.float32)  # [h,w] 米制深度
        mono = torch.from_numpy(d)[None, None]  # [1,1,h,w]
        fg = None
        if self.use_mask:
            mp = os.path.join(self.masks_dir, stem + ".png")
            if os.path.exists(mp):
                from PIL import Image
                m = np.asarray(Image.open(mp).convert("L"), dtype=np.float32) / 255.0
                fg = torch.from_numpy(m)[None, None]
        self._cache[stem] = (mono, fg)
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return self._cache[stem]

    # ---- 逐点深度损失 ------------------------------------------------------

    def loss(self, cam, xyz, opacity):
        """cam: 训练相机（用 full_proj_transform / world_view_transform 投影）；
        xyz: [N,3] 形变后高斯中心（可导）；opacity: [N] 或 [N,1]（会 detach）。
        返回 (loss_scalar or None, info_str or None)。
        """
        if not self.available:
            return None, None
        entry = self._load(cam.image_name)
        if entry is None:
            return None, "no npz"
        mono, fg = entry
        device = xyz.device
        mono = mono.to(device)
        if fg is not None:
            fg = fg.to(device)

        N = xyz.shape[0]
        ones = torch.ones(N, 1, device=device, dtype=xyz.dtype)
        pts_h = torch.cat([xyz, ones], dim=1)                       # [N,4]

        # 投影约定修正过（2026-10-16）：原照抄 train_mask.sample_fg_prob 的
        # gy=-ndc_y 是镜像 bug，已改 gy=+ndc_y（依据见下方 loss() 内注释）。
        clip = pts_h @ cam.full_proj_transform                      # [N,4]
        w_clip = clip[:, 3]
        ndc = clip[:, :3] / w_clip.clamp_min(1e-6).unsqueeze(1)
        p_view = pts_h @ cam.world_view_transform                   # [N,4]
        z_view = p_view[:, 2]                                       # 前方为正

        # ⚠️ 2026-10-16 修正：gy 不再取负。本 fork 投影矩阵 P[1][1]=+1/tanfovy、
        # 视图矩阵 y 轴朝下（COLMAP 约定）→ ndc_y>0 就是画面下半部；grid_sample 的
        # y=+1 也是画面底部 → 正确映射是 gy=+ndc_y（无翻转）。
        # 此前照抄 deformable_human/train_mask.py sample_fg_prob 的「gy=-ndc_y」，
        # 该实现标注「qc_proj_fg.py 验证过」实为镜像错误——qc_proj_fg.py 把叠加点
        # 画在与采样同一（可能翻转的）坐标系里，点色=落点处 mask 值，任何 gy 符号
        # 下都「红点落在 fg 上」，对符号天然不敏感（无效验证）。用光栅化器自渲染
        # 深度逐点核对才暴露：gy=+ndc_y 时 corr(z_view, rendered_depth)=+0.62~+0.88
        # （4 个测试视角），gy=-ndc_y 全负（-0.48~-0.50）。镜像版会让 (a,b) 对齐
        # 算出 a<0 → 深度损失整段被「align a<=0」门限静默跳过（E 案首次冒烟 10k
        # 步 0 条 [DEPTH] iter 日志）。
        gx, gy = ndc[:, 0], ndc[:, 1]
        in_frame = (w_clip > 0) & (gx.abs() <= 1.0) & (gy.abs() <= 1.0) & (z_view > 0.2)
        n_in = int(in_frame.sum())
        if n_in < self.min_points:
            return None, "too few in frame (%d)" % n_in

        grid = torch.stack([gx, gy], dim=1).view(1, 1, N, 2)
        mono_at = F.grid_sample(mono, grid, mode="bilinear",
                                padding_mode="zeros", align_corners=False).view(N)
        if fg is not None:
            fg_at = F.grid_sample(fg, grid, mode="bilinear",
                                  padding_mode="zeros", align_corners=False).view(N)
        else:
            fg_at = torch.ones_like(mono_at)

        valid = in_frame & (mono_at > 0) & torch.isfinite(mono_at)
        wgt = opacity.reshape(-1).detach() * fg_at                  # 只监督位置
        wgt = torch.where(valid, wgt, torch.zeros_like(wgt))
        w_sum = wgt.sum()
        if float(w_sum) < self.min_points * 0.01:                    # 加权后仍要有量
            return None, "weight too low"

        # 加权最小二乘解 (a, b)：a·mono + b ≈ z_view（detach 数据求解）
        mm = mono_at.detach()
        zz = z_view.detach()
        ww = wgt.detach()
        m_mean = (ww * mm).sum() / w_sum
        z_mean = (ww * zz).sum() / w_sum
        var_m = (ww * (mm - m_mean) ** 2).sum() / w_sum
        if float(var_m) < 1e-8:
            return None, "mono flat"
        a = ((ww * (mm - m_mean) * (zz - z_mean)).sum() / w_sum) / var_m
        b = z_mean - a * m_mean
        if not (torch.isfinite(a) and torch.isfinite(b)) or float(a) <= 0:
            # a≤0 → 该帧单目深度与几何结构性矛盾（BAD frame），跳过而非硬拉
            return None, "align a<=0"

        residual = (a * mono_at + b - z_view).abs()
        l = (wgt * residual).sum() / (w_sum + 1e-6)
        info = "a={:.3f} b={:.3f} loss={:.4f} n={}k".format(
            float(a), float(b), float(l), n_in // 1024)
        return l, info
