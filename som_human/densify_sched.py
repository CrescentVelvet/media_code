# -*- coding: utf-8 -*-
"""稠密化自动调度器（方案 D 移植，2026-10-13）

来源：服务器 Reconstruction 仓 DensificationScheduler（只有文字描述，本文件是
按描述的重实现，与官方仓解耦——不动 ~/repos/Deformable-3D-Gaussians 任何文件）。

要解决的问题（media_code 实测事故）：
  - v1：thr 0.0004 + until 2000 → 216k 点欠拟合（阈值/窗口靠人肉猜，猜高了饿死）
  - v2：thr 0.0002 + until 10000 → 点数爆炸，步速掉 13 倍，ETA 10.7h（猜低了爆炸）
  同一个固定阈值不可能适配所有场景，需要一个自调节闭环。

机制（按文字描述忠实移植）：
  1. 自适应软上限：max_n = init_n + momentum；
     momentum 初始 = init_n × mult（稀疏初始化 ×4 / 稠密初始化 ×0.5，
     auto 规则：init_n >= 10 万判稠密——COLMAP 稀疏点云典型 1~5 万）。
  2. 需求累计（只增不减）：每个稠密化周期后
     momentum = max(momentum, decay × momentum + min(demand_cap, 本轮超阈值候选数))
     ——decay=0.98 与 demand_cap=40000 把爆炸场景的需求增长限制成渐进曲线。
  3. topk 截断：本轮目标增量 = max_n(scale) − current_n（c2f 期间 ÷scale，
     我们分辨率调度默认关，scale 恒 1）；候选（grad ≥ thr，与官方口径完全一致）
     超过目标时只保留梯度最大的前 target 个（预过滤 grads 后调官方
     densify_and_clone/split，选择逻辑一字不改）。

与 SOM_MAX_GAUSSIANS 的关系：调度器是「软的」增长速率控制，MAX_GAUSSIANS 是
「硬的」兜底线（也作为 absolute_cap 喂给调度器），两者互补。
"""

import torch


class DensifyScheduler:
    def __init__(self, init_n, mult="auto", decay=0.98, demand_cap=40000,
                 absolute_cap=None):
        self.init_n = int(init_n)
        if mult == "auto":
            # 文字描述的口径：COLMAP 稀疏初始化 ×4；稠密初始化（深度/采集点云）×0.5
            mult = 0.5 if self.init_n >= 100000 else 4.0
        self.mult = float(mult)
        self.momentum = self.init_n * self.mult
        self.decay = float(decay)
        self.demand_cap = int(demand_cap)
        self.absolute_cap = int(absolute_cap) if absolute_cap else None

    def max_n(self, scale=1):
        cap = self.init_n + self.momentum
        if self.absolute_cap:
            cap = min(cap, self.absolute_cap)
        # c2f 耦合：低分辨率段不需要那么多点（我们 scale 恒 1，保留接口）
        return int(cap / max(scale, 1))

    def target_add(self, current_n, scale=1):
        """本轮允许新增的点数（>=0）。"""
        return max(0, self.max_n(scale) - int(current_n))

    def update(self, n_candidates):
        """每个稠密化周期后调用；n_candidates = 本轮 grad >= thr 的候选总数
        （截断前的「需求」信号）。"""
        grown = self.decay * self.momentum + min(self.demand_cap, int(n_candidates))
        self.momentum = max(self.momentum, grown)

    def state_str(self):
        return "momentum=%d max_n=%d" % (int(self.momentum), self.max_n())


def sched_densify_and_prune(gaussians, max_grad, min_opacity, extent,
                            max_screen_size, target_add):
    """官方 densify_and_prune 的 topk 截断版（fork，逻辑与官方逐行一致，
    唯一差异：候选超过 target_add 时按梯度只留前 target_add 个）。

    返回本轮截断前的候选数（供 DensifyScheduler.update 作为需求信号）。
    """
    grads = gaussians.xyz_gradient_accum / gaussians.denom
    grads[grads.isnan()] = 0.0

    # 候选口径与官方两路完全一致：clone/split 都用 norm(grads) >= thr，
    # 仅靠 scaling 大小区分去 clone 还是 split（互斥分家，并集 = grad >= thr）
    scores = torch.norm(grads, dim=-1).squeeze()  # [N]
    n_cand = int((scores >= max_grad).sum().item())

    if target_add <= 0:
        # 软上限已满：只 prune 不 densify（与 MAX_GAUSSIANS 硬护栏行为一致）
        pass
    else:
        if target_add < n_cand:
            # topk 截断：非前 target_add 名的梯度置零，官方阈值逻辑自然跳过
            # （kth 处有同分时可能略超 target_add，属可接受上溢）
            kth = torch.topk(scores, target_add).values[-1]
            grads = grads.clone()
            grads[scores < kth] = 0.0

        # 以下与官方 densify_and_prune 逐行一致（scene/gaussian_model.py:365）
        gaussians.densify_and_clone(grads, max_grad, extent)
        gaussians.densify_and_split(grads, max_grad, extent)

    prune_mask = (gaussians.get_opacity < min_opacity).squeeze()
    if max_screen_size:
        big_points_vs = gaussians.max_radii2D > max_screen_size
        big_points_ws = gaussians.get_scaling.max(dim=1).values > 0.1 * extent
        prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs),
                                      big_points_ws)
    gaussians.prune_points(prune_mask)

    torch.cuda.empty_cache()
    return n_cand
