# -*- coding: utf-8 -*-
"""稠密化自动调度器（方案 D 移植，2026-10-13；2026-10-16 按服务器实录对齐口径）

来源：服务器 Reconstruction 仓 DensificationScheduler（本文件是按描述的重实现，
与官方仓解耦——不动 ~/repos/Deformable-3D-Gaussians 任何文件）。

要解决的问题（media_code 实测事故）：
  - v1：thr 0.0004 + until 2000 → 216k 点欠拟合（阈值/窗口靠人肉猜，猜高了饿死）
  - v2：thr 0.0002 + until 10000 → 点数爆炸，步速掉 13 倍，ETA 10.7h（猜低了爆炸）
  同一个固定阈值不可能适配所有场景，需要一个自调节闭环。2026-10-15 profile 进一步
  证明点数 N 是步时唯一大杠杆（deform MLP 随 N 线性），所以「点数受控」这件事
  同时是质量与速度问题。

## 口径（2026-10-16 与服务器实现对齐后的最终形态）

1. **上限来源二选一**（服务器：显式 cap 才是默认路径）
   - 显式 cap（`cap > 0`，本仓 = SOM_MAX_GAUSSIANS 默认 600000）：`max_n = cap`，
     momentum 自适应**不参与**上限计算（仅继续累计供日志/诊断）。
     cap600k A/B 已证该值质量无损（@10k −0.09dB、@20k +0.18dB，2.4× 省点）。
   - 无 cap（`cap <= 0`）：走 momentum 自适应，`max_n = init_n + momentum`。
2. **momentum（只增不减的需求累计）**：每个稠密化周期后
   `momentum = max(momentum, int(decay × momentum + min(demand_cap, n_candidates)))`
   ——`n_candidates` = **topk 截断之前**、梯度超阈值的原始候选数（需求信号）；
   decay=0.98 与 demand_cap=40000 把爆炸场景的需求增长限成渐进曲线。
   momentum 初值 = init_n × mult（稀疏初始化 ×4 / 稠密 ×0.5，auto 以 10 万为界）。
3. **目标点数与 c2f 折扣**：`next_n(scale) = init_n + (max_n − init_n) / scale`
   ——**折扣只作用于增量项**（服务器公式；旧版误写成 max_n/scale，会把 init_n 也折掉）。
   som_human 分辨率调度默认关 → scale 恒 1 → next_n = max_n。
4. **rate 上限**（`max_rate`，服务器 max_densify_rate_per_step=1.0）：相邻周期
   next_n 的相对增长上限 `next_n ≤ prev_next_n × (1 + max_rate)`。
   【口径为推断】服务器只给了常量名与 1.0 这个值，未给公式上下文；scale 恒 1 时
   next_n 首个周期即取满，该限幅仅在开 c2f 时才会生效，故风险仅限 c2f 场景。
5. **topk 截断**：本轮允许新增 `target_add = max(0, next_n − current_n)`；
   候选数超过 target 时，按**分数**只保留前 target 个再走官方 clone/split。
   - 分数口径 = `norm(accum / denom.clamp_min(cnt_clamp))`，`cnt_clamp=3.0`
     （服务器 `v_means2d_accum / cnt.clamp_min(3.0)`）——可见次数 <3 的点被钝化，
     防止「刚出现就抖动」的点抢到分裂名额。
   - 累积量本身已是 **absgrad**（AbsGS 口径）：D-3DGS 自带的 densify 梯度分支在
     CUDA backward 里用 `fabsf(...)` 逐像素累加（不抵消），经
     `viewspace_points_densify` 暴露，无需改渲染器。**不要**改用
     `viewspace_points`（那是普通有符号梯度，会抵消，与服务器口径不符）。
6. **窗口结束后只剪不增**：`next_n = max_n`、`target_add = 0`，仍按期做 prune
   （官方 `iteration < densify_until_iter` 会连 prune 一起停掉）。
   - **但窗口后的 prune 只能按 opacity，不得带尺寸判据**（2026-10-16 冒烟事故）：
     尺寸判据（`max_radii2D > 20` / `scaling > 0.1·extent`）在窗口内是安全的，因为
     每轮 prune 之后紧跟 clone/split **补点**，删掉的点会被重新分配；一旦停了
     densify，删点就没有补充来源 → 剩余点为覆盖同样面积被迫长大 → 屏幕半径再次
     超标 → 再被删，形成**死亡螺旋**。
     实测（double_wave，三次独立重跑完全一致）：窗口关闭首轮 599k → 463k，
     随后每千步 ×0.685 指数衰减 463k→286k→196k→135k→…→9.6k，PSNR 20.52→5.34；
     对照组（cap600k，官方代码、窗口后完全不 prune）@20k = 21.81 正常。
     故 `sched_prune_only` 在窗口后传 `max_screen_size=None`（只剪 opacity<min_opacity）。

与 SOM_MAX_GAUSSIANS 的关系：调度器是「软的」增长节奏控制，MAX_GAUSSIANS 是
「硬的」兜底线（同时作为显式 cap 喂给调度器），两者互补。
"""

import torch


class DensifyScheduler:
    def __init__(self, init_n, mult="auto", decay=0.98, demand_cap=40000,
                 cap=None, max_rate=1.0, cnt_clamp=3.0):
        self.init_n = int(init_n)
        if mult == "auto":
            # 文字描述的口径：COLMAP 稀疏初始化 ×4；稠密初始化（深度/采集点云）×0.5
            mult = 0.5 if self.init_n >= 100000 else 4.0
        self.mult = float(mult)
        self.momentum = self.init_n * self.mult
        self.decay = float(decay)
        self.demand_cap = int(demand_cap)
        # cap <= 0 / None → 走 momentum 自适应；否则显式 cap 路径
        self.cap = int(cap) if cap and int(cap) > 0 else None
        self.max_rate = float(max_rate) if max_rate and max_rate > 0 else None
        self.cnt_clamp = float(cnt_clamp)
        self.prev_next_n = None

    # ---- 上限与目标 -------------------------------------------------------

    def max_n(self):
        """上限点数（不含 c2f 折扣）。"""
        if self.cap is not None:
            return self.cap
        return int(self.init_n + self.momentum)

    def next_n(self, scale=1):
        """本周期目标总点数：next_n = init_n + (max_n − init_n)/scale。"""
        scale = max(float(scale), 1.0)
        target = self.init_n + (self.max_n() - self.init_n) / scale
        if self.max_rate is not None and self.prev_next_n is not None:
            target = min(target, self.prev_next_n * (1.0 + self.max_rate))
        self.prev_next_n = target
        return int(target)

    def target_add(self, current_n, scale=1, window_closed=False):
        """本轮允许新增的点数（>=0）。窗口结束后恒为 0（只剪不增）。"""
        if window_closed:
            self.prev_next_n = max(self.prev_next_n or 0, self.max_n())
            return 0
        return max(0, self.next_n(scale) - int(current_n))

    # ---- 需求累计 ---------------------------------------------------------

    def update(self, n_candidates):
        """每个稠密化周期后调用；n_candidates = 本轮 grad >= thr 的候选总数
        （**topk 截断之前**的原始计数，服务器 momentum_step 同口径）。"""
        grown = self.decay * self.momentum + min(self.demand_cap, int(n_candidates))
        self.momentum = max(self.momentum, grown)

    def state_str(self):
        next_n = int(self.prev_next_n) if self.prev_next_n else self.max_n()
        return "mode=%s momentum=%d max_n=%d next_n=%d" % (
            "cap" if self.cap is not None else "momentum",
            int(self.momentum), self.max_n(), next_n)


def grad_scores(gaussians, cnt_clamp=3.0):
    """稠密化判定/排序用的梯度分数（服务器口径）：
    norm(accum / denom.clamp_min(cnt_clamp))，accum 为 absgrad 累加量。"""
    denom = gaussians.denom.clamp_min(cnt_clamp)
    grads = gaussians.xyz_gradient_accum / denom
    grads[grads.isnan()] = 0.0
    return grads


def _prune_components(gaussians, min_opacity, extent, max_screen_size,
                      diag_screen_size=20):
    """返回 `(prune_mask, parts)`。

    `parts` 恒给出三路判据的**独立计数**（供诊断：判断一次大删到底由哪路造成），
    而 `prune_mask` 严格按 `max_screen_size` 是否为真值决定是否并入尺寸判据。
    诊断路用的屏幕半径阈值固定 `diag_screen_size`，不参与剪枝决策。
    """
    opa = (gaussians.get_opacity < min_opacity).squeeze()
    big_vs = gaussians.max_radii2D > diag_screen_size
    big_ws = gaussians.get_scaling.max(dim=1).values > 0.1 * extent
    if max_screen_size:
        prune_mask = torch.logical_or(torch.logical_or(opa, big_vs), big_ws)
    else:
        prune_mask = opa
    parts = {
        "opa": int(opa.sum().item()),
        "vs": int(big_vs.sum().item()),
        "ws": int(big_ws.sum().item()),
        "applied": int(prune_mask.sum().item()),
    }
    return prune_mask, parts


def _prune_mask(gaussians, min_opacity, extent, max_screen_size):
    return _prune_components(gaussians, min_opacity, extent, max_screen_size)[0]


def sched_prune_only(gaussians, min_opacity, extent, max_screen_size, report=False):
    """只做 prune（点数到天花板 / densify 窗口结束后的行为）。

    官方 `densify_and_prune` 是「clone+split+prune」一体的，到天花板后若直接跳过
    整个调用，低不透明度点会永久堆积（旧版硬护栏分支就是这个 bug，注释写「只 prune
    不 densify」但实际两者都没做）。这里把 prune 单独抠出来。

    ⚠️ 窗口后调用务必传 `max_screen_size=None`：尺寸判据在没有 densify 补点的阶段
    会触发死亡螺旋（见模块 docstring 第 6 条）。
    """
    mask, parts = _prune_components(gaussians, min_opacity, extent, max_screen_size)
    gaussians.prune_points(mask)
    torch.cuda.empty_cache()
    return parts if report else None


def sched_densify_and_prune(gaussians, max_grad, min_opacity, extent,
                            max_screen_size, target_add, cnt_clamp=3.0,
                            report=False):
    """官方 densify_and_prune 的 topk 截断版（fork，逻辑与官方逐行一致，
    唯一差异：候选超过 target_add 时按分数只留前 target_add 个）。

    返回本轮截断前的候选数（供 DensifyScheduler.update 作为需求信号）；
    `report=True` 时返回 `(n_cand, parts)`，parts 为 prune 三路判据计数。
    """
    grads = grad_scores(gaussians, cnt_clamp)

    # 候选口径与官方两路完全一致：clone/split 都用 norm(grads) >= thr，
    # 仅靠 scaling 大小区分去 clone 还是 split（互斥分家，并集 = grad >= thr）
    scores = torch.norm(grads, dim=-1).squeeze()  # [N]
    n_cand = int((scores >= max_grad).sum().item())

    if target_add > 0:
        if target_add < n_cand:
            # topk 截断：非前 target_add 名的梯度置零，官方阈值逻辑自然跳过
            # （kth 处有同分时可能略超 target_add，属可接受上溢）
            kth = torch.topk(scores, target_add).values[-1]
            grads = grads.clone()
            grads[scores < kth] = 0.0

        # 以下与官方 densify_and_prune 逐行一致（scene/gaussian_model.py:365）
        gaussians.densify_and_clone(grads, max_grad, extent)
        gaussians.densify_and_split(grads, max_grad, extent)

    mask, parts = _prune_components(gaussians, min_opacity, extent, max_screen_size)
    gaussians.prune_points(mask)
    torch.cuda.empty_cache()
    return (n_cand, parts) if report else n_cand


# --------------------------------------------------------------------------
# 自测：python densify_sched.py（不需要 CUDA；用桩对象替代 GaussianModel）
# --------------------------------------------------------------------------
def _self_test():
    class _StubGaussians:
        """最小桩：只实现 clone/split/prune 的计数语义（+1/+1/-k）。"""

        def __init__(self, n, scores, small_ratio=0.5):
            self.n = int(n)
            self._scores = torch.as_tensor(scores, dtype=torch.float32).reshape(-1, 1)
            self.denom = torch.ones(self.n, 1)
            self.xyz_gradient_accum = self._scores.clone()
            self.n_small = int(self.n * small_ratio)
            self._opacity = torch.ones(self.n, 1)
            self.max_radii2D = torch.zeros(self.n)
            self._scaling = torch.ones(self.n, 3)
            self.clone_calls = 0
            self.split_calls = 0

        def _selected(self, grads, max_grad, small):
            sel = (torch.norm(grads, dim=-1) >= max_grad).squeeze()
            idx = torch.arange(self.n)
            size_ok = idx < self.n_small
            return int((sel & size_ok if small else sel & ~size_ok).sum().item())

        def densify_and_clone(self, grads, max_grad, extent):
            self.clone_calls += 1
            self._pending_clone = self._selected(grads, max_grad, small=True)

        def densify_and_split(self, grads, max_grad, extent):
            self.split_calls += 1
            self._pending_split = self._selected(grads, max_grad, small=False)

        def prune_points(self, mask):
            self.n -= int(mask.sum().item())

        # 与 GaussianModel 一致：这两个是 property，不是方法
        @property
        def get_opacity(self):
            return self._opacity

        @property
        def get_scaling(self):
            return self._scaling

        # 模拟一轮：返回本轮净新增
        def apply(self):
            added = getattr(self, "_pending_clone", 0) + getattr(self, "_pending_split", 0)
            self.n += added
            return added

    def new_stub(n, n_over=0):
        scores = torch.full((n, 1), 0.0001)
        if n_over:
            scores[:n_over, 0] = torch.linspace(0.001, 0.01, n_over)
        return _StubGaussians(n, scores.squeeze(-1).tolist())

    # 1) 显式 cap：max_n 恒为 cap，momentum 不参与
    s = DensifyScheduler(50000, cap=600000)
    assert s.max_n() == 600000, s.max_n()
    assert "mode=cap" in s.state_str(), s.state_str()  # 格式串别混用 {} 与 %d
    s.update(40000)
    s.update(40000)
    assert s.max_n() == 600000, "cap>0 时 momentum 不得影响上限"
    assert s.momentum > 50000 * 4, "momentum 仍应累计（供诊断）"

    # 2) 无 cap：momentum 自适应，单调不减
    s2 = DensifyScheduler(50000, cap=0)
    m0 = s2.momentum
    assert m0 == 50000 * 4, m0
    assert s2.max_n() == 50000 + m0
    s2.update(40000)
    assert s2.momentum >= m0 and s2.momentum == int(0.98 * m0 + 40000), s2.momentum
    m1 = s2.momentum
    s2.update(0)  # 需求为 0 也不回落
    assert s2.momentum == m1, "momentum 只增不减"

    # 3) c2f 折扣只作用于增量项（关掉 rate 限幅，单独验 c2f 公式）
    s3 = DensifyScheduler(50000, cap=650000, max_rate=None)
    assert s3.next_n(scale=4) == 50000 + (650000 - 50000) // 4, s3.next_n(scale=4)
    assert s3.next_n(scale=1) == 650000

    # 4) rate 上限（推断口径）：相邻周期相对增长 ≤ 1+max_rate
    s4 = DensifyScheduler(50000, cap=650000, max_rate=1.0)
    s4.prev_next_n = 200000.0
    assert s4.next_n(scale=1) == 400000, s4.next_n(scale=1)
    s5 = DensifyScheduler(50000, cap=650000, max_rate=None)
    s5.prev_next_n = 200000.0
    assert s5.next_n(scale=1) == 650000, "关闭 rate 时不受限"

    # 5) 窗口结束后只剪不增
    s6 = DensifyScheduler(50000, cap=600000)
    assert s6.target_add(600000, window_closed=True) == 0
    assert s6.target_add(300000, window_closed=True) == 0, "窗口后不得再加点"

    # 6) target_add 基本语义
    s7 = DensifyScheduler(50000, cap=600000)
    assert s7.target_add(599000) == 1000
    assert s7.target_add(700000) == 0, "已超 cap 时必须为 0"

    # 7) topk 精确截断 + 需求信号为截断前计数
    g = new_stub(1000, n_over=100)
    s8 = DensifyScheduler(1000, cap=1000000, cnt_clamp=1.0)
    n_cand = sched_densify_and_prune(g, 0.0005, 0.005, 1.0, None, 30, s8.cnt_clamp)
    added = g.apply()
    assert n_cand == 100, n_cand
    assert added == 30, "target=30 应精确新增 30（实际 %d）" % added

    # 8) target=0 → 不新增（只 prune）
    g2 = new_stub(1000, n_over=100)
    n_cand2 = sched_densify_and_prune(g2, 0.0005, 0.005, 1.0, None, 0, 1.0)
    added2 = g2.apply()
    assert n_cand2 == 100 and added2 == 0, (n_cand2, added2)
    assert g2.clone_calls == 0 and g2.split_calls == 0

    # 9) cnt_clamp 语义：可见次数少的点被钝化（1 次 vs 3 次）
    g3 = _StubGaussians(2, [0.9, 0.9])
    g3.xyz_gradient_accum = torch.tensor([[0.9], [0.9]])
    g3.denom = torch.tensor([[1.0], [3.0]])
    sc = grad_scores(g3, cnt_clamp=3.0).squeeze()
    assert abs(float(sc[0]) - 0.3) < 1e-6, float(sc[0])
    assert abs(float(sc[1]) - 0.3) < 1e-6, float(sc[1])
    sc_raw = grad_scores(g3, cnt_clamp=1.0).squeeze()
    assert abs(float(sc_raw[0]) - 0.9) < 1e-6 and abs(float(sc_raw[1]) - 0.3) < 1e-6

    # 10) prune-only 不再触发 clone/split
    g4 = new_stub(1000)
    g4._opacity[:10] = 0.0
    sched_prune_only(g4, 0.005, 1.0, None)
    assert g4.n == 990, g4.n

    # 11) 三路判据计数 + 窗口后尺寸判据必须被忽略（死亡螺旋防线）
    #     构造：10 个低透明、20 个屏幕半径超标、30 个世界尺度超标（互不重叠）
    g5 = _StubGaussians(100, [0.001] * 100)
    g5._opacity = torch.ones(100, 1)
    g5._opacity[:10, 0] = 0.0
    g5.max_radii2D[10:30] = 999.0
    g5._scaling = torch.full((100, 3), 0.01)  # < 0.1*extent(=1.0)
    g5._scaling[30:60] = 999.0
    # 带尺寸判据：三路并集 = 60
    _, parts = _prune_components(g5, 0.005, 1.0, 20)
    assert parts == {"opa": 10, "vs": 20, "ws": 30, "applied": 60}, parts
    # 不带尺寸判据：只按 opacity → 10
    _, parts2 = _prune_components(g5, 0.005, 1.0, None)
    assert parts2["opa"] == 10 and parts2["applied"] == 10, parts2
    assert parts2["vs"] == 20 and parts2["ws"] == 30, "诊断计数不受剪枝开关影响"
    # prune-only 端口径一致
    g6 = _StubGaussians(100, [0.001] * 100)
    g6._opacity = torch.ones(100, 1)
    g6._opacity[:10, 0] = 0.0
    g6.max_radii2D[10:30] = 999.0
    g6._scaling = torch.full((100, 3), 0.01)
    g6._scaling[30:60] = 999.0
    p = sched_prune_only(g6, 0.005, 1.0, None, report=True)
    assert g6.n == 90 and p["applied"] == 10, (g6.n, p)

    print("[OK] densify_sched 自测 11 组全过")


if __name__ == "__main__":
    _self_test()
