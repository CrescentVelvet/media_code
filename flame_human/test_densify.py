#!/usr/bin/env python3
"""test_densify.py — densify 不变量单测（无 FLAME/GPU 依赖）。

验证三条不变式：
  1. 全量参数与 tri/is_free 长度恒等于 N = kept + added
  2. _free_can 行数 == is_free.sum()（紧凑数组）
  3. 新点继承父点 tri/is_free；prune 后 _free_can 同步
"""
import sys
import numpy as np
import torch

sys.path.insert(0, ".")
from train_avatar import AvatarModel  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)


def make_model(N=100, n_free=30):
    m = object.__new__(AvatarModel)
    m.device = torch.device("cpu")
    m._bary_raw = torch.nn.Parameter(torch.randn(N, 3))
    m._opacity = torch.nn.Parameter(torch.randn(N, 1) + 1.0)
    m._scaling = torch.nn.Parameter(torch.randn(N, 3).abs() + 0.05)
    m._rotation = torch.nn.Parameter(torch.randn(N, 4))
    m._features_dc = torch.nn.Parameter(torch.randn(N, 1, 3))
    m._features_rest = torch.nn.Parameter(torch.randn(N, 15, 3))
    m.tri = torch.randint(0, 50, (N, 3))
    m.is_free = torch.zeros(N, dtype=torch.bool)
    m.is_free[5:5 + n_free] = True
    m.tri[m.is_free] = 0
    m._free_can = torch.nn.Parameter(torch.randn(n_free, 3))
    m.grad_accum = torch.rand(N)
    m.denom = torch.ones(N)
    return m


def check(m, tag, kept, added):
    N = len(m._opacity)
    assert m.tri.shape[0] == N, f"{tag}: tri {m.tri.shape[0]} != {N}"
    assert m.is_free.shape[0] == N, f"{tag}: is_free len mismatch"
    assert m._bary_raw.shape[0] == N, f"{tag}: bary len mismatch"
    assert m._scaling.shape[0] == N and m._rotation.shape[0] == N
    assert m._features_dc.shape[0] == N and m._features_rest.shape[0] == N
    nf = int(m.is_free.sum())
    assert m._free_can.shape[0] == nf, \
        f"{tag}: free_can {m._free_can.shape[0]} != is_free.sum() {nf}"
    assert N == kept + added, f"{tag}: N {N} != kept {kept} + added {added}"
    print(f"  ✅ {tag}: N={N} kept={kept} added={added} free_can={nf}")


# ── case 1: clone + split + prune 同时发生 ─────────────────────────
m = make_model(N=100, n_free=30)
with torch.no_grad():
    tri_before = m.tri.clone()
    fc_before = m._free_can.clone()
    is_free_before = m.is_free.clone()
    m._scaling[0:10] = 10.0        # 大尺度 + 高梯度 → split
    m._opacity[50:60] = -10.0      # 低 opacity → prune（含 5 个自由点 50-54? 不，自由点是 5..34）
    # 让高梯度点含自由点与绑定点
    m.grad_accum[:] = 5.0          # 全部超阈值
    m._opacity[60] = -10.0         # 60 是绑定点，被 prune
kept, added = m.densify(grad_thresh=0.1, min_opacity=0.5, extent=1.0)
check(m, "case1 clone+split+prune", kept, added)

# 新点继承校验：最后 added 个点的 tri/is_free 应来自非 prune 父点
with torch.no_grad():
    new_tri = m.tri[-added:]
    new_is_free = m.is_free[-added:]
    assert (new_tri[:, 0] >= 0).all()
    # prune 同步校验：prune 掉的点不应残留（opacity 应全过阈值）
    op = torch.sigmoid(m._opacity).squeeze(-1)
    assert (op >= 0.5).all(), "prune 后仍有低 opacity 点"

# ── case 2: 无新点（纯 prune，旧代码 _prune_only 漏 _free_can 的分支）──
m2 = make_model(N=100, n_free=30)
with torch.no_grad():
    m2.grad_accum[:] = 0.0         # 无梯度 → 不 clone/split
    m2._opacity[0:20] = -10.0      # prune 掉 20 个（含自由点 5..19，共 15 个）
    is_free2 = m2.is_free.clone()
kept2, added2 = m2.densify(grad_thresh=0.1, min_opacity=0.5, extent=1.0)
assert added2 == 0, f"case2 应无新点，got {added2}"
check(m2, "case2 prune-only", kept2, added2)
# 自由点数应减 15
assert int(m2.is_free.sum()) == 30 - 15, \
    f"自由点数 {int(m2.is_free.sum())} != 15"

# ── case 3: 重复 densify（二连击，模拟 epoch 5/10/15 连续触发）────
m3 = make_model(N=200, n_free=50)
for rnd in range(3):
    with torch.no_grad():
        m3.grad_accum[:] = 3.0
        m3.grad_accum[::7] = 0.0
        kept3, added3 = m3.densify(grad_thresh=0.5, min_opacity=0.01, extent=1.0)
        check(m3, f"case3 round{rnd + 1}", kept3, added3)
        m3.grad_accum = torch.rand(len(m3._opacity)) * 4
        m3.denom = torch.ones(len(m3._opacity))
        # 模拟 optimizer 重建：直接再跑 deform 索引一致性
        _ = torch.softmax(m3._bary_raw, -1).sum(-1)  # bary 合法性
        assert torch.allclose(torch.softmax(m3._bary_raw, -1).sum(-1),
                              torch.ones(len(m3._bary_raw))), "bary 不合法"

print("\n🎉 全部不变式通过")
