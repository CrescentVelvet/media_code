#!/usr/bin/env python3
"""bench_step_matmul.py — SoM step() gather-blend vs matmul-blend 微基准。

验证两件事：
1. 数值等价：matmul 版与 gather 版前向输出 + 对 rot6d/transl 的梯度一致。
   ⚠️ 容差必须用相对误差——梯度是 N 项累加和，量级 ~N，绝对容差无意义。
2. 加速倍率：N=216k（v1 规模）双路径对比；N=1.5M 只测新路径
   （旧路径在训练并发下有 OOM 前科，理论外推 ×7）。

用法（WSL，deformable_human env）：
    python som_human/bench_step_matmul.py
"""
import torch


def blend_old(w, rot6d, transl, t_idx):
    """现网旧实现：gather 出 N×B×6 再加权求和。"""
    r6 = (w.unsqueeze(-1) * rot6d[t_idx]).sum(dim=1)
    tr = (w.unsqueeze(-1) * transl[t_idx]).sum(dim=1)
    return r6, tr


def blend_new(w, rot6d, transl, t_idx):
    """matmul 版：训练单步内 t_idx 全场唯一 → [N,B]@[B,6] 严格等价。"""
    t = t_idx[0]
    assert bool((t_idx == t).all()), "训练步内 t_idx 应唯一"
    r6 = w @ rot6d[t]
    tr = w @ transl[t]
    return r6, tr


def rel_err(a, b):
    return ((a - b).abs().max() / (b.abs().max() + 1e-12)).item()


def bench(fn, w, rot6d, transl, t_idx, iters=50):
    for _ in range(5):
        r6, tr = fn(w, rot6d, transl, t_idx)
        (r6.sum() + tr.sum()).backward()
        rot6d.grad = transl.grad = None
    torch.cuda.synchronize()
    start = torch.cuda.Event(True)
    end = torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        r6, tr = fn(w, rot6d, transl, t_idx)
        (r6.sum() + tr.sum()).backward()
        rot6d.grad = transl.grad = None
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    torch.manual_seed(0)
    dev = "cuda"
    T, B = 160, 10
    for N, run_old in ((216_000, True), (1_500_000, False)):
        w = torch.softmax(torch.randn(N, B, device=dev), dim=-1)
        rot6d = torch.randn(T, B, 6, device=dev, requires_grad=True)
        transl = torch.randn(T, B, 3, device=dev, requires_grad=True)
        t_idx = torch.full((N,), 7, device=dev, dtype=torch.long)

        r6_n, tr_n = blend_new(w, rot6d, transl, t_idx)
        g_rot_n, g_tr_n = torch.autograd.grad(
            r6_n.sum() + tr_n.sum(), [rot6d, transl], retain_graph=False)
        t_new = bench(blend_new, w, rot6d, transl, t_idx)

        if run_old:
            r6_o, tr_o = blend_old(w, rot6d, transl, t_idx)
            g_rot_o, g_tr_o = torch.autograd.grad(
                r6_o.sum() + tr_o.sum(), [rot6d, transl], retain_graph=False)
            t_old = bench(blend_old, w, rot6d, transl, t_idx)
            print("N={:>9,} rel_err: out={:.2e}/{:.2e} grad={:.2e}/{:.2e}  "
                  "old={:7.2f}ms new={:6.2f}ms speedup={:.1f}x".format(
                      N, rel_err(r6_n, r6_o), rel_err(tr_n, tr_o),
                      rel_err(g_rot_n, g_rot_o), rel_err(g_tr_n, g_tr_o),
                      t_old, t_new, t_old / max(t_new, 1e-9)))
        else:
            print("N={:>9,} new={:6.2f}ms（旧路径跳过，按线性外推约 {:.0f}ms）".format(
                N, t_new, 495.87 * N / 216_000))


if __name__ == "__main__":
    main()
