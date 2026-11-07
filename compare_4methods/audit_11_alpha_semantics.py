#!/usr/bin/env python3
"""判定 gsplat 1.5.3 的 rasterization 是否对 opacities 再做一次 sigmoid。

做法：单个高斯正对相机，扫 opacity 输入值，读回 alpha。
  alpha ≈ 输入值         → rasterizer 期望「已激活」的 α（不再 sigmoid）
  alpha ≈ sigmoid(输入)  → rasterizer 内部还会 sigmoid（期望 logit）
"""
import torch
from gsplat import rasterization

dev = "cuda"
W = H = 64
means = torch.tensor([[0.0, 0.0, 2.0]], device=dev)
quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=dev)      # wxyz 恒等
scales = torch.tensor([[0.2, 0.2, 1e-4]], device=dev)
sh = torch.zeros(1, 1, 3, device=dev)                          # DC=0 → 黑色
K = torch.tensor([[[41.0, 0, 32.0], [0, 41.0, 32.0], [0, 0, 1]]], device=dev)
vm = torch.eye(4, device=dev)[None]

print(f"{'opacity 输入':>14} {'alpha 读回':>10} {'sigmoid(输入)':>14} {'判定':>16}")
for o in [1e-4, 0.036, 0.2, 0.5, 2.0, 8.0]:
    op = torch.tensor([o], device=dev)
    with torch.no_grad():
        img, alpha, _ = rasterization(means, quats, scales, op, sh, vm, K, W, H,
                                      sh_degree=0)
    a = float(alpha[0, 32, 32])
    s = float(torch.sigmoid(torch.tensor(o)))
    tag = "≈sigmoid(输入)" if abs(a - s) < 0.02 and abs(a - o) > 0.02 else (
        "≈输入" if abs(a - o) < 0.02 else "都不像")
    print(f"{o:>14.4f} {a:>10.4f} {s:>14.4f} {tag:>16}")
