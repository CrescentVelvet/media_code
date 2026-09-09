#!/usr/bin/env python3
"""验证 3DGS CUDA 子模块（diff-gaussian-rasterization / simple-knn）。"""
import torch

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
print("diff_gaussian_rasterization import OK")

from simple_knn._C import distCUDA2
print("simple_knn import OK")

x = torch.randn(100, 3).cuda().requires_grad_(True)
d = distCUDA2(x[None])
print("simple_knn cuda OK:", tuple(d.shape))

# rasterizer 构造（不跑真渲染，只验证 CUDA 符号链接完整）
settings = GaussianRasterizationSettings(
    image_height=64, image_width=64, tanfovx=0.5, tanfovy=0.5,
    bg=torch.zeros(3, device="cuda"), scale_modifier=1.0,
    viewmatrix=torch.eye(4, device="cuda"), projmatrix=torch.eye(4, device="cuda"),
    sh_degree=0, campos=torch.zeros(3, device="cuda"),
    prefiltered=False, debug=False,
    antialiasing=False,
)
r = GaussianRasterizer(raster_settings=settings)
print("GaussianRasterizer construct OK:", type(r).__name__)

print("ALL PASS")
