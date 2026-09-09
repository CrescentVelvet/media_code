#!/usr/bin/env python3
"""验证 pytorch3d CUDA 算子（阶段七 point-to-triangle 切分依赖）。"""
import torch
from pytorch3d import _C  # noqa: F401  编译扩展存在性
from pytorch3d.ops import knn_points, point_mesh_face_distance
from pytorch3d.structures import Meshes

print("pytorch3d _C OK")

# CUDA KNN（阶段六 AvatarGaussian init 用）
p = torch.randn(1000, 3).cuda()
out = knn_points(p[None], p[None], K=3)
print("cuda knn OK:", tuple(out.dists.shape))

# point-to-triangle（阶段七 BodyGaussian 切分核心算子）
verts = torch.rand(50, 3).cuda()
faces = torch.tensor([[i, (i + 1) % 50, (i + 2) % 50] for i in range(48)]).cuda()
mesh = Meshes(verts=[verts], faces=[faces])
pts = torch.rand(200, 3).cuda()
d = point_mesh_face_distance(mesh, Meshes(verts=[pts[None]], faces=[torch.tensor([[0, 1, 2]], device="cuda")]))
print("point_mesh_face_distance OK:", float(d))

print("ALL PASS")
