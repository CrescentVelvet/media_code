#!/usr/bin/env python3
"""AUDIT-15a · 2DGS 光栅器 API 探针（跑 I-03 之前的自证）

实测结论（把"读源码"变成"跑出来的事实"）：

  Q1 `rasterization_2dgs` 签名里**没有 `normals` 参数** —— 盘法向在光栅器内部由
     `quats` 推出的旋转矩阵第三列决定。所以我们既传不进法向、也改不了它：
     **法向完全由"轴重排把哪个轴放到第三列"决定**。返回 7 项：
     (colors, alphas, normals, normals_from_depth, distort, median_depth, meta)
  Q2 `scales` 只用前两维、第三维硬编码 1 —— s0→0 时覆盖面积真空塌缩。
     这是 E7（DA3 最小轴在 index 0，直接喂 = 零宽刀片）的机器证明。
  Q3 ⚠️ gsplat 1.5.3 的 `rasterization_2dgs` 在 `sh_degree=None` 时**不可用**：
     "Compute the per-view colors" 那一段被整块注释掉（rendering.py:1503-1520），
     colors 停在 (N,3)，下游断言 `colors.shape[:-2] == (B,C,N)` 直接失败。
     → 只能传 SH 系数 (N,K,3) + `sh_degree`。对 I-03 不是限制（本来就要全 SH），
     但任何人想传预激活颜色都会在这里撞墙。
  Q4 `depth_mode` 可选 expected / median；`distloss=True` 时 meta 带 ray_transforms
     与 gradient_2dgs。单位量级在这里看一眼，避免后面拿错。

不需要 PLY、不需要相机文件，秒级完成。
"""
import inspect
import torch
from gsplat import rasterization_2dgs, rasterization

torch.manual_seed(0)
dev = "cuda"
W = H = 64
fx = fy = 60.0
cx = cy = (W - 1) / 2
K = torch.tensor([[[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]], device=dev)
vm = torch.eye(4, device=dev)[None].clone()
vm[0, 2, 3] = 2.0                      # 相机在 z=2，OpenCV 约定看向 +z

print("=" * 78)
print("Q1 签名（实测）")
print("=" * 78)
sig = inspect.signature(rasterization_2dgs)
print("  参数：", ", ".join(sig.parameters))
print("  含 normals ?", "normals" in sig.parameters)

N = 5
means = torch.zeros(N, 3, device=dev)
means[:, 0] = torch.linspace(-0.03, 0.03, N, device=dev)
quats = torch.tensor([[1.0, 0, 0, 0]], device=dev).repeat(N, 1)
opac = torch.full((N,), 0.95, device=dev)
# SH：DC 项 = 颜色 − 0.5（gsplat 反变换是 SH(C)+0.5），这里让输出 ≈ (0.9,0.3,0.1)
sh = torch.zeros(N, 9, 3, device=dev)
sh[:, 0] = torch.tensor([0.4, -0.2, -0.4], device=dev)

print("\n" + "=" * 78)
print("Q2 判据：前两维决定足迹，第三维被忽略（E7 的机器证明）")
print("=" * 78)
print(f"  {'切平面 s0':>11} {'法向轴 s2':>11} {'alpha>0.01 像素':>16} {'覆盖率':>9}")
for s0, s2 in ((0.05, 1.0), (0.05, 1e-8), (1e-6, 1.0), (1e-6, 1e8)):
    scales = torch.tensor([[s0, s0, s2]], device=dev).repeat(N, 1)
    a = rasterization_2dgs(means, quats, scales, opac, sh, vm, K, W, H,
                           sh_degree=2, packed=False)[1][0].squeeze(-1)
    print(f"  {s0:>11.1e} {s2:>11.1e} {int((a > 0.01).sum()):>16} "
          f"{(a > 0.01).float().mean():>8.3%}")

print("\n" + "=" * 78)
print("Q3 SH 通道 + 返回项形状")
print("=" * 78)
scales = torch.tensor([[0.05, 0.05, 1.0]], device=dev).repeat(N, 1)
img, alpha, nrm, nfrm, dist, med, meta = rasterization_2dgs(
    means, quats, scales, opac, sh, vm, K, W, H, sh_degree=2, packed=False)
def shp(x):
    return tuple(x.shape) if torch.is_tensor(x) else f"None（{x!r}）"


print(f"  ✅ colors{shp(img)} alphas{shp(alpha)} normals{shp(nrm)}")
print(f"     normals_from_depth{shp(nfrm)} distort{shp(dist)} "
      f"median_depth{shp(med)} meta[{len(meta)} key]")
print("     ⚠️ render_mode='RGB' 时 normals_from_depth / distort / median_depth 全是 None，"
      "\n        要拿它们必须 RGB+ED 或 RGB+D")
fg = alpha[0].squeeze(-1) > 0.5
print(f"  前景 {int(fg.sum())} px，渲染色均值 "
      f"{[round(float(x), 3) for x in img[0][fg].mean(0)]}  （期望 ≈ [0.9,0.3,0.1]）")

print("\n" + "=" * 78)
print("Q4 深度语义：expected vs median；与 3DGS EWA 的 expected 对照")
print("=" * 78)
for dm in ("expected", "median"):
    out = rasterization_2dgs(means, quats, scales, opac, sh, vm, K, W, H,
                             sh_degree=2, render_mode="RGB+ED", distloss=True,
                             depth_mode=dm, packed=False)
    d = out[0][0, ..., 3]
    print(f"  depth_mode={dm:<9} min={float(d.min()):.4f} max={float(d.max()):.4f} "
          f"前景中位={float(d[d > 0].median()):.4f}  distort_max={float(out[4].max()):.4f}")
    print(f"     meta: ray_transforms={'ray_transforms' in out[6]}  "
          f"gradient_2dgs={'gradient_2dgs' in out[6]}")

i3, a3, m3 = rasterization(means, quats, scales, opac, sh, vm, K, W, H,
                           sh_degree=2, render_mode="RGB+ED", packed=False)
d3 = i3[0, ..., 3]
print(f"  3DGS EWA          min={float(d3.min()):.4f} max={float(d3.max()):.4f} "
      f"前景中位={float(d3[d3 > 0].median()):.4f}")
print(f"  → 两者都约为 2.0（相机在 z=2、盘在 z=0），单位一致")

print("\n" + "=" * 78)
print("Q5 法向来源自检：quats=单位 → 第三列 = +z；渲染法向应为 +z")
print("=" * 78)
n = nrm[0]
print(f"  前景 {int(fg.sum())} px，法向均值 "
      f"{[round(float(x), 4) for x in n[fg].mean(0)]}  （期望 ≈ [0,0,1]，"
      f"若为 [0,0,-1] 则法向符号与光栅器约定相反）")

# 绕 x 轴转 90°：第三列应变成 −y（quarterion wxyz）
q90 = torch.tensor([[0.7071068, 0.7071068, 0, 0]], device=dev).repeat(N, 1)
n2 = rasterization_2dgs(means, q90, scales, opac, sh, vm, K, W, H,
                        sh_degree=2, packed=False)[2][0]
fg2 = rasterization_2dgs(means, q90, scales, opac, sh, vm, K, W, H,
                         sh_degree=2, packed=False)[1][0].squeeze(-1) > 0.5
print(f"  绕 x 转 90° 后法向均值 "
      f"{[round(float(x), 4) for x in n2[fg2].mean(0)] if fg2.any() else 'N/A'}"
      f"  （盘已侧立，第三列 = 原 −y 方向，面积应变小）")
print(f"  侧立后覆盖率 {(fg2.float().mean()):.3%} vs 正对 {(fg.float().mean()):.3%}")
