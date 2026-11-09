#!/usr/bin/env python3
"""诊断 06 v3 的盘面法向朝向：identity quat 是否让法向锁在世界 +Z（掠射）。

gsplat 2DGS 的盘法向 = 旋转矩阵第三列（ERRATA E10），identity quat → 法向 [0,0,1]。
DA3 视角绕人一圈，若法向全朝世界 +Z，则多数视角是掠射 → 盘面投影成线 → 覆盖不足。

输出两组：
  1) cos(盘法线, 源相机方向) 分布（理想 |cos|≈1；接近 0 = 掠射）
  2) 零训练对照渲染 PSNR：as-is vs 强制盘面朝源相机
"""
import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization_2dgs

OUT = "/home/velvet/output/da3_human_results/da3_2dgs_v3/ours"
INFER = "/home/velvet/output/da3_human_results/da3_inference/studio_human"
IMGS = "/home/velvet/output/vggt_human_results/source/images"
DEV = "cuda"
W, H = 1440, 1920
EVAL_VIEWS = [0, 30, 60, 90]


def quat_to_third_col(q):
    """wxyz quat → 旋转矩阵第三列（2DGS 盘法线）"""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    nx = 2 * (x * z + w * y)
    ny = 2 * (y * z - w * x)
    nz = 1 - 2 * (x * x + y * y)
    return torch.stack([nx, ny, nz], -1)


def rot_between(a, b):
    """最短旋转 quat（wxyz），把 a 转到 b；处理反向退化"""
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    v = torch.cross(a, b, dim=-1)
    c = (a * b).sum(-1, keepdim=True)
    q = torch.cat([1.0 + c, v], -1)
    bad = (1.0 + c.squeeze(-1)) < 1e-6
    if bad.any():
        alt = torch.tensor([0.0, 1.0, 0.0, 0.0], device=a.device).expand_as(q)
        q = torch.where(bad[:, None], alt, q)
    return F.normalize(q, dim=-1)


def main():
    ck = np.load(f"{OUT}/checkpoint_final.npz")
    z = np.load(f"{INFER}/exports/mini_npz/results.npz")
    ext = z["extrinsics"]
    nv = len(ext)
    bottom = np.tile(np.array([[[0.0, 0.0, 0.0, 1.0]]], np.float32), (nv, 1, 1))
    w2c = np.concatenate([ext, bottom], axis=1)
    K = (z["intrinsics"] * (H / z["depth"].shape[1])).astype(np.float32)
    K[:, 2, 2] = 1.0

    imgs = sorted(__import__("pathlib").Path(IMGS).glob("*.jpg"))
    assert len(imgs) == nv, f"图像 {len(imgs)} != 视角 {nv}"

    means = torch.from_numpy(ck["means"]).to(DEV)
    quats = F.normalize(torch.from_numpy(ck["quats"]).to(DEV), dim=-1)
    scales = torch.from_numpy(ck["scales"]).to(DEV)
    opac = torch.from_numpy(ck["opacities"]).to(DEV)
    sh = torch.from_numpy(ck["sh"]).to(DEV)
    anchor = torch.from_numpy(ck["anchor"]).to(DEV)
    src = torch.from_numpy(ck["src_view"].astype(np.int64)).to(DEV)

    n_hat = quat_to_third_col(quats)
    ident_frac = (quats[:, 1:].abs().max(1).values < 1e-6).float().mean().item()

    w2c_t = torch.from_numpy(w2c).to(DEV)
    K_t = torch.from_numpy(K).to(DEV)
    c2w = torch.linalg.inv(w2c_t)
    campos = c2w[:, :3, 3]
    viewdir = F.normalize(campos[src] - anchor, dim=-1)  # anchor → 源相机

    cos_src = (n_hat * viewdir).sum(-1)
    cos_z = n_hat[:, 2]
    print("=" * 70)
    print(f"高斯数: {len(means):,} | identity quat 占比: {ident_frac:.4f}")
    print(f"cos(盘法线, 源相机方向): mean {cos_src.mean():+.3f}  med {cos_src.median():+.3f}  "
          f"p10 {cos_src.quantile(0.1):+.3f}  p90 {cos_src.quantile(0.9):+.3f}")
    print(f"  |cos|>0.9（正对）占比: {(cos_src.abs() > 0.9).float().mean():.4f} | "
          f"|cos|<0.3（掠射）占比: {(cos_src.abs() < 0.3).float().mean():.4f}")
    print(f"cos(盘法线, 世界+Z): med {cos_z.median():+.3f} | "
          f"|cos|>0.9 占比: {(cos_z.abs() > 0.9).float().mean():.4f}")
    print("=" * 70, flush=True)

    def render_psnr(tag, q_use, views=EVAL_VIEWS):
        ps, alphas = [], []
        for vi in views:
            rend, alpha, *_ = rasterization_2dgs(
                means, q_use, scales, opac, sh,
                w2c_t[vi:vi + 1], K_t[vi:vi + 1], W, H,
                render_mode="RGB", sh_degree=3,
                backgrounds=torch.zeros(1, 3, device=DEV),
                eps2d=0.3, near_plane=0.01, far_plane=100.0, packed=False)
            r = rend[0, :, :, :3].permute(2, 0, 1).clamp(0, 1)
            g = torch.from_numpy(
                cv2.cvtColor(cv2.imread(str(imgs[vi])), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            ).permute(2, 0, 1).to(DEV)
            ps.append(-10 * math.log10(max(((r - g) ** 2).mean().item(), 1e-12)))
            if vi == views[0]:
                a = alpha[0, :, :, 0]
                alphas = [a.mean().item(), (a > 0.5).float().mean().item()]
                arr = (r.detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                cv2.imwrite(f"/tmp/diag_{tag}_v{vi}.png", arr[:, :, ::-1])
        print(f"{tag:26s} PSNR {np.mean(ps):6.2f}  {np.round(ps, 1)} | "
              f"alpha mean {alphas[0]:.3f} |a>0.5| {alphas[1]:.3f}")
        return float(np.mean(ps))

    q_face = rot_between(torch.tensor([0.0, 0.0, 1.0], device=DEV).expand_as(viewdir), viewdir)

    render_psnr("as-is (训练后)", quats)
    q_ident = torch.tensor([1.0, 0.0, 0.0, 0.0], device=DEV).repeat(len(means), 1)
    render_psnr("identity(全朝世界+Z)", q_ident)
    render_psnr("盘面朝源相机", q_face)
    # 混合：只把掠射的改成朝相机，其余不动（避免破坏已学好的）
    graze = (cos_src.abs() < 0.3)
    q_mix = torch.where(graze[:, None], q_face, quats)
    print(f"  （掠射高斯数: {int(graze.sum()):,} / {len(graze):,} = {graze.float().mean():.3f}）")
    render_psnr("仅掠射档改朝相机", q_mix)


if __name__ == "__main__":
    main()
