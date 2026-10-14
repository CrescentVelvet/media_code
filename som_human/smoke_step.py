#!/usr/bin/env python3
"""smoke_step.py — 修补后 som_motion.step() 冒烟：形状/有限性/与旧公式对齐。"""
import sys

import torch

sys.path.insert(0, "/mnt/c/code/media_code/som_human")
sys.path.insert(0, "/home/velvet/repos/Deformable-3D-Gaussians")

from som_motion import SoMMotionModel  # noqa: E402


def main():
    torch.manual_seed(1)
    m = SoMMotionModel(num_frames=160, num_bases=10)
    N = 5000
    xyz = torch.randn(N, 3, device="cuda")
    t = torch.full((N, 1), 0.5, device="cuda")
    rot = torch.randn(N, 4, device="cuda")
    rot = rot / rot.norm(dim=-1, keepdim=True)

    d_xyz, d_rot, _ = m.step(xyz, t, rot)

    # 旧 gather 公式手算对照（同一权重与参数）
    w = torch.softmax(m.coef_net(xyz), dim=-1)
    t_idx = (t.squeeze(-1) * (m.num_frames - 1)).round().long()
    r6_ref = (w.unsqueeze(-1) * m.rot6d[t_idx]).sum(dim=1)
    tr_ref = (w.unsqueeze(-1) * m.transl[t_idx]).sum(dim=1)

    from som_motion import rot6d_to_matrix
    R_ref = rot6d_to_matrix(r6_ref)
    x_ref = torch.bmm(R_ref, xyz.unsqueeze(-1)).squeeze(-1) + tr_ref
    d_xyz_ref = x_ref - xyz

    err = (d_xyz - d_xyz_ref).abs().max().item()
    print("shapes:", tuple(d_xyz.shape), tuple(d_rot.shape))
    print("d_xyz max_abs_err vs old formula: %.2e" % err)
    print("finite:", bool(torch.isfinite(d_xyz).all() and torch.isfinite(d_rot).all()))
    assert err < 1e-5, "matmul 路径与旧公式不符"
    print("SMOKE OK")


if __name__ == "__main__":
    main()
