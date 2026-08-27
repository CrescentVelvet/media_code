#!/usr/bin/env python3
"""pose_adjuster.py — One-time pose adjustment before 3DGS training.

Three operations (each independently toggleable):
  1. Translation: center scene at the sight-line intersection of all cameras.
  2. Rotation: gravity-align using SVD on camera right-vectors.
  3. Scale: normalize camera-to-center median distance to 10.

Transform formulas (w2c convention, COLMAP / OpenCV):
  Camera:   new_w2c_t = scale * (w2c_t + w2c_r @ center)
            new_w2c_r = w2c_r @ pred_rot
  Points:   new_pts   = scale * (pts - center) @ pred_rot

Reverse (after training, for original-coord PLY):
  Camera:   old_w2c_r = w2c_r @ pred_rot.T
            old_w2c_t = (1/scale) * w2c_t - old_w2c_r @ center
  Points:   old_pts   = (1/scale) * pts @ pred_rot.T + center
  Gaussian: scaling  /= scale;  rotation @= pred_rot.T

All math in torch (float64 for least-squares stability, then cast to float32).
"""
import os
import json
import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def quat_to_rotmat(q):
    """Quaternion [w, x, y, z] -> 3x3 rotation matrix. Differentiable."""
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)


def rotmat_to_quat(R):
    """3x3 rotation matrix -> [w, x, y, z] (Shepperd's method, scalar-first)."""
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    q = torch.zeros(*R.shape[:-2], 4, device=R.device, dtype=R.dtype)
    # Positive trace branch (covers most cases; sufficient for our use)
    s = torch.sqrt(torch.clamp(trace + 1.0, min=1e-10)) * 2.0
    q[..., 0] = 0.25 * s
    q[..., 1] = (R[..., 2, 1] - R[..., 1, 2]) / s
    q[..., 2] = (R[..., 0, 2] - R[..., 2, 0]) / s
    q[..., 3] = (R[..., 1, 0] - R[..., 0, 1]) / s
    return q / (torch.norm(q, dim=-1, keepdim=True) + 1e-9)


# ---------------------------------------------------------------------------
# Sight-line intersection (least-squares)
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_sight_center(c2w_t_list, sight_dir_list):
    """Estimate scene center as the least-squares intersection of camera sight-lines.

    Each camera i: position c2w_t_i = center + t_i * sight_dir_i.
    Solve A @ [center; t_1..t_n] = c2w_t, where A is (3n, n+3):
      rows 3i:3i+3 = [I_3 | 0 ... -sight_dir_i ... 0]  (col 3+i = -sight_dir_i).
    """
    n = len(c2w_t_list)
    c2w_t = torch.stack(c2w_t_list)                   # (n, 3)
    sight = torch.stack(sight_dir_list)                # (n, 3)

    # Build A: (3n, n+3), b: (3n,)
    # Each camera i: c2w_t_i = center + t_i * sight_dir_i
    # Unknowns: center (3) + t_1..t_n (n) = n+3. For camera i, rows 3i:3i+3:
    #   cols 0:3 = I_3 (center), col 3+i = -sight_dir_i (t_i coefficient).
    A = torch.zeros(3 * n, n + 3, dtype=c2w_t.dtype, device=c2w_t.device)
    for i in range(n):
        A[3 * i:3 * i + 3, :3] = torch.eye(3, dtype=c2w_t.dtype, device=c2w_t.device)
        A[3 * i:3 * i + 3, 3 + i] = -sight[i]
    b = c2w_t.reshape(-1)                              # (3n,)

    # Normal equations: (A^T A) x = A^T b
    AtA = A.T @ A
    Atb = A.T @ b
    sol = torch.linalg.solve(AtA + 1e-6 * torch.eye(n + 3, dtype=AtA.dtype, device=AtA.device), Atb)
    center = sol[:3]                                    # (3,)

    # Determine inside-out vs outside-in: check if center is in front of cameras
    # If c2w_t + sight_dir is farther from center than c2w_t - sight_dir,
    # cameras look inward (outside-in) → center is correct.
    # Otherwise, flip sight_dir (inside-out).
    dist_plus = torch.norm(c2w_t + sight - center, dim=-1).mean()
    dist_minus = torch.norm(c2w_t - sight - center, dim=-1).mean()
    if dist_plus < dist_minus:
        # Inside-out: cameras look outward; center is behind them
        # Flip sight dirs and re-solve
        sight = -sight
        A = torch.zeros(3 * n, n + 3, dtype=c2w_t.dtype, device=c2w_t.device)
        for i in range(n):
            A[3 * i:3 * i + 3, :3] = torch.eye(3, dtype=c2w_t.dtype, device=c2w_t.device)
            A[3 * i:3 * i + 3, 3 + i] = -sight[i]
        AtA = A.T @ A
        Atb = A.T @ b
        sol = torch.linalg.solve(AtA + 1e-6 * torch.eye(n + 3, dtype=AtA.dtype, device=AtA.device), Atb)
        center = sol[:3]

    return center.float()


# ---------------------------------------------------------------------------
# Gravity estimation (SVD on camera right-vectors)
# ---------------------------------------------------------------------------
@torch.no_grad()
def estimate_down_vec(w2c_r_list):
    """Estimate gravity (down) direction via SVD on camera right-vectors.

    The right-vector (w2c_r[:, 0]) is perpendicular to gravity for orbit shots.
    The direction most perpendicular to all right-vectors = gravity = min singular vector.
    """
    right_vecs = torch.stack([r[:, 0] for r in w2c_r_list])     # (n, 3)
    cov = right_vecs.T @ right_vecs                               # (3, 3)
    _, _, V = torch.linalg.svd(cov)
    down = V.T[2]                                                  # smallest singular vector

    # Align with the mean down-vector (w2c_r[:, 1])
    mean_down = torch.stack([r[:, 1] for r in w2c_r_list]).mean(dim=0)
    if torch.dot(down, mean_down) < 0:
        down = -down
    return down


# ---------------------------------------------------------------------------
# PoseAdjuster
# ---------------------------------------------------------------------------
class PoseAdjuster:
    """One-time pose adjustment: translate + rotate + scale.

    Call apply_to_cameras() + apply_to_points() before training.
    Call reverse_gaussians() after training for original-coord PLY.
    """

    def __init__(self, w2c_r_list, w2c_t_list, c2w_t_list, sight_dir_list,
                 enable_trans=True, enable_rotate=True, enable_scale=True,
                 gravity_prior=False, device="cuda"):
        self.enable_trans = enable_trans
        self.enable_rotate = enable_rotate
        self.enable_scale = enable_scale
        self.device = device
        dtype = torch.float64

        w2c_r = [r.to(dtype).to(device) for r in w2c_r_list]
        w2c_t = [t.to(dtype).to(device) for t in w2c_t_list]
        c2w_t = [t.to(dtype).to(device) for t in c2w_t_list]
        sight = [s.to(dtype).to(device) for s in sight_dir_list]

        # 1. Translation: sight-line intersection
        if enable_trans:
            center = compute_sight_center(c2w_t, sight)
        else:
            center = torch.zeros(3, dtype=dtype, device=device)

        # 2. Rotation: gravity alignment
        if enable_rotate:
            if gravity_prior:
                down = torch.tensor([0.0, -1.0, 0.0], dtype=dtype, device=device)
            else:
                down = estimate_down_vec(w2c_r)
            first_right = w2c_r[0][:, 0]
            lookat = torch.nn.functional.normalize(torch.cross(first_right, down, dim=0), dim=0)
            right = torch.nn.functional.normalize(torch.cross(down, lookat, dim=0), dim=0)
            pred_rot = torch.stack([right, down, lookat], dim=1)   # (3, 3), columns = right/down/lookat
            pred_rot = pred_rot @ torch.diag(torch.tensor([1.0, -1.0, -1.0], dtype=dtype, device=device))
        else:
            pred_rot = torch.eye(3, dtype=dtype, device=device)

        # 3. Scale
        if enable_scale:
            dists = torch.stack([torch.norm(t - center) for t in c2w_t])
            extent = torch.median(dists).clamp(min=1e-6)
            scale = 10.0 / extent
        else:
            scale = torch.tensor(1.0, dtype=dtype, device=device)

        self.center = center.float().cpu()
        self.pred_rot = pred_rot.float().cpu()
        self.scale = scale.float().cpu()

        print(f"[PoseAdjuster] trans={enable_trans} rotate={enable_rotate} scale={enable_scale}")
        print(f"  center={self.center.tolist()}")
        print(f"  scale={self.scale.item():.4f}")
        if enable_rotate:
            print(f"  gravity={down.tolist()}")

    # ── Forward transforms ──────────────────────────────────────────────────
    def transform_camera(self, w2c_r, w2c_t):
        """Forward: original → adjusted coords. Returns (new_r, new_t)."""
        center = self.center.to(w2c_r)
        rot = self.pred_rot.to(w2c_r)
        scale = self.scale.to(w2c_r)
        new_t = scale * (w2c_t + w2c_r @ center)
        new_r = w2c_r @ rot
        return new_r, new_t

    def transform_points(self, points):
        """Forward: original → adjusted coords. points: (N, 3)."""
        center = self.center.to(points)
        rot = self.pred_rot.to(points)
        scale = self.scale.to(points)
        return scale * (points - center) @ rot

    # ── Reverse transforms (after training) ────────────────────────────────
    def reverse_points(self, points):
        """Reverse: adjusted → original coords. points: (N, 3)."""
        center = self.center.to(points)
        rot = self.pred_rot.to(points)
        scale = self.scale.to(points)
        return (1.0 / scale) * points @ rot.T + center

    def reverse_scaling(self, scaling):
        """Reverse gaussian scaling: divide by scale."""
        return scaling / self.scale.to(scaling.device)

    def reverse_rotation(self, rotation):
        """Reverse gaussian rotation: multiply by pred_rot.T."""
        rot = self.pred_rot.to(rotation.device)
        return rotation @ rot.T

    # ── Save / Load ────────────────────────────────────────────────────────
    def save(self, path):
        """Save params as JSON (alongside the 3DGS model)."""
        data = {
            "center": self.center.tolist(),
            "pred_rot": self.pred_rot.tolist(),
            "scale": self.scale.item(),
            "enable_trans": self.enable_trans,
            "enable_rotate": self.enable_rotate,
            "enable_scale": self.enable_scale,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[PoseAdjuster] saved -> {path}")

    @classmethod
    def load(cls, path, device="cuda"):
        """Load from JSON (for render_novel.py to transform virtual cameras)."""
        with open(path) as f:
            data = json.load(f)
        adj = cls.__new__(cls)
        adj.center = torch.tensor(data["center"], dtype=torch.float32)
        adj.pred_rot = torch.tensor(data["pred_rot"], dtype=torch.float32)
        adj.scale = torch.tensor(data["scale"], dtype=torch.float32)
        adj.enable_trans = data.get("enable_trans", True)
        adj.enable_rotate = data.get("enable_rotate", True)
        adj.enable_scale = data.get("enable_scale", True)
        adj.device = device
        return adj
