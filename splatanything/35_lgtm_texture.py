#!/usr/bin/env python3
"""35: SplatAnything + LGTM-style textured 2DGS controlled pilot.

Freeze SA pre-merge geometry. Both arms share surfels, source supervision and
images; the LGTM arm additionally receives projected texture features. Keep the
same frozen SA RGB base and learnable residual in both arms; the LGTM arm adds
its projected RGB residual to that base after activation. Projection identity
is checked independently before optimization. This is single-scene fitting,
not cross-scene training or 4K.

Requires LGTM's patched gsplat and texture_project.py.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
SH_C0 = 0.28209479177387814


def load_local(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M09 = load_local("sa_train09", "09_train_probe.py")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", required=True, help="Scene directory containing images/")
    p.add_argument("--out", required=True)
    p.add_argument("--res", type=int, default=252)
    p.add_argument("--n-source", type=int, default=3)
    p.add_argument("--camera-context", type=int, default=4,
                   help="Images in one teacher camera estimate; last image is held out")
    p.add_argument("--pose-mode", action="store_true",
                   help="Use scene/prior_camera.json, instead of joint estimated cameras")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--texture-size", type=int, default=4, choices=(1, 2, 4, 8, 16, 32))
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def subset(views, ids, n):
    return {k: v[:, ids] if torch.is_tensor(v) and v.ndim >= 2 and v.shape[1] == n else v
            for k, v in views.items()}


def camera_split(n, n_source, n_context, pose_mode=False):
    if n < n_source + 1 or (not pose_mode and n_context < n_source + 1):
        raise ValueError("Need at least n_source+1 context images and one distinct target")
    target = n - 1
    available = list(range(n - 1))
    positions = np.linspace(0, len(available) - 1, n_source).round().astype(int)
    source = [available[i] for i in positions]
    if pose_mode:
        return source, [target], source
    context = np.linspace(0, n - 1, n_context).round().astype(int).tolist()
    ids = sorted(set(source + [target] + context))
    if len(ids) < n_source + 1:
        raise ValueError("Camera context collapsed; increase --camera-context")
    return source, [target], ids


def rgb_metrics(rgb, gt, alpha):
    mse = float((rgb.float() - gt.float()).square().mean())
    return {"mse": mse, "mae": float((rgb - gt).abs().mean()),
            "psnr": -10 * np.log10(max(mse, 1e-12)),
            "alpha_mean": float(alpha.mean()),
            "coverage_05": float((alpha > .5).float().mean())}


def sh0_to_rgb(sh0):
    return sh0 * SH_C0 + 0.5


def rgb_to_sh0(rgb):
    return (rgb - 0.5) / SH_C0


def render_2d(sp, c2w, K, hw, textures=None, color_delta=None):
    from gsplat.rendering import rasterization_2dgs
    from gsplat.cuda._wrapper import TextureAlphaMode
    # Use identical activated SA RGB base in both paths. Patched gsplat adds
    # texture RGB after base activation, so no second SH0 conversion is needed.
    colors = sh0_to_rgb(sp["sh"][0][:, 0]).clamp_min(0)
    if color_delta is not None:
        colors = colors + color_delta
    extra = {}
    if textures is not None:
        extra = dict(texture_colors=textures, texture_alpha_mode=TextureAlphaMode.GAUSSIAN)
    rgb, alpha, _, _, _, _, _ = rasterization_2dgs(
        means=sp["means"][0], quats=F.normalize(sp["quats"][0], dim=-1),
        scales=sp["scales"][0], opacities=sp["opacities"][0],
        colors=colors, viewmats=torch.linalg.inv(c2w), Ks=K,
        width=hw[1], height=hw[0], near_plane=.01, far_plane=400., sh_degree=None,
        backgrounds=torch.zeros(len(c2w), 3, device=c2w.device), render_mode="RGB", **extra)
    return rgb.clamp(0, 1), alpha[..., 0], rgb


def render_3d(sp, c2w, K, hw):
    from gsplat import rasterization
    rgb, alpha, _ = rasterization(
        means=sp["means"][0], quats=sp["quats"][0], scales=sp["scales"][0],
        opacities=sp["opacities"][0], colors=sp["sh"][0],
        viewmats=torch.linalg.inv(c2w), Ks=K, width=hw[1], height=hw[0],
        sh_degree=0, eps2d=.01, near_plane=.01, far_plane=400.,
        backgrounds=torch.zeros(len(c2w), 3, device=c2w.device), render_mode="RGB")
    return rgb.clamp(0, 1), alpha[..., 0]


class AppearanceHead(nn.Module):
    """Shared image/base-color trunk and projected-texture adapter, RGB output.

    Common layers have identical initialization in both arms. Extra texture
    capacity is explicitly reported; this is not an equal-parameter comparison.
    """
    def __init__(self, size, channels):
        super().__init__()
        self.size = size
        self.texture_adapter = nn.Sequential(
            nn.Conv2d(3 * size * size, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU())
        self.trunk = nn.Sequential(
            nn.Conv2d(6 + channels, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU())
        self.out = nn.Conv2d(channels, 3 * size * size, 1)
        nn.init.zeros_(self.texture_adapter[-2].weight)
        nn.init.zeros_(self.texture_adapter[-2].bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, images, base_rgb, projected):
        B, S, _, H, W = images.shape
        feat = self.texture_adapter((projected * 2 - 1).reshape(B * S, -1, H, W))
        # LGTM uses [-1,1] image inputs. Its texture RGB is added to the base.
        x = torch.cat([images * 2 - 1, base_rgb.clamp(0, 1) * 2 - 1,
                       feat.reshape(B, S, -1, H, W)], 2).reshape(B * S, -1, H, W)
        return self.out(self.trunk(x)).reshape(B, S, 3 * self.size**2, H, W)


def to_matrix(q):
    w, x, y, z = F.normalize(q, dim=-1).unbind(-1)
    return torch.stack([
        1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], -1).reshape(*q.shape[:-1], 3, 3)


def to_quat(R):
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(R.detach().cpu().numpy().reshape(-1, 3, 3)).as_quat()
    q = torch.from_numpy(q[..., [3, 0, 1, 2]].copy()).to(R.device, R.dtype)
    return q.reshape(*R.shape[:-2], 4)


def surfel_rotation(quats, scales):
    """Flatten SA's smallest-scale axis without shrinking its two tangents."""
    R = to_matrix(quats)
    axis = scales.argmin(dim=-1)
    # A cyclic permutation preserves the original tangent frame and handedness.
    # Replacing both tangent scales by the normal scale would erase thin
    # primitives and create holes/moire unrelated to texture optimization.
    order = torch.stack(((axis + 1) % 3, (axis + 2) % 3, axis), dim=-1)
    R_surfel = R.gather(-1, order[..., None, :].expand(*order.shape[:-1], 3, 3))
    tangent = scales.gather(-1, order).clone()
    tangent[..., 2] = 0
    return R_surfel, tangent


def source_local_geometry(raw, keep, preds, views):
    """Preserve source UV identity; never voxel-merge splats across sources."""
    B, S, _, H, W = views["img"].shape
    # Keep [B,S*H*W,C] separate from the source-grid view used for UV projection.
    sp = {}
    for k in ("means", "quats", "scales", "opacities", "sh"):
        v = raw[k]
        sp[k] = torch.stack([v[b][keep[b]] for b in range(B)]).reshape(
            B, S, H, W, *v.shape[2:]).detach()
    rotation, tangent = surfel_rotation(sp["quats"], sp["scales"])
    c2w = preds["export_camera_poses"].float()
    R, t = c2w[..., :3, :3], c2w[..., :3, 3]
    local_means = torch.einsum("bsji,bshwj->bshwi", R, sp["means"] - t[:, :, None, None, :])
    local_rotation = torch.einsum("bsji,bshwjk->bshwik", R, rotation)
    local = {"means": local_means, "quats": to_quat(local_rotation), "scales": tangent}
    expected_normal = torch.einsum("bsji,bshwj->bshwi", R, rotation[..., :, 2])
    world = {"means": sp["means"], "quats": to_quat(rotation), "scales": tangent,
             "opacities": sp["opacities"], "sh": sp["sh"]}
    flat = {k: world[k].reshape(1, -1, *world[k].shape[4:]) for k in world}
    stats = {"scale_min": float(tangent.min()), "scale_median": float(tangent.median()),
             "scale_max": float(tangent.max()),
             "local_normal_error": float((to_matrix(local["quats"]) - local_rotation).abs().max()),
             "local_rotation_normal_error": float((local_rotation[..., :, 2] - expected_normal).abs().max()),
             "world_rotation_normal_error": float((rotation[..., :, 2] -
                to_matrix(sp["quats"]).gather(-1, sp["scales"].argmin(-1)[..., None, None].expand(
                    *sp["scales"].shape[:-1], 3, 1)).squeeze(-1)).abs().max()),
             "local_z_min": float(local_means[..., 2].min()),
             "local_z_median": float(local_means[..., 2].median())}
    return flat, local, stats


def project_source(lgtm_dir, local, images, K, size, return_debug=False):
    sys.path.insert(0, str(Path(lgtm_dir) / "src"))
    from gsplat.cuda._wrapper import fully_fused_projection_2dgs
    B, S, _, H, W = images.shape
    if K.ndim == 3:
        K = K[None]  # Renderer uses [S,3,3]; LGTM projection uses [B,3,3].
    if K.shape[:2] != (B, S):
        raise ValueError(f"Expected camera K [S,3,3] or [B,S,3,3], got {K.shape}")
    values, coordinates = [], []
    for b in range(B):
        for s in range(S):
            H, W = local["means"].shape[2:4]
            means = local["means"][b, s:s+1].reshape(1, -1, 3)
            scales = local["scales"][b, s:s+1].reshape(1, -1, 3)
            xyzws = local["quats"][b, s:s+1].reshape(1, -1, 4)[..., [1, 2, 3, 0]]
            image = images[b, s:s+1]
            intrinsics = K[b, s:s+1].float()
            wxyz = xyzws[..., [3, 0, 1, 2]]
            radii, _, _, ray_transforms, _ = fully_fused_projection_2dgs(
                means=means[0], quats=wxyz[0], scales=scales[0],
                viewmats=torch.eye(4, device=K.device, dtype=K.dtype)[None],
                Ks=intrinsics, width=W, height=H)
            valid = (radii > 0).all(dim=-1)[0]
            # Fused CUDA stores mathematical KWH in row-major order. Do not
            # transpose it: h_pixel = KWH @ [u,v,1], normalized by its z.
            texel = torch.arange(size, device=K.device, dtype=K.dtype) + .5
            v, u = torch.meshgrid(texel, texel, indexing="ij")
            uv = torch.stack((u, v), dim=-1).reshape(-1, 2) / size * 2 - 1
            coords_h = torch.einsum(
                "nij,tj->nti", ray_transforms[0], F.pad(uv, (0, 1), value=1))
            coords = (coords_h[..., :2] / coords_h[..., 2:3]).reshape(
                H, W, size, size, 2)
            if return_debug:
                uvh = F.pad(uv, (0, 1), value=1)
                center_h = ray_transforms[0][:, :, 2]
                center_h = center_h / center_h[:, 2:3]
                coordinates.append({
                    "ray_transforms": ray_transforms[0],
                    "texel_coords": coords,
                    "center_coords": center_h[:, :2].reshape(H, W, 2),
                    "uvh": uvh,
                    "valid": valid,
                })
                continue
            texture = image.new_zeros(H * W, 3, size, size)
            if valid.any():
                # SA lifting places its homogeneous center at integer (u,v).
                # grid_sample's center convention needs +0.5 here exactly once.
                # Official LGTM omits it because its lifting convention differs;
                # removing this shift causes a measured 0.5-pixel error on SA.
                grid = torch.stack((2 * (coords[..., 0] + .5) / W - 1,
                                    2 * (coords[..., 1] + .5) / H - 1), dim=-1)
                grid = grid.reshape(-1, size * size, 2)[valid].reshape(1, -1, 1, 2)
                sampled = F.grid_sample(
                    image, grid, mode="bilinear",
                    padding_mode="border", align_corners=False)
                texture[valid] = sampled.reshape(1, 3, -1, size, size)[0].permute(1, 0, 2, 3)
            values.append(texture.reshape(H, W, 3, size, size))
    if return_debug:
        return {"ray_transforms": torch.stack([x["ray_transforms"] for x in coordinates]).reshape(
                    B, S, H, W, 3, 3),
                "texel_coords": torch.stack([x["texel_coords"] for x in coordinates]).reshape(
                    B, S, H, W, size, size, 2),
                "center_coords": torch.stack([x["center_coords"] for x in coordinates]).reshape(
                    B, S, H, W, 2),
                "uvh": coordinates[0]["uvh"],
                "valid": torch.stack([x["valid"] for x in coordinates]).reshape(B, S, H, W)}
    return torch.stack(values).reshape(B, S, H, W, 3, size, size).detach()


def expected_texel_coordinates(local, K, size):
    """Analytically project the same source-local tangent-plane texels."""
    if K.ndim == 3:
        K = K[None]
    R = to_matrix(local["quats"])
    tangent = local["scales"][..., :2]
    t = (torch.arange(size, device=K.device, dtype=K.dtype) + .5) / size * 2 - 1
    v, u = torch.meshgrid(t, t, indexing="ij")
    uv = torch.stack((u, v), dim=-1)
    delta = torch.einsum("...ij,abj->...abi", R[..., :, :2] * tangent[..., None, :], uv)
    points = local["means"][..., None, None, :] + delta
    pixels = torch.einsum("bsij,bs...j->bs...i", K, points)
    return pixels[..., :2] / pixels[..., 2:3]



def projection_diagnostics(lgtm_dir, local, images, K, size):
    """Check centers and every texel against coordinates, not natural RGB alone."""
    from gsplat.cuda._wrapper import fully_fused_projection_2dgs
    B, S, _, H, W = images.shape
    actual_debug = project_source(lgtm_dir, local, images, K, size, return_debug=True)
    actual = actual_debug["texel_coords"]
    expected = expected_texel_coordinates(local, K, size)
    error = (actual - expected).abs()
    border = ((expected[..., 0] >= 0) & (expected[..., 0] <= W - 1) &
              (expected[..., 1] >= 0) & (expected[..., 1] <= H - 1))
    interior = ((expected[..., 0] >= 1) & (expected[..., 0] <= W - 2) &
                (expected[..., 1] >= 1) & (expected[..., 1] <= H - 2))

    # Pixel-valued ramps expose x/y/source swaps; border clamping is separate.
    ramps = images.clone()
    ramps[:, :, 0] = torch.arange(W, device=K.device)[None, None, None, :]
    ramps[:, :, 1] = torch.arange(H, device=K.device)[None, None, :, None]
    ramps[:, :, 2] = torch.arange(S, device=K.device)[None, :, None, None]
    sampled = project_source(lgtm_dir, local, ramps, K, size)
    # CUDA's forward projection uses boundaries (pixel j at j+0.5), while
    # grid_sample expects centers. Apply +0.5 here, not in the analytic
    # reference or anywhere else. SA source centers then return integer (u,v).
    synthetic = torch.stack(
        (expected[..., 0], expected[..., 1], expected[..., 0].clone()), dim=4)
    synthetic[:, :, :, :, 2] = torch.arange(S, device=K.device)[None, :, None, None, None, None]
    synthetic_error = (sampled - synthetic).abs()

    projected_center = project_source(lgtm_dir, local, images, K, 1)
    expected_center = images.permute(0, 1, 3, 4, 2)[..., None, None]
    center = actual_debug["center_coords"]
    center_ramps = project_source(lgtm_dir, local, ramps, K, 1)
    yy, xx = torch.meshgrid(torch.arange(H, device=K.device),
                            torch.arange(W, device=K.device), indexing="ij")
    center_xy = torch.stack((xx, yy), dim=-1)[None, None].float()
    center_synthetic_error = (center_ramps[:, :, :, :, :2, 0, 0] - center_xy).abs()
    yy, xx = torch.meshgrid(torch.arange(H, device=K.device),
                            torch.arange(W, device=K.device), indexing="ij")
    center_error = center - torch.stack((xx, yy), dim=-1)[None, None]
    R_local = to_matrix(local["quats"])
    M_local = torch.matmul(K[None, :, None, None],
                           R_local * local["scales"][..., None, :]).clone()
    M_local[..., 2] = torch.matmul(
        K[None, :, None, None], local["means"][..., None]
    )[..., 0]
    matrix_error = actual_debug["ray_transforms"] - M_local
    stats_matrix = {
        "analytic_matrix_max_abs": float(matrix_error.abs().max()),
        "analytic_matrix_center_max_abs": float(matrix_error[..., :, 2].abs().max()),
        "analytic_matrix_tangent_max_abs": float(matrix_error[..., :, :2].abs().max()),
    }
    lifting_pixels = torch.matmul(K[None, :, None, None], local["means"][..., None])[..., 0]
    lifting_pixels = lifting_pixels[..., :2] / lifting_pixels[..., 2:3]
    lifting_error = lifting_pixels - torch.stack((xx, yy), dim=-1)[None, None].float()
    stats_matrix.update({
        "local_lifting_center_mae": float(lifting_error.abs().mean()),
        "local_lifting_center_max_error": float(lifting_error.abs().max()),
    })
    analytic_center = actual_debug["ray_transforms"][..., :, 2]
    analytic_center = analytic_center[..., :2] / analytic_center[..., 2:3]
    stats_matrix["matrix_center_vs_local_max_px"] = float((analytic_center - lifting_pixels).abs().max())
    packed_stats = []
    for b in range(B):
        for s in range(S):
            means = local["means"][b, s].reshape(-1, 3)
            quats = local["quats"][b, s].reshape(-1, 4)
            scales = local["scales"][b, s].reshape(-1, 3)
            fused = actual_debug["ray_transforms"][b, s].reshape(-1, 3, 3)
            valid = actual_debug["valid"][b, s].reshape(-1)
            result = fully_fused_projection_2dgs(
                means=means, quats=quats, scales=scales,
                viewmats=torch.eye(4, device=K.device, dtype=K.dtype)[None],
                Ks=K[b, s:s+1] if K.ndim == 4 else K[s:s+1],
                width=W, height=H, packed=True)
            ids, primitive_ids, radii, means2d, depths, matrices, _ = result
            if not torch.equal(primitive_ids, valid.nonzero()[:, 0]):
                packed_stats.append({"valid_order_match": False})
                continue
            packed_stats.append({
                "valid_order_match": True,
                "matrix_max_error": float((matrices - fused[valid]).abs().max()),
            })
    stats_matrix["packed_fused_max_matrix_error"] = max(
        x.get("matrix_max_error", float("inf")) for x in packed_stats)
    stats_matrix["packed_fused_valid_order_match"] = all(
        x["valid_order_match"] for x in packed_stats)
    center_error_abs = center_error.abs()
    stats_matrix["center_quantiles_50_90_99_999_100"] = [float(x) for x in torch.quantile(
        center_error_abs.amax(dim=-1).reshape(-1),
        torch.tensor([.5, .9, .99, .999, 1.], device=center.device))]
    rgb_error = (projected_center - expected_center).abs()
    stats_matrix["rgb_quantiles_50_90_99_999_100"] = [float(x) for x in torch.quantile(
        rgb_error.amax(dim=2).reshape(-1),
        torch.tensor([.5, .9, .99, .999, 1.], device=rgb_error.device))]
    worst_rgb = torch.topk(rgb_error.amax(dim=2).reshape(-1), 8).indices
    stats_matrix["rgb_worst_source_height_width"] = [
        [int(i // (H * W)), int((i % (H * W)) // W), int(i % W)] for i in worst_rgb]
    rgb_l1 = (images[:, :, :, :, 1:] - images[:, :, :, :, :-1]).abs().amax(dim=2)
    rgb_l1 = F.pad(rgb_l1, (0, 1))
    vertical = (images[:, :, :, 1:] - images[:, :, :, :-1]).abs().amax(dim=2)
    vertical = F.pad(vertical, (0, 0, 0, 1))
    assert rgb_l1.shape == (B, S, H, W), rgb_l1.shape
    assert vertical.shape == (B, S, H, W), vertical.shape
    rgb_l1 = torch.maximum(rgb_l1, vertical)
    local_rgb_bound = 2 * rgb_l1[..., None] * (center - center_xy).abs().amax(
        dim=-1, keepdim=True)
    stats_matrix["rgb_local_linear_bound_max"] = float(local_rgb_bound.max())
    stats_matrix["rgb_local_linear_bound_at_max_error"] = float(
        local_rgb_bound.reshape(-1)[rgb_error.amax(dim=2).reshape(-1).argmax()])
    analytic_grid = torch.stack(
        (2 * (center[..., 0] + .5) / W - 1,
         2 * (center[..., 1] + .5) / H - 1), dim=-1)
    assert analytic_grid.ndim == 5 and analytic_grid.shape[:4] == (B, S, H, W), analytic_grid.shape
    analytic_rgb = F.grid_sample(
        images.reshape(B * S, 3, H, W),
        analytic_grid.reshape(B * S, H, W, 2), mode="bilinear",
        padding_mode="border", align_corners=False).reshape(B, S, 3, H, W)
    analytic_rgb = analytic_rgb.permute(0, 1, 3, 4, 2)
    stats_matrix["rgb_cuda_vs_analytic_center_max"] = float((
        projected_center[..., 0, 0] - analytic_rgb).abs().max())
    stats_matrix["rgb_direct_vs_analytic_center_max"] = float((
        expected_center[..., 0, 0] - analytic_rgb).abs().max())
    inner_synthetic_abs = synthetic_error.abs()
    interior_sample_error = inner_synthetic_abs[interior.unsqueeze(4).expand_as(synthetic_error)]
    stats_matrix["interior_synthetic_all_xy_max_error"] = float(
        interior_sample_error.max()) if interior_sample_error.numel() else 0.
    stats_matrix["interior_synthetic_xy_max_error"] = float(
        interior_sample_error.reshape(-1, 3)[:, :2].max()) if interior_sample_error.numel() else 0.
    inner_error = error[interior]
    inner_synthetic_error = synthetic_error.permute(
        0, 1, 2, 3, 5, 6, 4)[..., :2][interior]
    inner_synthetic_mae = (float(inner_synthetic_error.mean())
                           if inner_synthetic_error.numel() else 0.)
    stats_matrix["interior_source_id_max_error"] = float(
        synthetic_error.permute(0, 1, 2, 3, 5, 6, 4)[..., 2][interior].max()
    ) if interior.any() else 0.
    from lgtm.utils.texture_project import project_texture_colors
    official_differences = []
    for b in range(B):
        for s in range(S):
            official = project_texture_colors(
                means=local["means"][b, s:s+1].reshape(1, -1, 3),
                scales=local["scales"][b, s:s+1].reshape(1, -1, 3),
                xyzws=local["quats"][b, s:s+1].reshape(1, -1, 4)[..., [1, 2, 3, 0]],
                ims=images[b, s:s+1], Ks=K[s:s+1] if K.ndim == 3 else K[b, s:s+1],
                Ts=torch.eye(4, device=K.device, dtype=K.dtype)[None],
                texture_size=size, use_cuda_projection=True)[0]
            local_one = {k: v[b:b+1, s:s+1] for k, v in local.items()}
            current = project_source(
                lgtm_dir, local_one, images[b:b+1, s:s+1],
                K[s:s+1] if K.ndim == 3 else K[b:b+1, s:s+1], size)[0, 0]
            official_differences.append(float((official.reshape(-1, 3, size, size) -
                                               current.reshape(-1, 3, size, size)).abs().max()))
    stats_matrix["official_cuda_projection_max_rgb_error"] = max(official_differences)
    # Official LGTM's grid_sample mapping differs from SA's integer-center
    # lifting. Reuse the same fused matrix and replace only that source-grid
    # mapping; this isolates the half-pixel convention from any layout issue.
    convention_diffs = []
    direct_projected = project_source(lgtm_dir, local, images, K, size)
    for b in range(B):
        for s in range(S):
            actual_one = actual_debug["texel_coords"][b:b+1, s:s+1]
            valid_one = actual_debug["valid"][b:b+1, s:s+1].reshape(-1)
            grid = torch.stack((
                2 * actual_one[..., 0] / W - 1,
                2 * actual_one[..., 1] / H - 1), dim=-1)
            grid = grid.reshape(-1, size * size, 2)[valid_one].reshape(1, -1, 1, 2)
            sampled = F.grid_sample(
                images[b, s:s+1], grid, mode="bilinear",
                padding_mode="border", align_corners=False)
            expected_one = images.new_zeros(H * W, 3, size, size)
            expected_one[valid_one] = sampled.reshape(
                3, -1, size, size).permute(1, 0, 2, 3)
            expected_one = expected_one.reshape(1, 1, H, W, 3, size, size)
            convention_diffs.append(float((
                direct_projected[b:b+1, s:s+1] - expected_one).abs().max()))
    stats_matrix["official_grid_convention_max_rgb_error"] = max(convention_diffs)
    # Independent analytic reference: all texels, source identity, CUDA/packed
    # matrices and the actual rasterizer share the same SA lifting convention.
    expected_official = expected.reshape(-1, size * size, 2)
    official_grid = torch.stack(
        (2 * (expected_official[..., 0] + .5) / W - 1,
         2 * (expected_official[..., 1] + .5) / H - 1), dim=-1)
    official_rgb = F.grid_sample(
        images.reshape(B * S, 3, H, W),
        official_grid.reshape(B * S, H * W * size * size, 1, 2), mode="bilinear",
        padding_mode="border", align_corners=False)
    expected_projected = official_rgb.reshape(
        B, S, 3, H, W, size, size).permute(0, 1, 3, 4, 2, 5, 6)
    stats_matrix["independent_texel_rgb_max_error"] = float(
        (direct_projected - expected_projected).abs().max())
    interior_rgb = (direct_projected - expected_projected).abs()[
        interior.unsqueeze(4).expand_as(direct_projected)]
    stats_matrix["independent_interior_texel_rgb_max_error"] = float(interior_rgb.max())
    stats = {
        "texture_size": size,
        "rgb_mae": float((projected_center - expected_center).abs().mean()),
        "rgb_max_error": float((projected_center - expected_center).abs().max()),
        "coordinate_mae_px": float(error.mean()),
        "coordinate_max_error_px": float(error.max()),
        "interior_coordinate_mae_px": float(inner_error.mean()) if inner_error.numel() else 0.,
        "interior_coordinate_max_error_px": float(inner_error.max()) if inner_error.numel() else 0.,
        "synthetic_coordinate_mae_px": float(synthetic_error.mean()),
        "synthetic_coordinate_max_error_px": float(synthetic_error.max()),
        "interior_synthetic_coordinate_mae_px": inner_synthetic_mae,
        "interior_synthetic_coordinate_max_error_px": float(inner_synthetic_error.max()) if inner_synthetic_error.numel() else 0.,
        "center_pixel_mae": float(center_error.abs().mean()),
        "center_pixel_max_error": float(center_error.abs().max()),
        "interior_fraction": float(interior.float().mean()),
        "border_clamp_fraction": float((~border).float().mean()),
        "zero_fraction": float((projected_center == 0).all(dim=2).float().mean()),
        "mean_error_per_source": [float((projected_center[:, i] - expected_center[:, i]).abs().mean()) for i in range(S)],
        "k1_synthetic_mae_px": float(center_synthetic_error.mean()),
        "k1_synthetic_max_error_px": float(center_synthetic_error.max()),
        "thresholds": {"rgb_mae": 2e-5, "rgb_max_error": 2e-4,
                       "coordinate_max_error_px": 2e-3,
                       "interior_synthetic_coordinate_mae_px": 2e-3},
    }
    M = actual_debug["ray_transforms"].reshape(-1, 3, 3).repeat_interleave(size * size, dim=0)
    p_h = F.pad(expected.reshape(-1, 2), (0, 1), value=1)
    reference_uv = actual_debug["uvh"].reshape(size * size, 3)[None, :, :2].expand(
        expected.numel() // (2 * size * size), -1, -1)
    uv_cross = torch.empty_like(reference_uv)
    for start in range(0, reference_uv.numel() // 2, 16384):
        end = min(start + 16384, reference_uv.numel() // 2)
        ph = expected.reshape(-1, 2)[start:end]
        mat = actual_debug["ray_transforms"].reshape(-1, 3, 3).repeat_interleave(size * size, dim=0)[start:end]
        hu = ph[:, 0:1] * mat[:, 2] - mat[:, 0]
        hv = ph[:, 1:2] * mat[:, 2] - mat[:, 1]
        cross = torch.cross(hu, hv, dim=-1)
        uv_cross.reshape(-1, 2)[start:end] = cross[:, :2] / cross[:, 2:3]
    uv_cross = uv_cross.reshape(-1, 2)
    uv_ref = reference_uv.reshape(-1, 2)
    assert uv_ref.shape == uv_cross.shape == (expected.numel() // 2, 2)
    cross_error = (uv_cross - uv_ref).abs().amax(dim=-1)
    cross_quantile = torch.quantile(cross_error, torch.tensor(
        [.5, .9, .99, .999, .9999], device=cross_error.device))
    worst = torch.topk(cross_error, min(1024, len(cross_error))).indices
    # Cross products amplify cancellation on nearly edge-on surfels. Compare
    # the worst UVs in float64 before treating an outlier as a layout bug.
    worst_matrix = actual_debug["ray_transforms"].reshape(-1, 3, 3).double()
    worst_matrix = worst_matrix[(worst // (size * size)).cpu()]
    worst_pixel = expected.reshape(-1, 2).double()[worst.cpu()]
    hu = worst_pixel[:, 0:1] * worst_matrix[:, 2] - worst_matrix[:, 0]
    hv = worst_pixel[:, 1:2] * worst_matrix[:, 2] - worst_matrix[:, 1]
    cross64 = torch.cross(hu, hv, dim=-1)
    uv64 = cross64[:, :2] / cross64[:, 2:3]
    uv_ref64 = uv_ref[worst].double()
    stats["cross_uv_quantiles_50_90_99_999_9999"] = [float(x) for x in cross_quantile]
    stats["cross_uv_max_error"] = float(cross_error.max())
    stats["cross_uv_over_1e-3"] = int((cross_error > 1e-3).sum())
    stats["cross_uv_over_1e-2"] = int((cross_error > 1e-2).sum())
    stats["cross_uv_worst1024_max_float64"] = float((uv64 - uv_ref64).abs().max())
    stats["cross_uv_worst1024_mean_float64"] = float((uv64 - uv_ref64).abs().mean())
    stats["cross_uv_worst1024_max_denominator_float64"] = float(cross64[:, 2].abs().min())
    stats["cross_uv_worst_indices"] = worst[:10].cpu().tolist()
    # UV can be ill-conditioned even when the pixel-space map is correct.
    # Gate both regular UVs and the screen-space roundtrip for the worst UVs.
    condition = torch.linalg.cond(actual_debug["ray_transforms"].reshape(-1, 3, 3).double())
    regular = (condition < 1e3).to(cross_error.device).repeat_interleave(size * size)
    stats["cross_uv_regular_fraction"] = float(regular.float().mean())
    stats["cross_uv_regular_max_error"] = float(cross_error[regular].max()) if regular.any() else 0.
    stats.update(stats_matrix)
    regular_error = torch.where(regular, cross_error, torch.zeros_like(cross_error))
    worst_regular = torch.topk(regular_error, min(1024, len(regular_error))).indices
    condition_quantile = torch.quantile(condition, torch.tensor(
        [0., .01, .1, .5, .9, .99, 1.], device=condition.device, dtype=condition.dtype))
    stats["condition_quantiles_0_1_10_50_90_99_100"] = [float(x) for x in condition_quantile]
    regular_matrix = actual_debug["ray_transforms"].reshape(-1, 3, 3).double()
    regular_matrix = regular_matrix[(worst_regular // (size * size)).cpu()]
    regular_pixel = expected.reshape(-1, 2).double()[worst_regular.cpu()]
    regular_uv = uv_ref[worst_regular].double()
    regular_screen = torch.matmul(regular_matrix, F.pad(regular_uv, (0, 1), value=1)[..., None])[..., 0]
    regular_screen = regular_screen[:, :2] / regular_screen[:, 2:3]
    stats["cross_regular_worst1024_roundtrip_max_px_float64"] = float((regular_screen - regular_pixel).abs().max())
    # Recompute the problematic inverse from the independent analytic matrix,
    # not from CUDA's float32 cross products. This distinguishes ill-conditioned
    # float32 cross products from an incorrect primitive-to-texture assignment.
    regular_analytic = M_local.reshape(-1, 3, 3).double()
    regular_analytic = regular_analytic[(worst_regular // (size * size)).cpu()]
    regular_analytic_uv = torch.linalg.solve(
        regular_analytic, F.pad(regular_pixel, (0, 1), value=1)[..., None]
    )[..., 0]
    regular_analytic_uv = regular_analytic_uv[:, :2] / regular_analytic_uv[:, 2:3]
    stats["cross_regular_worst1024_analytic_uv_max_error_float64"] = float(
        (regular_analytic_uv - regular_uv).abs().max())
    worst_analytic = M_local.reshape(-1, 3, 3).double()[(worst // (size * size)).cpu()]
    worst_analytic_screen = torch.matmul(
        worst_analytic, F.pad(uv64, (0, 1), value=1)[..., None]
    )[..., 0]
    worst_analytic_screen = worst_analytic_screen[:, :2] / worst_analytic_screen[:, 2:3]
    stats["cross_analytic_worst1024_max_px_float64"] = float(
        (worst_analytic_screen - worst_pixel).abs().max())
    regular_max = int(regular_error.argmax())
    stats["cross_regular_worst_index"] = regular_max
    stats["cross_regular_worst_uv"] = uv_cross[regular_max].cpu().tolist()
    stats["cross_regular_worst_uv_expected"] = uv_ref[regular_max].cpu().tolist()
    stats["cross_regular_worst_condition"] = float(condition[regular_max // (size * size)])
    cross_screen = torch.matmul(worst_matrix, F.pad(uv64, (0, 1), value=1)[..., None])[..., 0]
    cross_screen = cross_screen[:, :2] / cross_screen[:, 2:3]
    stats["cross_roundtrip_worst1024_max_px_float64"] = float((cross_screen - worst_pixel).abs().max())
    stats["cross_roundtrip_worst1024_mean_px_float64"] = float((cross_screen - worst_pixel).abs().mean())
    stats["cross_worst_matrices"] = actual_debug["ray_transforms"].reshape(-1, 3, 3)[
        (worst[:2] // (size * size)).cpu()].cpu().tolist()
    stats["cross_worst_pixels"] = expected.reshape(-1, 2)[worst[:2]].cpu().tolist()
    stats["cross_worst_uv_float32"] = uv_ref[worst[:2]].cpu().tolist()
    stats["cross_worst_uv_recovered_float64"] = uv64[:2].cpu().tolist()
    del uv_cross, uv_ref, reference_uv, cross_error, worst, worst_matrix, worst_pixel
    del M, p_h, cross64, uv64, uv_ref64, hu, hv, cross_screen
    stats["thresholds"].update({
        "center_pixel_max_error": 2e-3,
        "k1_synthetic_max_error_px": 2e-3,
        "interior_synthetic_coordinate_max_error_px": 2e-3,
        "cross_uv_regular_max_error": 2e-4,
        "cross_roundtrip_worst1024_max_px_float64": 2e-3,
        "cross_regular_worst1024_roundtrip_max_px_float64": 2e-3,
        "analytic_matrix_max_abs": 2e-3,
        "packed_fused_max_matrix_error": 2e-5,
        "matrix_center_vs_local_max_px": 2e-3,
    })
    # Natural RGB is diagnostic: even a correct subpixel center interpolation
    # differs from direct texel lookup in high-gradient image regions.
    gate_fields = {
        "rgb_mae": 2e-5,
        "coordinate_max_error_px": 2e-3,
        "center_pixel_max_error": 2e-3,
        "k1_synthetic_max_error_px": 2e-3,
        "interior_synthetic_coordinate_max_error_px": 2e-3,
        "interior_source_id_max_error": 2e-6,
        "interior_coordinate_max_error_px": 2e-3,
        "cross_roundtrip_worst1024_max_px_float64": 2e-3,
        "cross_regular_worst1024_roundtrip_max_px_float64": 2e-3,
        "cross_analytic_worst1024_max_px_float64": 2e-3,
        "analytic_matrix_max_abs": 2e-3,
        "packed_fused_max_matrix_error": 2e-5,
        "matrix_center_vs_local_max_px": 2e-3,
        "rgb_cuda_vs_analytic_center_max": 2e-6,
        "independent_texel_rgb_max_error": 2e-4,
        "independent_interior_texel_rgb_max_error": 2e-4,
    }
    stats["projection_gates"] = {name: {
        "value": stats[name], "threshold": threshold,
        "passed": bool(stats[name] <= threshold)}
        for name, threshold in gate_fields.items()}
    stats["passed"] = bool(
        all(item["passed"] for item in stats["projection_gates"].values()) and
        stats["packed_fused_valid_order_match"])
    print("[projection_detail]", json.dumps(stats_matrix), flush=True)
    del ramps, sampled, synthetic
    return stats


def appearance(head, images, base_rgb, projected, size, textured):
    B, S = images.shape[:2]
    delta = head(images, base_rgb, projected).reshape(
        B, S, 3, size, size, *images.shape[-2:]).permute(0, 1, 5, 6, 2, 3, 4)
    if not textured:
        # Only the last two axes are texels. A K=1 control has [B,S,H,W,3,1,1]
        # here; indexing B,S would silently permute source and image pixels.
        if delta.shape[-2:] != (1, 1):
            raise ValueError(f"Plain control requires one texel, got {delta.shape}")
        delta_rgb = delta[..., 0, 0].reshape(-1, 3)
        if len(delta_rgb) != B * S * images.shape[-2] * images.shape[-1]:
            raise RuntimeError("Plain control shape does not match source splat grid")
        return None, delta_rgb
    # Both branches use activated RGB, not raw SH0, in the base renderer.
    # CUDA samples the learned projected RGB residual within each surfel.
    projected_rgb = projected - base_rgb.permute(0, 1, 3, 4, 2)[..., None, None]
    return (projected_rgb + delta).reshape(-1, 3, size, size), None


def save_alpha(root, name, alpha):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    for i in range(len(alpha)):
        Image.fromarray(image_array(alpha[i, ..., None].expand(-1, -1, 3))).save(
            d / f"{i:02d}_alpha.png")


def image_array(x):
    return np.uint8(np.clip(x.detach().float().cpu().numpy(), 0, 1) * 255)


def save_views(root, name, gt, before, after):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    for i in range(len(gt)):
        for kind, tensor in (("gt", gt), ("before", before), ("after", after)):
            Image.fromarray(image_array(tensor[i])).save(d / f"{i:02d}_{kind}.png")
        a, b, c = (Image.fromarray(image_array(x[i])) for x in (gt, before, after))
        diff = Image.fromarray(np.uint8(np.clip(np.abs(
            image_array(before[i]).astype(float) - image_array(after[i]).astype(float)) * 4, 0, 255)))
        w, h = a.size
        sheet = Image.new("RGB", (w * 4, h + 24), "#20252b")
        draw = ImageDraw.Draw(sheet)
        for j, (label, im) in enumerate(zip(["Target", "Before", "After", "|diff| x4"], [a, b, c, diff])):
            sheet.paste(im, (j*w, 24))
            draw.text((j*w+6, 5), label, fill="white")
        sheet.save(d / f"{i:02d}_comparison.png")


def git_commit(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def check_renderer():
    import gsplat
    from gsplat.cuda._wrapper import TextureAlphaMode
    if not hasattr(TextureAlphaMode, "GAUSSIAN"):
        raise RuntimeError("Vanilla gsplat loaded; install LGTM patched gsplat")
    return str(Path(gsplat.__file__).resolve())


def main():
    a = parse_args()
    if a.res % 14 or a.steps < 0 or a.n_source < 2:
        raise SystemExit("res must be divisible by 14; steps >= 0; n_source >= 2")
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(a.seed)
    out = Path(a.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    lgtm_dir = os.environ.get("LGTM_DIR")
    if not lgtm_dir or not (Path(lgtm_dir) / "src/lgtm/utils/texture_project.py").is_file():
        raise SystemExit("Set LGTM_DIR to the official ml-lgtm checkout")
    gsplat_file = check_renderer()
    repo = M09.official_repo()
    sys.path.insert(0, str(repo))
    _, model, teacher = M09.build_models(repo, dev, pose_mode=a.pose_mode)
    model.eval().requires_grad_(False)
    model.gs_renderer.filter_large_scale_infer = False
    model.gs_renderer.use_gt_camera = a.pose_mode
    views, scene, paths = M09.build_inputs(Path(a.scene) / "images", a.res, dev, a.pose_mode)
    src, tgt, context = camera_split(len(paths), a.n_source, a.camera_context, a.pose_mode)
    cv = subset(views, context, len(paths))
    si = [context.index(i) for i in src]
    sv = subset(cv, si, len(context))
    # Strict pose mode keeps target RGB/K/c2w outside the teacher context.
    tv = subset(views, tgt, len(paths)) if a.pose_mode else subset(cv, [context.index(i) for i in tgt], len(context))
    flags = [1, 0, 1] if a.pose_mode else [0, 0, 0]
    print(f"[LGTM] scene={scene} context={context} source={src} held_out={tgt} "
          f"K={a.texture_size} pose_mode={a.pose_mode}", flush=True)
    with torch.no_grad():
        joint_geometry = teacher(cv, cond_flags=flags)
    if a.pose_mode:
        geometry = subset(joint_geometry, si, len(context))
        target_geometry = None
    else:
        ti = [context.index(i) for i in tgt]
        geometry, target_geometry = (subset(joint_geometry, ids, len(context)) for ids in (si, ti))
    del joint_geometry
    stash, renderer = {}, model.gs_renderer
    original_merge = renderer.merge_voxels

    def capture(splats, keep):
        stash["raw_splats"], stash["keep"] = splats, keep
        return M09.merge_voxels_functional(splats, keep, renderer.voxel_size)

    renderer.merge_voxels = capture
    try:
        preds = M09.forward_train(model, sv, geometry, flags, M09.Chrono(dev), stash)
    finally:
        renderer.merge_voxels = original_merge
    raw, keep = stash["raw_splats"], stash["keep"]
    if not bool(keep.all()) or raw["means"].shape[1] != len(src) * a.res * a.res:
        raise RuntimeError("Unexpected SA filtering; preserve UV correspondence before continuing")
    flat, local, geometry_stats = source_local_geometry(raw, keep, preds, sv)
    flat3d = {k: raw[k].reshape(1, -1, *raw[k].shape[2:]).detach()
              for k in ("means", "quats", "scales", "opacities", "sh")}
    merged = {k: (v[0] if isinstance(v, list) else v)[None].detach()
              for k, v in preds["splats"].items()}
    c2w, K = preds["export_camera_poses"][0].float(), preds["export_camera_intrs"][0].float()
    if a.pose_mode:
        target_c2w, target_K = tv["camera_poses"][0].float().clone(), tv["camera_intrs"][0].float()
        # Estimate the common GSN factor using baselines, including the canonical
        # first camera whose center is exactly zero and cannot give a ratio.
        i, j = torch.triu_indices(len(src), len(src), 1, device=dev)
        prior_baseline = (sv["camera_poses"][0, i, :3, 3] -
                          sv["camera_poses"][0, j, :3, 3]).norm(dim=-1)
        export_baseline = (c2w[i, :3, 3] - c2w[j, :3, 3]).norm(dim=-1)
        factor = (export_baseline / prior_baseline.clamp_min(1e-8)).median()
        target_c2w[:, :3, 3] *= factor
        camera_protocol = "Supplied prior cameras; common official-GSN scale from source only"
        camera_stats = {"gsn_scale": float(factor),
                        "source_rotation_max_error": float((c2w[:, :3, :3] -
                            sv["camera_poses"][0, :, :3, :3]).abs().max()),
                        "source_center_max_error": float((c2w[:, :3, 3] -
                            factor * sv["camera_poses"][0, :, :3, 3]).abs().max())}
    else:
        target_c2w, target_K = target_geometry["camera_poses"][0].float(), target_geometry["camera_intrs"][0].float()
        camera_protocol = "Camera-context diagnostic only; not strict held-out evaluation"
        camera_stats = {}
    print(f"[LGTM] source_K={K.tolist()} camera_centers={c2w[:, :3, 3].tolist()}", flush=True)
    projected = project_source(lgtm_dir, local, sv["img"], K, a.texture_size)
    projection_identity = projection_diagnostics(
        lgtm_dir, local, sv["img"], K, a.texture_size)
    print(f"[LGTM] projection={projection_identity}", flush=True)
    images = sv["img"]
    base_rgb = sh0_to_rgb(flat["sh"][..., 0, :].reshape(
        1, len(src), a.res, a.res, 3).permute(0, 1, 4, 2, 3)).clamp_min(0)
    gt_source, gt_target = images[0].permute(0, 2, 3, 1), tv["img"][0].permute(0, 2, 3, 1)
    print(f"[LGTM] primitive={len(flat['means'][0])} merged={len(merged['means'][0])}", flush=True)
    baseline = {}
    with torch.no_grad():
        sa_rgb, sa_alpha = M09.render_splats(preds, dev, (a.res, a.res))
        baseline["sa3d_source"] = rgb_metrics(sa_rgb, gt_source, sa_alpha[..., 0])
        sa_t, alpha_t = render_3d(merged, target_c2w, target_K, (a.res, a.res))
        baseline["sa3d_held_out"] = rgb_metrics(sa_t, gt_target, alpha_t)
        raw3d_rgb, raw3d_alpha = render_3d(flat3d, c2w, K, (a.res, a.res))
        baseline["raw3d_source"] = rgb_metrics(raw3d_rgb, gt_source, raw3d_alpha)
        zero = torch.zeros(len(flat["means"][0]), 3, a.texture_size, a.texture_size, device=dev)
        plain_rgb, plain_alpha, _ = render_2d(flat, c2w, K, (a.res, a.res))
        zero_rgb, zero_alpha, _ = render_2d(flat, c2w, K, (a.res, a.res), zero)
        initial_texture = (projected -
            base_rgb.permute(0, 1, 3, 4, 2)[..., None, None]).reshape(
            -1, 3, a.texture_size, a.texture_size)
        texture_rgb, texture_alpha, texture_raw_rgb = render_2d(
            flat, c2w, K, (a.res, a.res), initial_texture)
    initial_rgb = {"plain": plain_rgb.detach(), "lgtm": texture_rgb.detach()}
    save_views(out, "initial_source", gt_source, plain_rgb, texture_rgb)
    save_alpha(out, "initial_plain_source", plain_alpha)
    save_alpha(out, "initial_texture_source", texture_alpha)
    smoke = {"constant_zero_max_rgb_diff": float((plain_rgb-zero_rgb).abs().max()),
             "constant_zero_max_alpha_diff": float((plain_alpha-zero_alpha).abs().max()),
             "projected_min": float(projected.min()), "projected_max": float(projected.max()),
             "texture_finite": bool(torch.isfinite(texture_rgb).all()),
             "initial_texture_unclamped": {
                 "min": float(texture_raw_rgb.min()), "max": float(texture_raw_rgb.max()),
                 "negative_fraction": float((texture_raw_rgb < 0).float().mean())},
             "initial_plain": rgb_metrics(plain_rgb, gt_source, plain_alpha),
             "initial_texture": rgb_metrics(texture_rgb, gt_source, texture_alpha),
             "camera": camera_stats,
             "projection_center_identity": projection_identity,
             "geometry": geometry_stats,
             "merged_count": len(merged["means"][0]),
             "raw_mean_min": float(flat3d["means"].min()),
             "raw_mean_max": float(flat3d["means"].max()),
             "raw_scale_min": float(flat3d["scales"].min()),
             "raw_scale_median": float(flat3d["scales"].median()),
             "raw_scale_max": float(flat3d["scales"].max())}
    with open(out / "smoke.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(smoke, f, ensure_ascii=False, indent=2)
        f.write("\n")
    if not projection_identity["passed"]:
        raise RuntimeError("Source texture projection identity failed; training is blocked: " +
                           str(projection_identity))
    if (smoke["constant_zero_max_rgb_diff"] > 2e-5 or
            smoke["constant_zero_max_alpha_diff"] > 2e-5 or not smoke["texture_finite"] or
            geometry_stats["local_normal_error"] > 1e-5):
        raise RuntimeError("Renderer/texture smoke failed; training is blocked")
    if a.pose_mode and (camera_stats["source_rotation_max_error"] > 1e-5 or
                        camera_stats["source_center_max_error"] > 1e-5):
        raise RuntimeError("GSN source cameras are not a pure common scale")
    del model, teacher, raw, preds, zero, plain_rgb, zero_rgb, texture_rgb, local, flat3d
    del stash, keep, plain_alpha, zero_alpha, texture_alpha, sa_rgb, sa_alpha
    del sa_t, alpha_t, raw3d_rgb, raw3d_alpha, initial_texture
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(a.seed)
    prototype = AppearanceHead(a.texture_size, a.channels).to(dev)
    torch.manual_seed(a.seed + 1)
    plain_initial = AppearanceHead(1, a.channels).to(dev)
    common = {k: v for k, v in prototype.state_dict().items()
              if k in plain_initial.state_dict() and v.shape == plain_initial.state_dict()[k].shape}
    plain_initial.load_state_dict(common, strict=False)
    results = {}
    for arm in ("lgtm", "plain"):
        size = a.texture_size if arm == "lgtm" else 1
        h = prototype if arm == "lgtm" else plain_initial.to(dev)
        h.load_state_dict((prototype if arm == "lgtm" else plain_initial).state_dict())
        h.requires_grad_(True)
        h.train()
        opt = torch.optim.AdamW(h.parameters(), lr=a.lr, weight_decay=1e-6)
        assert all(p.requires_grad for p in h.parameters()), "Appearance head is frozen"
        parameter_before = {name: p.detach().clone() for name, p in h.named_parameters()}
        # Pool projected RGB for the K=1 control, with axes kept as
        # [B,S,H,W,3,K,K]. This also avoids accidental axis permutation.
        arm_projected = projected if arm == "lgtm" else projected.mean(
            dim=(-2, -1), keepdim=True
        )
        records, grad_check, tensor_grads, initial_check = [], {}, {}, {}
        t0 = time.perf_counter()
        for step in range(a.steps + 1):
            tx, delta = appearance(h, images, base_rgb, arm_projected, size, arm == "lgtm")
            rgb, alpha, raw_rgb = render_2d(flat, c2w, K, (a.res, a.res), tx, delta)
            loss = F.mse_loss(rgb, gt_source)
            if step == 0:
                # Validate the actual training render, not just an auxiliary path.
                with torch.no_grad():
                    h.eval()
                    tx0, delta0 = appearance(h, images, base_rgb, arm_projected,
                                             size, arm == "lgtm")
                    eval_rgb = render_2d(flat, c2w, K, (a.res, a.res), tx0, delta0)[0]
                    h.train()
                baseline_rgb = initial_rgb[arm]
                train_eval_error = float((rgb - eval_rgb).abs().max())
                initial_error = float((eval_rgb - baseline_rgb).abs().max())
                initial_check = {"train_eval_max_rgb_diff": train_eval_error,
                                 "eval_smoke_max_rgb_diff": initial_error,
                                 "threshold": 2e-5}
                print(f"[{arm}] initial train/eval diff={train_eval_error:.8g} "
                      f"baseline diff={initial_error:.8g}", flush=True)
                if train_eval_error > 2e-5 or initial_error > 2e-5:
                    raise RuntimeError(f"{arm} initial rendering is not reproducible")
            if step < a.steps:
                opt.zero_grad(set_to_none=True)
                if step == 0:
                    if tx is not None:
                        tx.retain_grad()
                    if delta is not None:
                        delta.retain_grad()
                    print(f"[{arm}] graph rgb={rgb.requires_grad} loss={loss.requires_grad} "
                          f"tx={tx is not None and tx.requires_grad} delta={delta is not None and delta.requires_grad}", flush=True)
                    print(f"[{arm}] unclamped rgb min={float(raw_rgb.min()):.6f} max={float(raw_rgb.max()):.6f} "
                          f"mean={float(raw_rgb.mean()):.6f} negative={float((raw_rgb < 0).float().mean()):.6f} "
                          f"alpha={float(alpha.mean()):.6f}", flush=True)
                loss.backward()
                if step == 0:
                    grad_check = {n: {"finite": bool(g is not None and torch.isfinite(g).all()),
                                      "abs_sum": None if g is None else float(g.abs().sum())}
                                  for n, g in ((n, p.grad) for n, p in h.named_parameters())}
                    tensor_grads = {}
                    if tx is not None:
                        tensor_grads["texture_tensor"] = None if tx.grad is None else float(tx.grad.abs().sum())
                    if delta is not None:
                        tensor_grads["delta_tensor"] = None if delta.grad is None else float(delta.grad.abs().sum())
                    print(f"[{arm}] grad_check={grad_check} tensor_grads={tensor_grads}", flush=True)
                    valid_tensor_grads = [v for v in tensor_grads.values() if v is not None]
                    if (not all(v["finite"] for v in grad_check.values())
                            or not valid_tensor_grads or not any(v > 0 for v in valid_tensor_grads)
                            or not any(v["abs_sum"] and v["abs_sum"] > 0 for v in grad_check.values())):
                        raise RuntimeError("Invalid rendering gradient: " +
                                           str({"parameters": grad_check, "tensors": tensor_grads}))
                torch.nn.utils.clip_grad_norm_(h.parameters(), 1.)
                opt.step()
            if step % a.log_every == 0 or step == a.steps:
                rec = {"step": step, **rgb_metrics(rgb.detach(), gt_source, alpha.detach())}
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                rec["wall_s"] = time.perf_counter() - t0
                rec["peak_alloc_GB"] = torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else 0
                records.append(rec)
                print(f"[{arm}] step={step} psnr={rec['psnr']:.3f} alpha={rec['alpha_mean']:.3f}", flush=True)
            if step == 0:
                before_source = (rgb.detach(), alpha.detach())
                with torch.no_grad():
                    before_target = render_2d(flat, target_c2w, target_K,
                                              (a.res, a.res), tx, delta)[:2]
        with torch.no_grad():
            tx, delta = appearance(h, images, base_rgb, arm_projected, size, arm == "lgtm")
            after_source = render_2d(flat, c2w, K, (a.res, a.res), tx, delta)[:2]
            after_target = render_2d(flat, target_c2w, target_K, (a.res, a.res), tx, delta)[:2]
        parameter_delta = {name: float((p.detach() - parameter_before[name]).abs().max())
                           for name, p in h.named_parameters()}
        if a.steps and not any(delta > 0 for delta in parameter_delta.values()):
            raise RuntimeError("Optimizer completed without changing any appearance parameter")
        results[arm] = {"train": records, "step1_grad": grad_check, "step1_tensor_grad": tensor_grads,
                        "initial_consistency": initial_check,
                        "parameter_max_abs_update": parameter_delta,
                        "parameter_count": sum(p.numel() for p in h.parameters()),
                        "source_before": rgb_metrics(before_source[0], gt_source, before_source[1]),
                        "source_after": rgb_metrics(after_source[0], gt_source, after_source[1]),
                        "held_out_before": rgb_metrics(before_target[0], gt_target, before_target[1]),
                        "held_out_after": rgb_metrics(after_target[0], gt_target, after_target[1]),
                        "source_per_view_after": [rgb_metrics(after_source[0][i:i+1], gt_source[i:i+1],
                                                             after_source[1][i:i+1]) for i in range(len(src))],
                        "held_out_per_view_after": [rgb_metrics(after_target[0][i:i+1], gt_target[i:i+1],
                                                               after_target[1][i:i+1]) for i in range(len(tgt))]}
        save_views(out, arm + "_source", gt_source, before_source[0], after_source[0])
        save_views(out, arm + "_held_out", gt_target, before_target[0], after_target[0])
        save_alpha(out, arm + "_source", after_source[1])
        save_alpha(out, arm + "_held_out", after_target[1])
        torch.save({"head": h.state_dict(), "args": vars(a)}, out / (arm + "_head.pt"))
        del h, opt
    report = {"protocol": {"scene": scene, "source": src, "held_out": tgt, "camera_context": context,
              "source_files": [str(paths[i]) for i in src], "target_files": [str(paths[i]) for i in tgt],
              "context_files": [str(paths[i]) for i in context], "resolution": a.res,
              "texture_size": a.texture_size, "steps": a.steps, "lr": a.lr, "seed": a.seed,
              "geometry": "Frozen SA pre-merge elliptical surfels; smallest-scale axis flattened to local Z; two original tangent scales retained; no voxel merge in either arm",
              "alpha": "Frozen SA opacity, Gaussian only", "scope": "Single-scene fitting, not cross-scene training",
              "camera": camera_protocol,
              "target": "Target RGB never enters texture optimization; joint camera context is not GT calibration",
              "shared_initialization": "Same-seed common layer state; texture arms have extra capacity"},
              "provenance": {"splatanything_commit": git_commit(repo), "lgtm_commit": git_commit(lgtm_dir),
                             "gsplat_file": gsplat_file, "torch": torch.__version__,
                             "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
              "counts": {"raw": len(flat["means"][0]), "merged": len(merged["means"][0])},
              "smoke": smoke, "baseline": baseline, "arms": results}
    with open(out / "report.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("[Done]", out)


if __name__ == "__main__":
    main()
