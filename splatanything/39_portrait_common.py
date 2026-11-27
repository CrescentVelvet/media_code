#!/usr/bin/env python3
"""Shared, strictly source/target-separated utilities for SA portrait short runs.

No SA architecture is changed. Only the existing Gaussian heads are optimized;
the external geometry teacher and the source feature cache are frozen. The
functions that transform cameras/depth are also used by the CPU contract tests.
"""
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent


def load_local(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M09 = load_local("portrait_train09", "09_train_probe.py")


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def move_tree(value, device, detach=False):
    if torch.is_tensor(value):
        value = value.detach() if detach else value
        return value.to(device)
    if isinstance(value, dict):
        return {k: move_tree(v, device, detach) for k, v in value.items()}
    if isinstance(value, list):
        return [move_tree(v, device, detach) for v in value]
    if isinstance(value, tuple):
        return tuple(move_tree(v, device, detach) for v in value)
    return value


def load_manifest(path):
    m = read_json(path)
    if m.get("schema") != "ue53_portrait_finetune_manifest_v1":
        raise ValueError("Expected the frozen UE53 portrait manifest")
    return m


def sequence(manifest, seq_id):
    found = [s for s in manifest["sequences"] if s["sequence_id"] == seq_id]
    if len(found) != 1:
        raise ValueError("Unknown or duplicate sequence_id: " + seq_id)
    return found[0]


def select_sequences(manifest, split, limit=0, sequences=()):
    if sequences:
        rows = [sequence(manifest, s) for s in sequences]
    else:
        rows = [s for s in manifest["sequences"] if s["split"] == split]
    if limit:
        rows = rows[:limit]
    if not rows:
        raise ValueError("No sequences selected: " + split)
    if split == "train" and any(not s["training_eligible"] for s in rows):
        raise ValueError("c6-fail sequence cannot enter optimization")
    return rows


def resolve_view(manifest, row, cid):
    if cid not in row["views"]:
        raise ValueError("View not audited in manifest: %s/%s" % (row["sequence_id"], cid))
    v = dict(row["views"][cid])
    package = Path(manifest["dataset_root"]) / row["package"]
    v["rgb_path"] = package / v["rgb"]
    v["depth_path"] = package / v["depth_radial"]
    v["mask_path"] = package / v["foreground_mask"]
    v["K_np"] = np.asarray(v["K"], np.float32)
    v["c2w_np"] = np.asarray(v["c2w"], np.float32)
    v["cid"] = cid
    return v


def views_from_records(records, device):
    """Exactly reproduce build_views for these already ordered image records."""
    from PIL import Image
    from src.utils.scene_io import preprocess_view
    imgs, Ks, poses = [], [], []
    for rec in records:
        with Image.open(rec["rgb_path"]) as im:
            img_t, K = preprocess_view(im, rec["K_np"], rec.get("resolution_wh", (504, 504)))
        imgs.append(img_t)
        Ks.append(K)
        poses.append(torch.from_numpy(rec["c2w_np"]).double())
    views = {"img": torch.stack(imgs)[None].to(device),
             "camera_intrs": torch.stack(Ks)[None].to(device).float(),
             "camera_poses": torch.stack(poses)[None].to(device).float()}
    # UE w2c matrices carry small float roundoff. torch.float32 LU inversion
    # can amplify that roundoff into a wrong rotation; preserve the official
    # double-precision first-source alignment, then cast the rigid result.
    poses64 = torch.stack(poses)[None].double()
    first_inv = torch.linalg.inv(poses64[:, 0])
    relative = (first_inv[:, None] @ poses64).float().to(device)
    views["camera_poses"] = relative
    # Retain the one double-precision first-source inverse for raw target
    # cameras. Never invert the already relative, float32 first pose again.
    views["source_first_w2c"] = first_inv.to(device)
    views["camera_pose_source"] = "source_first_w2c @ original package c2w"
    first = views["camera_poses"][0, 0].double()
    ortho = views["camera_poses"][0, :, :3, :3].double()
    if ((first - torch.eye(4, dtype=torch.float64, device=device)).abs().max() > 1e-5
            or (ortho @ ortho.transpose(-1, -2) - torch.eye(3, dtype=torch.float64, device=device)).abs().max() > 1e-5
            or (torch.linalg.det(ortho) - 1).abs().max() > 1e-5):
        raise ValueError("Source first-camera alignment is not a valid rigid frame")
    return views


def preprocess_target(rec, res):
    """Same RGB/K as the official forward, plus nearest axial-Z and person mask.

    radial is in millimetres and converted BEFORE nearest resize. scene_io's
    historical final crop returns the pre-crop K; here the actually returned K
    is used for both RGB lifting and target projection, and separately audited
    against the native forward in the GPU smoke.
    """
    import cv2
    from PIL import Image
    from src.utils.scene_io import preprocess_view
    K = rec["K_np"]
    with Image.open(rec["rgb_path"]) as im:
        native_wh = im.size
        gt_t, K_t = preprocess_view(im, K, (res, res))
    radial = cv2.imread(str(rec["depth_path"]), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(rec["mask_path"]), cv2.IMREAD_GRAYSCALE)
    if radial is None or mask is None or radial.dtype != np.uint16:
        raise ValueError("Missing uint16 radial depth/person mask: " + rec["cid"])
    if radial.shape != mask.shape or (radial.shape[1], radial.shape[0]) != native_wh:
        raise ValueError("Target RGB/depth/mask mismatch: " + rec["cid"])
    radial_m = radial.astype(np.float32) * .001
    yy, xx = np.indices(radial_m.shape, dtype=np.float32)
    norm = np.sqrt(1. + ((xx - K[0, 2]) / K[0, 0]) ** 2 +
                   ((yy - K[1, 2]) / K[1, 1]) ** 2)
    axial = np.where(radial > 0, radial_m / norm, 0.).astype(np.float32)
    with Image.open(rec["rgb_path"]) as im:
        _, official_K, z = preprocess_view(im, K, (res, res), depthmap=axial)
        _, _, person = preprocess_view(im, K, (res, res),
                                       depthmap=(mask > 127).astype(np.float32))
    if not torch.equal(official_K, K_t):
        raise RuntimeError("Official preprocessing is not deterministic for target K")
    person = person > .5
    valid = (z > 0) & person
    return {"gt": gt_t.permute(1, 2, 0).float(), "K": K_t.float(),
            "depth_z_m": z.float(), "person": person, "valid_depth": valid,
            "c2w": torch.from_numpy(rec["c2w_np"]).float(), "cid": rec["cid"]}


def radial_to_axial(radial, K):
    """Metric radial distance -> camera axial Z. Pure numpy reference for tests."""
    radial = np.asarray(radial)
    K = np.asarray(K)
    if radial.shape[-2:] != (int(K[1, 2] * 2 + 1), int(K[0, 2] * 2 + 1)):
        # General image sizes are valid; this is only a diagnostic for callers
        # expecting a centered principal point. No geometry depends on it.
        pass
    yy, xx = np.indices(radial.shape[-2:], dtype=np.float64)
    z = radial / np.sqrt(1. + ((xx - K[0, 2]) / K[0, 0]) ** 2 +
                         ((yy - K[1, 2]) / K[1, 1]) ** 2)
    return np.where(radial > 0, z, 0.)


def package_to_source_camera(c2w, source_views):
    """Map an original package camera to the first-source coordinate frame.

    views_from_records computes P_i=P_0^-1 C_i. Therefore the corresponding
    target is also P_0^-1 C_target, not P_0 C_target. Exported cameras and the
    lifted cloud use the same relative frame; this must be applied exactly once.
    """
    if torch.is_tensor(c2w):
        c2w = c2w.detach().cpu().numpy()
    c2w = np.asarray(c2w, dtype=np.float64)
    if c2w.shape != (4, 4):
        raise ValueError("Expected a 4x4 original package c2w")
    prior = source_views["source_first_w2c"].detach().cpu().numpy().astype(np.float64)
    if prior.shape == (1, 4, 4):
        prior = prior[0]
    return torch.from_numpy(prior @ c2w).float()


def fit_source_sim3(source_views, exported_poses, tol=2e-4):
    """Source-only X_model=s R X_relative+T, with R_model=R R_relative."""
    prior = source_views["camera_poses"][0].detach().cpu().numpy().astype(np.float64)
    exported = exported_poses.detach().cpu().numpy().astype(np.float64)
    if prior.shape != exported.shape or prior.ndim != 3 or prior.shape[1:] != (4, 4):
        raise ValueError("Source/exported camera arrays must have equal [N,4,4] shapes")
    rotations = np.stack([e[:3, :3] @ p[:3, :3].T for e, p in zip(exported, prior)])
    R = rotations.mean(axis=0)
    u, _, vt = np.linalg.svd(R)
    if np.linalg.det(u @ vt) <= 0:
        raise ValueError("Source-only camera alignment reflected a coordinate frame")
    R = u @ vt
    pairs, s = [], []
    for i in range(len(prior)):
        for j in range(i):
            den = np.linalg.norm(prior[i, :3, 3] - prior[j, :3, 3])
            if den > 1e-8:
                pairs.append(float(np.linalg.norm(exported[i, :3, 3] - exported[j, :3, 3]) / den))
    if not pairs:
        raise ValueError("Cannot estimate scale from coincident source cameras")
    s = float(np.median(pairs))
    if not 0 < s < 1e5:
        raise ValueError("Invalid source-only GSN scale")
    T = exported[0, :3, 3] - s * R @ prior[0, :3, 3]
    dR = np.abs(rotations - R).max()
    residuals = [float(np.max(np.abs(s * R @ p[:3, 3] + T - e[:3, 3])))
                 for p, e in zip(prior, exported)]
    if dR > tol or max(residuals) > tol or max(pairs) - min(pairs) > tol:
        raise ValueError("Source camera alignment failed: dR=%g residual=%g spread=%g" %
                         (dR, max(residuals), max(pairs) - min(pairs)))
    return {"s": s, "R": R, "T": T, "pair_scales": pairs,
            "rotation_max_abs": float(dR), "center_max_abs": max(residuals)}


def target_to_model(c2w, fit, device):
    """Scale only the camera centre; never put s inside its rotation block."""
    if torch.is_tensor(c2w):
        c2w = c2w.detach().cpu().numpy()
    c2w = np.asarray(c2w, dtype=np.float64)
    out = np.eye(4, dtype=np.float64)
    # N1 target_camera first inverts its model->package fit. Here the fit is
    # already source/package -> model, so apply R @ R_source directly.
    out[:3, :3] = fit["R"] @ c2w[:3, :3]
    out[:3, 3] = fit["s"] * fit["R"] @ c2w[:3, 3] + fit["T"]
    return torch.from_numpy(out).float().to(device)


def model_to_package_camera(c2w, fit, device):

    """Inverse rigid+scale transform used for independent rendering of a PLY."""
    R, s, T = fit["R"], fit["s"], fit["T"]
    if torch.is_tensor(c2w):
        c2w = c2w.detach().cpu().numpy()
    c2w = np.asarray(c2w, np.float64)
    rotation = R.T @ c2w[:3, :3]
    center = R.T @ (c2w[:3, 3] - T) / s
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = rotation, center
    return torch.from_numpy(out).float().to(device)


def set_trainable(model):
    assert model.gs_renderer.use_gt_camera
    return M09.set_trainable(model)


def build_models(repo, device):
    repo = Path(repo).expanduser().resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    cfg, model, teacher = M09.build_models(repo, device, pose_mode=True)
    model.gs_renderer.use_gt_camera = True
    model.eval()
    teacher.eval().requires_grad_(False)
    return cfg, model, teacher


def source_cache(model, teacher, records, device):
    """Teacher/encoder sees exactly three source images, never a target view."""
    if len(records) != 3:
        raise ValueError("Portrait protocol requires exactly three sources")
    views = views_from_records(records, device)
    flags = [1, 0, 1]
    with torch.no_grad():
        geometry = teacher(views, cond_flags=flags)
        tokens, patch_start = model.encode(views, flags)
    return {"views": move_tree(views, "cpu", True),
            "geometry": move_tree(geometry, "cpu", True),
            "tokens": move_tree(tokens, "cpu", True),
            "patch_start": move_tree(patch_start, "cpu", True),
            "allowed_keys": {"views", "geometry", "tokens", "patch_start",
                             "source_ids", "allowed_keys"},
            "source_ids": [r["cid"] for r in records]}


def predict(model, cache, device, stash=None):
    views, geometry = (move_tree(cache[k], device) for k in ("views", "geometry"))
    tokens = move_tree(cache["tokens"], device)
    patch_start = cache["patch_start"]

    def features(start, end):
        return model.gs_head.forward_chunk(tokens, views["img"], patch_start, start, end)

    renderer = model.gs_renderer
    if not renderer.use_gt_camera:
        raise RuntimeError("Known-camera protocol is disabled")
    original = renderer.merge_voxels
    if stash is None:
        renderer.merge_voxels = lambda sp, keep: M09.merge_voxels_functional(
            sp, keep, renderer.voxel_size)
    else:
        def capture(sp, keep):
            stash["raw"], stash["keep"] = sp, keep
            return M09.merge_voxels_functional(sp, keep, renderer.voxel_size)
        renderer.merge_voxels = capture
    try:
        with torch.autocast(device_type=device.type, enabled=False):
            return renderer(views, geometry, features)
    finally:
        renderer.merge_voxels = original


def splat_list(preds):
    return {k: (v[0] if isinstance(v, list) else v[0]) for k, v in preds["splats"].items()}


def render_splats(preds, cameras, intrinsics, hw, background=0., eps2d=.01, mode="RGB+ED"):
    """Differentiable gsplat rendering; ED is alpha-normalized axial model Z.

    An auxiliary single-channel D pass preserves depth backward at empty pixels;
    RGB+ED's native 0/clamp(alpha) division is otherwise non-differentiable there.
    It uses the same projection and splats; evaluation can use the native ED.
    """
    from gsplat import rasterization
    sp = splat_list(preds)
    colors = sp["sh"][None].expand(len(cameras), -1, -1, -1).contiguous()
    bg = torch.full((len(cameras), 3), float(background), device=sp["means"].device)
    common = dict(means=sp["means"], quats=sp["quats"], scales=sp["scales"],
                  opacities=sp["opacities"], viewmats=torch.linalg.inv(cameras.float()),
                  Ks=intrinsics.float(), width=hw[1], height=hw[0], sh_degree=0,
                  eps2d=eps2d, near_plane=.01, far_plane=400.)
    rgb, alpha, info = rasterization(colors=colors, backgrounds=bg,
                                    render_mode=mode, **common)
    result = {"rgb": rgb[..., :3] if mode == "RGB+ED" else rgb,
              "alpha": alpha[..., 0], "depth": rgb[..., 3] if mode == "RGB+ED" else None,
              "info": info}
    if mode == "RGB+ED" and torch.is_grad_enabled() and torch.is_inference_mode_enabled() is False:
        # This gsplat build requires RGB-shaped colors even for scalar D.
        depth_colors = sp["means"][:, 2][None, :, None, None].expand(
            len(cameras), -1, 1, 3).contiguous()
        accumulated, _, _ = rasterization(
            colors=depth_colors,
            backgrounds=torch.zeros((len(cameras), 3), device=sp["means"].device),
            render_mode="D", **common)
        accum = accumulated[..., 0]
        # Compute ED from D and detached alpha. This is mathematically identical
        # to RGB+ED at forward, but avoids non-finite gradients at alpha==0.
        ed = torch.where(alpha[..., 0].detach() > 1e-10,
                         accum / alpha[..., 0].detach().clamp_min(1e-10),
                         torch.zeros_like(accum))
        result["depth_accum"] = accum
        result["depth_ed_for_metric"] = ed
    return result


def stable_depth_loss(rendered, target, fit):
    """GT-masked ED loss with differentiable D, including empty-pixel coverage."""
    valid = target["valid_depth"].to(rendered["rgb"].device)
    if "depth_accum" not in rendered:
        depth = rendered["depth"]
    else:
        alpha = rendered["alpha"].detach()
        depth = torch.where(alpha > 1e-10,
                            rendered["depth_accum"] / alpha.clamp_min(1e-10),
                            torch.zeros_like(alpha))
    value = (((depth / fit["s"] - target["depth_z_m"].to(depth.device)) ** 2)[valid].mean()
             if valid.any() else rendered["rgb"].sum() * 0)
    return value, int(valid.sum())


def masked_losses(rendered, target, fit, depth_weight=.1):
    rgb, alpha, depth = rendered["rgb"], rendered["alpha"], rendered["depth"]
    gt, person = target["gt"].to(rgb.device), target["person"].to(rgb.device)
    if not person.any():
        raise ValueError("Target person mask is empty")
    depth_loss, depth_valid_n = stable_depth_loss(rendered, target, fit)
    # Supervise both person and background; without an alpha term erasing the
    # subject can look like reduced ghosting in a lowpass/PSNR comparison.
    rgb_person = ((rgb - gt) ** 2).mean(-1)[person].mean()
    rgb_full = F.mse_loss(rgb, gt)
    alpha_loss = F.mse_loss(alpha, person.to(alpha.dtype))
    total = rgb_full + 2. * rgb_person + .1 * alpha_loss + depth_weight * depth_loss
    return {"loss": total, "rgb_full": rgb_full, "rgb_person": rgb_person,
            "alpha": alpha_loss, "depth_z_m2": depth_loss, "depth_valid_n": depth_valid_n}


def batch_targets(records, res, device):
    rows = [preprocess_target(r, res) for r in records]
    out = {k: torch.stack([r[k] for r in rows]).to(device)
           for k in ("gt", "K", "depth_z_m", "person", "valid_depth", "c2w")}
    return out, [r["cid"] for r in rows]


def checkpoint_names(rundir, explicit=()):
    if explicit:
        paths = [Path(p) for p in explicit]
    else:
        paths = [Path(rundir) / "B0.pt"]
        paths += sorted(Path(rundir).glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    for p in paths:
        if not p.is_file():
            raise ValueError("Missing checkpoint: " + str(p))
    return paths


def load_trainable(model, path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    state = blob["trainable"] if "trainable" in blob else blob["model"]
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain a trainable parameter dictionary")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in state:
                if not torch.is_tensor(state[name]) or param.shape != state[name].shape:
                    raise ValueError("Checkpoint shape/type mismatch: " + name)
                param.copy_(state[name])
    return blob


def save_checkpoint(path, step, trainable, optimizer, args, manifest, extra=None):
    state = {"schema": "sa_portrait_student_checkpoint_v1", "step": int(step),
             "trainable": {n: p.detach().cpu() for n, p in trainable},
             "optimizer": optimizer.state_dict(), "args": vars(args),
             "manifest": str(manifest), "wall_time": time.time()}
    if extra:
        state.update(extra)
    tmp = Path(str(path) + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def finite_gradients(params):
    return all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in params)


def target_camera_batch(target, fit, device, source_views):
    return torch.stack([target_to_model(
        package_to_source_camera(c, source_views), fit, device) for c in
        target["c2w"].detach().cpu().numpy()])
