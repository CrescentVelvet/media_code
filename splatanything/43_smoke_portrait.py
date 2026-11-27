#!/usr/bin/env python3
"""GPU smoke for SA portrait training without updating any official weight.

Requires one small synthetic scene, not the UE dataset. Checks:
  - source-only known-camera cache (exactly three sources);
  - official returned-K versus independent cropping reference;
  - target camera Sim(3) and inverse roundtrip;
  - differentiable target RGB+ED and finite, nonzero student gradient;
  - frozen teacher/source encoder and zero loss-impacting target leakage.
The official model architecture and optimizer update count remain unchanged.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from importlib.util import module_from_spec, spec_from_file_location
_spec = spec_from_file_location("sa_portrait_common", Path(__file__).with_name("39_portrait_common.py"))
C = module_from_spec(_spec)
_spec.loader.exec_module(C)


def synthetic(root, res=112):
    root.mkdir(parents=True, exist_ok=False)
    records = []
    for i in range(5):
        im = Image.new("RGB", (res, res), (30, 35, 40))
        d = ImageDraw.Draw(im)
        d.rectangle((i * 3 + 16, 12, i * 3 + 72, 102), fill=(140, 100, 90))
        d.ellipse((i * 3 + 22, 18, i * 3 + 66, 62), fill=(200, 160, 120))
        d.rectangle((i * 3 + 33, 75, i * 3 + 56, 94), fill=(55, 120, 160))
        path = root / ("s%02d.png" % i)
        im.save(path)
        angle = (i - 2) * .18
        c = np.array([3. * np.sin(angle), 0., 3. * np.cos(angle)])
        forward = -c / np.linalg.norm(c)
        right = np.cross(forward, np.array([0., 1., 0.]))
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        c2w = np.eye(4)
        c2w[:3, :3] = np.stack([right, down, forward], axis=1)
        assert np.linalg.det(c2w[:3, :3]) > .999
        c2w[:3, 3] = c
        records.append({"cid": path.stem, "rgb_path": path,
                        "K_np": np.array([[110., 0., (res - 1) / 2],
                                         [0., 110., (res - 1) / 2], [0., 0., 1.]], np.float32),
                        "c2w_np": c2w.astype(np.float32), "resolution_wh": (res, res)})
    return records


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--official-repo", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--res", type=int, default=112)
    args = p.parse_args(argv)
    if args.res % 14:
        raise SystemExit("res must be divisible by 14")
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    records = synthetic(out / "synthetic", args.res)
    _, model, teacher = C.build_models(Path(args.official_repo), device)
    trainable = C.set_trainable(model)
    cache = C.source_cache(model, teacher, records[:3], device)
    if cache["source_ids"] != [r["cid"] for r in records[:3]]:
        raise RuntimeError("Unexpected source ordering")
    preds = C.predict(model, cache, device)
    fit = C.fit_source_sim3(cache["views"], preds["export_camera_poses"][0])
    sp = C.splat_list(preds)
    means, scales, quats, opacities = (sp[k].detach().cpu().numpy() for k in
                                      ("means", "scales", "quats", "opacities"))
    np.savez(out / "gaussians_diagnostic.npz", means=means, scales=scales,
             quats=quats, opacities=opacities)
    stash = {}
    with torch.no_grad():
        diag_preds = C.predict(model, cache, device, stash)
    target_relative = C.package_to_source_camera(records[3]["c2w_np"], cache["views"])
    target = C.target_to_model(target_relative, fit, device)[None]
    rt = C.model_to_package_camera(target[0], fit, device).cpu().numpy()
    if np.max(np.abs(rt - target_relative.numpy())) > 2e-5:
        raise RuntimeError("Target camera roundtrip failed")
    from PIL import Image
    from src.utils.scene_io import preprocess_view
    with Image.open(records[0]["rgb_path"]) as im:
        _, actual_K = preprocess_view(im, records[0]["K_np"], (args.res, args.res))
        expected_K = C.views_from_records(records[:1], "cpu")["camera_intrs"][0, 0]
    if not torch.equal(actual_K, expected_K):
        raise RuntimeError("build_views and source cache K differ")
    # Audit the historical returned pre-crop K rather than silently repairing it.
    M21 = C.load_local("portrait_smoke_preprocess21", "21_geometry_audit_v2.py")
    ref_K = M21.official_K_reference(records[0]["K_np"], (args.res, args.res), (args.res, args.res))
    # Do not require K_reference to equal the actual forward K for non-square
    # real packages. The synthetic square is exactly the returned K by design.
    k_delta = float(np.max(np.abs(ref_K - actual_K.numpy())))
    Ks = preds["export_camera_intrs"][0].float()
    rendered = C.render_splats(preds, target, Ks[:1], (args.res, args.res))
    projected = (torch.linalg.inv(target[0]) @ torch.cat([
        sp["means"], torch.ones((len(sp["means"]), 1), device=device)], dim=-1).T).T
    z = projected[:, 2]
    uv = projected[:, :2] / z.clamp_min(1e-8)[:, None] * Ks[0, :2, :2].diagonal() + Ks[0, :2, 2]
    in_frame = (z > .01) & (z < 400.) & (uv[:, 0] >= 0) & (uv[:, 0] < args.res) & (uv[:, 1] >= 0) & (uv[:, 1] < args.res)
    projected_summary = {"n_gaussians": len(z), "z_min": float(z.min()), "z_max": float(z.max()),
                         "in_target_frustum": int(in_frame.sum()),
                         "scale_mean": sp["scales"].mean(dim=0).tolist(),
                         "opacity_mean": float(sp["opacities"].mean())}
    covered = rendered["alpha"] > .1
    ed_delta = (rendered["depth_ed_for_metric"] - rendered["depth"]).abs()
    depth_forward_check = {
        "covered_pixels": int(covered.sum()),
        "max_abs": float(ed_delta[covered].max()) if covered.any() else None,
        "empty_pixels": int((~covered).sum()),
    }
    if not covered.any() or float(rendered["alpha"].max()) <= .1:
        C.dump_json(out / "gradient_diagnostics.json", {"status": "uncovered_target",
                    "optimizer_steps": 0, "target_projection": projected_summary,
                    "alpha_max": float(rendered["alpha"].max()),
                    "depth_forward_check": depth_forward_check})
        raise RuntimeError("Synthetic target is uncovered; not a valid forward/gradient smoke")
    if not torch.isfinite(rendered["rgb"]).all() or not torch.isfinite(rendered["depth"]).all():
        raise RuntimeError("Non-finite target render")
    if float(ed_delta[covered].max()) > 1e-5:
        raise RuntimeError("Auxiliary D forward disagrees with native ED")
    raw = stash["raw"]
    geom = C.move_tree(cache["geometry"], device)
    geometry_summary = {}
    for key, value in geom.items():
        if torch.is_tensor(value):
            geometry_summary[key] = {"shape": list(value.shape), "finite": bool(torch.isfinite(value).all()),
                                     "min": float(value.min()), "max": float(value.max()),
                                     "median": float(value.median())}
    attrs = {}
    for key, value in raw.items():
        if torch.is_tensor(value):
            attrs[key] = {"shape": list(value.shape), "finite": bool(torch.isfinite(value).all()),
                          "min": float(value.min()), "max": float(value.max()),
                          "mean": float(value.float().mean())}
    C.dump_json(out / "pre_backward_diagnostics.json", {
        "render": {key: {"shape": list(value.shape), "finite": bool(torch.isfinite(value).all()),
                         "min": float(value.min()), "max": float(value.max()), "mean": float(value.float().mean())}
                   for key, value in rendered.items() if torch.is_tensor(value)},
        "fit": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in fit.items()},
        "source_camera": C.move_tree(cache["views"], "cpu")["camera_poses"].tolist(),
        "export_camera": preds["export_camera_poses"].tolist(),
        "target_camera": target.tolist(), "target_K": Ks[:1].tolist(),
        "geometry": geometry_summary, "raw_attributes": attrs,
        "target_projection": projected_summary})
    target_person = torch.ones((1, args.res, args.res), dtype=torch.bool, device=device)
    target_depth = torch.ones_like(target_person, dtype=torch.float32)
    targets = {"gt": rendered["rgb"].detach(), "K": Ks[:1], "person": target_person,
               "valid_depth": target_person, "depth_z_m": target_depth,
               "c2w": target}
    losses = C.masked_losses(rendered, targets, fit)
    losses["loss"].backward()
    grads = {n: None if p.grad is None else float(p.grad.abs().sum()) for n, p in trainable}
    partial = {"losses": {k: float(v) if torch.is_tensor(v) else v for k, v in losses.items()},
               "render_rgb": [float(rendered["rgb"].min()), float(rendered["rgb"].max())],
               "alpha": [float(rendered["alpha"].min()), float(rendered["alpha"].max())],
               "depth": [float(rendered["depth"].min()), float(rendered["depth"].max())],
               "fit": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in fit.items()},
               "target_camera": target.tolist(), "grads": grads}
    C.dump_json(out / "gradient_diagnostics.json", partial)
    if not C.finite_gradients([p for _, p in trainable]) or not any(g and g > 0 for g in grads.values()):
        raise RuntimeError("Non-finite/zero target gradient")
    for n, param in model.named_parameters():
        if n.startswith("visual_geometry_transformer") and param.grad is not None:
            raise RuntimeError("Source encoder gradient leaked")
    if any(p.grad is not None for p in teacher.parameters()):
        raise RuntimeError("Geometry teacher gradient leaked")
    report = {"schema": "sa_portrait_training_smoke_v1", "status": "passed",
              "optimizer_steps": 0, "architecture_modified": False,
              "source_ids": cache["source_ids"], "target_ids": [records[3]["cid"]],
              "camera_fit": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in fit.items()},
              "target_roundtrip_max_abs": float(np.max(np.abs(rt - target_relative.numpy()))),
              "source_actual_K_vs_reference_max_abs_square": k_delta,
              "actual_returned_K": actual_K.tolist(),
              "loss": {k: float(v) if torch.is_tensor(v) else v for k, v in losses.items()},
              "depth_forward_check": depth_forward_check,
              "target_gradient_abs_sum": grads,
              "teacher_frozen": True, "source_encoder_frozen": True,
              "rgb_shape": list(rendered["rgb"].shape),
              "n_merged": int(C.splat_list(preds)["means"].shape[0]),
              "real_data_used": False}
    C.dump_json(out / "smoke.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print("SMOKE_PASS", out, flush=True)


if __name__ == "__main__":
    main()
