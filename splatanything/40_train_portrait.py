#!/usr/bin/env python3
"""Strict source/target-separated, original-SA student-head short training.

Only the released Gaussian heads are optimized. Geometry teacher and source
encoder are frozen and cached per scene. Each step has three known-camera
sources and two distinct side-boundary train_target views. Held-out eval images
never enter the model, loss, cache or checkpoint selection.

Example (on an allocated GPU, not on an H100 login node):
  python 40_train_portrait.py --manifest manifest.json --out RUN \
      --steps 100 --max-train-sequences 2 --checkpoint-at 1,100
"""
import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from importlib.util import module_from_spec, spec_from_file_location
_spec = spec_from_file_location("sa_portrait_common", Path(__file__).with_name("39_portrait_common.py"))
C = module_from_spec(_spec)
_spec.loader.exec_module(C)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--official-repo", default=os.environ.get("SPLATANYTHING_DIR", ""))
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--res", type=int, default=504)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--depth-weight", type=float, default=.1)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--batch-sequences", type=int, default=1)
    p.add_argument("--accumulate", type=int, default=1)
    p.add_argument("--checkpoint-at", default="250,500,1000,2000")
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--max-train-sequences", type=int, default=0)
    p.add_argument("--sequences", nargs="*", default=())
    p.add_argument("--resume", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--review", default="",
                   help="Review JSON: every audited view of each selected sequence must be accepted before optimization")
    p.add_argument("--gate-only", action="store_true",
                   help="Real-data B0 forward/backward only; no optimizer updates or training checkpoint")
    p.add_argument("--max-hours", type=float, default=0)
    return p.parse_args(argv)


def camera_and_gradient_gate(model, teacher, cache, targets, device, trainable, depth_weight):
    """Before training: verify a target-only camera transform and real gradients."""
    stash = {}
    preds = C.predict(model, cache, device, stash)
    print("GATE_FORWARD_GAUSSIANS", int(C.splat_list(preds)["means"].shape[0]), flush=True)
    fit = C.fit_source_sim3(cache["views"], preds["export_camera_poses"][0])
    cameras = C.target_camera_batch(targets, fit, device, cache["views"])
    rd = C.render_splats(preds, cameras, targets["K"].to(device),
                         targets["gt"].shape[1:3])
    print("GATE_RENDERED", flush=True)
    loss = C.masked_losses(rd, targets, fit, depth_weight)
    loss["loss"].backward()
    print("GATE_BACKWARD", flush=True)
    grads = {n: (None if p.grad is None else float(p.grad.detach().abs().sum()))
             for n, p in trainable}
    if not C.finite_gradients([p for _, p in trainable]) or not any(v and v > 0 for v in grads.values()):
        raise RuntimeError("Invalid target rendering gradient: " + str(grads))
    for p in model.visual_geometry_transformer.parameters():
        if p.requires_grad or p.grad is not None:
            raise RuntimeError("Source encoder is not frozen")
    if any(p.requires_grad or p.grad is not None for p in teacher.parameters()):
        raise RuntimeError("External geometry teacher is not frozen")
    return {"fit": fit, "cameras": cameras, "rendered": rd, "preds": preds,
            "loss": {k: (float(v.detach()) if torch.is_tensor(v) else v) for k, v in loss.items()},
            "trainable_grad_abs_sum": grads,
            "trainable_parameter_count": int(sum(p.numel() for _, p in trainable)),
            "source_ids": cache["source_ids"]}


def main(argv=None):
    args = parse_args(argv)
    if not args.official_repo:
        raise SystemExit("Set --official-repo or SPLATANYTHING_DIR")
    if args.steps < 1 or args.steps > 2000 or args.res % 14:
        raise SystemExit("This short protocol requires 1..2000 steps and res divisible by 14")
    if args.batch_sequences != 1 or args.accumulate != 1:
        raise SystemExit("Frozen pilot uses one scene per optimizer step; accumulation is not enabled")
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = C.load_manifest(manifest_path)
    rows = C.select_sequences(manifest, "train", args.max_train_sequences, args.sequences)
    if any(s["split"] != "train" for s in rows):
        raise SystemExit("Only training-eligible identities can be selected for optimization")
    if args.gate_only:
        if len(args.sequences) != 1 or args.resume:
            raise SystemExit("--gate-only requires exactly one --sequences ID and no --resume")
        print("GATE_ONLY: zero optimizer updates; review is not checked", flush=True)
    else:
        if not args.review:
            raise SystemExit("Provide --review from 38; do not bypass per-view exposure QA")
        review = C.read_json(args.review)
        if review.get("schema") != "ue53_portrait_view_review_v1":
            raise SystemExit("Wrong per-view review schema")
        for s in rows:
            for cid, v in s["views"].items():
                entry = review.get("views", {}).get(s["sequence_id"] + "/" + cid, {})
                if entry.get("status") != "accepted":
                    raise SystemExit("Review does not accept %s/%s; revise the manifest/protocol instead of bypassing" %
                                     (s["sequence_id"], cid))
                if v.get("exposure_qa") is None:
                    raise SystemExit("Exposure QA absent; run 38 --audit-views first")
    out = Path(args.out).expanduser().resolve()
    if args.resume:
        out.mkdir(parents=True, exist_ok=True)
    else:
        out.mkdir(parents=True, exist_ok=False)
    if not args.resume and any(out.iterdir()):
        raise SystemExit("New output directory must be empty")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    repo = Path(args.official_repo).expanduser().resolve()
    _, model, teacher = C.build_models(repo, device)
    print("MODELS_READY", flush=True)
    trainable = C.set_trainable(model)
    if args.gate_only:
        before = {n: p.detach().clone() for n, p in trainable}
        row = rows[0]
        source_records = [C.resolve_view(manifest, row, cid) for cid in manifest["source_ids"]]
        for r in source_records:
            r["resolution_wh"] = (args.res, args.res)
        t0 = time.perf_counter()
        cache = C.source_cache(model, teacher, source_records, device)
        print("SOURCE_CACHE_READY", flush=True)
        targets, target_ids = C.batch_targets(
            [C.resolve_view(manifest, row, cid) for cid in manifest["train_target_ids"]], args.res, device)
        print("TARGETS_READY", flush=True)
        gate = camera_and_gradient_gate(model, teacher, cache, targets, device, trainable, args.depth_weight)
        rd, person = gate["rendered"], targets["person"]
        valid = targets["valid_depth"] & (rd["alpha"] > .1)
        delta = rd["depth_ed_for_metric"] - rd["depth"]
        covered = rd["alpha"] > .1
        cameras = gate["cameras"]
        sp = C.splat_list(gate["preds"])
        means_h = torch.cat([sp["means"], torch.ones_like(sp["means"][:, :1])], dim=-1)
        projected = (torch.linalg.inv(cameras) @ means_h.T).transpose(1, 2)
        z = projected[..., 2]
        uv = projected[..., :2] / z.clamp_min(1e-8)[..., None] * targets["K"][:, None, :2, :2].diagonal(dim1=-2, dim2=-1) + targets["K"][:, None, :2, 2]
        in_frame = ((z > .01) & (z < 400.) & (uv[..., 0] >= 0) & (uv[..., 0] < args.res) & (uv[..., 1] >= 0) & (uv[..., 1] < args.res))
        camera_diagnostics = {
            "target_relative_cameras": [
                C.package_to_source_camera(c, cache["views"]).numpy().tolist()
                for c in targets["c2w"].detach().cpu()],
            "target_model_cameras": cameras.detach().cpu().tolist(),
            "source_relative_cameras": cache["views"]["camera_poses"].tolist(),
            "exported_source_cameras": gate["preds"]["export_camera_poses"].detach().cpu().tolist(),
            "merged_mean_centroid": sp["means"].mean(dim=0).detach().cpu().tolist(),
            "per_target": [{"target_id": target_ids[i],
                            "z_min": float(z[i].min()), "z_max": float(z[i].max()),
                            "in_frustum": int(in_frame[i].sum()),
                            "alpha_max": float(rd["alpha"][i].max()),
                            "person_coverage01": float((rd["alpha"][i][person[i]] > .1).float().mean())}
                           for i in range(len(target_ids))]}
        C.dump_json(out / "camera_diagnostics.json", camera_diagnostics)
        if not covered.any():
            raise RuntimeError("Real target has no rendered support; see camera_diagnostics.json")
        if not bool(torch.isfinite(rd["rgb"]).all()) or float(delta[covered].abs().max()) > 1e-5:
            raise RuntimeError("Real target render or auxiliary/native ED forward check failed")
        changed = {n: float((p.detach() - before[n]).abs().max()) for n, p in trainable}
        if any(v != 0. for v in changed.values()):
            raise RuntimeError("Gate-only changed an official model parameter")
        report = {"schema": "sa_portrait_real_b0_gate_v1", "status": "passed",
                  "sequence_id": row["sequence_id"], "target_ids": target_ids,
                  "optimizer_steps": 0, "review_checked": False,
                  "gate_only_is_not_training_or_review_approval": True,
                  "source_ids": cache["source_ids"], "fit": {
                      k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in gate["fit"].items()},
                  "loss": gate["loss"], "gradients": gate["trainable_grad_abs_sum"],
                  "parameter_max_delta": changed, "n_merged": int(C.splat_list(gate["preds"])["means"].shape[0]),
                  "render": {"alpha_max": float(rd["alpha"].max()),
                             "person_alpha_mean": float(rd["alpha"][person].mean()),
                             "person_coverage05": float((rd["alpha"][person] > .5).float().mean()),
                             "valid_depth_coverage": float(valid.sum() / targets["valid_depth"].sum()),
                             "native_vs_stable_ED_max_abs_covered": float(delta[covered].abs().max())},
                  "peak_alloc_GB": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.,
                  "wall_s": time.perf_counter() - t0}
        C.dump_json(out / "gate.json", report)
        for i, cid in enumerate(target_ids):
            Image.fromarray(np.uint8(np.clip(targets["gt"][i].cpu().numpy(), 0, 1) * 255)).save(out / (cid + "_gt.png"))
            Image.fromarray(np.uint8(np.clip(rd["rgb"][i].detach().cpu().numpy(), 0, 1) * 255)).save(out / (cid + "_B0_rgb.png"))
            Image.fromarray(np.uint8(np.clip(rd["alpha"][i].detach().cpu().numpy(), 0, 1) * 255)).save(out / (cid + "_B0_alpha.png"))
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        print("GATE_ONLY_PASS", out, flush=True)
        return
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=args.lr,
                                  betas=(.9, .95), weight_decay=1e-6)
    checkpoints = sorted(set(int(s) for s in args.checkpoint_at.split(",") if s.strip()))
    if any(s < 1 or s > args.steps for s in checkpoints):
        raise SystemExit("Checkpoint steps must be within the requested run")
    step0 = 0
    if args.resume:
        blob = C.load_trainable(model, args.resume)
        optimizer.load_state_dict(blob["optimizer"])
        step0 = int(blob["step"])
        if step0 >= args.steps:
            raise SystemExit("Resume step is already at or past the requested end")
        print("RESUME", args.resume, "step", step0, flush=True)
    else:
        C.save_checkpoint(out / "B0.pt", 0, trainable, optimizer, args, manifest_path,
                          {"baseline": "Official released weights before any portrait update"})
    records = {"schema": "sa_portrait_short_train_v1", "args": vars(args),
               "manifest": str(manifest_path),
               "selected_sequences": [s["sequence_id"] for s in rows],
               "source_ids": manifest["source_ids"], "target_ids": manifest["train_target_ids"],
               "official_commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                                          text=True).strip(),
               "teacher_frozen": True, "source_encoder_frozen": True,
               "no_eval_in_training": True, "checkpoint_steps": checkpoints,
               "status": "initializing", "started": time.time()}
    C.dump_json(out / "run.json", records)
    metrics_path = out / "metrics.jsonl"
    csv_path = out / "metrics.csv"
    new_file = not metrics_path.exists()
    jl = open(metrics_path, "a", encoding="utf-8", newline="\n")
    cf = open(csv_path, "a", encoding="utf-8", newline="")
    fields = ("step", "wall_s", "step_s", "gate_s", "sequence_id", "target_ids", "loss",
              "rgb_full", "rgb_person", "alpha_loss", "depth_loss_m2",
              "depth_valid_n", "grad_norm", "n_merged", "alpha_person_mean",
              "alpha_coverage05_person", "depth_mae_mm_valid", "source_psnr_mean",
              "peak_alloc_GB", "peak_reserved_GB", "event")
    writer = csv.DictWriter(cf, fieldnames=fields)
    if new_file:
        writer.writeheader()
    cache = None
    t_start = time.perf_counter()
    step = step0
    try:
        while step < args.steps:
            if args.max_hours and time.perf_counter() - t_start > args.max_hours * 3600:
                print("TIME_LIMIT", flush=True)
                break
            row = rows[step % len(rows)]
            t_step = time.perf_counter()
            source_records = [C.resolve_view(manifest, row, cid) for cid in manifest["source_ids"]]
            for r in source_records:
                r["resolution_wh"] = (args.res, args.res)
            # Drop the previous cache and CUDA intermediates before rebuilding
            # a new scene, avoiding simultaneous teacher/source activations.
            cache = None
            torch.cuda.empty_cache() if device.type == "cuda" else None
            cache = C.source_cache(model, teacher, source_records, device)
            target_records = [C.resolve_view(manifest, row, cid) for cid in manifest["train_target_ids"]]
            targets, target_ids = C.batch_targets(target_records, args.res, device)
            optimizer.zero_grad(set_to_none=True)
            t_gate = time.perf_counter()
            gate = camera_and_gradient_gate(model, teacher, cache, targets, device, trainable,
                                            args.depth_weight)
            gate_s = time.perf_counter() - t_gate
            loss = gate["loss"]
            if not np.isfinite(loss["loss"]):
                raise FloatingPointError("Non-finite loss before optimizer: " + str(loss))
            norm = float(torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.))
            if not np.isfinite(norm):
                raise FloatingPointError("Non-finite gradient norm before optimizer")
            optimizer.step()
            step += 1
            rd = gate["rendered"]
            person = targets["person"].to(device)
            valid = targets["valid_depth"].to(device) & (rd["alpha"] > .1)
            depth_mae = (torch.abs(rd["depth"] / gate["fit"]["s"] -
                                   targets["depth_z_m"].to(device))[valid].mean() * 1000.
                         if valid.any() else torch.tensor(float("nan"), device=device))
            alpha_mean = float(rd["alpha"][person].mean())
            coverage = float((rd["alpha"][person] > .5).float().mean())
            # Source replay is diagnostic only; no claim of held-out quality.
            with torch.no_grad():
                sr = C.render_splats(gate["preds"], gate["preds"]["export_camera_poses"][0],
                                     gate["preds"]["export_camera_intrs"][0],
                                     (args.res, args.res))
                src_gt = C.move_tree(cache["views"], device)["img"][0].permute(0, 2, 3, 1)
                source_psnr = float(-10 * torch.log10(F.mse_loss(sr["rgb"], src_gt).clamp_min(1e-12)))
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t_step
            n_merged = int(C.splat_list(gate["preds"])["means"].shape[0])
            rec = {"step": step, "wall_s": time.perf_counter() - t_start, "step_s": dt,
                   "gate_s": gate_s, "sequence_id": row["sequence_id"],
                   "target_ids": target_ids, **loss, "depth_loss_m2": loss["depth_z_m2"],
                   "grad_norm": norm, "n_merged": n_merged,
                   "alpha_person_mean": alpha_mean, "alpha_coverage05_person": coverage,
                   "depth_mae_mm_valid": float(depth_mae), "source_psnr_mean": source_psnr,
                   "peak_alloc_GB": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.,
                   "peak_reserved_GB": torch.cuda.max_memory_reserved() / 2**30 if device.type == "cuda" else 0.,
                   "event": "step"}
            rec.pop("alpha", None)
            rec["alpha_loss"] = loss["alpha"]
            jl.write(json.dumps(rec, ensure_ascii=False) + "\n")
            jl.flush()
            writer.writerow({k: rec.get(k) for k in fields})
            cf.flush()
            print("STEP %d/%d %s loss %.7f person_alpha %.3f depth_valid %d %.3fs" %
                  (step, args.steps, row["sequence_id"], loss["loss"], alpha_mean,
                   loss["depth_valid_n"], dt), flush=True)
            if step == 1:
                records["first_step_gate"] = {"fit": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                                       for k, v in gate["fit"].items()},
                                              "trainable_grad_abs_sum": gate["trainable_grad_abs_sum"],
                                              "trainable_parameter_count": gate["trainable_parameter_count"],
                                              "source_ids": gate["source_ids"], "loss": loss}
                C.dump_json(out / "run.json", records)
            if step in checkpoints or step == args.steps:
                C.save_checkpoint(out / ("step_%04d.pt" % step), step, trainable, optimizer,
                                  args, manifest_path,
                                  {"selected_sequences": [s["sequence_id"] for s in rows],
                                   "last_metrics": rec})
                print("CHECKPOINT", step, flush=True)
            records["last_step"] = step
            records["last_step_s"] = dt
            records["status"] = "running"
            C.dump_json(out / "run.json", records)
            del gate, targets, rd
        records["status"] = "completed" if step == args.steps else "time_limited"
        records["completed_steps"] = step
        records["total_wall_s"] = time.perf_counter() - t_start
        records["measured_mean_step_s"] = (time.perf_counter() - t_start) / max(step - step0, 1)
        records["eta_2000_only_steps_hours"] = records["measured_mean_step_s"] * 2000 / 3600
        C.dump_json(out / "run.json", records)
    except BaseException as e:
        records["status"] = "failed"
        records["error"] = type(e).__name__ + ": " + str(e)
        records["last_step"] = step
        C.dump_json(out / "run.json", records)
        raise
    finally:
        jl.close()
        cf.close()
    print("DONE", out, "steps", step, "seconds", records["total_wall_s"], flush=True)


if __name__ == "__main__":
    main()
