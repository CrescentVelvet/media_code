#!/usr/bin/env python3
"""Evaluate B0 and multiple student checkpoints on strict held-out targets.

UE reports RGB/lowpass/person alpha and axial-Z depth. Nose/face are not inferred
from the person mask; full-frame/center-person crops are for visual review only.
Existing real packages use their frozen protocol and are development samples,
not new blind tests or metric-geometry evidence. No target fits Sim(3).
"""
import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from importlib.util import module_from_spec, spec_from_file_location
_spec = spec_from_file_location("sa_portrait_common", Path(__file__).with_name("39_portrait_common.py"))
C = module_from_spec(_spec)
_spec.loader.exec_module(C)
M21 = C.load_local("portrait_preprocess21", "21_geometry_audit_v2.py")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest")
    p.add_argument("--rundir", required=True)
    p.add_argument("--checkpoints", nargs="*", default=())
    p.add_argument("--split", choices=("val", "test", "hard_example_only"), default="test")
    p.add_argument("--sequences", nargs="*", default=())
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--res", type=int, default=504)
    p.add_argument("--official-repo", default=os.environ.get("SPLATANYTHING_DIR", ""))
    p.add_argument("--device", default="cuda")
    p.add_argument("--save-ply", action="store_true")
    p.add_argument("--real-protocol", default="")
    p.add_argument("--real-report", default="")
    p.add_argument("--real-res", type=int, default=756)
    return p.parse_args(argv)


def metrics(gt, rd, alpha, person=None):
    if person is None:
        person = np.ones(gt.shape[:2], bool)
    if int(person.sum()) < 64:
        return {"status": "N/A", "n": int(person.sum())}
    d = gt[person].astype(np.float64) - rd[person]
    lp_gt = cv2.GaussianBlur(gt, (0, 0), 2.)[person]
    lp_rd = cv2.GaussianBlur(rd, (0, 0), 2.)[person]
    mse = float(np.mean(d * d))
    return {"n": int(person.sum()), "psnr": -10 * np.log10(max(mse, 1e-12)),
            "mae": float(np.mean(np.abs(d))),
            "lowpass_sigma2_psnr": -10 * np.log10(max(float(np.mean((lp_gt - lp_rd)**2)), 1e-12)),
            "alpha_mean": float(alpha[person].mean()),
            "alpha_coverage05": float((alpha[person] > .5).mean())}


def depth_metrics(rd, target, fit, person, source=False):
    if source:
        return {"status": "source-only; no metric depth fitted or claimed"}
    valid = target["valid_depth"].detach().cpu().numpy() & person & (rd["alpha"] > .1).detach().cpu().numpy()
    if int(valid.sum()) < 64:
        return {"status": "N/A", "n": int(valid.sum())}
    error = (rd["depth"] / fit["s"] - target["depth_z_m"].to(rd["depth"].device)
             ).detach().cpu().numpy()
    e = error[valid]
    return {"n": int(valid.sum()), "mae_mm": float(np.mean(np.abs(e)) * 1000),
            "p90_abs_mm": float(np.percentile(np.abs(e), 90) * 1000),
            "bias_mm": float(e.mean() * 1000),
            "valid_depth_coverage": float(valid.sum() / max(person.sum(), 1))}


def image(x):
    return np.uint8(np.clip(x, 0, 1) * 255)


def comparison(path, gt, renders, person, extra=()):
    # Entire 504/756 frame plus a center-person crop, not an invented face ROI.
    names = ["GT"] + [r[0] for r in renders]
    tiles, rows = [], []
    for j, name in enumerate(names):
        arr = gt if j == 0 else renders[j - 1][1]
        tiles.append(Image.fromarray(image(arr)))
    for crop in (False, True):
        row = []
        for name, tile in zip(names, tiles):
            if crop:
                ys, xs = np.where(person)
                if len(xs):
                    x0, x1 = xs.min(), xs.max() + 1
                    y0, y1 = ys.min(), ys.max() + 1
                    pad = int(.08 * max(x1 - x0, y1 - y0))
                    tile = tile.crop((max(0, x0 - pad), max(0, y0 - pad),
                                      min(tile.width, x1 + pad), min(tile.height, y1 + pad)))
            tile = tile.resize((256, 256), Image.Resampling.LANCZOS)
            d = ImageDraw.Draw(tile)
            d.text((5, 5), name + (" / person bbox" if crop else ""), fill="white",
                   stroke_width=2, stroke_fill="black")
            row.append(np.asarray(tile))
        rows.append(np.concatenate(row, axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)
    if extra:
        rows = []
        for title, rd, alpha, ed in extra:
            tile = []
            for name, arr in (("RGB", image(rd)), ("lowpass", image(cv2.GaussianBlur(rd, (0, 0), 2.))),
                              ("alpha", np.repeat(image(alpha)[..., None], 3, 2)),
                              ("person ED m", image(ed))):
                t = Image.fromarray(arr).resize((256, 256))
                ImageDraw.Draw(t).text((5, 5), title + " " + name, fill="white",
                                       stroke_width=2, stroke_fill="black")
                tile.append(np.asarray(t))
            rows.append(np.concatenate(tile, axis=1))
        Image.fromarray(np.concatenate(rows, axis=0)).save(str(path).replace(".png", "_diagnostics.png"))


def model_source_fit(model, cache, device):
    preds = C.predict(model, cache, device)
    fit = C.fit_source_sim3(cache["views"], preds["export_camera_poses"][0])
    return preds, fit


def save_model_ply(preds, fit, path, source_ids):
    from src.utils.scene_io import export_gaussians
    path.mkdir(parents=True, exist_ok=True)
    # The evaluation PLY is the complete merged cloud. It does not slice a
    # fixed 3*H*W point order, and applies no extra opacity/quantile pruning.
    export_gaussians(preds, path, opacity_thresh=0., scale_quantile=None)
    C.dump_json(path / "model_to_package.json", {
        "schema": "sa_portrait_source_sim3_v1", "fit": {k: v.tolist() if isinstance(v, np.ndarray) else v
                                                         for k, v in fit.items()},
        "formula": "X_model=s R X_package+T; R_model=R R_package; C_model=s R C_package+T",
        "source_ids": source_ids,
        "inverse_camera": "R_package=R.T R_model; C_package=R.T(C_model-T)/s"})


def real_records(protocol, report):
    rows = []
    for pack in protocol["packages"]:
        src = json.loads((Path(pack["source_scene"]) / "prior_camera.json").read_text(encoding="utf-8"))
        priors = {c["camera_id"]: np.asarray(c["matrix"], np.float32)
                  for c in src["extrinsics"]}
        pack_targets = json.loads((Path(pack["target_scene"]) / "prior_camera.json").read_text(encoding="utf-8"))
        priors.update({c["camera_id"]: np.asarray(c["matrix"], np.float32)
                       for c in pack_targets["extrinsics"]})
        sources = []
        for f in pack["frames"]:
            if f["split"] != "source":
                continue
            sources.append({"cid": f["camera_id"], "rgb_path": Path(f["processed_path"]),
                            "K_np": np.asarray(next(c["matrix"] for c in src["intrinsics"]
                                                    if c["camera_id"] == f["camera_id"]), np.float32),
                            "c2w_np": priors[f["camera_id"]]})
        targets = []
        ref = report["packages"][pack["scene"]]["target_reference"]["auxiliary"]
        for f in pack["frames"]:
            if f["split"] != "target":
                continue
            stem = f["camera_id"]
            crop = ref[stem]["crop"]
            gt = M21.official_preprocess_rgb(f["processed_path"], crop)
            targets.append({"cid": stem, "gt": torch.from_numpy(gt).float(),
                            "K": torch.from_numpy(np.asarray(ref[stem]["target_K"], np.float32)),
                            "c2w_np": priors[stem], "roi_status": ref[stem]["roi_status"]})
        rows.append({"sequence_id": pack["scene"], "sources": sources, "targets": targets})
    return rows


def main(argv=None):
    args = parse_args(argv)
    if not args.official_repo:
        raise SystemExit("Set --official-repo or SPLATANYTHING_DIR")
    if bool(args.real_protocol) != bool(args.real_report):
        raise SystemExit("Real evaluation requires both --real-protocol and --real-report")
    if args.real_protocol and args.manifest:
        raise SystemExit("Use separate UE and real evaluation output directories")
    if not args.real_protocol and not args.manifest:
        raise SystemExit("UE evaluation requires --manifest")
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    _, model, teacher = C.build_models(Path(args.official_repo), device)
    paths = C.checkpoint_names(args.rundir, args.checkpoints)
    real = bool(args.real_protocol)
    manifest = C.load_manifest(args.manifest) if not real else None
    rows = real_records(C.read_json(args.real_protocol), C.read_json(args.real_report)) if real else \
        C.select_sequences(manifest, args.split, args.limit, args.sequences)
    res = args.real_res if real else args.res
    all_metrics, timing, summary = [], [], {}
    t_all = time.perf_counter()
    for row in rows:
        seq_id = row["sequence_id"]
        seq_out = out / seq_id
        seq_out.mkdir()
        source_records = row["sources"] if real else [
            C.resolve_view(manifest, row, cid) for cid in manifest["source_ids"]]
        for rec in source_records:
            rec["resolution_wh"] = (res, res)
        t0 = time.perf_counter()
        cache = C.source_cache(model, teacher, source_records, device)
        cache_s = time.perf_counter() - t0
        if real:
            target_rows = row["targets"]
            target_ids = [r["cid"] for r in target_rows]
            person = [np.ones(tuple(r["gt"].shape[:2]), bool) for r in target_rows]
        else:
            eval_ids = manifest["eval_near_ids"]
            target_records = [C.resolve_view(manifest, row, cid) for cid in eval_ids]
            target_rows, target_ids = C.batch_targets(target_records, res, device)
            target_rows = [{k: v[i] for k, v in target_rows.items()} for i in range(len(target_ids))]
            person = [r["person"].detach().cpu().numpy() for r in target_rows]
            if any(r["exposure_qa"] is None or r["exposure_qa"]["exposure_flag"] for r in target_records):
                print("EXPOSURE_QA", seq_id, "eval contains flagged/unreviewed view; report remains descriptive", flush=True)
        for r in target_rows:
            if "c2w" not in r:
                r["c2w"] = r["c2w_np"]
            elif torch.is_tensor(r["c2w"]):
                r["c2w"] = r["c2w"].detach().cpu().numpy()
        for ckpt in paths:
            name = ckpt.stem
            t0 = time.perf_counter()
            blob = C.load_trainable(model, ckpt)
            with torch.no_grad():
                preds, fit = model_source_fit(model, cache, device)
                cameras = torch.stack([C.target_to_model(C.package_to_source_camera(
                    r["c2w"], cache["views"]), fit, device)
                                       for r in target_rows])
                Ks = torch.stack([r["K"] for r in target_rows]).to(device)
                rendered = C.render_splats(preds, cameras, Ks, (res, res))
            summary.setdefault(name, [])
            extras = []
            for i, (target, cid) in enumerate(zip(target_rows, target_ids)):
                rd = {k: (v[i] if torch.is_tensor(v) else v) for k, v in rendered.items()}
                gt = target["gt"].numpy()
                rgb = rd["rgb"].detach().cpu().numpy()
                alpha = rd["alpha"].detach().cpu().numpy()
                rec = {"sequence_id": seq_id, "identity": "real_development" if real else row["character_id"],
                       "split": "real_development" if real else row["split"],
                       "checkpoint": name, "step": int(blob.get("step", 0)), "target": cid,
                       "role": "real_frozen_target" if real else target.get("role", "eval"),
                       "view_group": "real" if real else target.get("view_group", "near_input"),
                       "full": metrics(gt, rgb, alpha),
                       "person": metrics(gt, rgb, alpha, person[i]),
                       "depth": depth_metrics(rd, target, fit, person[i], source=real),
                       "roi_status": target.get("roi_status", "person_mask_not_face_roi")}
                all_metrics.append(rec)
                summary[name].append(rec)
                stem = seq_out / (cid + "_" + name)
                Image.fromarray(image(rgb)).save(str(stem) + "_rgb.png")
                Image.fromarray(image(alpha)).save(str(stem) + "_alpha.png")
                if real:
                    ed = np.zeros_like(alpha)
                else:
                    ed = np.where(person[i] & (rd["alpha"].detach().cpu().numpy() > .1),
                                  rd["depth"].detach().cpu().numpy() / fit["s"], np.nan)
                    finite = ed[np.isfinite(ed)]
                    vmax = float(np.percentile(finite, 95)) if len(finite) else 1.
                    ed = np.clip(ed / max(vmax, 1e-8), 0, 1)
                    ed = np.nan_to_num(ed)
                    Image.fromarray(image(ed)).save(str(stem) + "_person_ED.png")
                extras.append((cid, rgb, alpha, ed))
                comparison(Path(str(stem) + "_comparison.png"), gt,
                           [(name, rgb)], person[i])
            # One per-target sheet includes all diagnostic modalities; the B0
            # vs checkpoints sheet is generated below after every checkpoint.
            for i, cid in enumerate(target_ids):
                comparison(seq_out / (cid + "_" + name + "_diagnostic.png"),
                           target_rows[i]["gt"].numpy(),
                           [(name, rendered["rgb"][i].detach().cpu().numpy())], person[i],
                           [(name, rendered["rgb"][i].detach().cpu().numpy(),
                             rendered["alpha"][i].detach().cpu().numpy(), extras[i][3])])
            if args.save_ply:
                save_model_ply(preds, fit, seq_out / (name + "_ply"),
                               cache["source_ids"])
            timing.append({"sequence_id": seq_id, "checkpoint": name,
                           "cache_s_once": cache_s, "checkpoint_eval_s": time.perf_counter() - t0})
            print("EVAL", seq_id, name, "seconds", timing[-1]["checkpoint_eval_s"], flush=True)
            del preds, rendered
        # Compact cross-checkpoint RGB sheet. Re-read saved renders so this
        # does not retain one merged Gaussian cloud per checkpoint on the GPU.
        for i, cid in enumerate(target_ids):
            renders = [(p.stem, np.asarray(Image.open(seq_out / (cid + "_" + p.stem + "_rgb.png")),
                                           np.float32) / 255.) for p in paths]
            comparison(seq_out / (cid + "_B0_vs_checkpoints.png"), target_rows[i]["gt"].numpy(),
                       renders, person[i])
        del cache
        if device.type == "cuda":
            torch.cuda.empty_cache()
    flat = []
    for rec in all_metrics:
        flat.append({"sequence_id": rec["sequence_id"], "identity": rec["identity"],
                     "split": rec["split"], "checkpoint": rec["checkpoint"], "step": rec["step"],
                     "target": rec["target"], "role": rec["role"], "view_group": rec["view_group"],
                     "full_psnr": rec["full"].get("psnr"), "full_lowpass_psnr": rec["full"].get("lowpass_sigma2_psnr"),
                     "full_alpha_mean": rec["full"].get("alpha_mean"),
                     "person_psnr": rec["person"].get("psnr"),
                     "person_lowpass_psnr": rec["person"].get("lowpass_sigma2_psnr"),
                     "person_alpha_mean": rec["person"].get("alpha_mean"),
                     "person_coverage05": rec["person"].get("alpha_coverage05"),
                     "depth_mae_mm": rec["depth"].get("mae_mm"),
                     "depth_p90_mm": rec["depth"].get("p90_abs_mm"),
                     "depth_valid_coverage": rec["depth"].get("valid_depth_coverage")})
    with open(out / "per_view_metrics.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0]))
        w.writeheader()
        w.writerows(flat)
    aggregates = {}
    for name, recs in summary.items():
        def mean(key):
            vals = [r[key] for r in flat if r["checkpoint"] == name and r[key] is not None and np.isfinite(r[key])]
            return float(np.mean(vals)) if vals else None
        aggregates[name] = {key: mean(key) for key in flat[0] if key.endswith(("_psnr", "_mm", "_mean", "_coverage", "_coverage05"))}
    report = {"schema": "sa_portrait_checkpoint_eval_v1", "split": args.split,
              "is_real_development": real, "geometry_truth_available": not real,
              "person_is_not_face_or_nose_roi": True, "checkpoints": [str(p) for p in paths],
              "aggregate_unweighted_per_view": aggregates, "timing": timing,
              "total_wall_s": time.perf_counter() - t_all,
              "target_fitting_or_checkpoint_selection": False,
        "real_limitations": "Existing frozen real targets; no independent metric geometry, no blind-test claim" if real else None,
              "note": "Raw per-view CSV is authoritative; mean PSNR is descriptive, not an automatic pass gate."}
    C.dump_json(out / "checkpoint_eval.json", report)
    C.dump_json(out / "summary.json", {"checkpoints": aggregates, "total_wall_s": report["total_wall_s"]})
    print(json.dumps(aggregates, ensure_ascii=False, indent=2), flush=True)
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
