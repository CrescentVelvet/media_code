#!/usr/bin/env python3
"""UE53 portrait fine-tune: read-only audit and identity-isolated manifest.

This script does not copy or modify the dataset. It validates the standardized
UE packages, records c6 failures and exposure diagnostics, then freezes a
16/2/2 identity split from the 20 c6-pass identities. Seven c6-fail identities
are hard-example-only. All paths are written relative to the dataset root for
portability between the local machine and H100.

Use --audit-views N to inspect N additional main-camera source/eval views per
package. Exposure is a per-view QA flag, not an automatic deletion or face ROI.
Use --require-review review.json before training; the review schema is emitted
by --write-review-template. c6 and source/target RGB/depth/mask checks are hard
gates; c6 failure does not imply missing files.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

SCHEMA = "ue53_portrait_finetune_manifest_v1"
REVIEW_SCHEMA = "ue53_portrait_view_review_v1"
SOURCE_IDS = ("s03", "s06", "s09")
TRAIN_TARGET_IDS = tuple("e%02d_p00" % i for i in (6, 19))
EVAL_NEAR_IDS = tuple("e%02d_p00" % i for i in (2, 4, 21, 23))
EVAL_REAR_IDS = tuple("e%02d_p00" % i for i in (8, 11, 14, 17))
SPLIT_IDS = {
    "train": ("ada", "bernice", "bryan", "chandra", "cooper", "danielle",
              "dhruv", "dax", "emanuel", "emory", "erno", "ettore", "farrukh",
              "gavin", "hadley", "irene"),
    "val": ("jesse", "kai"),
    "test": ("keiji", "kellan"),
}
EXPECTED_C6_FAIL = {"amelia", "aoi", "bes", "fei", "glenda", "hana", "hudson"}


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dump_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def sha256(path, limit_bytes=64 * 1024 * 1024):
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(4 * 1024 * 1024)
            if not b:
                break
            h.update(b)
            n += len(b)
            if limit_bytes and n >= limit_bytes:
                break
    return {"sha256": h.hexdigest(), "hashed_bytes": n,
            "complete_file": n == path.stat().st_size}


def candidates(root):
    """Allow a root of packages or the supplied timestamp/subfolder layout."""
    direct = [p for p in root.iterdir() if p.is_dir() and (p / "cameras.json").is_file()]
    nested = [p.parent for p in root.glob("*/*/*/cameras.json") if p.is_file()]
    paths = sorted(set(direct + nested), key=lambda p: p.as_posix())
    missing = sorted({p.parent.name for p in root.glob("*/*/*/cameras.json")} -
                     {p.name for p in paths})
    if missing:
        raise SystemExit("Standardized package discovery missed: " + str(missing))
    if not paths:
        raise SystemExit("No standardized UE package containing cameras.json was found")
    names = [p.name for p in paths]
    if len(names) != len(set(names)):
        raise SystemExit("Duplicate sequence_id in dataset root")
    return paths


def exposure(image, mask):
    rgb = np.asarray(image.convert("RGB").resize((256, 256), Image.Resampling.BOX))
    m = np.asarray(mask.convert("L").resize((256, 256), Image.Resampling.NEAREST)) > 127
    # Restrict the exposure diagnostic to the upper person; no face/nose claim.
    roi = m & (np.indices(m.shape)[0] < 128)
    if int(roi.sum()) < 64:
        roi = m
    if int(roi.sum()) < 64:
        raise ValueError("Person mask is empty at the audit resolution")
    lum = rgb.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    values = lum[roi]
    # Model heads/garments are dark; use relative sub-black fractions and an
    # explicit visual review rather than assuming a lighting label per scene.
    mean = float(values.mean())
    p05 = float(np.percentile(values, 5))
    dark = float((values < 5).mean())
    very_dark = float((values < 15).mean())
    flag = bool(mean < 35 or dark > .05 or very_dark > .5)
    return {"region": "upper_person_mask_at_256_not_face_or_nose",
            "luma_mean_255": mean, "luma_p05_255": p05,
            "fraction_lt5": dark, "fraction_lt15": very_dark,
            "exposure_flag": flag,
            "rule": "mean<35 or fraction<5 > 0.05 or fraction<15 > 0.5; QA flag only"}


def verify_camera(cid, row):
    if row.get("camera_type") != "main":
        raise ValueError("Protocol view is not a main camera: " + cid)
    if (int(row["width"]), int(row["height"])) != (2048, 2048):
        raise ValueError("Unexpected main-camera image dimensions: " + cid)
    K = np.asarray(row["K"], np.float64)
    c2w = np.asarray(row["c2w"], np.float64)
    w2c = np.asarray(row["w2c"], np.float64)
    if (K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0
            or K[1, 1] <= 0 or K[0, 1] != 0):
        raise ValueError("Invalid intrinsics: " + cid)
    if c2w.shape != (4, 4) or not np.isfinite(c2w).all():
        raise ValueError("Invalid c2w: " + cid)
    if np.max(np.abs(c2w @ w2c - np.eye(4))) > 1e-4:
        raise ValueError("c2w/w2c roundtrip failed: " + cid)
    if row.get("depth_type") != "radial" or row.get("depth_unit") != "mm":
        raise ValueError("Expected radial uint16-mm depth: " + cid)
    return {"K": K.tolist(), "c2w": c2w.tolist(),
            "role": row["role"], "azimuth_deg": row.get("azimuth_deg"),
            "elevation_deg": row.get("elevation_deg"),
            "view_group": "source" if cid in SOURCE_IDS else (
                "train_side_boundary" if cid in TRAIN_TARGET_IDS else
                "eval_near_input" if cid in EVAL_NEAR_IDS else
                "eval_rear_secondary" if cid in EVAL_REAR_IDS else
                "additional_audit_" + row["role"])}


def inspect_view(package, row, cid, with_exposure):
    image_path = package / row["image"]
    depth_path = package / "depth_radial" / (cid + ".png")
    mask_path = package / "masks" / "foreground" / (cid + ".png")
    for p in (image_path, depth_path, mask_path):
        if not p.is_file():
            raise ValueError("Missing required training/eval input: " + str(p))
    with Image.open(depth_path) as im:
        depth = np.asarray(im)
    with Image.open(mask_path) as im:
        mask = np.asarray(im.convert("L"))
    with Image.open(image_path) as im:
        rgb = im.convert("RGB")
        wh = im.size
        stat = exposure(im, Image.fromarray(mask)) if with_exposure else None
    if depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError("Depth is not single-channel uint16: " + cid)
    if depth.shape != mask.shape or (depth.shape[1], depth.shape[0]) != wh:
        raise ValueError("RGB/depth/mask shape mismatch: " + cid)
    if wh != (int(row["width"]), int(row["height"])):
        raise ValueError("RGB dimensions differ from cameras.json: " + cid)
    if not np.any(depth > 0) or not np.any(mask > 127):
        raise ValueError("Empty depth or foreground mask: " + cid)
    fg = mask > 127
    fg_depth = depth[fg]
    if int((fg_depth == 0).sum()) > max(16, .05 * len(fg_depth)):
        raise ValueError("Too many invalid foreground depth pixels: " + cid)
    if not np.any(fg & (depth > 0)):
        raise ValueError("No valid foreground depth: " + cid)
    return {"view_id": cid, "rgb": image_path.relative_to(package).as_posix(),
            "depth_radial": depth_path.relative_to(package).as_posix(),
            "foreground_mask": mask_path.relative_to(package).as_posix(),
            "native_wh": list(wh), "valid_depth_fraction": float((depth > 0).mean()),
            "foreground_fraction": float(fg.mean()),
            "exposure_qa": stat}


def make_manifest(root, out, audit_views):
    rows = []
    for package in candidates(root):
        print("AUDIT", package.name, flush=True)
        meta = read_json(package / "meta.json")
        cameras = read_json(package / "cameras.json")
        report_path = package / "verify_out" / "verify_report.json"
        report = read_json(report_path)
        seq = meta["sequence_id"]
        identity = meta["character_id"].lower()
        if package.name != seq or not seq.startswith(identity + "_env"):
            raise ValueError("Identity/sequence naming mismatch: " + str(package))
        if len(cameras) != 141:
            raise ValueError("Expected 141 views: " + seq)
        counts = Counter(row["role"] for row in cameras.values())
        if counts != Counter(input=16, train_target=50, eval=75):
            raise ValueError("Unexpected role counts: " + seq + " " + str(counts))
        checks = report["items"]
        failed = [name for name, passed in checks.items()
                  if not (passed if isinstance(passed, bool) else passed.get("pass", False))]
        if failed != ([] if identity not in EXPECTED_C6_FAIL else ["c6_person_height_static"]):
            raise ValueError("Acceptance result differs from frozen audit: " + seq + " " + str(failed))
        # Review only the actual supervised/evaluated inputs, never arbitrary
        # rear train targets. Training defaults to e06/e19 at elevation zero.
        ids = list(SOURCE_IDS + TRAIN_TARGET_IDS + EVAL_NEAR_IDS + EVAL_REAR_IDS)
        if audit_views:
            remaining = sorted(k for k, v in cameras.items()
                               if k not in ids and v.get("camera_type") == "main"
                               and v["role"] in ("input", "eval"))
            # Deterministic coverage across the sorted input/eval set.
            pick = np.linspace(0, len(remaining) - 1, audit_views).round().astype(int)
            ids += [remaining[int(i)] for i in pick]
        views = {}
        for cid in ids:
            row = cameras[cid]
            # The 13 protocol views are always exposure-audited. Additional
            # source/eval views are structural QA only, not more formal review.
            protocol_view = cid in SOURCE_IDS + TRAIN_TARGET_IDS + EVAL_NEAR_IDS + EVAL_REAR_IDS
            v = inspect_view(package, row, cid, protocol_view)
            v.update(verify_camera(cid, row))
            views[cid] = v
        rows.append({
            "sequence_id": seq, "character_id": identity,
            "environment_id": meta["environment_id"],
            "environment_index": meta["environment_index"],
            "package": package.relative_to(root).as_posix(),
            "verify_report": report_path.relative_to(root).as_posix(),
            "verify_verdict": report["verdict"],
            "c6_pass": not failed, "c6_failure": failed,
            "training_eligible": not failed,
            "split": next((split for split, identities in SPLIT_IDS.items()
                           if identity in identities), "hard_example_only"),
            "character_height_source": meta.get("character_height_m_source"),
            "cross_view_depth_consistency": checks["c5_cross_view_depth_consistency"],
            "views": views,
        })
    counts = Counter((r["split"], r["character_id"]) for r in rows)
    id_counts = {s: sum(1 for (sp, _), n in counts.items() if sp == s) for s in
                 ("train", "val", "test", "hard_example_only")}
    seq_counts = Counter(r["split"] for r in rows)
    if id_counts != {"train": 16, "val": 2, "test": 2, "hard_example_only": 7}:
        raise ValueError("Unexpected identity split: " + str(id_counts))
    if seq_counts != Counter(train=32, val=4, test=4, hard_example_only=13):
        raise ValueError("Unexpected sequence split: " + str(seq_counts))
    by_identity = defaultdict(set)
    for r in rows:
        by_identity[r["character_id"]].add(r["split"])
    if any(len(s) != 1 for s in by_identity.values()):
        raise ValueError("Identity leakage across splits")
    return {
        "schema": SCHEMA, "dataset_root": str(root),
        "path_rule": "package plus per-view relative paths; relocate only dataset root",
        "source_ids": list(SOURCE_IDS), "train_target_ids": list(TRAIN_TARGET_IDS),
        "eval_near_ids": list(EVAL_NEAR_IDS), "eval_rear_ids": list(EVAL_REAR_IDS),
        "split_ids": {k: list(v) for k, v in SPLIT_IDS.items()},
        "split_scope": "train/val/test from c6-pass identities; c6 failures hard-example-only",
        "counts": {"packages": len(rows), "identities": len(by_identity),
                   "sequences_by_split": dict(seq_counts), "identities_by_split": id_counts},
        "supervision": {
            "training": "3 fixed sources + e06/e19 p00; no eval role or rear train_target in optimization",
            "validation": "same sources, only eval-role near/front targets",
            "test": "same sources, eval-role near/front targets; rear only secondary completeness diagnostic",
            "no_per_view_training": "Input-only teacher/student cache; target RGB/K/depth/mask never enters cache",
            "no_nose_roi": "Person mask and upper-person exposure only; no face/nose semantic truth in this delivery",
        },
        "limitations": [
            "53 sequences are 27 identities, not 53 independent identities.",
            "Thirteen packages fail c6 person-height/static and are excluded from optimization.",
            "Hana is checked by standardized-package training files, not by incomplete original raw cam_000 export.",
            "Exposure flags are per-view upper-person diagnostics, not face ROI approval or whole-environment exclusion.",
            "Cross-view report and same-camera reprojection do not independently certify millimetre facial geometry.",
            "Two test identities can support a short positive/negative signal, not broad real-person generalization.",
        ],
        "sequences": rows,
    }


def validate_output_directory(out, review_path=None):
    """Reserve a new directory, or finish one containing only its audit JSON."""
    out.mkdir(parents=True, exist_ok=True)
    existing = set(out.iterdir())
    if existing and existing != {out / "manifest.json"}:
        raise SystemExit("Refusing to overwrite a non-empty output directory: " + str(out))
    if review_path and Path(review_path).resolve() == out.resolve():
        raise SystemExit("Review template must be a file outside the output directory")


def write_outputs(root, manifest, out, audit_views):
    existing = set(out.iterdir())
    if existing and (existing != {out / "manifest.json"}
                     or read_json(out / "manifest.json") != manifest):
        raise SystemExit("Refusing to overwrite a different audit: " + str(out))
    dump_json(out / "manifest.json", manifest)
    with open(out / "sequence_audit.csv", "w", encoding="utf-8", newline="") as f:
        fields = ("sequence_id", "character_id", "environment_index", "split",
                  "training_eligible", "c6_pass", "verify_verdict", "package")
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in manifest["sequences"]:
            w.writerow({k: row[k] for k in fields})
    if audit_views:
        flagged = []
        with open(out / "view_exposure_qa.csv", "w", encoding="utf-8", newline="") as f:
            fields = ("sequence_id", "character_id", "split", "view_id", "role",
                      "luma_mean_255", "luma_p05_255", "fraction_lt5", "fraction_lt15",
                      "exposure_flag", "package")
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in manifest["sequences"]:
                for cid, v in row["views"].items():
                    q = v["exposure_qa"]
                    if q is None:
                        continue
                    rec = {"sequence_id": row["sequence_id"], "character_id": row["character_id"],
                           "split": row["split"], "view_id": cid, "role": v["role"],
                           "package": row["package"], **{k: q[k] for k in fields[5:-1]}}
                    w.writerow(rec)
                    if q["exposure_flag"]:
                        flagged.append(rec)
        summary = {"n_sequences": len(manifest["sequences"]),
                   "n_audited_views": sum(len(r["views"]) for r in manifest["sequences"]),
                   "n_exposure_flagged": len(flagged),
                   "flagged_by_split": dict(Counter(r["split"] for r in flagged)),
                   "flagged_by_environment": dict(Counter(
                       r["environment_index"] for r in manifest["sequences"]
                       for v in r["views"].values() if v["exposure_qa"] is not None
                       and v["exposure_qa"]["exposure_flag"])),
                   "note": "Flagged views remain in QA; require per-view review before changing the training pool."}
        dump_json(out / "audit_summary.json", summary)
        with open(out / "view_exposure_flagged.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(flagged[0]) if flagged else
                               ["sequence_id", "view_id", "exposure_flag"])
            w.writeheader()
            w.writerows(flagged)
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2), flush=True)
    print("MANIFEST", out / "manifest.json", flush=True)


def write_review_template(root, manifest, path):
    entries = {}
    for row in manifest["sequences"]:
        for cid, v in row["views"].items():
            key = row["sequence_id"] + "/" + cid
            entries[key] = {"package": row["package"], "rgb": v["rgb"],
                            "role": v["role"], "status": "pending",
                            "exposure_flag": v["exposure_qa"]["exposure_flag"]
                            if v["exposure_qa"] is not None else None,
                            "note": "Inspect native RGB: source/target readability and exposure; do not infer face/nose ROI."}
    blob = {"schema": REVIEW_SCHEMA, "dataset_root": str(root),
            "instructions": "Set every status to accepted or rejected before --require-review; rejected targets are replaced from unused side targets, sources are immutable.",
            "views": entries}
    if path.exists():
        old = read_json(path)
        if old != blob:
            raise SystemExit("Existing review differs or contains human decisions; use a new path: " + str(path))
        print("REUSE_IDENTICAL_PENDING_REVIEW_TEMPLATE", path, flush=True)
        return
    dump_json(path, blob)
    print("REVIEW_TEMPLATE", path, flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--audit-views", type=int, default=0,
                   help="Extra deterministic source/eval views per package, in addition to protocol views")
    p.add_argument("--write-review-template", default="")
    p.add_argument("--require-review", default="")
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args(argv)
    root = Path(args.dataset_root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit("Dataset root is not a directory: " + str(root))
    if args.check_only:
        out.mkdir(parents=True, exist_ok=True)
    else:
        validate_output_directory(out, args.write_review_template)
    if args.write_review_template and Path(args.write_review_template).exists():
        existing = read_json(args.write_review_template)
        if (existing.get("schema") != REVIEW_SCHEMA
                or existing.get("dataset_root") != str(root)
                or any(v.get("status") != "pending" for v in existing.get("views", {}).values())):
            raise SystemExit("Do not overwrite a different or human-reviewed template: " + args.write_review_template)
    manifest = make_manifest(root, out, args.audit_views)
    # Structural audit is complete; preserve it before optional reporting I/O.
    dump_json(out / "manifest.json", manifest)
    if args.write_review_template:
        if not args.audit_views:
            raise SystemExit("Review template requires --audit-views for exposure QA")
        write_review_template(root, manifest, Path(args.write_review_template))
    if args.require_review:
        review = read_json(args.require_review)
        if review.get("schema") != REVIEW_SCHEMA:
            raise SystemExit("Wrong review schema")
        for seq in manifest["sequences"]:
            for cid in seq["views"]:
                key = seq["sequence_id"] + "/" + cid
                status = review["views"].get(key, {}).get("status")
                if status not in ("accepted", "rejected"):
                    raise SystemExit("Unreviewed view: " + key)
    if not args.check_only:
        write_outputs(root, manifest, out, args.audit_views)


if __name__ == "__main__":
    main()
