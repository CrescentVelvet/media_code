#!/usr/bin/env python3
"""26b_new_gt_landmarks.py -- 新合成身份的原生 FaceMesh 与 ROI QA 材料。

必须在 sam3 环境中、运行新身份模型前执行。只检测 manifest 已冻结的帧。
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("ROOT", "/mnt/d/output/results/splatanything_results/"
                           "densitok_portrait_probe/protocol_v1"))
sys.path.insert(0, str(HERE))
import importlib.util
spec = importlib.util.spec_from_file_location("paired_landmarks", HERE / "21b_geometry_landmarks.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def main():
    protocol = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    pack = next((p for p in protocol["packages"] if p["kind"] == "static_synthetic"
                 and p.get("new_identity")), None)
    if pack is None:
        raise SystemExit("No new static identity in manifest")
    manifest = json.loads((Path(pack["pkg"]) / "cameras.json").read_text(encoding="utf-8"))
    cameras = {int(c["id"]): c for c in manifest["cameras"]}
    qa = ROOT / "daven_landmarks"
    qa.mkdir(exist_ok=True)
    result = {"schema": "paired_new_identity_landmarks_v1", "views": {},
              "visual_review_status": "see_roi_qa_json_manual_review_is_authoritative"}
    roi_qa = {"schema": "paired_roi_visual_qa_v1", "status": "pending_manual_review",
              "views": {}}
    failures = {}
    for rec in pack["frames"]:
        stem = rec["stem"]
        mc = cameras[rec["raw_id"]]
        im = Image.open(pack["pkg"] + "/" + mc["file_rgb"]).convert("RGB")
        rgb = np.asarray(im)
        try:
            points, factor = mod.detect(rgb)
        except RuntimeError as exc:
            failures[str(rec["raw_id"])] = {"split": rec["split"], "error": str(exc)}
            roi_qa["views"][str(rec["raw_id"])] = {
                "id": rec["raw_id"], "split": rec["split"],
                "status": "no_facemesh_target_rejected" if rec["split"] == "target"
                          else "source_only_no_facemesh", "excluded_rois": []}
            print("NO_FACEMESH", rec["split"], rec["raw_id"], flush=True)
            continue
        result["views"][str(rec["raw_id"])] = {"id": rec["raw_id"], "file_rgb": mc["file_rgb"],
                                 "file_depth": mc["file_depth"], "detection_scale": factor,
                                 "points": points.tolist()}
        roi_qa["views"][str(rec["raw_id"])] = {
            "id": rec["raw_id"], "split": rec["split"], "status": "pending",
            "excluded_rois": [], "qa_image": stem + "_landmarks.png"}
        ann = im.copy()
        draw = ImageDraw.Draw(ann)
        for name, inds in mod.REGIONS.items():
            color = (255, 40, 40) if name.startswith("nose") else (20, 160, 255)
            for x, y in points[inds]:
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
            draw.text(tuple(points[inds].mean(axis=0)), name, fill=color)
        ann.save(qa / (stem + "_landmarks.png"))
    result["failures"] = failures
    with open(ROOT / "daven_landmarks_native.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(ROOT / "roi_qa.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(roi_qa, f, ensure_ascii=False, indent=2)
    print("NEW_GT_LANDMARKS", result["visual_review_status"], "failures", failures, flush=True)
    if any(v["split"] == "target" for v in failures.values()):
        raise RuntimeError("Target FaceMesh failed; no approximate ROI is permitted")


if __name__ == "__main__":
    main()
