#!/usr/bin/env python3
"""28_object_control.py -- 冻结物体输入的三/五图前馈及外观控制。

每个 arm/condition 串行前向，输出目录拒绝覆盖。仅原三锚点估计对齐，
target 不参与拟合/选参；无训练、TokenBA、面部 ROI 或米制深度声明。
"""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent


def load():
    spec = importlib.util.spec_from_file_location("paired27_object", HERE / "27_evaluate_paired.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--protocol-root", required=True)
    ap.add_argument("--official-repo", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--forward", action="store_true", help="仅当 models 不存在时串行启动已冻结输入的前馈")
    args = ap.parse_args()
    if Path(sys.executable).resolve() != (Path.home()/"miniconda3/envs/splatanything/bin/python").resolve():
        raise RuntimeError("Object control must use splatanything env")
    M = load()
    sys.path.insert(0, str(Path(args.official_repo).resolve()))
    root, out, protocol_root = Path(args.control_root), Path(args.out), Path(args.protocol_root)
    manifest = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    if manifest["input_visual_review"]["status"] != "accepted" or manifest["training"] or manifest["token_ba"]:
        raise RuntimeError("Frozen object input review/protocol is not accepted")
    out.mkdir(parents=True, exist_ok=False)
    models = out/"models"
    if args.forward:
        for arm in ("source3", "source5"):
            scene = root/arm
            for name in ("SA_RGB", "SA_pose_aux"):
                mode = "rgb" if name == "SA_RGB" else "pose"
                result = models/("sa_"+mode)/("object_"+arm)
                peak_log = models/("sa_"+mode)/("object_"+arm+"_gpu_mib.log")
                peak_log.parent.mkdir(parents=True, exist_ok=True)
                if result.exists():
                    raise RuntimeError("Refuse to overwrite existing forward: "+str(result))
                env = dict(os.environ, CONDA_ENV="splatanything", INPUT=str(scene), MODE=mode,
                           RESOLUTION="756 756", OUT_ROOT=str(models/("sa_"+mode)), PROFILE="0",
                           SPLATANYTHING_MODEL_DIR="/mnt/d/wheel/splatanything_ms")
                print("OBJECT_FORWARD", arm, name, flush=True)
                started = time.perf_counter()
                with open(peak_log, "w", encoding="utf-8", newline="\n") as log:
                    if shutil.which("nvidia-smi"):
                        monitor = subprocess.Popen(["bash", "-c",
                            'while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.5; done'],
                            stdout=log, stderr=subprocess.DEVNULL)
                    else:
                        monitor = None
                    try:
                        subprocess.run(["bash", str(HERE/"02_run_inference.sh")], env=env, check=True)
                        run = json.loads((result/"run.json").read_text(encoding="utf-8"))
                        if run["output_path"] != str(result.parent) or \
                                Path(run["input_path"]).resolve() != scene.resolve():
                            raise RuntimeError("SA output was relocated; preserve it and use a new control OUT: "+str(result))
                    finally:
                        if monitor is not None:
                            monitor.terminate()
                            monitor.wait()
                elapsed = time.perf_counter()-started
                peak = max([float(x) for x in peak_log.read_text().split() if x.strip()] or [0.])
                M.dump(models/("sa_"+mode)/("object_"+arm+"_cost.json"),
                       {"elapsed_seconds": elapsed, "whole_gpu_peak_mib": peak if monitor is not None else None,
                        "gpu_scope": "whole GPU, includes this serial forward; not allocator-only peak" if monitor is not None else "unavailable"})
            result = models/"zip_rgb"/("object_"+arm)
            if result.exists():
                raise RuntimeError("Refuse to overwrite existing forward: "+str(result))
            env = dict(os.environ, IMAGES_DIR=str(scene/"images"), SCENE_NAME="object_"+arm,
                       WEIGHT_PATH=str(Path.home()/"model/zipsplat/zipsplat-da3g-504.tar"),
                       OUT_DIR=str(result), MAX_VIEWS="5", BA="0", SAVE_FEEDFORWARD="1", RENDER_VIDEO="0")
            print("OBJECT_FORWARD", arm, "Zip_FF", flush=True)
            subprocess.run(["bash", str(HERE.parent/"zipsplat/02_run_inference.sh")], env=env, check=True)
    frames = []
    for split in ("anchors", "additional", "targets"):
        for i, stem in enumerate(manifest[split]):
            arm = "target" if split == "targets" else ("source3" if split == "anchors" else "source5")
            rec = manifest["scenes"][arm][stem]
            path = root/arm/"images"/(stem+".png")
            if hashlib.sha256(path.read_bytes()).hexdigest() != rec["sha256"]:
                raise RuntimeError("Frozen object image changed: "+str(path))
            frame = {"raw_id": int(stem.rsplit("_", 1)[1]), "split": {"anchors": "anchor", "additional": "additional", "targets": "target"}[split],
                     "path": str(path), "processed_wh": rec["size"],
                     "rgb_sha256": hashlib.sha256(Image.open(path).convert("RGB").tobytes()).hexdigest(),
                     "scene_slots": {}}
            for arm_name in ("source3", "source5", "target"):
                if stem in manifest["scenes"][arm_name]:
                    frame["scene_slots"][arm_name] = stem
                    other = manifest["scenes"][arm_name][stem]
                    if not np.allclose(other["c2w"], rec["c2w"]) or not np.allclose(other["K"], rec["K"]):
                        raise RuntimeError("Object camera differs across arms: "+stem)
                    if arm_name != "target":
                        image_path = root/arm_name/"images"/(stem+".png")
                        frame[arm_name+"_input_file_sha256"] = hashlib.sha256(image_path.read_bytes()).hexdigest()
            frames.append(frame)
    pack = {"name": "object_control", "kind": "static_object_no_metric_truth", "frames": frames,
            "source3_scene": str(root/"source3"), "source5_scene": str(root/"source5"),
            "target_scene": str(root/"target")}
    (out/"images").mkdir()
    methods = {"source3": {}, "source5": {}}
    for arm in ("source3", "source5"):
        anchors = {f["scene_slots"][arm] for f in frames if f["split"] == "anchor"}
        for cond in ("SA_RGB", "Zip_FF", "SA_pose_aux"):
            m = M.model_meta(pack, out, arm, cond, anchors, args.device)
            m["provenance"] = M.validate_provenance(m, {}, protocol_root, pack)
            m["source_scene"] = pack[arm+"_scene"]
            m["frames"] = pack["frames"]
            m["arm"] = arm
            M.native_replay(m, out/"native"/(arm+"_"+cond), args.device)
            m["source_projection"] = M.source_projection_checks(
                m, pack, out/"source_projection"/(arm+"_"+cond), args.device)
            m["gaussians"] = len(m["g"]["means"])
            m["repeat_camera"] = M.target_setup(pack, arm, cond, M.M25.evaluation_geometry(
                np.asarray(manifest["scenes"]["target"][manifest["targets"][0]]["K"], np.float32)
                if cond == "SA_pose_aux" else None,
                tuple(manifest["scenes"]["target"][manifest["targets"][0]]["size"]),
                np.asarray(manifest["scenes"]["target"][manifest["targets"][0]]["K"], np.float32)),
                manifest["targets"][0], m["fit"])
            m["repeat_K"] = M.M25.evaluation_geometry(
                np.asarray(manifest["scenes"]["target"][manifest["targets"][0]]["K"], np.float32)
                if cond == "SA_pose_aux" else None,
                tuple(manifest["scenes"]["target"][manifest["targets"][0]]["size"]),
                np.asarray(manifest["scenes"]["target"][manifest["targets"][0]]["K"], np.float32))["K_reference"]
            m["repeatability"] = M.repeatability_check(
                m, out/"repeatability"/(arm+"_"+cond+".json"), args.device)
            if not m["repeatability"]["pass"]:
                raise RuntimeError("Object control repeat rendering exceeds tolerance")
            methods[arm][cond] = m
    report = {"schema": "paired_object_control_results_v1",
              "scope": "full-image appearance and alpha coverage only; central rectangle is a diagnostic crop, not a background or instance mask; no face ROI or metric depth",
              "central_rectangle_xy_fraction": [.15, .10, .85, .95], "targets": {}, "forward_models": str(models),
              "models": {arm: {cond: {"gaussians": m["gaussians"], "fit": m["fit"],
                                      "residual": m["residual"], "runtime": m["runtime"],
                                      "source_projection": m["source_projection"],
                                      "repeatability": m["repeatability"]}
                               for cond, m in methods[arm].items()} for arm in methods}}
    for fr in (x for x in frames if x["split"] == "target"):
        stem = fr["scene_slots"]["target"]
        prior = json.loads((root/"target/prior_camera.json").read_text(encoding="utf-8"))
        K = np.asarray(next(c["matrix"] for c in prior["intrinsics"] if c["camera_id"] == stem), np.float32)
        prep = M.M25.evaluation_geometry(None, tuple(fr["processed_wh"]), K)
        gt = M.M21.official_preprocess_rgb(fr["path"], prep)
        roi = np.zeros(gt.shape[:2], bool)
        roi[int(.10*756):int(.95*756), int(.15*756):int(.85*756)] = True
        view = {"raw_id": fr["raw_id"], "methods": {}}
        rows = [("GT", [("RGB", gt), ("lowpass", cv2.GaussianBlur(gt, (0, 0), 2.))])]
        for arm in ("source3", "source5"):
            for cond, m in methods[arm].items():
                track = "auxiliary" if cond == "SA_pose_aux" else "primary"
                target_prep = M.M25.evaluation_geometry(K if track == "auxiliary" else None,
                                                        tuple(fr["processed_wh"]), K)
                cam = M.target_setup(pack, arm, cond, target_prep, stem, m["fit"])
                rd, al, _ = M.M22.render(m["gpu"], m["degree"], cam, target_prep["K_reference"], 756, args.device, .01)
                target_gt = M.M21.official_preprocess_rgb(fr["path"], target_prep)
                values = {scope: M.metrics(target_gt, rd, al, mask)
                          for scope, mask in (("full", np.ones(roi.shape, bool)), ("central_rectangle", roi))}
                values["lowpass_full"] = M.metrics(cv2.GaussianBlur(target_gt, (0, 0), 2.),
                                                   cv2.GaussianBlur(rd, (0, 0), 2.), al, np.ones(roi.shape, bool))
                view["methods"].setdefault(cond, {})[arm] = values
                rows.append((cond+" "+arm, [("RGB", rd), ("lowpass", cv2.GaussianBlur(rd, (0, 0), 2.))]))
        M.comparison(out/"images"/(stem+"_comparison.png"), rows)
        report["targets"][stem] = view
    M.dump(out/"object_results.json", report)
    M.dump(out/"object_visual_review_template.json", {
        "status": "pending_visual_review", "central_rectangle_not_instance_mask": True,
        "review": [{"target": stem, "condition": cond, "observation": "pending", "obvious_regression": None}
                   for stem in report["targets"] for cond in ("SA_RGB", "Zip_FF", "SA_pose_aux")],
        "gate": "no obvious target appearance/coverage degradation; visual gate independent from PSNR"})
    print("OBJECT_CONTROL_MEASUREMENT_DONE", out, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        print(tb, end="", flush=True)
        if "--out" in sys.argv:
            p = Path(sys.argv[sys.argv.index("--out")+1])
            if p.is_dir():
                with open(p/"failure.log", "w", encoding="utf-8", newline="\n") as f:
                    f.write(tb)
        raise
