#!/usr/bin/env python3
"""25_n1_real_eval.py -- 两个真实近静态包的六组前馈外观与受限覆盖验收。

不读取/使用真实深度或 FBX，不产生毫米几何指标。仅用三对 source 相机拟合
X_prior=s R X_model+T，target 相机逆回模型系；不按 target 图拟合或选择参数。
正式主轨为 SA RGB 与 Zip ba=False，SA known-pose 单列辅助轨。
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M03 = load("real_render03", "03_render_eval.py")
M15 = load("real_sim15", "15_face_target_eval.py")
M21 = load("real_preprocess21", "21_geometry_audit_v2.py")
M22 = load("real_render22", "22_n1_head_compare.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(M21.jsonable(data), f, ensure_ascii=False, indent=2)


def image(path):
    return np.asarray(Image.open(path).convert("RGB"), np.float32) / 255


def evaluation_geometry(K, wh, projection_K=None):
    """crop布局与实际投影内参分开；均调用官方crop/resize算子。

    RGB-only 的虚拟K只决定居中裁图，target投影必须用真实采集K，不能拿
    虚拟焦距冒充物理相机。known-pose的布局K与投影K相同。
    preprocess_view返回K和21号旧CPU的重复offset均不用于共同GT。
    """
    from src.utils.scene_io import _centered_default_intrinsics
    from src.utils.cropping import (crop_image_depthmap, rescale_image_depthmap,
                                    camera_matrix_of_crop, bbox_from_intrinsics_in_out)
    image = Image.new("RGB", tuple(wh))
    native_K = _centered_default_intrinsics(tuple(wh)) if K is None else np.asarray(K, np.float32).copy()
    projection_K = native_K if projection_K is None else np.asarray(projection_K, np.float32).copy()
    K = native_K.copy()
    cx, cy = K[:2, 2].round().astype(int)
    mx, my = min(cx, wh[0] - cx), min(cy, wh[1] - cy)
    first_box = (int(cx - mx), int(cy - my), int(cx + mx), int(cy + my))
    image, _, K = crop_image_depthmap(image, None, K, first_box)
    projection_K = crop_image_depthmap(image, None, projection_K, first_box)[2]
    crop_size = image.size
    resized_size = tuple(np.floor(np.asarray(crop_size) * (
        max(np.asarray((756, 756)) / np.asarray(crop_size)) + 1e-8)).astype(int))
    # K 缩放必须读取 resize 前的图像尺寸；不能传入已经 resize 的 image。
    # 第三参始终是 K，返回值赋给投影 K，不得误赋给 image。
    projection_K = rescale_image_depthmap(
        image, None, projection_K, np.asarray((756, 756)))[2]
    image, _, K = rescale_image_depthmap(image, None, K, np.asarray((756, 756)))
    if image.size != resized_size:
        raise RuntimeError("Image/K resize layout diverged")
    if np.max(np.abs(projection_K - native_K)) < 1e-8:
        raise RuntimeError("Projection K was not rescaled")
    target_K = camera_matrix_of_crop(K, resized_size, (756, 756), offset_factor=0.5)
    bbox = bbox_from_intrinsics_in_out(K, target_K, (756, 756))
    _, _, cropped_K = crop_image_depthmap(image, None, projection_K, bbox)
    return {"K_reference": cropped_K, "first_box": first_box,
            "crop_size": list(crop_size), "resized_size": list(resized_size),
            "bbox_offset": list(bbox[:2]), "size": [756, 756],
            "scale_final": float(max(np.asarray((756, 756)) / np.asarray(crop_size)) + 1e-8),
            "native_principal_rounding": native_K[:2, 2].tolist(),
            "native_projection_K": projection_K,
            "target_K_before_integer_crop": target_K}


def native_k_from_saved(row, size=(378, 504)):
    """Zip cameras=[W,H,fx,fy,cx,cy]；原生渲染用长边等比内参。"""
    w, h, fx, fy, cx, cy = map(float, row)
    factor = max(size[0] / w, size[1] / h)
    return np.array([[fx * factor, 0, cx * factor],
                     [0, fy * factor, cy * factor], [0, 0, 1]], np.float32)


def native_metrics(common, rd):
    d = common.astype(np.float64) - rd
    return {"psnr": float(-10 * np.log10(max(np.mean(d*d), 1e-12))),
            "mae": float(np.mean(np.abs(d)))}


def metrics(gt, rd, alpha, mask=None):
    if mask is None:
        mask = np.ones(gt.shape[:2], bool)
    if mask.sum() < 32:
        return {"status": "N/A", "n": int(mask.sum())}
    d = gt[mask].astype(np.float64) - rd[mask]
    return {"n": int(mask.sum()), "psnr": float(-10 * np.log10(max(np.mean(d*d), 1e-12))),
            "mae": float(np.mean(np.abs(d))), "alpha_mean": float(alpha[mask].mean()),
            "coverage_05": float((alpha[mask] > .5).mean())}


def comparison(out, stem, gt, renders, roi=None):
    width = 360
    panels = [("Target RGB", gt)] + [(name, rd) for name, rd, _ in renders]
    rows = []
    for name, im in panels:
        tile = Image.fromarray(np.uint8(np.clip(im, 0, 1) * 255)).resize((width, width))
        d = ImageDraw.Draw(tile)
        d.text((8, 8), name, fill="white", stroke_width=2, stroke_fill="black")
        rows.append(np.asarray(tile))
    Image.fromarray(np.concatenate(rows, axis=1)).save(out / f"{stem}_target_rgb.png")
    rows = []
    for name, im in panels:
        lp = cv2.GaussianBlur(im, (0, 0), 2.)
        tile = Image.fromarray(np.uint8(np.clip(lp, 0, 1) * 255)).resize((width, width))
        d = ImageDraw.Draw(tile)
        d.text((8, 8), name + " lowpass sigma2", fill="white", stroke_width=2, stroke_fill="black")
        rows.append(np.asarray(tile))
    Image.fromarray(np.concatenate(rows, axis=1)).save(out / f"{stem}_target_lowpass.png")
    if roi is not None and roi.sum() >= 32:
        ys, xs = np.where(roi)
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        rows = []
        for name, im in panels:
            face = Image.fromarray(np.uint8(np.clip(im[y0:y1, x0:x1], 0, 1) * 255))
            face = face.resize((width, width), Image.Resampling.LANCZOS)
            ImageDraw.Draw(face).text((8, 8), name + " tight ROI", fill="white",
                                      stroke_width=2, stroke_fill="black")
            rows.append(np.asarray(face))
        Image.fromarray(np.concatenate(rows, axis=1)).save(out / f"{stem}_target_face.png")
    rows = []
    for name, rd, alpha in renders:
        tile = Image.fromarray(np.uint8(np.clip(alpha, 0, 1) * 255)).resize((width, width))
        ImageDraw.Draw(tile).text((8, 8), name + " full alpha", fill="white",
                                  stroke_width=2, stroke_fill="black")
        rows.append(np.asarray(tile.convert("RGB")))
    Image.fromarray(np.concatenate(rows, axis=1)).save(out / f"{stem}_target_alpha.png")


def target_camera(m, c2w):
    if m["formal_track"] == "auxiliary":
        return M21.model_camera(c2w, *m["sim3"][:3])
    s, R, T, _ = m["sim3"]
    return M21.model_camera(c2w, 1/s, R.T, -(R.T @ T)/s)


def shared_source_checks(methods, pack, out, device):
    """同一 source 投影回归与模型 target 质量分开。

    known-pose 的共同 K 必须能直接回放原生 source，且与 native_checks 的图一致；
    RGB-only/Zip 的预测相机、预测焦距换成共同物理相机/K 后只记录诊断，不硬调。
    """
    report = {}
    src_prior = json.loads((Path(pack["source_scene"]) / "prior_camera.json").read_text(encoding="utf-8"))
    priors = {c["camera_id"]: np.asarray(c["matrix"], np.float64) for c in src_prior["extrinsics"]}
    for name, m in methods.items():
        prep = m["evaluation_geometry"]
        report[name] = {}
        for stem, source_camera in m["source_cams"].items():
            auxiliary = m["formal_track"] == "auxiliary"
            if auxiliary:
                if np.max(np.abs(m["source_K"][stem] - prep["K_reference"])) > 1e-4:
                    raise RuntimeError("Known-pose shared source K differs from native lifting K: " + stem)
                camera = source_camera
            else:
                camera = target_camera(m, priors[stem])
            rd, alpha, _ = M22.render(m["gpu"], m["g"]["degree"], camera,
                                       prep["K_reference"], 756, device, .01)
            native = image(out / (pack["scene"] + "_native") / f"{stem}_{name}_native.png") if name != "Zip_FF" else None
            record = {"shared_K": prep["K_reference"], "native_K": m["source_K"][stem],
                      "camera": "saved native source camera" if auxiliary else "prior source camera in model coordinates",
                      "source_reconstruction": native_metrics(
                          image(m["result"] / "rgb" / (stem + ".png")), rd) if name != "Zip_FF" else None,
                      "interpretation": "source-only projection diagnostic; not held-out quality or geometry truth"}
            if native is not None:
                record["same_native_camera_rgb_max_abs"] = float(np.max(np.abs(native - rd)))
                if auxiliary and record["same_native_camera_rgb_max_abs"] > 3 / 255:
                    raise RuntimeError("Known-pose shared source replay differs from native: " + stem)
            report[name][stem] = record
            Image.fromarray(np.uint8(np.clip(rd, 0, 1) * 255)).save(
                out / pack["scene"] / f"{stem}_{m['formal_track']}_{name}_shared_source.png")
    return report


def model_setup(pack, root, device, official_repo):
    """只读六组已保存前馈结果；输入、相机、权重链分开核验。"""
    source = Path(pack["source_scene"])
    src_prior = json.loads((source / "prior_camera.json").read_text(encoding="utf-8"))
    tgt_prior = json.loads((Path(pack["target_scene"]) / "prior_camera.json").read_text(encoding="utf-8"))
    ids = [c["camera_id"] for c in src_prior["extrinsics"]]
    prior = {c["camera_id"]: np.asarray(c["matrix"], np.float64)
             for c in src_prior["extrinsics"] + tgt_prior["extrinsics"]}
    expected = [Path(f["processed_path"]) for f in pack["frames"] if f["split"] == "source"]
    expected.sort(key=lambda p: p.stem)
    if [p.stem for p in expected] != ids:
        raise RuntimeError("Frozen source order differs from camera order")
    methods = {}
    for name, condition in (("SA_RGB", "rgb"), ("SA_pose_aux", "pose")):
        result = root / "models" / ("sa_rgb" if name == "SA_RGB" else "sa_pose") / (pack["scene"] + "_source")
        run = json.loads((result / "run.json").read_text())
        if run["images"] != [str(p) for p in expected] or run["resolution"] != [756, 756]:
            raise RuntimeError("SA run is not on exact ordered sources: " + name)
        if run["pose_condition"] != (condition == "pose"):
            raise RuntimeError("SA condition mismatch")
        cams, K, cam_ids = M03.load_cameras(result / "cameras.json")
        if cam_ids != ids:
            raise RuntimeError("SA source camera order mismatch")
        prep = evaluation_geometry(
            np.asarray(src_prior["intrinsics"][0]["matrix"], np.float32)
            if name == "SA_pose_aux" else None, (1439, 1919),
            np.asarray(src_prior["intrinsics"][0]["matrix"], np.float32))
        # 官方 scene_io.py:45 错将旧 K 作为 K2 返回；旧 CPU 参考不应当作正确门。
        # 共同 GT 用真实整数裁剪后的 K；SA 原生 PLY 核验坚持用实际导出的 K。
        for stem in ids:
            gt = M21.official_preprocess_rgb(source / "images" / (stem + ".png"), prep)
            actual = image(result / "rgb" / (stem + ".png"))
            if np.max(np.abs(gt - actual)) > 2 / 255:
                raise RuntimeError("SA input RGB differs from official preprocess")
        from src.utils.scene_io import preprocess_view
        returned_K = []
        for j, p in enumerate(expected):
            k_input = np.asarray(src_prior["intrinsics"][j]["matrix"], np.float32) if name == "SA_pose_aux" else None
            expected_view, expected_K = preprocess_view(Image.open(p), k_input, (756, 756))
            returned_K.append(expected_K.numpy())
            actual = image(result / "rgb" / p.name)
            if np.max(np.abs(expected_view.moveaxis(0, -1).numpy() - actual)) > 1e-6:
                raise RuntimeError("SA official input RGB cross-check failed: " + name + " " + p.name)
        if name == "SA_pose_aux":
            if np.max(np.abs(np.stack(returned_K) - K)) > 1e-4:
                raise RuntimeError("SA known-pose native K differs from official forward input")
            if np.max(np.abs(np.stack(returned_K) - prep["K_reference"])) > 1e-4:
                raise RuntimeError("Known-pose common target K differs from actual lifting K")
            fit = M15.estimate_sim3(result, source)
            fit = (*fit[:3], {"rule": "known-pose N0 export camera fit; X_model=s R X_prior+T"})
        else:
            fit = M22.fit_sim3(np.stack([cams[i, :3, 3] for i in range(3)]),
                               np.stack([prior[s][:3, 3] for s in ids]))
            fit = (*fit[:3], {"rule": "source-only center Umeyama"})
        residual = M22.source_camera_residuals(
            np.stack([cams[i] for i in range(3)]), np.stack([prior[s] for s in ids]), fit, ids)
        methods[name] = {"result": result, "ply": result / "gaussians.ply", "sim3": fit,
                         "source_cams": dict(zip(ids, cams)), "source_K": dict(zip(ids, K)),
                         "source_residual": residual, "native_eps": .01,
                         "target_preprocess_K": np.asarray(src_prior["intrinsics"][0]["matrix"], np.float32),
                         "crop_preprocess_K": np.asarray(src_prior["intrinsics"][0]["matrix"], np.float32)
                         if name == "SA_pose_aux" else None,
                         "evaluation_geometry": evaluation_geometry(
                             None if name == "SA_RGB" else np.asarray(src_prior["intrinsics"][0]["matrix"], np.float32),
                             (1439, 1919), np.asarray(src_prior["intrinsics"][0]["matrix"], np.float32)),
                         "condition": "RGB only" if name == "SA_RGB" else "RGB + acquisition camera prior",
                         "formal_track": "primary" if name == "SA_RGB" else "auxiliary"}
    result = root / "models" / "zip_rgb" / (pack["scene"] + "_source")
    run = json.loads((result / "metrics.json").read_text(encoding="utf-8"))
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    if run["official_commit"] != protocol["software"]["zip_repo"]:
        raise RuntimeError("Zip commit differs from frozen protocol")
    if run["token_ba"] or run["image_files"] != [p.name for p in expected]:
        raise RuntimeError("Zip source/BA protocol mismatch")
    z = np.load(result / "predicted_cameras.npz", allow_pickle=False)
    if list(z["names"]) != [p.name for p in expected]:
        raise RuntimeError("Zip saved source order mismatch")
    cams = {n[:-4]: c for n, c in zip(z["names"], z["c2w"])}
    fit = M22.fit_sim3(np.stack([prior[s][:3, 3] for s in ids]),
                       np.stack([cams[s][:3, 3] for s in ids]))
    fit = (*fit[:3], {"rule": "source-only center Umeyama; no Wahba/target selection"})
    residual = M22.source_camera_residuals(np.stack([prior[s] for s in ids]),
                                           np.stack([cams[s] for s in ids]), fit, ids)
    methods["Zip_FF"] = {"result": result, "ply": result / "gaussians_feedforward.ply",
                         "sim3": fit, "source_cams": cams,
                         "source_K": {p.stem: native_k_from_saved(row)
                                      for p, row in zip(expected, z["cameras"])},
                         "source_residual": residual, "native_eps": .3,
                         "native_size": (378, 504),
                         "target_preprocess_K": np.asarray(src_prior["intrinsics"][0]["matrix"], np.float32),
                         "crop_preprocess_K": None,
                         "evaluation_geometry": methods["SA_RGB"]["evaluation_geometry"],
                         "condition": "RGB only, token_ba=False", "formal_track": "primary"}
    for name, m in methods.items():
        m["g"] = M22.load_ply(m["ply"], "wxyz")
        m["gpu"] = M22.to_gpu(m["g"], device)
        m["ply_sha256"] = sha(m["ply"])
        if name == "Zip_FF":
            m["weight_path"] = run["weight_path"]
        else:
            config = (m["result"] / "config.yaml").read_text(encoding="utf-8")
            if "enable_cond: " + ("true" if name == "SA_pose_aux" else "false") not in config:
                raise RuntimeError("SA config condition mismatch: " + name)
            key = "SplatAnything_pretrained:"
            m["weight_path"] = next(line.split(key, 1)[1].strip() for line in config.splitlines() if key in line)
            m["geometry_weight_path"] = next(line.split("worldmirror2_pretrained:", 1)[1].strip()
                                               for line in config.splitlines() if "worldmirror2_pretrained:" in line)
        m["weight_sha256"] = {p: sha(p) for p in [m["weight_path"]] + ([m["geometry_weight_path"]] if "geometry_weight_path" in m else [])}
        m["run_sha256"] = sha(m["result"] / ("metrics.json" if name == "Zip_FF" else "run.json"))
        m["config_sha256"] = sha(m["result"] / ("predicted_cameras.npz" if name == "Zip_FF" else "config.yaml"))
    return methods


def native_checks(methods, pack, out, device):
    """双读取器回放是硬门；输入图重建质量是诊断，不伪称原生渲染参考。"""
    out.mkdir()
    records = {}
    for name, m in methods.items():
        records[name] = {}
        if name == "Zip_FF":
            records[name] = {"reference": "separate v6 audit: official native render vs PLY low-level render",
                             "saved_native_reference_available": True}
            continue
        base = M03.load_ply(m["ply"])
        for stem, camera in m["source_cams"].items():
            K = m["source_K"][stem]
            rd, alpha, _ = M22.render(m["gpu"], m["g"]["degree"], camera, K, 756, device, .01)
            reference, reference_alpha = M03.render(base, camera, K, 756, 756,
                                                     background=0, device=device)
            rgb_diff = float(np.max(np.abs(rd - reference[0])))
            alpha_diff = float(np.max(np.abs(alpha - reference_alpha[0, ..., 0])))
            if rgb_diff > 3 / 255 or alpha_diff > 3 / 255:
                raise RuntimeError("SA independent native PLY replay failed: " + name)
            common = image(m["result"] / "rgb" / (stem + ".png"))
            replay = {"independent_reader_rgb_max_abs": rgb_diff,
                      "independent_reader_alpha_max_abs": alpha_diff,
                      "independent_reader_max_gate": 3 / 255,
                      "source_reconstruction": native_metrics(common, rd),
                      "source_reference": "preprocessed input RGB, not exact native renderer output"}
            replay["source_gt_canvas"] = "saved official RGB-only center crop" if name == "SA_RGB" else "saved official known-prior crop"
            replay["native_intrinsics_source"] = "saved export_camera_intrs (not the geometrically corrected common-target K)"
            records[name][stem] = replay
            Image.fromarray(np.uint8(np.clip(rd, 0, 1) * 255)).save(out / f"{stem}_{name}_native.png")
    dump(out / "sa_native_replay.json", {"status": "passed_two_reader_replay_not_target_acceptance",
                                        "checks": records})
    if device == "cuda":
        import torch
        torch.cuda.synchronize()
        print("NATIVE_REPLAY_COMPLETE", pack["scene"], flush=True)
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--official-repo", default=str(Path.home() / "repos/SplatAnything"))
    args = ap.parse_args()
    root = Path(args.protocol_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(Path(args.official_repo).resolve()))
    from src.utils.scene_io import preprocess_view
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    primary_review = json.loads((root / "roi_primary_review.json").read_text(encoding="utf-8"))
    primary_points = json.loads((root / "roi_primary_probe.json").read_text(encoding="utf-8"))
    for stem, q in primary_review["views"].items():
        detected = primary_points[stem]["points"] is not None
        if detected != (q["status"] == "accepted"):
            raise RuntimeError("Primary ROI review does not match detector record: " + stem)
    review = json.loads((root / "roi_review.json").read_text(encoding="utf-8"))
    if not review["review_status"].startswith("visual_review_complete"):
        raise RuntimeError("ROI visual review gate has not passed")
    for pack in protocol["packages"]:
        for f in pack["frames"]:
            if sha(f["processed_path"]) != f["processed_sha256"]:
                raise RuntimeError("Frozen source/target PNG hash mismatch")
    report = {"schema": "n1_real_appearance_v1", "status": "running",
              "protocol_sha256": sha(root / "protocol.json"),
              "roi_review_sha256": sha(root / "roi_review.json"),
              "primary_roi_review_sha256": sha(root / "roi_primary_review.json"),
              "primary_roi_points_sha256": sha(root / "roi_primary_probe.json"),
              "geometry": {"independent_truth": False, "metric_error_reported": False},
              "limitations": ["near-static, not rigid/expression ground truth", "camera prior is not independent metric geometry",
                              "tight face is not skin/hair segmentation", "Zip native width=378, height=504 versus SA 756 square; exact equality is original source PNGs only",
                              "primary uses RGB-only centered crop; known-pose auxiliary has its own acquisition-K crop, target metrics and ROI",
                              "raw147 face ROI unavailable; only three of four targets have face scores"],
              "packages": {}, "native_validation": {}}
    for pack in protocol["packages"]:
        print("PACKAGE", pack["scene"], flush=True)
        methods = model_setup(pack, root, args.device, args.official_repo)
        src_prior = json.loads((Path(pack["source_scene"]) / "prior_camera.json").read_text(encoding="utf-8"))
        tgt_prior = json.loads((Path(pack["target_scene"]) / "prior_camera.json").read_text(encoding="utf-8"))
        prior = {c["camera_id"]: np.asarray(c["matrix"], np.float64)
                 for c in src_prior["extrinsics"] + tgt_prior["extrinsics"]}
        native = native_checks(methods, pack, out / (pack["scene"] + "_native"), args.device)
        native["Zip_FF"] = json.loads((root / "audits" / (pack["scene"] + "_zip_native_v6") /
                                       "zip_native_audit.json").read_text(encoding="utf-8"))
        if not native["Zip_FF"]["status"].startswith("passed_"):
            raise RuntimeError("Zip native PLY replay gate has not passed: " + pack["scene"])
        if native["Zip_FF"]["ply_sha256"] != methods["Zip_FF"]["ply_sha256"]:
            raise RuntimeError("Zip native audit is not the requested PLY")
        for item in native["Zip_FF"]["source_images"]:
            if sha(item["path"]) != item["sha256"]:
                raise RuntimeError("Zip native audit source hash mismatch")
        if native["Zip_FF"]["saved_c2w"] != np.load(methods["Zip_FF"]["result"] / "predicted_cameras.npz")["c2w"].tolist():
            raise RuntimeError("Zip native audit camera mismatch")
        for name, m in methods.items():
            dump(out / f"{pack['scene']}_{name}_source_alignment.json", {
                "s_R_T": list(m["sim3"]), "source_camera_residuals": m["source_residual"],
                "condition": m["condition"], "track": m["formal_track"],
                "ply_sha256": m["ply_sha256"], "weight_sha256": m["weight_sha256"],
                "gaussians": len(m["g"]["means"]), "native_size": m.get("native_size", [756, 756]),
                "native_eps2d": m["native_eps"]})
        report["native_validation"][pack["scene"]] = native
        scene_out = out / pack["scene"]
        scene_out.mkdir()
        shared_sources = shared_source_checks(methods, pack, out, args.device)
        views = {}
        target_ref = {}
        for track, group in (("primary", ("SA_RGB", "Zip_FF")), ("auxiliary", ("SA_pose_aux",))):
            for f in pack["frames"]:
                if f["split"] != "target":
                    continue
                stem = f["camera_id"]
                m = methods[group[0]]
                prep = m["evaluation_geometry"]
                Image.fromarray(np.uint8(np.clip(M21.official_preprocess_rgb(
                    f["processed_path"], prep), 0, 1) * 255)).save(
                        scene_out / f"{stem}_{track}_target.png")
                gt = M21.official_preprocess_rgb(f["processed_path"], prep)
                ref, K_ref = preprocess_view(
                    Image.open(f["processed_path"]), m["crop_preprocess_K"], (756, 756))
                if np.max(np.abs(ref.moveaxis(0, -1).numpy() - gt)) > 1e-6:
                    raise RuntimeError("Common target RGB differs from official preprocessing")
                q = (primary_review["views"] if track == "primary" else review["views"][pack["scene"]])[stem]
                if track == "primary":
                    points = primary_points[stem]["points"]
                else:
                    points = f["landmark_qa"].get("points")
                roi = None
                if q["status"] == "accepted":
                    if points is None:
                        raise RuntimeError("Accepted ROI has no landmarks: " + stem)
                    roi = M21.region_masks(np.asarray(points), gt.shape[:2])["face_tight_landmark"]
                elif q["status"] != "unavailable" or points is not None:
                    raise RuntimeError("Unavailable/pending ROI cannot enter scoring: " + stem)
                rendered = []
                result = {}
                for name in group:
                    m = methods[name]
                    camera = target_camera(m, np.asarray(prior[stem], np.float64))
                    rd, alpha, _ = M22.render(m["gpu"], m["g"]["degree"], camera,
                                               prep["K_reference"], 756, args.device, .01)
                    values = {"full": metrics(gt, rd, alpha),
                              "lowpass_sigma2_full": metrics(cv2.GaussianBlur(gt, (0, 0), 2.),
                                                             cv2.GaussianBlur(rd, (0, 0), 2.), alpha),
                              "face_tight": metrics(gt, rd, alpha, roi) if roi is not None else {
                                  "status": "N/A", "reason": "Track-specific FaceMesh unavailable; no approximate ROI substituted"}}
                    result[name] = values
                    rendered.append((name, rd, alpha))
                    Image.fromarray(np.uint8(np.clip(rd, 0, 1) * 255)).save(
                        scene_out / f"{stem}_{track}_{name}.png")
                    Image.fromarray(np.uint8(np.clip(alpha, 0, 1) * 255)).save(
                        scene_out / f"{stem}_{track}_{name}_alpha.png")
                    print(pack["scene"], track, stem, name, values, flush=True)
                comparison(scene_out, f"{stem}_{track}", gt, rendered, roi)
                views.setdefault(stem, {"raw_frame_idx": f["raw_frame_idx"], "methods": {},
                                        "evaluation_tracks": {}})
                views[stem]["methods"].update(result)
                views[stem]["evaluation_tracks"][track] = {
                    "roi_status": q["status"], "face_roi_pixels": int(roi.sum()) if roi is not None else None,
                    "roi_scope": "tight landmark oval, may include hairline/hair; not whole head",
                    "crop": prep, "target_K": prep["K_reference"]}
                target_ref.setdefault(track, {})[stem] = {
                    "raw_frame_idx": f["raw_frame_idx"], "roi_status": q["status"],
                    "crop": prep, "target_K": prep["K_reference"],
                    "official_scene_io_return_K": K_ref.numpy(),
                    "K_basis": "actual target acquisition K transformed by identical official crop/resize operators; near-centered square crop is preregistered, scene_io returned K and legacy CPU double-offset K are excluded"}
        report["packages"][pack["scene"]] = {
            "package_id": pack["package_id"], "staticness": pack["near_static_qa"],
            "source_raw_frame_idx": pack["source_raw_frame_idx"],
            "target_raw_frame_idx": pack["target_raw_frame_idx"],
            "shared_source_projection_diagnostics": shared_sources,
            "target_reference": target_ref,
            "model_artifacts": {name: {"condition": m["condition"], "track": m["formal_track"],
                                       "gaussians": len(m["g"]["means"]), "ply_sha256": m["ply_sha256"],
                                       "weight_sha256": m["weight_sha256"], "run_sha256": m["run_sha256"],
                                       "config_sha256": m["config_sha256"],
                                       "source_alignment": m["source_residual"],
                                       "native_size": m.get("native_size", [756, 756]),
                                       "native_eps2d": m["native_eps"]} for name, m in methods.items()},
            "views": views,
            "methods": {name: {"condition": m["condition"], "track": m["formal_track"],
                               "gaussians": len(m["g"]["means"]), "source_alignment": m["source_residual"]}
                        for name, m in methods.items()}}
        del methods
    summary = {}
    for scene, pack in report["packages"].items():
        summary[scene] = {}
        for name in pack["methods"]:
            records = [v["methods"][name] for v in pack["views"].values()]
            summary[scene][name] = {
                "target_full_psnr_mean": float(np.mean([r["full"]["psnr"] for r in records])),
                "target_full_mae_mean": float(np.mean([r["full"]["mae"] for r in records])),
                "target_full_alpha_coverage05_mean": float(np.mean([r["full"]["coverage_05"] for r in records])),
                "target_lowpass_sigma2_psnr_mean": float(np.mean([r["lowpass_sigma2_full"]["psnr"] for r in records])),
                "target_face_alpha_coverage05_mean": float(np.mean([r["face_tight"]["coverage_05"] for r in records
                                                                  if r["face_tight"].get("coverage_05") is not None])) if any(
                    r["face_tight"].get("coverage_05") is not None for r in records) else None,
                "face_n_valid": sum(r["face_tight"].get("psnr") is not None for r in records),
                "target_face_psnr_mean": float(np.mean([r["face_tight"]["psnr"] for r in records
                                                       if r["face_tight"].get("psnr") is not None])) if any(
                    r["face_tight"].get("psnr") is not None for r in records) else None}
    report["summary"] = summary
    report["software"] = {"sa_repo": subprocess.check_output(["git", "-C", args.official_repo, "rev-parse", "HEAD"],
                                                           text=True).strip(),
                          "zip_repo": protocol["software"]["zip_repo"]}
    if report["software"]["sa_repo"] != protocol["software"]["sa_repo"]:
        raise RuntimeError("SA official repo differs from preregistration")
    report["code_sha256"] = {str(HERE / f): sha(HERE / f) for f in
                             ("23_n1_prepare_real.py", "24_zip_real_audit.py", "25_n1_real_eval.py",
                              "03_render_eval.py", "15_face_target_eval.py", "21_geometry_audit_v2.py",
                              "22_n1_head_compare.py")}
    dump(out / "audit_command.json", {"argv": sys.argv, "python": sys.executable,
                                      "official_repo": str(Path(args.official_repo).resolve())})
    rows = []
    for scene, pack in report["packages"].items():
        for stem, view in pack["views"].items():
            for name, methods in view["methods"].items():
                for region, values in methods.items():
                    if values.get("status") == "N/A":
                        continue
                    rows.append({"scene": scene, "target": stem, "raw_frame_idx": view["raw_frame_idx"],
                                 "method": name, "track": pack["methods"][name]["track"], "region": region,
                                 **{k: v for k, v in values.items() if k in ("n", "psnr", "mae", "alpha_mean", "coverage_05")}})
    with open(out / "per_view_metrics.csv", "w", encoding="utf-8", newline="\n") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report["status"] = "two packages, four held-out targets; appearance-only evidence; N2 remains frozen pending visual review"
    dump(out / "n1_real_eval.json", report)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
