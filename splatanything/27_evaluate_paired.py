#!/usr/bin/env python3
"""27_evaluate_paired.py -- 真实三图/五图的同 target、同 ROI 配对验收。

算法：每臂仅由原始三锚点估计一个 Sim(3)，把 target 刚体相机逆回模型
坐标。RGB-only / Zip 的 model→pkg 拟合把 ED 乘以 s 回米制；
known-pose 的 pkg→model 拟合把 ED 除以 s。深度只在有效 GT 和预登记 ROI 中评分；
另给共同有效像素上的配对分数。真人没有深度真值，不生成毫米几何。
无 target 拟合、无 TokenBA、无训练、无 PLY 拼接。
"""
import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M03 = load("paired_render03", "03_render_eval.py")
M13 = load("paired_depth13", "13_face_select.py")
M15 = load("paired_sim15", "15_face_target_eval.py")
M21 = load("paired_geometry21", "21_geometry_audit_v2.py")
M22 = load("paired_render22", "22_n1_head_compare.py")
M25 = load("paired_real25", "25_n1_real_eval.py")


SHA_CACHE = {}
CUDA_WARMED = False
WARMUP_GAUSSIANS = 20000
EVAL_CODE_VERSION = "paired_eval_v8_raw_id_landmarks"


def mark_cuda_warm():
    global CUDA_WARMED
    CUDA_WARMED = True
    return CUDA_WARMED


def sha(path):
    path = Path(path)
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in SHA_CACHE:
        SHA_CACHE[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return SHA_CACHE[key]


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(M21.jsonable(data), f, ensure_ascii=False, indent=2)


def rgb_sha(path):
    return hashlib.sha256(Image.open(path).convert("RGB").tobytes()).hexdigest()


def image(path):
    return np.asarray(Image.open(path).convert("RGB"), np.float32) / 255


def font(size):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def metrics(gt, rd, alpha, mask, lp=None):
    if int(mask.sum()) < 32:
        return {"n": int(mask.sum()), "status": "N/A"}
    d = gt[mask].astype(np.float64) - rd[mask].astype(np.float64)
    out = {"n": int(mask.sum()), "psnr": float(-10 * np.log10(max(np.mean(d*d), 1e-12))),
           "mae": float(np.mean(np.abs(d))), "alpha_mean": float(alpha[mask].mean()),
           "coverage_05": float((alpha[mask] > .5).mean())}
    if lp is not None:
        ys, xs = np.where(mask)
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max()+1, ys.max()+1
        import torch
        def crop(a):
            im = Image.fromarray(np.uint8(np.clip(a[y0:y1, x0:x1], 0, 1) * 255))
            im = im.resize((128, 128), Image.Resampling.LANCZOS)
            return torch.from_numpy(np.asarray(im, np.float32)/255).permute(2, 0, 1)[None].cuda()*2-1
        with torch.no_grad():
            out["lpips_crop128"] = float(lp(crop(gt), crop(rd)).item())
    return out


def depth_metrics(ed, gt, mask, paired=None):
    selected = mask & (ed > 1e-5)
    if paired is not None:
        selected &= paired
    err = (ed - gt)[selected]
    if len(err) < 32:
        return {"n": int(len(err)), "status": "N/A"}
    return {"n": int(len(err)), "mae_mm": float(np.abs(err).mean()*1000),
            "bias_mm": float(err.mean()*1000),
            "p90_abs_mm": float(np.percentile(np.abs(err), 90)*1000),
            "valid_fraction": float(selected.sum()/max(mask.sum(), 1))}


def landmark_raw_ids(path):
    """两个冻结版本的 key 不同，只以记录内的 raw id 为物理帧依据。"""
    views = json.loads(Path(path).read_text(encoding="utf-8"))["views"]
    landmarks = {}
    for slot, value in views.items():
        raw_id = int(value["id"])
        if int(slot.rsplit("_", 1)[1]) != raw_id or raw_id in landmarks:
            raise RuntimeError("Landmark slot/raw ID mismatch or duplicate: " + slot)
        landmarks[raw_id] = value
    return landmarks


def model_meta(pack, root, arm, condition, anchors, device):
    name = pack["name"] + "_" + arm
    if condition.startswith("SA"):
        mode = "rgb" if condition == "SA_RGB" else "pose"
        result = root / "models" / ("sa_" + mode) / name
        run = json.loads((result / "run.json").read_text(encoding="utf-8"))
        expected = [str(Path(pack[arm + "_scene"]) / "images" / f)
                    for f in sorted(x.name for x in (Path(pack[arm + "_scene"]) / "images").glob("*.png"))]
        if run["images"] != expected or run["resolution"] != [756, 756] or \
                bool(run["pose_condition"]) != (condition == "SA_pose_aux"):
            raise RuntimeError("SA images/resolution/condition differ from frozen arm: " + name)
        cams, K, ids = M03.load_cameras(result / "cameras.json")
        if ids != [Path(x).stem for x in expected]:
            raise RuntimeError("SA camera ordering mismatch")
        from src.utils.scene_io import preprocess_view
        prior = json.loads((Path(pack[arm + "_scene"]) / "prior_camera.json").read_text(encoding="utf-8"))
        for i, path in enumerate(expected):
            view, expected_K = preprocess_view(
                Image.open(path),
                np.asarray(next(c["matrix"] for c in prior["intrinsics"]
                                if c["camera_id"] == Path(path).stem), np.float32)
                if condition == "SA_pose_aux" else None, (756, 756))
            actual = image(result / "rgb" / Path(path).name)
            if np.max(np.abs(view.moveaxis(0, -1).numpy() - actual)) > 1e-6:
                raise RuntimeError("SA official input preprocess mismatch: " + Path(path).stem)
            if condition == "SA_pose_aux" and np.max(np.abs(expected_K.numpy() - K[i])) > 1e-4:
                raise RuntimeError("SA known-pose export K differs from forward input")
        prior = json.loads((Path(pack[arm + "_scene"]) / "prior_camera.json").read_text(encoding="utf-8"))
        priors = {c["camera_id"]: np.asarray(c["matrix"], np.float64) for c in prior["extrinsics"]}
        if condition == "SA_pose_aux":
            anchor_frames = [r for r in pack["frames"] if r["split"] == "anchor"]
            if len(anchor_frames) != 3:
                raise RuntimeError("Known-pose fit requires exactly three original anchors")
            anchor_slots = {r["scene_slots"][arm] for r in anchor_frames}
            if len(anchor_slots) != 3 or not anchor_slots <= set(ids):
                raise RuntimeError("Known-pose arm lost original anchors")
            prior_by_id = {c["camera_id"]: c for c in prior["extrinsics"]}
            fit = M15.estimate_sim3(result, Path(pack[arm + "_scene"]),
                                    source_ids=sorted(anchor_slots))
            s, R, T = fit[:3]
            fit = (s, R, T, {"rule": "three-anchor rigid camera fit with first-source orientation; pkg->model; depth /= s",
                             "source_only_fit": fit[3],
                             "anchor_raw_ids": [r["raw_id"] for r in anchor_frames]})
            residual = {"orientation_residual_deg": {}, "export_camera_max_abs": {},
                        "source_only_fit": fit[3]}
            for fr in anchor_frames:
                slot = fr["scene_slots"][arm]
                expected_cam = M21.model_camera(
                    np.asarray(prior_by_id[slot]["matrix"], np.float64), *fit[:3])
                exported = cams[ids.index(slot)]
                residual["orientation_residual_deg"][slot] = M22.angle(
                    (R @ np.asarray(prior_by_id[slot]["matrix"], np.float64)[:3, :3]).T @ exported[:3, :3])
                residual["export_camera_max_abs"][slot] = float(np.max(np.abs(expected_cam-exported)))
            if abs(np.linalg.det(R)-1) > 1e-4 or max(residual["export_camera_max_abs"].values()) > 2e-4:
                raise RuntimeError("Known-pose three-anchor rigid fit failed")
        else:
            fit = M22.fit_sim3(np.stack([priors[s][:3, 3] for s in ids if s in anchors]),
                               np.stack([cams[i, :3, 3] for i in range(len(ids)) if ids[i] in anchors]))
            fit = (*fit[:3], {"rule": "three-anchor source-only center Umeyama; model->pkg; depth *= s"})
            source_ids = [s for s in ids if s in anchors]
            residual = M22.source_camera_residuals(
                np.stack([priors[s] for s in source_ids]),
                np.stack([cams[ids.index(s)] for s in source_ids]),
                fit, source_ids)
        config = (result / "config.yaml").read_text(encoding="utf-8")
        if "enable_cond: " + ("true" if condition == "SA_pose_aux" else "false") not in config:
            raise RuntimeError("SA saved config condition mismatch: " + name)
        weight_path = next(line.split("SplatAnything_pretrained:", 1)[1].strip()
                           for line in config.splitlines() if "SplatAnything_pretrained:" in line)
        geometry_weight_path = next(line.split("worldmirror2_pretrained:", 1)[1].strip()
                                    for line in config.splitlines() if "worldmirror2_pretrained:" in line)
        return {"condition": condition, "result": result, "ply": result / "gaussians.ply",
                "fit": fit, "residual": residual, "source_cams": dict(zip(ids, cams)),
                "source_K": dict(zip(ids, K)), "degree": 0, "eps": .01, "size": [756, 756],
                "weight_paths": [weight_path, geometry_weight_path],
                "runtime": {"run": run}}
    result = root / "models" / "zip_rgb" / name
    run = json.loads((result / "metrics.json").read_text(encoding="utf-8"))
    z = np.load(result / "predicted_cameras.npz", allow_pickle=False)
    ids = [str(x)[:-4] for x in z["names"]]
    expected = [c["camera_id"] for c in json.loads(
        (Path(pack[arm + "_scene"]) / "prior_camera.json").read_text(encoding="utf-8"))["extrinsics"]]
    if ids != expected or list(z["names"]) != [s + ".png" for s in expected]:
        raise RuntimeError("Zip camera order differs from frozen slots")
    cams = dict(zip(ids, z["c2w"]))
    prior = json.loads((Path(pack[arm + "_scene"]) / "prior_camera.json").read_text(encoding="utf-8"))
    priors = {c["camera_id"]: np.asarray(c["matrix"], np.float64) for c in prior["extrinsics"]}
    if run["token_ba"] or run["image_files"] != [s + ".png" for s in expected]:
        raise RuntimeError("Zip image order/BA protocol mismatch")
    fit = M22.fit_sim3(np.stack([priors[s][:3, 3] for s in sorted(anchors)]),
                       np.stack([cams[s][:3, 3] for s in sorted(anchors)]))
    fit = (*fit[:3], {"rule": "three-anchor source-only center Umeyama; model->pkg; depth *= s"})
    residual = M22.source_camera_residuals(np.stack([priors[s] for s in sorted(anchors)]),
                                           np.stack([cams[s] for s in sorted(anchors)]), fit, sorted(anchors))
    native_K = {}
    for stem, row in zip(ids, z["cameras"]):
        w, h, fx, fy, cx, cy = map(float, row)
        original_size = Image.open(Path(pack[arm + "_scene"]) / "images" / (stem + ".png")).size
        if (round(w), round(h)) != original_size:
            raise RuntimeError("Zip native camera size differs from source image")
        f = max(378/w, 504/h)
        native_K[stem] = np.array([[fx*f, 0, cx*f], [0, fy*f, cy*f], [0, 0, 1]], np.float32)
        expected_size = [round(w*f), round(h*f)]
        if expected_size != run["model_input_size"]:
            raise RuntimeError("Zip native aspect ratio differs from saved model input")
    return {"condition": condition, "result": result, "ply": result / "gaussians_feedforward.ply",
            "fit": fit, "residual": residual, "source_cams": cams, "source_K": native_K,
            "eps": .3, "size": run["model_input_size"], "runtime": run,
            "weight_paths": [run["weight_path"]]}


def validate_provenance(m, protocol, root, pack):
    """把真实前向配置与权重链逐一对账；记录来源在首组推理后补记。"""
    provenance_path = root / "run_provenance.json"
    if not provenance_path.is_file():
        raise RuntimeError("Missing first-launch software/weight provenance")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    runtime = m["runtime"].get("run", m["runtime"])
    if m["condition"].startswith("SA"):
        repo = Path.home() / "repos/SplatAnything"
        commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        if commit != provenance["official_commits"][str(repo)]:
            raise RuntimeError("SA official commit differs from first launch")
        if runtime["ckpt_path"] is not None or runtime["geometry_ckpt_path"] is not None:
            raise RuntimeError("Paired SA requires frozen default config weights")
        if runtime["sky_seg"] or runtime["external_depth_dir"] is not None or \
                runtime["no_depth_rgb"] or runtime["no_cam_align"]:
            raise RuntimeError("Unexpected SA auxiliary input or altered alignment")
        if runtime["input_path"] != pack[("source5" if len(runtime["images"]) == 5 else "source3") + "_scene"]:
            raise RuntimeError("SA input scene differs from frozen arm")
        config_sha = sha(m["result"] / "config.yaml")
        if "infer_opacity_thresh: 0.05" not in (m["result"] / "config.yaml").read_text(encoding="utf-8") or \
                "gs_voxel_size: 0.002" not in (m["result"] / "config.yaml").read_text(encoding="utf-8"):
            raise RuntimeError("SA export config differs from preregistered inference settings")
    else:
        repo = Path.home() / "repos/ZipSplat"
        if runtime["official_commit"] != provenance["official_commits"][str(repo)] or \
                runtime["num_views"] != len(runtime["image_files"]):
            raise RuntimeError("Zip commit/native input protocol mismatch")
        config_sha = sha(m["result"] / "predicted_cameras.npz")
    weight_sha256 = {p: sha(p) for p in m["weight_paths"]}
    if weight_sha256 != {p: provenance["weights"][p] for p in m["weight_paths"]}:
        raise RuntimeError("Model weight hash differs from first launch")
    arm = "source5" if len(runtime["image_files"] if not m["condition"].startswith("SA") else runtime["images"]) == 5 else "source3"
    return {"provenance_timing": "recorded after first model launch and before evaluation; not claimed as pre-first-launch",
            "provenance_sha256": sha(provenance_path),
            "weight_sha256": weight_sha256,
            "run_sha256": sha(m["result"] / ("run.json" if m["condition"].startswith("SA") else "metrics.json")),
            "config_sha256": config_sha,
            "official_commit": commit if m["condition"].startswith("SA") else runtime["official_commit"],
            "source_prior_sha256": sha(Path(pack[arm + "_scene"]) / "prior_camera.json")}


def native_replay(m, out, device):
    """新臂先过原生 PLY 双读取器回放，再进入 target 评分。"""
    out.mkdir(exist_ok=False, parents=True)
    g = M22.load_ply(m["ply"], "wxyz")
    gpu = M22.to_gpu(g, device)
    m["g"], m["gpu"], m["degree"] = g, gpu, g["degree"]
    checks = {}
    for stem, camera in m["source_cams"].items():
        rd, al, _ = M22.render(gpu, m["degree"], camera, m["source_K"][stem],
                               m["size"][1], device, m["eps"], height=m["size"][0])
        reference = image(m["result"] / "renders" / f"{list(m['source_cams']).index(stem):03d}.png") if m["condition"] == "Zip_FF" else None
        if m["condition"] == "Zip_FF":
            delta = float(np.abs(rd-reference).max())
            if delta > 3/255:
                raise RuntimeError("Zip native PLY replay failed: " + stem)
            checks[stem] = {"official_vs_ply_rgb_max_abs": delta,
                            "official_vs_ply_rgb_mae": float(np.abs(rd-reference).mean())}
        else:
            base = M03.load_ply(m["ply"])
            ref, ref_al = M03.render(base, camera, m["source_K"][stem], m["size"][1],
                                     m["size"][0], background=0, device=device)
            delta = float(np.abs(rd-ref[0]).max())
            ad = float(np.abs(al-ref_al[0, ..., 0]).max())
            if delta > 3/255 or ad > 3/255:
                raise RuntimeError("SA independent native PLY replay failed: " + stem)
            checks[stem] = {"independent_reader_rgb_max_abs": delta,
                            "independent_reader_alpha_max_abs": ad}
        Image.fromarray(np.uint8(np.clip(rd, 0, 1)*255)).save(out / (stem + "_native.png"))
    m["native_replay"] = checks
    dump(out / "native_replay.json", {"status": "passed_native_only_not_target_acceptance",
                                     "checks": checks, "ply_sha256": sha(m["ply"])})


def source_projection_checks(m, pack, out, device):
    """原三锚点投影；源相机从 source_scene 读取，不能借用 target slot。"""
    source = json.loads((Path(m["source_scene"]) /
                         "prior_camera.json").read_text(encoding="utf-8"))
    m["camera_validation"] = validate_physical_cameras(pack)
    priors = {c["camera_id"]: np.asarray(c["matrix"], np.float64)
              for c in source["extrinsics"]}
    intrinsics = {c["camera_id"]: np.asarray(c["matrix"], np.float32)
                  for c in source["intrinsics"]}
    anchor_frames = [f for f in m["frames"] if f["split"] == "anchor"]
    if len(anchor_frames) != 3:
        raise RuntimeError("Source projection check requires three anchors")
    out.mkdir(parents=True, exist_ok=False)
    records = {}
    for fr in anchor_frames:
        stem = fr["scene_slots"][m["arm"]]
        physical_K = intrinsics[stem]
        layout_K = physical_K if m["condition"] == "SA_pose_aux" else None
        prep = M25.evaluation_geometry(layout_K, tuple(fr["processed_wh"]), physical_K)
        gt = M21.official_preprocess_rgb(fr["path"], prep)
        if m["condition"] == "SA_pose_aux":
            if np.max(np.abs(m["source_K"][stem] - prep["K_reference"])) > 1e-4:
                raise RuntimeError("Known-pose source projection K differs from frozen native K")
            cam = m["source_cams"][stem]
            expected_cam = M21.model_camera(priors[stem], *m["fit"][:3])
            if np.max(np.abs(expected_cam - cam)) > 2e-6:
                raise RuntimeError("Known-pose physical camera/export regression failed: " + stem)
        else:
            s, R, T, _ = m["fit"]
            cam = M21.model_camera(priors[stem], 1/s, R.T, -(R.T@T)/s)
        rd, al, _ = M22.render(m["gpu"], m["degree"], cam, prep["K_reference"],
                               756, device, .01)
        values = {"rgb_psnr": float(M03.psnr(gt, rd)),
                  "rgb_mae": float(np.abs(gt-rd).mean()),
                  "alpha_coverage_05": float((al > .5).mean()),
                  "raw_id": fr["raw_id"], "slot": stem,
                  "scope": "source-only physical-camera projection; not held-out quality or independent geometry truth"}
        records[stem] = values
        Image.fromarray(np.uint8(np.clip(rd, 0, 1)*255)).save(out / (stem + "_shared_source.png"))
    dump(out / "source_projection.json", {
        "records": records, "anchor_raw_ids": [f["raw_id"] for f in anchor_frames],
        "camera_validation": m["camera_validation"]})
    return {"status": "source_projection_checks_complete",
            "records_path": str(out / "source_projection.json"), "records": records,
            "anchor_raw_ids": [f["raw_id"] for f in anchor_frames]}


def repeatability_check(m, out, device):
    """同一 PLY/camera/K 的两次独立渲染；不是跨场景或模型重复前向。"""
    import torch
    if m["condition"].startswith("SA"):
        base = M03.load_ply(m["ply"])
        render = lambda: M13.render_expected_depth(
            base, m["repeat_camera"], m["repeat_K"], 756, 756,
            device=device, eps2d=.01)
        mark_cuda_warm()
    else:
        render = lambda: M22.render(m["gpu"], m["degree"], m["repeat_camera"],
                                    m["repeat_K"], 756, device, .01)[1:]
        mark_cuda_warm()
    start = time.perf_counter()
    ed1, al1 = render()
    first_s = time.perf_counter() - start
    start = time.perf_counter()
    ed2, al2 = render()
    second_s = time.perf_counter() - start
    if m["condition"].startswith("SA"):
        ed1, al1, ed2, al2 = ed1[0], al1[0, ..., 0], ed2[0], al2[0, ..., 0]
    common = (al1 > .01) & (al2 > .01) & (ed1 > 1e-5) & (ed2 > 1e-5)
    delta_ed = np.abs(ed1 - ed2)
    record = {"schema": "same_ply_camera_repeat_v1", "repeats": 2,
              "same_ply_sha256": sha(m["ply"]), "same_camera_and_K": True,
              "ed_unit": "model units, before metric scale", "alpha_threshold": .01,
              "warmup_separate_gaussians": WARMUP_GAUSSIANS,
              "max_abs_ed": float(delta_ed.max()),
              "max_abs_alpha": float(np.abs(al1-al2).max()),
              "mean_abs_ed_common": float(delta_ed[common].mean()) if common.any() else None,
              "p99_9_abs_ed_common": float(np.percentile(delta_ed[common], 99.9)) if common.any() else None,
              "common_pixels": int(common.sum()),
              "fixed_noise_tolerance": {"ed_max_model_units": 1e-4,
                                        "ed_mean_common_model_units": 2e-5,
                                        "alpha_max": 1e-5},
              "first_render_seconds_after_warmup": first_s, "second_render_seconds": second_s,
              "torch_peak_allocated_mib": float(torch.cuda.max_memory_allocated(device)/2**20),
              "scope": "render variability only; not model inference or a pass on target geometry"}
    record["pass"] = bool(record["max_abs_ed"] <= 1e-4 and
                           (record["mean_abs_ed_common"] or 0) <= 2e-5 and
                           record["max_abs_alpha"] <= 1e-5)
    dump(out, record)
    return record


def metric_depth_scale(fit, pkg_to_model=False):
    """世界相似变换 X_b=s R X_a+T 的深度倍率恒为 s。

    正式 common-camera 渲染逆变换相机时，深度要用逆世界变换的倍率：
    model→pkg 拟合为 s；pkg→model 拟合为 1/s。二者都不是额外的自由尺度。
    """
    return 1.0 / float(fit[0]) if pkg_to_model else float(fit[0])


def physical_camera(pack, role, stem):
    """读取与 GT 一致的物理相机；never alter frozen input or exported cameras.

    corrected_opencv 的 Daven manifest 已通过 look-at 验证，不能再施加 CRC；
    Rosemary 冻结 prior 已含一次 CRC。这里只核对，不用 target 拟合相机。
    """
    scene = json.loads((Path(pack[role + "_scene"]) /
                        "prior_camera.json").read_text(encoding="utf-8"))
    return np.asarray(next(c["matrix"] for c in scene["extrinsics"]
                           if c["camera_id"] == stem), np.float64)


def validate_physical_cameras(pack):
    """七个 raw_id 分角色逐一核验，slot 只负责定位，不负责决定物理帧。"""
    if pack["kind"] != "static_synthetic":
        return {"status": "real capture uses frozen opencv_c2w prior"}
    manifest = json.loads((Path(pack["pkg"]) / "cameras.json").read_text(encoding="utf-8"))
    by_id = {int(c["id"]): c for c in manifest["cameras"]}
    target = np.asarray(json.loads((Path(pack["pkg"]) / "meta.json").read_text(
        encoding="utf-8"))["target"], np.float64)
    K = np.asarray(manifest["intrinsics"]["K"], np.float64)
    flip = np.diag([1., -1., -1.])
    records = {}
    for fr in pack["frames"]:
        for role, stem in fr["scene_slots"].items():
            raw_id = fr["raw_id"]
            if int(stem.rsplit("_", 1)[1]) != raw_id:
                raise RuntimeError("Physical camera slot/raw_id mismatch: " + stem)
            expected = np.asarray(by_id[raw_id]["transform_cam2world"], np.float64).copy()
            if pack["camera_convention"] == "manifest_requires_one_crc":
                expected[:3, :3] = flip @ expected[:3, :3] @ flip
            elif pack["camera_convention"] != "corrected_opencv":
                raise RuntimeError("Unknown static camera convention")
            actual = physical_camera(pack, role, stem)
            pc = (target - actual[:3, 3]) @ actual[:3, :3]
            uv = K[:2, :2] @ pc[:2] / pc[2] + K[:2, 2]
            offset = float(np.linalg.norm(uv - K[:2, 2]))
            error = float(np.max(np.abs(actual - expected)))
            if error > 1e-8 or pc[2] <= 0 or offset > .01:
                raise RuntimeError("Static physical camera/look-at contract failed: " + stem)
            records[role + "/" + stem] = {
                "raw_id": raw_id, "prior_vs_expected_max_abs": error,
                "look_at_offset_from_principal_px": offset,
                "look_at_depth_m": float(pc[2])}
    return {"status": "passed_manifest_mapping_and_look_at",
            "rule": "Daven raw OpenCV; Rosemary prior=C@R_manifest@C once; no extra CRC",
            "records": records}


def target_setup(pack, arm, condition, prep, priors, fit):
    """同一个物理相机分别进入两种已冻结的模型坐标系，不接触 target 参数。"""
    target_cam = physical_camera(pack, "target", priors)
    if condition == "SA_pose_aux":
        return M21.model_camera(target_cam, *fit[:3])
    s, R, T, _ = fit
    return M21.model_camera(target_cam, 1/s, R.T, -(R.T@T)/s)


def comparison(path, rows):
    width, height, label = 310, 310, 32
    canvas = Image.new("RGB", (width*len(rows), (height+label)*max(len(r[1]) for r in rows)), "white")
    draw = ImageDraw.Draw(canvas)
    for j, (name, entries) in enumerate(rows):
        for i, (kind, im) in enumerate(entries):
            tile = Image.fromarray(np.uint8(np.clip(im, 0, 1)*255)).resize((width, height))
            canvas.paste(tile, (j*width, i*(height+label)))
            draw.text((j*width+8, i*(height+label)+height+7), name+" | "+kind,
                      fill="#1b2a41", font=font(16))
    canvas.save(path)


def evaluate_gt(pack, methods, root, out, lp, device):
    pkg = Path(pack["pkg"])
    manifest = json.loads((pkg / "cameras.json").read_text(encoding="utf-8"))
    by_id = {int(c["id"]): c for c in manifest["cameras"]}
    K = np.asarray(manifest["intrinsics"]["K"], np.float64)
    wh = (int(manifest["intrinsics"]["width"]), int(manifest["intrinsics"]["height"]))
    prep = M21.official_preprocess_geometry(K, wh, (756, 756))
    for arm in ("source3", "source5"):
        source = json.loads((Path(pack[arm + "_scene"]) / "prior_camera.json").read_text(encoding="utf-8"))
        for i, stem in enumerate([c["camera_id"] for c in source["extrinsics"]]):
            actual_K = methods[arm]["SA_pose_aux"]["source_K"][stem]
            actual_cam = methods[arm]["SA_pose_aux"]["source_cams"][stem]
            expected_cam = M21.model_camera(
                np.asarray(source["extrinsics"][i]["matrix"], np.float64),
                *methods[arm]["SA_pose_aux"]["fit"][:3])
            if np.max(np.abs(actual_K - prep["K_reference"])) > 1e-4 or \
                    np.max(np.abs(actual_cam - expected_cam)) > 2e-6:
                raise RuntimeError("Static known-pose camera/K differs from fitted export camera: " + stem)
    landmarks = landmark_raw_ids(pack["landmarks"])
    for raw_id in pack["target_ids"]:
        if int(landmarks[raw_id]["id"]) != raw_id:
            raise RuntimeError("Landmark raw ID differs from frozen target: " + str(raw_id))
    target_prior = json.loads((Path(pack["target_scene"]) / "prior_camera.json").read_text(encoding="utf-8"))
    exclusions = {}
    if pack["name"] == "rosemary":
        qa = json.loads(Path("/mnt/d/output/results/splatanything_results/geometry_audit_v2_20261122/roi_visual_qa.json").read_text(encoding="utf-8"))
        qa_by_raw = {int(v["id"] if "id" in v else k.rsplit("_", 1)[1]): v
                     for k, v in qa["views"].items()}
    else:
        qa = json.loads((root / "roi_qa.json").read_text(encoding="utf-8"))
        qa_by_raw = {int(v["id"] if "id" in v else k): v for k, v in qa["views"].items()}
    report = {"prep": prep, "targets": {}}
    for tidx, fr in enumerate([x for x in pack["frames"] if x["split"] == "target"]):
        stem = target_prior["extrinsics"][tidx]["camera_id"]
        mc = by_id[fr["raw_id"]]
        gt = M21.official_preprocess_rgb(pkg / mc["file_rgb"], prep)
        dep_native, valid_native = M21.load_depth(pkg / mc["file_depth"])
        _, al_native = M21.load_rgb(pkg / mc["file_rgb"], wh)
        valid_native &= al_native > .999
        gt_depth = M21.remap_nearest(np.where(valid_native, dep_native, 0), prep)
        valid = M21.remap_nearest(valid_native, prep)
        landmark_view = landmarks.get(fr["raw_id"])
        if landmark_view is None:
            raise RuntimeError("No native-frame landmarks for target raw_id: " + str(fr["raw_id"]))
        roi_native = M21.region_masks(np.asarray(landmark_view["points"]), dep_native.shape)
        excluded = qa_by_raw.get(fr["raw_id"], {}).get("excluded_rois", [])
        if qa_by_raw.get(fr["raw_id"], {}).get("status") not in ("accepted", "partial"):
            raise RuntimeError("Target ROI not visually accepted: " + str(fr["raw_id"]))
        masks = {"face_tight": roi_native["face_tight_landmark"],
                 **{n: roi_native[n] for n in ("nose_tip", "nose_ala", "eye_l", "eye_r")}}
        masks = {n: M21.remap_nearest(m, prep) & valid for n, m in masks.items()
                 if n not in excluded}
        rows = []
        rendered = {}
        view = {"stem": stem, "raw_id": fr["raw_id"], "roi_pixels": {n: int(m.sum()) for n, m in masks.items()}, "methods": {}}
        for arm in ("source3", "source5"):
            for name, m in methods[arm].items():
                cam = target_setup(pack, arm, name, prep, stem, m["fit"])
                if name.startswith("SA"):
                    base = M03.load_ply(m["ply"])
                    ed, al = M13.render_expected_depth(base, cam, prep["K_reference"], 756, 756,
                                                       device=device, eps2d=.01)
                    rd, _ = M03.render(base, cam, prep["K_reference"], 756, 756, background=0, device=device)
                    rd, ed, al = rd[0], ed[0], al[0, ..., 0]
                    ed *= metric_depth_scale(m["fit"], name == "SA_pose_aux")
                else:
                    rd, al, ed = M22.render(m["gpu"], m["degree"], cam, prep["K_reference"], 756, device, .01)
                    ed *= metric_depth_scale(m["fit"])
                values = {"appearance": {n: metrics(gt, rd, al, mask, lp if n == "face_tight" else None)
                                        for n, mask in masks.items()},
                          "depth": {n: depth_metrics(ed, gt_depth, mask) for n, mask in masks.items()},
                          "coverage_whole_gt": float((al[valid] > .5).mean())}
                view["methods"].setdefault(name, {})[arm] = values
                rendered[(name, arm)] = {"ed": ed, "alpha": al, "rgb": rd}
                (out / "images").mkdir(exist_ok=True)
                Image.fromarray(np.uint8(np.clip(rd, 0, 1)*255)).save(out / "images" / f"{pack['name']}_{stem}_{name}_{arm}.png")
                entries = [("RGB", rd), ("lowpass", cv2.GaussianBlur(rd, (0, 0), 2.))]
                if masks.get("face_tight") is not None and masks["face_tight"].any():
                    yy, xx = np.where(masks["face_tight"])
                    x0, y0, x1, y1 = xx.min(), yy.min(), xx.max()+1, yy.max()+1
                    entries.append(("face", rd[y0:y1, x0:x1]))
                rows.append((name+" "+arm, entries))
        # 同一 GT、同一 ROI、共同有效深度像素的严格配对。
        for name in methods["source3"]:
            pair = {"appearance": {}, "depth": {}}
            for n, mask in masks.items():
                a = rendered[(name, "source3")]
                b = rendered[(name, "source5")]
                common = (a["ed"] > 1e-5) & (b["ed"] > 1e-5) & \
                         (a["alpha"] > .01) & (b["alpha"] > .01)
                pair["depth"][n] = {"source3": depth_metrics(a["ed"], gt_depth, mask, common),
                                    "source5": depth_metrics(b["ed"], gt_depth, mask, common),
                                    "common_ed_valid_fraction": float(common[mask].mean()) if mask.any() else 0.0}
                pa, pb = pair["depth"][n]["source3"], pair["depth"][n]["source5"]
                if pa.get("p90_abs_mm", 0) > 0 and pb.get("p90_abs_mm") is not None:
                    pair["depth"][n]["p90_relative_reduction"] = float((pa["p90_abs_mm"]-pb["p90_abs_mm"])/pa["p90_abs_mm"])
            view["methods"][name]["paired_reduction"] = pair
        comparison(out / "images" / f"{pack['name']}_{stem}_comparison.png", [("GT", [("RGB", gt)])] + rows)
        report["targets"][stem] = view
    return report


def paired_geometry_summary(report):
    """相同 target/ROI/共同有效像素下的配对；未逐图 QA 前不做晋级。"""
    summary = {}
    for cond in ("SA_RGB", "Zip_FF", "SA_pose_aux"):
        summary[cond] = {}
        for scene, pack in report["packages"].items():
            if pack.get("geometry_truth") is False:
                continue
            per_target = []
            for stem, view in pack["targets"].items():
                pair = view["methods"][cond]["paired_reduction"]["depth"]
                per_target.append({"target": stem, "regions": pair})
            valid_nose = []
            insufficient_targets = []
            for view in per_target:
                vals = [view["regions"][r]["p90_relative_reduction"]
                        for r in ("nose_tip", "nose_ala")
                        if view["regions"].get(r, {}).get("p90_relative_reduction") is not None]
                cover = {r: view["regions"].get(r, {}).get("common_ed_valid_fraction", 0.)
                         for r in ("nose_tip", "nose_ala")}
                if len(vals) != 2 or min(cover.values()) < .5:
                    insufficient_targets.append({"target": view["target"], "common_nose_support": cover,
                                                 "reason": "both frozen nose ROIs require >=50% common alpha>.01 and ED support"})
                else:
                    valid_nose.append(float(np.mean(vals)))
            face_support = [view["regions"].get("face_tight", {}).get("common_ed_valid_fraction", 0.)
                            for view in per_target]
            sufficient_coverage = bool(per_target and not insufficient_targets and min(face_support) >= .5)
            face_worst = [view["regions"]["face_tight"]["source5"]["mae_mm"] -
                          view["regions"]["face_tight"]["source3"]["mae_mm"]
                          for view in per_target
                          if view["regions"].get("face_tight", {}).get("source3", {}).get("mae_mm") is not None]
            summary[cond][scene] = {
                "per_target": per_target,
                "mean_per_target_nose_p90_relative_reduction": float(np.mean(valid_nose)) if valid_nose else None,
                "worst_target_face_mae_increase_mm": max(face_worst) if face_worst else None,
                "nose_p90_gate_10pct": bool(sufficient_coverage and valid_nose and np.mean(valid_nose) >= .10),
                "face_mae_gate_1mm": bool(sufficient_coverage and face_worst and max(face_worst) <= 1.),
                "sufficient_common_support": sufficient_coverage,
                "insufficient_targets": insufficient_targets,
                "minimum_face_common_support": min(face_support) if face_support else 0.,
                "support_gate": "every target face and both nose ROIs require >=50% common alpha>.01 + ED support; stricter evidence validity, not a new improvement threshold"}
    return {"scope": "common valid ED pixels, paired targets and frozen ROIs; no target fitting",
            "conditions": summary,
            "status": "geometry measurements only; real visual QA, object control and repeatability remain separate gates"}


def real_coverage_summary(report):
    out = {"status": "appearance/coverage only; no independent real depth or metric face claim",
           "accepted_primary_roi_targets": 0, "unavailable_roi_targets": [], "conditions": {}}
    for cond in ("SA_RGB", "Zip_FF"):
        per_view = []
        for scene, pack in report["packages"].items():
            if pack.get("geometry_truth") is not False:
                continue
            for stem, view in pack["targets"].items():
                if view["tracks"]["primary"]["roi_status"] != "accepted":
                    out["unavailable_roi_targets"].append(scene + "/" + stem)
                    continue
                a = view["methods"][cond]["source3"]["face_tight"]
                b = view["methods"][cond]["source5"]["face_tight"]
                if "coverage_05" not in a or "coverage_05" not in b:
                    raise RuntimeError("Accepted real face ROI lacks paired alpha coverage")
                per_view.append({"scene": scene, "target": stem,
                                 "source3_coverage_05": a["coverage_05"],
                                 "source5_coverage_05": b["coverage_05"],
                                 "source5_minus_source3": b["coverage_05"]-a["coverage_05"]})
        out["conditions"][cond] = {
            "per_view": per_view, "n": len(per_view),
            "worst_coverage_drop": max([0.]+[-x["source5_minus_source3"] for x in per_view]),
            "gate_max_drop": .02,
            "pass": bool(per_view and all(x["source5_minus_source3"] >= -.02 for x in per_view)),
            "visual_double_image_gate": "pending independent image QA; not inferred from alpha"}
    out["accepted_primary_roi_targets"] = len({(x["scene"], x["target"])
                                               for v in out["conditions"].values() for x in v["per_view"]})
    return out


def cost_summary(report):
    rows = []
    for scene, pack in report["packages"].items():
        for arm, methods in pack["models"].items():
            for name, m in methods.items():
                run = m["runtime"].get("run", m["runtime"])
                repeat = m.get("repeatability", {})
                rows.append({"scene": scene, "arm": arm, "condition": name,
                             "gaussians": m["gaussians"],
                             "num_views": len(run.get("images", run.get("image_files", []))),
                             "forward_seconds": run.get("inference_seconds",
                                run.get("timing_seconds", {}).get("feedforward")),
                             "recorded_total_seconds": run.get("total_seconds",
                                run.get("timing_seconds", {}).get("total")),
                             "forward_peak_allocated_mib": run.get("gpu", {}).get("after_feedforward", {}).get("peak_allocated_mib"),
                             "recorded_gpu": run.get("gpu", run.get("profile")),
                             "evaluation_peak_allocated_mib": repeat.get("torch_peak_allocated_mib"),
                             "repeat_render_seconds_after_warmup": [repeat.get("first_render_seconds_after_warmup"),
                                                                    repeat.get("second_render_seconds")],
                             "note": "native run records versus separate replay evaluation; no new model forward"})
    return {"schema": "paired_cost_v1", "rows": rows,
            "repeatability_scope": "same PLY/camera/K; ED in model units; one separate native warmup before timing",
            "repeatability_pass": all(m.get("repeatability", {}).get("pass", False)
                                       for p in report["packages"].values()
                                       for methods in p["models"].values() for m in methods.values()),
            "missing_native_profile_fields": "retained as null; no wall-clock reconstruction from log order"}


def evaluate_real(pack, methods, root, out, device):
    old = json.loads(Path(pack["n1_eval"]).read_text(encoding="utf-8"))
    old_pack = old["packages"][pack["name"]]
    base_root = Path("/mnt/d/output/results/splatanything_results/n1_real_20261122/protocol_v1")
    primary_points = json.loads((base_root / "roi_primary_probe.json").read_text(encoding="utf-8"))
    primary_review = json.loads((base_root / "roi_primary_review.json").read_text(encoding="utf-8"))
    target_prior = json.loads((Path(pack["target_scene"]) / "prior_camera.json").read_text(encoding="utf-8"))
    target_K = {c["camera_id"]: np.asarray(c["matrix"], np.float32)
                for c in target_prior["intrinsics"]}
    old_protocol = json.loads((base_root / "protocol.json").read_text(encoding="utf-8"))
    old_pack_protocol = next(p for p in old_protocol["packages"] if p["scene"] == pack["name"])
    old_review = json.loads((base_root / "roi_review.json").read_text(encoding="utf-8"))
    views = {}
    for tidx, fr in enumerate([x for x in pack["frames"] if x["split"] == "target"]):
        stem = fr["scene_slots"]["target"]
        old_fr = next(f for f in old_pack_protocol["frames"] if f["raw_frame_idx"] == fr["raw_id"])
        old_stem = old_fr["camera_id"]
        view_ref = old_pack["views"][old_stem]
        view = {"stem": stem, "raw_id": fr["raw_id"], "methods": {}, "tracks": {}}
        rendered = {}
        for track, conds in (("primary", ("SA_RGB", "Zip_FF")), ("auxiliary", ("SA_pose_aux",))):
            physical_K = target_K[stem]
            prep = M25.evaluation_geometry(physical_K if track == "auxiliary" else None,
                                           tuple(fr["processed_wh"]), physical_K)
            old_prep = view_ref["evaluation_tracks"][track]["crop"]
            if np.max(np.abs(prep["K_reference"]-old_prep["K_reference"])) > 1e-5 or \
                    prep["first_box"] != old_prep["first_box"] or prep["bbox_offset"] != old_prep["bbox_offset"]:
                raise RuntimeError("Target crop/K differs from frozen ROI coordinate system")
            gt = M21.official_preprocess_rgb(fr["path"], prep)
            roi = None
            if track == "primary":
                q = primary_review["views"][old_stem]
                pts = primary_points[old_stem]["points"]
            else:
                q = old_review["views"][pack["name"]][old_stem]
                pts = old_fr["landmark_qa"].get("points")
            if q["status"] == "accepted":
                if pts is None:
                    raise RuntimeError("Accepted target has no frozen landmarks: " + stem)
                roi = M21.region_masks(np.asarray(pts), gt.shape[:2])["face_tight_landmark"]
            view["tracks"][track] = {"roi_status": q["status"], "prep": prep}
            for arm in ("source3", "source5"):
                for name in conds:
                    m = methods[arm][name]
                    cam = target_setup(pack, arm, name, prep, stem, m["fit"])
                    rd, al, _ = M22.render(m["gpu"], m["degree"], cam, prep["K_reference"], 756, device, .01)
                    values = {"full": metrics(gt, rd, al, np.ones(gt.shape[:2], bool)),
                              "lowpass": metrics(cv2.GaussianBlur(gt, (0, 0), 2.),
                                                cv2.GaussianBlur(rd, (0, 0), 2.), al, np.ones(gt.shape[:2], bool)),
                              "face_tight": metrics(gt, rd, al, roi) if roi is not None else {"status": "N/A"}}
                    view["methods"].setdefault(name, {})[arm] = values
                    rendered[(name, arm, track)] = {"rgb": rd, "roi": roi}
                    Image.fromarray(np.uint8(np.clip(rd, 0, 1)*255)).save(out / "images" / f"{pack['name']}_{stem}_{track}_{name}_{arm}.png")
        views[stem] = view
        compare_rows = [("GT", [("RGB", gt)])]
        for track in ("primary", "auxiliary"):
            available = [(name, arm) for arm in ("source3", "source5")
                         for name in ("SA_RGB", "Zip_FF", "SA_pose_aux")
                         if (name, arm, track) in rendered]
            entries = [("RGB", gt), ("lowpass", cv2.GaussianBlur(gt, (0, 0), 2.))]
            if any(rendered[(name, arm, track)]["roi"] is not None for name, arm in available):
                mask = next(rendered[(name, arm, track)]["roi"] for name, arm in available
                            if rendered[(name, arm, track)]["roi"] is not None)
                yy, xx = np.where(mask)
                entries.append(("face", gt[yy.min():yy.max()+1, xx.min():xx.max()+1]))
            compare_rows.append(("GT "+track, entries))
            for name, arm in available:
                rd = rendered[(name, arm, track)]["rgb"]
                entries = [("RGB", rd), ("lowpass", cv2.GaussianBlur(rd, (0, 0), 2.))]
                roi = rendered[(name, arm, track)]["roi"]
                if roi is not None:
                    yy, xx = np.where(roi)
                    entries.append(("face", rd[yy.min():yy.max()+1, xx.min():xx.max()+1]))
                compare_rows.append((name+" "+arm+" "+track, entries))
        comparison(out / "images" / f"{pack['name']}_{stem}_comparison.png", compare_rows)
    return {"targets": views, "geometry_truth": False, "render_cache": rendered}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--official-repo", default=str(Path.home() / "repos/SplatAnything"))
    ap.add_argument("--only-pack", default=None,
                    choices=("rosemary", "daven_holdout", "real_gray_0221a3dd", "real_black_3a8b3cc7"),
                    help="指定一包只读验收；四包完整验收仍是正式结论的前提")
    args = ap.parse_args()
    started = time.perf_counter()
    code_files = [HERE / f for f in ("27_evaluate_paired.py", "15_face_target_eval.py", "22_n1_head_compare.py",
                                    "25_n1_real_eval.py", "21_geometry_audit_v2.py", "03_render_eval.py", "13_face_select.py")]
    code_sha256 = {str(p): sha(p) for p in code_files}
    sys.path.insert(0, str(Path(args.official_repo).resolve()))
    root, out = Path(args.root), Path(args.out)
    protocol = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if args.only_pack:
        protocol["packages"] = [p for p in protocol["packages"] if p["name"] == args.only_pack]
    if (Path.home() / "miniconda3/envs/splatanything/bin/python").resolve() != Path(sys.executable).resolve():
        raise RuntimeError("Evaluation must use the splatanything environment")
    out.mkdir(parents=True, exist_ok=False)
    (out / "images").mkdir()
    dump(out / "evaluation_provenance.json", {
        "code_version": EVAL_CODE_VERSION, "code_sha256": code_sha256,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv, "python": sys.executable,
        "scope": "read-only PLY replay; no model inference, training or target fitting",
        "complete": False})
    for pack in protocol["packages"]:
        for fr in pack["frames"]:
            if rgb_sha(fr["path"]) != fr["rgb_sha256"]:
                raise RuntimeError("Frozen RGB content changed: " + fr["path"])
            for arm in ("source3", "source5"):
                if arm not in fr["scene_slots"]:
                    continue
                path = Path(pack[arm + "_scene"]) / "images" / (fr["scene_slots"][arm] + ".png")
                if rgb_sha(path) != fr["rgb_sha256"]:
                    raise RuntimeError("Frozen RGB pixels differ across arms: " + str(path))
            if fr["split"] != "target":
                arm = "source5" if fr["split"] == "additional" else "source3"
                path = Path(pack[arm + "_scene"]) / "images" / (fr["scene_slots"][arm] + ".png")
                if sha(path) != fr[arm + "_input_file_sha256"]:
                    raise RuntimeError("Frozen model input file changed: " + str(path))
    import lpips
    lp = lpips.LPIPS(net="alex", verbose=False).to(args.device).eval()
    M22.sim3_test()
    M22.rotation_self_test()
    rng = np.random.default_rng(273)
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    prior_centers = rng.normal(size=(3, 3))
    model_centers = 1.7 * (prior_centers @ Q.T) + np.array([.2, -.3, .7])
    fit = M22.fit_sim3(model_centers, prior_centers)
    if abs(fit[0] - 1.7) > 1e-9 or np.max(np.abs(fit[1] - Q)) > 1e-9:
        raise RuntimeError("Paired Sim3 direction self-test failed")
    point_pkg = rng.normal(size=(4, 3))
    point_model = fit[0] * (point_pkg @ fit[1].T) + fit[2]
    # 正向世界变换使 ED_model=s*ED_pkg；反向渲染同一表面须恢复 ED_pkg。
    if abs(metric_depth_scale(fit) - 1.7) > 1e-12 or \
            abs(metric_depth_scale(fit, True) - 1 / 1.7) > 1e-12 or \
            abs(metric_depth_scale(M22.inverse_sim3(fit)) - 1 / 1.7) > 1e-12 or \
            abs(metric_depth_scale(M22.inverse_sim3(fit), True) - 1.7) > 1e-12:
        raise RuntimeError("Metric ED inverse-Sim3 direction self-test failed")
    depth_camera = np.eye(4)
    depth_camera[:3, 3] = [0, 0, 2]
    forward_camera = M22.transform_camera(depth_camera, fit)
    inverse_camera = M21.model_camera(depth_camera, 1/fit[0], fit[1].T,
                                      -(fit[1].T@fit[2])/fit[0])
    if np.max(np.abs(forward_camera[:3, 3] - inverse_camera[:3, 3])) < 1e-8 or \
            np.max(np.abs(M22.transform_camera(inverse_camera, fit) - depth_camera)) > 1e-9:
        raise RuntimeError("Common-camera inverse-Sim3 roundtrip self-test failed")
    if abs(depth_metrics(np.full((10, 10), .5), np.full((10, 10), .45),
                         np.ones((10, 10), bool))["mae_mm"] - 50) > 1e-9:
        raise RuntimeError("Paired metric-unit self-test failed")
    final = {"schema": "paired_real_observations_results_v1", "manifest_sha256": sha(root / "manifest.json"),
             "evaluation_code_sha256": code_sha256, "only_pack": args.only_pack,
             "geometry_limit": "Synthetic GT depth only; real captures have no independent depth and no metric face claim.",
             "gates": protocol["gates"], "packages": {}, "native_validation": {}}
    for pack in protocol["packages"]:
        pack_anchors = {f["raw_id"] for f in pack["frames"] if f["split"] == "anchor"}
        pack_additional = {f["raw_id"] for f in pack["frames"] if f["split"] == "additional"}
        pack_targets = {f["raw_id"] for f in pack["frames"] if f["split"] == "target"}
        if pack_anchors & pack_additional or pack_anchors & pack_targets or pack_additional & pack_targets:
            raise RuntimeError("anchor/additional/target raw_id leakage")
        methods = {"source3": {}, "source5": {}}
        for arm in ("source3", "source5"):
            anchors = {f["scene_slots"][arm] for f in pack["frames"] if f["split"] == "anchor"}
            for cond in ("SA_RGB", "Zip_FF", "SA_pose_aux"):
                print("LOAD", pack["name"], arm, cond, flush=True)
                m = model_meta(pack, root, arm, cond, anchors, args.device)
                m["provenance"] = validate_provenance(m, protocol, root, pack)
                m["source_scene"] = pack[arm + "_scene"]
                m["frames"] = pack["frames"]
                m["arm"] = arm
                native_replay(m, out / "native" / f"{pack['name']}_{arm}_{cond}", args.device)
                m["source_projection"] = source_projection_checks(
                    m, pack, out / "source_projection" / f"{pack['name']}_{arm}_{cond}", args.device)
                m["gaussians"] = len(m["g"]["means"])
                m["ply_sha256"] = sha(m["ply"])
                methods[arm][cond] = m
        if pack["kind"] == "static_synthetic":
            gt_pkg = Path(pack["pkg"])
            gt_manifest = json.loads((gt_pkg / "cameras.json").read_text(encoding="utf-8"))
            prep = M21.official_preprocess_geometry(np.asarray(gt_manifest["intrinsics"]["K"], np.float64),
                (int(gt_manifest["intrinsics"]["width"]), int(gt_manifest["intrinsics"]["height"])), (756, 756))
        else:
            target_prior = json.loads((Path(pack["target_scene"]) / "prior_camera.json").read_text(encoding="utf-8"))
            physical_K = np.asarray(target_prior["intrinsics"][0]["matrix"], np.float32)
            prep = M25.evaluation_geometry(None, tuple(pack["frames"][0]["processed_wh"]), physical_K)
        for arm in ("source3", "source5"):
            for cond, m in methods[arm].items():
                target_fr = next(f for f in pack["frames"] if f["split"] == "target")
                m["repeat_camera"] = target_setup(pack, arm, cond, prep, target_fr["scene_slots"]["target"], m["fit"])
                m["repeat_K"] = prep["K_reference"]
                if not CUDA_WARMED:
                    warm = M22.load_ply(m["ply"], "wxyz")
                    if len(warm["means"]) > WARMUP_GAUSSIANS:
                        warm = {k: (v[:WARMUP_GAUSSIANS] if isinstance(v, np.ndarray) else v)
                                for k, v in warm.items()}
                    warm = M22.to_gpu(warm, args.device)
                    M22.render(warm, m["degree"], m["repeat_camera"], m["repeat_K"], 756, args.device, .01)
                    del warm
                m["repeatability"] = repeatability_check(m, out / "repeatability" / f"{pack['name']}_{arm}_{cond}.json", args.device)
                if not m["repeatability"]["pass"]:
                    raise RuntimeError("Same-PLY repeat rendering exceeds fixed tolerance: " + pack["name"] + " " + cond)
                print("REPEAT", pack["name"], arm, cond, m["repeatability"]["max_abs_ed"], flush=True)
        if pack["kind"] == "static_synthetic":
            result = evaluate_gt(pack, methods, root, out, lp, args.device)
        else:
            result = evaluate_real(pack, methods, root, out, args.device)
        final["packages"][pack["name"]] = {k: v for k, v in result.items() if k != "render_cache"}
        final["packages"][pack["name"]]["models"] = {
            arm: {n: {"fit": m["fit"], "residual": m["residual"], "gaussians": m["gaussians"],
                      "ply_sha256": m["ply_sha256"], "runtime": m["runtime"],
                      "provenance": m["provenance"], "repeatability": m["repeatability"],
                      "source_projection": m["source_projection"], "native_replay": m["native_replay"]} for n, m in methods[arm].items()}
            for arm in ("source3", "source5")}
        dump(out / (pack["name"] + "_results.json"), final["packages"][pack["name"]])
        del methods
        gc.collect()
        import torch
        torch.cuda.empty_cache()
    dump(out / "paired_results.json", final)
    dump(out / "paired_geometry_summary.json", paired_geometry_summary(final))
    dump(out / "real_coverage_summary.json", real_coverage_summary(final))
    dump(out / "cost_summary.json", cost_summary(final))
    rows = []
    for scene, pack in final["packages"].items():
        for stem, view in pack["targets"].items():
            for name, arms in view["methods"].items():
                paired = arms.get("paired_reduction")
                if paired:
                    for region, pair in paired["depth"].items():
                        for arm in ("source3", "source5"):
                            rows.append({"scene": scene, "target": stem, "condition": name,
                                         "arm": arm + "_common_ed", "region": region,
                                         **{k: val for k, val in pair[arm].items()
                                            if isinstance(val, (int, float))},
                                         "common_ed_valid_fraction": pair["common_ed_valid_fraction"],
                                         "p90_relative_reduction": pair.get("p90_relative_reduction")})
                for arm, v in arms.items():
                    if arm == "paired_reduction":
                        continue
                    for region, value in v.get("depth", v.get("appearance", {})).items():
                        rows.append({"scene": scene, "target": stem, "condition": name, "arm": arm,
                                     "region": region, **{k: val for k, val in value.items()
                                                          if isinstance(val, (int, float))}})
    fieldnames = ["scene", "target", "condition", "arm", "region", "n", "mae",
                  "psnr", "alpha_mean", "coverage_05", "lpips_crop128", "mae_mm",
                  "bias_mm", "p90_abs_mm", "valid_fraction", "common_ed_valid_fraction",
                  "p90_relative_reduction"]
    with open(out / "per_target.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    dump(out / "evaluation_provenance.json", {
        "code_version": EVAL_CODE_VERSION, "code_sha256": code_sha256,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter()-started,
        "argv": sys.argv, "python": sys.executable, "complete": True,
        "scope": "measurement replay complete; visual QA/object control/final promotion remain separate"})
    print("PAIRED_EVAL_DONE", out, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        print(tb, end="", flush=True)
        if "--out" in sys.argv:
            p = Path(sys.argv[sys.argv.index("--out") + 1])
            if p.is_dir():
                with open(p / "failure.log", "w", encoding="utf-8", newline="\n") as f:
                    f.write(tb)
        raise
