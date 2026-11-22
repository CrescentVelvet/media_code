#!/usr/bin/env python3
"""23_n1_prepare_real.py -- 冻结真实近静态包的五帧协议，生成可复现场景与 ROI QA 材料。

只依据原始输入和人工预注册做筛选，绝不读取模型输出。原始帧号固定，target 不得进
source/SA RGB/Zip RGB 输入目录；相同 source PNG 供两模型 RGB-only 受控轨使用。
这里不做独立身份的外部证明、静止性保证或深度/毫米几何声明。

Env vars:
  ROOT        实验根目录（必须为空或不存在），默认
              /mnt/d/output/results/splatanything_results/n1_real_20261122/protocol_v1
"""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get(
    "ROOT", "/mnt/d/output/results/splatanything_results/n1_real_20261122/protocol_v1"
)).expanduser()
PACKS = [
    {
        "package_id": "0221a3dde05546258a1972136ecf2afc",
        "scene": "real_gray_0221a3dd",
        "raw_dir": "/mnt/d/dataset/人像验收数据汇总-含佳佳/0221a3dde05546258a1972136ecf2afc_input",
        "description": "画面筛选为灰衣站立男性；以包 ID 区分，不以衣服作为身份证明",
        "sources": [0, 18, 43], "targets": [10, 32],
    },
    {
        "package_id": "3a8b3cc746304f49b9e3275e36aa9374",
        "scene": "real_black_3a8b3cc7",
        "raw_dir": "/mnt/d/dataset/手部微动样例/3a8b3cc746304f49b9e3275e36aa9374-男生室内",
        "description": "画面筛选为黑衣站立男性；以包 ID 区分，不与佳佳开发段混算",
        "sources": [0, 98, 196], "targets": [49, 147],
    },
]


def load_mod(name, filename):
    spec = importlib.util.spec_from_file_location(name, str(HERE / filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M07 = load_mod("n1_real_capture07", "07_capture_to_scene.py")
M21 = load_mod("n1_real_geometry21", "21_geometry_audit_v2.py")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def font(size):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/mnt/c/Windows/Fonts/arial.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def input_qa(records, output):
    tile_w, tile_h, label_h = 300, 400, 42
    sheet = Image.new("RGB", (tile_w * len(records), tile_h + label_h), "#f7f8fa")
    draw = ImageDraw.Draw(sheet)
    for i, rec in enumerate(records):
        im = Image.open(rec["processed_path"]).convert("RGB")
        im.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
        x = i * tile_w + (tile_w - im.width) // 2
        sheet.paste(im, (x, (tile_h - im.height) // 2))
        label = f"{rec['split'].upper()} | raw {rec['raw_frame_idx']} | slot {rec['converted_idx']}"
        draw.text((i * tile_w + 8, tile_h + 12), label, fill="#152235", font=font(15))
    sheet.save(output)


def landmark_qa(rec, output):
    """sam3 env 单独执行；在官方正方形画布检测，禁止固定框代替。"""
    scene = Path(rec["processed_path"]).parent.parent
    review = json.loads((ROOT / "roi_review.json").read_text(encoding="utf-8"))
    pack_key = scene.name.rsplit("_", 1)[0]
    if pack_key not in review.get("views", {}):
        pack_key = scene.name.removesuffix("_all5")
    if pack_key not in review.get("views", {}):
        raise ValueError("Cannot infer package key from " + scene.name)
    prior = json.loads((scene / "prior_camera.json").read_text(encoding="utf-8"))
    K = np.asarray(next(c["matrix"] for c in prior["intrinsics"]
                        if c["camera_id"] == rec["camera_id"]), np.float64)
    wh = tuple(rec["processed_wh"])
    prep = M21.official_preprocess_geometry(K, wh, (756, 756))
    rgb = np.uint8(np.clip(M21.official_preprocess_rgb(rec["processed_path"], prep), 0, 1) * 255)
    Image.fromarray(rgb).save(str(output).replace("_roi.png", "_square.png"))
    detection_crop = None
    try:
        points, scale = M21.detect_landmarks(rgb)
    except RuntimeError:
        # 检测框只辅助定位；地标必须由真实 FaceMesh 输出，并经同一几何映射。
        if output.name.endswith("_roi.png") and not (scene / "prior_camera.json").is_file():
            raise
        reason = review.get("unavailable_landmark_views", {}).get(
            pack_key, {}).get(rec["camera_id"])
        native_box = review.get("detection_crops_native_xyxy", {}).get(
            pack_key, {}).get(rec["camera_id"])
        if native_box and reason and rec["split"] == "target":
            if not reason.get("whole_head_or_full_face_unavailable"):
                raise ValueError("A partial-face ROI must state its visual scope")
            native = np.asarray(Image.open(rec["processed_path"]).convert("RGB"))
            nx0, ny0, nx1, ny1 = native_box
            points_crop, scale = M21.detect_landmarks(np.ascontiguousarray(native[ny0:ny1, nx0:nx1]))
            points_native = points_crop + np.array([nx0, ny0])
            sx = prep["resized_size"][0] / prep["crop_size"][0]
            sy = prep["resized_size"][1] / prep["crop_size"][1]
            points = (points_native - np.array(prep["first_box"][:2])) * np.array([sx, sy])
            points -= np.array(prep["bbox_offset"])
            detection_crop = {"native_xyxy": native_box, "mapped_to_square": True}
        elif reason and rec["split"] == "target":
            return {"status": "unavailable", "reason": reason["reason"],
                    "points": None, "roi_pixels": {},
                    "common_square_head_or_full_face_unavailable": True}
        else:
            box = rec.get("detection_crop_xyxy")
            if not box:
                raise
            x0, y0, x1, y1 = box
            detection_crop = box
            points_crop, scale = M21.detect_landmarks(rgb[y0:y1, x0:x1])
            points = points_crop + np.array([x0, y0])
    masks = M21.region_masks(points, rgb.shape[:2])
    im = Image.fromarray(rgb).copy()
    dr = ImageDraw.Draw(im)
    oval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
            365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
            132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
    dr.line([tuple(points[i]) for i in oval], fill=(0, 255, 0), width=4)
    for name, inds in M21.REGIONS.items():
        color = (255, 45, 45) if name.startswith("nose") else (20, 170, 255)
        for x, y in points[inds]:
            dr.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
        dr.text(tuple(points[inds].mean(axis=0)), name, fill=color, font=font(22))
    im.thumbnail((756, 1008), Image.Resampling.LANCZOS)
    im.save(output)
    return {"detection_scale": float(scale), "points": points.tolist(),
            "detection_crop_xyxy": detection_crop,
            "roi_pixels": {k: int(v.sum()) for k, v in masks.items()}}


def main():
    if ROOT.exists() and "--landmarks" not in sys.argv:
        sys.exit("❌ 协议输出已存在，不覆盖：" + str(ROOT))
    if "--landmarks" in sys.argv:
        protocol_path = ROOT / "protocol.json"
        if not protocol_path.is_file():
            sys.exit("❌ 先运行 prepare，再运行 --landmarks")
        import mediapipe  # noqa: F401  # 强制在 sam3 环境中检查依赖
        sys.path.insert(0, str(Path.home() / "repos/SplatAnything"))
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        review_path = ROOT / "roi_review.json"
        review = json.loads(review_path.read_text(encoding="utf-8")) if review_path.exists() else {}
        detection_boxes = review.get("detection_crops_756_xyxy", {})
        for pack in protocol["packages"]:
            for rec in pack["frames"]:
                rec["detection_crop_xyxy"] = detection_boxes.get(pack["scene"], {}).get(rec["camera_id"])
                roi_path = ROOT / f"{pack['scene']}_{rec['split']}_{rec['converted_idx']:03d}_roi.png"
                if roi_path.exists() and "landmark_qa" in rec:
                    print("⏭️ ROI 已生成，保留并继续:", roi_path.name, flush=True)
                    continue
                rec["landmark_qa"] = landmark_qa(rec, roi_path)
                print("ROI", pack["scene"], rec["converted_idx"],
                      rec["landmark_qa"].get("detection_scale", rec["landmark_qa"].get("status", "detected")), flush=True)
        protocol["landmark_qa"] = {
            "method": "MediaPipe FaceMesh; native processed PNG; multi-scale detection",
            "status": "automated proposals only; every view requires human visual review before scored evaluation",
            "not_semantic_skin_or_hair_segmentation": True,
            "review_file": "roi_review.json (to be completed after visual review)",
        }
        write_json(protocol_path, protocol)
        return

    (ROOT / "scenes").mkdir(parents=True)
    protocol = {
        "schema": "n1_real_preregistration_v1",
        "status": "preregistered_from_input_images_before_model_outputs",
        "created_before_inference": True,
        "model_outputs_seen_for_frame_selection": False,
        "staticness": "near_static_only",
        "identity_scope": "two visually distinct capture packages; package IDs are the experimental identities, no external identity certificate",
        "geometry_truth": {
            "independent_depth_or_mesh_available": False,
            "camera_prior_is_not_depth_gt": True,
            "metric_geometry_reporting_forbidden": True,
            "allowed_evidence": ["held-out RGB", "fixed ROI appearance", "alpha coverage", "fixed low-pass", "visual inspection", "source-only camera alignment diagnostics"],
        },
        "input_protocol": {
            "sources_per_pack": 3, "held_out_targets_per_pack": 2,
            "source_processing": "original HEIC -> EXIF transpose -> undistort -> ROI crop; original pixel scale; synchronized K",
            "camera_convention": "transforms.json OpenGL c2w @ diag(1,-1,-1,1) -> OpenCV c2w",
            "common_evaluation": "756x756 via official SA preprocessing for both models; black background; eps2d=.01",
            "primary_track": ["SA MODE=rgb", "Zip ba=False"], "controlled_auxiliary": "SA MODE=pose, reported separately",
            "zip_token_ba": False, "target_fitting_or_selection": False, "training": False, "ply_concatenation": False,
            "frame_substitution_after_model_output": False,
            "preprocess_command": "23_n1_prepare_real.py via 23_run_real.sh PREPARE=1",
        },
        "limitations": [
            "Head/hand position and expression may vary across the capture; near-static is not rigid ground truth.",
            "Camera priors support evaluation only; no independently certified metric geometry.",
            "FaceMesh tight oval is not skin/hair semantic segmentation and must be visually checked per view.",
            "These packages do not by themselves justify an N2 base selection.",
        ],
        "packages": [],
    }
    for pack in PACKS:
        raw = Path(pack["raw_dir"])
        tf = raw / "transforms.json"
        data = json.loads(tf.read_text(encoding="utf-8"))
        selected = pack["sources"] + pack["targets"]
        all_dir = ROOT / "scenes" / (pack["scene"] + "_all5")
        argv = sys.argv
        try:
            sys.argv = ["07_capture_to_scene.py", "--raw-dir", str(raw), "--out-dir", str(all_dir),
                        "--frames", ",".join(map(str, selected)), "--max-side", "0"]
            M07.main()
        finally:
            sys.argv = argv
        meta = json.loads((all_dir / "meta.json").read_text(encoding="utf-8"))
        if meta["picked_idx"] != sorted(selected):
            raise RuntimeError("Converted frame mapping differs from preregistration")
        by_raw = {f["idx"]: f for f in meta["frames"]}
        recs = []
        for split, raw_indices in (("source", pack["sources"]), ("target", pack["targets"])):
            dest = ROOT / "scenes" / (pack["scene"] + "_" + split)
            (dest / "images").mkdir(parents=True)
            cameras = {"num_cameras": len(raw_indices), "extrinsics": [], "intrinsics": []}
            for idx in raw_indices:
                fr = by_raw[idx]
                name = fr["out"]
                src_image = all_dir / "images" / name
                out_image = dest / "images" / name
                out_image.write_bytes(src_image.read_bytes())
                ci = all_dir / "prior_camera.json"
                cd = json.loads(ci.read_text(encoding="utf-8"))
                for typ in ("extrinsics", "intrinsics"):
                    cameras[typ].append(next(c for c in cd[typ] if c["camera_id"] == name[:-4]))
                original = raw / data["frames"][idx]["file_path"].replace("./", "")
                rec = {
                    "split": split, "raw_frame_idx": idx,
                    "converted_idx": int(name[:3]), "camera_id": name[:-4],
                    "original_heic": str(original), "original_sha256": sha256(original),
                    "processed_path": str(out_image), "processed_sha256": sha256(out_image),
                    "processed_wh": list(Image.open(out_image).size),
                    "qa_status": "pending_final_visual_review",
                    "notes": "Input-only near-static candidate; no model-output-based selection",
                }
                recs.append(rec)
            write_json(dest / "prior_camera.json", cameras)
            subset_meta = {"source_all5": str(all_dir), "picked_raw_frame_idx": raw_indices,
                           "picked_ids": [c["camera_id"] for c in cameras["extrinsics"]],
                           "source_or_target": split}
            write_json(dest / "meta.json", subset_meta)
        for rec in recs:
            assert rec["processed_sha256"] == sha256(all_dir / "images" / (rec["camera_id"] + ".png"))
        input_qa(recs, ROOT / (pack["scene"] + "_inputs.jpg"))
        pack_record = {k: v for k, v in pack.items() if k not in ("raw_dir", "sources", "targets")}
        pack_record.update({
            "raw_dir": str(raw), "transforms_sha256": sha256(tf),
            "n_original_frames": len(data["frames"]),
            "source_raw_frame_idx": pack["sources"], "target_raw_frame_idx": pack["targets"],
            "frame_order_in_all5_is_sorted_raw_index_not_split_order": True,
            "source_scene": str(ROOT / "scenes" / (pack["scene"] + "_source")),
            "target_scene": str(ROOT / "scenes" / (pack["scene"] + "_target")),
            "frames": recs,
            "near_static_qa": {
                "status": "near_static_candidate",
                "visual_observation": "standing subject with face/profile coverage; minor head/hand changes possible",
                "not_claimed": ["absolute stillness", "independent scanned geometry", "rigid expression across frames"],
            },
        })
        protocol["packages"].append(pack_record)
    protocol["code_hashes"] = {str(HERE / n): sha256(HERE / n) for n in (
        "23_n1_prepare_real.py", "07_capture_to_scene.py", "21_geometry_audit_v2.py")}
    protocol["software"] = {"python": sys.version.split()[0],
                            "sa_repo": subprocess.check_output(["git", "-C", str(Path.home() / "repos/SplatAnything"),
                                                              "rev-parse", "HEAD"], text=True).strip(),
                            "zip_repo": subprocess.check_output(["git", "-C", str(Path.home() / "repos/ZipSplat"),
                                                               "rev-parse", "HEAD"], text=True).strip()}
    write_json(ROOT / "protocol.json", protocol)
    print("🎉 五帧预注册与 SHA-256 已保存:", ROOT / "protocol.json", flush=True)
    print("⏭️ 下一步：sam3 env 执行 --landmarks，再逐帧目检 ROI；不得按模型输出换帧。", flush=True)


if __name__ == "__main__":
    main()
