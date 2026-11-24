#!/usr/bin/env python3
"""26_prepare_paired.py -- 冻结免训练真实三图/五图配对的输入和相机。

只读取原始 GT 包和 N1 真人协议，不读取任何新模型输出。原三锚点、额外
两帧和独立 target 分开落盘；target 不参与输入选择、对齐或 ROI 调整。

Env vars:
  ROOT  实验根，必须不存在；默认 densitok_portrait_probe/protocol_v1
"""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("ROOT", "/mnt/d/output/results/splatanything_results/"
                           "densitok_portrait_probe/protocol_v1"))


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M07 = load("paired_capture07", "07_capture_to_scene.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rgb_sha(path):
    """输入图是 RGB，原 GT 的 alpha 只用于有效 mask；hash 范围显式分离。"""
    return hashlib.sha256(Image.open(path).convert("RGB").tobytes()).hexdigest()


def dump(path, data):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def scene(out, records):
    (out / "images").mkdir(parents=True)
    cameras = {"num_cameras": len(records), "extrinsics": [], "intrinsics": []}
    for r in records:
        (out / "images" / r["filename"]).write_bytes(Path(r["path"]).read_bytes())
        for typ in ("extrinsics", "intrinsics"):
            cameras[typ].append({"camera_id": r["stem"], "matrix": r[typ].tolist()})
    dump(out / "prior_camera.json", cameras)
    return str(out)


def qa(records, out):
    width, height, label = 280, 380, 40
    tile = Image.new("RGB", (width * len(records), height + label), "#f6f8fb")
    draw = ImageDraw.Draw(tile)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    for i, rec in enumerate(records):
        im = Image.open(rec["path"]).convert("RGB")
        im.thumbnail((width, height), Image.Resampling.LANCZOS)
        x = i * width + (width - im.width) // 2
        tile.paste(im, (x, (height - im.height) // 2))
        draw.text((i * width + 7, height + 10),
                  f"{rec['split']} | {rec['stem']}", fill="#1b2a41", font=font)
    tile.save(out)


def gt_pose(meta, convention):
    m = np.asarray(meta["transform_cam2world"], np.float64)
    if convention == "corrected_opencv":
        return m
    if convention != "manifest_requires_one_crc":
        raise ValueError(convention)
    # 旧包 prior 的 C·R_manifest·C 已由 N0 五视角 look-at / 深度法向回归验证；
    # 不能根据新旧生成器注释猜测相机版本，也不能再次施加 C 共轭。
    m = m.copy()
    m[:3, :3] = np.diag([1., -1., -1.]) @ m[:3, :3] @ np.diag([1., -1., -1.])
    return m


def convention_needs_crc(spec):
    return spec["camera_convention"] == "manifest_requires_one_crc"


def add_gt_package(protocol, spec):
    pkg = Path(spec["pkg"])
    manifest = json.loads((pkg / "cameras.json").read_text(encoding="utf-8"))
    meta = json.loads((pkg / "meta.json").read_text(encoding="utf-8"))
    K = np.asarray(manifest["intrinsics"]["K"], np.float64)
    target = np.asarray(meta["target"], np.float64)
    poses = {}
    records = {}
    for c in manifest["cameras"]:
        cid = int(c["id"])
        pose = gt_pose(c, spec["camera_convention"])
        poses[cid] = pose
        d = pose[:3, 3] - target
        yaw = float(np.degrees(np.arctan2(d[1], d[0])))
        if spec.get("azimuth_override_deg") is not None:
            yaw = float(np.degrees(np.arctan2(d[1] - target[1], d[0] - target[0])))
        records[cid] = {"split": "additional", "raw_id": cid,
                        "stem": f"{len(records):03d}_{cid:06d}", "yaw_deg": yaw,
                        "path": str(pkg / c["file_rgb"]),
                        "depth_path": str(pkg / c["file_depth"]),
                        "extrinsics": pose, "intrinsics": K}
    chosen = spec["anchors"] + spec["additional"] + spec["targets"]
    if len(set(chosen)) != len(chosen) or any(x not in records for x in chosen):
        raise RuntimeError("GT split is missing or overlaps")
    if convention_needs_crc(spec):
        audit = json.loads(Path(spec["n0_eval"]).read_text(encoding="utf-8"))
        if audit["summary"]["measurement_gates"].get("manifest_file_mapping_verified") is not True:
            raise RuntimeError("Legacy CRC requires the frozen N0 camera audit")
        validated = audit["gt_camera_validation"]
        for c in spec["anchors"]:
            keys = [k for k, v in validated.items() if int(k.rsplit("_", 1)[1]) == c]
            if len(keys) != 1 or validated[keys[0]]["prior_vs_C_manifest_C_max"] > 1e-8:
                raise RuntimeError("Legacy camera CRC regression failed: " + str(c))
        # 旧 N0 未冻结 cameras.json 整体 hash；先校验原始 manifest 存在，
        # 再用其逐相机矩阵与 prior 数值对账，不伪造历史哈希。
        protocol.setdefault("camera_validation", {})[spec["name"]] = {
            "rule": "prior=C@R_manifest@C once, unchanged translation",
            "source": spec["n0_eval"], "selected_camera_regressions": {
                str(c): {k: validated[k] for k in validated if int(k.rsplit("_", 1)[1]) == c}
                for c in spec["anchors"] + spec["additional"] + spec["targets"]}}
    for role, ids in (("anchor", spec["anchors"]), ("additional", spec["additional"]),
                      ("target", spec["targets"])):
        for order, cid in enumerate(ids):
            records[cid].update(split=role, order=order)
    for role, ids in (("source3", spec["anchors"]),
                      ("source5", sorted(spec["anchors"] + spec["additional"])),
                      ("target", spec["targets"])):
        recs = [records[x] for x in ids]
        for i, r in enumerate(recs):
            r["stem_" + role] = f"{i:03d}_{r['raw_id']:06d}"
        scene_recs = []
        for r in recs:
            slot = r["stem_" + role]
            item = dict(r, stem=slot)
            item["filename"] = slot + ".png"
            r.setdefault("scene_slots", {})[role] = slot
            if role == "source3" and spec.get("reuse_n1"):
                # 原 N1 的三图目录逐字节复用，不通过重新保存改变像素。
                old = Path("~/output/splatanything_results/_scenes/rosemary33v").expanduser()
                item["path"] = str(old / "images" / item["filename"])
            scene_recs.append(item)
        scene(ROOT / "scenes" / (spec["name"] + "_" + role), scene_recs)
    seq = [records[x] for x in spec["anchors"] + spec["additional"] + spec["targets"]]
    n1_hashes = {r["stem"]: sha(r["path"]) for r in scene_recs} if spec.get("reuse_n1") else None
    qa(seq, ROOT / (spec["name"] + "_inputs.jpg"))
    dump(ROOT / (spec["name"] + "_source5_meta.json"),
         {"picked_raw_ids": [r["raw_id"] for r in seq],
          "camera_convention": spec["camera_convention"]})
    protocol["packages"].append({
        **{k: v for k, v in spec.items() if k not in ("anchors", "additional", "targets")},
            "kind": "static_synthetic", "anchor_ids": spec["anchors"],
            "additional_ids": spec["additional"], "target_ids": spec["targets"],
            "source3_scene": str(ROOT / "scenes" / (spec["name"] + "_source3")),
            "source5_scene": str(ROOT / "scenes" / (spec["name"] + "_source5")),
            "target_scene": str(ROOT / "scenes" / (spec["name"] + "_target")),
            "frames": [{k: v for k, v in r.items() if k not in ("extrinsics", "intrinsics")}
                       for r in seq],
            "source3_uses_original_n1_files": spec.get("reuse_n1", False),
            "source3_n1_input_sha256": n1_hashes,
            "new_identity": spec.get("new_identity", False),
        })


def frozen_prior(scene_name):
    prior = json.loads((Path(scene_name) / "prior_camera.json").read_text(encoding="utf-8"))
    return {c["camera_id"]: {typ: np.asarray(next(x["matrix"] for x in prior[typ]
            if x["camera_id"] == c["camera_id"]), np.float64)
            for typ in ("extrinsics", "intrinsics")} for c in prior["extrinsics"]}


def intermediate_real_frames(pack):
    """用相机实际绕转方向取近中点，raw 序号仅用于排除 target/保证轨迹顺序。"""
    target_ids = set(pack["target_raw_frame_idx"])
    chosen = []
    for lo, hi in zip(pack["source_raw_frame_idx"][:-1], pack["source_raw_frame_idx"][1:]):
        fr = next(f for f in pack["frames"] if f["raw_frame_idx"] == lo)
        gr = next(f for f in pack["frames"] if f["raw_frame_idx"] == hi)
        cams = frozen_prior(fr["processed_path"].rsplit("/images/", 1)[0])
        c0 = cams[fr["camera_id"]]["extrinsics"][:3, 3]
        c1 = cams[gr["camera_id"]]["extrinsics"][:3, 3]
        center = (c0 + c1) / 2
        score = lambda cam: np.linalg.norm(cam[:3, 3] - center)
        data = json.loads((Path(pack["raw_dir"]) / "transforms.json").read_text(encoding="utf-8"))
        candidates = []
        for idx in range(lo + 1, hi):
            if idx in target_ids:
                continue
            # 原始 index 在环绕中不是角度；须用实际 camera-to-anchor 方向选插值。
            raw = data["frames"][idx]
            mat = np.asarray(raw["transform_matrix"], np.float64) @ np.diag([1., -1., -1., 1.])
            angle0 = np.arctan2(*(c0 - center)[[1, 0]])
            angle1 = np.arctan2(*(c1 - center)[[1, 0]])
            angle = np.arctan2(*(mat[:3, 3] - center)[[1, 0]])
            delta = np.arctan2(np.sin(angle1 - angle0), np.cos(angle1 - angle0))
            rel = np.arctan2(np.sin(angle - angle0), np.cos(angle - angle0))
            progress = 0. if abs(delta) < 1e-10 else rel / delta
            candidates.append((abs(progress - .5) + 2 * float(not 0 < progress < 1),
                               abs(idx - (lo + hi) / 2), idx))
        if not candidates:
            raise RuntimeError("No target-disjoint intermediate frame in " + pack["scene"])
        chosen.append(min(candidates)[2])
    return chosen


def add_real_packages(protocol):
    old_root = Path("/mnt/d/output/results/splatanything_results/n1_real_20261122/protocol_v1")
    old = json.loads((old_root / "protocol.json").read_text(encoding="utf-8"))
    for pack in old["packages"]:
        anchors = pack["source_raw_frame_idx"]
        additional = intermediate_real_frames(pack)
        targets = pack["target_raw_frame_idx"]
        all_frames = additional
        converted = ROOT / "conversion" / pack["scene"]
        argv = sys.argv
        try:
            sys.argv = ["07_capture_to_scene.py", "--raw-dir", pack["raw_dir"],
                        "--out-dir", str(converted), "--frames", ",".join(map(str, all_frames)),
                        "--max-side", "0"]
            M07.main()
        finally:
            sys.argv = argv
        meta = json.loads((converted / "meta.json").read_text(encoding="utf-8"))
        mapping = {f["idx"]: f["out"][:-4] for f in meta["frames"]}
        records = []
        for role, ids in (("anchor", anchors), ("additional", additional), ("target", targets)):
            for order, idx in enumerate(ids):
                if role == "additional":
                    stem = mapping[idx]
                    path = converted / "images" / (stem + ".png")
                else:
                    fr = next(f for f in pack["frames"] if f["raw_frame_idx"] == idx)
                    stem, path = fr["camera_id"], Path(fr["processed_path"])
                records.append({"split": role, "raw_id": idx, "order": order, "stem": stem,
                                "path": str(path), "filename": stem + ".png"})
        cams = {}
        for scene_name in (pack["source_scene"], pack["target_scene"], str(converted)):
            prior = json.loads((Path(scene_name) / "prior_camera.json").read_text(encoding="utf-8"))
            for typ in ("extrinsics", "intrinsics"):
                for c in prior[typ]:
                    cams.setdefault(c["camera_id"], {})[typ] = np.asarray(c["matrix"], np.float64)
        for r in records:
            r.update(cams[r["stem"]])
        for role, ids in (("source3", anchors), ("source5", sorted(anchors + additional)),
                          ("target", targets)):
            recs = sorted((dict(r) for r in records if r["raw_id"] in ids),
                          key=lambda r: ids.index(r["raw_id"]))
            for i, r in enumerate(recs):
                # 槽位编号只表达原始 raw 轨迹顺序；hash 与相机均保持一一对应。
                slot = f"{i:03d}_{r['raw_id']:06d}"
                original = next(x for x in records if x["raw_id"] == r["raw_id"])
                original.setdefault("scene_slots", {})[role] = slot
                r["stem"], r["filename"] = slot, slot + ".png"
            scene(ROOT / "scenes" / (pack["scene"] + "_" + role), recs)
        qa(records, ROOT / (pack["scene"] + "_inputs.jpg"))
        protocol["packages"].append({
            "name": pack["scene"], "kind": "near_static_real", "new_identity": False,
            "package_id": pack["package_id"], "camera_convention": "opencv_c2w",
            "anchor_ids": anchors, "additional_ids": additional, "target_ids": targets,
            "selection_rule": "camera-centered near-midpoint between adjacent anchors; target excluded before scoring; no model outputs consulted",
            "source3_scene": str(ROOT / "scenes" / (pack["scene"] + "_source3")),
            "source5_scene": str(ROOT / "scenes" / (pack["scene"] + "_source5")),
            "target_scene": str(ROOT / "scenes" / (pack["scene"] + "_target")),
            "frames": [{k: v for k, v in r.items() if k not in ("extrinsics", "intrinsics")}
                       for r in records],
            "n1_protocol": str(old_root / "protocol.json"),
            "n1_eval": "/mnt/d/output/results/splatanything_results/n1_real_20261122/real_eval_v18/n1_real_eval.json",
            "geometry_truth": False,
        })


def main():
    if ROOT.exists():
        raise SystemExit("Protocol output must not exist: " + str(ROOT))
    ROOT.mkdir(parents=True)
    specs = [{"name": "rosemary", "pkg": "/mnt/d/dataset/human_fbx_gt/rosemary_face_v3",
              "camera_convention": "manifest_requires_one_crc", "anchors": [20, 30, 38],
              "additional": [25, 27], "targets": [24, 34], "reuse_n1": True,
              "landmarks": "/mnt/d/output/results/splatanything_results/geometry_audit_v2_20261122/landmarks_native/landmarks_native.json",
              "n1_eval": "/mnt/d/output/results/splatanything_results/n1_head_compare_20261122/rosemary_v27/n1_head_compare.json",
              "n0_eval": "/mnt/d/output/results/splatanything_results/geometry_audit_v2_20261122/five_view_rigid_v28/geometry_audit_v2.json"}]
    extra = os.environ.get("NEW_GT_PKG", "")
    if extra:
        specs.append({"name": "daven_holdout", "pkg": extra,
                      "camera_convention": "corrected_opencv", "anchors": [20, 30, 38],
                      "additional": [25, 27], "targets": [24, 34], "new_identity": True,
                      "landmarks": str(ROOT / "daven_landmarks_native.json")})
    protocol = {"schema": "paired_real_observations_v1", "created_before_inference": True,
                "purpose": "Test whether two real intermediate observations improve held-out face geometry; not training or token completion.",
                "variables": "A/B differ only by two real intermediate RGB views. The same three anchors fit each arm's single Sim(3).",
                "conditions": ["SA_RGB", "Zip_FF", "SA_pose_aux"],
                "target_fit_or_selection": False, "token_ba": False, "training": False,
                "size": [756, 756], "background": 0, "eps2d": .01,
              "nose_regions": ["nose_tip", "nose_ala"],
              "protocol_revision": "v5.1 evaluation hard gates added before final scoring; input/split unchanged",
              "evaluation_contract": {"common_ed_pixels_for_paired_geometry": True,
                                     "face_mae_and_nose_p90_same_common_mask": True,
                                     "real_coverage_scope": "accepted primary face ROI; full-frame auxiliary coverage separate",
                                     "repeatability": "same PLY/camera two repeated renders; differences below fixed noise tolerance",
                                     "object_control": "fixed 3/5-view appearance/coverage control; no facial ROI or metric geometry"},
              "gates": {"requires_two_static_identities": True,
                          "mean_nose_p90_relative_reduction_min": .10,
                          "worst_target_face_mae_increase_max_mm": 1.0,
                          "real_face_coverage_drop_max": .02,
                          "requires_no_new_double_image_on_originally_visible_real_faces": True},
                "packages": []}
    for spec in specs:
        add_gt_package(protocol, spec)
    add_real_packages(protocol)
    for pack in protocol["packages"]:
        groups = [set(pack[k]) for k in ("anchor_ids", "additional_ids", "target_ids")]
        if any(a & b for i, a in enumerate(groups) for b in groups[i + 1:]):
            raise RuntimeError("Input/target overlap: " + pack["name"])
        for role in ("source3", "source5", "target"):
            scene_dir = Path(pack[role + "_scene"])
            prior = json.loads((scene_dir / "prior_camera.json").read_text(encoding="utf-8"))
            ordered = [c["camera_id"] for c in prior["extrinsics"]]
            expected = sorted(pack["anchor_ids"] + (pack["additional_ids"] if role == "source5" else [])
                              if role != "target" else pack["target_ids"])
            expected_slots = {r["scene_slots"][role]: r["raw_id"] for r in pack["frames"]
                              if role in r.get("scene_slots", {})}
            actual = [expected_slots[cid] for cid in ordered]
            if actual != expected or set(expected_slots) != set(ordered) or \
                    [c["camera_id"] for c in prior["intrinsics"]] != ordered:
                raise RuntimeError("Frozen slot/camera order mismatch: " + pack["name"] + " " + role)
        for r in pack["frames"]:
            r["sha256"] = sha(r["path"])
            r["rgb_sha256"] = rgb_sha(r["path"])
            for role in ("source3", "source5"):
                if role not in r["scene_slots"]:
                    continue
                image_path = Path(pack[role + "_scene"]) / "images" / (r["scene_slots"][role] + ".png")
                r[role + "_input_file_sha256"] = sha(image_path)
                if rgb_sha(image_path) != r["rgb_sha256"]:
                    raise RuntimeError("Scene RGB pixels differ from selected original: " + str(image_path))
            r["processed_wh"] = list(Image.open(r["path"]).size)
            if r.get("depth_path"):
                r["depth_sha256"] = sha(r["depth_path"])
    protocol["code_sha256"] = {str(HERE / f): sha(HERE / f) for f in
                               ("26_prepare_paired.py", "07_capture_to_scene.py")}
    dump(ROOT / "manifest.json", protocol)
    print("PAIRED_MANIFEST", ROOT / "manifest.json", flush=True)
    print("Next: new identity FaceMesh/ROI visual QA before any model output.", flush=True)


if __name__ == "__main__":
    main()
