#!/usr/bin/env python3
"""Batch SAM 3D Body inference over a folder of images, with timing.

Loads the SAM 3D Body model + optional detector/segmentor/FOV estimator ONCE
(timed), then loops over every image in INPUT_DIR (walked recursively),
running full-body human mesh recovery and saving per image:
  $OUTPUT_DIR/result/<rel>.jpg   (rendered overlay, like the official demo.py)
  $OUTPUT_DIR/mesh/<rel stem>_mesh_<pid>.ply   (per-person 3D mesh)
  $OUTPUT_DIR/npz/<rel>.npz                    (per-person numeric outputs)

Mirrors the official demo.py's logic (ViTDet detector + MoGe2 FOV by default,
SAM2 segmentor only if a path is given, recursive relative-path output) but
adds timing, a recursive walk, and PLY/npz artifacts. Reads params from env
(set by 02_run_inference.sh).

Env vars (set by 02_run_inference.sh):
  SAM3D_DIR, CHECKPOINT_PATH, MHR_PATH, DEVICE,
  DETECTOR_NAME, DETECTOR_PATH, SEGMENTOR_NAME, SEGMENTOR_PATH,
  FOV_NAME, FOV_PATH, INPUT_DIR, OUTPUT_DIR,
  BBOX_THRESH, USE_MASK, INFERENCE_TYPE, SAVE_NPZ
"""
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

SAM3D_DIR = os.environ.get("SAM3D_DIR", "../sam-3d-body")
sys.path.insert(0, SAM3D_DIR)  # so `from sam_3d_body import ...` / `from tools...` resolve

import torch  # noqa: E402

from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator  # noqa: E402
from sam_3d_body.visualization.renderer import Renderer  # noqa: E402
from tools.vis_utils import visualize_sample_together  # noqa: E402

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH")
MHR_PATH = os.environ.get("MHR_PATH", "")
DEVICE = os.environ.get("DEVICE", "cuda")
DETECTOR_NAME = os.environ.get("DETECTOR_NAME", "vitdet")
DETECTOR_PATH = os.environ.get("DETECTOR_PATH", os.environ.get("SAM3D_DETECTOR_PATH", ""))
SEGMENTOR_NAME = os.environ.get("SEGMENTOR_NAME", "sam2")
SEGMENTOR_PATH = os.environ.get("SEGMENTOR_PATH", os.environ.get("SAM3D_SEGMENTOR_PATH", ""))
FOV_NAME = os.environ.get("FOV_NAME", "moge2")
FOV_PATH = os.environ.get("FOV_PATH", os.environ.get("SAM3D_FOV_PATH", ""))
INPUT_DIR = os.environ.get("INPUT_DIR")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR")
BBOX_THRESH = float(os.environ.get("BBOX_THRESH", "0.8"))
USE_MASK = os.environ.get("USE_MASK", "0") == "1"
INFERENCE_TYPE = os.environ.get("INFERENCE_TYPE", "full")
SAVE_NPZ = os.environ.get("SAVE_NPZ", "1") == "1"

# Mesh base color used by notebook/utils.py's save_mesh_results.
LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif"}


def build_estimator():
    device = DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        print("[!] CUDA not available — falling back to CPU (will be slow).")
        device = "cpu"

    print(f"[*] loading SAM 3D Body model (device={device}) ...")
    t0 = time.time()
    model, model_cfg = load_sam_3d_body(
        CHECKPOINT_PATH, device=device, mhr_path=MHR_PATH
    )
    print(f"[*] 模型加载耗时: {time.time() - t0:.2f}s")

    human_detector = human_segmentor = fov_estimator = None
    if DETECTOR_NAME:
        print(f"[*] loading human detector: {DETECTOR_NAME} ...")
        from tools.build_detector import HumanDetector

        human_detector = HumanDetector(
            name=DETECTOR_NAME, device=device, path=DETECTOR_PATH
        )
    # Same conditional as demo.py: SAM2 only loads when a path is given.
    if SEGMENTOR_NAME and (
        SEGMENTOR_NAME != "sam2" or (SEGMENTOR_NAME == "sam2" and len(SEGMENTOR_PATH))
    ):
        print(f"[*] loading human segmentor: {SEGMENTOR_NAME} ...")
        from tools.build_sam import HumanSegmentor

        human_segmentor = HumanSegmentor(
            name=SEGMENTOR_NAME, device=device, path=SEGMENTOR_PATH
        )
    if FOV_NAME:
        print(f"[*] loading FOV estimator: {FOV_NAME} ...")
        from tools.build_fov_estimator import FOVEstimator

        fov_estimator = FOVEstimator(
            name=FOV_NAME, device=device, path=FOV_PATH
        )

    return SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )


def save_mesh_and_npz(img_bgr, outputs, faces, rel, output_dir):
    """Save per-person PLY (+ optional npz) for one image's outputs."""
    mesh_dir = output_dir / "mesh" / rel.parent
    mesh_dir.mkdir(parents=True, exist_ok=True)
    npz_dir = output_dir / "npz" / rel.parent
    if SAVE_NPZ:
        npz_dir.mkdir(parents=True, exist_ok=True)

    npz_payload = {"n_persons": np.array(len(outputs))}
    for pid, out in enumerate(outputs):
        renderer = Renderer(
            focal_length=out["focal_length"], faces=faces
        )
        tmesh = renderer.vertices_to_trimesh(
            out["pred_vertices"], out["pred_cam_t"], LIGHT_BLUE
        )
        ply_path = mesh_dir / f"{rel.stem}_mesh_{pid:03d}.ply"
        tmesh.export(str(ply_path))

        if SAVE_NPZ:
            for k in (
                "pred_vertices", "pred_cam_t", "pred_keypoints_3d",
                "pred_keypoints_2d", "focal_length", "bbox", "global_rot",
                "body_pose_params", "hand_pose_params", "scale_params",
                "shape_params", "expr_params", "pred_joint_coords",
                "pred_global_rots", "mhr_model_params",
            ):
                v = out.get(k)
                if v is None:
                    continue
                npz_payload[f"{k}_p{pid}"] = np.asarray(v)
            if "lhand_bbox" in out:
                npz_payload[f"lhand_bbox_p{pid}"] = np.asarray(out["lhand_bbox"])
            if "rhand_bbox" in out:
                npz_payload[f"rhand_bbox_p{pid}"] = np.asarray(out["rhand_bbox"])
    if SAVE_NPZ and outputs:
        npz_path = npz_dir / f"{rel.stem}.npz"
        np.savez_compressed(str(npz_path), **npz_payload)


def main():
    if not CHECKPOINT_PATH:
        sys.exit("ERROR: CHECKPOINT_PATH not set.")
    if not INPUT_DIR:
        sys.exit("ERROR: INPUT_DIR not set.")
    if not OUTPUT_DIR:
        sys.exit("ERROR: OUTPUT_DIR not set.")

    estimator = build_estimator()

    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    result_dir = output_dir / "result"
    result_dir.mkdir(parents=True, exist_ok=True)

    images = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                images.append(Path(root) / f)
    images.sort(key=lambda x: str(x.relative_to(input_dir)))
    if not images:
        sys.exit(f"ERROR: no images in {input_dir}")
    print(f"[*] {len(images)} image(s): {input_dir} -> {result_dir}  "
          f"(detector={DETECTOR_NAME or 'none'} segmentor={SEGMENTOR_NAME or 'none'} "
          f"fov={FOV_NAME or 'none'} bbox_thr={BBOX_THRESH} use_mask={USE_MASK} "
          f"inference={INFERENCE_TYPE})")

    infer_times = []
    ok = 0
    t_loop0 = time.time()
    for i, fp in enumerate(images, 1):
        rel = fp.relative_to(input_dir)
        result_path = result_dir / rel.with_suffix(".jpg")
        result_path.parent.mkdir(parents=True, exist_ok=True)

        t1 = time.time()
        try:
            outputs = estimator.process_one_image(
                str(fp),
                bbox_thr=BBOX_THRESH,
                use_mask=USE_MASK,
                inference_type=INFERENCE_TYPE,
            )
            dt = time.time() - t1

            if not outputs:
                print(f"[{i}/{len(images)}] {fp.name}  | no humans detected ({dt:.2f}s)")
                continue

            img = cv2.imread(str(fp))
            rend_img = visualize_sample_together(img, outputs, estimator.faces)
            cv2.imwrite(str(result_path), rend_img.astype(np.uint8))
            save_mesh_and_npz(img, outputs, estimator.faces, rel, output_dir)

            infer_times.append(dt)
            ok += 1
            n_p = len(outputs)
            print(f"[{i}/{len(images)}] {fp.name}  ->  {rel.with_suffix('.jpg').as_posix()}  "
                  f"| {n_p} person(s) | 推理 {dt:.2f}s")
        except Exception as e:
            print(f"[{i}/{len(images)}] {fp.name}  ! failed: {e}", file=sys.stderr)

    loop_time = time.time() - t_loop0
    pure = sum(infer_times)
    print(f"[*] done. {ok}/{len(images)} succeeded. "
          f"循环 {loop_time:.2f}s (其中纯推理 {pure:.2f}s)")
    if infer_times:
        avg = pure / len(infer_times)
        print(f"[*] 单图推理耗时: avg {avg:.2f}s, min {min(infer_times):.2f}s, "
              f"max {max(infer_times):.2f}s, 共 {len(infer_times)} 张")


if __name__ == "__main__":
    main()
