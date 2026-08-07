#!/usr/bin/env python3
"""Pick the front-facing image from an orbit shoot and segment the person.

Pipeline:
1. Load SAM 3D Body (model + detector + segmentor + FOV estimator).
2. Walk INPUT_DIR/image/ recursively -> collect all images.
3. For each image: process_one_image -> 3D body outputs (global_rot, bbox, ...).
4. Compute a front-facing score for each (using global_rot).
5. Pick the image with the best (highest) score.
6. Segment the person in that image, trying in order:
   a. HumanSegmentor (SAM2/SAM3) — pixel-accurate mask.
   b. 3D mesh silhouette (pyrender render of pred_vertices).
   c. Bounding box with padding (last resort).
7. Apply mask: person pixels kept, background -> white (255,255,255).
8. Save: segmented_image.png + front_facing_original.jpg + debug info.

Env vars (set by 01_pick_and_segment.sh):
  SAM3D_DIR, CHECKPOINT_PATH, MHR_PATH, DEVICE,
  DETECTOR_NAME, DETECTOR_PATH, SEGMENTOR_NAME, SEGMENTOR_PATH,
  FOV_NAME, FOV_PATH, INPUT_DIR, OUTPUT_DIR,
  BBOX_THRESH, INFERENCE_TYPE, FRONTAL_SIGN, WHITE_BG, PADDING
"""
import os
import sys
import time
import csv
from pathlib import Path

import numpy as np
import cv2

SAM3D_DIR = os.environ.get("SAM3D_DIR", "../sam-3d-body")
sys.path.insert(0, SAM3D_DIR)

import torch  # noqa: E402

from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator  # noqa: E402

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "")
MHR_PATH = os.environ.get("MHR_PATH", "")
DEVICE = os.environ.get("DEVICE", "cuda")
DETECTOR_NAME = os.environ.get("DETECTOR_NAME", "vitdet")
DETECTOR_PATH = os.environ.get("DETECTOR_PATH", "")
SEGMENTOR_NAME = os.environ.get("SEGMENTOR_NAME", "sam2")
SEGMENTOR_PATH = os.environ.get("SEGMENTOR_PATH", "")
FOV_NAME = os.environ.get("FOV_NAME", "moge2")
FOV_PATH = os.environ.get("FOV_PATH", "")
INPUT_DIR = os.environ.get("INPUT_DIR", "")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "")
BBOX_THRESH = float(os.environ.get("BBOX_THRESH", "0.8"))
INFERENCE_TYPE = os.environ.get("INFERENCE_TYPE", "body")
FRONTAL_SIGN = int(os.environ.get("FRONTAL_SIGN", "1"))
WHITE_BG = os.environ.get("WHITE_BG", "1") == "1"
PADDING = float(os.environ.get("PADDING", "0.1"))

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def build_estimator():
    device = DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        print("[!] CUDA not available -- falling back to CPU (will be slow).")
        device = "cpu"

    print(f"[*] loading SAM 3D Body model (device={device}) ...")
    t0 = time.time()
    model, model_cfg = load_sam_3d_body(
        CHECKPOINT_PATH, device=device, mhr_path=MHR_PATH
    )
    print(f"[*] model loaded in {time.time() - t0:.1f}s")

    human_detector = human_segmentor = fov_estimator = None

    if DETECTOR_NAME:
        print(f"[*] loading detector: {DETECTOR_NAME}")
        from tools.build_detector import HumanDetector
        human_detector = HumanDetector(
            name=DETECTOR_NAME, device=device, path=DETECTOR_PATH
        )

    need_seg = SEGMENTOR_NAME and (SEGMENTOR_NAME != "sam2" or len(SEGMENTOR_PATH))
    if need_seg:
        print(f"[*] loading segmentor: {SEGMENTOR_NAME}")
        from tools.build_sam import HumanSegmentor
        human_segmentor = HumanSegmentor(
            name=SEGMENTOR_NAME, device=device, path=SEGMENTOR_PATH
        )

    if FOV_NAME:
        print(f"[*] loading FOV estimator: {FOV_NAME}")
        from tools.build_fov_estimator import FOVEstimator
        fov_estimator = FOVEstimator(
            name=FOV_NAME, device=device, path=FOV_PATH
        )

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )
    return estimator, human_segmentor


def _to_rotmat(rot):
    rot = np.asarray(rot, dtype=np.float64).squeeze()
    if rot.shape == (3, 3):
        return rot
    if rot.shape == (3,):
        angle = np.linalg.norm(rot)
        if angle < 1e-8:
            return np.eye(3)
        mat, _ = cv2.Rodrigues(rot)
        return mat
    if rot.shape == (4,):
        w, x, y, z = rot
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
    return np.eye(3)


def compute_frontal_score(output):
    """Higher = more front-facing.

    Uses global_rot to get the person's forward direction in camera frame.
    SMPL rest pose faces +Z; after rotation R, forward = R @ [0,0,1].
    Front-facing (toward camera at -Z): forward[2] < 0  =>  score = -forward[2].
    If your convention is the opposite, set FRONTAL_SIGN=-1.
    """
    gr = output.get("global_rot")
    if gr is None:
        return 0.0
    R = _to_rotmat(gr)
    forward = R @ np.array([0.0, 0.0, 1.0])
    return FRONTAL_SIGN * (-forward[2])


def _bbox_area(bbox):
    if bbox is None:
        return 0
    b = np.asarray(bbox).squeeze()
    if b.shape == (4,):
        return max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return 0


def get_padded_bbox(output, img_h, img_w, padding):
    bbox = output.get("bbox")
    if bbox is None:
        return None
    b = np.asarray(bbox).squeeze()
    if b.shape == (4,):
        x1, y1, x2, y2 = b
    elif b.ndim == 2 and b.shape[1] == 4:
        x1, y1, x2, y2 = b[0]
    else:
        return None
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * padding))
    y1 = max(0, int(y1 - bh * padding))
    x2 = min(img_w, int(x2 + bw * padding))
    y2 = min(img_h, int(y2 + bh * padding))
    return [x1, y1, x2, y2]


# ── segmentation methods (tried in order) ──

def segment_with_segmentor(human_segmentor, image_bgr, bboxes):
    """Try calling the HumanSegmentor with several plausible interfaces."""
    if human_segmentor is None or not bboxes:
        return None
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    np_bboxes = np.array(bboxes, dtype=np.float32) if bboxes else None

    for img in (image_rgb, image_bgr):
        for bboxes_arg in (bboxes, np_bboxes):
            for desc, fn in [
                ("__call__", lambda: human_segmentor(img, bboxes_arg)),
                ("predict", lambda: human_segmentor.predict(img, bboxes_arg)),
            ]:
                try:
                    result = fn()
                except Exception:
                    continue
                mask = _extract_mask(result)
                if mask is not None:
                    print(f"    [segmentor] OK via {desc}")
                    return mask
    return None


def _extract_mask(result):
    if result is None:
        return None
    if isinstance(result, np.ndarray):
        m = result.squeeze()
        if m.ndim == 2:
            return (m > 0).astype(np.uint8)
        if m.ndim == 3 and m.shape[0] == 1:
            return (m[0] > 0).astype(np.uint8)
    if isinstance(result, (list, tuple)):
        for item in result:
            mask = _extract_mask(item)
            if mask is not None:
                return mask
    return None


def segment_with_mesh_silhouette(estimator, output, img_h, img_w):
    """Render the 3D body mesh as a binary silhouette mask via pyrender."""
    try:
        from sam_3d_body.visualization.renderer import Renderer
        import pyrender
        import trimesh

        focal = output["focal_length"]
        if isinstance(focal, (list, tuple, np.ndarray)):
            fx = float(np.asarray(focal).ravel()[0])
        else:
            fx = float(focal)

        renderer = Renderer(focal_length=focal, faces=estimator.faces)
        tmesh = renderer.vertices_to_trimesh(
            output["pred_vertices"],
            output["pred_cam_t"],
            (1.0, 1.0, 1.0),
        )

        py_r = pyrender.OffscreenRenderer(
            viewport_width=img_w, viewport_height=img_h
        )
        scene = pyrender.Scene(bg_color=[0, 0, 0, 0])
        mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[1.0, 1.0, 1.0, 1.0],
            metallic=0.0,
            roughness=1.0,
        )
        pmesh = pyrender.Mesh.from_trimesh(tmesh, material=mat)
        scene.add(pmesh)

        yfov = 2.0 * np.arctan(img_h / (2.0 * fx))
        cam = pyrender.PerspectiveCamera(yfov=yfov)
        scene.add(cam, pose=np.eye(4))

        color, _ = py_r.render(scene)
        py_r.delete()

        mask = (color.sum(axis=2) > 10).astype(np.uint8)
        print("    [mesh_silhouette] OK")
        return mask
    except Exception as e:
        print(f"    [mesh_silhouette] failed: {e}")
        return None


def segment_with_bbox(bbox, img_h, img_w):
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    print("    [bbox] OK (rough mask)")
    return mask


def apply_mask(image_bgr, mask, white_bg=True):
    result = image_bgr.copy()
    bg = mask == 0
    result[bg] = (255, 255, 255) if white_bg else (0, 0, 0)
    return result


def main():
    if not CHECKPOINT_PATH:
        sys.exit("ERROR: CHECKPOINT_PATH not set.")
    if not INPUT_DIR:
        sys.exit("ERROR: INPUT_DIR not set.")
    if not OUTPUT_DIR:
        sys.exit("ERROR: OUTPUT_DIR not set.")

    image_dir = Path(INPUT_DIR) / "image"
    if not image_dir.is_dir():
        image_dir = Path(INPUT_DIR)
    if not image_dir.is_dir():
        sys.exit(f"ERROR: image dir not found: {image_dir}")

    images = []
    for root, _, files in os.walk(image_dir):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                images.append(Path(root) / f)
    images.sort(key=lambda x: str(x.relative_to(image_dir)))

    if not images:
        sys.exit(f"ERROR: no images in {image_dir}")

    print(f"[*] {len(images)} images in {image_dir}")

    estimator, human_segmentor = build_estimator()

    results = []
    t_loop0 = time.time()
    use_mask = human_segmentor is not None

    for i, fp in enumerate(images, 1):
        rel = fp.relative_to(image_dir)
        t1 = time.time()
        try:
            outputs = estimator.process_one_image(
                str(fp),
                bbox_thr=BBOX_THRESH,
                use_mask=use_mask,
                inference_type=INFERENCE_TYPE,
            )
            dt = time.time() - t1

            if not outputs:
                print(f"[{i}/{len(images)}] {fp.name}  | no person ({dt:.2f}s)")
                continue

            best = max(outputs, key=lambda o: _bbox_area(o.get("bbox")))
            score = compute_frontal_score(best)

            results.append({
                "path": str(fp),
                "rel": str(rel),
                "score": score,
                "n_persons": len(outputs),
                "output": best,
            })
            print(f"[{i}/{len(images)}] {fp.name}  | {len(outputs)}p  "
                  f"score={score:.3f}  ({dt:.2f}s)")
        except Exception as e:
            print(f"[{i}/{len(images)}] {fp.name}  ! failed: {e}", file=sys.stderr)

    loop_time = time.time() - t_loop0
    print(f"[*] processed {len(results)}/{len(images)} images in {loop_time:.1f}s")

    if not results:
        sys.exit("ERROR: no person detected in any image.")

    best_result = max(results, key=lambda r: r["score"])
    print(f"\n[*] picked: {best_result['rel']}  (score={best_result['score']:.3f})")

    image_bgr = cv2.imread(best_result["path"])
    if image_bgr is None:
        sys.exit(f"ERROR: cannot read image: {best_result['path']}")
    img_h, img_w = image_bgr.shape[:2]
    bbox = get_padded_bbox(best_result["output"], img_h, img_w, PADDING)

    mask = None

    if human_segmentor is not None:
        print("[*] segmenting with HumanSegmentor ...")
        mask = segment_with_segmentor(human_segmentor, image_bgr, [bbox] if bbox else None)

    if mask is None:
        print("[*] segmenting with 3D mesh silhouette ...")
        mask = segment_with_mesh_silhouette(estimator, best_result["output"], img_h, img_w)

    if mask is None:
        print("[*] falling back to bbox mask ...")
        mask = segment_with_bbox(bbox, img_h, img_w)

    if mask is None:
        sys.exit("ERROR: all segmentation methods failed.")

    print(f"[*] mask pixels: {int(mask.sum())} / {img_h * img_w}")

    result_img = apply_mask(image_bgr, mask, white_bg=WHITE_BG)

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    seg_path = out_dir / "segmented_image.png"
    cv2.imwrite(str(seg_path), result_img)
    print(f"[*] saved: {seg_path}")

    orig_path = out_dir / "front_facing_original.jpg"
    cv2.imwrite(str(orig_path), image_bgr)
    print(f"[*] saved: {orig_path}")

    mask_path = out_dir / "debug_mask.png"
    cv2.imwrite(str(mask_path), mask * 255)

    scores_path = out_dir / "frontal_scores.csv"
    with open(scores_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "frontal_score", "n_persons"])
        for r in sorted(results, key=lambda x: -x["score"]):
            writer.writerow([r["rel"], f"{r['score']:.4f}", r["n_persons"]])
    print(f"[*] saved: {scores_path}")

    script_dir = Path(__file__).resolve().parent
    print(f"\n[*] Done. Next:")
    print(f"    WEIGHT_PATH=<lora.safetensors> bash {script_dir}/02_generate_video.sh")


if __name__ == "__main__":
    main()
