#!/usr/bin/env python3
"""face_enhance.py — MediaPipe face detection + HYPIR enhancement + gradient blending.

Two modes:
  默认 (WHOLE_IMAGE=0): 检测人脸 bbox → 裁剪 → HYPIR → 羽化融合回原图。
  WHOLE_IMAGE=1 (优化点 2): 整张图就是人脸特写 (CLOSEUP_SIZE=512 的近景渲染),
    跳过检测/裁剪/融合, 整图单次前向进 HYPIR (无 tiling)。
    需要配 CENTER_BOX 无关 — 无任何检测。

For each image in the input directory:
  1. MediaPipe BlazeFace detects face bounding boxes. (WHOLE_IMAGE 模式跳过)
  2. Each bbox is enlarged by FACE_PADDING (default 0.2 = 20%).
  3. The crop is fed to the HYPIR model (SD2Enhancer with LoRA checkpoint)
     for face restoration/beautification (upscale=1, no super-resolution).
  4. The enhanced crop is blended back into the original image using a
     quadratic falloff mask (1 at center, 0 at edges) → seamless transition.
  5. The COLMAP sparse/ directory is copied unchanged (only images are enhanced).

Uses the vggt_human conda env (has diffusers/transformers/peft for HYPIR + mediapipe).

Env vars (set by 01_face_enhance.sh or 06_face_enhance.sh):
  HYPIR_DIR, HYPIR_BASE_MODEL, HYPIR_WEIGHT, INPUT_SOURCE_DIR, SOURCE_FACE_DIR,
  FACE_PADDING, UPSCALE, PATCH_SIZE, STRIDE, DEVICE, WHOLE_IMAGE
"""
import os
import sys
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image

HYPIR_DIR = os.environ.get("HYPIR_DIR", "../HYPIR")
BASE_MODEL_PATH = os.environ.get("HYPIR_BASE_MODEL", "")
WEIGHT_PATH = os.environ.get("HYPIR_WEIGHT", "")
INPUT_SOURCE_DIR = os.environ.get("INPUT_SOURCE_DIR", "")
SOURCE_FACE_DIR = os.environ.get("SOURCE_FACE_DIR", "")
FACE_PADDING = float(os.environ.get("FACE_PADDING", "0.2"))
# MediaPipe 检不出人脸时, 用画面中心框兜底 (近景相机 look-at 保证人脸在画面中央)。
# 3DGS 近景渲染的人脸过糊、MediaPipe 检不出是常态; 不开兜底 HYPIR 会整图透传。
CENTER_BOX_FALLBACK = os.environ.get("CENTER_BOX_FALLBACK", "0") == "1"
UPSCALE = int(os.environ.get("UPSCALE", "1"))
PATCH_SIZE = int(os.environ.get("PATCH_SIZE", "512"))
STRIDE = int(os.environ.get("STRIDE", "256"))
DEVICE = os.environ.get("DEVICE", "cuda")
LORA_RANK = int(os.environ.get("LORA_RANK", "256"))
LORA_MODULES = os.environ.get(
    "LORA_MODULES",
    "to_k,to_q,to_v,to_out.0,conv,conv1,conv2,conv_shortcut,conv_out,proj_in,proj_out,ff.net.2,ff.net.0.proj",
).split(",")
MODEL_T = int(os.environ.get("MODEL_T", "200"))
COEFF_T = int(os.environ.get("COEFF_T", "200"))
# 优化点 2: 整图增强 (配合 render_closeup CLOSEUP_SIZE=512)。
WHOLE_IMAGE = os.environ.get("WHOLE_IMAGE", "0") == "1"
# 优化点 3: sidecar _debug/ 标注副本 (增强结果上画参数, 不污染主输出)。
DEBUG_ANNOTATE = os.environ.get("DEBUG_ANNOTATE", "1") == "1"
# 优化点 4: 融合模式。
#   feather      = 旧的单次二次羽化融合 (默认, 兼容旧流程)
#   border_alpha = 两段式: ① border mask 融合增强结果 (crop 边缘 band 内衰减为 0,
#                  压住增强结果的边缘伪影); ② 3DGS 渲染 alpha 前景加权
#                  final = fused*alpha + orig*(1-alpha), 人脸边缘增强范围更精确。
#                  需要 ALPHA_DIR (与输入图同名的 3DGS alpha 渲染, 如 06c_closeup_alpha)。
FUSION_MODE = os.environ.get("FUSION_MODE", "feather").lower()
ALPHA_DIR = os.environ.get("ALPHA_DIR", "")
BORDER_MARGIN = float(os.environ.get("BORDER_MARGIN", "0.08"))  # 边缘 band 占 crop 短边比例

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


# ---------------------------------------------------------------------------
# Feather mask: quadratic falloff (1 at center, 0 at edges).
# ---------------------------------------------------------------------------
def create_feather_mask(h, w):
    """Create a 2D mask with quadratic falloff from center to edges."""
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    y = np.arange(h, dtype=np.float32) - cy
    x = np.arange(w, dtype=np.float32) - cx
    yy, xx = np.meshgrid(y, x, indexing="ij")
    r = np.sqrt(xx ** 2 + yy ** 2)
    r_max = max(min(cx, cy), 1.0)
    return np.clip(1.0 - (r / r_max) ** 2, 0.0, 1.0)


def create_border_mask(crop_h, crop_w, margin_frac):
    """优化点 4 ①: border mask — crop 内部为 1, 边缘 band 内平滑衰减为 0。

    与二次羽化的区别: 中心区域恒为 1 (羽化在中心就已衰减), 增强强度不被
    距离稀释; 只在边缘 band (margin_frac × 短边) 处过渡, 压住增强伪影。
    """
    band = max(int(min(crop_h, crop_w) * margin_frac), 2)
    band = min(band, (crop_h - 1) // 2, (crop_w - 1) // 2)  # 极小 crop 保护
    m = np.zeros((crop_h, crop_w), dtype=np.float32)
    m[band:crop_h - band, band:crop_w - band] = 1.0
    try:
        import cv2
        k = 2 * band + 1
        m = cv2.GaussianBlur(m, (k, k), 0)
    except ImportError:
        from scipy.ndimage import gaussian_filter
        m = gaussian_filter(m, sigma=band / 2.0)
    return np.clip(m, 0.0, 1.0)


# ---------------------------------------------------------------------------
# MediaPipe face detection.
# ---------------------------------------------------------------------------
def load_face_detector():
    import mediapipe as mp
    mp_face = mp.solutions.face_detection
    detector = mp_face.FaceDetection(
        model_selection=1,  # 1 = short-range (better for close-up faces)
        min_detection_confidence=0.3,
    )
    return detector


def detect_faces(detector, image_rgb):
    """Returns list of (x1, y1, x2, y2) in pixel coordinates."""
    import cv2
    h, w = image_rgb.shape[:2]
    results = detector.process(image_rgb)
    boxes = []
    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            x1 = max(int(bbox.xmin * w), 0)
            y1 = max(int(bbox.ymin * h), 0)
            x2 = min(int((bbox.xmin + bbox.width) * w), w)
            y2 = min(int((bbox.ymin + bbox.height) * h), h)
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
    return boxes


def enlarge_bbox(bbox, padding, img_w, img_h):
    """Enlarge bbox by padding fraction (e.g., 0.2 = 20%)."""
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    nw, nh = bw * (1 + padding), bh * (1 + padding)
    nx1 = max(int(cx - nw / 2), 0)
    ny1 = max(int(cy - nh / 2), 0)
    nx2 = min(int(cx + nw / 2), img_w)
    ny2 = min(int(cy + nh / 2), img_h)
    return nx1, ny1, nx2, ny2


def draw_annotations(pil_img, lines):
    """在图像副本左上角画标注条 (黑描边 + 黄字, 半透明底), 返回新图。"""
    import cv2
    arr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    strip_h = 14 + 17 * len(lines)
    overlay = arr.copy()
    cv2.rectangle(overlay, (0, 0), (370, strip_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, arr, 0.55, 0, arr)
    y = 20
    for line in lines:
        cv2.putText(arr, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(arr, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (60, 220, 255), 1, cv2.LINE_AA)
        y += 17
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main():
    if not INPUT_SOURCE_DIR:
        sys.exit("❌ INPUT_SOURCE_DIR not set")
    if not SOURCE_FACE_DIR:
        sys.exit("❌ SOURCE_FACE_DIR not set")
    if not WEIGHT_PATH:
        sys.exit("❌ HYPIR_WEIGHT not set")
    if not os.path.isfile(WEIGHT_PATH):
        sys.exit(f"❌ HYPIR weight not found: {WEIGHT_PATH}")
    if not BASE_MODEL_PATH:
        sys.exit("❌ HYPIR_BASE_MODEL not set")
    if not os.path.isdir(BASE_MODEL_PATH):
        sys.exit(f"❌ HYPIR base model not found: {BASE_MODEL_PATH}")

    t0 = time.time()

    # ── 1. Load MediaPipe face detector (WHOLE_IMAGE 模式跳过) ─────────────
    detector = None
    if not WHOLE_IMAGE:
        print("🔍 [1/3] loading MediaPipe face detector...")
        detector = load_face_detector()
        print("  ✅ MediaPipe loaded")
    else:
        print("🔍 [1/3] WHOLE_IMAGE=1 → 跳过 MediaPipe (整图即人脸特写)")

    # ── 2. Load HYPIR model ──────────────────────────────────────────────────
    print("🏋️ [2/3] loading HYPIR model (SD2Enhancer + LoRA)...")
    sys.path.insert(0, HYPIR_DIR)
    from HYPIR.enhancer.sd2 import SD2Enhancer  # noqa: E402
    from accelerate.utils import set_seed  # noqa: E402
    from torchvision import transforms  # noqa: E402

    set_seed(231)
    model = SD2Enhancer(
        base_model_path=BASE_MODEL_PATH,
        weight_path=WEIGHT_PATH,
        lora_modules=LORA_MODULES,
        lora_rank=LORA_RANK,
        model_t=MODEL_T,
        coeff_t=COEFF_T,
        device=DEVICE,
    )
    model.init_models()
    print(f"  ✅ HYPIR loaded (weight={WEIGHT_PATH})")

    # ── 3. Process images ────────────────────────────────────────────────────
    # Input can be: COLMAP scene (images/ subdir), test_task folder (image/ subdir),
    # or a plain image folder (images directly in INPUT_SOURCE_DIR).
    input_images_dir = os.path.join(INPUT_SOURCE_DIR, "images")
    if not os.path.isdir(input_images_dir):
        input_images_dir = os.path.join(INPUT_SOURCE_DIR, "image")
    if not os.path.isdir(input_images_dir):
        input_images_dir = INPUT_SOURCE_DIR  # plain image folder
    output_images_dir = os.path.join(SOURCE_FACE_DIR, "images")
    output_sparse_dir = os.path.join(SOURCE_FACE_DIR, "sparse", "0")
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_sparse_dir, exist_ok=True)

    images = sorted([f for f in os.listdir(input_images_dir)
                     if os.path.splitext(f)[1].lower() in IMG_EXTS])
    if not images:
        sys.exit(f"❌ no images in {input_images_dir}")

    print(f"🖼️ [3/3] enhancing faces in {len(images)} images..."
          + (f" (FUSION_MODE={FUSION_MODE}, ALPHA_DIR={ALPHA_DIR})"
             if FUSION_MODE != "feather" else ""))
    to_tensor = transforms.ToTensor()

    total_faces = 0
    for i, name in enumerate(images, 1):
        img_path = os.path.join(input_images_dir, name)
        img_pil = Image.open(img_path).convert("RGB")
        img_np = np.array(img_pil)
        h, w = img_np.shape[:2]

        if WHOLE_IMAGE:
            # 整图单次前向: 无检测、无裁剪、无融合 (优化点 2)
            t_img = time.time()
            tensor = to_tensor(img_pil).unsqueeze(0)
            pad_w = (8 - w % 8) % 8
            pad_h = (8 - h % 8) % 8
            if pad_w > 0 or pad_h > 0:
                import torch.nn.functional as F
                tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
            try:
                result = model.enhance(
                    lq=tensor,
                    prompt="",
                    scale_by="factor",
                    upscale=UPSCALE,
                    patch_size=max(tensor.shape[-1], tensor.shape[-2]),  # 单次前向, 无 tiling
                    stride=max(tensor.shape[-1], tensor.shape[-2]),
                    return_type="pil",
                )[0]
            except Exception as e:
                print(f"  [{i}/{len(images)}] ⚠️ HYPIR failed (whole image): {e}", file=sys.stderr)
                continue
            if pad_w > 0 or pad_h > 0:
                rw, rh = result.size
                result = result.crop((0, 0, rw - pad_w, rh - pad_h))
            if result.size != (w, h):
                result = result.resize((w, h), Image.LANCZOS)
            result.save(os.path.join(output_images_dir, name))
            # 优化点 3: sidecar _debug/ 标注副本
            if DEBUG_ANNOTATE:
                debug_dir = os.path.join(SOURCE_FACE_DIR, "_debug")
                os.makedirs(debug_dir, exist_ok=True)
                dt = time.time() - t_img
                lines = [
                    f"{name} | WHOLE_IMAGE enhance",
                    f"HYPIR sd2 t={MODEL_T} coeff={COEFF_T} up={UPSCALE}",
                    f"size={w}x{h} patch=full(no tiling) {dt:.1f}s",
                ]
                draw_annotations(result, lines).save(
                    os.path.join(debug_dir, name))
            total_faces += 1
            print(f"  [{i}/{len(images)}] {name} — whole-image enhanced")
            continue

        # Detect faces
        import cv2
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        # MediaPipe expects RGB input actually; let me check...
        # Actually MediaPipe's FaceDetection.process expects RGB.
        img_rgb_mp = img_np  # already RGB (PIL → numpy)
        boxes = detect_faces(detector, img_rgb_mp)

        if not boxes and CENTER_BOX_FALLBACK:
            # 中心框兜底: 近景相机 look-at 对准人脸, 人脸必在画面中央区域
            boxes = [(int(0.25 * w), int(0.15 * h), int(0.75 * w), int(0.75 * h))]

        if not boxes:
            # No face → copy as-is
            img_pil.save(os.path.join(output_images_dir, name))
            print(f"  [{i}/{len(images)}] {name} — no face, copy as-is")
            continue

        # Enhance each face
        enhanced = img_np.copy()
        face_count = 0
        for bbox in boxes:
            x1, y1, x2, y2 = enlarge_bbox(bbox, FACE_PADDING, w, h)
            crop_w, crop_h = x2 - x1, y2 - y1
            if crop_w < 32 or crop_h < 32:
                continue

            crop_pil = img_pil.crop((x1, y1, x2, y2))
            crop_tensor = to_tensor(crop_pil).unsqueeze(0)

            # Pad to nearest multiple of 8 (SD2 VAE downsampling factor).
            # Without this, VAE encode→decode produces size mismatch (e.g. 31 vs 30).
            pad_w = (8 - crop_w % 8) % 8
            pad_h = (8 - crop_h % 8) % 8
            if pad_w > 0 or pad_h > 0:
                import torch.nn.functional as F
                crop_tensor = F.pad(crop_tensor, (0, pad_w, 0, pad_h), mode='reflect')

            try:
                result = model.enhance(
                    lq=crop_tensor,
                    prompt="",
                    scale_by="factor",
                    upscale=UPSCALE,
                    patch_size=min(PATCH_SIZE, max(crop_tensor.shape[-1], crop_tensor.shape[-2])),
                    stride=min(STRIDE, max(crop_tensor.shape[-1], crop_tensor.shape[-2]) // 2),
                    return_type="pil",
                )[0]
            except Exception as e:
                print(f"  [{i}] ⚠️ HYPIR failed on face: {e}", file=sys.stderr)
                continue

            # Crop padding off the result (right/bottom), then resize to original crop
            if pad_w > 0 or pad_h > 0:
                rw, rh = result.size
                result = result.crop((0, 0, rw - pad_w, rh - pad_h))
            # Resize back to crop size if HYPIR changed resolution
            if result.size != (crop_w, crop_h):
                result = result.resize((crop_w, crop_h), Image.LANCZOS)
            result_np = np.array(result).astype(np.float32)
            original_crop = img_np[y1:y2, x1:x2].astype(np.float32)

            if FUSION_MODE == "border_alpha" and ALPHA_DIR:
                # 优化点 4: 两段式融合
                # ① border mask 融合增强结果
                border = create_border_mask(crop_h, crop_w, BORDER_MARGIN)[..., None]
                fused = original_crop * (1 - border) + result_np * border
                # ② 3DGS alpha 前景加权: final = fused*alpha + orig*(1-alpha)
                alpha_path = os.path.join(ALPHA_DIR, name)
                if os.path.isfile(alpha_path):
                    a_full = Image.open(alpha_path).convert("L")
                    if a_full.size != (w, h):
                        a_full = a_full.resize((w, h), Image.LANCZOS)
                    a = np.asarray(a_full, dtype=np.float32)[y1:y2, x1:x2] / 255.0
                    blended = fused * a[..., None] + original_crop * (1 - a[..., None])
                else:
                    # 该视角无 alpha → 退化为仅 border 融合
                    blended = fused
            else:
                # 旧路径: 单次二次羽化融合
                mask = create_feather_mask(crop_h, crop_w)
                mask_3d = mask[..., np.newaxis]  # (H, W, 1)
                blended = original_crop * (1 - mask_3d) + result_np * mask_3d
            enhanced[y1:y2, x1:x2] = blended.astype(np.uint8)
            face_count += 1

        total_faces += face_count
        Image.fromarray(enhanced).save(os.path.join(output_images_dir, name))
        print(f"  [{i}/{len(images)}] {name} — {face_count} face(s) enhanced")

    # ── 4. Copy COLMAP sparse/ (unchanged, only in post-processing mode) ──────
    input_sparse = os.path.join(INPUT_SOURCE_DIR, "sparse", "0")
    if os.path.isdir(input_sparse):
        for fname in os.listdir(input_sparse):
            shutil.copy(os.path.join(input_sparse, fname),
                        os.path.join(output_sparse_dir, fname))
        print(f"  ✅ copied sparse/0/ ({len(os.listdir(output_sparse_dir))} files)")
    else:
        print("  (no sparse/ — pre-processing mode)")

    # Free GPU
    del model
    import torch
    torch.cuda.empty_cache()

    print(f"\n🎉 Done. {total_faces} faces enhanced in {len(images)} images. "
          f"{time.time() - t0:.1f}s")
    print(f"  output: {SOURCE_FACE_DIR}")
    print(f"  Next: bash vggt_human/07_train_denoise.sh")


if __name__ == "__main__":
    main()
