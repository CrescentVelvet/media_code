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
  SPEED_MERGE_LORA, SPEED_CACHE_TEXT, SPEED_COMPILE, SPEED_COMPILE_MODE,
  SPEED_COMPILE_VAE, SPEED_COMPILE_FORCE
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
# 优化点 1 的替代提速路线（TorchScript 经 grill-me 否决后定的三项）：
#   SPEED_MERGE_LORA  LoRA 合并进 base 权重 → 去掉每层额外两路小矩阵乘
#   SPEED_CACHE_TEXT  prompt 恒定（本流程恒为 ""）→ text_encoder 只跑一次
#   SPEED_COMPILE     torch.compile UNet（需输入形状恒定，见 main() 门控）
# 三项互不依赖，可单独关掉做 A/B；任一项失败都降级回原路径并告警，不影响正确性。
SPEED_MERGE_LORA = os.environ.get("SPEED_MERGE_LORA", "1") == "1"
SPEED_CACHE_TEXT = os.environ.get("SPEED_CACHE_TEXT", "1") == "1"
# compile 默认 auto：在 merge_lora + cache_text 已生效的基础上，compile 的增量只有
# 142→130ms（省 ~12ms/张），一次性编译却要 ~28s → 约 2300 张才回本（3090 / 512²/
# vggt_human env 实测）。小批量编译是净亏，故按图片数自动决定；=1 强制、=0 关闭。
SPEED_COMPILE = os.environ.get("SPEED_COMPILE", "auto").lower()
SPEED_COMPILE_MIN_IMAGES = int(os.environ.get("SPEED_COMPILE_MIN_IMAGES", "2400"))
SPEED_COMPILE_MODE = os.environ.get("SPEED_COMPILE_MODE", "reduce-overhead")
SPEED_COMPILE_VAE = os.environ.get("SPEED_COMPILE_VAE", "0") == "1"
SPEED_COMPILE_FORCE = os.environ.get("SPEED_COMPILE_FORCE", "0") == "1"

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
# 推理提速三项（merge LoRA / text embed 缓存 / torch.compile）
# ---------------------------------------------------------------------------
def merge_lora(model):
    """把 LoRA 合并进 UNet 的 base 权重。

    SD2Enhancer 用 diffusers 的 add_adapter（底层 peft.inject_adapter_in_model）
    注入 LoRA，所以没有 LoraModel 的 merge_and_unload()，需要逐层调用
    peft LoraLayer.merge()：delta = (B @ A) * scaling 累加进 base_layer.weight。
    合并后推理不再走 lora_A/lora_B 两路额外矩阵乘（r=256、SD2.1 UNet 共 257 层，
    这部分开销可观）。实测 161→139ms/张（1.16x，0 预热成本）。

    数值：peft 的 merge 把 delta 累加进 base weight，bf16 累加引入误差。对齐调用
    序号对比（见下方 cache_text_embed 关于随机流的说明），merge 前后输出
    mean|Δ| = 0.35/255（≈0.14%），可忽略。
    """
    n = 0
    for m in model.G.modules():
        if not hasattr(m, "lora_A"):
            continue
        try:
            if bool(getattr(m, "merged", False)):   # peft 里 merged 是已合并 adapter 列表
                continue
            m.merge()
            n += 1
        except Exception as e:
            print(f"  ⚠️ merge LoRA 失败于 {type(m).__name__}: {e}", file=sys.stderr)
    return n


def cache_text_embed(model):
    """缓存空 prompt 的 text embed。

    BaseEnhancer.enhance() 每张图都调 prepare_inputs() → tokenizer + CLIP
    text_encoder 前向。本流程 prompt 恒为 ""，embed 恒定，只需算一次。
    实测 170→161ms/张（1.05x，0 预热成本）。数值上完全等价：text_encoder 前向
    是确定性的、也不消耗 RNG，缓存只是跳过重复计算，输出逐位不变。
    实现为 monkey patch：命中缓存时直接把 self.inputs 指回上次的结果
    （下游 forward_generator 只读不写）。

    ⚠️ 做 A/B 正确性对比时的坑：HYPIR 的 vae.encode().latent_dist.sample() 每次
    前向都从同一个（set_seed 固定的）随机流取新样本，所以输出对「第几次调用」
    敏感 —— 相邻两次前向就有 mean|Δ|≈0.011（0-1 尺度）的差异，但这不是任何改动
    造成的。对比提速前后必须对齐调用序号（例如跑两次完整脚本比同名图片），
    否则会量到 0.01 量级的假差异。固定 seed 下同一调用序号是逐位可复现的
    （实测两次独立运行 mean|Δ|=0.000/255）。
    """
    orig = model.prepare_inputs
    cache = {}

    def wrapped(batch_size, prompt):
        key = (batch_size, prompt)
        if key in cache:
            model.inputs = cache[key]
            return
        orig(batch_size, prompt)
        cache[key] = model.inputs

    model.prepare_inputs = wrapped
    return cache


def compile_unet(model, mode, compile_vae=False):
    """torch.compile UNet（单步扩散的主要开销）。

    两个必须踩过的坑（3090 / 512×512 / vggt_human env 实测）：
    1. diffusers 的 Attention.forward 里有 inspect.signature(self.processor.__call__)，
       dynamo 会对每个 attention block 的 processor 对象 id 加 guard，UNet 十几个
       block 各要一份编译。默认 cache_size_limit=8 撑爆后 dynamo 放弃编译、退回
       eager 并保留 dynamo 开销 → 实测 175ms 反而变成 1687ms（慢 10 倍）。
       抬到 64 之后 3 次 warmup 内收敛，稳态 129ms 且不再重编译。
    2. 形状必须恒定，否则按新形状反复重编译；reduce-overhead 还会抓 CUDA graph，
       遇到未捕获的新形状可能直接报错。故由调用方按输入尺寸做门控。
    """
    import torch
    try:
        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.accumulated_cache_size_limit = 128
    except Exception as e:
        print(f"  ⚠️ 设置 dynamo cache_size_limit 失败: {e}", file=sys.stderr)
    n = 1
    model.G = torch.compile(model.G, mode=mode)
    if compile_vae:
        # VAE decode 在 512 下也是可观开销。风险：decode 返回的是
        # DecoderOutput，dynamo 拆包失败时会抛异常 → 捕获后静默跳过。
        try:
            model.vae.decode = torch.compile(model.vae.decode, mode=mode)
            n += 1
        except Exception as e:
            print(f"  ⚠️ VAE compile 失败，跳过: {e}", file=sys.stderr)
    return n


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

    # ── 2b. 推理提速三项（任一项失败都降级回原路径，不影响正确性） ──────────
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

    if SPEED_MERGE_LORA:
        n = merge_lora(model)
        print(f"  ⚡ merge LoRA: {n} 层已合并进 base 权重")
    if SPEED_CACHE_TEXT:
        cache_text_embed(model)
        print("  ⚡ text embed 缓存已启用（prompt 恒定，text_encoder 只跑一次）")
    if SPEED_COMPILE == "1":
        do_compile = True
    elif SPEED_COMPILE in ("0", "false", "off", "no"):
        do_compile = False
    else:  # auto：按批量大小与回本线比较（一次性编译 ~30s，每张省 ~41ms）
        do_compile = len(images) >= SPEED_COMPILE_MIN_IMAGES
        if not do_compile:
            print(f"  ℹ️ compile=auto：{len(images)} 张 < 回本线 {SPEED_COMPILE_MIN_IMAGES} 张"
                  f"（一次性编译 ~28s，每张仅省 ~12ms）→ 本次不编译")
    if do_compile:
        sizes = {Image.open(os.path.join(input_images_dir, f)).size for f in images}
        if len(sizes) == 1 and (WHOLE_IMAGE or SPEED_COMPILE_FORCE):
            n = compile_unet(model, SPEED_COMPILE_MODE, SPEED_COMPILE_VAE)
            print(f"  ⚡ torch.compile({SPEED_COMPILE_MODE}) 已启用: "
                  f"{'UNet+VAE' if n > 1 else 'UNet'}（首图含 ~30s 编译开销，后续摊销）")
        else:
            reason = (f"{len(sizes)} 种输入尺寸" if len(sizes) > 1
                      else "非 WHOLE_IMAGE 模式，crop/tile 尺寸不恒定")
            print(f"  ⚠️ 跳过 compile：{reason}（形状不恒定会反复重编译，"
                  f"SPEED_COMPILE_FORCE=1 可强制）")

    # ── 3. Process images ────────────────────────────────────────────────────
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
