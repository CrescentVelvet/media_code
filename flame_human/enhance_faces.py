#!/usr/bin/env python3
"""enhance_faces.py — 阶段 01d：HYPIR 增强 p0 人脸区域。

动机（2026-09-10 诊断链）：head 分支柔化的真因不是高斯粒度/监督区/优化不足，
而是 **L1/SSIM 在残余不确定性（7px 对齐误差+时序不一致）下偏好平滑解**。
提升监督图的人脸高频上限是对症手段之一（pipeline 设计里 HYPIR 即为阶段五/八
的人脸增强）。

做法（复用 vggt_human/face_enhance.py 的裁剪+羽化融合思路）：
  1. 03_match 给出 p0 每帧脸框 → 按 FACE_PADDING 外扩裁剪；
  2. 裁剪图 pad 到 8 的倍数 → SD2Enhancer.enhance() → 裁掉 pad、resize 回原尺寸；
  3. 二次羽化 mask 融合回原图（中心 1 → 边缘 0），保留人脸外像素不变；
  4. 输出整幅图到 OUT_DIR/images/（08 用 IMAGES_DIR 指向它）；无脸帧原样拷贝。

Env: MATCH_JSON / SRC_IMAGES_DIR / OUT_DIR / HYPIR_DIR / HYPIR_BASE_MODEL /
     HYPIR_WEIGHT / LORA_RANK / MODEL_T / COEFF_T / FACE_PADDING / UPSCALE /
     PATCH_SIZE / STRIDE / DEVICE / LIMIT / DEBUG_DIR
"""
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HYPIR_DIR = os.environ.get("HYPIR_DIR", "")
if HYPIR_DIR:
    sys.path.insert(0, HYPIR_DIR)

LORA_RANK = int(os.environ.get("LORA_RANK", "256"))
LORA_MODULES = os.environ.get(
    "LORA_MODULES",
    "to_k,to_q,to_v,to_out.0,conv,conv1,conv2,conv_shortcut,conv_out,"
    "proj_in,proj_out,ff.net.2,ff.net.0.proj").split(",")
MODEL_T = int(os.environ.get("MODEL_T", "200"))
COEFF_T = int(os.environ.get("COEFF_T", "200"))
FACE_PADDING = float(os.environ.get("FACE_PADDING", "0.25"))
UPSCALE = int(os.environ.get("UPSCALE", "2"))
PATCH_SIZE = int(os.environ.get("PATCH_SIZE", "512"))
STRIDE = int(os.environ.get("STRIDE", "256"))
DEVICE = os.environ.get("DEVICE", "cuda")
LIMIT = int(os.environ.get("LIMIT", "0"))          # >0 只处理前 N 帧（冒烟）
DEBUG_DIR = os.environ.get("DEBUG_DIR", "")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def create_feather_mask(h, w):
    """中心 1 → 边缘 0 的二次羽化（同 vggt_human/face_enhance.py）。"""
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    y = np.arange(h, dtype=np.float32) - cy
    x = np.arange(w, dtype=np.float32) - cx
    yy, xx = np.meshgrid(y, x, indexing="ij")
    r = np.sqrt(xx ** 2 + yy ** 2)
    r_max = max(min(cx, cy), 1.0)
    return np.clip(1.0 - (r / r_max) ** 2, 0.0, 1.0)


def main():
    import json
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    from accelerate.utils import set_seed
    from HYPIR.enhancer.sd2 import SD2Enhancer

    match_json = Path(os.environ["MATCH_JSON"])
    src_dir = Path(os.environ["SRC_IMAGES_DIR"])
    out_dir = Path(os.environ["OUT_DIR"])
    out_images = out_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)
    dbg = Path(DEBUG_DIR) if DEBUG_DIR else None
    if dbg:
        dbg.mkdir(parents=True, exist_ok=True)

    match = json.loads(match_json.read_text())["frames"]
    imgs = sorted([p for p in src_dir.iterdir()
                   if p.suffix.lower() in IMG_EXTS])
    if LIMIT > 0:
        imgs = imgs[:LIMIT]
    print(f"🖼️  {len(imgs)} 张源图 → {out_images}")

    print("🏋️ 加载 HYPIR (SD2Enhancer + LoRA)...")
    set_seed(231)
    model = SD2Enhancer(
        base_model_path=os.environ["HYPIR_BASE_MODEL"],
        weight_path=os.environ["HYPIR_WEIGHT"],
        lora_modules=LORA_MODULES, lora_rank=LORA_RANK,
        model_t=MODEL_T, coeff_t=COEFF_T, device=DEVICE)
    model.init_models()
    print("  ✅ HYPIR 就绪")

    to_tensor = transforms.ToTensor()
    n_enh = 0
    t0 = time.time()
    for i, p in enumerate(imgs):
        stem = p.stem
        dst = out_images / p.name
        # 09 渲染帧文件名带帧号前缀（000_<stem>.png）→ 剥掉再查匹配表
        key = stem
        if "_" in stem and stem.split("_", 1)[0].isdigit():
            key = stem.split("_", 1)[1]
        rec = match.get(key, {}).get("persons", {}).get("0")
        img_pil = Image.open(p).convert("RGB")
        if not rec or "bbox" not in rec:
            img_pil.save(dst)          # 无 p0 脸 → 原样
            continue
        W, H = img_pil.size
        x1, y1, x2, y2 = rec["bbox"]
        bw, bh = x2 - x1, y2 - y1
        x1 = max(0, int(x1 - bw * FACE_PADDING))
        y1 = max(0, int(y1 - bh * FACE_PADDING))
        x2 = min(W, int(x2 + bw * FACE_PADDING))
        y2 = min(H, int(y2 + bh * FACE_PADDING))
        crop = img_pil.crop((x1, y1, x2, y2))
        cw, ch = crop.size
        t = to_tensor(crop).unsqueeze(0)
        pad_w = (8 - cw % 8) % 8
        pad_h = (8 - ch % 8) % 8
        if pad_w or pad_h:
            t = F.pad(t, (0, pad_w, 0, pad_h), mode="reflect")
        try:
            res = model.enhance(
                lq=t, prompt="", scale_by="factor", upscale=UPSCALE,
                patch_size=min(PATCH_SIZE, max(t.shape[-1], t.shape[-2])),
                stride=min(STRIDE, max(t.shape[-1], t.shape[-2]) // 2),
                return_type="pil")[0]
        except Exception as e:
            print(f"  ⚠️ [{stem}] HYPIR 失败，保留原图: {e}")
            img_pil.save(dst)
            continue
        if pad_w or pad_h:
            rw, rh = res.size
            res = res.crop((0, 0, rw - pad_w, rh - pad_h))
        if res.size != (cw, ch):
            res = res.resize((cw, ch), Image.LANCZOS)
        enh = np.asarray(res, dtype=np.float32)
        orig = np.asarray(img_pil, dtype=np.float32).copy()
        f = create_feather_mask(ch, cw)[..., None]
        orig[y1:y2, x1:x2] = orig[y1:y2, x1:x2] * (1 - f) + enh * f
        Image.fromarray(orig.clip(0, 255).astype(np.uint8)).save(dst)
        n_enh += 1
        if dbg and i < 3:                      # 冒烟：裁剪对 + 整帧对
            side = np.concatenate(
                [np.asarray(crop), np.asarray(res)], axis=1)
            Image.fromarray(side).save(dbg / f"enh_{i:03d}_{stem}.png")
            full = np.concatenate(
                [np.asarray(img_pil),
                 orig.clip(0, 255).astype(np.uint8)], axis=1)
            Image.fromarray(full).save(dbg / f"full_{i:03d}_{stem}.png")
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(imgs)}] {stem} "
                  f"({time.time()-t0:.0f}s)")
    print(f"✅ 增强 {n_enh} 帧（无脸帧原样拷贝）→ {out_images}  "
          f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
