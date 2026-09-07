#!/usr/bin/env python3
"""vis_region_masks.py — 三模型监督 mask 的可视化与覆盖率自检。

把 head / body / scene 三类 mask 叠加到原图上（红=head 绿=body 蓝=scene），
并统计覆盖率与重叠/空洞，确认「三模型区域互斥且覆盖完整」。

用法（WSL）：
  python vis_region_masks.py [--n 3]
"""
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from face_center_3d import parse_colmap_cameras  # noqa: E402

RESULTS_DIR = os.environ.get("RESULTS_DIR", "/mnt/d/output/vggt_human_ms")
SOURCE_DIR = os.environ.get("SOURCE_DIR", f"{RESULTS_DIR}/03_source")
MASKS_DIR = os.environ.get("MASKS_DIR", f"{RESULTS_DIR}/03i_region_masks")
OUT_DIR = os.environ.get("OUT_DIR", f"{MASKS_DIR}/vis")
IMAGES_DIR = os.environ.get("IMAGES_DIR", f"{SOURCE_DIR}/images")
PIDS = [p.strip() for p in os.environ.get("PIDS", "00,01,02").split(",") if p.strip()]
N_VIS = int(os.environ.get("N_VIS", "3"))


def find_image(stem, images_dir):
    for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG"):
        p = Path(images_dir) / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def main():
    views = {v["stem"]: v for v in parse_colmap_cameras(SOURCE_DIR)}
    masks_dir = Path(MASKS_DIR)
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    stems = sorted({f.name.split(".")[0] for f in masks_dir.glob("*.scene.png")})
    if not stems:
        sys.exit(f"❌ 未找到 mask: {masks_dir}/*.scene.png")
    idx = np.linspace(0, len(stems) - 1, min(N_VIS, len(stems))).astype(int)
    picked = [stems[i] for i in sorted(set(idx.tolist()))]

    print(f"📊 mask 覆盖率自检（{len(stems)} 帧，抽样 {len(picked)}）")
    print(f"{'stem':>18} {'head%':>7} {'body%':>7} {'scene%':>7} {'sum%':>7} {'未覆盖%':>8}")
    overlays = []
    for stem in stems:
        scene = np.asarray(Image.open(masks_dir / f"{stem}.scene.png").convert("L"),
                           dtype=np.float32) / 255.0
        H, W = scene.shape
        head = np.zeros((H, W), dtype=np.float32)
        body = np.zeros((H, W), dtype=np.float32)
        for pid in PIDS:
            hp = masks_dir / f"{stem}.p{pid}.head.png"
            bp = masks_dir / f"{stem}.p{pid}.body.png"
            if hp.is_file():
                head = np.maximum(head, np.asarray(
                    Image.open(hp).convert("L"), dtype=np.float32) / 255.0)
            if bp.is_file():
                body = np.maximum(body, np.asarray(
                    Image.open(bp).convert("L"), dtype=np.float32) / 255.0)
        tot = H * W
        h_pct, b_pct, s_pct = head.mean() * 100, body.mean() * 100, scene.mean() * 100
        sum_pct = h_pct + b_pct + s_pct
        if stem in picked:
            print(f"{stem:>18} {h_pct:7.2f} {b_pct:7.2f} {s_pct:7.2f} "
                  f"{sum_pct:7.2f} {100-sum_pct:8.2f}")
        if stem not in picked:
            continue
        img_path = find_image(stem, IMAGES_DIR)
        if img_path is None:
            continue
        orig = np.asarray(Image.open(img_path).convert("RGB").resize((W, H)),
                          dtype=np.float32) / 255.0
        # 彩色叠加：head=红 body=绿 scene=蓝
        col = np.stack([head, body, scene], axis=-1)
        a = 0.55
        ov = (orig * (1 - a) + col * a).clip(0, 1)
        ov = (ov * 255).astype(np.uint8)
        Image.fromarray(ov).save(out_dir / f"{stem}_region_overlay.png")
        overlays.append(ov)

    if overlays:
        h = min(o.shape[0] for o in overlays)
        w = min(o.shape[1] for o in overlays)
        mos = np.concatenate([o[:h, :w] for o in overlays], axis=1)
        Image.fromarray(mos).save(out_dir / "region_mosaic.png")
        print(f"🖼️  {out_dir/'region_mosaic.png'}")


if __name__ == "__main__":
    main()
