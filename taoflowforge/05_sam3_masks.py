#!/usr/bin/env python3
"""05_sam3_masks.py — 带 prompt 回退链的 SAM3 分割驱动（sam3 conda env 运行）。

替代对 sam3_face_masks_worker.py 的直接调用：worker 只接受单一 SAM3_PROMPT，
本驱动对每个物体接收逗号分隔的**回退链**（如 "blender,juicer,kitchen appliance"），
逐图从链首开始尝试，取第一个 score >= MIN_SCORE 的实例，避免个别视角
词表盲区（实测：X 形路由器 "router" 全灭、"x shaped router" 命中；
某视角榨汁机 "blender" 全灭、"kitchen appliance" 命中）。

产物命名与 worker 完全一致（{stem}.p00.{alpha,mask}.png + masks_manifest.json），
下游 05_compose_preprocessed.py 无需改动。

用法：
  SAM3_PROMPTS="blender,juicer" python 05_sam3_masks.py <IMAGES_DIR> <MASKS_DIR>

Env: SAM3_CKPT / SAM3_BPE / MIN_SCORE（默认 0.5，同 worker 语义）
"""
import json
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, "/home/velvet/repos/sam3")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SAM3_CKPT = os.environ.get("SAM3_CKPT", "/mnt/d/wheel/vggt_human_ms/sam3/sam3.pt")
SAM3_BPE = os.environ.get(
    "SAM3_BPE",
    "/home/velvet/repos/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
)
MIN_SCORE = float(os.environ.get("MIN_SCORE", "0.5"))
PROMPTS = [p.strip() for p in os.environ.get("SAM3_PROMPTS", "").split(",") if p.strip()]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

from sam3.model_builder import build_sam3_image_model  # noqa: E402
from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402


def save_pair(out_dir, stem, mask_bool, alpha_u8):
    """与 worker 命名一致：{stem}.p00.{alpha,mask}.png（单实例即 p00）。"""
    binary = mask_bool.astype(np.uint8) * 255
    Image.fromarray(alpha_u8).save(os.path.join(out_dir, f"{stem}.p00.alpha.png"))
    Image.fromarray(binary).save(os.path.join(out_dir, f"{stem}.p00.mask.png"))


def main():
    images_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(images_dir)
                   if os.path.splitext(f)[1].lower() in IMG_EXTS)
    if not names:
        sys.exit(f"no images in {images_dir}")
    if not PROMPTS:
        sys.exit("ERROR: SAM3_PROMPTS 未设置")

    print(f"🧑 SAM3 fallback driver: {len(names)} images, prompts={PROMPTS}")
    t0 = time.time()
    model = build_sam3_image_model(
        bpe_path=SAM3_BPE, checkpoint_path=SAM3_CKPT,
        load_from_HF=False, device="cuda", eval_mode=True,
    )
    proc = Sam3Processor(model)
    print(f"  model ready in {time.time()-t0:.1f}s")

    manifest, prompt_stats = {}, {}
    for idx, name in enumerate(names, 1):
        stem = os.path.splitext(name)[0]
        img = Image.open(os.path.join(images_dir, name)).convert("RGB")
        state = proc.set_image(img)
        hit_prompt, best = None, None
        for p in PROMPTS:
            out = proc.set_text_prompt(prompt=p, state=state)
            insts = []
            for i in range(len(out["scores"])):
                m = out["masks"][i].detach().cpu().numpy().squeeze().astype(bool)
                s = float(out["scores"][i])
                if s >= MIN_SCORE and m.sum() > 0:
                    box = ([round(float(x)) for x in out["boxes"][i]]
                           if i < len(out.get("boxes", [])) else None)
                    insts.append((m, s, box))
            if insts:
                insts.sort(key=lambda t: -t[1])
                hit_prompt, best = p, insts[0]
                break
        prompt_stats[hit_prompt] = prompt_stats.get(hit_prompt, 0) + 1
        if best is None:
            print(f"  [{idx}/{len(names)}] {name}: 全链无实例 ⚪")
            continue
        m, s, box = best
        alpha = (m.astype(np.float32))
        alpha_u8 = (alpha * 255).astype(np.uint8)
        save_pair(out_dir, stem, m, alpha_u8)
        manifest[stem] = [{
            "pid": 0, "score": round(s, 4), "box": box,
            "coverage": round(m.sum() / m.size * 100, 2),
            "prompt": hit_prompt,
        }]
        print(f"  [{idx}/{len(names)}] {name}: '{hit_prompt}' "
              f"cov={manifest[stem][0]['coverage']}% score={s:.3f}")

    with open(os.path.join(out_dir, "masks_manifest.json"), "w",
              encoding="utf-8", newline="\n") as f:
        json.dump({"backend": "sam3-fallback", "prompts": PROMPTS,
                   "frames": manifest}, f, ensure_ascii=False, indent=2)
    print(f"✅ {len(manifest)}/{len(names)} masked, "
          f"prompt 命中分布={prompt_stats}, {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
