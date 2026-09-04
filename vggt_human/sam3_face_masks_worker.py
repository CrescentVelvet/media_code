#!/usr/bin/env python3
"""sam3_face_masks_worker.py — SAM3 face masks worker (runs inside the `sam3` conda env).

Dispatched by sam2_face_masks.py when MASK_BACKEND=sam3. Lives in a separate
process because SAM3 needs its own conda env (torch 2.5.1 + source-installed
sam3 + patched perflib/fused.py), while the main pipeline env holds SAM2.

Two modes (MASK_MODE env var):
  image : per-frame image model (build_sam3_image_model + Sam3Processor).
          Fast (~0.3-0.7s/img). Per-frame instance IDs are NOT consistent
          across frames — pid ordering follows detection score (descending).
  video : video predictor session (build_sam3_video_predictor). sam3.pt ships
          with tracker weights, so obj_id IS consistent across frames — the
          natural "group by person" solution for multi-person data.

Output naming (same dir layout as the SAM2 backend):
  {stem}.p{pid:02d}.mask.png   — per-person binary mask (eroded)
  {stem}.p{pid:02d}.alpha.png  — per-person feathered soft mask
  Single-person back-compat: when a frame has exactly 1 instance, the legacy
  {stem}.mask.png / {stem}.alpha.png pair is also written.
  masks_manifest.json — per-frame instance list (pid, score, box, coverage).

Env vars:
  SAM3_CKPT    : checkpoint (default /mnt/d/wheel/vggt_human_ms/sam3/sam3.pt)
  SAM3_BPE     : BPE vocab (default /home/velvet/repos/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz)
  SAM3_PROMPT  : text prompt (default "face")
  MASK_MODE    : image | video (default image)
  MIN_SCORE    : min detection score, image mode only (default 0.5)
  SOFT_FEATHER / ERODE_PX : same semantics as sam2_face_masks.py
"""
import os
import re
import sys
import json
import time
import shutil
import argparse

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sam2_face_masks import (  # noqa: E402
    IMG_EXTS,
    SOFT_FEATHER,
    ERODE_PX,
    MIN_SCORE,
    create_feather_alpha,
    erode_mask,
)

SAM3_CKPT = os.environ.get("SAM3_CKPT", "/mnt/d/wheel/vggt_human_ms/sam3/sam3.pt")
SAM3_BPE = os.environ.get(
    "SAM3_BPE",
    "/home/velvet/repos/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
)
SAM3_PROMPT = os.environ.get("SAM3_PROMPT", "face")
MASK_MODE = os.environ.get("MASK_MODE", "image").lower()


def mask_to_bool(m):
    if hasattr(m, "detach"):
        m = m.detach().cpu().numpy()
    return np.asarray(m).squeeze().astype(bool)


def save_pair(out_dir, stem, pid, binary_bool):
    """Save per-person mask+alpha; pid=None → legacy single-face naming."""
    binary = binary_bool.astype(np.uint8) * 255
    binary_eroded = erode_mask(binary, ERODE_PX)
    if SOFT_FEATHER:
        alpha = create_feather_alpha(binary_bool, padding_px=5)
        alpha_u8 = (alpha * 255).astype(np.uint8)
    else:
        alpha_u8 = binary.copy()
    tag = f".p{pid:02d}" if pid is not None else ""
    Image.fromarray(alpha_u8).save(os.path.join(out_dir, f"{stem}{tag}.alpha.png"))
    Image.fromarray(binary_eroded).save(os.path.join(out_dir, f"{stem}{tag}.mask.png"))


def run_image_mode(names, images_dir, out_dir):
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    import torch

    t0 = time.time()
    model = build_sam3_image_model(
        bpe_path=SAM3_BPE,
        checkpoint_path=SAM3_CKPT,
        load_from_HF=False,
        device="cuda",
        eval_mode=True,
    )
    processor = Sam3Processor(model)
    print(f"  SAM3 image model ready in {time.time()-t0:.1f}s "
          f"(VRAM {torch.cuda.memory_allocated()/1024**3:.2f} GB)")

    manifest = {}
    processed = skipped = 0
    t_start = time.time()
    total = len(names)
    for idx, name in enumerate(names):
        stem = os.path.splitext(name)[0]
        img = Image.open(os.path.join(images_dir, name)).convert("RGB")
        t1 = time.time()
        state = processor.set_image(img)
        out = processor.set_text_prompt(state=state, prompt=SAM3_PROMPT)
        dt = time.time() - t1

        insts = []
        for i in range(len(out["scores"])):
            m = mask_to_bool(out["masks"][i])
            s = float(out["scores"][i])
            if s < MIN_SCORE or m.sum() == 0:
                continue
            box = [round(float(x)) for x in out["boxes"][i]] if i < len(out.get("boxes", [])) else None
            insts.append((m, s, box))
        insts.sort(key=lambda t: -t[1])

        if not insts:
            skipped += 1
            if (idx + 1) % 20 == 0 or idx == 0:
                print(f"  [{idx+1}/{total}] {name}: no instance >= {MIN_SCORE} (skip)")
            continue

        frame_manifest = []
        for pid, (m, s, box) in enumerate(insts):
            save_pair(out_dir, stem, pid, m)
            frame_manifest.append({
                "pid": pid,
                "score": round(s, 4),
                "box": box,
                "coverage": round(m.sum() / m.size * 100, 2),
            })
        if len(insts) == 1:
            save_pair(out_dir, stem, None, insts[0][0])  # legacy compat
        manifest[stem] = frame_manifest
        processed += 1

        if (idx + 1) % 20 == 0 or idx == 0:
            covs = ",".join(f"p{x['pid']}:{x['coverage']:.1f}%" for x in frame_manifest)
            print(f"  [{idx+1}/{total}] {name}: {len(insts)} instance(s) [{covs}] {dt:.2f}s")

    return manifest, processed, skipped, time.time() - t_start


def outputs_to_pairs(outs):
    """Normalize a propagate response['outputs'] to [(obj_id, mask_bool, score, box), ...].

    Real structure (verified on sam3.pt video session):
      {"out_obj_ids": (N,), "out_probs": (N,), "out_boxes_xywh": (N,4),
       "out_binary_masks": (N,H,W), "frame_stats": {...}}
    """
    if isinstance(outs, dict) and "out_obj_ids" in outs:
        masks = outs["out_binary_masks"]
        if hasattr(masks, "detach"):
            masks = masks.detach().cpu().numpy()
        masks = np.asarray(masks)
        probs = np.asarray(outs.get("out_probs", [None] * len(masks)))
        boxes = np.asarray(outs.get("out_boxes_xywh", [[None] * 4] * len(masks)))
        pairs = []
        for i, oid in enumerate(outs["out_obj_ids"]):
            m = masks[i].astype(bool)
            score = float(probs[i]) if i < len(probs) and probs[i] is not None else None
            box = [round(float(x), 4) for x in boxes[i]] if i < len(boxes) else None
            pairs.append((int(oid), m, score, box))
        return pairs
    pairs = []
    for obj_id, mask in outs.items():
        try:
            oid = int(obj_id)
        except (TypeError, ValueError):
            continue
        pairs.append((oid, mask_to_bool(mask), None, None))
    return pairs


def run_video_mode(names, images_dir, out_dir):
    """Cross-frame tracking: obj_id is the person ID across all frames."""
    from sam3.model_builder import build_sam3_video_predictor

    # Stage a session folder with numeric-stem symlinks → exact frame_index↔stem mapping
    tmp = os.path.join(out_dir, "_video_session_frames")
    os.makedirs(tmp, exist_ok=True)
    stems = []
    for i, name in enumerate(names):
        src = os.path.abspath(os.path.join(images_dir, name))
        dst = os.path.join(tmp, f"{i:06d}.png")
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(src, dst)
        stems.append(os.path.splitext(name)[0])

    try:
        t0 = time.time()
        predictor = build_sam3_video_predictor(
            checkpoint_path=SAM3_CKPT,
            bpe_path=SAM3_BPE,
            gpus_to_use=[0],
            async_loading_frames=False,
        )
        print(f"  SAM3 video predictor ready in {time.time()-t0:.1f}s")

        resp = predictor.handle_request(dict(type="start_session", resource_path=tmp))
        sid = resp["session_id"]
        print(f"  session {sid} started ({len(names)} frames)")

        t1 = time.time()
        predictor.handle_request(dict(
            type="add_prompt", session_id=sid, frame_index=0, text=SAM3_PROMPT,
        ))
        frame_outs = {}
        for response in predictor.handle_stream_request(dict(
            type="propagate_in_video", session_id=sid,
        )):
            frame_outs[response["frame_index"]] = response["outputs"]
        print(f"  propagate done in {time.time()-t1:.1f}s ({len(frame_outs)} frames)")

        manifest = {}
        processed = skipped = 0
        total_ids = set()
        for fi in sorted(frame_outs):
            stem = stems[fi]
            pids = []
            for obj_id, m, score, box in outputs_to_pairs(frame_outs[fi]):
                if m.sum() == 0:
                    continue
                save_pair(out_dir, stem, obj_id, m)
                pids.append(obj_id)
                total_ids.add(obj_id)
                manifest.setdefault(stem, []).append({
                    "pid": obj_id,
                    "score": round(score, 4) if score is not None else None,
                    "box_xywh": box,
                    "coverage": round(m.sum() / m.size * 100, 2),
                })
            if len(pids) == 1:
                save_pair(out_dir, stem, None, m)
            if pids:
                processed += 1
            else:
                skipped += 1

        _ = predictor.handle_request(dict(type="close_session", session_id=sid))
        predictor.shutdown()
        print(f"  tracked person IDs across video: {sorted(total_ids)}")
        return manifest, processed, skipped, 0.0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description="SAM3 face masks worker (sam3 conda env).")
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    print(f"🧑 SAM3 face segmentation (backend=sam3, mode={MASK_MODE})")
    print(f"  📂 images: {args.images_dir}")
    print(f"  💾 output: {args.output_dir}")
    print(f"  🔧 ckpt: {SAM3_CKPT}")
    print(f"  🔧 prompt: '{SAM3_PROMPT}'")
    print("")

    os.makedirs(args.output_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(args.images_dir)
                   if os.path.splitext(f)[1].lower() in IMG_EXTS)
    if not names:
        sys.exit(f"no images in {args.images_dir}")
    print(f"  found {len(names)} images")

    if MASK_MODE == "video":
        manifest, processed, skipped, _ = run_video_mode(names, args.images_dir, args.output_dir)
    else:
        manifest, processed, skipped, elapsed = run_image_mode(names, args.images_dir, args.output_dir)
        print(f"\n✅ done: {processed} frames, {skipped} skipped, {elapsed:.1f}s ({elapsed/max(len(names),1):.1f}s/img)")

    with open(os.path.join(args.output_dir, "masks_manifest.json"), "w") as f:
        json.dump({
            "backend": "sam3",
            "mode": MASK_MODE,
            "prompt": SAM3_PROMPT,
            "ckpt": SAM3_CKPT,
            "frames": manifest,
        }, f, indent=2, ensure_ascii=False)
    print(f"  manifest: {os.path.join(args.output_dir, 'masks_manifest.json')}")
    print(f"  output: {args.output_dir}")


if __name__ == "__main__":
    main()
