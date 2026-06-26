#!/usr/bin/env python3
"""Batch TripoSplat inference over a folder of images.

For each image in INPUT_DIR, produce a single NUM_GAUSSIANS-density gaussian
(default 262144) saved as <stem>.ply and <stem>.splat in OUTPUT_DIR, named
after the input image (e.g. house.webp -> house.ply / house.splat). The
pipeline is loaded ONCE and reused across images (much faster than relaunching
python per image). The official run_example.py's multi-density example is not
used — only one density is produced.

Env vars (set by 02_run_inference.sh):
  TRIPOSPLAT_DIR, INPUT_DIR, OUTPUT_DIR, NUM_GAUSSIANS, DEVICE
"""
import os
import sys
import time
from pathlib import Path

TRIPOSPLAT_DIR = os.environ.get("TRIPOSPLAT_DIR", "../TripoSplat")
sys.path.insert(0, TRIPOSPLAT_DIR)  # so `from triposplat import ...` resolves

from triposplat import TripoSplatPipeline

INPUT_DIR = os.environ.get("INPUT_DIR") or os.path.join(TRIPOSPLAT_DIR, "static", "example_inputs")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR") or os.path.join(TRIPOSPLAT_DIR, "output")
# Nest outputs under a subfolder named after the input folder, so different
# INPUT_DIR runs don't clobber each other.
INPUT_NAME = os.path.basename(os.path.normpath(INPUT_DIR)) or "input"
OUTPUT_DIR = os.path.join(OUTPUT_DIR, INPUT_NAME)
PREP_DIR = os.path.join(OUTPUT_DIR, "preprocessed")
NUM_GAUSSIANS = int(os.environ.get("NUM_GAUSSIANS", "262144"))
DEVICE = os.environ.get("DEVICE", "cuda")

CKPT = os.path.join(TRIPOSPLAT_DIR, "ckpts/diffusion_models/triposplat_fp16.safetensors")
DECODER = os.path.join(TRIPOSPLAT_DIR, "ckpts/vae/triposplat_vae_decoder_fp16.safetensors")
DINOV3 = os.path.join(TRIPOSPLAT_DIR, "ckpts/clip_vision/dino_v3_vit_h.safetensors")
FLUX2VAE = os.path.join(TRIPOSPLAT_DIR, "ckpts/vae/flux2-vae.safetensors")
RMBG = os.path.join(TRIPOSPLAT_DIR, "ckpts/background_removal/birefnet.safetensors")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PREP_DIR, exist_ok=True)

print(f"[*] loading pipeline (device={DEVICE}) ...")
pipe = TripoSplatPipeline(
    ckpt_path=CKPT,
    decoder_path=DECODER,
    dinov3_path=DINOV3,
    flux2_vae_encoder_path=FLUX2VAE,
    rmbg_path=RMBG,
    device=DEVICE,
)
print("[*] pipeline ready")

IMG_EXTS = {".webp", ".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR)
         if os.path.splitext(f)[1].lower() in IMG_EXTS]
files = sorted(set(files))
if not files:
    sys.exit(f"ERROR: no images in {INPUT_DIR}")

print(f"[*] {len(files)} image(s): {INPUT_DIR} -> {OUTPUT_DIR}  (num_gaussians={NUM_GAUSSIANS})")
ok = 0
recon_times = []
for i, f in enumerate(files, 1):
    stem = Path(f).stem
    ply = os.path.join(OUTPUT_DIR, f"{stem}.ply")
    splat = os.path.join(OUTPUT_DIR, f"{stem}.splat")
    print(f"[{i}/{len(files)}] {os.path.basename(f)}  ->  {stem}.ply / {stem}.splat")
    try:
        t0 = time.time()
        gaussian, prepared = pipe.run(f, num_gaussians=NUM_GAUSSIANS, show_progress=True)
        t_recon = time.time() - t0
        gaussian.save_ply(ply)
        gaussian.save_splat(splat)
        prepared.save(os.path.join(PREP_DIR, f"{stem}.webp"))
        recon_times.append(t_recon)
        ok += 1
        print(f"    reconstructed in {t_recon:.2f}s")
    except Exception as e:
        print(f"    ! failed: {e}", file=sys.stderr)

print(f"[*] done. {ok}/{len(files)} succeeded. outputs in {OUTPUT_DIR}")
if recon_times:
    avg = sum(recon_times) / len(recon_times)
    print(f"[*] reconstruction time: avg {avg:.2f}s, min {min(recon_times):.2f}s, max {max(recon_times):.2f}s over {len(recon_times)} image(s)")
