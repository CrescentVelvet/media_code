#!/usr/bin/env python3
"""Preview HYPIR's RealESRGAN degradation: HQ -> synthetic LQ.

Reuses the EXACT official degradation (RealESRGANDataset kernel gen +
RealESRGANBatchTransform two-stage blur/sinc/noise/jpeg) so the LQ matches what
the trainer synthesizes online. HQ is resized to OUT_SIZE(512) —
RealESRGANDataset otherwise asserts exactly 512 for crop_type=none.
queue_size is forced to 0 so each HQ's LQ is returned directly (no training-pool
caching / random return), and the queue_size % batch_size divisibility check is
bypassed.

Env:
  HQ_DIR        (required) folder of HQ images (any size)
  LQ_OUT        (required) output folder for LQ PNGs
  OUT_SIZE      (default 512) resize HQ to this before degrading
  SEED          (default 231)
  NUM_PER_IMAGE (default 1) generate N LQ per HQ (different random degradation)
  DEVICE        (default cpu) 'cpu' is fine for preview; 'cuda' faster for many
  TEMPLATE      (default $HYPIR_DIR/configs/sd2_train.yaml) official config to
                pull the degradation params from (no copy)
"""
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf

HYPIR_DIR = os.environ.get("HYPIR_DIR", "../HYPIR")
sys.path.insert(0, HYPIR_DIR)

from HYPIR.dataset.realesrgan import RealESRGANDataset
from HYPIR.dataset.batch_transform import RealESRGANBatchTransform

HQ_DIR = os.environ.get("HQ_DIR")
LQ_OUT = os.environ.get("LQ_OUT")
OUT_SIZE = int(os.environ.get("OUT_SIZE", "512"))
SEED = int(os.environ.get("SEED", "231"))
NUM_PER_IMAGE = int(os.environ.get("NUM_PER_IMAGE", "1"))
DEVICE = os.environ.get("DEVICE", "cpu")
TEMPLATE = os.environ.get("TEMPLATE") or os.path.join(HYPIR_DIR, "configs", "sd2_train.yaml")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


class _ResizeHQDataset(RealESRGANDataset):
    """Override load_gt_image to resize HQ to out_size, so any input size works
    with crop_type=none (which otherwise asserts image == out_size)."""

    def load_gt_image(self, image_path, max_retry=5):
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception:
            return None
        img = img.resize((self.out_size, self.out_size), Image.BICUBIC)
        return np.array(img)


def main():
    if not HQ_DIR:
        sys.exit("ERROR: set HQ_DIR.")
    if not LQ_OUT:
        sys.exit("ERROR: set LQ_OUT.")
    if not os.path.isdir(HQ_DIR):
        sys.exit(f"ERROR: HQ_DIR not found: {HQ_DIR}")
    if not os.path.isfile(TEMPLATE):
        sys.exit(f"ERROR: template not found: {TEMPLATE} (run run_all.sh to clone HYPIR)")
    os.makedirs(LQ_OUT, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    cfg = OmegaConf.load(TEMPLATE)
    ds_params = OmegaConf.to_container(cfg.data_config.train.dataset.params, resolve=True)
    bt_params = OmegaConf.to_container(cfg.data_config.train.batch_transform.params, resolve=True)

    # Collect HQ files and build a temp parquet (image_path + prompt) for the dataset.
    import polars as pl
    hq_files = []
    for r, _, fs in os.walk(HQ_DIR):
        for f in fs:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                hq_files.append(os.path.abspath(os.path.join(r, f)))
    hq_files.sort()
    if not hq_files:
        sys.exit(f"ERROR: no images in {HQ_DIR}")
    tmp_parquet = os.path.join(LQ_OUT, "_hq_preview_meta.parquet")
    pl.from_dict({"image_path": hq_files, "prompt": [""] * len(hq_files)}).write_parquet(tmp_parquet)

    # Dataset: point file_meta at the temp parquet; no aug; resize-to-out_size via subclass.
    ds_params["file_meta"] = {
        "file_list": tmp_parquet, "image_path_prefix": "",
        "image_path_key": "image_path", "prompt_key": "prompt",
    }
    ds_params["file_backend_cfg"] = {"target": "HYPIR.dataset.file_backend.HardDiskBackend"}
    ds_params["crop_type"] = "none"
    ds_params["use_hflip"] = False
    ds_params["use_rot"] = False
    ds_params["out_size"] = OUT_SIZE
    ds_params["p_empty_prompt"] = 0.0
    dataset = _ResizeHQDataset(**ds_params)

    # BatchTransform: force queue_size=0 -> return current LQ directly, skip the
    # training-pool (and the queue_size % batch_size divisibility check).
    # Filter bt_params to what this clone's __init__ actually accepts — your
    # batch_transform.py may be a simplified (Gaussian-blur-only) version with a
    # smaller __init__ signature than the official one.
    import inspect
    _accepted = set(inspect.signature(RealESRGANBatchTransform.__init__).parameters) - {"self"}
    bt_params = {kk: vv for kk, vv in bt_params.items() if kk in _accepted}
    bt_params["queue_size"] = 0
    bt = RealESRGANBatchTransform(**bt_params)

    print(f"[*] {len(hq_files)} HQ -> {LQ_OUT}  (out_size={OUT_SIZE}, seed={SEED}, "
          f"num_per_image={NUM_PER_IMAGE}, device={DEVICE})")
    n = 0
    for idx in range(len(hq_files)):
        stem = Path(hq_files[idx]).stem
        for k in range(NUM_PER_IMAGE):
            sample = dataset[idx]  # {hq: CHW, txt, maybe kernel1/2/sinc} — be key-agnostic:
            # your clone's dataset may not generate kernels if its batch_transform
            # doesn't use them (Gaussian-blur-only mod); build the batch from
            # whatever keys the sample actually has.
            batch = {}
            for kk, vv in sample.items():
                if isinstance(vv, torch.Tensor):
                    batch[kk] = vv.unsqueeze(0).to(DEVICE)
                else:  # txt (str) -> list-of-one for the batch dim
                    batch[kk] = [vv]
            out = bt(batch)
            lq = out["LQ"][0].clamp(0, 1).cpu().numpy()  # CHW [0,1]
            lq_arr = (np.transpose(lq, (1, 2, 0)) * 255).astype(np.uint8)
            suffix = f"_{k}" if NUM_PER_IMAGE > 1 else ""
            out_path = os.path.join(LQ_OUT, f"{stem}{suffix}.png")
            Image.fromarray(lq_arr).save(out_path)
            n += 1
            if idx < 3 or (idx % 50 == 0 and k == 0):
                print(f"  [{idx + 1}/{len(hq_files)}] {Path(hq_files[idx]).name}{suffix} -> {stem}{suffix}.png")

    try:
        os.remove(tmp_parquet)
    except OSError:
        pass
    print(f"[*] done. {n} LQ images -> {LQ_OUT}")


if __name__ == "__main__":
    main()
