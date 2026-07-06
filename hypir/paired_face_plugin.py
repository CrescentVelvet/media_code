#!/usr/bin/env python3
"""HYPIR paired face-dataset plug-ins — train on REAL LQ+HQ pairs (no synthetic
RealESRGAN degradation).

Three drop-in classes referenced by ``configs/sd2_train_paired.yaml`` via
``instantiate_from_config`` (the ``target:`` field). They let HYPIR learn from
your own real-degradation face pairs instead of synthesizing LQ from HQ.

  data_config.train.dataset.target          -> PairedFaceDataset
  data_config.train.batch_transform.target  -> PairedFaceBatchTransform
  train_paired.py (entry point)             -> FineTuneSD2Trainer   # warm-start LoRA

No official HYPIR file is modified — this module is importable because
``04b_train_paired.sh`` puts this folder on ``PYTHONPATH``.

Data flow (per sample):
  parquet(hq_path, lq_path, prompt)
    -> PairedFaceDataset.__getitem__: load HQ+LQ, resize/paired-crop to out_size,
       same flip/rot on both -> {hq, lq, txt}            (RGB CHW float [0,1])
    -> PairedFaceBatchTransform.__call__: USM-sharpen HQ, rename keys
       -> {GT, LQ, txt}                                   (what the trainer expects)
    -> trainer: VAE-encode LQ -> z_lq, one-step UNet -> VAE-decode -> compare to GT
       with L2 + LPIPS + GAN losses (unchanged).
"""

import os
import random
from typing import Dict, List

import numpy as np
import polars as pl
import torch
from PIL import Image
from torch.utils import data

from HYPIR.dataset.utils import augment, USMSharp  # official: flip/rot helper, sharpener
from HYPIR.trainer.sd2 import SD2Trainer            # to subclass for LoRA warm-start


# --------------------------------------------------------------------------- #
#  parquet reading                                                            #
# --------------------------------------------------------------------------- #
def _read_paired_meta(file_meta: Dict[str, str]) -> List[Dict[str, str]]:
    """Read the (hq_path, lq_path, prompt) parquet built by build_paired_dataset.py.

    Mirrors HYPIR.dataset.utils.load_file_meta but also pulls the LQ column.
    """
    path = file_meta["file_list"]
    prefix = file_meta.get("image_path_prefix", "")
    hq_key = file_meta.get("hq_path_key", "hq_path")
    lq_key = file_meta.get("lq_path_key", "lq_path")
    pmt_key = file_meta.get("prompt_key", "prompt")
    rows = []
    for row in pl.read_parquet(path).iter_rows(named=True):
        rows.append({
            "hq_path": os.path.join(prefix, row[hq_key]),
            "lq_path": os.path.join(prefix, row[lq_key]),
            "prompt": row[pmt_key] if pmt_key != "none" else "",
        })
    return rows


# --------------------------------------------------------------------------- #
#  paired crop                                                                #
# --------------------------------------------------------------------------- #
def _resize(pil: Image.Image, size: int) -> Image.Image:
    return pil.resize((size, size), Image.BICUBIC)


def _paired_crop(hq: Image.Image, lq: Image.Image, out: int, crop_type: str):
    """Return (hq_out, lq_out), both out×out, depicting the SAME face region.

    none   -> resize the whole face to out×out on both sides (recommended:
              the face crops from crop_faces_paired.py are ~square, so this
              keeps the full face with minimal distortion).
    center -> center out×out patch from HQ + the matching relative region from LQ.
    random -> random out×out patch from HQ + the matching relative region from LQ
              (more augmentation diversity; needs HQ larger than out).
    """
    W, H = hq.size
    if crop_type == "none" or H < out or W < out:
        return _resize(hq, out), _resize(lq, out)

    if crop_type == "center":
        top, left = (H - out) // 2, (W - out) // 2
    else:  # random
        top, left = random.randint(0, H - out), random.randint(0, W - out)

    hq_c = hq.crop((left, top, left + out, top + out))
    # the same relative box on the (much smaller) LQ, then upscale to out×out
    wl, hl = lq.size
    l = int(round(left / W * wl))
    t = int(round(top / H * hl))
    r = int(round((left + out) / W * wl))
    b = int(round((top + out) / H * hl))
    lq_c = lq.crop((max(0, l), max(0, t), max(1, r), max(1, b)))
    return hq_c, _resize(lq_c, out)


# --------------------------------------------------------------------------- #
#  dataset                                                                    #
# --------------------------------------------------------------------------- #
class PairedFaceDataset(data.Dataset):
    """Loads REAL LQ+HQ face pairs and returns {hq, lq, txt}.

    Parameters (from the config's dataset.params):
      file_meta        : {file_list, image_path_prefix, hq_path_key, lq_path_key, prompt_key}
      out_size         : training resolution (default 512, matches HYPIR's VAE patch)
      crop_type        : none | center | random  (see _paired_crop)
      use_hflip/use_rot: augmentation (applied identically to HQ and LQ)
      p_empty_prompt   : probability of dropping the prompt (null-text training)
      return_file_name : include "filename" in the output (for logging)
    """

    def __init__(self, file_meta, out_size=512, crop_type="none",
                 use_hflip=True, use_rot=False, p_empty_prompt=0.0,
                 return_file_name=False):
        super().__init__()
        self.rows = _read_paired_meta(file_meta)
        assert crop_type in ("none", "center", "random"), f"bad crop_type {crop_type}"
        self.out_size = out_size
        self.crop_type = crop_type
        self.use_hflip = use_hflip
        self.use_rot = use_rot
        self.p_empty_prompt = p_empty_prompt
        self.return_file_name = return_file_name

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.rows[index]
        try:
            hq = Image.open(row["hq_path"]).convert("RGB")
            lq = Image.open(row["lq_path"]).convert("RGB")
        except Exception:
            # be resilient to a single corrupt file (mirror RealESRGANDataset)
            return self.__getitem__(random.randint(0, len(self) - 1))

        hq_c, lq_c = _paired_crop(hq, lq, self.out_size, self.crop_type)
        hq_arr = np.array(hq_c, dtype=np.float32) / 255.0   # HWC, RGB, [0,1]
        lq_arr = np.array(lq_c, dtype=np.float32) / 255.0

        # identical flip/rotation on both (augment applies the same op to a list)
        hq_arr, lq_arr = augment([hq_arr, lq_arr], self.use_hflip, self.use_rot)

        prompt = row["prompt"]
        if np.random.uniform() < self.p_empty_prompt:
            prompt = ""

        out = {
            "hq": torch.from_numpy(np.ascontiguousarray(hq_arr.transpose(2, 0, 1))),
            "lq": torch.from_numpy(np.ascontiguousarray(lq_arr.transpose(2, 0, 1))),
            "txt": prompt,
        }
        if self.return_file_name:
            out["filename"] = os.path.basename(row["hq_path"])
        return out


# --------------------------------------------------------------------------- #
#  batch transform                                                            #
# --------------------------------------------------------------------------- #
class PairedFaceBatchTransform:
    """No degradation — just prepare the keys the trainer expects.

    Optionally USM-sharpen HQ so the GT matches the preprocessing the released
    HYPIR LoRA was trained with (the official RealESRGANBatchTransform also
    USM-sharpens HQ). Output keys: {GT, LQ, txt}.

    Parameters:
      hq_key / lq_key / txt_key : source keys in the dataset batch
      use_sharpener             : apply USMSharp to HQ (default True)
    """

    def __init__(self, hq_key="hq", lq_key="lq", txt_key="txt", use_sharpener=True):
        self.hq_key = hq_key
        self.lq_key = lq_key
        self.txt_key = txt_key
        self.usm = USMSharp() if use_sharpener else None

    @torch.no_grad()
    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        hq = batch[self.hq_key]
        if self.usm is not None:
            self.usm.to(hq)          # move the gaussian kernel onto hq's device
            hq = self.usm(hq)
        return {"GT": hq, "LQ": batch[self.lq_key], self.txt_key: batch[self.txt_key]}


# --------------------------------------------------------------------------- #
#  trainer subclass: warm-start LoRA from the released weights                #
# --------------------------------------------------------------------------- #
class FineTuneSD2Trainer(SD2Trainer):
    """SD2Trainer that loads a pretrained LoRA state_dict before the optimizer
    is built, so you FINE-TUNE the released HYPIR_sd2.pth on your face data
    instead of training a fresh LoRA from scratch (which needs far more data).

    Set ``config.lora_weight_path`` to a .pth whose keys are the UNet's
    requires_grad parameters (exactly what HYPIR checkpoints store — the official
    HYPIR_sd2.pth is one). Omit it to fall back to gaussian-init from-scratch
    training (the official behaviour).

    NB: ``resume_ema`` should be false when warm-starting from a raw LoRA file
    (no EMA state ships with HYPIR_sd2.pth); the EMA is initialised from the
    loaded weights instead.
    """

    def init_generator(self):
        # 官方 SD2Trainer.init_generator 读 config.weight_path 来加载 LoRA(暖启动)；
        # 我们的 config 用的是 lora_weight_path，所以在这里映射过去。
        # 无 lora_weight_path 时也设 weight_path=None(从零训)，避免官方代码访问缺键报错。
        lora_wp = getattr(self.config, "lora_weight_path", None)
        if not hasattr(self.config, "weight_path"):
            from omegaconf import OmegaConf
            OmegaConf.set_struct(self.config, False)
            self.config.weight_path = lora_wp   # 暖启动=权重路径；从零训=None
            OmegaConf.set_struct(self.config, True)
        super().init_generator()