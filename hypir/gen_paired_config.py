#!/usr/bin/env python3
"""Fill the TODO fields in sd2_train_paired.yaml and write a derived config for
paired face fine-tuning. No official file is modified — a filled copy is written
under OUTPUT_DIR and passed (via 04b_train_paired.sh) to train_paired.py.

Env:
  TEMPLATE        (default $SCRIPT_DIR/sd2_train_paired.yaml)
  CONFIG_OUT      (default $OUTPUT_DIR/sd2_train_paired_filled.yaml)
  OUTPUT_DIR      experiment output dir (REQUIRED)
  PARQUET_PATH    parquet from build_paired_dataset.py (REQUIRED)
  BASE_MODEL_PATH (default $MODEL_DIR/sd2_base)
  LORA_WEIGHT_PATH(default $MODEL_DIR/HYPIR_sd2.pth)  set to "" to train from scratch
  HQ_PATH_KEY / LQ_PATH_KEY / PROMPT_KEY  (default hq_path / lq_path / prompt)
  IMAGE_PATH_PREFIX (default "")
  CROP_TYPE       (default none)
  OUT_SIZE        (default 512)
"""
import os
import sys

from omegaconf import OmegaConf

TEMPLATE = os.environ.get("TEMPLATE")
CONFIG_OUT = os.environ.get("CONFIG_OUT")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR")
PARQUET_PATH = os.environ.get("PARQUET_PATH")
BASE_MODEL_PATH = os.environ.get("BASE_MODEL_PATH")
LORA_WEIGHT_PATH = os.environ.get("LORA_WEIGHT_PATH")
HQ_PATH_KEY = os.environ.get("HQ_PATH_KEY", "hq_path")
LQ_PATH_KEY = os.environ.get("LQ_PATH_KEY", "lq_path")
PROMPT_KEY = os.environ.get("PROMPT_KEY", "prompt")
IMAGE_PATH_PREFIX = os.environ.get("IMAGE_PATH_PREFIX", "")
CROP_TYPE = os.environ.get("CROP_TYPE", "none")
OUT_SIZE = int(os.environ.get("OUT_SIZE", "512"))


def need(name, val):
    if not val:
        sys.exit(f"ERROR: set {name}.")


def main():
    need("OUTPUT_DIR", OUTPUT_DIR)
    need("PARQUET_PATH", PARQUET_PATH)
    if not TEMPLATE or not os.path.isfile(TEMPLATE):
        sys.exit(f"ERROR: template not found: {TEMPLATE}")
    if not os.path.isfile(PARQUET_PATH):
        sys.exit(f"ERROR: parquet not found: {PARQUET_PATH}")
    config_out = CONFIG_OUT or os.path.join(OUTPUT_DIR, "sd2_train_paired_filled.yaml")
    config_out = os.path.abspath(config_out)
    os.makedirs(os.path.dirname(config_out) or ".", exist_ok=True)

    cfg = OmegaConf.load(TEMPLATE)

    # --- dataset file_meta ---
    fm = cfg.data_config.train.dataset.params.file_meta
    fm.file_list = os.path.abspath(PARQUET_PATH)
    fm.image_path_prefix = IMAGE_PATH_PREFIX
    fm.hq_path_key = HQ_PATH_KEY
    fm.lq_path_key = LQ_PATH_KEY
    fm.prompt_key = PROMPT_KEY
    cfg.data_config.train.dataset.params.crop_type = CROP_TYPE
    cfg.data_config.train.dataset.params.out_size = OUT_SIZE

    # --- paths ---
    cfg.output_dir = os.path.abspath(OUTPUT_DIR)
    if BASE_MODEL_PATH:
        cfg.base_model_path = os.path.abspath(BASE_MODEL_PATH)
    # lora_weight_path: empty/None => from scratch; otherwise the released .pth
    if LORA_WEIGHT_PATH:
        cfg.lora_weight_path = os.path.abspath(LORA_WEIGHT_PATH)
    else:
        cfg.lora_weight_path = None

    OmegaConf.save(cfg, config_out)
    print(f"[*] wrote filled paired config -> {config_out}")
    print(f"    output_dir        = {cfg.output_dir}")
    print(f"    base_model_path   = {cfg.base_model_path}")
    print(f"    lora_weight_path  = {cfg.lora_weight_path}")
    print(f"    file_list         = {fm.file_list}")
    print(f"    crop_type/out_size= {CROP_TYPE}/{OUT_SIZE}")


if __name__ == "__main__":
    main()
