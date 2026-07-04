#!/usr/bin/env python3
"""Fill the TODO fields in HYPIR's configs/sd2_train.yaml and write a derived
config for training. No official file is modified — a filled copy is written
under OUTPUT_DIR and passed to train.py.

Env:
  TEMPLATE        (default $HYPIR_DIR/configs/sd2_train.yaml)
  CONFIG_OUT      (default $OUTPUT_DIR/sd2_train_filled.yaml)
  OUTPUT_DIR      experiment output dir (REQUIRED)
  PARQUET_PATH    parquet from 03_build_dataset.sh (REQUIRED)
  BASE_MODEL_PATH (default $MODEL_DIR/sd2_base)  overrides base_model_path
  IMAGE_PATH_PREFIX (default "")
  IMAGE_PATH_KEY    (default image_path)
  PROMPT_KEY        (default prompt)
  CROP_TYPE         (default none)   none|center|random
  OUT_SIZE          (default 512)
"""
import os
import sys

from omegaconf import OmegaConf

TEMPLATE = os.environ.get("TEMPLATE")
CONFIG_OUT = os.environ.get("CONFIG_OUT")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR")
PARQUET_PATH = os.environ.get("PARQUET_PATH")
BASE_MODEL_PATH = os.environ.get("BASE_MODEL_PATH")
IMAGE_PATH_PREFIX = os.environ.get("IMAGE_PATH_PREFIX", "")
IMAGE_PATH_KEY = os.environ.get("IMAGE_PATH_KEY", "image_path")
PROMPT_KEY = os.environ.get("PROMPT_KEY", "prompt")
CROP_TYPE = os.environ.get("CROP_TYPE", "none")
OUT_SIZE = int(os.environ.get("OUT_SIZE", "512"))


def need(name, val):
    if not val:
        sys.exit(f"ERROR: set {name}.")


def main():
    need("OUTPUT_DIR", OUTPUT_DIR)
    need("PARQUET_PATH", PARQUET_PATH)
    need("TEMPLATE (configs/sd2_train.yaml)", TEMPLATE)
    if not os.path.isfile(PARQUET_PATH):
        sys.exit(f"ERROR: parquet not found: {PARQUET_PATH}")
    if not os.path.isfile(TEMPLATE):
        sys.exit(f"ERROR: template not found: {TEMPLATE}")
    config_out = CONFIG_OUT or os.path.join(OUTPUT_DIR, "sd2_train_filled.yaml")
    config_out = os.path.abspath(config_out)
    os.makedirs(os.path.dirname(config_out) or ".", exist_ok=True)

    cfg = OmegaConf.load(TEMPLATE)

    # Fill the dataset TODOs.
    cfg.data_config.train.dataset.params.file_meta.file_list = os.path.abspath(PARQUET_PATH)
    cfg.data_config.train.dataset.params.file_meta.image_path_prefix = IMAGE_PATH_PREFIX
    cfg.data_config.train.dataset.params.file_meta.image_path_key = IMAGE_PATH_KEY
    cfg.data_config.train.dataset.params.file_meta.prompt_key = PROMPT_KEY
    cfg.data_config.train.dataset.params.crop_type = CROP_TYPE
    cfg.data_config.train.dataset.params.out_size = OUT_SIZE

    cfg.output_dir = os.path.abspath(OUTPUT_DIR)
    if BASE_MODEL_PATH:
        cfg.base_model_path = os.path.abspath(BASE_MODEL_PATH)

    OmegaConf.save(cfg, config_out)
    print(f"[*] wrote filled config -> {config_out}")
    print(f"    output_dir        = {cfg.output_dir}")
    print(f"    base_model_path   = {cfg.base_model_path}")
    fm = cfg.data_config.train.dataset.params.file_meta
    print(f"    file_list         = {fm.file_list}")
    print(f"    crop_type/out_size= {CROP_TYPE}/{OUT_SIZE}")


if __name__ == "__main__":
    main()
