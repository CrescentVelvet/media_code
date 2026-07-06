#!/usr/bin/env python3
"""Training entry point for PAIRED face fine-tuning.

Identical to the official ``train.py``, except it instantiates
``FineTuneSD2Trainer`` (from paired_face_plugin) which warm-starts the LoRA
from ``config.lora_weight_path`` (e.g. the released HYPIR_sd2.pth) when set.
Launched by 04b_train_paired.sh as ``accelerate launch train_paired.py --config ...``.
"""
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                                    # 让 paired_face_plugin 可导入
# HYPIR 官方包(HYPIR.*)不在本目录：优先用环境变量 HYPIR_DIR，否则用当前工作目录
# (04b_train_paired.sh 已 cd 到 HYPIR_DIR 并 export HYPIR_DIR / 加入 PYTHONPATH)。
sys.path.insert(0, os.environ.get("HYPIR_DIR") or os.getcwd())
from paired_face_plugin import FineTuneSD2Trainer  # noqa: E402

parser = ArgumentParser()
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()

config = OmegaConf.load(args.config)
trainer = FineTuneSD2Trainer(config)
trainer.run()
