#!/usr/bin/env python3
"""Training entry point for PAIRED face fine-tuning.

Identical to the official ``train.py``, except it instantiates
``FineTuneSD2Trainer`` (from paired_face_plugin) which warm-starts the LoRA
from ``config.lora_weight_path`` (e.g. the released HYPIR_sd2.pth) when set.
Launched by 04_train_paired.sh as ``accelerate launch train_paired.py --config ...``.
"""
import sys
from argparse import ArgumentParser
from pathlib import Path

from omegaconf import OmegaConf

# Make paired_face_plugin importable regardless of the launch cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_face_plugin import FineTuneSD2Trainer  # noqa: E402

parser = ArgumentParser()
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()

config = OmegaConf.load(args.config)
trainer = FineTuneSD2Trainer(config)
trainer.run()
