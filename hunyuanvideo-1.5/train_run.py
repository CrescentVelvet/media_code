#!/usr/bin/env python3
"""Launcher for HunyuanVideo-1.5 training that uses a real dataset.

The official train.py hardcodes a placeholder `create_dummy_dataloader()`.
Rather than fork the official file, this launcher:
  1. puts the official repo on sys.path (so `import train` + `from hyvideo...`),
  2. replaces train.create_dummy_dataloader with one that reads DATA_DIR via
     train_dataset.HunyuanVideoDataset,
  3. calls train.main() — which parses the remaining CLI args exactly like the
     official script and runs the full HunyuanVideoTrainer loop.

So all official training flags (--learning_rate, --use_lora, --sp_size, ...) are
passed through unchanged; the only addition is the DATA_DIR env var.

Run via torchrun (see 03_train.sh):
    DATA_DIR=/data/my_videos torchrun --nproc_per_node=N train_run.py \\
        --pretrained_model_root ./ckpts --learning_rate 1e-5 ...
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HYVIDEO_DIR = os.environ.get("HYVIDEO_DIR", os.path.join(HERE, "..", "HunyuanVideo-1.5"))
HYVIDEO_DIR = os.path.abspath(HYVIDEO_DIR)

# Make `import train` (official repo) and `import hyvideo` resolve, then our dir
# for `import train_dataset`.
sys.path.insert(0, HYVIDEO_DIR)
sys.path.insert(0, HERE)

DATA_DIR = os.environ.get("DATA_DIR", "").strip()
if not DATA_DIR:
    sys.exit("ERROR: DATA_DIR is not set. Point it at your dataset root "
             "(see hunyuanvideo-1.5/README.md for the expected layout).")

# torchrun sets LOCAL_RANK via env (modern) but some versions still inject a
# stray --local-rank / --local_rank argv; train.py's argparse doesn't define it
# and would error. Strip those here (harmless when absent).
_argv = list(sys.argv)
_clean = [_argv[0]]
_i = 1
while _i < len(_argv):
    _a = _argv[_i]
    if _a.startswith("--local-rank") or _a.startswith("--local_rank"):
        if "=" not in _a and _i + 1 < len(_argv) and not _argv[_i + 1].startswith("-"):
            _i += 1  # consume the value form "--local-rank <v>"
    else:
        _clean.append(_a)
    _i += 1
sys.argv = _clean

import train as official_train      # noqa: E402  (official repo's train.py)
import train_dataset                # noqa: E402  (our dataset module, same dir)


def _make_dataloader(config):
    return train_dataset.create_dataloader(config, DATA_DIR)


# Inject our real dataloader into the official main() flow (it calls the
# module-global create_dummy_dataloader). No official code is modified.
official_train.create_dummy_dataloader = _make_dataloader

if __name__ == "__main__":
    official_train.main()
