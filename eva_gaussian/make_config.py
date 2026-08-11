#!/usr/bin/env python3
"""Generate EVA-Gaussian config YAML files from env vars.

Writes:
  $EVA_DIR/config/pretrain.yaml  — for depthnet_pretrain.py (stage 1)
  $EVA_DIR/config/train.yaml     — for train.py (stage 2)

depthnet_pretrain.py hardcodes cfg.load("config/pretrain.yaml") and output
paths "experiments/...", so the pretrain.yaml MUST be at $EVA_DIR/config/
and depthnet_pretrain.py is run from $EVA_DIR. train.py accepts --cfg_path
and --log_path but we still write train.yaml to $EVA_DIR/config/ for consistency.

Env vars:
  EVA_DIR          — official repo root (required)
  DATA_ROOT        — dataset root with train/ and val/ subdirs (required)
  STAGE1_CKPT      — path to stage-1 checkpoint for train.yaml (optional)
  ANCHOR           — "1" to enable anchor loss (default 0)
  LR               — learning rate override
  WDECAY           — weight decay override
  BATCH_SIZE       — batch size override
  NUM_STEPS        — num training steps override
  SOURCE_ID        — comma-sep source view IDs (default "0,1")
  TRAIN_NOVEL_ID   — comma-sep train novel view IDs (default "2,3,4")
  VAL_NOVEL_ID     — comma-sep val novel view IDs (default "3")
  USE_HR_IMG       — "1" to use high-res images (default 0)
"""
import os
import sys


def _list_yaml(vals):
    """Turn '2,3,4' into '[2, 3, 4]' for yacs YAML."""
    items = [v.strip() for v in vals.split(",") if v.strip() != ""]
    return "[" + ", ".join(items) + "]"


def _bool_yaml(val):
    return "True" if str(val) in ("1", "true", "True") else "False"


def gen_pretrain():
    eva_dir = os.environ.get("EVA_DIR")
    if not eva_dir:
        sys.exit("❌ EVA_DIR not set")
    data_root = os.environ.get("DATA_ROOT", "")
    if not data_root:
        sys.exit("❌ DATA_ROOT not set")

    anchor = _bool_yaml(os.environ.get("ANCHOR", "0"))
    source_id = _list_yaml(os.environ.get("SOURCE_ID", "0,1"))
    lr = os.environ.get("LR", "0.0002")
    wdecay = os.environ.get("WDECAY", "1e-5")
    batch_size = os.environ.get("BATCH_SIZE", "6")
    num_steps = os.environ.get("NUM_STEPS", "100000")
    loss_freq = os.environ.get("LOSS_FREQ", "200")
    eval_freq = os.environ.get("EVAL_FREQ", "5000")

    yaml = f"""\
name: 'pretrain'

lr: {lr}
wdecay: {wdecay}
batch_size: {batch_size}
num_steps: {num_steps}

dataset:
  anchor: {anchor}
  source_id: {source_id}
  use_hr_img: False
  data_root: '{data_root}'

record:
  loss_freq: {loss_freq}
  eval_freq: {eval_freq}
"""
    out = os.path.join(eva_dir, "config", "pretrain.yaml")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(yaml)
    print(f"✅ wrote {out}")
    return out


def gen_train():
    eva_dir = os.environ.get("EVA_DIR")
    if not eva_dir:
        sys.exit("❌ EVA_DIR not set")
    data_root = os.environ.get("DATA_ROOT", "")
    if not data_root:
        sys.exit("❌ DATA_ROOT not set")
    stage1_ckpt = os.environ.get("STAGE1_CKPT", "")

    anchor = _bool_yaml(os.environ.get("ANCHOR", "0"))
    source_id = _list_yaml(os.environ.get("SOURCE_ID", "0,1"))
    train_novel_id = _list_yaml(os.environ.get("TRAIN_NOVEL_ID", "2,3,4"))
    val_novel_id = _list_yaml(os.environ.get("VAL_NOVEL_ID", "3"))
    use_hr_img = _bool_yaml(os.environ.get("USE_HR_IMG", "0"))
    lr = os.environ.get("LR", "0.0005")
    wdecay = os.environ.get("WDECAY", "1e-5")
    batch_size = os.environ.get("BATCH_SIZE", "2")
    num_steps = os.environ.get("NUM_STEPS", "100000")
    loss_freq = os.environ.get("LOSS_FREQ", "200")
    eval_freq = os.environ.get("EVAL_FREQ", "2000")

    yaml = f"""\
name: 'train_EVAGaussian'

stage1_ckpt: '{stage1_ckpt}'
lr: {lr}
wdecay: {wdecay}
batch_size: {batch_size}
num_steps: {num_steps}

dataset:
  anchor: {anchor}
  source_id: {source_id}
  train_novel_id: {train_novel_id}
  val_novel_id: {val_novel_id}
  use_hr_img: {use_hr_img}
  data_root: '{data_root}'

record:
  loss_freq: {loss_freq}
  eval_freq: {eval_freq}
"""
    out = os.path.join(eva_dir, "config", "train.yaml")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(yaml)
    print(f"✅ wrote {out}")
    return out


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("all", "pretrain"):
        gen_pretrain()
    if mode in ("all", "train"):
        gen_train()
