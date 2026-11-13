#!/bin/bash
# R2-N 鼻部指认包生成（flame_human env 有 mediapipe）
set -e
cd /mnt/c/code/media_code
source ~/miniconda3/etc/profile.d/conda.sh
conda activate flame_human

OUT="${OUT:-/mnt/d/tmp/r2n/sheets}"
EXTRA="${EXTRA:-}"
python splatanything/10_nose_sheet.py --out "$OUT" --pairs /mnt/d/splatanything_figures/pairs $EXTRA
