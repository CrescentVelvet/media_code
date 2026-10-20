#!/usr/bin/env bash
set -u
PY=/home/velvet/miniconda3/envs/vggt_human/bin/python
SCRIPT=/mnt/c/code/media_code/.workbuddy/memory/_diag_hand_blur.py
"$PY" "$SCRIPT" > /tmp/diag.log 2>&1
echo "EXIT=$?"
