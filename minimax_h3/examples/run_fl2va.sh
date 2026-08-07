#!/usr/bin/env bash
# run_fl2va.sh — reproduce the official "reproducible-768p FL2VA" (I2VA) case.
# Prereq: FL2VA server running (bash minimax_h3/02_serve.sh, default :30010).
# Uses the official remote first-frame image URL (public CDN). To use your own
# image, set FIRST_FRAME=/path/to/your.png (local paths are auto file://-prefixed).
# Output: ../MiniMax-H3/results/fl2va/fl2va.mp4
set -o pipefail
EX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Official first-frame image (ramen bowl), from the H3 I2VA demo testset.
OFFICIAL_FIRST_FRAME="https://cdn.hailuoai.com/prod/hailuo_demo/testsets/H3_AA_I2VA/gallery/sr_v17_variants_seed42_43_20260724/inputs/4a3a90bf9100_KDmcbkhzYo5sjjxr9FqcVmWVnzb.png"

TASK=fl2va \
PROMPT_FILE="$EX_DIR/fl2va_prompt.txt" \
FIRST_FRAME="${FIRST_FRAME:-$OFFICIAL_FIRST_FRAME}" \
DURATION=8 \
SHORT_EDGE=768 \
SEED=0 \
OUTPUT_NAME=fl2va.mp4 \
bash "$EX_DIR/../03_generate.sh"
