#!/usr/bin/env bash
# run_t2va.sh — reproduce the official "reproducible-768p T2VA" case.
# Prereq: FL2VA server running (bash minimax_h3/02_serve.sh, default :30010).
# Output: ../MiniMax-H3/results/t2va/t2va.mp4
set -euo pipefail
EX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TASK=t2va \
PROMPT_FILE="$EX_DIR/t2va_prompt.txt" \
DURATION=10 \
ASPECT_RATIO=16:9 \
SHORT_EDGE=768 \
SEED=0 \
OUTPUT_NAME=t2va.mp4 \
bash "$EX_DIR/../03_generate.sh"
