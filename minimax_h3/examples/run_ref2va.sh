#!/usr/bin/env bash
# run_ref2va.sh — reproduce the official "reproducible-768p Ref2VA" case.
# Prereq: Ref2VA server running (MODEL_VARIANT=ref2va bash minimax_h3/02_serve.sh, :30011).
# Uses the official remote reference video + audio URLs (public CDN) via the
# verbatim conditions JSON (video first, then audio — the prompt numbers the
# video's own soundtrack as <Audio 1> and the audio file as <Audio 2>).
# Output: ../MiniMax-H3/results/ref2va/ref2va.mp4
set -o pipefail
EX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SERVER_URL="${SERVER_URL:-http://localhost:30011}" \
TASK=ref2va \
PROMPT_FILE="$EX_DIR/ref2va_prompt.txt" \
CONDITIONS_FILE="$EX_DIR/ref2va_conditions.json" \
DURATION=5 \
SHORT_EDGE=768 \
SEED=0 \
OUTPUT_NAME=ref2va.mp4 \
bash "$EX_DIR/../03_generate.sh"
