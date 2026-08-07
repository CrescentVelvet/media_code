#!/usr/bin/env bash
# 03_generate.sh — submit a MiniMax-H3 generation job to the SGLang server
# (started by 02_serve.sh), poll until done, and download the mp4.
#
# One job per invocation; loop over a prompt file yourself, or run the example
# scripts under examples/ for the official reproducible 768p cases.
#
# Quick examples (server assumed running; see 02_serve.sh):
#   # T2VA — text -> video+audio (uses a built-in default prompt)
#   TASK=t2va bash minimax_h3/03_generate.sh
#   # T2VA with your own prompt
#   TASK=t2va PROMPT="a drone shot over alpine peaks at golden hour" bash minimax_h3/03_generate.sh
#   # I2VA — first frame -> video (FL2VA weights)
#   TASK=fl2va FIRST_FRAME=/data/imgs/first.png DURATION=8 bash minimax_h3/03_generate.sh
#   # Ref2VA — image + audio references (Ref2VA weights, server on :30011)
#   SERVER_URL=http://localhost:30011 TASK=ref2va \
#     REF_IMAGES=/data/refs/subject.png,REF_AUDIOS=/data/refs/voice.mp3 \
#     PROMPT="Use <Picture 1> as the subject and <Audio 1> as the voice." bash minimax_h3/03_generate.sh
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# _env.sh gives proxy + CA bundle + conda env. The client itself only needs
# `requests`; to run it from a machine without conda, call generate.py directly
# with the env vars below instead of this wrapper.
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

REPO_DIR2="$(dirname "$SCRIPT_DIR")"

# --- server endpoint ---
PORT="${PORT:-30010}"
SERVER_URL="${SERVER_URL:-http://localhost:$PORT}"

# --- task & prompt ---
TASK="${TASK:-t2va}"                            # t2va | fl2va | ref2va
PROMPT="${PROMPT:-}"                            # inline prompt (wins over PROMPT_FILE)
PROMPT_FILE="${PROMPT_FILE:-}"                  # read prompt from a file
if [ -z "$PROMPT" ] && [ -z "$PROMPT_FILE" ] && [ "$TASK" = "t2va" ]; then
    # Bare T2VA with no prompt -> generate.py uses a built-in default; that's fine.
    :
fi

# --- generation target ---
DURATION="${DURATION:-5}"                       # 4-15 seconds
ASPECT_RATIO="${ASPECT_RATIO:-}"                # 16:9 / 9:16 / 1:1 / auto; blank -> per-task default
SHORT_EDGE="${SHORT_EDGE:-768}"
SEED="${SEED:-0}"

# --- sampling (SGLang cookbook defaults) ---
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
FLOW_SHIFT="${FLOW_SHIFT:-12.0}"
AUDIO_FLOW_SHIFT="${AUDIO_FLOW_SHIFT:-3.0}"
NUM_OUTPUTS="${NUM_OUTPUTS:-}"

# --- FL2VA endpoint frames (local paths or http URLs) ---
FIRST_FRAME="${FIRST_FRAME:-}"
LAST_FRAME="${LAST_FRAME:-}"

# --- Ref2VA references (comma-separated lists; local paths or http URLs) ---
REF_IMAGES="${REF_IMAGES:-}"
REF_AUDIOS="${REF_AUDIOS:-}"
REF_VIDEOS="${REF_VIDEOS:-}"
REF_VIDEO_STARTS="${REF_VIDEO_STARTS:-}"         # comma-sep seconds, parallel to REF_VIDEOS
VIDEO_AS_AUDIO_REF="${VIDEO_AS_AUDIO_REF:-0}"    # 1 -> type=video_audio (carry soundtrack)
CONDITIONS_FILE="${CONDITIONS_FILE:-}"           # JSON array of conditions (overrides REF_* env)

# --- output ---
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR2/../MiniMax-H3/results/$TASK}"
OUTPUT_NAME="${OUTPUT_NAME:-}"                  # blank -> <task>_seed<seed>.mp4

# --- polling ---
POLL_INTERVAL="${POLL_INTERVAL:-10}"
TIMEOUT_MINS="${TIMEOUT_MINS:-30}"

echo "=== [03] MiniMax-H3 generate ==="
echo "  服务地址:   $SERVER_URL"
echo "  任务:       $TASK"
echo "  时长/比例:  ${DURATION}s / ${ASPECT_RATIO:-<per-task default>}  short_edge=$SHORT_EDGE  seed=$SEED"
if [ -n "$PROMPT" ]; then
    echo "  prompt:     (inline, ${#PROMPT} chars)"
elif [ -n "$PROMPT_FILE" ]; then
    echo "  prompt:     file=$PROMPT_FILE"
else
    echo "  prompt:     <built-in default>"
fi
[ -n "$FIRST_FRAME" ] && echo "  首帧:       $FIRST_FRAME"
[ -n "$LAST_FRAME" ]  && echo "  末帧:       $LAST_FRAME"
[ -n "$REF_IMAGES" ]  && echo "  参考图:     $REF_IMAGES"
[ -n "$REF_AUDIOS" ]  && echo "  参考音频:   $REF_AUDIOS"
[ -n "$REF_VIDEOS" ]  && echo "  参考视频:   $REF_VIDEOS"
echo "  输出:       $OUTPUT_DIR/${OUTPUT_NAME:-${TASK}_seed${SEED}.mp4}"

export SERVER_URL TASK PROMPT PROMPT_FILE
export DURATION ASPECT_RATIO SHORT_EDGE SEED
export NUM_INFERENCE_STEPS FLOW_SHIFT AUDIO_FLOW_SHIFT NUM_OUTPUTS
export FIRST_FRAME LAST_FRAME
export REF_IMAGES REF_AUDIOS REF_VIDEOS REF_VIDEO_STARTS VIDEO_AS_AUDIO_REF
export CONDITIONS_FILE
export OUTPUT_DIR OUTPUT_NAME POLL_INTERVAL TIMEOUT_MINS

python "$SCRIPT_DIR/generate.py"

echo "=== [03] Done. ==="
