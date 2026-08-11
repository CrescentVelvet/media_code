#!/usr/bin/env bash
# run.sh — one-shot: ensure server up -> submit generate -> download mp4.
#
# Wraps 02_serve.sh + 03_generate.sh into a single command. The 33B model
# takes minutes to load, so this script:
#   1) pokes $SERVER_URL/health — if already ready (e.g. a previous run left
#      the server up), skip step 2 and go straight to generate;
#   2) otherwise launches `02_serve.sh` in BG mode (which itself waits for
#      /health) — server is LEFT RUNNING after the job for the next call;
#   3) calls `03_generate.sh` to submit + poll + download.
#
# All knobs are passed through env (same names as 02/03). Serve-side vars go
# to 02_serve.sh, generate-side vars go to 03_generate.sh. See README for the
# full list. Minimal example (T2VA, 4× A100 FSDP):
#
#   GPU=0,1,2,3 MODEL_PATH=../../model/MiniMax-H3 \
#     TASK=t2va PROMPT="a drone shot over alpine peaks at golden hour" \
#     OUTPUT_DIR=../MiniMax-H3/results/t2va OUTPUT_NAME=t2va.mp4 \
#     bash minimax_h3/run.sh
#
# Server left running afterwards; stop with:  kill $(cat ../MiniMax-H3/logs/serve_<variant>_<port>.pid)
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- server endpoint (derive from MODEL_VARIANT/PORT like 02_serve.sh) ---
MODEL_VARIANT="${MODEL_VARIANT:-fl2va}"            # fl2va | ref2va
if [ -z "${PORT:-}" ]; then
    case "$MODEL_VARIANT" in
        ref2va) PORT=30011 ;;
        *)      PORT=30010 ;;
    esac
fi
SERVER_URL="${SERVER_URL:-http://localhost:$PORT}"

echo "🚀 [run] one-shot MiniMax-H3 generate"
echo "  🤖 变体/端口:  $MODEL_VARIANT / $PORT  ($SERVER_URL)"
echo "  🎯 任务:       ${TASK:-t2va}"

# --- step 1: is the server already up? ---
server_up=0
if curl -sf --max-time 5 "$SERVER_URL/health" >/dev/null 2>&1 \
   || curl -sf --max-time 5 "$SERVER_URL/health_generate" >/dev/null 2>&1; then
    server_up=1
    echo "  ✅ server already ready at $SERVER_URL — skip launch"
fi

# --- step 2: launch server (BG + wait ready) if not up ---
if [ "$server_up" != "1" ]; then
    if [ -z "${GPU:-}" ]; then
        echo "❌ ERROR: GPU not set (e.g. GPU=0,1,2,3). Required to launch server." >&2
        exit 1
    fi
    if [ -z "${MODEL_PATH:-}" ]; then
        echo "❌ ERROR: MODEL_PATH not set (e.g. MODEL_PATH=../../model/MiniMax-H3). Required to launch server." >&2
        exit 1
    fi
    # A100 80GB capacity recipe defaults (override via env). NUM_GPUS must
    # match the card count in GPU.
    export NUM_GPUS="${NUM_GPUS:-4}"
    export ULYSSES_DEGREE="${ULYSSES_DEGREE:-$NUM_GPUS}"
    export USE_FSDP="${USE_FSDP:-1}"
    echo "--- 📦 launching server (BG, waits for /health; loading 33B takes minutes) ---"
    echo "  🎮 GPU=$GPU  NUM_GPUS=$NUM_GPUS  ULYSSES=$ULYSSES_DEGREE  FSDP=$USE_FSDP${TP_SIZE:+  TP=$TP_SIZE}"
    echo "  🏋️ MODEL_PATH=$MODEL_PATH  MODEL_VARIANT=$MODEL_VARIANT"
    BG=1 bash "$SCRIPT_DIR/02_serve.sh"
    if [ $? -ne 0 ]; then
        echo "❌ ERROR: server failed to start. Check $(dirname "$SCRIPT_DIR")/../MiniMax-H3/logs/serve_${MODEL_VARIANT}_${PORT}.log" >&2
        exit 1
    fi
fi

# --- step 3: submit generate request ---
echo "--- 🎬 submitting generate request ---"
bash "$SCRIPT_DIR/03_generate.sh"
if [ $? -ne 0 ]; then
    echo "❌ ERROR: generate failed." >&2
    exit 1
fi

echo "🎉 [run] Done."
echo "    Server still running at $SERVER_URL (reuse for next job)."
echo "    Stop:  kill \$(cat $(dirname "$SCRIPT_DIR")/../MiniMax-H3/logs/serve_${MODEL_VARIANT}_${PORT}.pid 2>/dev/null)"
