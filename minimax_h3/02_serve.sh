#!/usr/bin/env bash
# 02_serve.sh — launch the SGLang Diffusion server for MiniMax-H3 (H3-Base 768p).
#
# This is the ONLY locally-reproducible piece of MiniMax-H3: the H3-Base model
# generates 768p video + native stereo audio. (H3-Context-IR & H3-Regenerate-2K
# are hosted MiniMax APIs, not open-source — see README "Full 2K workflow".)
#
# SGLang shards the 33B Omni-Transformer + Qwen3-VL-32B encoder across GPUs.
# Parallelism is fully configurable; below are the VERIFIED recipes from the
# SGLang cookbook, mapped to A100:
#
#   4× A100 80GB (capacity, safest):   GPU=0,1,2,3 NUM_GPUS=4 ULYSSES_DEGREE=4 USE_FSDP=1
#   4× A100 80GB (fastest):             GPU=0,1,2,3 NUM_GPUS=4 TP_SIZE=2 ULYSSES_DEGREE=2
#   2× RTX 5090 32GB (offload, slow):   see README offload note
#
# A100 is Ampere — not in the officially-verified list, but 4× A100 80GB has
# enough per-card VRAM for the 80GB recipes. If the default (Ulysses4,
# resident) OOMs, set USE_FSDP=1 (capacity path) or TP_SIZE=2 ULYSSES_DEGREE=2
# (lowers peak memory).
#
# Usage:
#   GPU=              bash minimax_h3/02_serve.sh              # foreground, default FL2VA on :30010
#   BG=1               bash minimax_h3/02_serve.sh              # background + wait for /health ready
#   MODEL_VARIANT=ref2va bash minimax_h3/02_serve.sh           # Ref2VA server on :30011
#   NUM_GPUS=4 USE_FSDP=1 bash minimax_h3/02_serve.sh          # A100 capacity path
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../../model/MiniMax-H3}"
MODEL_PATH="${MODEL_PATH:-$MODEL_DIR}"          # SGLang --model-path = HF snapshot root

MODEL_VARIANT="${MODEL_VARIANT:-fl2va}"         # fl2va | ref2va
HOST="${HOST:-0.0.0.0}"
# Default ports match the official reproducible scripts (FL2VA=30010, Ref2VA=30011).
if [ -z "${PORT:-}" ]; then
    case "$MODEL_VARIANT" in
        ref2va) PORT=30011 ;;
        *)      PORT=30010 ;;
    esac
fi

NUM_GPUS="${NUM_GPUS:-4}"
ULYSSES_DEGREE="${ULYSSES_DEGREE:-$NUM_GPUS}"
TP_SIZE="${TP_SIZE:-}"                           # unset by default; set 2 for A100 fastest
USE_FSDP="${USE_FSDP:-0}"
PERFORMANCE_MODE="${PERFORMANCE_MODE:-speed}"   # speed | memory
# Extra flags appended verbatim (e.g. offload components on low-VRAM cards).
EXTRA_SGLANG_FLAGS="${EXTRA_SGLANG_FLAGS:-}"

BG="${BG:-0}"                                   # 0=foreground (default), 1=background+wait
# Logs land under the official code dir (../MiniMax-H3/logs) by default.
LOG_DIR="${LOG_DIR:-$REPO_DIR/../MiniMax-H3/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/serve_${MODEL_VARIANT}_${PORT}.log}"

# --- checks ---
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ ERROR: model snapshot not found at $MODEL_PATH. Run 01_download_models.sh first." >&2; exit 1
fi
if [ ! -f "$MODEL_PATH/model_index.json" ]; then
    echo "❌ ERROR: $MODEL_PATH/model_index.json not found (incomplete snapshot). Rerun 01_download_models.sh." >&2; exit 1
fi
if ! command -v sglang >/dev/null 2>&1 && ! python -c "import sglang" 2>/dev/null; then
    echo "❌ ERROR: sglang not installed in env '$CONDA_ENV'. Run INSTALL_DEPS=1 bash minimax_h3/00_setup_env.sh." >&2; exit 1
fi

# --- build the flag list ---
SERVE_ARGS=(
    --model-path "$MODEL_PATH"
    --model-variant "$MODEL_VARIANT"
    --num-gpus "$NUM_GPUS"
    --ulysses-degree "$ULYSSES_DEGREE"
    --performance-mode "$PERFORMANCE_MODE"
    --host "$HOST"
    --port "$PORT"
)
[ -n "$TP_SIZE" ] && SERVE_ARGS+=(--tp-size "$TP_SIZE")
[ "$USE_FSDP" = "1" ] && SERVE_ARGS+=(--use-fsdp-inference true)
# shellcheck disable=SC2206
[ -n "$EXTRA_SGLANG_FLAGS" ] && SERVE_ARGS+=($EXTRA_SGLANG_FLAGS)

SERVER_URL="http://localhost:$PORT"

echo "🚀 [02] SGLang serve MiniMax-H3 (H3-Base 768p)"
echo "  🤖 模型路径: $MODEL_PATH"
echo "  变体:       $MODEL_VARIANT  (FL2VA=T2VA/I2VA/L2VA/FL2VA; Ref2VA=参考生成)"
echo "  并行:       num_gpus=$NUM_GPUS ulysses=$ULYSSES_DEGREE${TP_SIZE:+  tp=$TP_SIZE}${USE_FSDP:+  fsdp=on}"
echo "  performance: $PERFORMANCE_MODE"
echo "  📡 服务地址: $SERVER_URL  (host=$HOST)"
echo "  📝 日志:     $LOG_FILE"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "  🎮 GPU:       physical $CUDA_VISIBLE_DEVICES  [GPU=N to restrict; leave unset to use all --num-gpus cards]"
else
    echo "  🎮 GPU:       all visible  [set GPU=N to restrict to specific cards]"
fi
if [ "$NUM_GPUS" = "4" ] && [ "$ULYSSES_DEGREE" = "4" ] && [ "$USE_FSDP" != "1" ] && [ -z "$TP_SIZE" ]; then
    echo "  ⚠️  4× A100 80GB 若默认 resident OOM，改用: USE_FSDP=1 (capacity) 或 TP_SIZE=2 ULYSSES_DEGREE=2"
fi

run_serve() {
    # shellcheck disable=SC1091
    exec sglang serve "${SERVE_ARGS[@]}"
}

run_serve_bg() {
    echo "--- 📦 launching SGLang in background (log: $LOG_FILE) ---"
    nohup sglang serve "${SERVE_ARGS[@]}" >"$LOG_FILE" 2>&1 &
    SERVER_PID=$!
    echo "$SERVER_PID" > "$LOG_DIR/serve_${MODEL_VARIANT}_${PORT}.pid"
    echo "  server PID: $SERVER_PID"
    echo "  ⏳ waiting for server to become ready (loading 33B model, this takes minutes) ---"
    # Poll /health. SGLang returns 200 once the engine is up.
    ready=0
    for i in $(seq 1 "${HEALTH_TIMEOUT_MINS:-30}"); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "❌ ERROR: server process died before becoming ready. Last log lines:" >&2
            tail -n 40 "$LOG_FILE" >&2 || true
            exit 1
        fi
        if curl -sf --max-time 10 "$SERVER_URL/health" >/dev/null 2>&1 \
           || curl -sf --max-time 10 "$SERVER_URL/health_generate" >/dev/null 2>&1; then
            ready=1
            echo "  [$(printf '%02d' $i)min] ✅ server ready"
            break
        fi
        echo "  [$(printf '%02d' $i)min] ⏳ not ready yet... (tail: $(tail -n1 "$LOG_FILE" 2>/dev/null | cut -c1-80))"
        sleep 60
    done
    if [ "$ready" != "1" ]; then
        echo "❌ ERROR: server not ready after ${HEALTH_TIMEOUT_MINS:-30} min. Check $LOG_FILE." >&2
        exit 1
    fi
    echo "🎉 [02] Server ready at $SERVER_URL (PID $SERVER_PID)"
    echo "    Submit jobs:  bash minimax_h3/03_generate.sh"
    echo "    Stop server:  kill \$(cat $LOG_DIR/serve_${MODEL_VARIANT}_${PORT}.pid)"
}

if [ "$BG" = "1" ]; then
    run_serve_bg
else
    echo "  (foreground — Ctrl-C to stop. For background+wait set BG=1)"
    run_serve
fi
