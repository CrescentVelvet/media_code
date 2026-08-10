#!/usr/bin/env bash
# run_all.sh — one-click MiniMax-H3 (H3-Base 768p) reproduction pipeline:
#   clone GitHub reference repo -> setup env (install SGLang) -> download HF
#   weights -> launch SGLang server (background) -> run the T2VA example.
#
# SGLang serving the 33B model takes minutes to load; this script starts it in
# the background, waits for /health, then fires one generation request. The
# server is LEFT RUNNING so you can send more jobs (stop command printed at the
# end). Set KEEP_SERVER=0 to auto-kill it after the example.
#
# Uses the existing conda env (torch preinstalled); set INSTALL_DEPS=1 once to
# install SGLang. Pass through parallelism env (NUM_GPUS / USE_FSDP / ...).
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Optional proxy (must be set before git clone).
if [ -f "$REPO_DIR/proxy.env" ]; then
    set -a; # shellcheck disable=SC1090
    source "$REPO_DIR/proxy.env"; set +a
fi

# --- Corporate proxy TLS interception workaround (git clone over HTTPS) ---
SYS_CA=/etc/ssl/certs/ca-certificates.crt
if [ -f "$SYS_CA" ]; then
    : "${GIT_SSL_CAINFO:=$SYS_CA}"
    export GIT_SSL_CAINFO
fi

MINIMAX_H3_DIR="${MINIMAX_H3_DIR:-$REPO_DIR/../MiniMax-H3}"
MINIMAX_H3_REPO="${MINIMAX_H3_REPO:-https://github.com/MiniMax-AI/MiniMax-H3.git}"
export MINIMAX_H3_DIR

KEEP_SERVER="${KEEP_SERVER:-1}"
export CONDA_ENV="${CONDA_ENV:-${CONDA_DEFAULT_ENV:-base}}"

echo "🚀 [run_all] MiniMax-H3 one-click pipeline (conda env: $CONDA_ENV)"

# 0. Clone the GitHub reference repo if absent (scripts/skills live here; SGLang
#    serving itself does NOT depend on it — it only needs the HF weight snapshot).
if [ ! -d "$MINIMAX_H3_DIR" ]; then
    mkdir -p "$(dirname "$MINIMAX_H3_DIR")"
    echo "--- 📦 cloning reference repo -> $MINIMAX_H3_DIR ---"
    git clone "$MINIMAX_H3_REPO" "$MINIMAX_H3_DIR" || \
        git -c http.sslVerify=false clone "$MINIMAX_H3_REPO" "$MINIMAX_H3_DIR" || \
        echo "WARNING: reference repo clone failed (network); serving does not need it — continuing." >&2
else
    echo "--- 📦 reference repo already present: $MINIMAX_H3_DIR ---"
fi

# 1. Env + SGLang (first run installs deps).
if [ "${INSTALL_DEPS:-1}" = "1" ]; then
    INSTALL_DEPS=1 bash "$SCRIPT_DIR/00_setup_env.sh"
else
    bash "$SCRIPT_DIR/00_setup_env.sh"
fi

# 2. Weights (verifies the snapshot if already downloaded).
bash "$SCRIPT_DIR/01_download_models.sh"

# 3. Launch SGLang server in background + wait for /health.
#    Forward the caller's parallelism env (NUM_GPUS / USE_FSDP / TP_SIZE ...).
BG=1 bash "$SCRIPT_DIR/02_serve.sh"
PID_FILE="$MINIMAX_H3_DIR/logs/serve_fl2va_30010.pid"

# 4. Run the T2VA example against the now-ready server.
echo "--- 🎬 running T2VA example ---"
bash "$SCRIPT_DIR/examples/run_t2va.sh"

echo "🎉 [run_all] Pipeline finished."
if [ -f "$PID_FILE" ]; then
    PID="$(cat "$PID_FILE")"
    if [ "$KEEP_SERVER" = "1" ]; then
        echo "    SGLang server still running (PID $PID) at http://localhost:30010"
        echo "    Send more jobs:"
        echo "      bash $SCRIPT_DIR/examples/run_fl2va.sh          # first-frame -> video"
        echo "      PROMPT=\"your text\" TASK=t2va bash $SCRIPT_DIR/03_generate.sh"
        echo "    Stop the server:  kill $PID"
    else
        echo "    ⏹️ stopping SGLang server (KEEP_SERVER=0) ---"
        kill "$PID" 2>/dev/null || true
    fi
else
    echo "    (no server PID file — server may have run in foreground)"
fi
