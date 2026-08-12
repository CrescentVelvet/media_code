#!/usr/bin/env bash
# stop.sh — kill 残留的 sglang serve 进程（按端口找，不依赖进程名）。
#
# 任务失败后 worker 僵死（HTTP server 还活着但不占显存），任务卡 status=queue。
# 这时 grep sglang 找不到（进程名是 python 不是 sglang），nvidia-smi 也看不到
# （不占显存）。最可靠是按端口找 PID：lsof -ti :PORT。
#
# Usage:
#   bash minimax_h3/stop.sh                  # 默认 fl2va :30010
#   MODEL_VARIANT=ref2va bash minimax_h3/stop.sh   # ref2va :30011
#   PORT=30010 bash minimax_h3/stop.sh       # 指定端口
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh" 2>/dev/null || true

MODEL_VARIANT="${MODEL_VARIANT:-fl2va}"
if [ -z "${PORT:-}" ]; then
    case "$MODEL_VARIANT" in
        ref2va) PORT=30011 ;;
        *)      PORT=30010 ;;
    esac
fi

REPO_DIR2="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${LOG_DIR:-$REPO_DIR2/../MiniMax-H3/logs}"
PID_FILE="$LOG_DIR/serve_${MODEL_VARIANT}_${PORT}.pid"

echo "🛑 [stop] killing sglang serve on port $PORT (variant=$MODEL_VARIANT)"

# 按端口找 PID（lsof 优先，ss fallback，PID 文件兜底）
PIDS=""
if command -v lsof >/dev/null 2>&1; then
    PIDS=$(lsof -ti ":$PORT" 2>/dev/null | sort -u)
fi
if [ -z "$PIDS" ] && command -v ss >/dev/null 2>&1; then
    PIDS=$(ss -tlnp 2>/dev/null | grep ":$PORT " | grep -oP 'pid=\K[0-9]+' | sort -u)
fi
if [ -z "$PIDS" ] && [ -f "$PID_FILE" ]; then
    PIDS=$(cat "$PID_FILE" 2>/dev/null)
    [ -n "$PIDS" ] && echo "  (端口找不到，用 PID 文件 $PID_FILE: $PIDS)"
fi

if [ -z "$PIDS" ]; then
    echo "  ✅ no process on port $PORT (already clean)"
    rm -f "$PID_FILE" 2>/dev/null || true
    exit 0
fi

echo "  🎯 found PIDs: $(echo $PIDS | tr '\n' ' ')"

# 先 SIGTERM（优雅退出），5s 后 SIGKILL
for p in $PIDS; do
    kill "$p" 2>/dev/null || true
done
echo "  ⏳ sent SIGTERM, waiting 5s..."
sleep 5

# 还活着就 SIGKILL
REMAINING=""
for p in $PIDS; do
    if kill -0 "$p" 2>/dev/null; then
        REMAINING="$REMAINING $p"
    fi
done
if [ -n "$REMAINING" ]; then
    echo "  💀 still alive, SIGKILL: $REMAINING"
    for p in $REMAINING; do
        kill -9 "$p" 2>/dev/null || true
    done
    sleep 1
fi

# 确认端口释放
if command -v lsof >/dev/null 2>&1 && lsof -ti ":$PORT" 2>/dev/null | grep -q .; then
    echo "  ⚠️ port $PORT still occupied after kill" >&2
else
    echo "  ✅ port $PORT free"
fi

# 清 PID 文件
rm -f "$PID_FILE" 2>/dev/null || true

echo "🎉 [stop] Done."
