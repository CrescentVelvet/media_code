#!/usr/bin/env bash
# 02_start_reward_servers.sh — 启动 3D + general reward server (后台)。
#
# 用于多节点训练或单独调试 reward server。
# 单节点训练通常不需要单独跑这个——03_run_training.sh 默认调官方的
# run_single_node.sh, 会自动启动 reward server + 训练。
#
# Ctrl+C 停止 (trap 自动 kill 两个 server)。
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

# reward server 用的 GPU (3D + general 共用)
SERVER_VISIBLE_DEVICES="${SERVER_VISIBLE_DEVICES:-0,1}"
SERVER_PORT="${SERVER_PORT:-18089}"
GENERAL_REWARD_PORT="${GENERAL_REWARD_PORT:-18090}"

export PYTHONPATH="${WORLD_R1_DIR}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export no_proxy="${no_proxy:-127.0.0.1,localhost}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"

LOG_DIR="${EXPERIMENTS_DIR}/reward_servers"
mkdir -p "$LOG_DIR"
SERVER_LOG="$LOG_DIR/reward_3d_server.log"
GENERAL_REWARD_LOG="$LOG_DIR/general_reward_server.log"

echo "🚀 [02] 启动 reward server"
echo "  🎮 server GPU:   $SERVER_VISIBLE_DEVICES"
echo "  📐 3D reward:     http://127.0.0.1:$SERVER_PORT  (log: $SERVER_LOG)"
echo "  🤖 general reward: http://127.0.0.1:$GENERAL_REWARD_PORT  (log: $GENERAL_REWARD_LOG)"
echo ""

# 前置检查
if [ ! -d "$WORLD_R1_DIR/scripts" ]; then
    echo "❌ World-R1 代码不在 $WORLD_R1_DIR" >&2
    echo "   Run: INSTALL_DEPS=1 bash $SCRIPT_DIR/00_setup_env.sh" >&2
    exit 1
fi

export REWARD_3D_SERVER_URL="http://127.0.0.1:$SERVER_PORT"
export GENERAL_REWARD_SERVER_URL="http://127.0.0.1:$GENERAL_REWARD_PORT"

cd "$WORLD_R1_DIR"

# --- 启动 3D reward server ---
echo "📦 starting 3D reward server..."
CUDA_VISIBLE_DEVICES="$SERVER_VISIBLE_DEVICES" \
    python -u scripts/serve_reward_3d.py --port "$SERVER_PORT" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "  PID: $SERVER_PID"

# --- 启动 general reward server ---
echo "📦 starting general reward server..."
CUDA_VISIBLE_DEVICES="$SERVER_VISIBLE_DEVICES" \
    python -u scripts/serve_general_reward.py --port "$GENERAL_REWARD_PORT" \
    > "$GENERAL_REWARD_LOG" 2>&1 &
GENERAL_REWARD_PID=$!
echo "  PID: $GENERAL_REWARD_PID"

# --- 等待启动 ---
echo "⏱️ 等待 reward server 初始化 (加载 DA3 + Qwen3-VL, 约 60-90s)..."
sleep 70
echo ""
echo "--- 3D reward server log (tail) ---"
tail -n 20 "$SERVER_LOG" 2>/dev/null || echo "  (no log yet)"
echo ""
echo "--- general reward server log (tail) ---"
tail -n 10 "$GENERAL_REWARD_LOG" 2>/dev/null || echo "  (no log yet)"
echo ""

# --- 检查是否存活 ---
_ok=1
if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "❌ 3D reward server 已退出! 见 $SERVER_LOG" >&2
    _ok=0
fi
if ! kill -0 "$GENERAL_REWARD_PID" 2>/dev/null; then
    echo "❌ general reward server 已退出! 见 $GENERAL_REWARD_LOG" >&2
    _ok=0
fi

if [ "$_ok" = "1" ]; then
    echo "✅ reward server 已就绪"
    echo "  REWARD_3D_SERVER_URL=$REWARD_3D_SERVER_URL"
    echo "  GENERAL_REWARD_SERVER_URL=$GENERAL_REWARD_SERVER_URL"
    echo ""
    echo "现在可以在另一个终端跑训练:"
    echo "  EXTERNAL_REWARD=1 bash $SCRIPT_DIR/03_run_training.sh"
    echo ""
    echo "Ctrl+C 停止 reward server"
fi

# --- trap: Ctrl+C 清理 ---
cleanup() {
    echo ""
    echo "🛑 stopping reward servers..."
    kill "$SERVER_PID" 2>/dev/null || true
    kill "$GENERAL_REWARD_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    wait "$GENERAL_REWARD_PID" 2>/dev/null || true
    echo "🎉 stopped."
}
trap cleanup EXIT INT TERM

# wait 让脚本保持运行 (Ctrl+C 触发 cleanup)
wait
