#!/usr/bin/env bash
# 03_run_demo.sh —— 本地起 Gradio demo（等价 HF Space）
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

PORT="${PORT:-7860}"
HOST="${HOST:-127.0.0.1}"

echo "🚀 [03] 启动 Idol Gradio demo"
echo "  🌐 http://$HOST:$PORT"
echo "  ⚠️  demo 默认从 HF 拉 speridlabs/idol；本机已缓存，首次加载会复用"

# demo 需要 gradio，00 里只装了主依赖，这里按需补
python -c "import gradio" 2>/dev/null || {
    echo "📦 装 gradio==5.49.1"
    pip install "gradio==5.49.1" || { echo "❌ gradio 安装失败" >&2; exit 1; }
}

cd "$IDOL_REPO/demo" || { echo "❌ demo 目录不存在" >&2; exit 1; }
GRADIO_SERVER_PORT="$PORT" GRADIO_SERVER_NAME="$HOST" python app.py
