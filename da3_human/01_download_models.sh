#!/usr/bin/env bash
# 01_download_models.sh — 下载 Depth-Anything-3 权重（HF → DA3_MODEL_DIR）
# 用法: bash da3_human/01_download_models.sh
#   DA3_MODEL_REPO 可换模型（见 EXPERIMENT_PLAN.md §3 模型许可注意点）
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

export DA3_MODEL_REPO="${DA3_MODEL_REPO:-depth-anything/DA3NESTED-GIANT-LARGE-1.1}"

echo "🚀 [01] 下载 DA3 权重"
echo "  🤖 repo: $DA3_MODEL_REPO"
echo "  💾 目标: $DA3_MODEL_DIR"
echo "  ⚠️ 许可注意: NESTED/GIANT/LARGE 系列权重为 CC BY-NC 4.0（非商用）；"
echo "     Apache 2.0 的 DA3-BASE/DA3METRIC-LARGE 不带位姿/度量组合，详见 EXPERIMENTS.md"

mkdir -p "$DA3_MODEL_DIR"

# HF 直连不通时回退 hf-mirror（网络会漂移，每次先测 API 小请求，不拉首页大 body）
if [ -z "${HF_ENDPOINT:-}" ]; then
    if ! curl -sI --max-time 8 -o /dev/null "https://huggingface.co/api/models/$DA3_MODEL_REPO"; then
        echo "⚠️ huggingface.co API 直连失败，回退 hf-mirror.com"
        export HF_ENDPOINT="https://hf-mirror.com"
    fi
fi
echo "  🌐 HF_ENDPOINT: ${HF_ENDPOINT:-huggingface.co (direct)}"

python - <<'PYEOF'
import os
from huggingface_hub import snapshot_download

repo = os.environ["DA3_MODEL_REPO"]
dst = os.environ["DA3_MODEL_DIR"]
print(f"📦 snapshot_download {repo} -> {dst}")
path = snapshot_download(repo, local_dir=dst)
print(f"✅ 权重已下载: {path}")
PYEOF
if [ $? -ne 0 ]; then echo "❌ 下载失败（网络问题先看 proxy.env / HF_ENDPOINT）" >&2; exit 1; fi

du -sh "$DA3_MODEL_DIR"
echo "🎉 Done."
