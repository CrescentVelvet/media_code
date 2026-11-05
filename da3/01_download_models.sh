#!/usr/bin/env bash
# 01_download_models.sh — DA3 权重（depth-anything/DA3NESTED-GIANT-LARGE-1.1）
# ⚠️ 许可：NESTED/GIANT/LARGE 系列权重为 CC BY-NC 4.0（非商用）；
#    Apache 2.0 的 DA3-BASE/DA3METRIC-LARGE 不带位姿/度量组合（见 da3_human/EXPERIMENTS.md）
# HF 缓存里已有（da3_human 跑过）时 snapshot_download 秒过。
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

mkdir -p "$DA3_MODEL_DIR"

if [ -z "${HF_ENDPOINT:-}" ]; then
    if ! curl -sI --max-time 8 -o /dev/null "https://huggingface.co/api/models/$DA3_MODEL_REPO"; then
        echo "⚠️ huggingface.co API 直连失败，回退 aifasthub.com 镜像"
        export HF_ENDPOINT="https://aifasthub.com"
    fi
fi
echo "🚀 [01] 下载 DA3 权重"
echo "  🤖 $DA3_MODEL_REPO"
echo "  💾 目标: $DA3_MODEL_DIR"

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
