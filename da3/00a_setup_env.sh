#!/usr/bin/env bash
# 00a_setup_env.sh — WSL 验证 da3 链路依赖（DA3 推理在 da3_human env，3DGS 在 vggt_human env）
# 本仓是编排层：不新建重复 env，只校验两个被转发的 env 可用。
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

echo "🚀 [00a] da3 链路依赖检查"

ok=1
for env in da3_human vggt_human; do
    if [ -x "$HOME/miniconda3/envs/$env/bin/python" ]; then
        echo "✅ env $env 存在"
    else
        echo "❌ env $env 缺失" >&2
        ok=0
    fi
done

# DA3 权重：HF 缓存（da3_human 已下载过 DA3NESTED-GIANT-LARGE-1.1）或本地目录二选一
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--depth-anything--DA3NESTED-GIANT-LARGE-1.1"
if [ -d "$DA3_MODEL_DIR" ] && [ -n "$(ls -A "$DA3_MODEL_DIR" 2>/dev/null)" ]; then
    echo "✅ DA3 权重目录: $DA3_MODEL_DIR"
elif [ -d "$HF_CACHE_DIR" ]; then
    echo "✅ DA3 权重在 HF 缓存: $HF_CACHE_DIR（推理走 from_pretrained 自动命中）"
else
    echo "⚠️ 未找到 DA3 权重，跑 bash da3/01_download_models.sh 下载"
fi

# 3DGS CUDA 扩展（vggt_human env 编译）
if "$HOME/miniconda3/envs/vggt_human/bin/python" -c "import diff_gaussian_rasterization, simple_knn" 2>/dev/null; then
    echo "✅ 3DGS CUDA 扩展可 import（vggt_human env）"
else
    echo "❌ 3DGS CUDA 扩展不可用（参照 vggt_human/00a_setup_env.sh 编译）" >&2
    ok=0
fi

[ "$ok" = "1" ] && echo "🎉 Done. 下一步: bash da3/01_download_models.sh（如权重缺失）" || { echo "❌ 有缺失项" >&2; exit 1; }
