#!/usr/bin/env bash
# 90a_check_da3_deps.sh — 横评 DA3 基线链路 preflight（收编自原 da3/00a_setup_env.sh，2026-11-10 da3/ 撤销）
# 不新建 env，只校验被转发的两个 env（da3_human 推理 / vggt_human 3DGS）+ DA3 权重 + 3DGS CUDA 扩展。
# 用法: bash compare_4methods/90a_check_da3_deps.sh
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDIA_CODE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "🚀 [90a] 横评 DA3 基线链路依赖检查"

ok=1
for env in da3_human vggt_human; do
    if [ -x "$HOME/miniconda3/envs/$env/bin/python" ]; then
        echo "✅ env $env 存在"
    else
        echo "❌ env $env 缺失" >&2
        ok=0
    fi
done

# DA3 权重：本地目录（默认与 da3_human 共用）或 HF 缓存二选一
DA3_MODEL_DIR="${DA3_MODEL_DIR:-${DA3_HUMAN_MODEL_DIR:-$HOME/model/da3_human}}"
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--depth-anything--DA3NESTED-GIANT-LARGE-1.1"
if [ -d "$DA3_MODEL_DIR" ] && [ -n "$(ls -A "$DA3_MODEL_DIR" 2>/dev/null)" ]; then
    echo "✅ DA3 权重目录: $DA3_MODEL_DIR"
elif [ -d "$HF_CACHE_DIR" ]; then
    echo "✅ DA3 权重在 HF 缓存: $HF_CACHE_DIR（推理走 from_pretrained 自动命中）"
else
    echo "⚠️ 未找到 DA3 权重，先跑 bash da3_human/01_download_models.sh"
fi

# 3DGS CUDA 扩展（vggt_human env 编译）
if "$HOME/miniconda3/envs/vggt_human/bin/python" -c "import diff_gaussian_rasterization, simple_knn" 2>/dev/null; then
    echo "✅ 3DGS CUDA 扩展可 import（vggt_human env）"
else
    echo "❌ 3DGS CUDA 扩展不可用（参照 vggt_human/00a_setup_env.sh 编译）" >&2
    ok=0
fi

[ "$ok" = "1" ] && echo "🎉 Done. 下一步: bash compare_4methods/90_run_all_scenes.sh" || { echo "❌ 有缺失项" >&2; exit 1; }
