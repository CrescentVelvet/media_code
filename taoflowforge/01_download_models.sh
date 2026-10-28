#!/usr/bin/env bash
# 01_download_models.sh — 下载 TaoFlowForge 权重（stage0/1/2.pt）。
#
# 权重托管在 ModelScope（非 gated，无需 token）：
#   https://www.modelscope.cn/models/Taobao3D/TaoFlowForge/files
# 快照下载后归一到 $MODEL_DIR/stage{0,1,2}.pt（快照内实际子路径不保证，
# 脚本会递归找到后硬链接/复制到标准位置）。
#
# 用法：
#   bash taoflowforge/01_download_models.sh              # ModelScope SDK 下载
#   SRC_DIR=/mnt/d/downloads/TaoFlowForge bash taoflowforge/01_download_models.sh
#                                                        # 导入手工/迅雷下好的目录
#   FORCE=1 bash taoflowforge/01_download_models.sh      # 重下
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

SRC_DIR="${SRC_DIR:-}"
FORCE="${FORCE:-0}"

echo "🚀 [01] 准备 TaoFlowForge 权重"
echo "  🏋️ 目标目录: $MODEL_DIR"
echo "  🌐 ModelScope: $TFF_MODELSCOPE_ID"
echo ""

mkdir -p "$MODEL_DIR"

# ── 归一函数：在某目录下找到 stage{0,1,2}.pt 并放到 $MODEL_DIR 根 ─────────
# 优先硬链接（同盘秒完、不占双份空间），跨盘回退复制。
_normalize() {
    local search_root="$1"
    local name f sz
    for name in stage0.pt stage1.pt stage2.pt; do
        if [ -f "$MODEL_DIR/$name" ]; then
            continue
        fi
        f="$(find "$search_root" -name "$name" -type f 2>/dev/null | head -1)"
        if [ -z "$f" ]; then
            echo "  ❌ 在 $search_root 下找不到 $name" >&2
            return 1
        fi
        ln "$f" "$MODEL_DIR/$name" 2>/dev/null || cp -f "$f" "$MODEL_DIR/$name" || {
            echo "  ❌ 归一失败: $f -> $MODEL_DIR/$name" >&2
            return 1
        }
        sz=$(stat -c %s "$MODEL_DIR/$name")
        echo "  ✅ $name  ($((sz / 1024 / 1024)) MiB)  <- $f"
    done
    return 0
}

# ── 情况 A：导入外部已下载目录（迅雷 / 手工）───────────────────────────────
if [ -n "$SRC_DIR" ]; then
    if [ ! -d "$SRC_DIR" ]; then
        echo "❌ ERROR: SRC_DIR 不存在: $SRC_DIR" >&2
        exit 1
    fi
    echo "📦 从 $SRC_DIR 导入 ..."
    _normalize "$SRC_DIR" || exit 1
    echo ""
    echo "🎉 [01] Done. 下一步："
    echo "  GPU=0 INPUT=/path/to/image.png bash taoflowforge/02_run_inference.sh"
    exit 0
fi

# ── 情况 B：已存在且完整，跳过 ────────────────────────────────────────────
if [ "$FORCE" != "1" ] && [ -f "$TFF_STAGE0" ] && [ -f "$TFF_STAGE1" ] && [ -f "$TFF_STAGE2" ]; then
    echo "⏭️  三个 stage 权重已存在，跳过下载（FORCE=1 可强制重下）"
    ls -lh "$TFF_STAGE0" "$TFF_STAGE1" "$TFF_STAGE2"
    echo ""
    echo "🎉 [01] Done."
    exit 0
fi

# ── 情况 C：ModelScope SDK 下载 ───────────────────────────────────────────
if ! python -c "import modelscope" 2>/dev/null; then
    echo "📦 modelscope 未安装，先装 ..."
    pip install -i "${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}" modelscope || {
        echo "❌ modelscope 安装失败" >&2
        exit 1
    }
fi

MS_CACHE="$MODEL_DIR/_ms_snapshot"
mkdir -p "$MS_CACHE"
echo "📦 modelscope snapshot 下载（支持断点续传，中断后重跑即可）..."
python - "$TFF_MODELSCOPE_ID" "$MS_CACHE" <<'PY'
import sys
from modelscope import snapshot_download

model_id, local_dir = sys.argv[1], sys.argv[2]
# allow_patterns 只取权重与配置，跳过 README/截图等
path = snapshot_download(
    model_id,
    local_dir=local_dir,
    allow_patterns=["*.pt", "*.json", "*.txt", "*.md"],
)
print(f"  ✅ snapshot at: {path}")
PY
if [ $? -ne 0 ]; then
    echo "❌ ModelScope 下载失败" >&2
    echo "  备用：浏览器/迅雷打开 https://www.modelscope.cn/models/$TFF_MODELSCOPE_ID/files" >&2
    echo "       逐个下 stage0.pt / stage1.pt / stage2.pt，再：" >&2
    echo "       SRC_DIR=<存放目录> bash taoflowforge/01_download_models.sh" >&2
    exit 1
fi

echo "📦 归一权重到 $MODEL_DIR ..."
_normalize "$MS_CACHE" || exit 1

# 归一完成后快照目录里的 .pt 若是复制（跨盘）会占双份空间，提示清理
echo ""
echo "📁 最终权重："
ls -lh "$TFF_STAGE0" "$TFF_STAGE1" "$TFF_STAGE2"
echo ""
echo "  （快照缓存 $MS_CACHE 可手动删除——硬链接场景删除安全，复制场景删了省空间）"

echo ""
echo "🎉 [01] Done. 下一步："
echo "  GPU=0 INPUT=/path/to/image.png bash taoflowforge/02_run_inference.sh"
