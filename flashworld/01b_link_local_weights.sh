#!/usr/bin/env bash
# 01b_link_local_weights.sh — 把「迅雷手动下载的权重目录」挂进 HF cache 布局。
#
# 为什么需要：app.py 里 `model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"` 是字符串常量，
# 只能靠 HF_HOME 里的 cache 结构解析；直接改代码去指本地目录会把「官方代码不复制不
# 修改」这条规矩破掉。所以这里造一个 cache 快照目录，里面全是**软链接**（不占空间）。
#
# 预期输入（目录结构照 HF 仓库原样）：
#   $FLASHWORLD_MODEL_DIR/Wan2.2-TI2V-5B-Diffusers/{vae,transformer,text_encoder,...}
#   $FLASHWORLD_MODEL_DIR/FlashWorld/model.ckpt
#
# 用法：
#   bash flashworld/01b_link_local_weights.sh
#   WAN_SRC=/mnt/d/迅雷下载/Wan2.2-TI2V-5B-Diffusers bash flashworld/01b_link_local_weights.sh
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

WAN_SRC="${WAN_SRC:-$FLASHWORLD_MODEL_DIR/Wan2.2-TI2V-5B-Diffusers}"
CKPT_SRC="${CKPT_SRC:-$FLASHWORLD_MODEL_DIR/FlashWorld/model.ckpt}"

echo "🚀 [01b] 挂载本地权重到 HF cache"
echo "  📦 Wan 源:   $WAN_SRC"
echo "  🏋️ ckpt 源:  $CKPT_SRC"
echo "  📁 HF cache: $HF_HOME"
echo ""

_link_snapshot() {  # $1=repo_id  $2=snapshot 源目录  $3=额外要挂的文件（可选，逐个以空格分隔）
    local repo="$1" src="$2"; shift 2
    local tag="models--${repo//\//--}"
    local snap="$HF_HOME/hub/$tag/snapshots/local"
    mkdir -p "$snap"
    local n=0
    # 顶层条目逐个软链（不清场：已存在的链接跳过，避免手滑删源文件）
    for entry in "$src"/*; do
        [ -e "$entry" ] || continue
        local base; base="$(basename "$entry")"
        [ -e "$snap/$base" ] && { echo "  ⏭️  跳过已存在: $base"; continue; }
        ln -s "$entry" "$snap/$base"
        n=$((n + 1))
    done
    for extra in "$@"; do
        local base; base="$(basename "$extra")"
        [ -f "$extra" ] || continue
        [ -e "$snap/$base" ] && { echo "  ⏭️  跳过已存在: $base"; continue; }
        ln -s "$extra" "$snap/$base"
        n=$((n + 1))
    done
    # refs/main 指向 local，离线解析（HF_HUB_OFFLINE=1）时才找得到
    mkdir -p "$HF_HOME/hub/$tag/refs"
    echo -n "local" > "$HF_HOME/hub/$tag/refs/main"
    echo "  ✅ $repo: 挂载 $n 个条目 -> $snap"
}

if [ -d "$WAN_SRC" ]; then
    _link_snapshot "Wan-AI/Wan2.2-TI2V-5B-Diffusers" "$WAN_SRC"
else
    echo "  ⚠️ Wan 源目录不存在，跳过: $WAN_SRC" >&2
fi

if [ -f "$CKPT_SRC" ]; then
    # ckpt 源是单文件，临时造一个只含它的目录挂进去
    _tmp="$FLASHWORLD_MODEL_DIR/.flashworld_ckpt_stage"
    mkdir -p "$_tmp"
    [ -e "$_tmp/model.ckpt" ] || ln -s "$CKPT_SRC" "$_tmp/model.ckpt"
    _link_snapshot "imlixinyang/FlashWorld" "$_tmp"
else
    echo "  ⚠️ ckpt 不存在，跳过: $CKPT_SRC" >&2
fi

echo ""
echo "=== verify ==="
CHECK_ONLY=1 python "$SCRIPT_DIR/01_download_models.py"
