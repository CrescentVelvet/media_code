#!/usr/bin/env bash
# 01f_depth_maps.sh — 逐帧单目深度离线推理（方案 E 离线半）
#
# 模型：UniDepth V1 ViT-L14（本机已有权重 + torch.hub 缓存，完全离线）。
# 运行环境：conda env `gavs`（xformers 0.0.25.post1 是 UniDepth V1 decoder 的硬依赖；
#   deformable_human env 是 torch 2.5.1 没钉版 xformers，勿用）。
#
# 输出：$SOURCE_PATH/depths/{stem}.npz（float16 米制深度 + 可选置信度 + meta），
#       $SOURCE_PATH/depths/_qc/{stem}.jpg（伪彩抽检图，务必目视检查）。
# 训练侧（train_som.py 读 SOM_DEPTH_*，由 02_train.sh 从 DEPTH_* 透传）从
# $SOURCE_PATH/depths/ 读。
#
# Env (all optional, defaults shown):
#   SCENE_NAME=human_seq
#   SOURCE_PATH=        # 默认 $RESULTS_DIR/$SCENE_NAME/colmap_scene
#                       # ⚠️ 必须与 02_train.sh 的 SOURCE_PATH 一致（训练按它找 depths/）
#   DEPTH_ENV=gavs      # UniDepth 所在 conda env
#   DEPTH_VERSION=v1  DEPTH_BACKBONE=vitl14
#   DEPTH_LIMIT=0       # 只算前 N 张（调试用）
#   DEPTH_FORCE=0       # 1=重算已有
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCENE_NAME="${SCENE_NAME:-human_seq}"
DEPTH_ENV="${DEPTH_ENV:-gavs}"
DEPTH_VERSION="${DEPTH_VERSION:-v1}"
DEPTH_BACKBONE="${DEPTH_BACKBONE:-vitl14}"
DEPTH_LIMIT="${DEPTH_LIMIT:-0}"
DEPTH_FORCE="${DEPTH_FORCE:-0}"

# _env.sh 会激活 $CONDA_ENV；这里钉成 gavs（UniDepth 的 xformers 依赖在该 env）
export CONDA_ENV="$DEPTH_ENV"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

SOURCE_PATH="${SOURCE_PATH:-$RESULTS_DIR/$SCENE_NAME/colmap_scene}"
IMAGES_DIR="$SOURCE_PATH/images"
DEPTHS_DIR="$SOURCE_PATH/depths"

echo "🌊 [01f] 单目深度推理（UniDepth ${DEPTH_VERSION}-${DEPTH_BACKBONE}）"
echo "  📂 images:  $IMAGES_DIR"
echo "  💾 depths:  $DEPTHS_DIR"
echo "  🐍 env:     $CONDA_ENV（$(which python 2>/dev/null || echo ?)）"
[ "$DEPTH_LIMIT" != "0" ] && echo "  🧪 limit:   $DEPTH_LIMIT（调试用）"
echo ""

if [ ! -d "$IMAGES_DIR" ]; then
    echo "❌ ERROR: 图像目录不存在: $IMAGES_DIR（先跑 01/01c）" >&2
    exit 1
fi
if ! python -c "import xformers.ops" 2>/dev/null; then
    echo "❌ ERROR: 当前 env（$CONDA_ENV）没有 xformers.ops——UniDepth V1 decoder 需要。" >&2
    echo "       用 DEPTH_ENV=gavs（默认）跑，或先跑 gavs/00a_setup_env.sh 搭环境。" >&2
    exit 1
fi

EXTRA_ARGS=()
[ "$DEPTH_LIMIT" != "0" ] && EXTRA_ARGS+=(--limit "$DEPTH_LIMIT")
[ "$DEPTH_FORCE" = "1" ] && EXTRA_ARGS+=(--force)

python "$SCRIPT_DIR/depth_est.py" \
    --images "$IMAGES_DIR" \
    --out "$DEPTHS_DIR" \
    --version "$DEPTH_VERSION" \
    --backbone "$DEPTH_BACKBONE" \
    "${EXTRA_ARGS[@]}"
rc=$?
# ⚠️ 必须查退出码：2026-10-16 事故——torch.hub 联网失败抛异常，但脚本照样打印
# 成功横幅，导致「01f 已跑完」被误判（depths/ 里其实一个 npz 都没有）。
if [ "$rc" -ne 0 ]; then
    echo "❌ FAILED: depth_est.py 退出码 $rc（详情见上方 traceback）" >&2
    exit "$rc"
fi

# 产物兜底校验：没有 npz 就是失败，不管退出码怎么说
N_NPZ="$(ls "$DEPTHS_DIR"/*.npz 2>/dev/null | wc -l)"
if [ "$N_NPZ" -eq 0 ]; then
    echo "❌ FAILED: $DEPTHS_DIR 下没有任何 .npz（推理未产出）" >&2
    exit 1
fi

echo ""
echo "🎉 [01f] Done（$N_NPZ 个 npz）"
echo "  💾 深度:   $DEPTHS_DIR/*.npz"
echo "  🖼️  抽检:   $DEPTHS_DIR/_qc/*.jpg（务必目视检查！）"
echo "  → 训练开关: DEPTH_LAMBDA=0.05 bash som_human/02_train.sh"
