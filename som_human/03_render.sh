#!/usr/bin/env bash
# 03_render.sh — SoM 运动基模型渲染 + 评测。
#
# MODE=render   测试视角渲染 + PSNR/SSIM/LPIPS（03 之后接 metrics，见下）
# MODE=time     固定视角扫时间（时间一致性检查）
# MODE=original 时间×视角组合渲染——目视检查重影/撕裂的标准工具
#               （与 deformable_human README_wsl.md 的验收口径一致）
#
# Env (all optional, defaults shown):
#   SCENE_NAME=human_seq
#   MODEL_PATH=            # 默认 $RESULTS_DIR/$SCENE_NAME/model
#   MODE=render
#   NUM_BASES=10           # ⚠️ 必须与训练时一致（ckpt 里有断言）
#   ITERATION=-1           # 加载哪个迭代（-1 = 最新）
#   SKIP_TRAIN=0 SKIP_TEST=0
#   RUN_METRICS=1          # render 模式后自动跑官方 metrics.py
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

SCENE_NAME="${SCENE_NAME:-human_seq}"
MODEL_PATH="${MODEL_PATH:-$RESULTS_DIR/$SCENE_NAME/model}"
MODE="${MODE:-render}"
NUM_BASES="${NUM_BASES:-10}"
ITERATION="${ITERATION:--1}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_TEST="${SKIP_TEST:-0}"
RUN_METRICS="${RUN_METRICS:-1}"

echo "🚀 [03] 渲染 SoM 模型（mode=$MODE）"
echo "  💾 模型: $MODEL_PATH"
echo ""

if [ ! -d "$MODEL_PATH/som_motion" ]; then
    echo "❌ ERROR: 未找到 som_motion/ 权重目录: $MODEL_PATH（先跑 02_train.sh）" >&2
    exit 1
fi

RENDER_FLAGS=(-m "$MODEL_PATH" --mode "$MODE" --iteration "$ITERATION")
[ "$SKIP_TRAIN" = "1" ] && RENDER_FLAGS+=(--skip_train)
[ "$SKIP_TEST" = "1" ] && RENDER_FLAGS+=(--skip_test)

export DG_DIR SOM_NUM_BASES="$NUM_BASES"
( cd "$DG_DIR" && python "$SCRIPT_DIR/render_som.py" "${RENDER_FLAGS[@]}" )
if [ $? -ne 0 ]; then
    echo "❌ FAILED: render_som.py 没跑完。常见原因:" >&2
    echo "    - num_bases 断言: NUM_BASES 与训练不一致" >&2
    echo "    - \"Could not recognize scene type!\": cfg_args 里 source_path 是旧绝对路径，" >&2
    echo "      sed -i 's|旧路径|新路径|' $MODEL_PATH/cfg_args" >&2
    exit 1
fi

if [ "$MODE" = "render" ] && [ "$RUN_METRICS" = "1" ]; then
    echo "📊 计算 PSNR/SSIM/LPIPS（官方 metrics.py）"
    ( cd "$DG_DIR" && python metrics.py -m "$MODEL_PATH" ) || \
        echo "⚠️ metrics.py 失败（渲染产物已生成，不影响目视检查）" >&2
fi

echo ""
echo "🎉 [03] Done."
echo "  🖼️ 渲染图: $MODEL_PATH/{train,test}/"
