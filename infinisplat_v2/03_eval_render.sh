#!/usr/bin/env bash
# 03_eval_render.sh — 自接 gsplat 渲染 + 两种口径评测。
#
# 口径 A（自洽 re-render）：用 02 导出的相机渲染输入视角。指标好只说明属性学对了。
# 口径 B（NVS holdout）：留出若干目标视角不参与重建，用它们的 GT 相机渲染。
#   这是唯一能与论文表 1 类比的口径（论文每例 4 个 held-out 目标视角）。
#
# 用法：
#   # A) 对已有结果做自洽渲染
#   RESULT=$HOME/output/infinisplat_v2_results/dtu_provided \
#   IMAGES_DIR=$INFINISPLAT_V2/examples/data/dtu_provided/images bash 03_eval_render.sh
#
#   # B) 从示例里划出 holdout，重跑推理并评测
#   EXAMPLE=dtu_provided MODE=holdout N_INPUT=8 N_TARGET=4 bash 03_eval_render.sh
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

: "${MODE:=rerender}"
: "${N_INPUT:=8}"
: "${N_TARGET:=4}"
: "${EPS2D:=0.01}"

if ! python -c "import gsplat" 2>/dev/null; then
    echo "❌ ERROR: gsplat 未安装（发布包不带渲染器，评测需自接）" >&2
    echo "      WITH_GSPLAT=1 bash $SCRIPT_DIR/00_setup_env.sh" >&2
    exit 1
fi

if [ "$MODE" = "holdout" ]; then
    # ---- 从示例或自有数据划 holdout ----
    if [ -n "${EXAMPLE:-}" ]; then
        SRC_IMAGES="$INFINISPLAT_V2/examples/data/$EXAMPLE/images"
        SRC_CAMS="$INFINISPLAT_V2/examples/data/$EXAMPLE/cameras.npz"
        : "${OUT_NAME:=$EXAMPLE}"
    else
        SRC_IMAGES="${IMAGES_DIR:?ERROR: 需要 EXAMPLE 或 IMAGES_DIR}"
        SRC_CAMS="${CAMERAS:?ERROR: holdout 模式需要 CAMERAS（GT 相机 npz）}"
        : "${OUT_NAME:=$(basename "$SRC_IMAGES")}"
    fi
    : "${SPLIT_DIR:=$RESULTS_DIR/_splits/$OUT_NAME}"
    if [ -d "$SPLIT_DIR" ]; then
        echo "⏭️  复用已有划分: $SPLIT_DIR"
    else
        echo "🔍 划分 holdout: $N_INPUT 输入 / $N_TARGET 目标"
        python "$SCRIPT_DIR/make_holdout.py" --images-dir "$SRC_IMAGES" --cameras "$SRC_CAMS" \
            --out "$SPLIT_DIR" --n-input "$N_INPUT" --n-target "$N_TARGET" || exit 1
    fi

    echo "🚀 用输入子集重建"
    # EXAMPLE 必须清空：否则 02 会优先用官方示例的完整图像目录，忽略这里的子集
    EXAMPLE="" \
    IMAGES_DIR="$SPLIT_DIR/input/images" CAMERAS="$SPLIT_DIR/input/cameras.npz" \
    OUT_NAME="${OUT_NAME}_holdout" MAX_VIEWS=50 LONG_SIDE="${LONG_SIDE:-518}" \
        bash "$SCRIPT_DIR/02_run_inference.sh" || exit 1

    RESULT="$RESULTS_DIR/${OUT_NAME}_holdout"
    # 02 在目录冲突时会加时间戳后缀，取最新的一个
    [ -d "$RESULT" ] || RESULT=$(ls -td "$RESULTS_DIR/${OUT_NAME}_holdout"* | head -1)
    EVAL_IMAGES="$SPLIT_DIR/target/images"
    EVAL_CAMS="$SPLIT_DIR/target/cameras.npz"
    # 目标相机的原始 GT 坐标系与输出的 RDNorm 空间差一个 Sim(3)，必须给出参考相机做对齐
    REF_CAMS="$SPLIT_DIR/input/cameras.npz"
    TAG=holdout
else
    RESULT="${RESULT:?ERROR: rerender 模式需要 RESULT（02 的输出目录）}"
    EVAL_IMAGES="${IMAGES_DIR:?ERROR: rerender 模式需要 IMAGES_DIR（原图目录）}"
    EVAL_CAMS=""
    REF_CAMS=""
    TAG=rerender
fi

echo "🎨 [03] 渲染评测 ($TAG)"
echo "  💾 结果: $RESULT"
echo "  🖼️  图像: $EVAL_IMAGES"

ARGS=(--result "$RESULT" --out "$RESULT/eval" --images-dir "$EVAL_IMAGES" --tag "$TAG" --eps2d "$EPS2D")
[ -n "$EVAL_CAMS" ] && ARGS+=(--cameras "$EVAL_CAMS")
[ -n "$REF_CAMS" ] && ARGS+=(--ref-cameras "$REF_CAMS" --align-out-pose)
python "$SCRIPT_DIR/render_check.py" "${ARGS[@]}" || exit 1

echo "🎉 Done: $RESULT/eval"
