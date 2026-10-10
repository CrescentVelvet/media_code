#!/usr/bin/env bash
# 03_run_eval.sh — GaVS 仅评估（用已微调好的 checkpoint 换参数重渲染）。
#
# 典型用途：同一份微调结果，换稳定强度 σ_s 重新渲染（平滑与重建解耦，
# 不需要重新微调）。对应官方 README 的 eval-only 命令。
#
# 前置：02 已跑过（产物在 $GAVS_RESULTS_DIR/<scene>/checkpoints 等）。
#
# 用法：
#   SCENE=dynamic_dance bash gavs/03_run_eval.sh                 # 默认 σ_s=4
#   SCENE=dynamic_dance STABILITY=8 bash gavs/03_run_eval.sh     # σ_s=8 更稳
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

SCENE="${SCENE:-dynamic_dance}"
STABILITY="${STABILITY:-4}"
DATA_PATH="$GAVS_DATA_ROOT/dataset/$SCENE"
RUN_DIR="$GAVS_RESULTS_DIR/$SCENE"
EVAL_DIR="${EVAL_DIR:-stability$STABILITY}"

echo "🚀 [03] GaVS 仅评估: $SCENE (σ_s=$STABILITY)"
echo "  📁 数据: $DATA_PATH"
echo "  🏋️ 模型: $RUN_DIR/checkpoints"
echo "  💾 输出: $RUN_DIR/$EVAL_DIR"

[ -d "$RUN_DIR/checkpoints" ] || {
    echo "❌ 未找到微调产物: $RUN_DIR/checkpoints（先跑 02）" >&2; exit 1; }
[ -d "$DATA_PATH" ] || { echo "❌ 场景数据不存在: $DATA_PATH" >&2; exit 1; }

export WANDB_MODE=disabled
export WANDB_SILENT=true

cd "$GAVS_DIR" || exit 1
# evaluate.py 会 chdir 到 hydra.run.dir，并在那里找 checkpoints/ 和
# depth_scale.json —— 02 已把这两件搬进 $RUN_DIR，恰好满足。
python evaluate.py \
    train.mode='eval' \
    hydra.run.dir="$RUN_DIR" \
    +experiment=layered_gavs_eval \
    dataset.data_path="$DATA_PATH" \
    config.eval_dir="$EVAL_DIR" \
    dataset.stability="$STABILITY"
_rc=$?
[ $_rc -ne 0 ] && { echo "❌ evaluate.py failed (rc=$_rc)" >&2; exit 1; }

echo ""
echo "🎉 [03] Done."
echo "  📊 指标: $RUN_DIR/$EVAL_DIR/numerical_results.json"
echo "  🎬 视频: $RUN_DIR/$EVAL_DIR/*.mp4"
