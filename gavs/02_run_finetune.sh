#!/usr/bin/env bash
# 02_run_finetune.sh — GaVS 测试时微调（3 epoch，仅解码器）+ 自动评估。
#
# 对应官方 README 的：
#   python train.py hydra.run.dir=./exp/<scene>/ +experiment=layered_gavs_overfit \
#       dataset.data_path=./gavs-data/dataset/<scene>
#
# 机制说明（读源码+实测确认，别乱改 cwd）：
#   - train.py 不主动 chdir，但 hydra.run.dir=./exp/<scene>/ 会让 hydra 把工作目录
#     切过去 —— checkpoints/、depth_scale.json、visualization/、train.log 实际都落在
#     $GAVS_DIR/exp/<scene>/ 下（2026-10-12 实测，不是仓根）；
#   - 训练结束后 train.py 用 os.system("python evaluate.py") 自动评估，
#     该相对路径要求 cwd = 仓根 —— 所以本脚本固定 cd 到 $GAVS_DIR 运行；
#   - 跑完把 exp/<scene>/ 下的产物搬到 $GAVS_RESULTS_DIR/<scene>/，
#     防止同场景重跑误 resume（train.py 检测到 exp/<scene>/checkpoints/ 存在会续训）。
#
# 用法：
#   SCENE=dynamic_dance bash gavs/02_run_finetune.sh
#   SCENE=mild_walk GPU=0 bash gavs/02_run_finetune.sh
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

SCENE="${SCENE:-dynamic_dance}"
DATA_PATH="$GAVS_DATA_ROOT/dataset/$SCENE"
CKPT_DIR="$GAVS_DATA_ROOT/re10k_v2_checkpoints"
OUT_DIR="$GAVS_RESULTS_DIR/$SCENE"

echo "🚀 [02] GaVS 微调 + 评估: $SCENE"
echo "  📁 数据:   $DATA_PATH"
echo "  🏋️ 预训练: $CKPT_DIR"
echo "  💾 输出:   $OUT_DIR"

# 前置检查
[ -d "$DATA_PATH" ] || { echo "❌ 场景数据不存在: $DATA_PATH（先跑 01）" >&2; exit 1; }
[ -s "$CKPT_DIR/model_re10k_v2.pth" ] || { echo "❌ 预训练权重缺失（先跑 01）" >&2; exit 1; }
[ -f "$GAVS_DIR/train.py" ] || { echo "❌ 官方仓不存在: $GAVS_DIR（先跑 00a）" >&2; exit 1; }

# 防同场景误 resume：checkpoints 在 exp/<scene>/ 下，上次没搬走就报警
EXP_DIR="$GAVS_DIR/exp/$SCENE"
if [ -d "$EXP_DIR/checkpoints" ]; then
    if [ "${FORCE:-0}" = "1" ]; then
        echo "⚠️ FORCE=1，删除残留 $EXP_DIR/checkpoints（上次微调结果将被丢弃）"
        rm -rf "$EXP_DIR/checkpoints"
    else
        echo "❌ $EXP_DIR/checkpoints 已存在（上次残留？）。" >&2
        echo "   train.py 检测到它会续训而不是从头微调。确认无用后删除，或 FORCE=1 重跑。" >&2
        exit 1
    fi
fi

mkdir -p "$OUT_DIR"
cd "$GAVS_DIR" || exit 1

# wandb 不联网（wandb.ai 本机不可达）；train.py 默认 train.logging=true，
# disabled 模式下 init/log 全部变 no-op，不影响训练与自动评估。
export WANDB_MODE=disabled
export WANDB_SILENT=true

echo "🏋️ 开始微调（3 epoch，仅 GS 解码器）..."
python train.py \
    hydra.run.dir=./exp/"$SCENE"/ \
    +experiment=layered_gavs_overfit \
    dataset.data_path="$DATA_PATH" \
    train.load_weights_folder="$CKPT_DIR"
_rc=$?
if [ $_rc -ne 0 ]; then
    echo "❌ train.py failed (rc=$_rc)" >&2
    exit 1
fi

# 收尾搬运（产物在 hydra run dir：exp/<scene>/，不是仓根）
echo "📦 搬运产物 -> $OUT_DIR"
for _item in checkpoints visualization depth_scale.json train.log evaluate.log; do
    if [ -e "$EXP_DIR/$_item" ]; then
        rm -rf "${OUT_DIR:?}/$_item"
        mv "$EXP_DIR/$_item" "$OUT_DIR/"
        echo "  ✅ $_item"
    else
        echo "  ⚠️ 未找到 $_item" >&2
    fi
done

echo ""
echo "🎉 [02] Done: $SCENE"
echo "  📊 指标:   $OUT_DIR/visualization/numerical_results.json"
echo "  🎬 视频:   $OUT_DIR/visualization/*.mp4"
echo "  下一步（换稳定强度重渲染）:"
echo "    SCENE=$SCENE STABILITY=8 bash gavs/03_run_eval.sh"
