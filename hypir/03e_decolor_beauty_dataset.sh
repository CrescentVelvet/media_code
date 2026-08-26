#!/usr/bin/env bash
# 03e_decolor_beauty_dataset.sh — 对 03d 的美颜图做"去红润"后处理并建 parquet。
#
# 依赖 03d 已产出(像素级对齐)的 hq_orig/ + hq_beauty/ [+ lq_gauss/]，输出:
#   hq_beauty_decolor/<name>.png = wavelet_reconstruction(content=hq_beauty, style=hq_orig)
#                                = 美颜高频(磨皮纹理) + 原图低频(原色调)
#   rest_beauty_decolor.parquet  = lq_gauss -> hq_beauty_decolor  (供 04b 直接训练)
#
# 原理：RetouchFormer 的红润是权重偏置带来的低频色偏(VRT selective self-attention 把
#   瑕疵区推向周围皮肤统计，偏暖)；而磨皮/去瑕疵是高频变化。用原图低频替换美颜低频即
#   去红润、保留美颜高频即保留磨皮。小波三函数 copy 自 HYPIR/utils/common.py:32-80，
#   自包含(仅依赖 torch)，无 HYPIR import 依赖——便于塞进任意人脸增强 pipeline。
#
# 训练/推理脚本均不改：把 rest_beauty_decolor.parquet 喂 04b 训一个新 LoRA(OUTPUT_DIR 分开)，
#   再用该 LoRA 走 02 推理即出"去红润 + 磨皮"效果——红润与否由 HQ 训练目标决定，
#   与推理无关，故 02 无需改动。
#
# 必填：BEAUTY_DIR=/path/to/03d_output  (含 hq_orig/ + hq_beauty/ [+ lq_gauss/])
# 常用覆盖：
#   HQ_ORIG_NAME=hq_orig              原图子目录名(03d 默认 hq_orig)
#   HQ_BEAUTY_NAME=hq_beauty          美颜子目录名(03d 默认 hq_beauty)
#   HQ_DECOLOR_NAME=hq_beauty_decolor 去红润输出子目录名
#   PARQUET_OUT=.../rest_beauty_decolor.parquet
#   SKIP_PARQUET=1                    只产图不建 parquet(之后手动 03b)
#   SAVE_COMPARE=1                    存 compare_decolor/<name>.png = [orig|beauty|decolor]
#   DEVICE=cuda                       默认 cuda(不可用回退 cpu；wavelet 极轻，cpu 也很快)
#   GPU=0                             选卡
# 例：
#   GPU=0 BEAUTY_DIR=../HYPIR/dataset/beauty_guojia_datas_20260708 \
#     SAVE_COMPARE=1 bash hypir/03e_decolor_beauty_dataset.sh
#   # 只产图核对(不建 parquet)：
#   GPU=0 BEAUTY_DIR=.../beauty_xxx SAVE_COMPARE=1 SKIP_PARQUET=1 bash hypir/03e_decolor_beauty_dataset.sh
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"   # conda 激活、代理/CA、GPU=N 选卡(同 hypir 风格)

: "${BEAUTY_DIR:?set BEAUTY_DIR (03d output root, contains hq_orig/ + hq_beauty/ [+ lq_gauss/])}"
[ -d "$BEAUTY_DIR" ] || { echo "ERROR: BEAUTY_DIR not found: $BEAUTY_DIR" >&2; exit 1; }
BEAUTY_DIR="$(cd "$BEAUTY_DIR" && pwd)"

HQ_ORIG_NAME="${HQ_ORIG_NAME:-hq_orig}"
HQ_BEAUTY_NAME="${HQ_BEAUTY_NAME:-hq_beauty}"
HQ_DECOLOR_NAME="${HQ_DECOLOR_NAME:-hq_beauty_decolor}"
LQ_NAME="${LQ_NAME:-lq_gauss}"
PARQUET_OUT="${PARQUET_OUT:-$BEAUTY_DIR/rest_beauty_decolor.parquet}"

export BEAUTY_DIR HQ_ORIG_NAME HQ_BEAUTY_NAME HQ_DECOLOR_NAME
export DEVICE="${DEVICE:-cuda}"
export SAVE_COMPARE="${SAVE_COMPARE:-0}"
export SKIP_PARQUET="${SKIP_PARQUET:-0}"

echo "=== [03e] 去红润后处理: hq_beauty高频 + hq_orig低频 -> $HQ_DECOLOR_NAME ==="
echo " 💎 输入根(03d 输出): $BEAUTY_DIR"
echo " 💎 原图(style,低频):  $BEAUTY_DIR/$HQ_ORIG_NAME"
echo " 💎 美颜(content,高频):$BEAUTY_DIR/$HQ_BEAUTY_NAME  (红润)"
echo " 💎 输出(去红润):      $BEAUTY_DIR/$HQ_DECOLOR_NAME"
[ "${SKIP_PARQUET:-0}" = "1" ] || \
    echo " 💎 parquet:          $PARQUET_OUT  (LQ=$LQ_NAME -> HQ=$HQ_DECOLOR_NAME)"
echo " 💎 参数: device=$DEVICE save_compare=$SAVE_COMPARE skip_parquet=$SKIP_PARQUET (wavelet levels=5, HYPIR fixed)"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "  GPU:                physical $CUDA_VISIBLE_DEVICES  [GPU=N to change]"
else
    echo "  GPU:                default cuda:0  [set GPU=N to pin a card]"
fi

# --- 前置检查 ---
[ -d "$BEAUTY_DIR/$HQ_ORIG_NAME" ] || {
    echo "ERROR: $BEAUTY_DIR/$HQ_ORIG_NAME not found (run 03d_build_beauty_dataset first)." >&2; exit 1; }
[ -d "$BEAUTY_DIR/$HQ_BEAUTY_NAME" ] || {
    echo "ERROR: $BEAUTY_DIR/$HQ_BEAUTY_NAME not found (run 03d_build_beauty_dataset first)." >&2; exit 1; }

# 缺包兜底(同 03d 风格)：wavelet 只需 torch，但读/存图要 torchvision/PIL。
if ! python -c "import torch, torchvision, PIL" 2>/dev/null; then
    echo "--- 缺 torch/torchvision/PIL，pip 兜底安装 ---"
    pip install --trusted-host pypi.org --trusted-host pypi.python.org \
        --trusted-host files.pythonhosted.org --timeout 600 --retries 10 torch torchvision pillow
fi

# ─── Phase A: wavelet 去红融融合 ───
python "$SCRIPT_DIR/decolor_beauty_dataset.py"
if [ $? -ne 0 ]; then
    echo "❌ [03e] Phase A FAILED" >&2
    exit 1
fi
echo "=== [03e] Phase A done. $HQ_DECOLOR_NAME under: $BEAUTY_DIR ==="

# ─── Phase B: 建配对 parquet (复用 03b) ───
# rest_beauty_decolor.parquet : lq_gauss -> hq_beauty_decolor  (D 组 去红润美颜)
# 与 03d 的 A/B/C 并列：共用同一 lq_gauss，单变量只 HQ 目标不同(原图/美颜/加强美颜/去红润美颜)。
if [ "${SKIP_PARQUET:-0}" = "1" ]; then
    echo "    SKIP_PARQUET=1 -> 跳过 parquet。手动建："
    echo "      HQ_DIR=$BEAUTY_DIR/$HQ_DECOLOR_NAME LQ_DIR=$BEAUTY_DIR/$LQ_NAME PARQUET_OUT=$PARQUET_OUT bash $SCRIPT_DIR/03b_build_paired_dataset.sh"
    exit 0
fi
if [ ! -d "$BEAUTY_DIR/$LQ_NAME" ]; then
    echo "⚠️  $BEAUTY_DIR/$LQ_NAME 不存在(03d 用了 SKIP_BLUR=1?) -> 无法配对，跳过 parquet。" >&2
    echo "    如需配对训练，先跑 03d 产出 $LQ_NAME/，或手动指定 LQ_DIR 重跑 03b。" >&2
    exit 0
fi

echo "--- [03e] build parquet: LQ=$BEAUTY_DIR/$LQ_NAME  HQ=$BEAUTY_DIR/$HQ_DECOLOR_NAME -> $PARQUET_OUT ---"
# 03b 自带 conda 激活 + 缺 polars 就 pip 装的兜底；传 CONDA_ENV=当前 env 使其不强制切到 hypir。
HQ_DIR="$BEAUTY_DIR/$HQ_DECOLOR_NAME" LQ_DIR="$BEAUTY_DIR/$LQ_NAME" PARQUET_OUT="$PARQUET_OUT" \
    CONDA_ENV="${CONDA_DEFAULT_ENV:-base}" \
    bash "$SCRIPT_DIR/03b_build_paired_dataset.sh"

echo "=== [03e] Done. 去红润美颜数据集就绪: $BEAUTY_DIR ==="
echo " 💎   rest_beauty_decolor.parquet (lq_gauss -> hq_beauty_decolor  D 去红润美颜): $PARQUET_OUT"
echo " 💎   next(训一个新 LoRA, OUTPUT_DIR 与 A/B/C 分开):"
echo " 💎     PARQUET_PATH=$PARQUET_OUT OUTPUT_DIR=../HYPIR/experiments/beauty_rest_beauty_decolor bash $SCRIPT_DIR/04b_train_paired.sh"
echo " 💎   训完用 02 推理(WEIGHT_PATH 指向新 checkpoint)，即出'去红润 + 磨皮'效果："
echo " 💎     GPU=0 LQ_DIR=.../test_faces WEIGHT_PATH=.../beauty_rest_beauty_decolor/checkpoint-N/ema_state_dict.pth bash $SCRIPT_DIR/02_run_inference.sh"
echo " 💎   与 03d 的 A/B/C 对比(肉眼或 05_eval)，看 D 是否红润减弱、磨皮保留。"
