#!/usr/bin/env bash
# 03d_build_beauty_dataset.sh — 用 RetouchFormer 美颜构建「配对蒸馏数据集」
# (hq=美颜目标, lq=对齐的原图预处理)。
#
# 与 03b(真实配对 LQ+HQ) / 03c(只输入 HQ 在线合成退化) 对照：这里只输入一个
# 「原图人脸」文件夹，对每张跑 RetouchFormer 美颜，同时保存：
#   hq/<name>.png = 美颜结果(训练目标)
#   lq/<name>.png = 模型实际看到的 Resize(512)+CenterCrop(512) 原图(训练输入)
# 两者像素级对齐(来自同一 src 张量)，任意尺寸/宽高比输入都安全——模型 VRT 写死
# 512×512，非方形输入会被 CenterCrop，而 lq 保存的正是这个被 crop 后的版本，故与 hq
# 严丝合缝。这是「直接拷贝原图当 LQ」做不到的(只有输入本就 512×512 方形时拷贝才行)。
#
# 产出可直接喂 03b -> 04b：HYPIR 学到的映射是 LQ(原图) -> HQ(美颜)，即把
# RetouchFormer 的「去瑕疵 + 磨皮保结构」retouching 蒸馏进 HYPIR 的一步扩散框架，
# 实现「人脸增强 + 一点点美颜磨皮」(推理时还享受 HYPIR 的速度与 tiled 任意分辨率)。
#
# ⚠️ 双 conda env：
#   Phase A(美颜) 用 retouchformer env(python3.8 + torch1.13.1，含 stylegan2 CUDA 算子)；
#   Phase B(建 parquet) 切到 hypir env(只需 polars + pillow)。
#   本脚本自动切换——用 RETOUCH_CONDA_ENV / HYPIR_CONDA_ENV 覆盖环境名(默认
#   retouchformer / hypir)。
#
# 必填：INPUT_DIR=/path/to/faces  (原图人脸文件夹，可含子目录)
# 常用覆盖：
#   OUTPUT_DIR=/.../beauty_faces   hq/lq 输出根目录(默认在 INPUT_DIR 同级建 beauty_<input_name>/)
#   SAVE_COMPARE=1                 同时存 compare/<name>.png (LQ|HQ 横向拼接，便于核对对齐/美颜度)
#   RESIZE_MODE=square             square(默认,CenterCrop512,任意输入安全) | smallest(仅 Resize512,需方形输入)
#   SKIP_PARQUET=1                 只产 hq/lq，不建 parquet(之后再 03b)；默认会建 parquet 供 04b 直接用
#   GPU=0                           美颜用哪张卡(Phase B 建 parquet 不用卡)
#
# 例：
#   GPU=0 INPUT_DIR=/data_3d/w00xxxxxx/code/HYPIR/dataset/guojia_datas_20260708 \
#     SAVE_COMPARE=1 bash hypir/03d_build_beauty_dataset.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

: "${INPUT_DIR:?set INPUT_DIR (folder of original face images)}"
[ -d "$INPUT_DIR" ] || { echo "ERROR: INPUT_DIR not found: $INPUT_DIR" >&2; exit 1; }
INPUT_DIR="$(cd "$INPUT_DIR" && pwd)"
INPUT_NAME="$(basename "$INPUT_DIR")"

# 默认输出到 INPUT_DIR 同级的 beauty_<input_name>/ (hq/ + lq/ [+ compare/] 在其下)
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "$INPUT_DIR")/beauty_$INPUT_NAME"}"

# ─── Phase A: RetouchFormer 美颜 (retouchformer env) ───
# 复用 retouchformer/_env.sh：代理 + CA bundle + 选卡(GPU=N) + 激活 retouchformer env。
# 强制 CONDA_ENV=retouchformer(避免用户传 CONDA_ENV=hypir 时误激活错 env)；可用
# RETOUCH_CONDA_ENV 改名。注意：_env.sh 用 SCRIPT_DIR 算 REPO_DIR，这里把 SCRIPT_DIR
# 设成本目录(hypir/) 也能得到正确的 media_code 根 —— proxy.env 路径一致。
export CONDA_ENV="${RETOUCH_CONDA_ENV:-retouchformer}"
# shellcheck disable=SC1091
source "$REPO_DIR/retouchformer/_env.sh"

# RetouchFormer 代码与权重目录(默认与仓库布局一致，可被 RETOUCH_DIR/MODEL_DIR 覆盖)。
# 路径取值与 retouchformer/02_run_inference.sh 完全一致 -> 美颜输出与官方推理对齐。
RETOUCH_DIR="${RETOUCH_DIR:-$REPO_DIR/../RetouchFormer}"
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../../model/RetouchFormer}"
CKPT_DIR_NAME="${CKPT_DIR_NAME:-release_model}"
EPOCH="${EPOCH:-best}"
WEIGHT_FILE="${WEIGHT_FILE:-gen_${EPOCH}.pth}"
CKPT_DIR="${CKPT_DIR:-$MODEL_DIR/$CKPT_DIR_NAME}"
WEIGHT_PATH="${WEIGHT_PATH:-$CKPT_DIR/$WEIGHT_FILE}"

MODEL_NAME="${MODEL_NAME:-RetouchFormer}"
RESIZE_MODE="${RESIZE_MODE:-square}"          # square | smallest
SIZE="${SIZE:-512}"                           # model is fixed to 512
DEVICE="${DEVICE:-cuda}"
export SAVE_COMPARE="${SAVE_COMPARE:-0}"      # 1=额外存 compare/<name>.png (LQ|HQ 横拼)

echo "=== [03d] Phase A: RetouchFormer 美颜 -> 配对 hq/lq (512x512, 对齐) ==="
echo "  美颜env:        retouchformer (CONDA_ENV=$CONDA_ENV)"
echo "  代码路径:       $RETOUCH_DIR"
echo "  权重:           $WEIGHT_PATH"
echo "  输入(原图):     $INPUT_DIR"
echo "  输出根:         $OUTPUT_DIR  (hq/ + lq/+compare/)"
echo "  参数:           resize=$RESIZE_MODE size=$SIZE device=$DEVICE save_compare=$SAVE_COMPARE"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "  GPU:            physical $CUDA_VISIBLE_DEVICES  [GPU=N to change]"
else
    echo "  GPU:            default cuda:0  [set GPU=N to pin a card]"
fi

# --- 前置检查 ---
[ -d "$RETOUCH_DIR" ] || {
    echo "ERROR: RetouchFormer code dir not found at $RETOUCH_DIR." >&2
    echo "       Run bash retouchformer/run_all.sh first (clones the official repo)." >&2; exit 1; }
[ -f "$WEIGHT_PATH" ] || {
    echo "ERROR: checkpoint not found at $WEIGHT_PATH." >&2
    echo "       Run bash retouchformer/01_download_models.sh first (Baidu manual step), or set WEIGHT_PATH." >&2
    exit 1; }

mkdir -p "$OUTPUT_DIR"

export RETOUCH_DIR WEIGHT_PATH MODEL_NAME INPUT_DIR OUTPUT_DIR RESIZE_MODE SIZE DEVICE
python "$SCRIPT_DIR/build_beauty_dataset.py"

echo "=== [03d] Phase A done. hq/lq under: $OUTPUT_DIR ==="

# ─── Phase B: 建配对 parquet (hypir env) ───
if [ "${SKIP_PARQUET:-0}" = "1" ]; then
    echo "    SKIP_PARQUET=1 -> 跳过 parquet 构建。之后手动建："
    echo "      HQ_DIR=$OUTPUT_DIR/hq LQ_DIR=$OUTPUT_DIR/lq bash $SCRIPT_DIR/03b_build_paired_dataset.sh"
    exit 0
fi

HQ_DIR="$OUTPUT_DIR/hq"
LQ_DIR="$OUTPUT_DIR/lq"
PARQUET_OUT="${PARQUET_OUT:-$OUTPUT_DIR/hypir_beauty.parquet}"

echo "=== [03d] Phase B: 建配对 parquet (hypir env) ==="
echo "  HQ_DIR=$HQ_DIR"
echo "  LQ_DIR=$LQ_DIR"
echo "  -> $PARQUET_OUT"
# 03b 会自己 source hypir/_env.sh 激活 hypir env；显式传 CONDA_ENV=hypir(可用
# HYPIR_CONDA_ENV 改名)。Phase A 激活的 retouchformer env 会被 03b 子 shell 覆盖。
HQ_DIR="$HQ_DIR" LQ_DIR="$LQ_DIR" PARQUET_OUT="$PARQUET_OUT" \
    CONDA_ENV="${HYPIR_CONDA_ENV:-hypir}" \
    bash "$SCRIPT_DIR/03b_build_paired_dataset.sh"

echo "=== [03d] Done. 美颜配对数据集就绪: $OUTPUT_DIR ==="
echo "    parquet: $PARQUET_OUT"
echo "    next(训练·暖启动 HYPIR_sd2.pth): PARQUET_PATH=$PARQUET_OUT bash $SCRIPT_DIR/04b_train_paired.sh"
