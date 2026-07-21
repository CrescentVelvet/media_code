#!/usr/bin/env bash
# 03d_build_beauty_dataset.sh — 用 RetouchFormer 美颜 + 高斯模糊构建「配对对比数据集」
# (hq_orig=原图对齐 crop, hq_beauty=美颜, lq_gauss=高斯模糊退化)。
#
# 与 03b(真实配对 LQ+HQ) / 03c(只输入 HQ 在线合成退化) 对照：这里只输入一个
# 「原图人脸」文件夹，对每张同时保存 THREE 张像素级对齐的 512×512 PNG：
#   hq_orig/<name>.png   = 原图对齐 crop(Resize512+CenterCrop512)  —— 复原实验的 HQ 目标
#   hq_beauty/<name>.png = RetouchFormer 美颜结果                  —— 美颜实验的 HQ 目标
#   lq_gauss/<name>.png  = 同一 crop 的高斯模糊退化                 —— 两个实验共用的 LQ 输入
# 三者都派生自模型输入 src 张量(同一 crop)，故像素级对齐——任意尺寸/宽高比输入都安全
# (模型 VRT 写死 512×512，非方形会被 CenterCrop，而 hq_orig/lq_gauss 存的正是这个 crop 版本)。
#
# 为什么是三套(两张 parquet)而不是一套：
#   现有 03c/04c 在线退化路径(LQ=高斯模糊, HQ=原图)虽能复原模糊，但会「长痘变丑」——
#   模型过度增强、凭空 invent 皮肤瑕疵。把 HQ 目标换成 RetouchFormer 的美颜版(已去瑕疵+磨皮
#   保结构)而 LQ 保持同样的模糊，模型仍学去模糊(增强不失)但目标变成干净光滑皮肤，故不再
#   invent 瑕疵。同时建两套可 A/B 对比：
#     rest.parquet        : lq_gauss -> hq_orig   (基线 = 现有 03c 风格复原)
#     rest_beauty.parquet : lq_gauss -> hq_beauty (复原 + 美颜，修掉长痘问题)
#   各用 04b 训一个(OUTPUT_DIR 分开)，再用 05/02 评测算指标，对比哪个不毁脸。
#
# 高斯模糊复现本仓 HYPIR clone 里简化版 batch_transform.py 的逻辑(随机 kernel 3/5/7/9/11、
# sigma 1-2、重复 1-5 次)，每图一个 FIXED seeded 实现(离线，非每 epoch 重随机)。
#   NB: 模糊作用于 raw 对齐 crop(非 USM(orig))，与 03c 的 LQ=blur(USM(orig)) 略有偏差；
#       但 A/B 共用同一 lq_gauss，对比仍是单变量(HQ 目标不同)。BLUR_SEED 可复现地重随机。
#
# conda env：不强制——默认沿用当前已激活 env(CONDA_DEFAULT_ENV)，缺包就 pip 兜底装。
#   想强制专 env 就设 RETOUCH_CONDA_ENV(Phase A)/ HYPIR_CONDA_ENV(Phase B)。
#   (官方推荐 Phase A 用 retouchformer env: python3.8 + torch1.13.1 含 stylegan2 CUDA 算子；
#    但别的 env 也能跑——op/ 非 Linux 或 torch 版本不匹配会回退纯 PyTorch，慢但能出图。)
#
# 必填：INPUT_DIR=/path/to/faces  (原图人脸文件夹，可含子目录)
# 常用覆盖：
#   OUTPUT_DIR=/.../beauty_faces   hq_orig/hq_beauty/lq_gauss 输出根(默认在 INPUT_DIR 同级建 beauty_<input>/)
#   SAVE_COMPARE=1                 同时存 compare/<name>.png ([LQ|orig|beauty] 横拼，便于核对对齐/美颜/模糊度)
#   RESIZE_MODE=square             square(默认,CenterCrop512,任意输入安全) | smallest(仅 Resize512,需方形输入)
#   SKIP_BLUR=1                    只产 hq_orig+hq_beauty(不建 lq_gauss)，则也跳过 parquet(无 LQ 没法配对)
#   BLUR_SEED=231                  高斯模糊随机种子(复现用)
#   SKIP_PARQUET=1                 只产图，不建 parquet(之后再 03b)；默认会建两张 parquet 供 04b 直接用
#   GPU=0                           美颜用哪张卡(Phase B 建 parquet 不用卡)
#   NPROC=4                         多卡分片加速 Phase A(NPROC<=可见卡数; 不设 GPU 用全部可见卡, 或 GPU=0,1,2,3)
#
# 例：
#   GPU=0 INPUT_DIR=/data_3d/w00xxxxxx/code/HYPIR/dataset/guojia_datas_20260708 \
#     SAVE_COMPARE=1 bash hypir/03d_build_beauty_dataset.sh
#   # 多卡加速(4 卡，不设 GPU 用全部可见卡)：
#   NPROC=4 INPUT_DIR=/data_3d/w00xxxxxx/code/HYPIR/dataset/guojia_datas_20260708 \
#     SAVE_COMPARE=1 bash hypir/03d_build_beauty_dataset.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

: "${INPUT_DIR:?set INPUT_DIR (folder of original face images)}"
[ -d "$INPUT_DIR" ] || { echo "ERROR: INPUT_DIR not found: $INPUT_DIR" >&2; exit 1; }
INPUT_DIR="$(cd "$INPUT_DIR" && pwd)"
INPUT_NAME="$(basename "$INPUT_DIR")"

# 默认输出到 INPUT_DIR 同级的 beauty_<input_name>/ (hq_orig/ + hq_beauty/ + lq_gauss/ [+ compare/])
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "$INPUT_DIR")/beauty_$INPUT_NAME}"

# ─── Phase A: RetouchFormer 美颜 + 高斯模糊 ───
# 复用 retouchformer/_env.sh 做代理 + CA bundle + 选卡(GPU=N) + conda 激活，但不强制
# retouchformer env——默认沿用当前已激活 env(CONDA_DEFAULT_ENV)，想强制专 env 就显式设
# RETOUCH_CONDA_ENV=retouchformer(或你起的名)。缺包(torch/torchvision/PIL)就 pip 兜底装。
# 注意：_env.sh 用 SCRIPT_DIR 算 REPO_DIR，这里 SCRIPT_DIR=本目录(hypir/) 也得到正确的
# media_code 根 —— proxy.env 路径一致。
export CONDA_ENV="${RETOUCH_CONDA_ENV:-${CONDA_DEFAULT_ENV:-base}}"
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
export SAVE_COMPARE="${SAVE_COMPARE:-0}"      # 1=额外存 compare/<name>.png ([LQ|orig|beauty] 横拼)
export SKIP_BLUR="${SKIP_BLUR:-0}"            # 1=只产 hq_orig+hq_beauty(不建 lq_gauss，也跳过 parquet)
export BLUR_SEED="${BLUR_SEED:-231}"          # 高斯模糊随机种子(复现用)

echo "=== [03d] Phase A: RetouchFormer 美颜 + 高斯模糊 -> 配对对比 hq_orig/hq_beauty/lq_gauss ==="
echo " 💎 美颜env:        retouchformer (CONDA_ENV=$CONDA_ENV)"
echo " 💎 代码路径:       $RETOUCH_DIR"
echo " 💎 权重:           $WEIGHT_PATH"
echo " 💎 输入(原图):     $INPUT_DIR"
echo " 💎 输出根:         $OUTPUT_DIR  (hq_orig/ + hq_beauty/ + lq_gauss/ + compare/)"
echo " 💎 参数:           resize=$RESIZE_MODE size=$SIZE device=$DEVICE save_compare=$SAVE_COMPARE skip_blur=$SKIP_BLUR blur_seed=$BLUR_SEED nproc=${NPROC:-1}"
if [ "${NPROC:-1}" -gt 1 ]; then
    echo " 💎 多卡:           NPROC=${NPROC} (torchrun 分片, 见下)"
fi
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
    echo "       Run bash retouchformer/01_download_models.sh first (Baidu manual step), or set WEIGHT_PATH." >&2; exit 1; }

mkdir -p "$OUTPUT_DIR"

export RETOUCH_DIR WEIGHT_PATH MODEL_NAME INPUT_DIR OUTPUT_DIR RESIZE_MODE SIZE DEVICE
export NPROC="${NPROC:-1}"                 # >1 -> torchrun 多卡分片(每进程绑一张卡跑自己的图片子集)

# 缺包就装（沿用当前 env；torch/torchvision/PIL 缺任一就 pip 兜底装。想用特定 torch 版本
# 或 CUDA build 请提前在当前 env 装好，这里只兜底默认 build。）
if ! python -c "import torch, torchvision, PIL" 2>/dev/null; then
    echo "--- 缺 torch/torchvision/PIL，pip 兜底安装 ---"
    pip install --trusted-host pypi.org --trusted-host pypi.python.org \
        --trusted-host files.pythonhosted.org --timeout 600 --retries 10 torch torchvision pillow
fi

# 多卡：torchrun 把图片列表 strided 分片到 N 个进程，每个进程在 cuda:LOCAL_RANK 上独立加载
# 模型跑自己的子集（输出按相对路径写、互不重叠，无需同步——等价于 N 个单卡推理并行）。
# 单卡(NPROC=1)走 python，同现状。多卡用法：NPROC=4 (不设 GPU 用全部可见卡) 或 GPU=0,1,2,3 NPROC=4。
if [ "$NPROC" -gt 1 ]; then
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        _NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')
    else
        _NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$_NGPU" -eq 0 ] && _NGPU=1
    fi
    if [ "$_NGPU" -lt "$NPROC" ]; then
        echo "ERROR: NPROC=$NPROC 但只有 $_NGPU 张可见 GPU。设 NPROC<=可见卡数(或 GPU=0,1,2,3 + NPROC=4)。" >&2; exit 1
    fi
    PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()' 2>/dev/null)
    : "${PORT:=29531}"
    echo " 💎 多卡: NPROC=$NPROC (可见 $_NGPU 卡), torchrun --master_port=$PORT"
    torchrun --nproc_per_node="$NPROC" --master_port="$PORT" "$SCRIPT_DIR/build_beauty_dataset.py"
else
    python "$SCRIPT_DIR/build_beauty_dataset.py"
fi

echo "=== [03d] Phase A done. hq_orig/hq_beauty/lq_gauss under: $OUTPUT_DIR ==="

# ─── Phase B: 建两张配对 parquet (hypir env) ───
# rest.parquet        : lq_gauss -> hq_orig   (基线复原)
# rest_beauty.parquet : lq_gauss -> hq_beauty  (复原+美颜)
# 两张共用同一 lq_gauss(LQ)，只 HQ 目标不同 -> 单变量对比 HQ 美颜 vs 原图对「长痘」的影响。
if [ "${SKIP_BLUR:-0}" = "1" ]; then
    echo "    SKIP_BLUR=1 -> 无 lq_gauss，跳过 parquet。如需配对训练，去掉 SKIP_BLUR 重跑。"
    exit 0
fi
if [ "${SKIP_PARQUET:-0}" = "1" ]; then
    echo "    SKIP_PARQUET=1 -> 跳过 parquet 构建。之后手动建："
    echo "      HQ_DIR=$OUTPUT_DIR/hq_orig  LQ_DIR=$OUTPUT_DIR/lq_gauss PARQUET_OUT=$OUTPUT_DIR/rest.parquet        bash $SCRIPT_DIR/03b_build_paired_dataset.sh"
    echo "      HQ_DIR=$OUTPUT_DIR/hq_beauty LQ_DIR=$OUTPUT_DIR/lq_gauss PARQUET_OUT=$OUTPUT_DIR/rest_beauty.parquet bash $SCRIPT_DIR/03b_build_paired_dataset.sh"
    exit 0
fi

build_parquet() {  # <hq_subdir> <parquet_name>
    local hq_sub="$1" pq="$2"
    local hq="$OUTPUT_DIR/$hq_sub"
    local lq="$OUTPUT_DIR/lq_gauss"
    local out="$OUTPUT_DIR/$pq"
    echo "--- [03d] build parquet: LQ=$lq  HQ=$hq  -> $out ---"
    # 03b 会自己 source hypir/_env.sh 做 conda 激活；传 CONDA_ENV=当前 env(或 HYPIR_CONDA_ENV
    # 覆盖)使其不强制切到 hypir——沿用 Phase A 同一 env。03b 自带缺 polars 就 pip 装的兜底。
    HQ_DIR="$hq" LQ_DIR="$lq" PARQUET_OUT="$out" \
        CONDA_ENV="${HYPIR_CONDA_ENV:-${CONDA_DEFAULT_ENV:-base}}" \
        bash "$SCRIPT_DIR/03b_build_paired_dataset.sh"
}

build_parquet "hq_orig"   "rest.parquet"
build_parquet "hq_beauty" "rest_beauty.parquet"

echo "=== [03d] Done. 美颜对比数据集就绪: $OUTPUT_DIR ==="
echo " 💎   rest.parquet        (lq_gauss->hq_orig  基线复原): $OUTPUT_DIR/rest.parquet"
echo " 💎   rest_beauty.parquet (lq_gauss->hq_beauty 复原+美颜): $OUTPUT_DIR/rest_beauty.parquet"
echo " 💎   next(各自训一个, OUTPUT_DIR 分开):"
echo " 💎     A 基线:   PARQUET_PATH=$OUTPUT_DIR/rest.parquet        OUTPUT_DIR=$OUTPUT_DIR/exp_rest        bash $SCRIPT_DIR/04b_train_paired.sh"
echo " 💎     B 美颜:   PARQUET_PATH=$OUTPUT_DIR/rest_beauty.parquet OUTPUT_DIR=$OUTPUT_DIR/exp_rest_beauty bash $SCRIPT_DIR/04b_train_paired.sh"
echo " 💎   训完用 05_eval / 02_run_inference 对比两组复原图，看 B 是否修掉了「长痘变丑」。"
