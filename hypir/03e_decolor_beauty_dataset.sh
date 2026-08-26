#!/usr/bin/env bash
# 03e_decolor_beauty_dataset.sh — 独立构建"去红润美颜"数据集（不依赖 03d，并列）。
#
# 输入原始人脸文件夹 INPUT_DIR，对每张图：
#   1. Resize512+CenterCrop512 对齐 -> src ([1,3,512,512] in [-1,1])
#   2. RetouchFormer 美颜 -> beauty ([-1,1]，偏红润；内存中间量，不存盘)
#   3. wavelet_reconstruction(content=beauty, style=src) -> decolor
#      = 美颜高频(磨皮纹理) + 原图低频(原色调)  -> 去红润、保留磨皮
#   4. 高斯模糊 src -> lq_gauss (LQ)
# 产出（像素级对齐，同源 src 派生，任意尺寸/宽高比输入都安全——模型 VRT 写死
#       512×512，非方形会被 CenterCrop，hq_beauty_decolor/lq_gauss 存的正是这个 crop）：
#   hq_beauty_decolor/<name>.png = 去红润美颜（D 组 HQ 目标）
#   lq_gauss/<name>.png         = 高斯模糊（LQ；与 03d 同 BLUR_SEED 逐像素一致）
#   compare/<name>.png          = [LQ|orig|beauty|decolor] 横拼（SAVE_COMPARE=1）
#   rest_beauty_decolor.parquet = lq_gauss -> hq_beauty_decolor（04b 直接训）
#
# 原理：RetouchFormer 红润是权重低频色偏(VRT 把瑕疵区推向周围皮肤统计，偏暖)；
#   磨皮是高频变化。wavelet 融合保留美颜高频(磨皮)+原图低频(原色调)即去红润保磨皮。
#   HQ 目标去红润 -> LoRA 学到的还原也去红润 -> 训练(04b)/推理(02)脚本均不改，
#   只换 PARQUET_PATH/WEIGHT_PATH。
#
# 与 03d 并列(非依次)：03d 产 A/B/C，03e 产 D，各自独立跑 RetouchFormer(不先跑 03d)。
#   RetouchFormer 加载/transform/blur 逻辑 copy 自 build_beauty_dataset.py(并列脚本，
#   代码重复正常——各自自包含便于塞进 pipeline)；wavelet 三函数 copy 自
#   HYPIR/utils/common.py:32-80(仅依赖 torch)。零脚本依赖。
#   A/B/C(03d) vs D(03e) 单变量对比：同 BLUR_SEED(默认 231)+同图顺序 => lq_gauss 逐像素一致。
#
# conda env：不强制——默认沿用当前已激活 env(CONDA_DEFAULT_ENV)，缺包就 pip 兜底装。
#   想强制专 env 就设 RETOUCH_CONDA_ENV。官方推荐 retouchformer env(python3.8+torch1.13.1，
#   含 stylegan2 CUDA 算子)，但别的 env 也能跑(op/ 非 Linux 或 torch 版本不匹配会回退纯 PyTorch，
#   慢但能出图)。**前置**：放好 gen_best.pth(retouchformer/01_download_models.sh，百度网盘手动下，提取码 reto)。
#
# 必填：INPUT_DIR=/path/to/faces  (原图人脸文件夹，可含子目录)
# 常用覆盖：
#   OUTPUT_DIR=/.../beauty_decolor_<input>  输出根(默认在 INPUT_DIR 同级建 beauty_decolor_<input_name>/)
#   SAVE_COMPARE=1       额外存 compare/<name>.png ([LQ|orig|beauty|decolor] 横拼，一眼核对去红润效果)
#   SKIP_BLUR=1          只产 hq_beauty_decolor(不建 lq_gauss，也跳过 parquet)
#   SKIP_PARQUET=1       只产图不建 parquet(之后再 03b)
#   BLUR_SEED=231        高斯模糊随机种子(与 03d 同值，保证 lq_gauss 逐像素一致，单变量对比)
#   RESIZE_MODE=square   square(默认,CenterCrop512,任意输入安全) | smallest(仅 Resize512,需方形输入)
#   GPU=0                美颜用哪张卡(Phase B 建 parquet 不用卡)
#   NPROC=4              多卡分片加速(NPROC<=可见卡数; 不设 GPU 用全部可见卡, 或 GPU=0,1,2,3)
#   RETOUCH_DIR/MODEL_DIR/WEIGHT_PATH  RetouchFormer 代码/权重路径(默认与 03d/retouchformer 一致)
# 例：
#   GPU=0 INPUT_DIR=../HYPIR/input/test_faces_hq SAVE_COMPARE=1 SKIP_PARQUET=1 bash hypir/03e_decolor_beauty_dataset.sh
#   GPU=0 INPUT_DIR=../HYPIR/dataset/guojia_datas_20260708 SAVE_COMPARE=1 bash hypir/03e_decolor_beauty_dataset.sh
#   GPU=0,1,2,3 NPROC=4 INPUT_DIR=../HYPIR/dataset/guojia_datas_20260708 SAVE_COMPARE=1 bash hypir/03e_decolor_beauty_dataset.sh
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

: "${INPUT_DIR:?set INPUT_DIR (folder of original face images)}"
[ -d "$INPUT_DIR" ] || { echo "ERROR: INPUT_DIR not found: $INPUT_DIR" >&2; exit 1; }
INPUT_DIR="$(cd "$INPUT_DIR" && pwd)"
INPUT_NAME="$(basename "$INPUT_DIR")"

# 默认输出到 INPUT_DIR 同级的 beauty_decolor_<input_name>/ (hq_beauty_decolor/ + lq_gauss/ [+ compare/])
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "$INPUT_DIR")/beauty_decolor_$INPUT_NAME}"

# ─── Phase A: RetouchFormer 美颜 + wavelet 去红润 + 高斯模糊 ───
# 复用 retouchformer/_env.sh 做代理 + CA bundle + 选卡(GPU=N) + conda 激活，但不强制
# retouchformer env——默认沿用当前已激活 env(CONDA_DEFAULT_ENV)，想强制专 env 就显式设
# RETOUCH_CONDA_ENV=retouchformer(或你起的名)。缺包(torch/torchvision/PIL)就 pip 兜底装。
export CONDA_ENV="${RETOUCH_CONDA_ENV:-${CONDA_DEFAULT_ENV:-base}}"
# shellcheck disable=SC1091
source "$REPO_DIR/retouchformer/_env.sh"

# RetouchFormer 代码与权重目录(默认与仓库布局一致，可被 RETOUCH_DIR/MODEL_DIR 覆盖)。
# 路径取值与 03d/retouchformer/02_run_inference.sh 完全一致 -> 去红润输出与官方推理对齐。
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
export SAVE_COMPARE="${SAVE_COMPARE:-0}"      # 1=额外存 compare/<name>.png ([LQ|orig|beauty|decolor])
export SKIP_BLUR="${SKIP_BLUR:-0}"            # 1=只产 hq_beauty_decolor(不建 lq_gauss，也跳过 parquet)
export BLUR_SEED="${BLUR_SEED:-231}"          # 高斯模糊随机种子(与 03d 同值，保证 lq_gauss 逐像素一致)

echo "=== [03e] Phase A: RetouchFormer 美颜 + wavelet 去红润 -> hq_beauty_decolor[/lq_gauss] ==="
echo " 💎 美颜env:     retouchformer (CONDA_ENV=$CONDA_ENV)"
echo " 💎 代码路径:    $RETOUCH_DIR"
echo " 💎 权重:        $WEIGHT_PATH"
echo " 💎 输入(原图):  $INPUT_DIR"
echo " 💎 输出根:      $OUTPUT_DIR  (hq_beauty_decolor/[+lq_gauss/][+compare/])"
echo " 💎 参数:        resize=$RESIZE_MODE size=$SIZE device=$DEVICE save_compare=$SAVE_COMPARE skip_blur=$SKIP_BLUR blur_seed=$BLUR_SEED nproc=${NPROC:-1}"
if [ "${NPROC:-1}" -gt 1 ]; then
    echo " 💎 多卡:       NPROC=${NPROC} (torchrun 分片, 见下)"
fi
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "  GPU:          physical $CUDA_VISIBLE_DEVICES  [GPU=N to change]"
else
    echo "  GPU:          default cuda:0  [set GPU=N to pin a card]"
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
# 单卡(NPROC=1)走 python，同 03d。
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
    torchrun --nproc_per_node="$NPROC" --master_port="$PORT" "$SCRIPT_DIR/decolor_beauty_dataset.py"
else
    python "$SCRIPT_DIR/decolor_beauty_dataset.py"
fi

echo "=== [03e] Phase A done. hq_beauty_decolor[/lq_gauss] under: $OUTPUT_DIR ==="

# ─── Phase B: 建配对 parquet (复用 03b) ───
# rest_beauty_decolor.parquet : lq_gauss -> hq_beauty_decolor  (D 去红润美颜)
# 与 03d 的 A/B/C 并列：共用同一 lq_gauss(同 seed 逐像素一致)，单变量只 HQ 目标不同。
if [ "${SKIP_BLUR:-0}" = "1" ]; then
    echo "    SKIP_BLUR=1 -> 无 lq_gauss，跳过 parquet。如需配对训练，去掉 SKIP_BLUR 重跑。" >&2
    exit 0
fi
if [ "${SKIP_PARQUET:-0}" = "1" ]; then
    echo "    SKIP_PARQUET=1 -> 跳过 parquet 构建。之后手动建："
    echo "      HQ_DIR=$OUTPUT_DIR/hq_beauty_decolor LQ_DIR=$OUTPUT_DIR/lq_gauss PARQUET_OUT=$OUTPUT_DIR/rest_beauty_decolor.parquet bash $SCRIPT_DIR/03b_build_paired_dataset.sh"
    exit 0
fi

PARQUET_OUT="${PARQUET_OUT:-$OUTPUT_DIR/rest_beauty_decolor.parquet}"
echo "--- [03e] build parquet: LQ=$OUTPUT_DIR/lq_gauss  HQ=$OUTPUT_DIR/hq_beauty_decolor -> $PARQUET_OUT ---"
# 03b 会自己 source hypir/_env.sh 做 conda 激活；传 CONDA_ENV=当前 env(或 HYPIR_CONDA_ENV
# 覆盖)使其不强制切到 hypir——沿用 Phase A 同一 env。03b 自带缺 polars 就 pip 装的兜底。
HQ_DIR="$OUTPUT_DIR/hq_beauty_decolor" LQ_DIR="$OUTPUT_DIR/lq_gauss" PARQUET_OUT="$PARQUET_OUT" \
    CONDA_ENV="${HYPIR_CONDA_ENV:-${CONDA_DEFAULT_ENV:-base}}" \
    bash "$SCRIPT_DIR/03b_build_paired_dataset.sh"

echo "=== [03e] Done. 去红润美颜数据集就绪: $OUTPUT_DIR ==="
echo " 💎   rest_beauty_decolor.parquet (lq_gauss -> hq_beauty_decolor  D 去红润美颜): $PARQUET_OUT"
echo " 💎   next(训一个新 LoRA, OUTPUT_DIR 与 A/B/C 分开):"
echo " 💎     PARQUET_PATH=$PARQUET_OUT OUTPUT_DIR=../HYPIR/experiments/beauty_rest_beauty_decolor bash $SCRIPT_DIR/04b_train_paired.sh"
echo " 💎   训完用 02 推理(WEIGHT_PATH 指向新 checkpoint)，即出'去红润 + 磨皮'效果："
echo " 💎     GPU=0 LQ_DIR=../HYPIR/input/test_faces WEIGHT_PATH=../HYPIR/experiments/beauty_rest_beauty_decolor/checkpoint-N/ema_state_dict.pth bash $SCRIPT_DIR/02_run_inference.sh"
echo " 💎   与 03d 的 A/B/C 对比(肉眼或 05_eval)，看 D 是否红润减弱、磨皮保留。"
