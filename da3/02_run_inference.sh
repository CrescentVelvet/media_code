#!/usr/bin/env bash
# 02_run_inference.sh — 华为 Remy 采集包 → DA3（metric 深度+位姿+内参）→ COLMAP 稀疏场景
# 推理转发 da3_human/02（run_da3_inference.py，EXPORT_FORMAT 含 colmap），
# 该脚本在 da3_human env 下验证过；本脚本负责采集包转换与参数转发。
# 输出: $RESULTS_DIR/<场景名>/{images, mini_npz/, colmap/（sparse/0）, depth_vis/}
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

export RAW_DIR="${RAW_DIR:?❌ 请指定采集包路径 RAW_DIR=/mnt/d/dataset/手部微动样例/<包名>}"
export SCENE_NAME="${SCENE_NAME:-$(basename "$RAW_DIR" | sed 's/^[0-9a-f]\{32\}-//' | tr -d '[:space:]')}"
export N_VIEWS="${N_VIEWS:-32}"
export PROCESS_RES="${PROCESS_RES:-504}"
export MAX_VIEWS="${MAX_VIEWS:-0}"   # 0 = 全部（转换已抽 N_VIEWS 帧）
# INFER_GS=1 → 走 DA3 官方 GSDPT 高斯头，一次前向直出 gaussians.ply（不需要 COLMAP，
# 也不需要 30k 步 3DGS 训练）；此时 EXPORT_FORMAT 默认只写 gs_ply。
export INFER_GS="${INFER_GS:-0}"
if [ "$INFER_GS" = "1" ]; then
    export EXPORT_FORMAT="${EXPORT_FORMAT:-gs_ply}"
else
    export EXPORT_FORMAT="${EXPORT_FORMAT:-mini_npz-colmap-depth_vis}"
fi

# Triton 首次运行要编译 CUDA utils（torch.matrix_exp → bmm_outer_product），
# 本机 WSL 没有系统 gcc，借用任一 conda env 的 conda-gcc 即可（无需 apt）。
if [ -z "${CC:-}" ] && ! command -v cc >/dev/null 2>&1; then
    for e in querysplat vggt_human da3_human; do
        g="$HOME/miniconda3/envs/$e/bin/x86_64-conda-linux-gnu-gcc"
        if [ -x "$g" ]; then
            export CC="$g"
            export CXX="${g/gcc/g++}"
            echo "🔧 未找到系统 cc，借用 $e env 的 conda-gcc"
            break
        fi
    done
fi

SCENE_RESULTS="$RESULTS_DIR/$SCENE_NAME"
CAP_DIR="$RESULTS_DIR/_captures/$SCENE_NAME"

echo "🚀 [02] DA3 推理: $SCENE_NAME"
echo "  📁 采集包: $RAW_DIR（$N_VIEWS 视角, process_res=$PROCESS_RES）"
echo "  💾 输出:   $SCENE_RESULTS"

# 1) 采集包 → 去畸变 png（infinisplat_v2 转换器，HEIC 解码在 infinisplat_v2 env）
CONVERT_PYTHON="${CONVERT_PYTHON:-$HOME/miniconda3/envs/infinisplat_v2/bin/python}"
[ -x "$CONVERT_PYTHON" ] || CONVERT_PYTHON=python
if [ -d "$CAP_DIR/images" ] && [ "$(ls "$CAP_DIR/images" | wc -l)" -eq "$N_VIEWS" ] && [ "${FORCE_CONVERT:-0}" != "1" ]; then
    echo "⏭️ 转换产物已存在: $CAP_DIR/images"
else
    echo "🔍 转换采集包 → $CAP_DIR"
    "$CONVERT_PYTHON" "$MEDIA_CODE_ROOT/infinisplat_v2/07_capture_to_npz.py" \
        --raw-dir "$RAW_DIR" --out-dir "$CAP_DIR" --n-views "$N_VIEWS"
    if [ $? -ne 0 ]; then echo "❌ 采集包转换失败" >&2; exit 1; fi
fi

# 2) DA3 推理（转发 da3_human/02；DA3_MODEL_DIR 有本地权重则优先用，否则走 HF repo 缓存）
if [ "$INFER_GS" = "1" ]; then
    echo "🤖 DA3 前馈 + 官方 GSDPT 高斯头（一次前向出 ply，无 COLMAP/无训练）"
else
    echo "🤖 DA3 前馈推理（da3_human env）"
fi
CONDA_ENV=da3_human \
INPUT_DIR="$CAP_DIR/images" \
SCENE_NAME="$SCENE_NAME" \
DA3_MODEL_DIR="$DA3_MODEL_DIR" \
DA3_MODEL_REPO="$DA3_MODEL_REPO" \
PROCESS_RES="$PROCESS_RES" MAX_VIEWS="$MAX_VIEWS" \
INFER_GS="$INFER_GS" \
EXPORT_FORMAT="$EXPORT_FORMAT" \
OUTPUT_DIR="$SCENE_RESULTS" \
bash "$MEDIA_CODE_ROOT/da3_human/02_run_da3_inference.sh"
if [ $? -ne 0 ]; then echo "❌ FAILED" >&2; exit 1; fi

echo "🎉 Done. 产物在 $SCENE_RESULTS"
echo "  📂 深度/位姿: mini_npz/ · COLMAP 稀疏: colmap/ · 深度可视化: depth_vis/"
echo "  下一步: RAW_DIR=... bash da3/03_train_3dgs.sh"
