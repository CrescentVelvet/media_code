#!/usr/bin/env bash
# 01_colmap.sh — 对自备数据跑 COLMAP SfM（从零重建，精化 AR/VIO 位姿）。
#
# 输入目录结构（官方约定）：
#   <SCENE_DIR>/images/*.png
# 输出（与官方 colmap.py 相同布局，dataset_readers 直接可读）：
#   <SCENE_DIR>/sparse/0/{cameras,images,points3D}.bin   （loader 优先读 bin）
#   <SCENE_DIR>/sparse/0_txt/                            （txt 副本，调试用）
#
# ⚠️ 不用官方 scripts/colmap.py：它写死 --SiftExtraction.use_gpu（COLMAP 3.x 参数名），
#    4.x 已拆分为 FeatureExtraction.use_gpu（通用）+ SiftExtraction.*（SIFT 专有），会
#    直接报 "unrecognised option"。本脚本直调 colmap 4.2.1（同 som_human/01c 的实测环境）。
#
# 内参固定（PINHOLE + ba_refine_* 全关）：AR 采集内参可信，BA 只动位姿和点，
# 避免 focal 漂移和图像时序耦合（fid→time 映射依赖帧序稳定）。
#
# 用法：
#   SCENE_DIR=/path/to/scene CAM_PARAMS_FILE=/path/to/cameras.txt bash 01_colmap.sh
#   或显式： SCENE_DIR=... FX=1380.8 FY=1384.2 CX=717.7 CY=952.9 bash 01_colmap.sh
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

SCENE_DIR="${SCENE_DIR:?❌ SCENE_DIR not set (dir containing images/)}"
COLMAP_ENV="${COLMAP_ENV:-colmap}"
MATCHER="${MATCHER:-exhaustive}"   # exhaustive | sequential
USE_GPU="${USE_GPU:-1}"

if [ ! -d "$SCENE_DIR/images" ]; then
    echo "❌ ERROR: $SCENE_DIR/images not found" >&2
    exit 1
fi

# ── 内参：CAM_PARAMS_FILE（COLMAP cameras.txt 格式）或 FX/FY/CX/CY ─────────
if [ -n "${CAM_PARAMS_FILE:-}" ]; then
    read -r _camid _model _w _h FX FY CX CY < <(grep -v '^#' "$CAM_PARAMS_FILE" | head -1)
    echo "📐 内参来自 $CAM_PARAMS_FILE: fx=$FX fy=$FY cx=$CX cy=$CY (${_w}x${_h})"
fi
: "${FX:?❌ 需要 FX/FY/CX/CY 或 CAM_PARAMS_FILE}" "${FY:?}" "${CX:?}" "${CY:?}"

COLMAP_BIN_DIR="$(conda info --base)/envs/$COLMAP_ENV/bin"
if [ ! -x "$COLMAP_BIN_DIR/colmap" ]; then
    echo "❌ ERROR: colmap not found in env '$COLMAP_ENV'" >&2
    echo "       Install: conda create -n colmap --override-channels -c conda-forge 'colmap=*cuda*' openimageio=3.1 -y" >&2
    exit 1
fi
COLMAP_BIN="$COLMAP_BIN_DIR/colmap"
# conda env 的 colmap 直调二进制需 env 自己的 lib 优先（libOpenImageIO 等）
export LD_LIBRARY_PATH="$(conda info --base)/envs/$COLMAP_ENV/lib:${LD_LIBRARY_PATH:-}"

N_IMG="$(ls "$SCENE_DIR/images" | wc -l)"
echo "🚀 [01] COLMAP SfM（从零，固定内参）"
echo "  📁 scene: $SCENE_DIR  📸 $N_IMG images"
echo "  🔧 $($COLMAP_BIN -v 2>/dev/null | head -1)"

DB="$SCENE_DIR/database.db"
SPARSE_BIN="$SCENE_DIR/sparse_bin"
OUT="$SCENE_DIR/sparse/0"
rm -f "$DB"
# COLMAP 4.x 要求 mapper 的 output_path 是已存在的目录
rm -rf "$SPARSE_BIN" "$SCENE_DIR/sparse"
mkdir -p "$SPARSE_BIN" "$OUT"

echo "🔍 [1/4] feature_extractor ..."
"$COLMAP_BIN" feature_extractor \
    --database_path "$DB" \
    --image_path "$SCENE_DIR/images" \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model PINHOLE \
    --ImageReader.camera_params "$FX,$FY,$CX,$CY" \
    --FeatureExtraction.use_gpu "$USE_GPU" \
    > "$SCENE_DIR/colmap_log.txt" 2>&1 || { echo "❌ feature_extractor FAILED（见 colmap_log.txt）" >&2; exit 1; }

echo "🔗 [2/4] $MATCHER matcher ..."
if [ "$MATCHER" = "sequential" ]; then
    "$COLMAP_BIN" sequential_matcher --database_path "$DB" \
        --FeatureMatching.use_gpu "$USE_GPU" \
        --SequentialMatching.loop_detection 1 --SequentialMatching.overlap 20 \
        >> "$SCENE_DIR/colmap_log.txt" 2>&1 || { echo "❌ matcher FAILED" >&2; exit 1; }
else
    "$COLMAP_BIN" exhaustive_matcher --database_path "$DB" \
        --FeatureMatching.use_gpu "$USE_GPU" \
        >> "$SCENE_DIR/colmap_log.txt" 2>&1 || { echo "❌ matcher FAILED" >&2; exit 1; }
fi

echo "🗺️ [3/4] mapper（BA 固定内参）..."
"$COLMAP_BIN" mapper \
    --database_path "$DB" \
    --image_path "$SCENE_DIR/images" \
    --output_path "$SPARSE_BIN" \
    --Mapper.ba_refine_focal_length 0 \
    --Mapper.ba_refine_principal_point 0 \
    --Mapper.ba_refine_extra_params 0 \
    >> "$SCENE_DIR/colmap_log.txt" 2>&1 || { echo "❌ mapper FAILED" >&2; exit 1; }

# ── 4. 选注册帧数最多的子模型（防场景断裂时误取小模型）────────────────────
echo "🎯 [4/4] 选最大子模型并转 bin/txt ..."
BEST=""
BEST_N=-1
TMP_TXT="$SCENE_DIR/.models_txt_tmp"
rm -rf "$TMP_TXT"; mkdir -p "$TMP_TXT"
for m in "$SPARSE_BIN"/*/; do
    [ -f "$m/images.bin" ] || continue
    mkdir -p "$TMP_TXT/$(basename "$m")"
    "$COLMAP_BIN" model_converter --input_path "$m" \
        --output_path "$TMP_TXT/$(basename "$m")" --output_type TXT >/dev/null 2>&1
    n=$(( $(grep -cv '^#' "$TMP_TXT/$(basename "$m")/images.txt" 2>/dev/null || echo 0) / 2 ))
    echo "  📁 model $(basename "$m"): $n registered"
    if [ "$n" -gt "$BEST_N" ]; then BEST_N="$n"; BEST="$m"; fi
done
if [ -z "$BEST" ]; then
    echo "❌ ERROR: mapper 没产出模型（特征匹配失败？见 colmap_log.txt）" >&2
    exit 1
fi
N_MODELS="$(ls -d "$SPARSE_BIN"/*/ 2>/dev/null | wc -l)"
[ "$N_MODELS" -gt 1 ] && echo "⚠️ mapper 产出 $N_MODELS 个子模型（场景断裂），取最大的 $(basename "$BEST")"

"$COLMAP_BIN" model_converter --input_path "$BEST" --output_path "$OUT" --output_type BIN >/dev/null 2>&1
mkdir -p "$SCENE_DIR/sparse/0_txt"
"$COLMAP_BIN" model_converter --input_path "$BEST" --output_path "$SCENE_DIR/sparse/0_txt" --output_type TXT >/dev/null 2>&1
rm -rf "$TMP_TXT"

if [ "$BEST_N" -lt "$N_IMG" ]; then
    echo "⚠️ WARNING: 只注册 $BEST_N/$N_IMG 帧（缺帧会让 fid→time 映射错位，建议查日志）" >&2
fi
echo "🎉 Done. $BEST_N/$N_IMG registered → $OUT"
