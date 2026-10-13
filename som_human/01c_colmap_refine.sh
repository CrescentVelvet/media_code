#!/usr/bin/env bash
# 01c_colmap_refine.sh — COLMAP SfM 精化手机 VIO 位姿（治根因）
#
# 动机：01b 重投影诊断 inlier 436~112750 摆幅 250 倍 → App VIO 位姿漂移是
# double_wave PSNR 钉死 13~14 的根因（点数×7、有无运动模型均无效）。
# 与 deformable_human 证伪清单「手机 VI 位姿直接替 COLMAP -4dB」互证。
#
# 做法：对 01 产出的去畸变图像**从零跑 SfM**（GPU SIFT + exhaustive match + mapper BA）。
# 内参固定为 01 写出的 PINHOLE 标定（cv2 去畸变 + alpha=0 裁剪后有效，160 图
# 自标定风险大），BA 只精化位姿与三角化点云——点云同时替换 App 的 pcd.ply 作
# 3DGS 初始化。
#
# 输入：$SCENE_DIR/{images/, sparse/0/cameras.txt}（01_convert_capture 产物）
# 输出：$OUT_DIR/
#         images -> 符号链接到 $SCENE_DIR/images（不复制，~2GB）
#         sparse/0/{cameras,images,points3D}.txt   # COLMAP 精化模型（txt）
#
# Env (all optional, defaults shown):
#   SCENE_NAME=double_wave
#   SCENE_DIR=$RESULTS_DIR/$SCENE_NAME/colmap_scene
#   OUT_DIR=$RESULTS_DIR/$SCENE_NAME/colmap_refined
#   COLMAP_BIN=colmap          # CUDA 版：conda create -n colmap -c conda-forge 'colmap=*cuda*'
#   USE_GPU_SIFT=1             # 0 = CPU SIFT（慢数倍，应急用）
#   MAX_IMAGE_SIZE=3200        # SIFT 提取最长边（COLMAP 默认 3200）
# 用法：SCENE_NAME=double_wave bash som_human/01c_colmap_refine.sh
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

SCENE_NAME="${SCENE_NAME:-double_wave}"
SCENE_DIR="${SCENE_DIR:-$RESULTS_DIR/$SCENE_NAME/colmap_scene}"
OUT_DIR="${OUT_DIR:-$RESULTS_DIR/$SCENE_NAME/colmap_refined}"
COLMAP_BIN="${COLMAP_BIN:-colmap}"
# 自动发现：conda env 'colmap'（CUDA 版，本机安装路径）
if ! command -v "$COLMAP_BIN" >/dev/null 2>&1 && [ -x "$HOME/miniconda3/envs/colmap/bin/colmap" ]; then
    COLMAP_BIN="$HOME/miniconda3/envs/colmap/bin/colmap"
fi
USE_GPU_SIFT="${USE_GPU_SIFT:-1}"
MAX_IMAGE_SIZE="${MAX_IMAGE_SIZE:-3200}"

echo "🧭 [01c] COLMAP SfM 精化位姿"
echo "  📂 输入场景:   $SCENE_DIR"
echo "  💾 输出:       $OUT_DIR"
echo "  🔧 colmap:     $COLMAP_BIN（GPU SIFT=$USE_GPU_SIFT, max_image_size=$MAX_IMAGE_SIZE）"

# ── 前置检查 ───────────────────────────────────────────────────────────────
if [ ! -d "$SCENE_DIR/images" ] || [ ! -f "$SCENE_DIR/sparse/0/cameras.txt" ]; then
    echo "❌ ERROR: 场景不完整: $SCENE_DIR（先跑 01_convert_capture.sh）" >&2
    exit 1
fi
if ! command -v "$COLMAP_BIN" >/dev/null 2>&1; then
    echo "❌ ERROR: colmap 不可用: $COLMAP_BIN" >&2
    echo "       装 CUDA 版: conda create -n colmap --override-channels -c conda-forge 'colmap=*cuda*'" >&2
    exit 1
fi
# conda env 里的 colmap 直接调二进制会缺动态库（libOpenImageIO 等）——
# 自动把「bin 的上级目录/lib」加进 LD_LIBRARY_PATH（系统 colmap 则无害）
_COLMAP_LIB="$(dirname "$(dirname "$(readlink -f "$(command -v "$COLMAP_BIN")")")")/lib"
if [ -d "$_COLMAP_LIB" ]; then
    export LD_LIBRARY_PATH="$_COLMAP_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if ! "$COLMAP_BIN" -h >/dev/null 2>&1; then
    echo "❌ ERROR: colmap 无法启动（动态库缺失？）: $COLMAP_BIN" >&2
    exit 1
fi

# 从 01 写出的 cameras.txt 取 PINHOLE 内参（单相机假设，与 01 一致）
CAM_LINE="$(grep -v '^#' "$SCENE_DIR/sparse/0/cameras.txt" | head -1)"
read -r _CID MODEL W H FX FY CX CY _ <<<"$CAM_LINE"
if [ "$MODEL" != "PINHOLE" ]; then
    echo "❌ ERROR: 期望 PINHOLE，实际 $MODEL" >&2
    exit 1
fi
N_IMG="$(find "$SCENE_DIR/images" -maxdepth 1 -name '*.png' | wc -l)"
echo "  📷 相机: PINHOLE ${W}x${H} fx=$FX fy=$FY cx=$CX cy=$CY；共 $N_IMG 帧"
if [ "$N_IMG" -lt 10 ]; then
    echo "❌ ERROR: 图像数异常（$N_IMG）" >&2
    exit 1
fi

mkdir -p "$OUT_DIR/sparse"
DB="$OUT_DIR/database.db"
rm -f "$DB"
# COLMAP 4.x 要求 mapper 的 output_path 是已存在的目录（3.x 会自动建）
rm -rf "$OUT_DIR/sparse_bin"
mkdir -p "$OUT_DIR/sparse_bin"

# ── 1. 特征提取（GPU SIFT，单相机固定内参）────────────────────────────────
echo ""
echo "🔍 [01c.1] feature_extractor ..."
"$COLMAP_BIN" feature_extractor \
    --database_path "$DB" \
    --image_path "$SCENE_DIR/images" \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model PINHOLE \
    --ImageReader.camera_params "$FX,$FY,$CX,$CY" \
    --SiftExtraction.use_gpu "$USE_GPU_SIFT" \
    --SiftExtraction.max_image_size "$MAX_IMAGE_SIZE"

# ── 2. 匹配（160 帧规模用 exhaustive，不漏帧间弱连接）─────────────────────
echo ""
echo "🔗 [01c.2] exhaustive_matcher ..."
"$COLMAP_BIN" exhaustive_matcher \
    --database_path "$DB" \
    --SiftMatching.use_gpu "$USE_GPU_SIFT"

# ── 3. 增量 SfM（内参固定，BA 只动位姿+点）────────────────────────────────
echo ""
echo "🗺️ [01c.3] mapper（ba_refine_focal_length/principal_point/extra_params 全关）..."
"$COLMAP_BIN" mapper \
    --database_path "$DB" \
    --image_path "$SCENE_DIR/images" \
    --output_path "$OUT_DIR/sparse_bin" \
    --Mapper.ba_refine_focal_length 0 \
    --Mapper.ba_refine_principal_point 0 \
    --Mapper.ba_refine_extra_params 0

# ── 4. 取最大子模型转 txt，组装成 D-3DGS 可读场景 ─────────────────────────
BEST="$(ls -d "$OUT_DIR/sparse_bin"/*/ 2>/dev/null | sort -V | tail -1)"
if [ -z "$BEST" ]; then
    echo "❌ ERROR: mapper 没产出模型（特征匹配失败？）" >&2
    exit 1
fi
N_MODELS="$(ls -d "$OUT_DIR/sparse_bin"/*/ | wc -l)"
[ "$N_MODELS" -gt 1 ] && echo "⚠️ mapper 产出 $N_MODELS 个子模型（场景断裂），取最大的 ${BEST}"
rm -rf "$OUT_DIR/sparse/0"
mkdir -p "$OUT_DIR/sparse/0"
"$COLMAP_BIN" model_converter \
    --input_path "$BEST" \
    --output_path "$OUT_DIR/sparse/0" \
    --output_type TXT
ln -sfn "$SCENE_DIR/images" "$OUT_DIR/images"

# ── 5. 完整性校验：所有帧必须注册（缺帧会让 D-3DGS 的 fid 映射错位）────────
N_REG="$(grep -cv '^#' "$OUT_DIR/sparse/0/images.txt" || true)"   # 每帧 2 行（位姿+points2D）
N_REG=$((N_REG / 2))
N_PTS="$(grep -cv '^#' "$OUT_DIR/sparse/0/points3D.txt" || true)"
echo ""
echo "📊 [01c] 注册帧数 $N_REG / $N_IMG，三角化点 $N_PTS"
if [ "$N_REG" -ne "$N_IMG" ]; then
    echo "❌ ERROR: 有 $((N_IMG - N_REG)) 帧未注册。不能进训练（fid 错位）。" >&2
    echo "       排查方向: ① --Mapper.* 放宽（min_num_matches 15→10）" >&2
    echo "                 ② 退化路径: point_triangulator 固定 VIO 位姿 + bundle_adjuster 只精化" >&2
    exit 1
fi
echo "🎉 [01c] Done -> $OUT_DIR"
echo "  → 校验: bash som_human/01b_check_reproj.py 对比精化前后 inlier"
echo "  → 训练: SCENE_NAME=$SCENE_NAME SOURCE_PATH=$OUT_DIR MODEL_PATH=\$RESULTS_DIR/$SCENE_NAME/model_colmap bash som_human/02_train.sh"
