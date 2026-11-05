#!/usr/bin/env bash
# 02_run_inference.sh — 华为 Remy 采集包 → QuerySplat 前馈重建 → 3DGS ply
# 步骤: 07_capture_to_npz 转换（去畸变+等间隔抽帧）→ scripts/infer.py
# 输出: $RESULTS_DIR/<场景名>/{gaussians ply, cameras, renders, vggt 深度点云等}
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

export RAW_DIR="${RAW_DIR:?❌ 请指定采集包路径 RAW_DIR=/mnt/d/dataset/手部微动样例/<包名>}"
export SCENE_NAME="${SCENE_NAME:-$(basename "$RAW_DIR" | sed 's/^[0-9a-f]\{32\}-//' | tr -d '[:space:]')}"
export N_VIEWS="${N_VIEWS:-32}"
export OPACITY_THRES="${OPACITY_THRES:-0.05}"        # infer.py 默认保存阈值（论文 yaml 是 0.005）
export SAVE_VGGT_PCD="${SAVE_VGGT_PCD:-1}"           # 顺带存 VGM 深度反投影点云（几何对照用）
export TTO="${TTO:-0}"                               # 1 = 开 test-time optimization（20 步，慢但提质）

CAP_DIR="$RESULTS_DIR/_captures/$SCENE_NAME"
OUT_DIR="$RESULTS_DIR/$SCENE_NAME"
CONFIG="$QS_CHECKPOINT_DIR/querysplat_vggto_1B_512_8192.yaml"
CHECKPOINT="$QS_CHECKPOINT_DIR/querysplat_vggto_1B_512_8192.safetensors"

if [ ! -d "$RAW_DIR" ]; then echo "❌ ERROR: 采集包不存在: $RAW_DIR" >&2; exit 1; fi
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ ERROR: 缺 QuerySplat 权重 $CHECKPOINT" >&2
    echo "       先跑: bash $SCRIPT_DIR/01_download_models.sh" >&2
    exit 1
fi
if [ ! -f "$QS_CHECKPOINT_DIR/vggt_omega_1b_512.pt" ]; then
    echo "❌ ERROR: 缺 VGGT-Omega 骨干权重 $QS_CHECKPOINT_DIR/vggt_omega_1b_512.pt（gated，见 01 脚本提示）" >&2
    exit 1
fi

echo "🚀 [02] QuerySplat 重建: $SCENE_NAME"
echo "  📁 采集包: $RAW_DIR（$N_VIEWS 视角）"
echo "  💾 输出:   $OUT_DIR"

# 1) 采集包 → 去畸变 png（复用 infinisplat_v2 的转换器，含 HEIC 解码与畸变修正）
#    转换依赖 pillow_heif，infinisplat_v2 env 自带；其余 env 跑会静默退或解码失败
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

# 2) 前馈推理（必须在官方仓目录下跑：yaml 内 vggt_checkpoint 是相对路径）
echo "🤖 QuerySplat 前馈推理"
INFER_FLAGS=(
    --config "$CONFIG"
    --checkpoint "$CHECKPOINT"
    --input_folder "$CAP_DIR/images"
    --output_dir "$OUT_DIR"
    --gaussian_save_opacity_threshold "$OPACITY_THRES"
    --save_predicted_input_cameras
)
[ "$SAVE_VGGT_PCD" = "1" ] && INFER_FLAGS+=(--save_vggt_depth_pointcloud)
if [ "$TTO" = "1" ]; then
    INFER_FLAGS+=(--use_tto --tto_n_steps "${TTO_STEPS:-20}")
fi
( cd "$QUERYSPLAT_REPO" && python scripts/infer.py "${INFER_FLAGS[@]}" )
if [ $? -ne 0 ]; then echo "❌ FAILED" >&2; exit 1; fi

echo "🎉 Done. 产物在 $OUT_DIR"
echo "  📂 高斯 PLY / 相机 / 渲染见输出目录；预测相机存了 JSON，可与 GT 位姿对齐评测"
