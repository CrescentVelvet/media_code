#!/usr/bin/env bash
# 04_generate_views.sh — Flux1-dev + ControlNet(depth) + IP-Adapter(可选) 生成
# N 个视角的静止人体图像, 用作无模糊三维重建的多视角输入。
#
# 纯 flux_human env (diffusers FluxControlNetPipeline)。
# 输入:
#   $DEPTH_DIR/depth_<view>.png    03 渲染的骨骼深度图 (ControlNet condition)
#   $REF_IMAGE                      参考帧原图 (IP-Adapter 锁外观; 可选)
# 输出:
#   $VIEWS_DIR/view_<view>.png      生成的多视角图像
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

RESULTS_DIR="${RESULTS_DIR:-$REPO_DIR/../flux_human_results}"
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../../model/flux_human}"
DEPTH_DIR="${DEPTH_DIR:-$RESULTS_DIR/depth}"
VIEWS_DIR="${VIEWS_DIR:-$RESULTS_DIR/views}"
SMPL_OUT="${SMPL_OUT:-$RESULTS_DIR/smpl}"

MODEL_PATH="${MODEL_PATH:-$MODEL_DIR/FLUX.1-dev}"
CONTROLNET_PATH="${CONTROLNET_PATH:-$MODEL_DIR/controlnet}"
IPADAPTER_PATH="${IPADAPTER_PATH:-$MODEL_DIR/ip-adapter}"

# 参考帧原图 (02 写进 reference.npz 的 ref_image; 这里 fallback 读文件)
if [ -z "${REF_IMAGE:-}" ] && [ -f "$SMPL_OUT/reference.npz" ]; then
    REF_IMAGE=$(python -c "import numpy as np; d=np.load('$SMPL_OUT/reference.npz', allow_pickle=True); print(str(d['ref_image']))" 2>/dev/null || echo "")
fi

# --- generation params ---
PROMPT="${PROMPT:-a person standing still, full body, detailed clothing, studio lighting, neutral background, photorealistic, high detail}"
NUM_VIEWS="${NUM_VIEWS:-24}"
SEED="${SEED:-231}"                     # 同 seed 锁随机性 (跨视角一致性靠 depth condition + seed)
HEIGHT="${HEIGHT:-1024}"
WIDTH="${WIDTH:-1024}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-28}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-3.5}"
CONTROLNET_SCALE="${CONTROLNET_SCALE:-0.7}"    # ControlNet depth 强度 (0-1)
DTYPE="${DTYPE:-bf16}"
OFFLOAD="${OFFLOAD:-model}"             # model | sequential | none
USE_IPADAPTER="${USE_IPADAPTER:-0}"     # 1 = 加 IP-Adapter (需 xlabs; API 待验证)

echo "=== [04] Flux1 多视角生成 (flux_human env) ==="
echo "  模型路径:    $MODEL_PATH  (FLUX.1-dev)"
echo "  controlnet: $CONTROLNET_PATH"
[ "$USE_IPADAPTER" = "1" ] && echo "  ip-adapter: $IPADAPTER_PATH"
[ -n "$REF_IMAGE" ] && [ "$USE_IPADAPTER" = "1" ] && echo "  参考帧:     $REF_IMAGE"
echo "  深度图目录:  $DEPTH_DIR  (depth_*.png)"
echo "  输出目录:    $VIEWS_DIR  (view_*.png)"
echo "  生成参数:    $NUM_VIEWS 视角, ${WIDTH}x${HEIGHT}, steps=$NUM_INFERENCE_STEPS cfg=$GUIDANCE_SCALE seed=$SEED"
echo "  controlnet:  scale=$CONTROLNET_SCALE  dtype=$DTYPE offload=$OFFLOAD"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "  GPU:         physical $CUDA_VISIBLE_DEVICES"
else
    echo "  GPU:         default cuda:0"
fi

# --- checks ---
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ ERROR: FLUX.1-dev 不存在: $MODEL_PATH. 跑 01_download_models.sh" >&2; exit 1
fi
if [ ! -d "$CONTROLNET_PATH" ]; then
    echo "❌ ERROR: controlnet 不存在: $CONTROLNET_PATH. 跑 01_download_models.sh" >&2; exit 1
fi
if [ ! -d "$DEPTH_DIR" ]; then
    echo "❌ ERROR: 深度图目录不存在: $DEPTH_DIR. 跑 03_render_depth.sh" >&2; exit 1
fi
if [ "$USE_IPADAPTER" = "1" ] && [ ! -d "$IPADAPTER_PATH" ]; then
    echo "❌ ERROR: ip-adapter 不存在: $IPADAPTER_PATH. 跑 01 (或 SKIP_IPADAPTER=1)." >&2; exit 1
fi

mkdir -p "$VIEWS_DIR"

export MODEL_PATH CONTROLNET_PATH IPADAPTER_PATH DEPTH_DIR VIEWS_DIR REF_IMAGE
export PROMPT NUM_VIEWS SEED HEIGHT WIDTH NUM_INFERENCE_STEPS GUIDANCE_SCALE
export CONTROLNET_SCALE DTYPE OFFLOAD USE_IPADAPTER

python "$SCRIPT_DIR/generate_views.py"
if [ $? -ne 0 ]; then
    echo "❌ generate_views.py 失败" >&2; exit 1
fi

echo "🎉 [04] Done. 多视角图像: $VIEWS_DIR/view_*.png"
echo "    Next: 05_reconstruct (3DGS/NeuS 重建) — 见 README 待办"
