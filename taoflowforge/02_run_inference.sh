#!/usr/bin/env bash
# 02_run_inference.sh — 单图 → 原生三角网格（TaoFlowForge 三阶段全流程）。
#
# 输入：单张 RGB/RGBA 图（建议主体居中、背景简单；有 TFF_REMBG_DIR 时会先抠背景）
# 输出：$RESULTS_DIR/<name>/
#   ├── input.png          # 预处理后实际送入模型的图
#   ├── stage0/1/2.npz     # 各阶段中间产物（含 RNG 状态，可 --resume-from 续跑）
#   ├── mesh_raw.obj       # Stage 2 直接输出（未修复）
#   ├── mesh.obj           # 补洞+法向校正后的最终网格
#   ├── mesh.glb           # 归一到 [-0.5,0.5]³ 的可视化版
#   └── metadata.json
#
# 用法：
#   GPU=0 INPUT=/path/to/image.png bash taoflowforge/02_run_inference.sh
#   GPU=0 INPUT=a.png NAME=my_obj bash taoflowforge/02_run_inference.sh
#   GPU=0 INPUT=a.png OFFLOAD=off bash taoflowforge/02_run_inference.sh  # 显存充裕时全程驻留
#   GPU=0 INPUT=a.png EXTRA_ARGS="--no-fill-holes" bash taoflowforge/02_run_inference.sh
#
# ⚠️ --offload 取值以官方 cli.py 为准：auto | on | off（2026-10-28 核对，默认 auto）。
# 显存：默认 auto（按需 offload）；官方未公布显存数字，
#       首次跑留意 nvidia-smi，OOM 可先试 OFFLOAD=on。
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

INPUT="${INPUT:-}"
SEED="${SEED:-42}"
OFFLOAD="${OFFLOAD:-auto}"        # auto | on | off（上游 choices，非 stage）
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [ -z "$INPUT" ]; then
    echo "❌ ERROR: INPUT 未设置（输入图像路径）" >&2
    echo "  用法: GPU=0 INPUT=/path/to/image.png bash taoflowforge/02_run_inference.sh" >&2
    exit 1
fi
if [ ! -f "$INPUT" ]; then
    echo "❌ ERROR: 输入文件不存在: $INPUT" >&2
    exit 1
fi
for f in "$TFF_STAGE0" "$TFF_STAGE1" "$TFF_STAGE2"; do
    if [ ! -f "$f" ]; then
        echo "❌ ERROR: 权重缺失: $f（先跑 taoflowforge/01_download_models.sh）" >&2
        exit 1
    fi
done
# 发布版 stage0 未内嵌 DINO 权重（2026-10-28 实测 mmap 探测确认），必配 --dino-checkpoint
if [ ! -f "$TFF_DINO_DIR/config.json" ]; then
    echo "❌ ERROR: DINOv3 目录缺失: $TFF_DINO_DIR（ModelScope facebook/dinov3-vitl16-pretrain-lvd1689m）" >&2
    exit 1
fi
if [ ! -d "$TFF_PKG_DIR/taoflowforge" ]; then
    echo "❌ ERROR: 官方代码不存在: $TFF_PKG_DIR（先跑 taoflowforge/00a_setup_env.sh）" >&2
    exit 1
fi

# 输出子目录名：默认用输入文件名（去扩展名），可用 NAME= 覆盖
_base="$(basename "$INPUT")"
NAME="${NAME:-${_base%.*}}"
OUT_DIR="$RESULTS_DIR/$NAME"

echo "🚀 [02] TaoFlowForge 单图 → 网格"
echo "  🖼️ 输入:   $INPUT"
echo "  💾 输出:   $OUT_DIR"
echo "  🏋️ stage0: $TFF_STAGE0"
echo "  🏋️ stage1: $TFF_STAGE1"
echo "  🏋️ stage2: $TFF_STAGE2"
echo "  🧬 DINO:   $TFF_DINO_DIR"
echo "  🌱 seed:   $SEED   offload: $OFFLOAD"
if [ -n "$TFF_REMBG_DIR" ]; then
    echo "  ✂️ RMBG:   $TFF_REMBG_DIR"
else
    echo "  ⚠️ 未配置 TFF_REMBG_DIR —— 不做背景移除，原图背景会保留（白底图最佳）"
fi
echo ""

mkdir -p "$OUT_DIR"

# 组装参数
ARGS=(run
    --image "$INPUT"
    --stage0-checkpoint "$TFF_STAGE0"
    --stage1-checkpoint "$TFF_STAGE1"
    --stage2-checkpoint "$TFF_STAGE2"
    --dino-checkpoint "$TFF_DINO_DIR"
    --output-dir "$OUT_DIR"
    --seed "$SEED"
    --offload "$OFFLOAD"
)
[ -n "$TFF_REMBG_DIR" ] && ARGS+=(--rembg-model "$TFF_REMBG_DIR")
# shellcheck disable=SC2206
[ -n "$EXTRA_ARGS" ] && ARGS+=($EXTRA_ARGS)

# python -m taoflowforge.cli 的包发现依赖 cwd（官方未发布 pip 包）
cd "$TFF_PKG_DIR" || exit 1

echo "⏱️ 开始推理 ..."
_start=$(date +%s)
python -m taoflowforge.cli "${ARGS[@]}"
_rc=$?
_end=$(date +%s)

if [ $_rc -ne 0 ]; then
    echo "❌ FAILED (rc=$_rc)" >&2
    exit 1
fi

echo ""
echo "⏱️ 总耗时: $((_end - _start))s"
echo "📁 产物:"
ls -lh "$OUT_DIR" 2>/dev/null
[ -f "$OUT_DIR/mesh.glb" ] && echo "  🎯 最终网格: $OUT_DIR/mesh.glb"

echo ""
echo "🎉 [02] Done."
