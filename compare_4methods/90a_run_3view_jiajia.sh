#!/usr/bin/env bash
# 90a_run_3view_jiajia.sh — 佳佳抱枕三图输入（N_VIEWS=3）四方法批量
#
# 口径对齐 32v 轮（90_run_all_scenes.sh / FINDINGS §2）：
#   同采集包、同 linspace 抽帧规则（07_capture_to_npz.py:45 含首末帧，3v=[0,40,81]）、
#   同 ITERATIONS=30000、InfiniSplat 带默认 3000 步精调。
# 产物隔离：SCENE_NAME=佳佳抱枕_3v，四方法输出全部落在 *_3v 目录，不碰 32v 产物
# （32v 的 infinisplat 目录名是 佳佳抱枕_32v，其余三件是 佳佳抱枕，已确认存在）。
#
# 用法:
#   bash compare_4methods/90a_run_3view_jiajia.sh          # 全部四方法串行
#   STEPS="is qs" bash compare_4methods/90a_run_3view_jiajia.sh   # 只跑指定步骤
#
# 可跑性依据：FINDINGS §7 代码级核实（四件均可吃 N=3；原版 COLMAP+3DGS 不参与）。
# 日志: ~/output/batch_4methods/3v_<step>_佳佳抱枕_3v.log
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDIA_CODE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export RAW_DIR="/mnt/d/dataset/手部微动样例/01e954de1b8a4cde9315c976f4dcb964-佳佳抱枕"
export SCENE_NAME="佳佳抱枕_3v"
export N_VIEWS=3
export ITERATIONS=30000

STEPS="${STEPS:-is qs vggt da3}"
LOG_DIR="$HOME/output/batch_4methods"
mkdir -p "$LOG_DIR"

echo "🚀 [90a] 三图批量: $SCENE_NAME（RAW=$RAW_DIR, N_VIEWS=$N_VIEWS, ITERS=$ITERATIONS）"
echo "  步骤: $STEPS"

run() {  # $1=step 名, 其余=命令
    local STEP="$1"; shift
    local LOG="$LOG_DIR/3v_${STEP}_${SCENE_NAME}.log"
    echo ""
    echo "🚀 [3v][$STEP] 开始  ⏱️  $(date +%H:%M:%S)  (log: $LOG)"
    local T0=$SECONDS
    "$@" > "$LOG" 2>&1
    local RC=$?
    local SEC=$((SECONDS - T0))
    if [ $RC -ne 0 ]; then
        echo "❌ [3v][$STEP] FAILED (exit=$RC, ${SEC}s)，日志: $LOG"
        echo "ALL_DONE rc=1 failed_step=$STEP"
        exit 1
    fi
    echo "✅ [3v][$STEP] 完成  ⏱️  ${SEC}s"
}

for S in $STEPS; do
    case "$S" in
        is)
            # InfiniSplat v2：采集包 → 3v 转换 → 前馈+显存测量 → 默认带 3000 步精调
            run "$S" env OUT_NAME="$SCENE_NAME" bash "$MEDIA_CODE_ROOT/infinisplat_v2/07_run_capture.sh" ;;
        qs)
            # QuerySplat：前馈出 gaussians.ply（OPACITY_THRES 默认 0.05，与 32v 首跑同口径）
            run "$S" bash "$MEDIA_CODE_ROOT/querysplat/02_run_inference.sh" ;;
        vggt)
            # VGGT-Ω + 3DGS 30k
            run "$S" bash -c "bash '$MEDIA_CODE_ROOT/vggt_omega/02_run_inference.sh' && bash '$MEDIA_CODE_ROOT/vggt_omega/03_train_3dgs.sh'" ;;
        da3)
            # DA3 前馈 + 3DGS 30k
            run "$S" bash -c "bash '$MEDIA_CODE_ROOT/da3/02_run_inference.sh' && bash '$MEDIA_CODE_ROOT/da3/03_train_3dgs.sh'" ;;
        *)
            echo "⚠️ 未知步骤: $S（可选 is qs vggt da3）" >&2
            exit 1 ;;
    esac
done

echo ""
echo "🎉 [90a] 全部步骤完成: $STEPS"
echo "ALL_DONE rc=0"
