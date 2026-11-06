#!/usr/bin/env bash
# 90_run_all_scenes.sh — 四方法横评批量跑全部场景（WSL，串行占一张 3090）
# 用法:
#   bash compare_4methods/90_run_all_scenes.sh            # 全部场景（已完成的有 ply 则跳过）
#   SCENES="女生弹琴 男生手机" bash compare_4methods/90_run_all_scenes.sh   # 只跑指定场景
#   N_VIEWS=64 METHODS="infinisplat querysplat" bash ...   # 覆盖视角数/方法子集
#   FORCE=1 bash ...                                       # 忽略已有 ply 重跑
# 日志: ~/output/batch_4methods/<method>_<scene>.log；汇总: ~/output/batch_4methods/summary.tsv
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDIA_CODE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATASET_DIR="${DATASET_DIR:-/mnt/d/dataset/手部微动样例}"
N_VIEWS="${N_VIEWS:-32}"
ITERATIONS="${ITERATIONS:-30000}"
METHODS="${METHODS:-infinisplat querysplat vggt_omega da3}"
LOG_DIR="$HOME/output/batch_4methods"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/summary.tsv"
[ -f "$SUMMARY" ] || printf "scene\tmethod\tstatus\tseconds\tlog\n" > "$SUMMARY"

scene_name() { basename "$1" | sed 's/^[0-9a-f]\{32\}-//' | tr -d '[:space:]'; }

# 产物落点（与四方法脚本约定一致）
ply_of() {
    case "$1" in
        infinisplat) echo "$HOME/output/infinisplat_v2_results/${2}_32v/scene.ply" ;;
        querysplat)  echo "$HOME/output/querysplat_results/$2/gaussians.ply" ;;
        vggt_omega)  echo "$HOME/output/vggt_omega_results/$2/04_model_3dgs/point_cloud/iteration_${ITERATIONS}/point_cloud.ply" ;;
        da3)         echo "$HOME/output/da3_results/$2/04_model_3dgs/point_cloud/iteration_${ITERATIONS}/point_cloud.ply" ;;
    esac
}

run_one() {  # $1=scene_dir $2=method
    local RAW="$1" M="$2" SN LOG SEC T0 RC PLY
    SN="$(scene_name "$RAW")"
    LOG="$LOG_DIR/${M}_${SN}.log"
    PLY="$(ply_of "$M" "$SN")"
    if [ -f "$PLY" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "⏭️  [$M] $SN 已有产物，跳过（FORCE=1 重跑）"
        printf "%s\t%s\tSKIP\t0\t%s\n" "$SN" "$M" "$LOG" >> "$SUMMARY"
        return 0
    fi
    echo "🚀 [$M] $SN  (log: $LOG)"
    T0=$SECONDS
    export RAW_DIR="$RAW" N_VIEWS="$N_VIEWS" ITERATIONS="$ITERATIONS"
    case "$M" in
        infinisplat)
            OUT_NAME="${SN}_32v" bash "$MEDIA_CODE_ROOT/infinisplat_v2/07_run_capture.sh" > "$LOG" 2>&1 ;;
        querysplat)
            bash "$MEDIA_CODE_ROOT/querysplat/02_run_inference.sh" > "$LOG" 2>&1 ;;
        vggt_omega)
            { bash "$MEDIA_CODE_ROOT/vggt_omega/02_run_inference.sh" && \
              bash "$MEDIA_CODE_ROOT/vggt_omega/03_train_3dgs.sh"; } > "$LOG" 2>&1 ;;
        da3)
            { bash "$MEDIA_CODE_ROOT/da3/02_run_inference.sh" && \
              bash "$MEDIA_CODE_ROOT/da3/03_train_3dgs.sh"; } > "$LOG" 2>&1 ;;
    esac
    RC=$?
    SEC=$((SECONDS - T0))
    if [ $RC -eq 0 ] && [ -f "$PLY" ]; then
        echo "✅ [$M] $SN  ${SEC}s → $PLY"
        printf "%s\t%s\tOK\t%d\t%s\n" "$SN" "$M" "$SEC" "$LOG" >> "$SUMMARY"
    else
        echo "❌ [$M] $SN 失败（exit=$RC, ${SEC}s），日志: $LOG"
        printf "%s\t%s\tFAIL\t%d\t%s\n" "$SN" "$M" "$SEC" "$LOG" >> "$SUMMARY"
    fi
}

# 场景清单
if [ -n "${SCENES:-}" ]; then
    SCENE_DIRS=()
    for s in $SCENES; do
        d="$(ls -d "$DATASET_DIR"/*"-$s" 2>/dev/null | head -1)"
        [ -n "$d" ] && SCENE_DIRS+=("$d") || echo "⚠️ 场景未找到: $s"
    done
else
    mapfile -t SCENE_DIRS < <(ls -d "$DATASET_DIR"/*/ 2>/dev/null)
fi
echo "📋 ${#SCENE_DIRS[@]} 场景 × 方法集 [$METHODS]，N_VIEWS=$N_VIEWS, ITERATIONS=$ITERATIONS"

for RAW in "${SCENE_DIRS[@]}"; do
    for M in $METHODS; do
        run_one "$RAW" "$M"
    done
done

echo ""
echo "🎉 批量完成。汇总: $SUMMARY"
column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
