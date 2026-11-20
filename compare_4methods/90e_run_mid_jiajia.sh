#!/usr/bin/env bash
# 90e_run_mid_jiajia.sh — 佳佳抱枕中间档（1008）四方法批量：效果 / 显存 / 耗时
#
# 背景：「八百多分辨率也能跑通」的核查实验。2K（1920）已判全灭（FINDINGS §9），
# 中间档定 1008（= 2K 的 52.5%，504 的 2 倍，14 的整数倍；DA3-GS 在 P0-1 已实测能跑）。
# QuerySplat 不测（options.py:103 硬锁 512²，800/1008 同样拒收）。
#
# 口径与 90d（2K 批量）完全对齐：同采集包、N_VIEWS=32、ITERATIONS=30000、
# REF_STRATEGY=first、3DGS 回炉 RES=1；唯一变量 = 输入分辨率 504/518/512 → 1008。
# 产物隔离：SCENE_NAME=佳佳抱枕_mid，输出全部落 *_mid 目录。
# 每步附带 nvidia-smi 采样器（5s 间隔）记录峰值显存 → mid_<step>_*.vram 日志。
#
# 用法:
#   bash compare_4methods/90e_run_mid_jiajia.sh           # 四方法串行
#   STEPS="is vggt" bash compare_4methods/90e_run_mid_jiajia.sh
# 日志: ~/output/batch_4methods/mid_<step>_佳佳抱枕_mid.log（.vram 为显存采样）
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDIA_CODE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export RAW_DIR="/mnt/d/dataset/手部微动样例/01e954de1b8a4cde9315c976f4dcb964-佳佳抱枕"
export SCENE_NAME="佳佳抱枕_mid"
export N_VIEWS=32
export ITERATIONS=30000
export REF_STRATEGY=first
export MID_RES=1008

STEPS="${STEPS:-is vggt da3 da3gs}"   # 快的先跑；QuerySplat 权重锁 512² 不测
LOG_DIR="$HOME/output/batch_4methods"
mkdir -p "$LOG_DIR"

echo "🚀 [90e] 中间档（$MID_RES）批量: $SCENE_NAME（N_VIEWS=$N_VIEWS, ITERS=$ITERATIONS）"
echo "  步骤: $STEPS"

FAILED=""
run() {  # $1=step 名, 其余=命令；失败不中断（能否跑通本身就是结果）
    local STEP="$1"; shift
    local LOG="$LOG_DIR/mid_${STEP}_${SCENE_NAME}.log"
    local VRL="$LOG_DIR/mid_${STEP}_${SCENE_NAME}.vram"
    echo ""
    echo "🚀 [mid][$STEP] 开始  ⏱️  $(date +%H:%M:%S)  (log: $LOG)"
    # 显存采样器：5s 间隔，步骤结束后取峰值
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 5 > "$VRL" 2>/dev/null &
    local VPID=$!
    local T0=$SECONDS
    "$@" > "$LOG" 2>&1
    local RC=$?
    local SEC=$((SECONDS - T0))
    kill $VPID 2>/dev/null; wait $VPID 2>/dev/null
    local MAXV=$(sort -n "$VRL" 2>/dev/null | tail -1)
    echo "📊 [mid][$STEP] 峰值显存: ${MAXV:-未知} MiB"
    if [ $RC -ne 0 ]; then
        echo "❌ [mid][$STEP] FAILED (exit=$RC, ${SEC}s)，日志: $LOG —— 继续下一家"
        FAILED="$FAILED $STEP(rc=$RC)"
        return
    fi
    echo "✅ [mid][$STEP] 完成  ⏱️  ${SEC}s"
}

for S in $STEPS; do
    case "$S" in
        is)
            # InfiniSplat v2：LONG_SIDE 518→1008（含默认 3000 步精调，timeout 4h）
            # 官方训练上限 1022，1008 在设计区间内
            run "$S" env OUT_NAME="$SCENE_NAME" LONG_SIDE=$MID_RES \
                 timeout 14400 bash "$MEDIA_CODE_ROOT/infinisplat_v2/07_run_capture.sh" ;;
        vggt)
            # VGGT-Ω 1008（timeout 3h：512 前向 ~2min，token ×3.9 + naive 注意力平方项）
            # + 3DGS 30k 全分辨率回炉（RES=1，timeout 8h）
            run "$S" bash -c \
              "timeout 10800 env RESOLUTION=$MID_RES bash '$MEDIA_CODE_ROOT/vggt_omega/02_run_inference.sh' \
               && timeout 28800 env RES=1 bash '$MEDIA_CODE_ROOT/vggt_omega/03_train_3dgs.sh'" ;;
        da3)
            # DA3 前馈 1008（P0-1 实测 282s）+ 3DGS 30k 全分辨率回炉（RES=1）
            run "$S" bash -c \
              "timeout 7200 env PROCESS_RES=$MID_RES bash '$MEDIA_CODE_ROOT/compare_4methods/90b_run_da3_inference.sh' \
               && timeout 28800 env RES=1 bash '$MEDIA_CODE_ROOT/compare_4methods/90c_train_da3_3dgs.sh'" ;;
        da3gs)
            # DA3-GS：97 全 SH 导出 + 98 相机 @1008（各 timeout 2h）
            # 注意 P0-1 教训：1008 下 24.4M 高斯 full-SH 导出曾 RAM OOM（plyfile 序列化），
            # 若复现则改走 gs_ply_dc 口径并在 FINDINGS 注明
            run "$S" bash -c \
              "timeout 7200 env IMAGES_DIR='$HOME/output/da3_results/_captures/$SCENE_NAME/images' \
               OUT_DIR='$HOME/output/da3_gs_results/$SCENE_NAME' \
               PROCESS_RES=$MID_RES REF_STRATEGY=first MAX_VIEWS=0 \
               DA3_MODEL_DIR=/mnt/d/wheel/da3_human_ms \
               ~/miniconda3/envs/da3_human/bin/python '$MEDIA_CODE_ROOT/compare_4methods/97_da3_gs_export_full_sh.py' \
               && timeout 7200 env IMAGES_DIR='$HOME/output/da3_results/_captures/$SCENE_NAME/images' \
               OUT_DIR='$HOME/output/da3_gs_results/${SCENE_NAME}_ff' \
               PROCESS_RES=$MID_RES REF_STRATEGY=first \
               DA3_MODEL_DIR=/mnt/d/wheel/da3_human_ms \
               ~/miniconda3/envs/da3_human/bin/python '$MEDIA_CODE_ROOT/compare_4methods/98_da3_ff_init.py'" ;;
        *)
            echo "⚠️ 未知步骤: $S（可选 is vggt da3 da3gs；qs 权重锁不测）" >&2
            exit 1 ;;
    esac
done

echo ""
if [ -n "$FAILED" ]; then
    echo "⚠️ [90e] 完成但有失败步骤:$FAILED"
else
    echo "🎉 [90e] 全部步骤完成: $STEPS"
fi
echo "ALL_DONE rc=0 failed='${FAILED# }'"
