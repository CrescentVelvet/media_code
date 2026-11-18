#!/usr/bin/env bash
# 90d_run_2k_jiajia.sh — 佳佳抱枕原生 2K（1440×1920 不降采样）五方法批量
#
# 口径对齐 32v 轮（90_run_all_scenes / FINDINGS §2/§8）：同采集包、同 linspace 抽帧
# （N_VIEWS=32）、同 ITERATIONS=30000、InfiniSplat 带默认 3000 步精调。
# 唯一变量 = 各方法输入/训练分辨率：
#   DA3 PROCESS_RES 504→1920 ｜ QuerySplat 512²→1920×1440（vggt_input_size 512→1920）
#   InfiniSplat LONG_SIDE 518→1920 ｜ VGGT-Ω RESOLUTION 512→1920 ｜ 3DGS 回炉 RES 2→1
# 产物隔离：SCENE_NAME=佳佳抱枕_2k，输出全部落 *_2k 目录，不碰 32v/3v 产物。
#
# 注意（2026-11-18）：采集包 HEIC 原生上限就是 1440×1920，不存在更高素材；
# 本实验测的是「不降采样直喂原生」的支持度（显存/耗时/画质），不是上采样伪 2K。
#
# 用法:
#   bash compare_4methods/90d_run_2k_jiajia.sh           # 五方法串行
#   STEPS="qs is" bash compare_4methods/90d_run_2k_jiajia.sh   # 只跑指定步骤
# 日志: ~/output/batch_4methods/2k_<step>_佳佳抱枕_2k.log
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDIA_CODE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export RAW_DIR="/mnt/d/dataset/手部微动样例/01e954de1b8a4cde9315c976f4dcb964-佳佳抱枕"
export SCENE_NAME="佳佳抱枕_2k"
export N_VIEWS=32
export ITERATIONS=30000
export REF_STRATEGY=first   # 与 90a/90b 三图轮一致，DA3 世界系钉视角 0

STEPS="${STEPS:-qs is vggt da3 da3gs}"   # 快的/不确定的先跑；DA3 系最慢压尾
LOG_DIR="$HOME/output/batch_4methods"
mkdir -p "$LOG_DIR"

echo "🚀 [90d] 2K 批量: $SCENE_NAME（N_VIEWS=$N_VIEWS, ITERS=$ITERATIONS, 原生 1440×1920）"
echo "  步骤: $STEPS"

FAILED=""
run() {  # $1=step 名, 其余=命令；失败不中断（"能否支持"里 OOM/超时本身就是结果）
    local STEP="$1"; shift
    local LOG="$LOG_DIR/2k_${STEP}_${SCENE_NAME}.log"
    echo ""
    echo "🚀 [2k][$STEP] 开始  ⏱️  $(date +%H:%M:%S)  (log: $LOG)"
    local T0=$SECONDS
    "$@" > "$LOG" 2>&1
    local RC=$?
    local SEC=$((SECONDS - T0))
    if [ $RC -ne 0 ]; then
        echo "❌ [2k][$STEP] FAILED (exit=$RC, ${SEC}s)，日志: $LOG —— 继续下一家"
        FAILED="$FAILED $STEP(rc=$RC)"
        return
    fi
    echo "✅ [2k][$STEP] 完成  ⏱️  ${SEC}s"
}

for S in $STEPS; do
    case "$S" in
        da3)
            # DA3 前馈 1920（timeout 4h：单次前向 >70min 未完已实测，超时就判不实用）
            # + 3DGS 30k 全分辨率回炉（RES=1，timeout 8h）
            run "$S" bash -c \
              "timeout 14400 env PROCESS_RES=1920 bash '$MEDIA_CODE_ROOT/compare_4methods/90b_run_da3_inference.sh' \
               && timeout 28800 env RES=1 bash '$MEDIA_CODE_ROOT/compare_4methods/90c_train_da3_3dgs.sh'" ;;
        da3gs)
            # DA3 原生高斯头（DA3-GS）：97 全 SH 导出 + 98 相机，均 PROCESS_RES=1920（各 timeout 4h）
            run "$S" bash -c \
              "timeout 14400 env IMAGES_DIR='$HOME/output/da3_results/_captures/$SCENE_NAME/images' \
               OUT_DIR='$HOME/output/da3_gs_results/$SCENE_NAME' \
               PROCESS_RES=1920 REF_STRATEGY=first MAX_VIEWS=0 \
               DA3_MODEL_DIR=/mnt/d/wheel/da3_human_ms \
               ~/miniconda3/envs/da3_human/bin/python '$MEDIA_CODE_ROOT/compare_4methods/97_da3_gs_export_full_sh.py' \
               && timeout 14400 env IMAGES_DIR='$HOME/output/da3_results/_captures/$SCENE_NAME/images' \
               OUT_DIR='$HOME/output/da3_gs_results/${SCENE_NAME}_ff' \
               PROCESS_RES=1920 REF_STRATEGY=first \
               DA3_MODEL_DIR=/mnt/d/wheel/da3_human_ms \
               ~/miniconda3/envs/da3_human/bin/python '$MEDIA_CODE_ROOT/compare_4methods/98_da3_ff_init.py'" ;;
        qs)
            # QuerySplat 2K：复制 yaml 改 img_size/vggt_input_size（02 已支持 QS_CONFIG 覆盖）
            run "$S" bash -c "
               CKPT_DIR=\"\$HOME/repos/QuerySplat/checkpoints\"
               Y2K=\"\$CKPT_DIR/querysplat_vggto_1B_2k_8192.yaml\"
               if [ ! -f \"\$Y2K\" ]; then
                   sed -e 's/img_size: \[512, 512\]/img_size: [1920, 1440]/' \
                       -e 's/vggt_input_size: 512/vggt_input_size: 1920/' \
                       \"\$CKPT_DIR/querysplat_vggto_1B_512_8192.yaml\" > \"\$Y2K\"
                   echo '📝 2K yaml: ' \"\$Y2K\"
               fi
               QS_CONFIG=\"\$Y2K\" timeout 10800 bash '$MEDIA_CODE_ROOT/querysplat/02_run_inference.sh'" ;;
        is)
            # InfiniSplat v2：LONG_SIDE 518→1920（含默认 3000 步精调，timeout 4h）
            run "$S" env OUT_NAME="$SCENE_NAME" LONG_SIDE=1920 \
                 timeout 14400 bash "$MEDIA_CODE_ROOT/infinisplat_v2/07_run_capture.sh" ;;
        vggt)
            # VGGT-Ω 1920（timeout 2h）+ 3DGS 30k 全分辨率回炉（RES=1，timeout 8h）
            run "$S" bash -c \
              "timeout 7200 env RESOLUTION=1920 bash '$MEDIA_CODE_ROOT/vggt_omega/02_run_inference.sh' \
               && timeout 28800 env RES=1 bash '$MEDIA_CODE_ROOT/vggt_omega/03_train_3dgs.sh'" ;;
        *)
            echo "⚠️ 未知步骤: $S（可选 da3 da3gs qs is vggt）" >&2
            exit 1 ;;
    esac
done

echo ""
if [ -n "$FAILED" ]; then
    echo "⚠️ [90d] 完成但有失败步骤:$FAILED"
else
    echo "🎉 [90d] 全部步骤完成: $STEPS"
fi
echo "ALL_DONE rc=0 failed='${FAILED# }'"
