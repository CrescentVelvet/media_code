#!/usr/bin/env bash
# 90b_run_3view_da3gs.sh — 三图轮的 DA3 原生高斯头（DA3-GS）全 SH 导出
#
# 口径对齐 32v 轮正式口径（FINDINGS：官方等价导出、不裁剪 + 全 SH，对应 97 的 gs_ply_full）。
# 必须与 90a 的 da3 步骤同 REF_STRATEGY=first（DA3 世界系由参考视角定义，钉死视角 0），
# 这样本脚本的 PLY 与 da3_results/佳佳抱枕_3v 的位姿同坐标系。
# 前置：90a 的转换产物（_captures/佳佳抱枕_3v/images，3 张）；与 90a 串行执行（GPU）。
#
# 用法: bash compare_4methods/90b_run_3view_da3gs.sh
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDIA_CODE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export IMAGES_DIR="$HOME/output/da3_results/_captures/佳佳抱枕_3v/images"
export OUT_DIR="$HOME/output/da3_gs_results/佳佳抱枕_3v"
export PROCESS_RES=504
export MAX_VIEWS=0
export REF_STRATEGY=first          # 与 90a 的 da3 步骤一致（可比对齐）

if [ ! -d "$IMAGES_DIR" ] || [ "$(ls "$IMAGES_DIR" | wc -l)" -ne 3 ]; then
    echo "❌ ERROR: 三图输入未就绪（需要 90a 先完成 is/da3 任一转换）: $IMAGES_DIR" >&2
    exit 1
fi

echo "🚀 [90b] DA3-GS 全 SH 导出（三图）: $IMAGES_DIR → $OUT_DIR"
DA3_MODEL_DIR="${DA3_MODEL_DIR:-/mnt/d/wheel/da3_human_ms}" \
    ~/miniconda3/envs/da3_human/bin/python "$MEDIA_CODE_ROOT/compare_4methods/97_da3_gs_export_full_sh.py"
RC=$?
if [ $RC -ne 0 ]; then
    echo "❌ 97 导出失败 (exit $RC)" >&2
    exit 1
fi

# 98：同 REF_STRATEGY=first 再跑一次前馈，专取 cameras.npz（w2c 米制）。
# 97 不产 cameras.npz，而评测对齐需要「与 gs_ply_full 同世界系」的相机；
# 同输入同权重前馈是确定性的，两次运行共享同一世界系（first = 都钉视角 0）。
FF_DIR="$HOME/output/da3_gs_results/佳佳抱枕_3v_ff"
echo "📷 [90b] 98 前馈取 cameras.npz → $FF_DIR"
DA3_MODEL_DIR="${DA3_MODEL_DIR:-/mnt/d/wheel/da3_human_ms}" \
IMAGES_DIR="$IMAGES_DIR" OUT_DIR="$FF_DIR" \
PROCESS_RES=504 REF_STRATEGY=first \
    ~/miniconda3/envs/da3_human/bin/python "$MEDIA_CODE_ROOT/compare_4methods/98_da3_ff_init.py"
RC=$?
if [ $RC -ne 0 ]; then
    echo "❌ 98 前馈失败 (exit $RC)" >&2
    exit 1
fi
echo "🎉 Done. 全 SH PLY: $OUT_DIR/gs_ply_full/0000.ply ｜ 相机: $FF_DIR/cameras.npz"
