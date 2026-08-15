#!/usr/bin/env bash
# run_all.sh — 一键: 验证 env -> 下权重 -> SMPL 提取 -> 渲染深度 -> Flux 多视角生成。
# 05 重建 / 06 评估为待办 (见 README)。flux_human 无官方代码仓 (纯 diffusers)。
#
# ⚠️ 前置: 需先建 sam_3d_body env + 下其权重 (02 跨 env 调用 sam_3d_body)。
#    conda env list | grep sam_3d_body  或  ls ../../model/sam-3d-body/
#
# 用法: VIDEO=../data/subject.mp4 HF_TOKEN=hf_xxx GPU=0 bash flux_human/run_all.sh
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== [run_all] flux_human 一键流程 (env: ${CONDA_ENV:-flux_human}) ==="
echo "    输入视频: ${VIDEO:-❌ 未设 (02 需要 VIDEO 或 FRAMES_DIR)}"
echo "    HF_TOKEN: ${HF_TOKEN:+已设}${HF_TOKEN:-❌ 未设 (01 下 FLUX.1-dev gated 需要)}"
echo "    GPU:      ${GPU:-0}"

# 前置检查: sam_3d_body env (02 跨 env 调用)
if ! conda env list 2>/dev/null | grep -qE "(^| )sam_3d_body( |$)"; then
    echo "⚠️ WARNING: conda env 'sam_3d_body' 未建. 02 会失败." >&2
    echo "   先跑: INSTALL_DEPS=1 bash sam_3d_body/00_setup_env.sh" >&2
    echo "         HF_TOKEN=hf_xxx bash sam_3d_body/01_download_models.sh" >&2
fi

if [ "${INSTALL_DEPS:-1}" = "1" ]; then
    INSTALL_DEPS=1 bash "$SCRIPT_DIR/00_setup_env.sh"
else
    bash "$SCRIPT_DIR/00_setup_env.sh"
fi
bash "$SCRIPT_DIR/01_download_models.sh"
VIDEO="${VIDEO:-}" GPU="${GPU:-0}" bash "$SCRIPT_DIR/02_extract_smpl.sh"
GPU="${GPU:-0}" bash "$SCRIPT_DIR/03_render_depth.sh"
GPU="${GPU:-0}" bash "$SCRIPT_DIR/04_generate_views.sh"

echo "=== [run_all] 00-04 完成. ==="
echo "    生成结果: ../flux_human_results/views/view_*.png"
echo "    待办 (见 README):"
echo "      05_reconstruct (3DGS/NeuS 重建 — 用 view_*.png + cameras.npz known pose)"
echo "      06_evaluate (Chamfer/F-score 对比 PIFuHD/GT)"
