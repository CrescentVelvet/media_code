#!/usr/bin/env bash
# 09_eval_bench_exports.sh — 评估 08 导出的四档 PLY（official / full_sh / noclip / max）
# 用法: wsl -d Ubuntu2404 -- bash /mnt/c/code/media_code/da3_human/09_eval_bench_exports.sh [BASE]
#
# ⚠️ 必须 conda activate：直接调 envs/vggt_human/bin/python 时 nvcc 不在 PATH，
#    gsplat 会打印 "No CUDA toolkit found. gsplat will be disabled."，
#    随后 _C=None → AttributeError: 'NoneType' object has no attribute 'CameraModelType'。
set +u   # conda 的 activate.d 脚本里有未定义变量（SYS_SYSROOT），set -u 会让它直接退出
BASE="${1:-/home/velvet/output/da3_human_results/bench_504_32v}"
SCRIPT="/mnt/c/code/media_code/da3_human/07_eval_da3_gs.py"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate vggt_human || { echo "❌ conda activate vggt_human 失败"; exit 1; }
if ! python -c "from gsplat.cuda._backend import _C; assert _C is not None" 2>/dev/null; then
    echo "❌ gsplat CUDA 后端未加载（nvcc=$(command -v nvcc || echo 缺失)）"
    exit 1
fi

for T in official full_sh noclip max; do
    P="$BASE/gs_$T/0000.ply"
    if [ ! -f "$P" ]; then
        echo "⏭️  skip $T（无 $P）"
        continue
    fi
    echo "=============== $T ==============="
    PLY="$P" NPZ="$BASE/cameras.npz" SF=1.0 python "$SCRIPT" eval 2>&1 | tail -20
done
