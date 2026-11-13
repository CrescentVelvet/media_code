#!/bin/bash
#SBATCH -J env_verify
#SBATCH -p h100a011
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH -o /data/00950754/logs/env_verify_%j.out

# 不要用 set -u：conda 的 activate.d 钩子引用未定义变量
# （activate-gcc_linux-64.sh: SYS_SYSROOT），在 set -u 下直接崩。
set -eo pipefail
echo "job=$SLURM_JOB_ID node=$(hostname) start=$(date -Is)"

SCRIPT_DIR="${VERIFY_SCRIPT_DIR:-${SLURM_SUBMIT_DIR:-/data/00950754/code/media_code/canada_h100}}"
echo "脚本目录: $SCRIPT_DIR"

source /data/00950754/envs/activate_splatanything.sh

echo "python : $(which python) — $(python -V 2>&1)"
echo "nvcc   : $(which nvcc 2>/dev/null || echo 'NOT FOUND')"
echo "gcc    : $(gcc --version 2>/dev/null | head -1 || echo 'NOT FOUND')"
echo "glm    : $(ls -d /data/00950754/repos/glm_src/glm 2>/dev/null || echo 'NOT FOUND')"
echo "arch   : ${TORCH_CUDA_ARCH_LIST:-auto}"

cd "$SCRIPT_DIR"
python 01_verify_env.py
echo "end=$(date -Is)"
