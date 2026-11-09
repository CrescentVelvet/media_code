#!/usr/bin/env bash
# audit_30_native_ghost.py 的封装（wsl.exe 会吞掉内联命令里的 $VAR，带变量一律走脚本）
set -eo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate querysplat
exec python "$(dirname "$0")/audit_30_native_ghost.py" "$@"
