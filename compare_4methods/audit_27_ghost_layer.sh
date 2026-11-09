#!/usr/bin/env bash
# audit_27_ghost_layer.py 的封装（wsl.exe 会吞掉内联命令里的 $VAR，带变量一律走脚本）
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate querysplat
exec python "$(dirname "$0")/audit_27_ghost_layer.py" "$@"
