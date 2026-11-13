#!/usr/bin/env bash
# 09_run_train_probe.sh — WB-4 / R3：量 SplatAnything 的训练改造成本。
#
#   bash 09_run_train_probe.sh                          # 全套（fwd+step1+100步+存恢复）
#   bash 09_run_train_probe.sh --stages fwd             # 只做等价前向对照
#   bash 09_run_train_probe.sh --stages sweep --sweep-lr 1e-3,1e-4,1e-5
#
# 默认 3v / 504²（14 的倍数）/ rgb 模式 / lr 1e-5，与主计划 WB-4 第 4 条一致。
# 输出：$SPLATANYTHING_RESULTS_DIR/train_probe.json（+ train_probe_ckpt.pt）
#
# Env：
#   SCENE=   场景 images 目录，默认 jiajia_381（帧 0/40/81）
#   RES=504  长边，必须是 14 的倍数
#   POSE=0   1 = pose 条件模式（需要 prior_camera.json）
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_gsplat_env.sh"

: "${SCENE:=$SPLATANYTHING_RESULTS_DIR/_scenes/jiajia_381/images}"
: "${RES:=504}"
ARGS=("--scene" "$SCENE" --res "$RES" --out "$SPLATANYTHING_RESULTS_DIR/train_probe.json")
[ "${POSE:-0}" = "1" ] && ARGS+=("--pose-mode")

exec python "$SCRIPT_DIR/09_train_probe.py" "${ARGS[@]}" "$@"
