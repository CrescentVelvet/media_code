#!/usr/bin/env bash
# 01b_prepare_jiajia.sh — 佳佳抱枕 Remy HEIC 包 → ZipSplat PNG 输入
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

RAW_DIR="${1:-${RAW_DIR:-}}"
RAW_DIR="${RAW_DIR:?❌ 指定 RAW_DIR=/mnt/d/dataset/手部微动样例/<佳佳抱枕包>，或作为第一个参数传入}"
CAP_DIR="${CAP_DIR:-$RESULTS_DIR/_captures/jiajia_baozhen}"
N_VIEWS="${N_VIEWS:-16}"
FRAMES="${FRAMES:-}"
CONVERT_PYTHON="${CONVERT_PYTHON:-$HOME/miniconda3/envs/infinisplat_v2/bin/python}"

[ -d "$RAW_DIR" ] || { echo "❌ 数据包不存在: $RAW_DIR" >&2; exit 1; }
[ -f "$RAW_DIR/transforms.json" ] || { echo "❌ 缺 transforms.json" >&2; exit 1; }
[ -x "$CONVERT_PYTHON" ] || CONVERT_PYTHON=python
ARGS=(--raw-dir "$RAW_DIR" --out-dir "$CAP_DIR")
if [ -n "$FRAMES" ]; then ARGS+=(--frames "$FRAMES"); else ARGS+=(--n-views "$N_VIEWS"); fi
if [ "${REUSE_CONVERTED:-1}" = "1" ] && [ -f "$CAP_DIR/meta.json" ] && \
    python3 - "$CAP_DIR/meta.json" "$RAW_DIR" "$N_VIEWS" "$FRAMES" <<'PY'
import json, sys
m = json.load(open(sys.argv[1], encoding="utf-8"))
ok = m.get("source") == sys.argv[2]
if sys.argv[4]:
    ok = ok and [str(i) for i in m.get("picked_idx", [])] == sys.argv[4].split(",")
else:
    ok = ok and m.get("n_views") == int(sys.argv[3])
raise SystemExit(0 if ok else 1)
PY
then
    echo "⏭️ 转换配置一致，复用: $CAP_DIR"
else
    "$CONVERT_PYTHON" "$MEDIA_CODE_ROOT/infinisplat_v2/07_capture_to_npz.py" "${ARGS[@]}" || exit 1
fi
echo "🎉 PNG 输入: $CAP_DIR/images（GT 相机仅供独立对齐评测，不喂 ZipSplat）"
