#!/usr/bin/env bash
# Replay the camera experiment first, then run the independent depth diagnostic.
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:?Set ROOT to the frozen protocol_v5 directory}"
OUT="${OUT:?Set OUT to the experiment output directory}"
STAGE="${STAGE:-camera}"
export CONDA_ENV=splatanything
source "$SCRIPT_DIR/_env.sh"
source "$SCRIPT_DIR/_gsplat_env.sh"
SA_PY="$CONDA_PREFIX/bin/python"
ENV_ROOT="$(dirname "$CONDA_PREFIX")"
export PYTHONPATH="$SPLATANYTHING_DIR:${PYTHONPATH:-}"
mapfile -t names < <("$SA_PY" -c 'import json,sys; print("\n".join(p["name"] for p in json.load(open(sys.argv[1]))["packages"]))' "$ROOT/manifest.json")
[ "${#names[@]}" -gt 0 ] || { echo "❌ Empty protocol" >&2; exit 1; }

case "$STAGE" in
camera)
    [ ! -e "$OUT" ] || { echo "❌ Use a fresh OUT for a complete camera replay" >&2; exit 1; }
    for name in "${names[@]}"; do
        dest="$OUT/$name"
        mkdir -p "$dest/pi3x" || exit 1
        echo "📷 Three-source pose: $name"
        IMAGES_DIR="$ROOT/scenes/${name}_source3/images" SAVE_DIR="$dest/pi3x" \
            PI3_DIR="${PI3_DIR:-$HOME/repos/Pi3}" PI3X_CKPT="${PI3X_CKPT:-$HOME/model/pi3x/model.safetensors}" \
            CONDITIONS_NPZ="" "$ENV_ROOT/pi3x/bin/python" "$SCRIPT_DIR/../pi3x/pi3x_recon.py" || exit 1
        "$SA_PY" "$SCRIPT_DIR/29_camera_probe.py" prepare "$ROOT" "$OUT" "$name" || exit 1
        for arm in SA Pi3X; do
            INPUT="$dest/prior_$arm" MODE=pose RESOLUTION="756 756" OUT_ROOT="$dest/models" \
                bash "$SCRIPT_DIR/02_run_inference.sh" || exit 1
        done
        "$SA_PY" "$SCRIPT_DIR/29_camera_probe.py" heldout "$ROOT" "$OUT" "$name" || exit 1
    done
    ;;
nose)
    "$SA_PY" "$SCRIPT_DIR/30_nose_probe.py" prepare "$ROOT" "$OUT" || exit 1
    for name in "${names[@]}"; do
        dest="$OUT/$name"
        [ -f "$dest/heldout.json" ] || { echo "❌ Missing camera experiment for $name" >&2; exit 1; }
        [ ! -e "$dest/dp2/inference.json" ] || { echo "❌ Existing DP2 result for $name" >&2; exit 1; }
        echo "📐 Joint three-frame depth: $name"
        INPUT_DIR="$dest/dp_input" OUT_DIR="$dest/dp2" \
            DEPTH_PRO_2_DIR="${DEPTH_PRO_2_DIR:-$HOME/repos/ml-depth-pro-2}" \
            CKPT_PATH="${CKPT_PATH:?Set the DP2 H+ checkpoint}" \
            FOV_CKPT_PATH="${FOV_CKPT_PATH:?Set the DP2 FOV checkpoint}" \
            "$ENV_ROOT/sharp/bin/python" "$SCRIPT_DIR/../depth_pro_2/run_inference.py" || exit 1
    done
    for action in score real; do
        "$SA_PY" "$SCRIPT_DIR/30_nose_probe.py" "$action" "$ROOT" "$OUT" || exit 1
    done
    ;;
*) echo "❌ STAGE must be camera or nose" >&2; exit 1 ;;
esac
echo "✅ $STAGE complete: $OUT"
