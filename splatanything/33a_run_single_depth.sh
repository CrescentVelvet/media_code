#!/usr/bin/env bash
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"
DP_PY="${DP_PY:-$(dirname "$CONDA_PREFIX")/sharp/bin/python}"
export DEPTH_PRO_2_DIR="${DEPTH_PRO_2_DIR:-$HOME/repos/ml-depth-pro-2}"
export CKPT_PATH="${CKPT_PATH:?Set the DP2 H+ checkpoint}"
export FOV_CKPT_PATH="${FOV_CKPT_PATH:?Set the DP2 FOV checkpoint}"
export OFFICIAL_REVISION="$(git -C "$DEPTH_PRO_2_DIR" rev-parse HEAD)"
for name in rosemary daven_holdout; do
    export INPUT_DIR="${OUT:?Set results directory}/$name/dp_input"
    export OUT_DIR="$INPUT_DIR/single_frame"
    "$DP_PY" "$SCRIPT_DIR/33_prepare_single_depth.py" "${DEPTH:?Set 30 diagnostics root}/$name/dp_input" "$INPUT_DIR" || exit 1
    "$DP_PY" "$SCRIPT_DIR/../depth_pro_2/run_single_frame_inference.py" || exit 1
done
