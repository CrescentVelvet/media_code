#!/usr/bin/env bash
# Install the official package into the existing Python 3.13 / CUDA PyTorch env.
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DEPTH_PRO_2_DIR="${DEPTH_PRO_2_DIR:-$HOME/repos/ml-depth-pro-2}"
export MODEL_DIR="${MODEL_DIR:-$HOME/model/depth_pro_2}"
WEIGHTS_DIR="${WEIGHTS_DIR:-/mnt/d/wheel/depth_pro_2_ms}"
export RESULTS_DIR="${RESULTS_DIR:-$HOME/output/depth_pro_2_results}"
source "$SCRIPT_DIR/_env.sh"
PROXY_ENV="$REPO_DIR/proxy.env"
if ! grep -q "DEPTH_PRO_2_DIR=" "$PROXY_ENV" 2>/dev/null; then
    cat >> "$PROXY_ENV" <<'ENVEOF'

# Depth Pro 2 local paths (this block is only read by depth_pro_2/_env.sh).
export DEPTH_PRO_2_DIR="$HOME/repos/ml-depth-pro-2"
export MODEL_DIR="$HOME/model/depth_pro_2"
export RESULTS_DIR="$HOME/output/depth_pro_2_results"
ENVEOF
    source "$SCRIPT_DIR/_env.sh"
fi
WHEELS_DIR="${WHEELS_DIR:-/mnt/d/wheel}"
PIP_FLAGS=(-i "${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}" --find-links "$WHEELS_DIR" --timeout 600 --retries 5)
mkdir -p "$(dirname "$DEPTH_PRO_2_DIR")" "$MODEL_DIR" "$RESULTS_DIR"
if [ ! -d "$DEPTH_PRO_2_DIR/.git" ]; then
    LD_LIBRARY_PATH= git clone --depth 1 --filter=blob:none --single-branch "$DEPTH_PRO_2_REPO" "$DEPTH_PRO_2_DIR" || exit 1
fi
python -c "import sys; assert sys.version_info >= (3, 11); import torch; assert torch.__version__ >= '2.7'; print('✅', sys.version, torch.__version__)" || exit 1
pip install "${PIP_FLAGS[@]}" "timm>=1.0.27,<1.1" "av>=14.1" opencv-python-headless scipy plyfile pillow-heif || exit 1
pip install --no-deps -e "$DEPTH_PRO_2_DIR" || exit 1
python -c "import depth_pro_2, cv2, av, timm; print('✅ Depth Pro 2:', depth_pro_2.__file__); print('✅ timm:', timm.__version__); print('✅ cv2:', cv2.__version__)" || exit 1
if [ -f "$WEIGHTS_DIR/depthpro2-h-eivtm5w7m5.pt" ]; then
    ln -sfn "$WEIGHTS_DIR/depthpro2-h-eivtm5w7m5.pt" "$CKPT_PATH"
elif [ -f "/mnt/d/迅雷下载/depthpro2-h-eivtm5w7m5.pt" ]; then
    ln -sfn "/mnt/d/迅雷下载/depthpro2-h-eivtm5w7m5.pt" "$CKPT_PATH"
fi
if [ -f "$WEIGHTS_DIR/depthpro2-fov-ew55zg4kq7.pt" ]; then
    ln -sfn "$WEIGHTS_DIR/depthpro2-fov-ew55zg4kq7.pt" "$FOV_CKPT_PATH"
elif [ ! -f "$FOV_CKPT_PATH" ]; then
    curl -L --fail --max-time 300 "https://ml-site.cdn-apple.com/models/depth-pro-2/depthpro2-fov-ew55zg4kq7.pt" -o "$FOV_CKPT_PATH" || exit 1
fi
echo "🎉 Environment ready: $CONDA_ENV"
