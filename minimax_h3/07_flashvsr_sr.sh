#!/usr/bin/env bash
# 07_flashvsr_sr.sh — FlashVSR 视频超分（把 int8 量化 MiniMax 生成的低分辨率视频 4× 超分到高清）。
#
# 用 OpenImagingLab/FlashVSR（CVPR 2026，one-step diffusion streaming VSR）。
# 默认 v1.1 + full pipeline（最高质量，MiniMax 短视频 ~5s 够用）；
# PIPELINE=tiny_long 切到 tiny + 长视频流式管线（低显存 / 长视频用，需 TCDecoder.ckpt）。
#
# FlashVSR 复用 minimax_h3 env（不建独立 env）：不 pin torch/transformers/numpy
# （用 env 现有版本——torch 是 cu124 即可、transformers 5.x 也能跑，因 VSR 运行时
# 加载预计算 prompt tensor，不实例化 transformers 模型；diffsynth prompters 只用 AutoTokenizer）。
# 只装 diffsynth（--no-deps，避免其 requirements 的 torch 2.6/transformers 4.46.2 降级 env）
# + Block-Sparse-Attention（编译绑定当前 torch）+ 版本无关运行时依赖。GPU 选卡用 GPU=N。
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 复用当前 env（默认沿用 _env.sh 的 CONDA_DEFAULT_ENV，即 minimax_h3）。不强制切 env。
source "$SCRIPT_DIR/_env.sh"

# 路径（都用 ${VAR:-default} 允许外部覆盖）
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../../model}"
FLASHVSR_DIR="${FLASHVSR_DIR:-$REPO_DIR/../FlashVSR}"           # 官方仓（diffsynth + examples/WanVSR/utils）
BSA_DIR="${BSA_DIR:-$REPO_DIR/../Block-Sparse-Attention}"       # Block-Sparse-Attention 源码
FLASHVSR_MODEL_DIR="${FLASHVSR_MODEL_DIR:-$MODEL_DIR/FlashVSR}" # 权重（用户已下）
RESULTS_DIR="${RESULTS_DIR:-$REPO_DIR/../MiniMax-H3/results_sr}"
export MODEL_DIR FLASHVSR_DIR BSA_DIR FLASHVSR_MODEL_DIR RESULTS_DIR

PIP_FLAGS=(--trusted-host pypi.org --trusted-host pypi.python.org \
    --trusted-host files.pythonhosted.org --timeout 600 --retries 10)

# ── 1. clone 官方仓（diffsynth 包 + examples/WanVSR/utils） ──
if [ ! -d "$FLASHVSR_DIR/diffsynth" ]; then
    echo "📦 cloning FlashVSR repo -> $FLASHVSR_DIR"
    mkdir -p "$(dirname "$FLASHVSR_DIR")"
    LD_LIBRARY_PATH= git clone --depth 1 https://github.com/OpenImagingLab/FlashVSR.git "$FLASHVSR_DIR" || \
        LD_LIBRARY_PATH= git -c http.sslVerify=false clone --depth 1 https://github.com/OpenImagingLab/FlashVSR.git "$FLASHVSR_DIR"
fi

# ── 2. clone Block-Sparse-Attention（LCSA 必需） ──
if [ ! -d "$BSA_DIR/block_sparse_attn" ]; then
    echo "📦 cloning Block-Sparse-Attention -> $BSA_DIR"
    mkdir -p "$(dirname "$BSA_DIR")"
    LD_LIBRARY_PATH= git clone --depth 1 https://github.com/mit-han-lab/Block-Sparse-Attention.git "$BSA_DIR" || \
        LD_LIBRARY_PATH= git -c http.sslVerify=false clone --depth 1 https://github.com/mit-han-lab/Block-Sparse-Attention.git "$BSA_DIR"
fi

# ── 3. 依赖检测 ──
need_install=0
if ! python -c "from diffsynth import ModelManager" 2>/dev/null; then
    need_install=1
fi
if ! python -c "import block_sparse_attn" 2>/dev/null; then
    need_install=1
fi
if ! python -c "import imageio, einops, tqdm, PIL" 2>/dev/null; then
    need_install=1
fi

# ── 4. 装依赖（INSTALL_DEPS=1 或检测到缺失时） ──
# 默认复用当前 env 的 torch/transformers/numpy（不 pin 版本）。
# INSTALL_DEPS_USE_PINS=1（独立 env 兜底用）：装 FlashVSR 官方 requirements.txt（pin torch 2.6
# /transformers 4.46.2/numpy 1.26.4）——只在独立 flashvsr env 里用，别在 minimax_h3 env 用。
if [ "${INSTALL_DEPS:-0}" = "1" ] || [ "$need_install" = "1" ]; then
    # Block-Sparse-Attention（先装 packaging/ninja，再 setup.py install；编译吃内存）
    python -m pip install "${PIP_FLAGS[@]}" packaging ninja || true
    if [ -f "$BSA_DIR/setup.py" ]; then
        ( cd "$BSA_DIR" && python setup.py install ) || \
            { echo "❌ Block-Sparse-Attention build failed (needs nvcc + sufficient RAM)" >&2; exit 1; }
    else
        echo "❌ BSA setup.py missing at $BSA_DIR" >&2; exit 1
    fi
    if [ "${INSTALL_DEPS_USE_PINS:-0}" = "1" ]; then
        echo "📦 [pins mode] installing FlashVSR full requirements.txt (torch 2.6/transformers 4.46.2/numpy 1.26.4)..."
        python -m pip install "${PIP_FLAGS[@]}" -e "$FLASHVSR_DIR" || \
            { echo "❌ pip install -e FlashVSR failed" >&2; exit 1; }
        python -m pip install "${PIP_FLAGS[@]}" -r "$FLASHVSR_DIR/requirements.txt" || \
            echo "⚠️ requirements.txt install partially failed, continuing..."
        python -m pip install "${PIP_FLAGS[@]}" --force-reinstall --no-deps numpy==1.26.4 || true
    else
        echo "📦 installing FlashVSR deps into env '$CONDA_ENV' (reusing its torch/transformers/numpy)..."
        # diffsynth 包本体：--no-deps 跳过 requirements.txt 的 pin（会降级 env 的 transformers 5.12.1/numpy）。
        python -m pip install "${PIP_FLAGS[@]}" -e "$FLASHVSR_DIR" --no-deps || \
            { echo "❌ pip install -e FlashVSR (--no-deps) failed" >&2; exit 1; }
        # 版本无关运行时依赖（不与 env 冲突；env 已有的会被跳过）
        python -m pip install "${PIP_FLAGS[@]}" einops imageio imageio-ffmpeg \
            opencv-python-headless tqdm safetensors pillow sentencepiece ftfy accelerate || \
            echo "⚠️ some runtime deps install failed, continuing..."
    fi
fi

python -c "from diffsynth import ModelManager; import block_sparse_attn; print('✅ diffsynth + block_sparse_attn ok')" 2>/dev/null || \
    { echo "❌ deps not ready (或 diffsynth 与本 env 的 transformers 不兼容)." >&2
      echo "   重试:  INSTALL_DEPS=1 bash minimax_h3/07_flashvsr_sr.sh" >&2
      echo "   仍失败(diffsynth 在 transformers 5.x 上 import 炸): 单独建 env →" >&2
      echo "     conda create -n flashvsr python=3.11 -y && conda activate flashvsr" >&2
      echo "     pip install torch --index-url https://download.pytorch.org/whl/cu124" >&2
      echo "     CONDA_ENV=flashvsr INSTALL_DEPS_USE_PINS=1 bash minimax_h3/07_flashvsr_sr.sh" >&2
      exit 1; }

# ── 5. 权重文件检查 ──
PIPELINE="${PIPELINE:-full}"
DIT="diffusion_pytorch_model_streaming_dmd.safetensors"
LQ="LQ_proj_in.ckpt"
VAE="Wan2.1_VAE.pth"
TCD="TCDecoder.ckpt"
need=("$DIT" "$LQ")
if [ "$PIPELINE" = "full" ]; then
    need+=("$VAE")
elif [ "$PIPELINE" = "tiny_long" ]; then
    need+=("$TCD")
else
    echo "❌ unknown PIPELINE=$PIPELINE (full | tiny_long)" >&2; exit 1
fi
miss=()
for f in "${need[@]}"; do
    [ -f "$FLASHVSR_MODEL_DIR/$f" ] || miss+=("$f")
done
if [ ${#miss[@]} -gt 0 ]; then
    echo "❌ missing weights in $FLASHVSR_MODEL_DIR: ${miss[*]}" >&2
    echo "   从 https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1 下载放到 $FLASHVSR_MODEL_DIR/" >&2
    exit 1
fi

# ── 6. 参数 + 前置检查 ──
INPUT="${INPUT:-}"
if [ -z "$INPUT" ]; then
    echo "❌ INPUT not set (video path or frame directory)" >&2
    echo "   e.g. INPUT=../MiniMax-H3/results_int8/rotate_360.mp4 bash $0" >&2
    exit 1
fi
export INPUT PIPELINE

echo "🚀 [07] FlashVSR 4× SR ($PIPELINE pipeline)"
echo "  🖼️ input:  $INPUT"
echo "  🏋️ model:  $FLASHVSR_MODEL_DIR"
echo "  📐 scale:  ${SCALE:-4}×  pipeline: $PIPELINE"
echo "  💾 output: $RESULTS_DIR/${OUTPUT_NAME:-<input>_sr.mp4}"

python "$SCRIPT_DIR/07_flashvsr_sr.py"
if [ $? -ne 0 ]; then
    echo "❌ FAILED" >&2; exit 1
fi
echo "🎉 [07] Done."
