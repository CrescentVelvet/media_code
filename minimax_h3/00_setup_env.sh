#!/usr/bin/env bash
# 00_setup_env.sh — activate conda env & install SGLang (with diffusion support).
#
# MiniMax-H3 本地复现走 SGLang Diffusion 路径（官方推荐的三种推理框架之一）。
# SGLang 自带 MiniMax-H3 的模型实现 + 扩散运行时，所以只需要：
#   1) 一个带 CUDA torch 的 conda env；
#   2) `pip install "sglang[all]"`（含 diffusion 支持，近版本已把 MiniMax-H3
#      的 transformer/vae/scheduler 打包进去）。
#
# Set INSTALL_DEPS=1 to install SGLang (first time).
#
# SGLang pulls its own torch + flashinfer + CUDA kernels; its version pins may
# CONFLICT with other algos in this repo (hunyuanvideo wants diffusers 0.35,
# hypir pins diffusers 0.32 / transformers 4.49). Use a DEDICATED env:
#   conda create -n minimax_h3 --clone doll -y   # 克隆现有 doll，本地复制不走 conda 通道，绕开 TUNA 镜像 403 + 代理 SSL
#   conda activate minimax_h3
#   CONDA_ENV=minimax_h3 INSTALL_DEPS=1 bash minimax_h3/00_setup_env.sh
# doll 不在时新建见 README「可能遇到的问题」第 2 条（--override-channels + conda config ssl_verify）。
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

echo "🚀 [00] Verify torch in conda env '$CONDA_ENV'"
python - <<'PY'
import sys
try:
    import torch
    print(f"torch: {torch.__version__}  cuda: {torch.version.cuda}  available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit("ERROR: torch.cuda not available — install a CUDA-enabled torch or check GPU visibility.")
except ModuleNotFoundError:
    raise SystemExit("ERROR: torch not installed in this env. Install with: pip install torch --index-url https://download.pytorch.org/whl/cu124")
print(f"python: {sys.version.split()[0]}")
PY

# Install SGLang on demand. sglang[all] bundles the SRT serving runtime + the
# diffusion extras (diffusers/transformers/einops...) that MiniMax-H3 needs.
# If your sglang build splits diffusion into a separate extra and [all] didn't
# bring it, additionally run: pip install "sglang[diffusion]"
# PIP_FLAGS 提到块外，自检段（git fallback）也能用。
PIP_FLAGS=(--trusted-host pypi.org --trusted-host pypi.python.org \
    --trusted-host files.pythonhosted.org --timeout 600 --retries 10)
if [ "${INSTALL_DEPS:-0}" = "1" ]; then
    # 升级 g++ 到 12（支持 C++20 <concepts>）：sglang JIT 融合 QKNorm+RoPE kernel 需要。
    # 不升级也能跑（fallback 到非融合 Python 实现，略慢）。conda install 走 conda 通道
    # 可能 403/SSL，失败不中断。pin python=3.10 避免 gxx 把 CPython 降级成 GraalPy。
    if ! echo '#include <concepts>' | g++ -std=c++20 -x c++ -fsyntax-only - 2>/dev/null; then
        echo "📦 g++ too old (no C++20 <concepts>), upgrading gxx_linux-64=12 (optional, for JIT kernel fusion) ---"
        conda install -y -c conda-forge gxx_linux-64=12 python=3.10 || \
            echo "⚠️ g++ upgrade failed (non-fatal: JIT kernel fallback to slower Python impl)" >&2
    else
        echo "✅ g++ supports C++20 (JIT kernel fusion ok)"
    fi
    echo "📦 installing SGLang (pip install -U \"sglang[all]\") ---"
    python -m pip install --upgrade pip "${PIP_FLAGS[@]}"
    # -U 强制升级：SGLang Diffusion（--model-variant / --performance-mode 等参数）
    # 是 2025/11 后新增，旧版 sglang 不认识这些参数会报 unrecognized arguments。
    python -m pip install "${PIP_FLAGS[@]}" -U "sglang[all]"
    # Diffusion extra is explicit in the sglang docker image
    # (python -m pip install -e ...python[diffusion]); install it too as a
    # no-op-ish safety net in case [all] didn't pin the diffusion deps.
    python -m pip install "${PIP_FLAGS[@]}" -U "sglang[diffusion]" || \
        echo "    (sglang[diffusion] extra not separately installable — assume [all] covers it)"
    # sglang git main pin transformers==5.12.1（2026/08 最新），需要配套升级。
    # transformers 5.x 重命名了 PreTrainedConfig → PretrainedConfig，sglang git main
    # 已适配新名；如果降级到 4.x 会报 `cannot import name 'PreTrainedConfig'`。
    # HybridCache 在 5.x 中已移除（用 DynamicCache 替代），sglang 内部不依赖它。
    echo "📦 aligning diffusers/peft/transformers ---"
    python -m pip install "${PIP_FLAGS[@]}" -U diffusers peft "transformers==5.12.1"
    # huggingface_hub 版本须与 transformers 5.x 兼容（旧版缺 is_offline_mode 等 API）。
    python -m pip install "${PIP_FLAGS[@]}" -U huggingface_hub
    echo "📦 installed. Verify with: python -c 'import sglang; print(\"ok\")'"
fi

echo "🔍 [00] Checking SGLang availability ==="
if python -c "import sglang" 2>/dev/null; then
    python - <<'PY'
import sglang
print(f"sglang: {getattr(sglang, '__version__', 'unknown')}")
PY
else
    echo "⚠️ WARNING: sglang not importable in env '$CONDA_ENV'. Run INSTALL_DEPS=1 bash minimax_h3/00_setup_env.sh." >&2
fi

# 自检 sglang serve 是否支持 MiniMax-H3 的 diffusion 参数（--model-variant / --performance-mode）。
# SGLang Diffusion（2025/11 后新增）的 --model-variant 等参数，PyPI 的 sglang[all]
# 可能不带（cookbook 的 docker 命令是从源码 pip install -e ".../python[diffusion]"）。
# 自检失败则从 git clone sglang 源码 + editable 安装 [diffusion] extra。
echo "🔍 [00] Checking SGLang Diffusion args (--model-variant) ==="
if sglang serve --help 2>&1 | grep -q -- '--model-variant'; then
    echo "✅ sglang serve supports --model-variant (Diffusion ok)"
else
    echo "⚠️ PyPI sglang 不带 --model-variant — 从 git 源码装 [diffusion] extra (cookbook 做法) ---" >&2
    SGLANG_SRC="${SGLANG_SRC:-/tmp/sglang-src}"
    if [ ! -d "$SGLANG_SRC/python" ]; then
        echo "📦 cloning sglang repo -> $SGLANG_SRC"
        LD_LIBRARY_PATH= git clone --depth 1 https://github.com/sgl-project/sglang.git "$SGLANG_SRC" || \
            LD_LIBRARY_PATH= git -c http.sslVerify=false clone --depth 1 https://github.com/sgl-project/sglang.git "$SGLANG_SRC"
    fi
    if [ -d "$SGLANG_SRC/python" ]; then
        echo "📦 pip install -e $SGLANG_SRC/python[diffusion] --no-deps --config-settings editable_mode=compat ---"
        # git main 的 pyproject.toml pin torch==2.13.0 / cuda-python>=13.0（CUDA 13），
        # A100 是 CUDA 12.4 装不上 cuda 13 wheel。--no-deps 只装 sglang 本体
        # （带 --model-variant / --performance-mode 等 diffusion serving 参数），
        # torch/flashinfer 用 env 现有版本；[diffusion] extra 依赖单独装。
        # SGLANG_BUILD_RUST_EXTS=none 跳过 Rust 扩展（cargo 不在时；运行时报错再装 rustup）。
        # editable_mode=compat：PEP 660 默认 strict 模式用 finder 映射包路径，可能漏掉
        # multimodal_gen 等子包（报 KeyError: 'sglang.multimodal_gen'）；compat 模式加
        # 源码目录到 sys.path，Python 直接遍历子目录，所有子包都能 import。
        SGLANG_BUILD_RUST_EXTS=none python -m pip install "${PIP_FLAGS[@]}" -e "$SGLANG_SRC/python[diffusion]" --no-deps --config-settings editable_mode=compat || \
            SGLANG_BUILD_RUST_EXTS=none python -m pip install "${PIP_FLAGS[@]}" -e "$SGLANG_SRC/python" --no-deps --config-settings editable_mode=compat
        # 修复 sglang git main 的 qwen3_asr 重复注册（transformers 5.12+ 内置同名配置，
        # sglang 代码 AutoConfig.register 没 exist_ok=True 会报 ValueError）。
        QWEN3_ASR_PY="$SGLANG_SRC/python/sglang/srt/configs/qwen3_asr.py"
        if [ -f "$QWEN3_ASR_PY" ] && ! grep -q 'exist_ok=True' "$QWEN3_ASR_PY"; then
            python - "$QWEN3_ASR_PY" <<'PYFIX'
import sys
path = sys.argv[1]
with open(path, 'r') as f:
    content = f.read()
content = content.replace(
    'AutoConfig.register("qwen3_asr", Qwen3ASRConfig)',
    'try:\n    AutoConfig.register("qwen3_asr", Qwen3ASRConfig, exist_ok=True)\nexcept ValueError:\n    pass'
)
content = content.replace(
    'AutoConfig.register("qwen3_asr_thinker", Qwen3ASRThinkerConfig)',
    'try:\n    AutoConfig.register("qwen3_asr_thinker", Qwen3ASRThinkerConfig, exist_ok=True)\nexcept ValueError:\n    pass'
)
with open(path, 'w') as f:
    f.write(content)
print(f"    ✅ patched {path} (exist_ok=True)")
PYFIX
        fi
        # 修复 flash_attention_v3.py 的 only_qv 参数不兼容旧 kernel
        # （sglang-kernel==0.4.1 的 flash_attn_varlen_func 不支持 only_qv，
        # sglang git main 传了它）。用 inspect.signature 动态过滤不支持的参数。
        FA3_PY="$SGLANG_SRC/python/sglang/kernels/ops/attention/flash_attention_v3.py"
        if [ -f "$FA3_PY" ] && ! grep -q 'inspect.signature' "$FA3_PY"; then
            python - "$FA3_PY" <<'PYFIX'
import sys
path = sys.argv[1]
with open(path, 'r') as f:
    content = f.read()
if 'only_qv=only_qv' not in content:
    print("    (flash_attention_v3.py: only_qv not used, skip)")
    sys.exit(0)
old = '''    return _call_fa3_kernel(
        _load_fa3_kernels()["flash_attn_varlen_func"],
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        page_table=page_table,
        softmax_scale=softmax_scale,
        causal=causal,
        qv=qv,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        window_size=window_size,
        attention_chunk=attention_chunk,
        softcap=softcap,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        only_qv=only_qv,
        sm_margin=sm_margin,
        return_softmax_lse=return_softmax_lse,
        sinks=sinks,
        out=out,
    )'''
new = '''    kernel = _load_fa3_kernels()["flash_attn_varlen_func"]
    import inspect as _inspect
    _supported = set(_inspect.signature(kernel).parameters)
    _kwargs = dict(
        q=q, k=k, v=v,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
        seqused_q=seqused_q, seqused_k=seqused_k,
        page_table=page_table, softmax_scale=softmax_scale,
        causal=causal, qv=qv,
        q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
        window_size=window_size, attention_chunk=attention_chunk,
        softcap=softcap, num_splits=num_splits, pack_gqa=pack_gqa,
        only_qv=only_qv, sm_margin=sm_margin,
        return_softmax_lse=return_softmax_lse, sinks=sinks, out=out,
    )
    _kwargs = {k: v for k, v in _kwargs.items() if k in _supported}
    return _call_fa3_kernel(kernel, **_kwargs)'''
if old not in content:
    print("    ⚠️ flash_attention_v3.py: expected pattern not found (file changed?)")
    sys.exit(0)
content = content.replace(old, new)
with open(path, 'w') as f:
    f.write(content)
print(f"    ✅ patched {path} (filter unsupported kernel args)")
PYFIX
        fi
        # --no-deps 跳过了 [diffusion] extra 的依赖，单独对齐。
        # transformers 须 pin ==5.12.1（5.15+ 有 qwen3_asr 重复注册冲突；4.x 缺 PreTrainedConfig）。
        # huggingface_hub 须与 transformers 5.x 配套（旧版缺 is_offline_mode）。
        # xgrammar 须 >=0.2.1（sglang git main 用了 AnyTokensFormat 等新 API）。
        python -m pip install "${PIP_FLAGS[@]}" -U diffusers peft "transformers==5.12.1" huggingface_hub xgrammar
    else
        echo "❌ ERROR: clone sglang repo failed. Manual: LD_LIBRARY_PATH= git clone https://github.com/sgl-project/sglang.git $SGLANG_SRC && SGLANG_BUILD_RUST_EXTS=none pip install -e \"$SGLANG_SRC/python[diffusion]\" --no-deps" >&2
    fi
    # 再自检
    if sglang serve --help 2>&1 | grep -q -- '--model-variant'; then
        echo "✅ sglang serve now supports --model-variant (Diffusion ok after git install)"
    else
        echo "❌ ERROR: sglang serve still does NOT recognize --model-variant." >&2
        echo "   Manual fallback: pip install -e \"$SGLANG_SRC/python[diffusion]\" then check deps." >&2
    fi
fi

echo "🎉 [00] Done. Env '$CONDA_ENV' ready. (Missing sglang? INSTALL_DEPS=1 bash this)"
