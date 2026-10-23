#!/usr/bin/env bash
# 01_check_models.sh — 校验 Qwen-Image-2.1 权重完整性。
# 按官方仓库（HF Qwen/Qwen-Image-2.1 / ModelScope 同源）API 字节数逐一比对 25 个文件。
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

MODEL_PATH="${MODEL_PATH:-/mnt/d/wheel/qwen_image21_ms}"

echo "🔍 [01] 校验权重: $MODEL_PATH"

if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ ERROR: 模型目录不存在: $MODEL_PATH" >&2
    exit 1
fi

# 官方文件清单（2026-10-22 从 ModelScope API 抓取，与 HF 同源）
EXPECTED=$(cat <<'EOF'
model_index.json 447
configuration.json 34
scheduler/scheduler_config.json 485
transformer/config.json 370
transformer/diffusion_pytorch_model.safetensors.index.json 30283
transformer/diffusion_pytorch_model-00001-of-00002.safetensors 9968332504
transformer/diffusion_pytorch_model-00002-of-00002.safetensors 4261951904
text_encoder/config.json 1517
text_encoder/generation_config.json 213
text_encoder/model.safetensors.index.json 67795
text_encoder/model-00001-of-00004.safetensors 4998056552
text_encoder/model-00002-of-00004.safetensors 4915962464
text_encoder/model-00003-of-00004.safetensors 4915962496
text_encoder/model-00004-of-00004.safetensors 2704357976
vae/config.json 2079
vae/diffusion_pytorch_model.safetensors 1350989512
processor/added_tokens.json 707
processor/chat_template.jinja 5292
processor/merges.txt 1671853
processor/preprocessor_config.json 782
processor/special_tokens_map.json 613
processor/tokenizer.json 11422654
processor/tokenizer_config.json 5445
processor/video_preprocessor_config.json 817
processor/vocab.json 2776833
EOF
)

FAIL=0
TOTAL_BYTES=0
while read -r relpath size; do
    [ -z "$relpath" ] && continue
    f="$MODEL_PATH/$relpath"
    actual=$(stat -c%s "$f" 2>/dev/null || echo MISSING)
    if [ "$actual" = "$size" ]; then
        echo "  ✅ $relpath"
    else
        echo "  ❌ $relpath  expect=$size actual=$actual" >&2
        FAIL=1
    fi
    TOTAL_BYTES=$((TOTAL_BYTES + size))
done <<< "$EXPECTED"

if [ $FAIL -ne 0 ]; then
    echo "❌ 权重不完整或有损坏，见上方 ❌ 项" >&2
    exit 1
fi
echo "✅ 25 个文件全部字节级匹配（共 $((TOTAL_BYTES / 1024 / 1024 / 1024)) GiB）"
echo "🎉 Done."
