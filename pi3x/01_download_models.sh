#!/usr/bin/env bash
# 01_download_models.sh — 下载 Pi3X 权重 (yyfz233/Pi3X, ~4GB safetensors)。
#
# 落盘策略（AGENTS.md 权重复用池）：下载到 $PI3X_WHEEL_DIR（默认 /mnt/d/wheel/pi3x_ms），
# 消费侧 $MODEL_DIR/model.safetensors 做软链。
#
# 通道优先级（WSL 实测 2026-11-20）：
#   1. hf-mirror.com 直连 ~2.1MB/s（国内 CDN，必须绕开 proxy.env 的代理，否则被拖慢）
#   2. huggingface.co 走 proxy.env 代理 ~0.1MB/s（兜底，断点续传）
#
# 用法:
#   bash pi3x/01_download_models.sh
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

MIRROR_URL="${PI3X_CKPT_URL_MIRROR:-https://hf-mirror.com/yyfz233/Pi3X/resolve/main/model.safetensors}"
OUT="$PI3X_WHEEL_DIR/model.safetensors"
SEG=6

echo "🚀 [01] 下载 Pi3X 权重"
echo "  💾 落盘: $OUT"

if [ -f "$OUT" ]; then
    echo "⏭️  wheel 池已有，跳过下载"
else
    mkdir -p "$PI3X_WHEEL_DIR"
    # --- 通道 1：hf-mirror 多段并行（绕开代理）---
    SIZE="$(env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
             curl -sIL --max-time 15 "$MIRROR_URL" | grep -i '^content-length' | tail -1 | tr -dc '0-9')"
    if [ -n "$SIZE" ] && [ "$SIZE" -gt 1000000000 ]; then
        echo "  🌐 hf-mirror 直连，文件 $((SIZE / 1024 / 1024)) MiB，$SEG 段并行"
        CHUNK=$(( (SIZE + SEG - 1) / SEG ))
        pids=()
        for i in $(seq 0 $((SEG - 1))); do
            start=$((i * CHUNK))
            end=$(( (i + 1) * CHUNK - 1 ))
            [ "$end" -ge "$SIZE" ] && end=$((SIZE - 1))
            env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
                curl -sfL --retry 3 -r "$start-$end" -o "$OUT.part$i" "$MIRROR_URL" &
            pids+=($!)
        done
        ok=1
        for p in "${pids[@]}"; do wait "$p" || ok=0; done
        if [ "$ok" = "1" ]; then
            cat "$OUT".part* > "$OUT"
            rm -f "$OUT".part*
            GOT="$(stat -c %s "$OUT")"
            if [ "$GOT" != "$SIZE" ]; then
                echo "⚠️ 分段合并后大小不符 ($GOT != $SIZE)，转单流续传"
                mv "$OUT" "$OUT.bak"
            fi
        else
            echo "⚠️ 分段下载有失败，转单流续传"
            rm -f "$OUT".part*
        fi
    fi
    # --- 通道 2：HF 代理单流断点续传（兜底；兼容通道 1 失败后的重下）---
    if [ ! -f "$OUT" ]; then
        echo "  🌐 走代理下载（慢，兜底）: $PI3X_CKPT_URL"
        curl -fL -C - --retry 3 -o "$OUT" "$PI3X_CKPT_URL"
        if [ $? -ne 0 ]; then
            echo "❌ FAILED: 下载失败（可浏览器/迅雷下好后手动放到 $OUT）" >&2
            exit 1
        fi
    fi
fi

# 消费侧软链
mkdir -p "$MODEL_DIR"
if [ ! -e "$MODEL_DIR/model.safetensors" ]; then
    ln -s "$OUT" "$MODEL_DIR/model.safetensors"
fi

FINAL="$(stat -c %s "$MODEL_DIR/model.safetensors" 2>/dev/null || echo 0)"
echo "  📦 大小: $((FINAL / 1024 / 1024)) MiB"
if [ "$FINAL" -lt 1000000000 ]; then
    echo "❌ FAILED: 文件异常小（可能下到 HTML 错误页），删掉重下" >&2
    exit 1
fi
echo "🎉 [01] Done. ckpt = $MODEL_DIR/model.safetensors"
