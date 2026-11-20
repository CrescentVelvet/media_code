#!/usr/bin/env bash
# 01_download_models.sh — ZipSplat 真实图像用 504px 权重（6,002,349,727 bytes）
# 下载实体进 D:\wheel 复用池，$WEIGHT_PATH 为消费侧软链。
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

WEIGHT_NAME="${WEIGHT_NAME:-zipsplat-da3g-504.tar}"
EXPECTED_BYTES="${EXPECTED_BYTES:-6002349727}"
MIRROR_URL="${MIRROR_URL:-https://hf-mirror.com/veichta/zipsplat/resolve/main/$WEIGHT_NAME}"
OFFICIAL_URL="${OFFICIAL_URL:-https://huggingface.co/veichta/zipsplat/resolve/main/$WEIGHT_NAME}"
OUT="$WHEEL_ZIPSPLAT_DIR/$WEIGHT_NAME"
mkdir -p "$WHEEL_ZIPSPLAT_DIR" "$MODEL_DIR"

echo "🚀 [01] ZipSplat 权重: $WEIGHT_NAME"
if [ ! -f "$OUT" ]; then
    # 镜像服务器接受 Range；按段保存，任一段失败则不拼成伪造的完整权重。
    SIZE="$(env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
        curl -sIL --connect-timeout 15 --max-time 60 "$MIRROR_URL" | \
        awk 'BEGIN{IGNORECASE=1} /^content-length:/{n=$2} END{gsub("\\r", "", n); print n}')"
    if [ "$SIZE" != "$EXPECTED_BYTES" ]; then
        echo "⚠️ 镜像未返回预期长度 ($SIZE != $EXPECTED_BYTES)，改单流断点下载"
    else
        echo "📦 镜像直连：$((SIZE / 1024 / 1024)) MiB，6 段并行"
        SEG=6
        CHUNK=$(( (SIZE + SEG - 1) / SEG ))
        pids=()
        for i in $(seq 0 $((SEG - 1))); do
            start=$((i * CHUNK))
            end=$(( (i + 1) * CHUNK - 1 ))
            [ "$end" -ge "$SIZE" ] && end=$((SIZE - 1))
            env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
                curl -sfL --retry 3 --connect-timeout 20 -r "$start-$end" \
                -o "$OUT.part$i" "$MIRROR_URL" &
            pids+=($!)
        done
        ok=1
        for p in "${pids[@]}"; do wait "$p" || ok=0; done
        if [ "$ok" = "1" ]; then
            for i in $(seq 0 $((SEG - 1))); do cat "$OUT.part$i"; done > "$OUT"
            GOT="$(stat -c %s "$OUT")"
            if [ "$GOT" = "$SIZE" ]; then
                # 已通过完整文件长度校验，再逐段清理临时分片。
                for i in $(seq 0 $((SEG - 1))); do rm "$OUT.part$i"; done
            else
                echo "⚠️ 合并长度校验失败 ($GOT != $SIZE)，保留分段，转单流下载"
                mv "$OUT" "$OUT.invalid"
            fi
        else
            echo "⚠️ 分段下载未全部成功，转单流下载"
        fi
    fi
    if [ ! -f "$OUT" ]; then
        curl -fL -C - --retry 3 --connect-timeout 20 -o "$OUT" "$OFFICIAL_URL" || {
            echo "❌ 下载失败，可用代理重跑，或把官方文件放到 $OUT" >&2; exit 1; }
    fi
fi
GOT="$(stat -c %s "$OUT")"
if [ "$GOT" != "$EXPECTED_BYTES" ]; then
    echo "❌ 权重长度 $GOT != $EXPECTED_BYTES，不加载不完整文件" >&2
    exit 1
fi
if [ -L "$WEIGHT_PATH" ] || [ ! -e "$WEIGHT_PATH" ]; then
    ln -sfn "$OUT" "$WEIGHT_PATH"
elif [ "$(stat -c %s "$WEIGHT_PATH")" != "$EXPECTED_BYTES" ]; then
    echo "❌ 消费侧已有不同权重，人工确认: $WEIGHT_PATH" >&2
    exit 1
fi
sha256sum "$OUT"
python - "$WEIGHT_PATH" <<'PY'
import sys, torch
ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert set(("model", "conf")) <= set(ckpt)
print("✅ checkpoint keys:", sorted(ckpt))
print("✅ gaussians/token:", ckpt["conf"]["model"]["gaussians_per_token"])
print("✅ resize:", ckpt["conf"]["data"]["preprocessing"]["resize"])
PY
if [ $? -ne 0 ]; then echo "❌ 权重无法加载" >&2; exit 1; fi
echo "🎉 Done. $WEIGHT_PATH -> $OUT"
