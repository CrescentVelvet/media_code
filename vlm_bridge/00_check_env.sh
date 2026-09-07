#!/usr/bin/env bash
# 00_check_env.sh — one-shot connectivity check for vlm_bridge (no conda needed).
# Verifies vlm.env exists, then sends a tiny text-only request to the VLM API.
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo "🚀 [vlm_bridge] env check"
if [ ! -f "$SCRIPT_DIR/vlm.env" ]; then
    echo "📦 creating vlm.env from vlm.env.example ..."
    cp "$SCRIPT_DIR/vlm.env.example" "$SCRIPT_DIR/vlm.env"
    echo "✅ created: $SCRIPT_DIR/vlm.env"
fi

# Pick a python: any python3 works (script is stdlib-only)
PY=""
for c in python python3 py; do
    if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
[ -z "$PY" ] && { echo "❌ ERROR: no python found on PATH" >&2; exit 1; }

# Text-only ping to verify key + network (kept stdlib-only, zero deps)
cd "$REPO_DIR"
"$PY" - <<'EOF'
import json, sys, urllib.request
from pathlib import Path

# load vlm_bridge/vlm.env (same parsing rules as ask_vlm.py)
p = Path("vlm_bridge/vlm.env")
env = {}
for line in p.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    if line.startswith("export "):
        line = line[7:]
    k, _, v = line.partition("=")
    env[k.strip()] = v.strip().strip('"').strip("'")

url, key = env.get("VLM_API_URL"), env.get("VLM_API_KEY")
model = env.get("VLM_MODEL", "qwen3.8-max")
if not url or not key:
    sys.exit("ERROR: vlm.env missing VLM_API_URL / VLM_API_KEY")
print(f"🔍 endpoint: {url}")
print(f"🤖 model:    {model}")

payload = {"model": model,
           "messages": [{"role": "user", "content": "Reply with the single word: pong"}],
           "max_tokens": 8}
req = urllib.request.Request(url, data=json.dumps(payload).encode(),
    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
try:
    with urllib.request.build_opener().open(req, timeout=30) as r:
        body = json.loads(r.read().decode())
    text = body["choices"][0]["message"]["content"]
    print(f"✅ API reachable, reply: {text!r}")
except Exception as e:
    sys.exit(f"ERROR: API check failed: {e}")
EOF
if [ $? -ne 0 ]; then
    echo "❌ FAILED" >&2
    exit 1
fi
echo "🎉 vlm_bridge ready. Try: python vlm_bridge/ask_vlm.py <image> \"what is this?\""
