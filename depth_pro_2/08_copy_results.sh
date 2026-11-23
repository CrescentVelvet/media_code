#!/usr/bin/env bash
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"
export SCENE_DIR="${SCENE_DIR:?❌ SCENE_DIR is required}"
export FINAL_DIR="${FINAL_DIR:-/mnt/d/output/results/depth_pro_2_results}"
for name in human router router_unscaled router_sfm; do
    [ -d "$SCENE_DIR/$name" ] || { echo "❌ Missing result directory: $SCENE_DIR/$name" >&2; exit 1; }
done
mkdir -p "$FINAL_DIR"
echo "🚀 Copy verified Depth Pro 2 results without deleting WSL cache"
echo "  📁 From: $SCENE_DIR"
echo "  💾 To: $FINAL_DIR"
# Copy replaces named outputs only; unrelated experiments in results/ are untouched.
# DrvFs does not support preserving all source timestamps; content hashes are authoritative.
cp -r "$SCENE_DIR/human" "$SCENE_DIR/router" "$SCENE_DIR/router_unscaled" "$SCENE_DIR/router_sfm" "$FINAL_DIR/" || exit 1
python - "$SCENE_DIR" "$FINAL_DIR" <<'PY'
import hashlib
import sys
from pathlib import Path
source, dest = map(Path, sys.argv[1:])
for name in ("human", "router", "router_unscaled", "router_sfm"):
    files = sorted(p for p in (source / name).rglob("*") if p.is_file())
    for path in files:
        other = dest / name / path.relative_to(source / name)
        if not other.is_file():
            raise RuntimeError(f"Missing copy: {other}")
        if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(other.read_bytes()).digest():
            raise RuntimeError(f"Copy hash mismatch: {other}")
    print(f"✅ {name}: {len(files)} files, SHA256 verified")
PY
