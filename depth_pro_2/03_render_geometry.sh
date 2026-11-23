#!/usr/bin/env bash
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"
export SCENE_DIR="${SCENE_DIR:?❌ SCENE_DIR is required}"
echo "🚀 PCD / splat / PCD-depth generation: $SCENE_DIR"
python "$SCRIPT_DIR/render_geometry.py" || exit 1
python "$SCRIPT_DIR/render_crossview_video.py" || exit 1
python "$SCRIPT_DIR/evaluate_consistency.py" || exit 1
python - "$SCENE_DIR" <<'PY'
import sys
from pathlib import Path
import av
from PIL import Image
root = Path(sys.argv[1])
files = sorted((root / "video_frames").glob("*.jpg"))
out = root / "rgb_depth_pcd_comparison.mp4"
with av.open(str(out), "w") as container:
    stream = container.add_stream("libx264", rate=1)
    first = av.VideoFrame.from_image(Image.open(files[0]))
    stream.width, stream.height = first.width, first.height
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18", "preset": "medium"}
    for path in files:
        for _ in range(3):
            frame = av.VideoFrame.from_image(Image.open(path))
            for packet in stream.encode(frame):
                container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
print(f"🎬 Video: {out}")
PY
