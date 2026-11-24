#!/usr/bin/env bash
# 26_run_paired.sh -- 三图/五图前馈：SA RGB、SA pose、Zip FF；绝不训练。
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/mnt/d/output/results/splatanything_results/densitok_portrait_probe/protocol_v1}"
ONLY_SCENE="${ONLY_SCENE:-}"
ONLY_ARM="${ONLY_ARM:-}"
export ROOT
[ -f "$ROOT/manifest.json" ] || { echo "❌ 先运行 26_prepare_paired.sh" >&2; exit 1; }
"$HOME/miniconda3/envs/splatanything/bin/python" - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
p = json.load(open(root / "manifest.json", encoding="utf-8"))
for pack in p["packages"]:
    if not pack.get("new_identity"):
        continue
    qa = json.load(open(root / "roi_qa.json", encoding="utf-8"))
    if not qa.get("status", "").startswith("visual_review_complete"):
        raise SystemExit("New identity ROI visual review has not passed")
    for fr in pack["frames"]:
        if fr["split"] == "target" and qa["views"][str(fr["raw_id"])]["status"] not in ("accepted", "partial"):
            raise SystemExit("Target ROI is not accepted: " + str(fr["raw_id"]))
PY
[ $? -eq 0 ] || exit 1
mkdir -p "$ROOT/models"
"$HOME/miniconda3/envs/splatanything/bin/python" - "$ROOT" <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path
root = Path(sys.argv[1])
p = json.load(open(root / "manifest.json", encoding="utf-8"))
protocol = json.load(open("/mnt/d/output/results/splatanything_results/n1_real_20261122/protocol_v1/protocol.json", encoding="utf-8"))
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
weights = ["/mnt/d/wheel/splatanything_ms/SplatAnything.ckpt",
           "/mnt/d/wheel/splatanything_ms/worldmirror2.safetensors",
           str(Path.home() / "model/zipsplat/zipsplat-da3g-504.tar")]
record = {"created_at_first_model_launch": True,
          "scope": "fixed paths/content hashes, not inference wall time",
          "training": False, "token_ba": False,
          "weights": {path: sha(path) for path in weights},
          "official_commits": {path: subprocess.check_output(
              ["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
              for path in [str(Path.home() / "repos/SplatAnything"),
                           str(Path.home() / "repos/ZipSplat")]},
          "n1_protocol_software": protocol.get("software", {})}
with open(root / "run_provenance.json", "w", encoding="utf-8", newline="\n") as f:
    json.dump(record, f, ensure_ascii=False, indent=2)
print("RUN_PROVENANCE", root / "run_provenance.json", flush=True)
PY
[ $? -eq 0 ] || exit 1
[ "${PROVENANCE_ONLY:-0}" != "1" ] || { echo "✅ 仅记录运行来源，未启动模型。"; exit 0; }
"$HOME/miniconda3/envs/splatanything/bin/python" - "$ROOT/manifest.json" > "$ROOT/model_inputs.txt" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
for pack in p["packages"]:
    for arm in ("source3", "source5"):
        print(pack["name"] + "|" + arm + "|" + pack[arm + "_scene"])
PY
while IFS='|' read -r SCENE ARM INPUT; do
  [ -n "$ONLY_SCENE" ] && [ "$SCENE" != "$ONLY_SCENE" ] && continue
  [ -n "$ONLY_ARM" ] && [ "$ARM" != "$ONLY_ARM" ] && continue
  export CONDA_ENV=splatanything
  source "$SCRIPT_DIR/_env.sh"
  for MODE in rgb pose; do
    OUT="$ROOT/models/sa_${MODE}/${SCENE}_${ARM}"
    [ ! -e "$OUT" ] || { echo "❌ 输出已存在: $OUT" >&2; exit 1; }
    echo "🚀 SA $MODE / $SCENE / $ARM"
    INPUT="$INPUT" MODE="$MODE" RESOLUTION="756 756" \
      SPLATANYTHING_MODEL_DIR="${SPLATANYTHING_MODEL_DIR:-/mnt/d/wheel/splatanything_ms}" \
      OUT_ROOT="$ROOT/models/sa_${MODE}" PROFILE=1 \
      bash "$SCRIPT_DIR/02_run_inference.sh" || exit 1
    [ -f "$OUT/gaussians.ply" ] || { echo "❌ 缺 SA PLY: $OUT" >&2; exit 1; }
  done
  export CONDA_ENV=zipsplat
  source "$SCRIPT_DIR/../zipsplat/_env.sh"
  OUT="$ROOT/models/zip_rgb/${SCENE}_${ARM}"
  [ ! -e "$OUT" ] || { echo "❌ 输出已存在: $OUT" >&2; exit 1; }
  echo "🚀 Zip FF / $SCENE / $ARM"
  IMAGES_DIR="$INPUT/images" SCENE_NAME="${SCENE}_${ARM}" \
    WEIGHT_PATH="${WEIGHT_PATH:-$HOME/model/zipsplat/zipsplat-da3g-504.tar}" \
    OUT_DIR="$OUT" MAX_VIEWS=5 BA=0 SAVE_FEEDFORWARD=1 RENDER_VIDEO=0 \
    bash "$SCRIPT_DIR/../zipsplat/02_run_inference.sh" || exit 1
  [ -f "$OUT/gaussians_feedforward.ply" ] || { echo "❌ 缺 Zip PLY: $OUT" >&2; exit 1; }
done < "$ROOT/model_inputs.txt"
echo "🎉 所有指定配对前馈完成；下一步用 27 号脚本核验原生回放并逐 target 评分。"
