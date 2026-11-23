#!/usr/bin/env bash
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"
export SCENE_DIR="${SCENE_DIR:?❌ SCENE_DIR is required}"
export HUMAN_RAW_DIR="${HUMAN_RAW_DIR:?❌ HUMAN_RAW_DIR is required}"
export ROUTER_RAW_DIR="${ROUTER_RAW_DIR:?❌ ROUTER_RAW_DIR is required}"
export ROUTER_POSE_DIR="${ROUTER_POSE_DIR:?❌ ROUTER_POSE_DIR is required}"
for scene in human router; do
    raw="$HUMAN_RAW_DIR"
    if [ "$scene" = router ]; then raw="$ROUTER_RAW_DIR"; fi
    INPUT_DIR="$SCENE_DIR/$scene" OUT_DIR="$SCENE_DIR/$scene" \
        SCENE="$scene" RAW_DIR="$raw" POSE_DIR="$ROUTER_POSE_DIR" \
        bash "$SCRIPT_DIR/01_prepare_data.sh" || exit 1
    INPUT_DIR="$SCENE_DIR/$scene" OUT_DIR="$SCENE_DIR/$scene" \
        bash "$SCRIPT_DIR/02_run_inference.sh" || exit 1
    if [ "$scene" = router ]; then
        SCENE_DIR="$SCENE_DIR/$scene" POSE_DIR="$ROUTER_POSE_DIR" \
            bash "$SCRIPT_DIR/03a_audit_colmap.sh" || exit 1
    fi
    SCENE_DIR="$SCENE_DIR/$scene" bash "$SCRIPT_DIR/03_render_geometry.sh" || exit 1
done
