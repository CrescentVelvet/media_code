#!/usr/bin/env bash
# 03_run_batch.sh — 批量推理：物体三图数据集（每物体 3 个视角抽帧）。
#
# 数据集结构（D:\dataset\物体三图数据 → WSL /mnt/d/dataset/物体三图数据）：
#   <dataset>/<物体名>/<帧号>.jpg   # 每物体 3 张，环绕视频抽帧
#
# TaoFlowForge 是单图模型：每张图独立产一个网格。VIEW_SEL 控制用哪些视角：
#   first —— 每物体只用第 1 张（默认，8 张，快速验证）
#   all   —— 每物体 3 张全跑（24 次推理，对比视角影响）
#
# 用法：
#   GPU=0 bash taoflowforge/03_run_batch.sh                  # 每物体首视角
#   GPU=0 VIEW_SEL=all bash taoflowforge/03_run_batch.sh     # 全部 3 视角
#   GPU=0 ONLY=剃须刀,鼠标 bash taoflowforge/03_run_batch.sh # 只跑指定物体
#   FORCE=1 ...                                              # 忽略已有产物重跑
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

DATASET_DIR="${DATASET_DIR:-/mnt/d/dataset/物体三图数据}"
VIEW_SEL="${VIEW_SEL:-first}"       # first | all
ONLY="${ONLY:-}"                    # 逗号分隔的物体名白名单
FORCE="${FORCE:-0}"

if [ ! -d "$DATASET_DIR" ]; then
    echo "❌ ERROR: 数据集目录不存在: $DATASET_DIR" >&2
    exit 1
fi

echo "🚀 [03] 批量推理: $DATASET_DIR"
echo "  视角选择: $VIEW_SEL   白名单: ${ONLY:-全部}   FORCE=$FORCE"
echo "  💾 输出根: $RESULTS_DIR"
echo ""

_total=0; _ok=0; _skip=0; _fail=0
_failed_names=""

for obj_dir in "$DATASET_DIR"/*/; do
    obj="$(basename "$obj_dir")"
    # 白名单过滤
    if [ -n "$ONLY" ]; then
        case ",$ONLY," in
            *",$obj,"*) ;;
            *) echo "⏭️  $obj（不在 ONLY 白名单）"; continue ;;
        esac
    fi

    # 收集图片（按文件名排序 = 帧号顺序）
    mapfile -t imgs < <(find "$obj_dir" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.png' -o -iname '*.jpeg' \) | sort)
    if [ ${#imgs[@]} -eq 0 ]; then
        echo "⚠️  $obj 目录无图片，跳过"
        continue
    fi
    if [ "$VIEW_SEL" = "first" ]; then
        imgs=("${imgs[0]}")
    fi

    for img in "${imgs[@]}"; do
        base="$(basename "$img")"
        name="${obj}_${base%.*}"
        _total=$((_total + 1))
        # 已有最终产物则跳过（断点重跑友好）
        if [ "$FORCE" != "1" ] && [ -f "$RESULTS_DIR/$name/mesh.glb" ]; then
            echo "⏭️  $name（已有 mesh.glb）"
            _skip=$((_skip + 1))
            continue
        fi
        echo ""
        echo "──── [$name] ────"
        if GPU="${GPU:-}" INPUT="$img" NAME="$name" bash "$SCRIPT_DIR/02_run_inference.sh"; then
            _ok=$((_ok + 1))
        else
            echo "❌ $name 推理失败（继续下一个）" >&2
            _fail=$((_fail + 1))
            _failed_names="$_failed_names $name"
        fi
    done
done

echo ""
echo "════════ 批量汇总 ════════"
echo "  总计: $_total   ✅ 成功: $_ok   ⏭️ 跳过: $_skip   ❌ 失败: $_fail"
[ -n "$_failed_names" ] && echo "  失败清单:$_failed_names"
echo "  💾 产物在: $RESULTS_DIR"

if [ "$_fail" -gt 0 ]; then
    exit 1
fi
echo "🎉 [03] Done."
