#!/usr/bin/env bash
# _probe_3view.sh — 核实四方法前馈推理在 N=3 输入下的代码级约束（只读检查，不跑推理）
set -o pipefail
exec > /mnt/c/code/media_code/compare_4methods/_probe_3view.log 2>&1

echo "########## 1) InfiniSplat v2 CLI 包 ##########"
INFER_BIN=$(which infinisplat 2>/dev/null)
echo "BIN=$INFER_BIN"
if [ -n "$INFER_BIN" ]; then
    PYBIN=$(head -1 "$INFER_BIN" | sed 's/^#!//')
    SITE=$("$PYBIN" -c 'import infinisplat, os; print(os.path.dirname(infinisplat.__file__))' 2>/dev/null)
    echo "SITE=$SITE"
    if [ -n "$SITE" ]; then
        echo "--- CLI 里 max-views / 视角数解析 ---"
        grep -rn "max.views\|max_views" "$SITE" --include="*.py" | head -8
        echo "--- 视角数断言/下限 ---"
        grep -rn "assert" "$SITE" --include="*.py" | grep -i "view\|image\|len" | head -10
    fi
fi
echo "--- 官方仓内 CLI 定义 ---"
grep -rn "max.views\|max_views" ~/repos/InfiniSplat/InfiniSplat_v2 --include="*.py" 2>/dev/null | grep -v test | head -8

echo ""
echo "########## 2) QuerySplat infer.py 输入视角处理 ##########"
grep -n "input_folder\|sorted\|glob\|image_files\|num_img\|len(" ~/repos/QuerySplat/scripts/infer.py | head -20
echo "--- infer.py 视角数断言 ---"
grep -n "assert" ~/repos/QuerySplat/scripts/infer.py | head -10
echo "--- yaml 视角相关 ---"
grep -n "view" ~/repos/QuerySplat/checkpoints/querysplat_vggto_1B_512_8192.yaml | head -10

echo ""
echo "########## 3) DA3 (da3_human) 视角数处理 ##########"
grep -n "MAX_VIEWS\|max_views\|len(\|assert" /mnt/c/code/media_code/da3_human/run_da3_inference.py 2>/dev/null | head -15
echo "--- DA3 官方仓视角数采样 ---"
grep -rn "num_views\|n_views" ~/repos/Depth-Anything-3 --include="*.py" 2>/dev/null | grep -iv test | head -8

echo ""
echo "########## 4) VGGT-Ω 视角数 ##########"
grep -n "assert\|len(\|views" /mnt/c/code/media_code/vggt_human/run_batch.py 2>/dev/null | head -12

echo "PROBE_DONE"
