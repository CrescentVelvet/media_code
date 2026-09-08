#!/usr/bin/env bash
# 01_download_models.sh — 检查 FLAME 2020 / DECA 权重是否就位。
#
# 两者都要官网注册才能下载，脚本不做自动下载：缺哪个就打印对应指引并退出。
# 清单与 URL 见 download_urls.md。
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

echo "🚀 [flame_human 01] 权重检查"
echo "  🏋️ MODEL_DIR: $MODEL_DIR"

mkdir -p "$MODEL_DIR" "$(dirname "$FLAME_MODEL")"

missing=0

# ── 1. FLAME 2020 ───────────────────────────────────────────────────────────
if [ -f "$FLAME_MODEL" ]; then
    echo "  ✅ FLAME: $FLAME_MODEL"
else
    echo "  ❌ FLAME 缺失: $FLAME_MODEL"
    echo "     → https://flame.is.tue.mpg.de/download.php 注册后下载 FLAME 2020 的 generic_model.pkl"
    echo "     → 不要用 FLAME_neutral_*.pkl（不带 expression 空间）"
    missing=1
fi

# ── 2. DECA ─────────────────────────────────────────────────────────────────
if [ -f "$DECA_CKPT" ]; then
    echo "  ✅ DECA: $DECA_CKPT"
else
    echo "  ❌ DECA 缺失: $DECA_CKPT"
    echo "     → 见 $DECA_DIR/README.md 里的 Google Drive 链接（需先有 FLAME 2020）"
    missing=1
fi

# ── 3. HYPIR（阶段五/八用；缺了只是不能增强，不阻塞前面阶段）──────────────────
if [ -f "$HYPIR_WEIGHT" ] && [ -d "$HYPIR_BASE_MODEL" ]; then
    echo "  ✅ HYPIR: $HYPIR_WEIGHT"
else
    echo "  ⚠️  HYPIR 缺失（仅影响阶段五/八增强）: $HYPIR_WEIGHT"
    echo "     → 与 vggt_human 共用，先跑 vggt_human/00a_setup_env.sh"
fi

if [ "$missing" -ne 0 ]; then
    echo ""
    echo "❌ 有必需权重缺失，补齐后重跑本脚本。清单见 download_urls.md"
    exit 1
fi

# ── 4. 校验 FLAME 能被 smplx 加载 ───────────────────────────────────────────
echo ""
echo "🔍 校验 FLAME 可加载："
FLAME_MODEL="$FLAME_MODEL" python - <<'PY'
import os, sys
import numpy as np
try:
    import smplx
except ImportError:
    print("  ❌ smplx 未安装（bash 00a_setup_env.sh）")
    sys.exit(1)
p = os.environ["FLAME_MODEL"]
try:
    # FLAME 走 SMPL-X loader；num_expression_coeffs=100 是 FLAME 2020 的 expr 维度
    m = smplx.create(model_path=p, model_type="flame",
                     num_expression_coeffs=100, use_face_contour=False)
    v = m().vertices.detach().cpu().numpy().squeeze()
    print(f"  ✅ 加载成功: {v.shape[0]} 顶点, expr 维度 {m.num_expression_coeffs}")
except Exception as e:
    print(f"  ❌ 加载失败: {type(e).__name__}: {e}")
    print("     → 确认下载的是 FLAME 2020 的 generic_model.pkl（含 expression）")
    sys.exit(1)
PY
if [ $? -ne 0 ]; then
    exit 1
fi

echo ""
echo "🎉 权重就位。下一步：bash 01b_build_lm468_embedding.sh"
