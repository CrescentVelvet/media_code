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

# ── 3b. canonical_face_model.obj（01b 要用；新版 mediapipe wheel 不内置）────
CANON_OBJ="$MODEL_DIR/mediapipe_aux/canonical_face_model.obj"
if [ ! -s "$CANON_OBJ" ]; then
    mkdir -p "$MODEL_DIR/mediapipe_aux"
    # 找 mediapipe 包内副本；没有则从官方仓镜像下载
    IN_PKG="$(python - <<'PY' 2>/dev/null
try:
    import mediapipe as mp, pathlib
    hits = sorted(pathlib.Path(mp.__file__).parent.rglob("canonical_face_model.obj"))
    print(hits[0] if hits else "")
except Exception:
    print("")
PY
)"
    if [ -n "$IN_PKG" ]; then
        cp "$IN_PKG" "$CANON_OBJ"
        echo "  ✅ canonical_face_model.obj（mediapipe 包内副本）"
    else
        curl -sL -m 60 -o "$CANON_OBJ" \
            "https://ghfast.top/https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj" \
        || curl -sL -m 60 -o "$CANON_OBJ" \
            "https://gh-proxy.com/https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj" \
        || true
        if [ -s "$CANON_OBJ" ] && head -1 "$CANON_OBJ" | grep -q "^v "; then
            echo "  ✅ canonical_face_model.obj（镜像下载）"
        else
            rm -f "$CANON_OBJ"
            echo "  ⚠️  canonical_face_model.obj 下载失败（01b 前需手动放置）"
            echo "     → 放到 $CANON_OBJ"
        fi
    fi
else
    echo "  ✅ canonical_face_model.obj"
fi

if [ "$missing" -ne 0 ]; then
    echo ""
    echo "❌ 有必需权重缺失，补齐后重跑本脚本。清单见 download_urls.md"
    exit 1
fi

# ── 4. 搭建 smplx 约定布局 + 校验 FLAME 能被 smplx 加载 ─────────────────────
# smplx.create() 传目录时拼 <dir>/flame/FLAME_NEUTRAL.pkl，且 FLAME 构造函数
# 无条件读同目录 flame_static_embedding.pkl（68 点静态嵌入，FLAME 官网单独
# 分发、不在 model.pkl 里）。布局：
#   $FLAME2020/flame/FLAME_NEUTRAL.pkl        → 符号链接 ../generic_model.pkl
#   $FLAME2020/flame/flame_static_embedding.pkl → 从 DECA 仓 landmark_embedding.npy 转换
# 幂等：链接与转换产物存在则跳过。
echo ""
echo "🔍 搭建 smplx 布局 + 校验 FLAME 可加载："
FLAME_DIR="$(dirname "$FLAME_MODEL")"
FLAME_SMPLX_SUB="$FLAME_DIR/flame"
mkdir -p "$FLAME_SMPLX_SUB"

if [ ! -e "$FLAME_SMPLX_SUB/FLAME_NEUTRAL.pkl" ]; then
    ln -s ../generic_model.pkl "$FLAME_SMPLX_SUB/FLAME_NEUTRAL.pkl" \
        || { echo "❌ 无法创建 FLAME_NEUTRAL.pkl 链接"; exit 1; }
    echo "  🔗 已建链接 flame/FLAME_NEUTRAL.pkl -> ../generic_model.pkl"
fi

if [ ! -s "$FLAME_SMPLX_SUB/flame_static_embedding.pkl" ] \
    || head -c4 "$FLAME_SMPLX_SUB/flame_static_embedding.pkl" | grep -q "404"; then
    if [ -f "$DECA_DIR/data/landmark_embedding.npy" ]; then
        python "$SCRIPT_DIR/convert_deca_lmk_to_smplx.py" \
            --src "$DECA_DIR/data/landmark_embedding.npy" \
            --dst "$FLAME_SMPLX_SUB/flame_static_embedding.pkl" \
            || { echo "❌ flame_static_embedding 转换失败"; exit 1; }
    else
        echo "  ❌ 缺 flame_static_embedding.pkl 且无 DECA 仓可转换"
        echo "     → 需 FLAME 官网 landmark embeddings 包，或先装好 DECA 仓"
        exit 1
    fi
else
    echo "  ⏭️  flame_static_embedding.pkl 已存在"
fi

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
    # smplx 约定：传目录，内部拼 flame/FLAME_NEUTRAL.pkl
    m = smplx.create(model_path=os.path.dirname(p), model_type="flame",
                     num_expression_coeffs=100, use_face_contour=False)
    v = m().vertices.detach().cpu().numpy().squeeze()
    print(f"  ✅ smplx 加载成功: {v.shape[0]} 顶点, expr 维度 {m.num_expression_coeffs}")
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
