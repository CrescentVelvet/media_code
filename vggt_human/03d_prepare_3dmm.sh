#!/usr/bin/env bash
# 03d_prepare_3dmm.sh — 下载 ICT-FaceKit 3DMM 头模板并生成拟合用模板 npz
#
# 头/身/景拆分的前置：人头不用 SfM 点（稀疏 + 后脑缺失），改用 3DMM 模板
# 多视角拟合，这样 head_gs 是完整人头（含后脑），closeup 推近的 alpha 覆盖率
# 才有物理意义（当前椭球代理下 40% 目标不可达，见 2026-09-06 日志）。
#
# 产出:
#   $MODEL_3DMM_DIR/FaceXModel/generic_neutral_mesh.obj   # 通用头（26719 顶点，含后脑/颈）
#   $MODEL_3DMM_DIR/FaceXModel/identity0XX.obj            # 24 个身份 → identity PCA 基
#   $MODEL_3DMM_DIR/mediapipe/canonical_face_model.obj    # MediaPipe 468 canonical
#   $MODEL_3DMM_DIR/3dmm_template.npz                     # skin 网格 + PCA + landmark 对应
#   $MODEL_3DMM_DIR/3dmm_align_check.png                  # 对齐验证图（必看）
#
# ICT-FaceKit 是 MIT 许可（USC-ICT/ICT-FaceKit），无需注册；
# 相比 FLAME / BFM 需要签协议，这是唯一能直接 curl 下来的完整头模型。
#
# Env (all optional, defaults shown):
#   MODEL_3DMM_DIR=   # 模型根（默认 $WEIGHTS_ROOT/ICT-FaceKit，WSL: ~/model/ICT-FaceKit）
#   SKIP_IDENTITIES=0 # 1 = 只下通用头，不建 identity PCA（省 60MB）
#   ICP_ITERS=40      # landmark 配准迭代数
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

# ⚠️ 不要拿 WEIGHTS_ROOT 拼路径：proxy.env 里它被设成了 /mnt/d/... 的遗留值，
#    会盖掉调用方传入的值（_env.sh 用 set -a source proxy.env）。
MODEL_3DMM_DIR="${MODEL_3DMM_DIR:-$HOME/model/ICT-FaceKit}"
SKIP_IDENTITIES="${SKIP_IDENTITIES:-0}"
ICP_ITERS="${ICP_ITERS:-40}"

BASE_URL="https://raw.githubusercontent.com/USC-ICT/ICT-FaceKit/master/FaceXModel"
MP_URL="https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj"

echo "🧱 [03d] 准备 3DMM 头模板 (ICT-FaceKit, MIT)"
echo "  📂 模型根: $MODEL_3DMM_DIR"
echo ""

mkdir -p "$MODEL_3DMM_DIR/FaceXModel" "$MODEL_3DMM_DIR/mediapipe"

# ── 1) 通用头 ──────────────────────────────────────────────────────────────
if [ -f "$MODEL_3DMM_DIR/FaceXModel/generic_neutral_mesh.obj" ]; then
    echo "  ✔️ 通用头已存在，跳过下载"
else
    echo "  ⬇️ 下载 generic_neutral_mesh.obj (2.5MB)"
    curl -sL -o "$MODEL_3DMM_DIR/FaceXModel/generic_neutral_mesh.obj" \
        "$BASE_URL/generic_neutral_mesh.obj" || { echo "❌ 下载失败" >&2; exit 1; }
fi

# ── 2) MediaPipe canonical 468 ─────────────────────────────────────────────
if [ -f "$MODEL_3DMM_DIR/mediapipe/canonical_face_model.obj" ]; then
    echo "  ✔️ MediaPipe canonical 已存在，跳过下载"
else
    echo "  ⬇️ 下载 canonical_face_model.obj"
    curl -sL -o "$MODEL_3DMM_DIR/mediapipe/canonical_face_model.obj" \
        "$MP_URL" || { echo "❌ 下载失败" >&2; exit 1; }
fi

# ── 3) identity 网格（identity PCA 形状基） ────────────────────────────────
if [ "$SKIP_IDENTITIES" = "1" ]; then
    echo "  ⏭️ SKIP_IDENTITIES=1，不下载身份网格"
else
    n_existing=$(ls "$MODEL_3DMM_DIR"/FaceXModel/identity0*.obj 2>/dev/null | wc -l)
    if [ "$n_existing" -ge 24 ]; then
        echo "  ✔️ identity 网格已存在 ($n_existing)，跳过下载"
    else
        echo "  ⬇️ 下载 identity000-023 (24×2.5MB)…"
        for i in $(seq -w 0 23); do
            [ -f "$MODEL_3DMM_DIR/FaceXModel/identity0$i.obj" ] && continue
            curl -sL -o "$MODEL_3DMM_DIR/FaceXModel/identity0$i.obj" \
                "$BASE_URL/identity0$i.obj"
        done
        echo "  ✅ identity 网格: $(ls "$MODEL_3DMM_DIR"/FaceXModel/identity0*.obj | wc -l) 个"
    fi
fi

# ── 4) 生成模板 npz ────────────────────────────────────────────────────────
echo ""
echo "  🔧 生成 3dmm_template.npz"
MODEL_3DMM_DIR="$MODEL_3DMM_DIR" \
ICP_ITERS="$ICP_ITERS" \
python "$SCRIPT_DIR/prepare_3dmm_template.py"
if [ $? -ne 0 ]; then
    echo "❌ FAILED" >&2
    exit 1
fi

echo ""
echo "✅ [03d] 3DMM 模板就绪"
echo "  📄 $MODEL_3DMM_DIR/3dmm_template.npz"
echo "  🖼️ $MODEL_3DMM_DIR/3dmm_align_check.png   ← 先看这张图确认 468 点贴在人脸上"
