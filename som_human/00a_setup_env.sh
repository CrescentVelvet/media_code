#!/usr/bin/env bash
# 00a_setup_env.sh —【WSL 变体】som_human 环境校验（不新建 env）。
#
# 方案决定（media_paper 手部微动方案 §9.2）：训练侧直接复用 deformable_human env
# （torch 2.5.1+cu121 + fork 光栅化器），官方仓也共用 ~/repos/Deformable-3D-Gaussians。
# 所以这里只做体检，不做安装：
#   1. deformable_human env 存在；2. 官方仓在位；3. fork 光栅化器可 import。
# 体检不过 → 按提示回 deformable_human 目录跑它的 00a。
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

echo "🚀 [00a] som_human 环境体检（复用 deformable_human env，不新建）"

FAIL=0
if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    echo "❌ env '$CONDA_ENV' 不存在。先跑：INSTALL_DEPS=1 BUILD_CUDA=1 bash deformable_human/00a_setup_env.sh" >&2
    FAIL=1
fi
if [ ! -f "$DG_DIR/train.py" ]; then
    echo "❌ 官方仓未就绪: $DG_DIR（先跑 deformable_human/00a_setup_env.sh）" >&2
    FAIL=1
fi
if [ "$FAIL" = "0" ]; then
    if ! python -c "import torch, diff_gaussian_rasterization, simple_knn; print('  🎮 torch', torch.__version__, 'cuda', torch.version.cuda)" 2>&1; then
        echo "❌ CUDA 扩展不可 import。修复：BUILD_CUDA=1 bash deformable_human/00a_setup_env.sh" >&2
        FAIL=1
    fi
fi
[ "$FAIL" != "0" ] && exit 1

# SoM 模块自身可 import（语法/依赖自检，不需要 CUDA 前向）
python - "$SCRIPT_DIR" "$DG_DIR" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])  # som_human 目录
sys.path.insert(0, sys.argv[2])  # 官方仓（utils 依赖）
import som_motion
m = som_motion.SoMMotionModel(num_frames=4, num_bases=10)
print("✅ [00a] som_motion 模块自检通过（num_frames=4, num_bases=10）")
PY
[ $? -ne 0 ] && { echo "❌ som_motion.py import 失败" >&2; exit 1; }

echo "🎉 [00a] Done. 环境就绪，直接进 02_train.sh。"
