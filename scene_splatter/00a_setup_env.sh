#!/usr/bin/env bash
# Scene Splatter WSL 本机构建 env（无 sudo，全部走 conda + pip）
#
# 官方依赖（README）：python 3.10 + torch 2.1.2/cu118 + requirements.txt +
# 编译 gaussianSplatting/submodules/diff-gaussian-rasterization（需要 nvcc 11.8）。
# ⚠️ 2026-11-09 网络漂移：repo.anaconda.com / download.pytorch.org / nvidia conda 通道
#    全部不可达，实际方案（详见 README_wsl.md 网络备忘）：
#    conda 主通道=NJU 镜像；torch=阿里云 pytorch-wheels/cu118；pip=阿里镜像；
#    nvcc/gcc 编译期借用 flame_human env 的 11.8 工具链（TOOLCHAIN_ENV 可换）。
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

INSTALL_DEPS="${INSTALL_DEPS:-1}"
PY_VER="${PY_VER:-3.10}"
PIP_INDEX="${PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple}"
WHEELS_DIR="${WHEELS_DIR:-/mnt/d/wheel}"

echo "🚀 [00a] Scene Splatter WSL env setup"
echo "  📁 官方仓: $SS_DIR"
echo "  🐍 conda:  $CONDA_ENV ($PY_VER)"
echo "  📦 pip:    $PIP_INDEX"
echo "  🎮 GPU:    ${CUDA_VISIBLE_DEVICES:-all}"

# ── 1) clone 官方仓（GitHub 本机直连可用；仓内 gaussianSplatting/flash3d/viewcrafter
#       全是 vendored 普通目录，无 submodule，plain clone 即可）──
if [ ! -f "$SS_DIR/scenesplatter.py" ]; then
    if [ -d "$SS_DIR" ]; then
        echo "❌ $SS_DIR 已存在但缺 scenesplatter.py（残留半成品，先人工处理）" >&2; exit 1
    fi
    echo "📦 cloning Scene-Splatter → $SS_DIR"
    mkdir -p "$(dirname "$SS_DIR")"
    LD_LIBRARY_PATH= git clone https://github.com/shengjun-zhang/Scene-Splatter "$SS_DIR" || {
        echo "❌ clone failed" >&2; exit 1; }
else
    echo "⏭️  repo exists: $SS_DIR"
fi

# ── 2) conda env ──
# ⚠️ 2026-11-09 网络漂移：repo.anaconda.com 与 download.pytorch.org 均超时（直连与代理都死）。
#    conda 主通道走 NJU 镜像（USTC 已停 anaconda 镜像并 302 到 NJU），pip/torch 走阿里镜像。
CONDA_MAIN_CHANNEL="${CONDA_MAIN_CHANNEL:-https://mirrors.nju.edu.cn/anaconda/pkgs/main}"
if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    echo "📦 creating conda env $CONDA_ENV (python=$PY_VER, channel=$CONDA_MAIN_CHANNEL)"
    conda create -n "$CONDA_ENV" python="$PY_VER" --override-channels -c "$CONDA_MAIN_CHANNEL" -y || {
        echo "⚠️  NJU create failed, retry with tuna"
        conda create -n "$CONDA_ENV" python="$PY_VER" --override-channels \
            -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main -y || {
        echo "❌ conda create failed" >&2; exit 1; }
    }
else
    echo "⏭️  env exists: $CONDA_ENV"
fi

if [ "$INSTALL_DEPS" != "1" ]; then
    echo "⏭️  INSTALL_DEPS=0, skip pip install"
    echo "🎉 Done (env only)."
    exit 0
fi

PY="$HOME/miniconda3/envs/$CONDA_ENV/bin/python"
PIP_FLAGS=(-i "$PIP_INDEX" --timeout 600 --retries 5)
[ -d "$WHEELS_DIR" ] && PIP_FLAGS+=(--find-links "$WHEELS_DIR")

# ── 3) torch 2.1.2 + torchvision 0.16.2（官方指定 cu118）──
# 2026-11-09 实测：pytorch.org 超时死；阿里云 pytorch-wheels 的索引页非 PEP503
#（pip index-url 找不到包），但精确 wheel URL 可下（torch 2.3GB / torchvision 7MB）。
# 方案：curl 精确 URL 落 /mnt/d/wheel（复用池），pip 本地文件直装。
TORCH_WHEEL="${TORCH_WHEEL:-$WHEELS_DIR/torch-2.1.2+cu118-cp310-cp310-linux_x86_64.whl}"
TV_WHEEL="${TV_WHEEL:-$WHEELS_DIR/torchvision-0.16.2+cu118-cp310-cp310-linux_x86_64.whl}"
TORCH_WHEEL_URL="${TORCH_WHEEL_URL:-https://mirrors.aliyun.com/pytorch-wheels/cu118/torch-2.1.2%2Bcu118-cp310-cp310-linux_x86_64.whl}"
TV_WHEEL_URL="${TV_WHEEL_URL:-https://mirrors.aliyun.com/pytorch-wheels/cu118/torchvision-0.16.2%2Bcu118-cp310-cp310-linux_x86_64.whl}"
if [ ! -s "$TORCH_WHEEL" ]; then
    echo "⬇️  downloading torch wheel → $TORCH_WHEEL"
    curl -sL --retry 3 -o "$TORCH_WHEEL" "$TORCH_WHEEL_URL" || {
        echo "❌ torch wheel download failed" >&2; exit 1; }
fi
if [ ! -s "$TV_WHEEL" ]; then
    echo "⬇️  downloading torchvision wheel → $TV_WHEEL"
    curl -sL --retry 3 -o "$TV_WHEEL" "$TV_WHEEL_URL" || {
        echo "❌ torchvision wheel download failed" >&2; exit 1; }
fi
echo "📦 installing torch/torchvision from local wheels"
"$PY" -m pip install "$TORCH_WHEEL" "$TV_WHEEL" || {
    echo "❌ torch install failed" >&2; exit 1; }

# ── 4) requirements.txt（去 torch 三件套，官方 README 单独装）──
# ⚠️ requirements 里 pip==23.0.1 会降级 pip，属官方原样行为，不动它。
# ⚠️ av==10.0.0 PyPI 只有 sdist（需 pkg-config+ffmpeg 头，无 sudo 编不了）——但 av 是
#    主路径依赖（viewcrafterpipe.py save_video 走 torchvision.io.write_video → PyAV），
#    不能剔除，换成有 cp310 wheel 的 av==12.1.0（首跑实测 torchvision 0.16.2 兼容）。
# ⚠️ scipy==1.9.1 要求 numpy<1.25，与 numpy==1.26.0 互斥（官方 requirements 自相矛盾，
#    当年大概靠旧 resolver 硬装）。scipy 1.9.1 的 wheel 编译期 ABI 虽兼容 numpy 1.26
#    （numpy 2.0 才断 ABI），但让 pip 解析通过需放宽——升 scipy==1.11.4（有 cp310
#    wheel，sklearn 1.2.2 / skimage 0.20 的 scipy 下限都远低于它）。
echo "📦 installing requirements.txt"
REQ_FILTERED="$(mktemp)"
sed -e 's/^av==10\.0\.0/av==12.1.0/' -e 's/^scipy==1\.9\.1/scipy==1.11.4/' \
    "$SS_DIR/requirements.txt" > "$REQ_FILTERED"
"$PY" -m pip install "${PIP_FLAGS[@]}" -r "$REQ_FILTERED" || {
    echo "❌ requirements install failed" >&2; exit 1; }
rm -f "$REQ_FILTERED"

# ── 4b) 官方 requirements 漏写/钉错的运行期依赖（2026-11-09 首跑实测）──
# diffusers：scenesplatter.py 直接 `from diffusers.utils import ...`，requirements.txt 没写；
# timm：requirements 装出 0.6.13，没有 timm.layers（flash3d UniDepth ConvNeXt 要 ≥0.9）→ 0.9.16；
# h5py / ema-pytorch：dust3r / flash3d 运行期 import；
# pytorch3d：scenesplatter.py L18 import。pip 无 wheel，官方 conda 包在 NJU cloud 有
#   py310_cu118_pyt212 精确匹配版，但依赖 pytorch-cuda metapackage（nvidia 通道已墙）
#   且 libmamba 不认 --no-deps——手工下载 tar.bz2 用 python tarfile 解到 env 根（WSL 无 bzip2）。
echo "📦 installing missing runtime deps (first-run findings)"
# nvidia-cuda-runtime/cublas-cu12：PyPI 版 xformers 全是 cu121 构建，torch 是 cu118，
# 缺的 libcudart.so.12/libcublas.so.12 由这两个 pip 包补齐（LD 路径在 _env.sh 挂）
"$PY" -m pip install "${PIP_FLAGS[@]}" "diffusers==0.24.0" "timm==0.9.16" h5py ema-pytorch \
    nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 || {
    echo "❌ runtime deps install failed" >&2; exit 1; }

if ! (cd /tmp && "$PY" -c "import pytorch3d" 2>/dev/null); then
    echo "📦 pytorch3d 0.7.7 (manual untar from NJU cloud)"
    P3D_URL="https://mirrors.nju.edu.cn/anaconda/cloud/pytorch3d/linux-64/pytorch3d-0.7.7-py310_cu118_pyt212.tar.bz2"
    P3D_TARBALL="$(mktemp --suffix=.tar.bz2)"
    curl -sL --retry 3 -o "$P3D_TARBALL" "$P3D_URL" || {
        echo "❌ pytorch3d download failed" >&2; exit 1; }
    "$PY" -c "import tarfile,sys; tarfile.open(sys.argv[1]).extractall(sys.argv[2])" \
        "$P3D_TARBALL" "$CONDA_PREFIX" || { echo "❌ pytorch3d untar failed" >&2; exit 1; }
    rm -f "$P3D_TARBALL"
    (cd /tmp && "$PY" -c "import pytorch3d") || {
        echo "❌ pytorch3d import failed after untar" >&2; exit 1; }
fi

# ── 5) nvcc 11.8 工具链（编译 diff-gaussian-rasterization 用）──
# nvidia conda 通道全被墙（tuna/NJU/BFSU 无 nvidia cloud，SJTU 跳 jcloud 403），
# pip 的 nvcc 又没有 gcc。借 flame_human env 的完整 11.8 工具链（nvcc+gcc11.4+
# cuda_runtime.h+crt+libcudart 已实测齐全），仅编译期借用 PATH/CUDA_HOME，不装任何包。
TOOLCHAIN_ENV="${TOOLCHAIN_ENV:-flame_human}"
TC="$HOME/miniconda3/envs/$TOOLCHAIN_ENV"
if [ ! -x "$TC/bin/nvcc" ] || [ ! -f "$TC/include/cuda_runtime.h" ]; then
    echo "❌ toolchain env $TOOLCHAIN_ENV 缺 nvcc 或 cuda_runtime.h（nvcc=[$TC/bin/nvcc]）" >&2
    echo "   换一个含 nvcc 11.8 的 env：TOOLCHAIN_ENV=<env> bash $0" >&2
    exit 1
fi
echo "🔧 borrowing toolchain: $TC (nvcc $("$TC/bin/nvcc" --version | grep -oE 'release [0-9.]+' | head -1))"
export CUDA_HOME="$TC"
export PATH="$TC/bin:$PATH"
export LD_LIBRARY_PATH="$TC/lib:${LD_LIBRARY_PATH:-}"
gcc --version >/dev/null 2>&1 || { echo "❌ no gcc in toolchain env" >&2; exit 1; }

# torch 2.1 cpp_extension 依赖 pkg_resources，而 conda 默认 setuptools 84.x 已移除
# 该模块（上游 2025 后彻底删除）——降到 <81（如 80.9.0，仍随 pkg_resources 分发）。
"$PY" -c "import pkg_resources" 2>/dev/null || {
    echo "📦 downgrading setuptools (restore pkg_resources for torch cpp_extension)"
    "$PY" -m pip install "setuptools<81" -i "$PIP_INDEX" || {
        echo "❌ setuptools downgrade failed" >&2; exit 1; }
}

# ── 6) 编译 diff-gaussian-rasterization + simple-knn（3DGS 两个 CUDA 扩展）──
# ⚠️ 判活必须 cd 到 /tmp 再 import：源码目录自带同名包，在 submodule cwd 里
#    `import x` / `pip show` 都会被源码树假阳性遮蔽（pip show 曾实测在脚本里误报）。
echo "🔨 building diff-gaussian-rasterization"
cd "$SS_DIR/gaussianSplatting/submodules/diff-gaussian-rasterization" || {
    echo "❌ submodule dir missing" >&2; exit 1; }
if ! (cd /tmp && "$PY" -c "import diff_gaussian_rasterization" 2>/dev/null); then
    "$PY" -m pip install . || { echo "❌ rasterizer build failed" >&2; exit 1; }
else
    echo "⏭️  diff_gaussian_rasterization already installed"
fi
# gaussian_model.py 还依赖 simple_knn._C（3DGS 标准邻居扩展，官方 README 没提）
echo "🔨 building simple-knn"
if [ -d "$SS_DIR/gaussianSplatting/submodules/simple-knn" ]; then
    cd "$SS_DIR/gaussianSplatting/submodules/simple-knn"
    if ! (cd /tmp && "$PY" -c "import simple_knn._C" 2>/dev/null); then
        "$PY" -m pip install . || { echo "❌ simple-knn build failed" >&2; exit 1; }
    else
        echo "⏭️  simple_knn already installed"
    fi
fi
cd "$SS_DIR"

# ── 7) 打本仓补丁（官方 main() 不落盘 PLY，补 scene.ply 导出）──
echo "📦 applying media_code patches"
python3 "$SCRIPT_DIR/patch_save_ply.py" || true

# ── 8) 验证 ──
echo "🔍 verifying imports"
cd "$SS_DIR" || exit 1
"$PY" - <<'EOF'
import torch
print(f"✅ torch {torch.__version__} cuda={torch.cuda.is_available()} "
      f"dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
from gaussianSplatting.scene import GaussianModel
from gaussianSplatting.gaussian_renderer import render
print("✅ gaussianSplatting import ok")
import flash3d  # noqa: F401
print("✅ flash3d import ok")
EOF
if [ $? -ne 0 ]; then
    echo "❌ import check failed" >&2
    exit 1
fi

echo "🔍 官方示例相机轨迹结构（核对约定用，详见 NOTES.md 第 4 节）"
python3 "$SCRIPT_DIR/02_gen_traj.py" --inspect-example || true

echo "🎉 Done. 下一步：bash scene_splatter/01_download_models.sh"
