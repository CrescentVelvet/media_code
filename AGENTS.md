# AGENTS.md — AI 代码编写规范（复现新论文时参考此文件）

> 本文件给 AI 看。用户让你复现新论文/算法时，按此规范建目录、写脚本、写 README。

## 1. 目录结构

```
<code-dir>/                           # e.g. /data_3d/<uid>/code
├── media_code/                       # 本仓（编排脚本，不含官方代码）
│   ├── AGENTS.md                     # ← 本文件
│   ├── .gitattributes                # 强制 LF（*.sh *.py）
│   ├── .gitignore                    # proxy.env / __pycache__ / *.pyc
│   ├── proxy.env.example             # 代理模板（用户复制为 proxy.env 填密码）
│   ├── README.md                     # 仓级 README（列出所有算法一行摘要）
│   └── <algorithm_name>/             # 每个算法一个子目录
│       ├── _env.sh                   # 共享环境设置（含 conda fallback，兼容 WSL）
│       ├── 00_setup_env.sh           # clone 官方仓 + 装依赖 + 验证（服务器）
│       ├── 00a_setup_env.sh          # WSL 变体（从零建 env，无 doll 依赖）
│       ├── 01_*.sh                   # 下载权重 / 数据集构建
│       ├── 02_run_inference.sh       # 推理
│       ├── *.py                      # Python 脚本（被 .sh 调用）
│       ├── 08_move_output.sh         # WSL 专用：训练完把结果从 Linux fs 搬到 Windows 盘
│       ├── run_all.sh                # 一键全流程
│       ├── README.md                 # 算法级 README（服务器）
│       └── README_wsl.md             # WSL 复现指南（与 README.md 并列，不删旧的）
├── <official-repo>/                  # 官方代码（自动 clone，sibling of media_code）
│   ├── 服务器: $REPO_DIR/../<repo>   # 与 media_code 同级
│   └── WSL: ~/repos/<repo>           # Linux fs（编译快，proxy.env 覆盖路径）
├── ../../model/                      # 权重根（各算法共享，在 code-dir 上一级）
│   └── <algorithm>/                  # 各算法子目录
│       ├── 服务器: $REPO_DIR/../../model/
│       └── WSL: ~/model/             # proxy.env 覆盖
└── <algorithm>_results/              # 输出（repo 外）
    ├── 服务器: $REPO_DIR/../<algo>_results
    └── WSL: ~/output/<algo>_results   # proxy.env 覆盖；跑完用 08 搬到 /mnt/d/output/
```

### 命名规范
- 算法目录名：小写 + 下划线，如 `wan22_rotate`、`sam_3d_body`、`hunyuanvideo_1.5`
- **脚本必须数字编号**：`00_setup_env.sh` → `01_download_models.sh` → `02_run_inference.sh` → `03_build_dataset.sh` → `04_train_lora.sh`。**不要写无编号的 `run_xxx.sh` / `stop_xxx.sh`**（如一键生成、停服务等 wrapper 也要编号：`04_run.sh`、`05_stop.sh`）。类型相似（同一步骤的多个变体）用同一数字编号 + 字母区分：`01_pick_and_segment.sh` / `01b_pick_and_segment.sh` / `01c_pick_and_segment.sh`。**跨平台变体也用字母区分**：`00_setup_env.sh`（服务器） / `00a_setup_env.sh`（WSL 本机）。
- **WSL 专用脚本也编号**：如 `08_move_output.sh`（训练完把结果从 Linux fs 搬到 Windows 盘），不要写无编号的 `move_output.sh`。
- Python 脚本：动词 + 名词，如 `run_inference.py`、`build_dataset.py`、`pick_and_segment.py`
- 官方仓目录名：与 GitHub repo 同名（去掉 `.git`），如 `sam-3d-body`、`DiffSynth-Studio`

## 2. _env.sh 模板

每个算法目录的 `_env.sh` 负责共享设置，被所有 `.sh` 脚本 source。内容：

```bash
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# 1. 代理（从 proxy.env 读）
if [ -f "$REPO_DIR/proxy.env" ]; then
    set -a; source "$REPO_DIR/proxy.env"; set +a
fi
[ -n "${http_proxy:-}" ]  && export HTTP_PROXY="$http_proxy"
[ -n "${https_proxy:-}" ] && export HTTPS_PROXY="$https_proxy"

# 2. CA bundle（公司代理 TLS 拦截）
SYS_CA=/etc/ssl/certs/ca-certificates.crt
USER_CA="$HOME/.ca-bundle.crt"
if [ -f "$USER_CA" ]; then CA_FILE="$USER_CA"
elif [ -f "$SYS_CA" ]; then CA_FILE="$SYS_CA"
else CA_FILE=""; fi
if [ -n "$CA_FILE" ]; then
    : "${REQUESTS_CA_BUNDLE:=$CA_FILE}"
    : "${SSL_CERT_FILE:=$CA_FILE}"
    : "${GIT_SSL_CAINFO:=$CA_FILE}"
    : "${PIP_CERT:=$CA_FILE}"
    export REQUESTS_CA_BUNDLE SSL_CERT_FILE GIT_SSL_CAINFO PIP_CERT
fi
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

# 3. conda env 激活
CONDA_ENV="${CONDA_ENV:-<algorithm_name>}"
export CONDA_ENV
# Fallback: conda 不在 PATH 时自动找常见安装位置（WSL 本机未 conda init 时触发）
# 服务器上 conda 已在 PATH，此块不会执行
if ! command -v conda >/dev/null 2>&1; then
    for _cb in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
        if [ -f "$_cb/etc/profile.d/conda.sh" ]; then
            source "$_cb/etc/profile.d/conda.sh"
            break
        fi
    done
    unset _cb
fi
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found on PATH (need env '$CONDA_ENV')." >&2
    echo "       Install miniconda or run: source ~/miniconda3/etc/profile.d/conda.sh" >&2
    exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" 2>/dev/null || true  # env 可能还没建

# 4. GPU 选卡
if [ -n "${GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

# 5. 路径（都用 ${VAR:-default} 允许外部覆盖）
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../../model}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_DIR/../<algorithm>_results}"
export REPO_DIR MODEL_DIR RESULTS_DIR
```

**关键规则：**
- 所有路径变量用 `${VAR:-default}`，允许命令行覆盖
- `conda activate` 加 `|| true`，env 不存在时不中断（00 负责创建）
- 不用 `set -e` / `set -u`（conda activate 会触发未绑定变量导致静默退出）

## 3. .sh 脚本规范

### shebang + set
```bash
#!/usr/bin/env bash
set -o pipefail
```
**不要用 `set -e`（出错静默退出看不到报错）和 `set -u`（conda activate 触发未绑定变量）。**

### 结构
```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

# 参数（全用 ${VAR:-default}）
export FOO="${FOO:-default}"

# 前置检查（文件存在、import 成功等）
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ ERROR: ..." >&2
    exit 1
fi

# 信息打印（带 emoji）
echo "🚀 [脚本名] 做什么"
echo "  📁 输入: ..."
echo "  💾 输出: ..."

# 执行
python "$SCRIPT_DIR/xxx.py"
if [ $? -ne 0 ]; then
    echo "❌ FAILED" >&2
    exit 1
fi
echo "🎉 Done."
```

### Emoji 用法
| Emoji | 用途 | 示例 |
|---|---|---|
| 🚀 | 脚本启动 | `echo "🚀 [02] 开始生成"` |
| 📦 | 安装/下载 | `echo "📦 installing ..."` |
| 🔍 | 检测/搜索 | `echo "🔍 detecting ..."` |
| ✂️ | 分割 | `echo "✂️ segmenting ..."` |
| 🎯 | 选图/结果 | `echo "🎯 picked: ..."` |
| ✅ | 成功 | `echo "✅ saved: ..."` |
| ❌ | 报错 | `echo "❌ ERROR: ..."` |
| ⚠️ | 警告 | `echo "⚠️ WARNING: ..."` |
| 🎮 | GPU | `echo "🎮 GPU: ..."` |
| 🎬 | 视频 | `echo "🎬 generation done"` |
| 🎉 | 完成 | `echo "🎉 Done."` |
| ⏱️ | 耗时 | `echo "⏱️ 12.3s"` |
| 🖼️ | 图像 | `echo "🖼️ input: ..."` |
| 🏋️ | 模型/权重 | `echo "🏋️ LoRA: ..."` |
| 🤖 | 模型名 | `echo "🤖 Wan2.2-TI2V-5B"` |
| 📐 | 分辨率 | `echo "📐 704x1248"` |
| ⏭️ | 跳过 | `echo "⏭️ skip"` |

## 4. .py 脚本规范

- 从 `os.environ` 读参数（和 .sh 的 export 对应）
- 用 `print()` 带 emoji 输出进度
- 错误用 `sys.exit("❌ ...")`
- **不要吞异常** — `except Exception as e: print(f"⚠️ failed: {e}")`，不要 `except: continue`
- 对未知 API 接口，先 `print(dir(obj))` introspect，贴给用户确认后再写正确调用
- **注释**：允许加**简洁注释**说明非显然逻辑（魔术常数、为何这么取、绕坑的坑点、易错陷阱），但别流水账复述代码做了什么。一行能说清就一行；复杂逻辑可多行。坑点注释要写清「为什么」而非「是什么」

```python
#!/usr/bin/env python3
"""一句话描述脚本做什么。

Env vars:
  VAR1, VAR2, ...
"""
import os, sys, time

VAR1 = os.environ.get("VAR1", "default")

def main():
    if not VAR1:
        sys.exit("❌ VAR1 not set")
    print(f"🚀 doing something with {VAR1}")
    # ...

if __name__ == "__main__":
    main()
```

## 5. README.md 格式

服务器和 WSL 各一个 README，**并列共存，不删旧的**：

```markdown
# 算法名 — 一句话描述

一段话概述：做什么、用什么官方代码、权重在哪。

## 常用命令
> 假设已进入容器；首次跑前先做下方「首次准备」。
> **铁律：每条命令都必须显式写出模型地址、输入路径、输出路径，不能全靠脚本里的默认值。** 用具体路径，不要用 `...` 占位。

\```bash
# 一键
GPU=0 INPUT_DIR=../data/subject_folder \
  WEIGHT_PATH=../../model/<algo>/step-N.safetensors \
  RESULTS_DIR=../../output/<algo>_results \
  bash <algo>/run_all.sh

# 分步
# 1) ...
GPU=0 INPUT_DIR=../data/subject_folder \
  RESULTS_DIR=../../output/<algo>_results \
  bash <algo>/01_xxx.sh
# 2) ...
GPU=0 MODEL_PATH=../../model/<algo>/model.safetensors \
  INPUT=../../output/<algo>_results/input_data \
  RESULTS_DIR=../../output/<algo>_results \
  bash <algo>/02_xxx.sh
\```

## 首次准备
\```bash
# clone 本仓 + proxy.env
cd <code-dir>
git clone ... && cd media_code && cp proxy.env.example proxy.env
# ⚠️ 确认 proxy.env 中 http_proxy / https_proxy 已取消注释

# 建 env + 装依赖
INSTALL_DEPS=1 bash <algo>/00_setup_env.sh

# 下权重
bash <algo>/01_download_models.sh
\```

权重目录布局：
\```
$MODEL_DIR/
  ...
\```

---

以下为详细参考。

## Pipeline（流程详解）
\```
输入 -> [01] 步骤 -> [02] 步骤 -> 输出
\```

## Config (env vars)
| var | default | note |
|---|---|---|
| ... | ... | ... |

## 可能遇到的问题
1. ...
2. ...

## 目录布局
\```
<code-dir>/
├── media_code/<algo>/
├── <official-repo>/
└── <algo>_results/
\```
```

### README_wsl.md 格式（WSL 本机复现指南）

与 `README.md` 并列，不删旧的。内容侧重 WSL 差异：

```markdown
# 算法名 (WSL Ubuntu 24.04) — 本地复现指南

本文件是 [`README.md`](README.md) 的 WSL 本地复现版。照着做即可从零跑完全流程。

## 与服务器版的核心差异
| | 服务器 | WSL |
|---|---|---|
| conda env | clone doll | conda create from scratch |
| CUDA toolkit | 系统 /usr/local/cuda | conda cuda-nvcc |
| pip 源 | 默认 PyPI | 清华镜像 |
| HF 源 | 直连 huggingface.co | hf-mirror.com 镜像 |
| 仓库位置 | /mnt/c/code/ | ~/repos/ (Linux fs) |

## Windows 路径 → WSL 路径
| Windows | WSL |
|---|---|
| D:\dataset\sample | /mnt/d/dataset/sample |

## 前提条件
（WSL + NVIDIA 驱动 + Miniconda）

## 首次准备
（00a_setup_env.sh → proxy.env 配置 → 权重下载）

## 全流程命令
（用 /mnt/d/ 路径读输入，~/output/ 写输出，08 搬运到 D 盘）

**格式规范**（照此写，可读性最高）：

\```bash
cd /mnt/c/code/media_code

# ── 0) 安装环境（首次，见上方「首次准备」）──

# ── 1) 步骤名（一句话说明做什么，括号补充前提/坑点）──
#    输入：源路径（说明）
#    输出：产物路径（说明）
#    可选开关：FOO=1 含义（默认 0）
GPU=0 \
INPUT_DIR=/mnt/d/dataset/sample/image \
RESULTS_DIR=~/output/<algo>_results \
bash <algo>/01_xxx.sh

# 输出：~/output/<algo>_results/<dir>/
#   file_a      # 说明
#   file_b      # 说明

# ── 2) 下一步（可选）──
#    DENOISER 可选: diffbir（质量高）| swinir（快）| none（跳过）
#    启用的步骤保留注释掉的命令，不要删（方便以后直接取消注释）
# GPU=0 INPUT_DIR=... \
#   bash <algo>/02_xxx.sh
\```

**要点**：
- 开头一条 `cd /mnt/c/code/media_code`，后续命令都是相对路径。
- 每步用 `# ── N) 标题 ──` 分隔（`──` 是 box-drawing 字符，不是 `--`）。标题写中文，括号内补前提或坑点。
- 命令前用注释写清**输入/输出/开关含义**，缩进对齐，一眼看出数据从哪来到哪去。
- 命令**一行一个 `VAR=value \`**，续行对齐；**所有关键路径显式写出**（输入、模型、输出），不靠脚本默认值。
- 命令后紧跟 `# 输出：` 块，逐行列出产物 + `#` 说明，让人不用跑就知道会生成什么。
- 可选步骤标题标「（可选）」，并在注释里列出各选项取值及适用场景。
- 暂时跳过的步骤**保留完整命令但注释掉**，并注明何时恢复。

## 可能遇到的问题（WSL 专属）
（conda not found、pip 慢、HF 连不上、OOM、vhdx 压缩）
```

## 6. 依赖安装踩坑总结

### pip
- **公司代理封 download.pytorch.org（403）**：改用 PyPI 默认源（用户的 pip.conf 通常配了清华 PyPI 镜像）
- **SSL 证书验证失败**：`_env.sh` 设了 `PIP_CERT`，但代理根 CA 可能不在系统 bundle；先 `bash hypir/setup_ca_bundle.sh` 建 `~/.ca-bundle.crt`
- **手动下载 whl → 统一放 `D:\wheel`（WSL: `/mnt/d/wheel`）**：本机网络 PyPI CDN 被限速到 ~8 KB/s，torch/nvidia 等大 wheel（~3GB）必须手动下载。**所有算法共用 `D:\wheel` 这个本地 wheel 缓存**——复现新算法前先查这里有没有现成 wheel，有就别重下。脚本里用 `--find-links /mnt/d/wheel` 让 pip 优先装本地、缺的再从镜像补：
  ```bash
  WHEELS_DIR="${WHEELS_DIR:-/mnt/d/wheel}"
  PIP_FLAGS=(-i https://mirrors.aliyun.com/pypi/simple --find-links "$WHEELS_DIR" --timeout 600)
  pip install "${PIP_FLAGS[@]}" -r requirements.txt
  ```
  下载清单写到 `<algo>/download_urls.md`（阿里源 URL，用户用 IDM/迅雷多线程下）。**清华源（pypi.tuna.tsinghua.edu.cn）会间歇性打不开，改用阿里源（mirrors.aliyun.com/pypi，~840 KB/s 可用）**；华为源（repo.huaweicloud.com）路径不通。wheel 的 `packages/<hash>/` 路径各镜像相同，只换域名即可。torch 的 nvidia-cu12 依赖版本从 `pip download -v torch==x.y` 的输出或 wheel 的 `.whl.metadata` 里拿（`Requires-Dist: nvidia-xxx-cu12==版本`）。
- **模型权重同理放 `D:\wheel`**：HF 在本机连不上（huggingface.co / hf-mirror.com 都不通），用 modelscope.cn 镜像（~230 KB/s 可用）或用户代理手动下。权重下到 `D:\wheel\<algo>_ms\`（按 HF repo 相对路径建子目录），安装脚本再拷到 `$MODEL_DIR`。

### conda
- **`gxx_linux-64` 会把 Python 从 CPython 降级成 GraalPy**：加 `python=3.10` 显式 pin，`--no-update-deps` 不够
  ```bash
  conda install -y -c conda-forge gxx_linux-64=12 python=3.10
  ```
  装完校验：`python -c "import platform; print(platform.python_implementation())"` 必须是 CPython

### git clone
- **`LD_LIBRARY_PATH` 导致系统 git 崩**（conda libffi 和系统 libp11-kit 冲突）：git 命令前加 `LD_LIBRARY_PATH=`
  ```bash
  LD_LIBRARY_PATH= git clone https://github.com/xxx/xxx.git "$DIR" || \
      LD_LIBRARY_PATH= git -c http.sslVerify=false clone ...
  ```
  手动 clone 后再 `pip install -e <local_dir>`，不用 `pip install 'git+https://...'`

### webfetch / curl 下载文件（查文档/源码）
- **webfetch 下载 `raw.githubusercontent.com` 报 `Transport error`**：URL 本身没问题，是公司代理 TLS 拦截导致 webfetch 的 HTTP 客户端连不上。**别反复重试 webfetch**（换 URL/加参数都没用），直接换 curl：
  ```bash
  # curl 默认也会 SSL 失败（代理根 CA 不在系统 bundle），加 --insecure 绕过
  curl.exe -sL --insecure --max-time 120 "https://raw.githubusercontent.com/xxx/xxx.md" -o "$env:TEMP\opencode\doc.md"
  # 然后用 Read 工具分析（别用 bash 的 Get-Content）
  ```
- **PowerShell 里用 `curl.exe` 不是 `curl`**：`curl` 是 `Invoke-WebRequest` 的别名，参数不兼容（`-sL --insecure` 会报错）。显式写 `curl.exe` 调真正的 curl。
- **下载到 `$env:TEMP\opencode\` 再 Read**：这个目录已预批准可访问，Read 工具能直接读。别下到工作区（会污染 git）。
- 临时目录路径：`$env:TEMP\opencode\`（PowerShell 展开），或 `C:\Users\<user>\AppData\Local\Temp\opencode\`（绝对路径，Read 工具用这个）。

### 版本 pin（最后装，覆盖依赖升级）
```bash
# numpy 2.x 和 detectron2 不兼容
pip install --force-reinstall --no-deps numpy==1.26.4
# setuptools>=70 去掉了 pkg_resources（detectron2 用）
pip install --force-reinstall --no-deps "setuptools<70"
```

### detectron2
- 用 `--no-build-isolation --no-deps` 装，避免 pin 冲突
- ViTDet 权重 `model_final_f05665.pkl` 运行时用 urllib 下载（SSL 被拦），预下载到 `$MODEL_DIR/ViTDet/`，设 `DETECTOR_PATH` 指向本地目录

### 模型加载
- **优先用 `ModelConfig(path=...)` 直接指定文件路径**，不走 `model_id` + `DIFFSYNTH_MODEL_BASE_PATH` + `Wan-AI` 符号链接
- 用户手动下载的模型在 `$MODEL_DIR/<model_name>/`，直接指向这个目录

### WSL 本机复现
- **conda ToS（conda 26.x）**：conda create 报 `CondaToSNonInteractiveError`。00a 自动 accept：
  ```bash
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
  ```
- **pip 慢（国内 PyPI）**：用清华镜像 `-i https://pypi.tuna.tsinghua.edu.cn/simple`
- **huggingface.co 连不上**：用 HF 镜像 `export HF_ENDPOINT=https://hf-mirror.com`（在 proxy.env 里设）
- **无 sudo（WSL 本机）**：CUDA toolkit + gcc 全走 conda（`cuda-nvcc` + `gxx_linux-64=12`），不装 `cuda-toolkit` 全包（太大）
  ```bash
  # 精简装 nvcc（~50MB，不装全包 ~2GB）
  conda install -y -c nvidia/label/cuda-12.1.1 cuda-nvcc cuda-cudart-dev cuda-cccl
  ```
- **CUDA 扩展编译必须放 Linux fs**：`/mnt/c` `/mnt/d` 是 drvfs（9p），symlink 会坏、I/O 慢 5-10x。仓库 clone 到 `~/repos/`，通过 `proxy.env` 覆盖路径
- **Windows 路径映射**：`D:\dataset` → `/mnt/d/dataset`（反斜杠改正斜杠），输入用 `/mnt/d/` 路径，输出写 `~/output/`（Linux fs），跑完用 `08_move_output.sh` 搬到 D 盘
- **WSL vhdx 空间有限**：训练完跑 `08_move_output.sh` 把结果剪切到 `/mnt/d/output/`；vhdx 不缩小时在 PowerShell 里 `wsl --shutdown` + `diskpart compact vdisk`
- **pip 优先从 `D:\wheel` 装本地 wheel**：所有算法共用 `/mnt/d/wheel` 这个 wheel 缓存。`00a_setup_env.sh` 的 pip install 必须加 `--find-links /mnt/d/wheel`，让 pip 先查本地、缺的再从清华镜像补。conda 包也缓存在这里（`.tar.bz2` / `.conda`），conda install 加 `--offline` 优先用缓存：
  ```bash
  WHEELS_DIR="${WHEELS_DIR:-/mnt/d/wheel}"
  # pip: 先查本地 wheel，缺的从清华镜像补
  PIP_FLAGS=(-i https://pypi.tuna.tsinghua.edu.cn/simple --find-links "$WHEELS_DIR" --timeout 600 --retries 5)
  pip install "${PIP_FLAGS[@]}" package1 package2 ...
  # conda: 优先用本地缓存的包（~/.cache/conda 或 /mnt/d/wheel 里的 .conda）
  conda install -y --offline -c conda-forge gxx_linux-64=12 python=3.10
  ```
- **pip cache 定期清理**：pip 的 HTTP 缓存存在 `~/.cache/pip`（Linux fs，吃 vhdx 空间）。装完环境后跑 `pip cache purge` 清掉，释放 vhdx 空间。需要保留的 wheel 用 `pip download -d /mnt/d/wheel` 提取到 D 盘再清缓存

## 7. 仓级 README.md

在 `media_code/README.md` 里，每个算法一行摘要：
```markdown
- [`<algo>/`](<algo>/) — 一句话描述。See its [README](<algo>/README.md).
```
新增算法时追加一行。

## 8. commit 规范

- 中文 commit message
- 格式：`<算法名>：简短描述`（如 `wan22_rotate：NUM_FRAMES 默认 121 -> 81`）
- **默认只写标题行，不写正文**；确有必要时正文最多 1-2 行
- 简洁平实，不要写论文（参考 git log --oneline 的风格）
- 一个逻辑改动一个 commit，不要混合多个不相关改动
- **多行 message 用 `git commit -F -` 接 heredoc**：`-m "标题\n正文"` 里的 `\n`
  不会展开，会原样存成字面量 `\n`

## 9. 路径规范

### 服务器
| 路径 | 默认值 | 说明 |
|---|---|---|
| 代码仓 | `$REPO_DIR` = `media_code/` | 编排脚本在这里 |
| 官方代码 | `$REPO_DIR/../<official-repo>` | clone 到 media_code 的 sibling |
| 权重根 | `$REPO_DIR/../../model` | 在 code-dir 上一级，各算法共享 |
| 输出 | `$REPO_DIR/../<algo>_results` | 在 media_code 的 sibling |
| 训练产物 | `$REPO_DIR/../<algo>_experiments` | checkpoint / 日志 |
| proxy.env | `$REPO_DIR/proxy.env` | 代理密码，gitignored |

### WSL 本机（proxy.env 覆盖默认值）
| 路径 | WSL 路径 | 说明 |
|---|---|---|
| 官方代码 | `~/repos/<repo>` | Linux fs，编译快 |
| 权重根 | `~/model/<algo>` | Linux fs，读大文件快 |
| 输出 | `~/output/<algo>_results` | Linux fs，训练写文件快 |
| 最终结果 | `/mnt/d/output/<algo>_results` | Windows 盘，08 搬运后存这里 |
| 输入数据 | `/mnt/d/dataset/...` | Windows 盘，只读无所谓慢 |

README 命令示例用**相对路径**（`../../model/...`、`../Reconstruction/...`），不要用绝对路径。WSL README 用 `~/` 和 `/mnt/d/` 路径。

## 10. 注意事项

- **不要在脚本里写死绝对路径**，用 `${VAR:-default}` 允许覆盖
- **不要用 PowerShell 的 Set-Content 改文件**，会破坏 UTF-8 编码；用 `edit` 工具
- **`.gitattributes` 强制 LF**：Windows push 的 .sh / .py 在 Ubuntu 上跑没问题
- **`proxy.env` 不入库**：代理密码只在本地 `proxy.env`，gitignored
- **不复制官方代码**：只写编排脚本，clone 官方仓到外部
- **WSL 与服务器脚本共用**：01-08 的 .sh 和 .py 不区分平台，靠 `_env.sh` + `proxy.env` 覆盖路径。只有 setup 脚本分 `00_setup_env.sh`（服务器）和 `00a_setup_env.sh`（WSL）
- **WSL 路径覆盖写 proxy.env**：`00a_setup_env.sh` 自动生成 `proxy.env`（含 `VGGT_DIR=~/repos/...` 等路径覆盖 + `HF_ENDPOINT` 镜像），脚本 01-08 通过 `_env.sh` source proxy.env 自动继承，不需改脚本
- **`_env.sh` 加 conda fallback**：conda 不在 PATH 时自动找 `~/miniconda3`。服务器上 conda 已在 PATH，fallback 不触发，行为不变

## 11. git 远端与破坏性操作纪律

### 远端：只允许 pull / fetch，禁止 push

- AI **不得**执行 `git push`（含 `--force`）、删远端分支、改远端 tag
- 只允许 `git fetch` / `git pull` / `git ls-remote` 等只读操作
- **push 由用户手动执行**

**理由**：push 是唯一能改动远端状态的 git 操作。让远端始终停在"用户手动确认过的
干净快照"，本地出事时能整库拉回。2026-09-04 本机 `.git/objects/pack` 被中断的
gc 撕掉，533 条历史全靠 `git fetch origin` 从远端恢复，只丢了最后 2 条未推的提交。

**前提（需用户配合）**：这条纪律的恢复能力 = 最后一次 push 的新鲜度。
本地未推的提交一旦对象库损毁就永久丢失。**建议在跑完一个可验证的阶段后及时 push**，
不要攒几天。

### 本地：禁止会被中断撕裂仓库的写操作

- **不要在同一条命令里串多个 git 写操作**，尤其是 `git stash` / `git stash pop` /
  `git commit --amend` / `git reset`。Bash 工具可能被 SIGTERM 打断，
  断在半途会撕裂 `.git/refs/` 或对象库
- **不要顺手用 `git commit --amend`**：确需改上一条 commit message 时，
  先 `git log --oneline -1` 确认 HEAD 是哪条，再单条执行
- **不要执行** `git gc` / `git prune` / `git repack` / `git clean -fd` /
  `git reset --hard`
- git 写操作一律**单条执行**，并加
  `-c gc.auto=0 -c maintenance.auto=false`
  （`git commit` 会自动触发 gc，此时被中断即损坏对象库）
- 动 `.git/` 之前先 `cp -a .git <仓库外的备份路径>`

## 12. 人脸 finetune 消融结论（2026-09-04，3090 实测）

对照实验（同等训练量 20k 步 @ lr_scale=0.2，唯一差异 face_weight 0 vs 1.0）证明：
**人脸监督（masked-L1 → HYPIR 增强图）对人脸区锐度无可测量贡献**（+4.04% vs +3.98%，
逐帧差在噪声内）。此前 v1~v6 的全部人脸区提升来自"30k 基线欠收敛 + 更高 lr 继续训练"。

- **推荐配方**：`06d_continue_train.sh`（ITERATION=30000, EXTRA_ITERS=20000,
  LR_SCALE=0.2, FACE_WEIGHT=0）。训练视角人脸区 Laplacian +4~7%，全图 PSNR
  27.32→27.44+，LPIPS 同步下降，无过平滑代价
- **权重已饱和**：face_weight 1.0→2.0 结果逐像素持平（瓶颈是 lr_scale 不是权重）
- **lr_scale 是瓶颈**：0.1→0.2 把增益从 +2.0% 解锁到 +2.7%/10k 步
- **增益曲线**：~+1.3%/万步递减到 ~+0.8%/万步（80k 处未完全收敛）
- **新视角注意**：常规（人物居中）新视角明显更清晰且无伪影；极端离轴暗部视角
  纤维状条纹伪影增多，此类场景用较少续训步数（50k~60k）折中
- 若要超越 GT 收敛的人脸细节，masked-L1 机制无效，需换机制（对抗损失等），未验证

### 06e：增强近景整视角注入（2026-09-04，有效，与 masked-L1 机制相反）

把 06c 的 512 增强近景作为**完整训练视角**（真实相机位姿 + COLMAP 注入）喂回续训，
与 masked-L1（增强图仅作人脸区 GT）机制不同，**实测有效**（ms 数据集，4 人 × 18 近景）：

- 归因对照（同 20k 步 @ lr_scale=0.2）：
  - 06f 纯续训无注入：近景 LPIPS 持平（0.3048→0.3049 等）、锐度 +3~6%（噪声级）
  - **06e 注入**：近景 LPIPS 全降（p00 0.305→0.260、p03 0.202→0.137），
    锐度 +28%~+324%；训练视角人脸区 Laplacian +45% 且 LPIPS vs 原图不升（无退化）
- 结论：masked-L1 无效 ≠ 增强图无用；**整视角注入（整图参与 loss）才是正确用法**
- 链路：`inject_closeup_cameras.py`（pycolmap 4.2 `add_camera_with_trivial_rig` +
  `add_image_with_trivial_frame`，pose json R/T 即 COLMAP w2c）→ 场景 06e_source_closeup
  （222 views）→ 06d 配方续训
- 坑：start_ply 的 exposure.json 只有原图条目，注入图名在 Scene.save()/testing 会
  KeyError —— trainer 已补恒等 exposure fallback（30k exposure 本就收敛为恒等阵）
- p03（此前疑 p00 重复轨迹）实为真实第 4 人（红衣），近景提升最大，无需排除

### 06d 脚本注意

- `train_face_finetune.py` 的 `FaceData` 只扫 `face_images_dir` **直接子文件**，
  目录参数必须指到图片平铺层（如 `06c_merged_face_images/images`，不是其父目录）
- `render_novel.py` 只读 txt 格式 COLMAP 模型；BA 场景（03b）sparse/0 只有 bin，
  已加 pycolmap 自动补 txt 的 fallback
