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
│       ├── _env.sh                   # 共享环境设置
│       ├── 00_setup_env.sh           # clone 官方仓 + 装依赖 + 验证
│       ├── 01_*.sh                   # 下载权重 / 数据集构建
│       ├── 02_run_inference.sh       # 推理
│       ├── *.py                      # Python 脚本（被 .sh 调用）
│       ├── run_all.sh                # 一键全流程
│       └── README.md                 # 算法级 README
├── <official-repo>/                  # 官方代码（自动 clone，sibling of media_code）
├── ../../model/                      # 权重根（各算法共享，在 code-dir 上一级）
│   └── <algorithm>/                  # 各算法子目录
└── <algorithm>_results/              # 输出（repo 外）
```

### 命名规范
- 算法目录名：小写 + 下划线，如 `wan22_rotate`、`sam_3d_body`、`hunyuanvideo_1.5`
- 脚本编号：`00_setup_env.sh` → `01_download_models.sh` → `02_run_inference.sh` → `03_build_dataset.sh` → `04_train_lora.sh`
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

## 6. 依赖安装踩坑总结

### pip
- **公司代理封 download.pytorch.org（403）**：改用 PyPI 默认源（用户的 pip.conf 通常配了清华 PyPI 镜像）
- **SSL 证书验证失败**：`_env.sh` 设了 `PIP_CERT`，但代理根 CA 可能不在系统 bundle；先 `bash hypir/setup_ca_bundle.sh` 建 `~/.ca-bundle.crt`
- **手动下载 whl**：用户手动下 whl 放 `$MODEL_DIR/`，脚本里用 glob 匹配本地安装：
  ```bash
  for w in "$MODEL_DIR"/torch-*.whl "$MODEL_DIR"/nvidia_*.whl; do
      [ -f "$w" ] && LOCAL_WHEELS+=("$w")
  done
  pip install --force-reinstall --no-deps "${LOCAL_WHEELS[@]}"
  ```

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

## 7. 仓级 README.md

在 `media_code/README.md` 里，每个算法一行摘要：
```markdown
- [`<algo>/`](<algo>/) — 一句话描述。See its [README](<algo>/README.md).
```
新增算法时追加一行。

## 8. commit 规范

- 中文 commit message
- 格式：`<算法名>：简短描述`（如 `wan22_rotate：NUM_FRAMES 默认 121 -> 81`）
- 简洁平实，不要写论文（参考 git log --oneline 的风格）
- 一个逻辑改动一个 commit，不要混合多个不相关改动

## 9. 路径规范

| 路径 | 默认值 | 说明 |
|---|---|---|
| 代码仓 | `$REPO_DIR` = `media_code/` | 编排脚本在这里 |
| 官方代码 | `$REPO_DIR/../<official-repo>` | clone 到 media_code 的 sibling |
| 权重根 | `$REPO_DIR/../../model` | 在 code-dir 上一级，各算法共享 |
| 输出 | `$REPO_DIR/../<algo>_results` | 在 media_code 的 sibling |
| 训练产物 | `$REPO_DIR/../<algo>_experiments` | checkpoint / 日志 |
| proxy.env | `$REPO_DIR/proxy.env` | 代理密码，gitignored |

README 命令示例用**相对路径**（`../../model/...`、`../Reconstruction/...`），不要用绝对路径。

## 10. 注意事项

- **不要在脚本里写死绝对路径**，用 `${VAR:-default}` 允许覆盖
- **不要用 PowerShell 的 Set-Content 改文件**，会破坏 UTF-8 编码；用 `edit` 工具
- **`.gitattributes` 强制 LF**：Windows push 的 .sh / .py 在 Ubuntu 上跑没问题
- **`proxy.env` 不入库**：代理密码只在本地 `proxy.env`，gitignored
- **不复制官方代码**：只写编排脚本，clone 官方仓到外部
