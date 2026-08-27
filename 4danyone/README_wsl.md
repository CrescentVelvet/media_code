# 4DAnyone (WSL Ubuntu 24.04) — 本地复现指南

本文件是 [`README.md`](README.md) 的 WSL 本地复现版。照着做即可从零装好环境 + 权重。

> ⚠️ **推理需 ~43 GiB 显存**，RTX 3090 (24 GiB) 跑不了——本机只做**环境 + 权重准备**，推理放服务器（≥48 GiB 卡）。`torch.cuda.is_available()` 能 True（WSL GPU passthrough 正常），但 02 推理会 OOM。

## 与服务器版的核心差异

| | 服务器 | WSL 本机 |
|---|---|---|
| github.com | 直连 clone | 不通（WSL/Windows git 都 Connection reset）→ **codeload.github.com 下 zip** |
| PyPI wheel | 直连 | files.pythonhosted 8 KB/s 极慢 → **阿里源 + D:\wheel 本地 wheel 缓存** |
| 清华源 | — | 间歇性打不开 → **阿里源** (`mirrors.aliyun.com/pypi`, ~840 KB/s) |
| HuggingFace | 直连 | huggingface.co / hf-mirror.com 都不通 → **modelscope.cn** 镜像 |
| torch 安装 | `pip install` | **`--no-index --find-links`** 强制本地 wheel（`-i index` 不用本地会重下 888 MB） |
| DPVO 子模块 | 装上 | 不装（4DAnyone 设 `use_dpvo=false`） |

## Windows 路径 → WSL 路径

| Windows | WSL |
|---|---|
| `C:\code\media_code` | `/mnt/c/code/media_code` |
| `C:\code\4DAnyone` | `/mnt/c/code/4DAnyone` |
| `D:\wheel` | `/mnt/d/wheel` |

## 前提条件

- WSL2 Ubuntu 24.04（`wsl -l -v` 显示 Ubuntu2404 Running）
- NVIDIA 驱动 + WSL GPU passthrough（`nvidia-smi` 在 Windows 能看到 RTX 3090）
- Miniconda3 装在 `~/miniconda3`（没装见下方「装 Miniconda」）

## 首次准备

### 0. 装 Miniconda（如已有可跳过）

```bash
# WSL 里
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/mc.sh
bash /tmp/mc.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

### 1. 获取代码（github 不通，用 codeload zip）

github.com 直连失败，但 **codeload.github.com**（zip 下载）快（5-10 MB/s）。在 Windows PowerShell 下载 zip + .NET 解压（PowerShell `Expand-Archive` 对 Linux 生成的 zip 会坏）：

```powershell
# 下载 4DAnyone + GVHMR zip（共 ~27 MB，几秒）
curl.exe -sL --insecure --retry 3 --max-time 300 "https://codeload.github.com/ant-research/4DAnyone/zip/refs/heads/main" -o "$env:TEMP\4d.zip"
curl.exe -sL --insecure --retry 3 --max-time 300 "https://codeload.github.com/zju3dv/GVHMR/zip/refs/heads/main" -o "$env:TEMP\gvhmr.zip"

# .NET 解压（Expand-Archive 会报"尾部记录"错误）
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::ExtractToDirectory("$env:TEMP\4d.zip", "C:\code\_4d")
[System.IO.Compression.ZipFile]::ExtractToDirectory("$env:TEMP\gvhmr.zip", "C:\code\_gvhmr")

# 组装：4DAnyone-main -> C:\code\4DAnyone，GVHMR-main -> third_party\GVHMR
Remove-Item -Recurse -Force "C:\code\4DAnyone" -ErrorAction SilentlyContinue
Move-Item "C:\code\_4d\4DAnyone-main" "C:\code\4DAnyone"
Remove-Item -Recurse -Force "C:\code\4DAnyone\third_party\GVHMR" -ErrorAction SilentlyContinue
Move-Item "C:\code\_gvhmr\GVHMR-main" "C:\code\4DAnyone\third_party\GVHMR"
Remove-Item -Recurse -Force "C:\code\_4d","C:\code\_gvhmr"
```

### 2. 下载 wheel + 模型到 `D:\wheel`

按 [`4danyone/download_urls.md`](download_urls.md) 清单，用 **IDM/迅雷多线程**下载（突破单连接限速）：

| 部分 | 内容 | 大小 | 保存到 |
|---|---|---|---|
| A | torch 2.8 + torchvision + 14 nvidia-cu12 + triton | ~3 GB | `D:\wheel\` |
| B | 4DAnyone 模型权重 (modelscope) | ~10-15 GB | `D:\wheel\4danyone_ms\<相对路径>\` |
| C | BiRefNet 前景模型 (4 文件) | ~423 MB | `D:\wheel\birefnet\` |
| D | SMPL-X (`models_smplx_v1_1.zip`) | ~50 MB | `D:\wheel\` |

> A 用**阿里源** URL（清华源打不开）；B 用 **modelscope.cn**（HF 不通）。最大文件 `4danyone/models_t5_umt5-xxl-enc-bf16.pth`（~11 GB）。SMPL-X 需先在 https://smpl-x.is.tue.mpg.de/ 注册 + 接受许可。

### 3. 建 conda env + 装依赖（00a）

```bash
cd /mnt/c/code/media_code
INSTALL_DEPS=1 bash 4danyone/00a_setup_env.sh
```

00a 分步装（已规避 pip 不用本地 wheel 的坑）：
1. **[3a]** torch 的小依赖（filelock/sympy/networkx/jinja2/fsspec）从阿里源（几十 MB，快）
2. **[3b]** torch 2.8 + triton + 14 nvidia-cu12 用 `--no-index` 从本地 wheel **秒装**（不下载）
3. **[3c]** numpy + pillow 从阿里源
4. **[3d]** torchvision 0.23 用 `--no-index` 从本地 wheel 秒装
5. **[3e]** 4DAnyone requirements 从阿里源（--find-links 复用本地 wheel）
6. **[3f]** numba（lapx 运行时缺它，setup.py 没声明）
7. 验证 `torch.cuda.is_available()` → True（RTX 3090 24 GB）

### 4. 装模型权重（01）

```bash
cd /mnt/c/code/media_code
FDANYONE_MODEL_DIR=/mnt/d/wheel/4danyone_ms bash 4danyone/01_download_models.sh
```

- `MODEL_DIR` 指向 `D:\wheel\4danyone_ms`（**in-place，不拷 25 GB 大模型**）
- BiRefNet（423 MB）拷到 `4danyone_ms\birefnet\`
- 调官方 `download_model.py` 建 GVHMR 兼容 symlink
- SMPL-X 若已下 zip，加 `SMPLX_ARCHIVE=/mnt/d/wheel/models_smplx_v1_1.zip`

### 5. SMPL-X（若步骤4 没装）

```bash
SMPLX_ARCHIVE=/mnt/d/wheel/models_smplx_v1_1.zip \
  FDANYONE_MODEL_DIR=/mnt/d/wheel/4danyone_ms \
  bash /mnt/c/code/media_code/4danyone/01_download_models.sh
```

## 推理（放服务器）

本机 RTX 3090 (24 GiB) 跑不了（6-view 最低 43 GiB）。把 `D:\wheel\4danyone_ms` 权重拷到服务器，在服务器用服务器版 `00_setup_env.sh` + `02_run_inference.sh` 跑。

## WSL 专属问题

1. **github.com 不通**：WSL git 连不上 github（Connection reset）。用 codeload.github.com 下 zip（步骤 1）。GVHMR 的嵌套子模块 DPVO 不用（4DAnyone `use_dpvo=false`）。
2. **pip 不用本地 wheel**：`pip install -i 阿里源 --find-links /mnt/d/wheel torch` 仍从阿里源下载 888 MB（不用本地，验证过）。**必须 `--no-index --find-links` 强制本地**——00a 已分步处理（小依赖阿里源，大 wheel --no-index）。
3. **lapx → `lap` 模块**：lapx 装成 `lap` 模块（`import lap`，不是 `import lapx`）。4DAnyone 不直接 import，不影响。但 lapx 运行时缺 **numba**（setup.py 没声明），00a 的 [3f] 补装。
4. **PyPI 清华源打不开**：用阿里源（`mirrors.aliyun.com/pypi/simple`，~840 KB/s）。华为源路径不通。
5. **HuggingFace 不通**：huggingface.co / hf-mirror.com 都连不上。用 **modelscope.cn** 镜像（`AntResearch/4DAnyone`，~230 KB/s）下权重。
6. **.NET 解压 zip**：PowerShell `Expand-Archive` 对 Linux 生成的 zip 报「尾部记录」错误。用 `[System.IO.Compression.ZipFile]::ExtractToDirectory`（步骤 1）。
