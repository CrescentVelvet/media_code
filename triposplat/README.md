# TripoSplat runner

One-click orchestration to run [TripoSplat](https://github.com/VAST-AI-Research/TripoSplat) inference on an Ubuntu + NVIDIA GPU server.
This folder holds **only orchestration scripts** — no official code, no weights. The official TripoSplat repo is cloned automatically; weights are downloaded from HuggingFace.

## Design
- **Reuses an existing conda env** (default name `doll`) that already has a CUDA-enabled torch — no torch download, no venv creation.
- Runtime deps (numpy/safetensors/pillow/tqdm/huggingface_hub) are installed **on demand**; or run `INSTALL_DEPS=1 bash triposplat/00_setup_env.sh` to install the known set in one shot (torch is NOT reinstalled).
- Code and weights live outside this repo (see Layout).

## Layout (when this repo is cloned under your code dir)
```
<code-dir>/
├── media_code/              # this repo
│   ├── proxy.env            # proxy + optional overrides, gitignored
│   └── triposplat/
│       ├── _env.sh              # shared: proxy + CA bundle + conda activate
│       ├── 00_setup_env.sh
│       ├── 01_download_models.sh
│       ├── 02_run_inference.sh
│       ├── run_all.sh
│       ├── setup_ca_bundle.sh   # one-time: extract proxy CA -> ~/.ca-bundle.crt
│       └── _extract_ca.py       #   helper used by setup_ca_bundle.sh
├── TripoSplat/              # official code (auto-cloned to ../TripoSplat)
│   └── ckpts -> ../model/TripoSplat   # symlink to shared weights
└── model/
    └── TripoSplat/          # weights (hf download)
```
Defaults: official code at `../TripoSplat`, weights at `../model/TripoSplat` (relative to this repo). Override with `TRIPOSPLAT_DIR` / `MODEL_DIR`.

## Prerequisites
- Ubuntu, NVIDIA driver (CUDA 11.8+ OK), `git`, `conda`
- A conda env with a CUDA-enabled torch already installed (default env name `doll`). Create one if needed:
  ```bash
  conda create -n doll python=3.10 -y && conda activate doll
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```
- NVIDIA GPU (A100 40G/80G ideal; ~8–16G VRAM)

## Setup (on the server)
```bash
cd <your-code-dir>   # e.g. /data_3d/<uid>/code
git -c http.sslVerify=false clone https://github.com/CrescentVelvet/media_code.git
cd media_code
cp proxy.env.example proxy.env
# edit proxy.env: http_proxy / https_proxy (and CONDA_ENV if your env isn't 'doll')
bash triposplat/run_all.sh
```
`run_all.sh`: activate conda env → verify torch → clone official repo → download weights → symlink `ckpts` → run inference.

## Step-by-step
```bash
bash triposplat/00_setup_env.sh        # activate env + verify torch (set INSTALL_DEPS=1 to install deps)
bash triposplat/01_download_models.sh  # hf download + ckpts symlink
bash triposplat/02_run_inference.sh    # run_example.py
```
Missing a package? Just `pip install <pkg>` in the conda env and rerun the failed step.

## 可能遇到的问题

公司代理做 HTTPS 中间人解密，下面按流水线阶段列出常见报错与修法（命令在服务器上、`doll` 环境已激活时执行）。

**1. clone 本仓报 SSL / 认证**
公开仓免认证，加 `-c http.sslVerify=false` 即可（见上文 Setup）。若克隆的是私有仓且提示不能用账号密码，是 GitHub 已停用密码认证——改用公开仓或只读 PAT。

**2. pip 装 torch 报 `SSL:CERTIFICATE_VERIFY_FAILED`**
脚本已内置 `--trusted-host`。仍失败时手动信任：
```bash
pip config set global.trusted-host "pypi.org pypi.python.org files.pythonhosted.org download.pytorch.org"
bash triposplat/00_setup_env.sh
```

**3. pip 装 torch 报 `HTTPSConnectionPool`（连接超时/断开）**
不是 torch 版本问题，是代理对大文件超时。加超时重试，或退回 PyPI 默认 torch（自带 CUDA，A100 可用）：
```bash
pip install --timeout 600 --retries 10 --trusted-host download.pytorch.org \
  --index-url https://download.pytorch.org/whl/cu118 torch torchvision
# 仍不行：
pip install --timeout 600 --retries 10 torch torchvision
python -c "import torch;print(torch.cuda.is_available(), torch.version.cuda)"  # 需 True + 12.x
```

**4. `hf download` 报 `CAS service error : ReqwestMiddleware`**
HF 的 Xet/Rust 通道不认代理。`_env.sh` 已设 `HF_HUB_DISABLE_XET=1`；仍报则彻底卸载：
```bash
pip uninstall -y hf_xet
bash triposplat/01_download_models.sh
```

**5. `hf download` 报 `SSLCertVerificationError`**
代理根 CA 不在系统证书包。一次性建包，之后 `_env.sh` 自动用 `~/.ca-bundle.crt`：
```bash
bash triposplat/setup_ca_bundle.sh    # 抓代理证书链 -> ~/.ca-bundle.crt，并自检
bash triposplat/01_download_models.sh
```
- 自检 `[OK]` → 直接重跑 `01`。
- 自检 `[FAIL]`（代理握手没带根 CA）→ 把公司根 CA 追加后再重跑：
  ```bash
  cat /path/to/corporate_root_ca.crt >> ~/.ca-bundle.crt
  ```
  公司根 CA 常见于 `/usr/local/share/ca-certificates/`（脚本已自动并入该目录；在那里就不用手动加）。

**6. 缺包 `ModuleNotFoundError`**
按需补，或一次性装齐已知小依赖：
```bash
pip install <包名>
# 或：
INSTALL_DEPS=1 bash triposplat/00_setup_env.sh
```

> 通用：`proxy.env`（代理凭证）在仓内 gitignored，`~/.ca-bundle.crt` 在家目录，都不入库；切勿把凭证写进脚本。

## Config (env vars, all optional)
| var | default | note |
|---|---|---|
| `CONDA_ENV` | `doll` | conda env to activate (must already have torch) |
| `TRIPOSPLAT_DIR` | `../TripoSplat` | official code path |
| `MODEL_DIR` | `../model/TripoSplat` | weights path (ckpts symlinks here) |
| `TRIPOSPLAT_REPO` | official GitHub URL | clone source |
| `HF_REPO_ID` | `VAST-AI/TripoSplat` | weights repo |
| `INSTALL_DEPS` | `0` | set `1` to install known runtime deps in 00 |
| `HF_HUB_DISABLE_XET` | `1` | disable HF Xet/CAS Rust path (proxy-unfriendly) |
| `HF_HUB_ENABLE_HF_TRANSFER` | `0` | Rust accel; may ignore proxy, off by default |

## Outputs
Written to the official code dir (`../TripoSplat` by default):
- `output.ply` / `output.splat` — 262144 gaussians (view in [SuperSplat](https://superspl.at/editor))
- `preprocessed_image.webp`
- `output_{32768,65536,131072,262144}.ply` — multi-density variants

## Notes
- Official code & weights follow their own license (TripoSplat = MIT). This folder only orchestrates; no official code is copied.
- `.gitattributes` (repo root) forces LF so Windows-pushed scripts run cleanly on Ubuntu.
- `proxy.env` (proxy creds / path / env overrides) is gitignored — never committed. Don't put credentials in scripts.
- SSL behind a TLS-intercepting corporate proxy: pip uses `--trusted-host`; `hf`/`git` use the CA bundle (`_env.sh` prefers `~/.ca-bundle.crt`, built by `setup_ca_bundle.sh`).
