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
│       ├── _env.sh          # shared: proxy + CA bundle + conda activate
│       ├── 00_setup_env.sh
│       ├── 01_download_models.sh
│       ├── 02_run_inference.sh
│       └── run_all.sh
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

## Config (env vars, all optional)
| var | default | note |
|---|---|---|
| `CONDA_ENV` | `doll` | conda env to activate (must already have torch) |
| `TRIPOSPLAT_DIR` | `../TripoSplat` | official code path |
| `MODEL_DIR` | `../model/TripoSplat` | weights path (ckpts symlinks here) |
| `TRIPOSPLAT_REPO` | official GitHub URL | clone source |
| `HF_REPO_ID` | `VAST-AI/TripoSplat` | weights repo |
| `INSTALL_DEPS` | `0` | set `1` to install known runtime deps in 00 |
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
- SSL behind a TLS-intercepting corporate proxy: pip uses `--trusted-host`; `hf`/`git` use the system CA bundle. See `proxy.env.example` if you must extract the proxy CA.
