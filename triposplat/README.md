# TripoSplat runner

One-click orchestration to run [TripoSplat](https://github.com/VAST-AI-Research/TripoSplat) inference on an Ubuntu + NVIDIA GPU server.
This folder holds **only orchestration scripts** — no official code, no weights. The official TripoSplat repo is cloned automatically; weights are downloaded from HuggingFace.

## Layout (when this repo is cloned under your code dir)
```
<code-dir>/
├── media_code/              # this repo
│   ├── proxy.env            # your proxy (+ optional path overrides), gitignored
│   └── triposplat/
│       ├── 00_setup_env.sh
│       ├── 01_download_models.sh
│       ├── 02_run_inference.sh
│       └── run_all.sh
├── TripoSplat/              # official code (auto-cloned to ../TripoSplat)
│   └── ckpts -> ../model/TripoSplat   # symlink to shared weights
└── model/
    └── TripoSplat/          # weights (hf download)
```
Defaults: official code at `../TripoSplat`, weights at `../model/TripoSplat` (relative to this repo). Override with `TRIPOSPLAT_DIR` / `MODEL_DIR` if you clone elsewhere.

## Prerequisites
- Ubuntu, NVIDIA driver (nvcc 12.4 OK), `git`, `python3` (≥3.9)
- NVIDIA GPU (A100 40G/80G ideal; ~8–16G VRAM)
- ~10G disk (weights + venv)

## Setup (on the server)
```bash
git -c http.sslVerify=false clone https://github.com/CrescentVelvet/media_code.git
cd media_code
cp proxy.env.example proxy.env
# edit proxy.env: fill http_proxy / https_proxy
# (set TRIPOSPLAT_DIR / MODEL_DIR only if you cloned this repo somewhere unusual)
bash triposplat/run_all.sh
```
`run_all.sh`: clone official repo → build venv → download weights → symlink `ckpts` → run inference.

## Step-by-step
```bash
bash triposplat/00_setup_env.sh        # venv + torch(cu124) + deps
bash triposplat/01_download_models.sh  # hf download + ckpts symlink
bash triposplat/02_run_inference.sh    # run_example.py
```

## Config (env vars, all optional)
| var | default | note |
|---|---|---|
| `TRIPOSPLAT_DIR` | `../TripoSplat` | official code path |
| `MODEL_DIR` | `../model/TripoSplat` | weights path (ckpts symlinks here) |
| `TRIPOSPLAT_REPO` | official GitHub URL | clone source |
| `VENV_DIR` | `<algo>/.venv` | venv location |
| `CUDA_TAG` | `cu124` | cu118 / cu121 / cu124 / cu126 |
| `INSTALL_GRADIO` | `1` | install gradio (web demo) |
| `HF_REPO_ID` | `VAST-AI/TripoSplat` | weights repo |
| `HF_HUB_ENABLE_HF_TRANSFER` | `0` | Rust accel; may ignore proxy, off by default |

## Outputs
Written to the official code dir (`../TripoSplat` by default):
- `output.ply` / `output.splat` — 262144 gaussians (view in [SuperSplat](https://superspl.at/editor))
- `preprocessed_image.webp`
- `output_{32768,65536,131072,262144}.ply` — multi-density variants

## Notes
- Official code & weights follow their own license (TripoSplat = MIT). This folder only orchestrates; no official code is copied.
- `.gitattributes` (repo root) forces LF so Windows-pushed scripts run cleanly on Ubuntu.
- `proxy.env` (proxy creds / path overrides) is gitignored — never committed. Don't put credentials in scripts.
- SSL behind a TLS-intercepting corporate proxy: pip uses `--trusted-host`; `hf`/`git` use the system CA bundle. See `proxy.env.example` if you must extract the proxy CA.
