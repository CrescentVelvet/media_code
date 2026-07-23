# TRELLIS.2 runner

One-click orchestration to run [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) (image-to-3D, 4B) inference on an Ubuntu + NVIDIA GPU server.
This folder holds **only orchestration scripts** — no official code, no weights. The official TRELLIS.2 repo is cloned automatically (`--recursive`); weights are downloaded from HuggingFace.

## Design
- **Reuses an existing conda env** (default name `trellis2`) that already has a CUDA-enabled torch — no torch download, no venv creation. Set `CREATE_ENV=1` to instead let the official `setup.sh --new-env` create the env fresh (torch 2.6.0 + cu124) and install everything in one shot.
- TRELLIS.2 needs several compiled CUDA extensions (flash-attn, nvdiffrast, nvdiffrec, cumesh, flexgemm, o-voxel). `00_setup_env.sh` installs them via the **official `setup.sh`** (sourced from the cloned repo) into the active env. Set `INSTALL_DEPS=0` to skip.
- The pipeline loads weights through `Trellis2ImageTo3DPipeline.from_pretrained(<local dir>)`, so **no symlink** into the official repo is needed — local weights just work.
- Code and weights live outside this repo (see Layout).

## Layout (when this repo is cloned under your code dir)
```
<code-dir>/
├── media_code/              # this repo
│   ├── proxy.env            # proxy + optional overrides, gitignored
│   └── trellis-2/
│       ├── _env.sh              # shared: proxy + CA bundle + conda activate + TRELLIS.2 runtime env
│       ├── 00_setup_env.sh      # activate env + verify torch + run official setup.sh
│       ├── 01_download_models.sh
│       ├── 02_run_inference.sh
│       ├── run_all.sh
│       ├── setup_ca_bundle.sh   # one-time: extract proxy CA -> ~/.ca-bundle.crt
│       ├── _extract_ca.py       #   helper used by setup_ca_bundle.sh
│       ├── _hf_download.py      #   snapshot_download with SSL verify off (01 fallback)
│       ├── run_batch.py         #   batch inference (load pipeline once, loop images -> GLB + .latent.npz)
│       ├── 03_render_video.sh   # decode .latent.npz -> mp4 (PBR turntable / composite)
│       └── render_video.py      #   decode + render helper (render_utils + o_voxel)
├── TRELLIS.2/               # official code (auto-cloned to ../TRELLIS.2, --recursive for o-voxel)
└── ../../model/             # weights live one dir above <code-dir> (shared by all algos)
    └── TRELLIS.2-4B/        # weights (hf download)
```
Defaults: official code at `../TRELLIS.2`, weights at `../../model/TRELLIS.2-4B` (relative to this repo). Override with `TRELLIS_DIR` / `MODEL_DIR`.

## Prerequisites
- Ubuntu, NVIDIA driver (CUDA 12.4 Toolkit recommended for compiling extensions), `git`, `conda`
- **CUDA Toolkit 12.4** installed and `CUDA_HOME` pointing at it (default `/usr/local/cuda`). Needed to compile flash-attn/nvdiffrast/cumesh/flexgemm/o-voxel.
- A conda env with a CUDA-enabled torch already installed (default env name `trellis2`). Create one if needed:
  ```bash
  conda create -n trellis2 python=3.10 -y && conda activate trellis2
  pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
  ```
  Or skip this and use `CREATE_ENV=1 bash trellis-2/run_all.sh` (the official `setup.sh --new-env` creates it).
- NVIDIA GPU with **>=24GB VRAM** (A100 40G/80G or H100 ideal; verified on A100/H100). For V100 (no flash-attn), `pip install xformers` and set `ATTN_BACKEND=xformers`.

## Setup (on the server)
```bash
cd <your-code-dir>   # e.g. /data_3d/<uid>/code
git -c http.sslVerify=false clone https://github.com/CrescentVelvet/media_code.git
cd media_code
cp proxy.env.example proxy.env
# edit proxy.env: http_proxy / https_proxy (and CONDA_ENV if your env isn't 'trellis2')
# bash trellis-2/run_all.sh
# `run_all.sh`: clone official repo (--recursive) -> install deps (or create env) -> download weights -> run inference.

## Step-by-step
# bash trellis-2/00_setup_env.sh        # activate env + verify torch + install extensions (INSTALL_DEPS=1)
# CREATE_ENV=1 bash trellis-2/00_setup_env.sh   # alt: create the trellis2 env fresh via setup.sh --new-env
# HF_DISABLE_SSL=1 bash trellis-2/01_download_models.sh   # hf download microsoft/TRELLIS.2-4B
# GPU=7 INPUT_DIR=/path/to/images bash trellis-2/02_run_inference.sh    # each image -> <stem>.glb + <stem>.latent.npz
# GPU=7 LATENT_INPUT=../TRELLIS.2/output/<set> bash trellis-2/03_render_video.sh  # .latent.npz -> mp4 (PBR turntable)
```
Missing a package? Just `pip install <pkg>` in the conda env and rerun the failed step. If a specific extension fails to compile, re-run `00` with a reduced `SETUP_FLAGS` (e.g. drop `--flash-attn` and set `ATTN_BACKEND=xformers`).

## Outputs
`02` batch-processes every image in `INPUT_DIR`. Outputs are nested under `OUTPUT_DIR/<input_folder_name>/`. For each `<stem>.<ext>`:
- `<stem>.glb` — PBR 3D asset (O-Voxel -> GLB via `o_voxel.postprocess.to_glb`). Open in [https://3dviewer.net](https://3dviewer.net) or Blender; the alpha channel is in the texture but inactive by default (connect it to the material's opacity to enable transparency).
- `<stem>.latent.npz` — cached latents (shape/tex slat + coords + res) so `03` can re-render without re-running the 3-stage generation (mirrors the official `app.py` `pack_state`/`unpack_state`).
- `<stem>.mp4` — (only if `RENDER_VIDEO=1`) a quick shaded turntable, rendered in the same pass while the mesh is in memory.

`RESOLUTION` selects the pipeline: `512` (`~3s`), `1024` (`~17s`, default, cascade), `1536` (`~60s`) — times from the paper on H100. Sampler params use the `app.py` recommended defaults (SS 12 steps / 7.5 cfg, shape 12 / 7.5, tex 12 / 1.0).

## Render to video (.latent.npz -> mp4)
Decode the cached latents and render a turntable with TRELLIS.2's PBR renderer. No generation is repeated. Output path mirrors `02`: `VIDEOS_DIR/<input_folder_name>/<stem>.mp4` (+ a `<stem>.png` first frame).
```bash
git -c http.sslVerify=false pull
GPU=7 LATENT_INPUT=../TRELLIS.2/output/setA bash trellis-2/03_render_video.sh
# -> ../TRELLIS.2/videos/setA/<stem>.mp4
```
`RENDER_MODE=shaded` (default) writes a clean turntable of the PBR-shaded image; `RENDER_MODE=pbr` writes the official composite grid (shaded + normal + base_color + metallic + roughness + alpha, as in `example.py`'s `make_pbr_vis_frames`). Lighting comes from an HDRI env map under `<TRELLIS_DIR>/assets/hdri/` (`ENVMAP=forest|sunset|courtyard|none`). Tweak the camera via `NUM_FRAMES FPS R FOV RENDER_RES`.

## 可能遇到的问题

公司代理做 HTTPS 中间人解密，下面按流水线阶段列出常见报错与修法（命令在服务器上、`trellis2` 环境已激活时执行）。

**1. clone 本仓 / 官方仓报 SSL / 认证**
公开仓免认证，加 `-c http.sslVerify=false` 即可（见上文 Setup）。`run_all.sh` 克隆官方仓失败时也会自动带 `-c http.sslVerify=false` 重试。

**2. pip 装 torch / 扩展报 `SSL:CERTIFICATE_VERIFY_FAILED`**
`_env.sh` 已设 `PIP_CERT` 指向 CA bundle，`setup.sh` 的 pip 命令会继承。仍失败时手动信任：
```bash
pip config set global.trusted-host "pypi.org pypi.python.org files.pythonhosted.org download.pytorch.org"
bash trellis-2/00_setup_env.sh
```

**3. 编译扩展报 CUDA 找不到 / 版本不对**
扩展需要 CUDA Toolkit（推荐 12.4），不是驱动版本。确认 `CUDA_HOME` 指向 toolkit：
```bash
echo $CUDA_HOME                       # 应为 /usr/local/cuda 或 /usr/local/cuda-12.4
nvcc --version                        # 看 release 版本
ls $CUDA_HOME/include/cuda_runtime.h  # 必须存在
# 多版本时显式指定：
export CUDA_HOME=/usr/local/cuda-12.4
bash trellis-2/00_setup_env.sh
```

**4. `--basic` 里 `sudo apt install libjpeg-dev` 失败（docker 无 sudo）**
直接在容器里装（若已是 root 则去掉 sudo）：
```bash
apt install -y libjpeg-dev
# 或跳过 basic、手动装其余小依赖后用精简 flags：
SETUP_FLAGS="--flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm" bash trellis-2/00_setup_env.sh
```

**5. flash-attn 编译失败 / 显卡不支持（如 V100）**
V100 不支持 flash-attn。装 xformers 并切后端：
```bash
pip install xformers
ATTN_BACKEND=xformers SETUP_FLAGS="--basic --xformers-instead" bash trellis-2/00_setup_env.sh   # 见下方注
```
> `setup.sh` 没有 `--xformers` 选项；用 `SETUP_FLAGS="--basic --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm"`（去掉 `--flash-attn`），再 `pip install xformers`，并 `export ATTN_BACKEND=xformers`（`_env.sh` 会透传）。

**6. `hf download` 报 `CAS service error : ReqwestMiddleware`**
HF 的 Xet/Rust 通道不认代理。`_env.sh` 已设 `HF_HUB_DISABLE_XET=1`；仍报则彻底卸载：
```bash
pip uninstall -y hf_xet
bash trellis-2/01_download_models.sh
```

**7. `hf download` 报 `SSLCertVerificationError`**
代理根 CA 不在系统证书包。先一次性建包，`_env.sh` 会自动用 `~/.ca-bundle.crt`：
```bash
bash trellis-2/setup_ca_bundle.sh    # 抓代理证书链 -> ~/.ca-bundle.crt，并自检
bash trellis-2/01_download_models.sh
```
- 自检 `[OK]` → 直接重跑 `01`。
- 自检 `[FAIL]`（代理握手没带根 CA）→ 把公司根 CA 追加后再重跑：
  ```bash
  cat /path/to/corporate_root_ca.crt >> ~/.ca-bundle.crt
  ```
  公司根 CA 常见于 `/usr/local/share/ca-certificates/`（脚本已自动并入该目录）。

> 若建包后仍报 SSL（CDN 端点 `us.aws.cdn.hf.co` 等用了不同的 MITM 证书），`01` 会自动回退到禁用 SSL 校验的下载器（`_hf_download.py`）；或直接 `HF_DISABLE_SSL=1 bash trellis-2/01_download_models.sh` 跳过首次尝试。代理已全程 MITM，此处关掉校验可接受。

**8. 缺包 `ModuleNotFoundError`**
按需补：
```bash
pip install <包名>
```

> 通用：`proxy.env`（代理凭证）在仓内 gitignored，`~/.ca-bundle.crt` 在家目录，都不入库；切勿把凭证写进脚本。

## Config (env vars, all optional)
| var | default | note |
|---|---|---|
| `CONDA_ENV` | `trellis2` | conda env to activate (must already have torch, unless `CREATE_ENV=1`) |
| `CREATE_ENV` | `0` | `1` = let official `setup.sh --new-env` create the env + install torch + deps |
| `INSTALL_DEPS` | `1` | `0` = skip running official `setup.sh` in `00` (env already fully set up) |
| `SETUP_FLAGS` | `--basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm` | components passed to official `setup.sh` |
| `GPU` | _(unset)_ | physical GPU id to pin, e.g. `GPU=7`; remaps `CUDA_VISIBLE_DEVICES` so in-process `cuda:0` == that card |
| `TRELLIS_DIR` | `../TRELLIS.2` | official code path |
| `MODEL_DIR` | `../../model/TRELLIS.2-4B` | weights path (passed to `from_pretrained`) |
| `OUTPUT_DIR` | `../TRELLIS.2/output` | where inference outputs are written |
| `INPUT_DIR` | `../TRELLIS.2/assets/example_image` | folder of images to batch-process (02) |
| `TRELLIS_REPO` | official GitHub URL | clone source |
| `HF_REPO_ID` | `microsoft/TRELLIS.2-4B` | weights repo |
| `SEED` | `0` | generation seed (same seed + image => same mesh) |
| `RESOLUTION` | `1024` | `512` / `1024` / `1536` -> pipeline_type `512` / `1024_cascade` / `1536_cascade` |
| `DECIMATION_TARGET` | `1000000` | GLB face-count target (02) |
| `TEXTURE_SIZE` | `4096` | GLB texture resolution (02) |
| `RENDER_VIDEO` | `0` | `1` = also write a quick shaded mp4 during 02 (mesh already in memory) |
| `LOW_VRAM` | `1` | `1` = move sub-models on/off device between stages (safe on 24G); `0` = keep all on GPU (needs more VRAM) |
| `ATTN_BACKEND` | _(unset=flash-attn)_ | set `xformers` for V100 |
| `CUDA_HOME` | `/usr/local/cuda` | CUDA Toolkit dir (for compiling extensions) |
| `HF_HUB_DISABLE_XET` | `1` | disable HF Xet/CAS Rust path (proxy-unfriendly) |
| `HF_DISABLE_SSL` | `0` | `1` = download weights with SSL verification disabled |
| `LATENT_INPUT` | `../TRELLIS.2/output` | .latent.npz file or folder to render (03) |
| `VIDEOS_DIR` | `../TRELLIS.2/videos` | base video dir; mp4s go to `VIDEOS_DIR/<input_folder_name>/` |
| `RENDER_MODE` | `shaded` | `shaded` (turntable) / `pbr` (composite grid, official `example.py`) (03) |
| `RENDER_RES` | `1024` | render resolution (03) |
| `NUM_FRAMES`/`FPS` | `120`/`15` | turntable length (03) |
| `R`/`FOV` | `2`/`40` | camera distance (in normalized units) / field of view in degrees (03) |
| `ENVMAP` | `forest` | HDRI: `forest` / `sunset` / `courtyard` / `none` (03; and 02 if `RENDER_VIDEO=1`) |

## Notes
- Official code & weights follow their own license (TRELLIS.2 = MIT; nvdiffrast/nvdiffrec have their own licenses). This folder only orchestrates; no official code is copied.
- `.gitattributes` (repo root) forces LF so Windows-pushed scripts run cleanly on Ubuntu.
- `proxy.env` (proxy creds / path / env overrides) is gitignored — never committed. Don't put credentials in scripts.
- SSL behind a TLS-intercepting corporate proxy: pip uses `PIP_CERT`/`--trusted-host`; `hf`/`git` use the CA bundle (`_env.sh` prefers `~/.ca-bundle.crt`, built by `setup_ca_bundle.sh`).
- Sampler defaults (SS/shape/tex steps=12, cfg 7.5/7.5/1.0, rescale_t 5.0/3.0/3.0) and the resolution->pipeline_type mapping match `app.py`. `example.py` uses pipeline defaults (same). Override `SEED`/`RESOLUTION` via env; for finer sampler control edit `run_batch.py`.
