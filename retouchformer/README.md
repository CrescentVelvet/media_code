# RetouchFormer runner

在 Ubuntu + NVIDIA 服务器上跑 [RetouchFormer](https://github.com/Davidcoach/RetouchFormer_AAAI_24)（AAAI 2024 人脸 retouching）的**推理**。本目录只含编排脚本——官方代码自动 clone、权重按 README 走 Baidu 网盘手动下载。

> 参考骨架来自 `hypir/`（同为「编排脚本 + 自动 clone + 一键推理」范式）。本目录只做**推理**；训练/评测请用官方 `train.py` / `eval.py`。

## 常用命令

> 假设已进入容器并 `conda activate retouchformer`；路径取各脚本默认值（可改）；`GPU=0` 按需换卡。首次跑前先做下方「首次准备」。

```bash
# ── 推理(02) ──
# 1) 对一个文件夹的人脸做 retouching（递归遍历，保留相对目录结构）
GPU=0 INPUT_DIR=/data/faces bash retouchformer/02_run_inference.sh
# 2) 用别的 checkpoint（默认 release_model/gen_best.pth）/ 换输出目录
GPU=0 INPUT_DIR=/data/faces WEIGHT_PATH=/data/RetouchFormer/model/RetouchFormer/release_model/gen_best.pth \
  OUTPUT_DIR=../RetouchFormer/results/myset bash retouchformer/02_run_inference.sh
# 3) 严格复现官方 wildDataset（仅 Resize(512)，不 CenterCrop）——仅当输入已是 512×512 时安全
GPU=0 INPUT_DIR=/data/faces RESIZE_MODE=smallest bash retouchformer/02_run_inference.sh
```

- 结果：`../RetouchFormer/results/<输入夹名>/result/*.png`（512×512，相对目录结构保留）。
- 想要定量指标（PSNR/SSIM/LPIPS）用官方 `eval.py`（注意：仓库里的 `eval.py` 有 `target_tensor_list`/`path` 等未定义变量的 bug，直接跑会崩，需自行修）。

## 首次准备
```bash
cd <your-code-dir>            # e.g. /data_3d/<uid>/code
git -c http.sslVerify=false clone https://github.com/CrescentVelvet/media_code.git
cd media_code && cp proxy.env.example proxy.env   # 填 http_proxy / https_proxy（公司代理；自用可跳过）
conda create -n retouchformer python=3.8 -y && conda activate retouchformer
# torch 1.13.1 (cu117) — A100 / 4090 及更老卡用这个
pip install torch==1.13.1 torchvision==0.14.1
# H100 (sm90) 需要 cu118+，改用更新版 torch（op/ 的 CUDA 扩展会按当前 torch 重新 JIT 编译）：
#   pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
INSTALL_DEPS=1 bash retouchformer/00_setup_env.sh          # 装推理依赖（requirements_inference.txt）
bash retouchformer/01_download_models.sh                   # ⚠ 见下方「权重」——需手动下 Baidu
```
⚠ RetouchFormer 的依赖（torch==1.13.1 / python==3.8）与本仓其他算法冲突，务必用专用 env（`CONDA_ENV=retouchformer`），别装进共享的 `doll`。

### 权重（Baidu 网盘，必须手动下）
官方**只**通过百度网盘发布 `gen_best.pth`，没有可脚本化下载的 HTTP/HuggingFace 镜像。`01_download_models.sh` 检测到缺失时会打印步骤并退出。一次即可：
1. 浏览器打开 https://pan.baidu.com/s/1eVgPN12KJN8GSdOw544ZdQ ，提取码 `reto`，下载 `gen_best.pth`。
2. 放到位：`mkdir -p ../../model/RetouchFormer/release_model && cp gen_best.pth $_`。
3. 重跑 `bash retouchformer/01_download_models.sh` 应显示 `checkpoint already present`。
4. 若你在内网镜像了该文件：`WEIGHT_URL=https://your.host/gen_best.pth bash retouchformer/01_download_models.sh` 可直接下载（含代理 + CA bundle + SSL 兜底）。

---

以下为详细参考（流程原理 / 各脚本参数 / 排错 / 目录布局）。

## Inference（02 — 推理流程详解）
`02_run_inference.sh` 调 `run_inference.py`：构建一次 `model.RetouchFormer.InpaintGenerator()`、加载 `gen_best.pth`，然后循环遍历 `INPUT_DIR`（递归）全图，逐图打印分辨率与耗时，输出 `OUTPUT_DIR/result/<rel>.png`（512×512）。

对应官方 `img_retouching.py` 的等价逻辑：
```python
net  = importlib.import_module('model.RetouchFormer')
model = net.InpaintGenerator().to(device)
model.load_state_dict(torch.load('release_model/gen_best.pth', map_location=device))
model.eval()
for name, source_tensor in loader:
    pred_img, _ = model(source_tensor.to(device))
    save_image(pred_img, f'{name}_out.png', normalize=True, value_range=(-1, 1))
```

`run_inference.py` 与官方的差异（均为改进，不影响已对齐的 FFHQR 测试集结果）：
- **不硬编码 `CUDA_VISIBLE_DEVICES="0"`**：由 `GPU=N`（`_env.sh` 映射）选卡。
- **保留相对目录结构**：输出 `result/<rel>.png`，而非平铺 `<name>_out.png`。对嵌套输入更友好。
- **`RESIZE_MODE=square`（默认）**：`Resize(512) + CenterCrop(512)`，确保非方形 wild 图也能跑（模型 VRT 的 `input_resolution=(6,64,64)` 写死 512×512，非方形会崩）。官方 `wildDataset` 仅 `Resize(512)`（最小边到 512），对 FFHQR 测试集（本就 512×512）两者等价；处理 wild 非方形图时设 `RESIZE_MODE=smallest` 会复现官方行为但很可能在 VRT 分窗时崩，故默认 `square`。
- **单图计时 + 汇总**：avg/min/max/total，官方只有 `tqdm` 进度条。

### 模型前向（一步，无扩散迭代）
RetouchFormer 不是扩散模型，单次前向即出结果。对一张 512×512 人脸：
1. **GPEN 编码器**（`model/gpen_model.py::Encoder`，stylegan2 架构）：3→32→…→512 通道，下采样到 64×64 的多尺度噪声特征 `decoder_noise[0..5]`（分辨率 64/128/256/512… 经 `conv_512/256/128` 统一到 512 通道）。
2. **瑕疵注意力**（`model/modules/soft_mask_generation.py::VQVAEMaskGAN`）：对输入预测瑕疵（痘/斑等）概率图 `attention_feat`（sigmoid 后 >0.5 即瑕疵）。
3. **VRT Transformer**（`model/network_vrt_pair_qkv.py::Stage`，`input_resolution=(6,64,64)`，depth=7，window=[6,16,16]）：把 6 个尺度特征堆成 `[B,C,6,H,W]`，带瑕疵图做 **selective self-attention**——瑕疵 token 只与正常皮肤 K/V 交互（`soft inpainting`），输出 `vrt_out`。
4. **选择性融合**：`mask = attention>0.5`；瑕疵位置取 `vrt_out`（修复后），正常位置取原 `decoder_noise`（保留），经 `back_*` 残差加回。
5. **GPEN 解码器**（`model/gpen_model.py::Decoder`）：逐级上采样 + `StyledConv` + `ToRGB`，输出 512×512 修复图 `[-1,1]`。

> 一句话：**GPEN 编/解码器做骨干 → VQVAE 预测瑕疵图 → VRT 带选择性自注意力「软修复」瑕疵区 → 解码回 512×512**。

### 依赖：op/ 的 stylegan2 CUDA 算子
`model/gpen_model.py` 里 `from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d`。`op/fused_act.py` 与 `op/upfirdn2d.py` 在 **Linux + CUDA** 下用 `torch.utils.cpp_extension.load` **JIT 编译** `.cpp/.cu`（首次 import `model.RetouchFormer` 时触发，缓存到 `~/.cache/torch_extensions/`，之后免编译）。需要：
- `ninja`（`requirements_inference.txt` 已含）；
- `nvcc`（CUDA toolkit，版本要与 torch 匹配：torch 1.13.1 → CUDA 11.7；torch 2.1 → CUDA 12.1）。
- 非 Linux 或无 CUDA 时回退到纯 PyTorch 实现（`upfirdn2d_native`、`F.leaky_relu` 版的 `fused_leaky_relu`），慢但能跑（GPU 服务器上不会触发）。

## Config (env vars, all optional)
| var | default | note |
|---|---|---|
| `CONDA_ENV` | `retouchformer` | conda env to activate (dedicated recommended — pins conflict with other algos) |
| `GPU` | _(unset)_ | physical GPU id to pin, e.g. `GPU=0`; maps `CUDA_VISIBLE_DEVICES` |
| `RETOUCH_DIR` | `../RetouchFormer` | official code path |
| `MODEL_DIR` | `../../model/RetouchFormer` | weights path |
| `RETOUCH_REPO` | official GitHub URL | clone source |

### Inference (02)
| var | default | note |
|---|---|---|
| `INPUT_DIR` | `../RetouchFormer/datasets/test` | folder of face images (walked recursively) |
| `OUTPUT_DIR` | `../RetouchFormer/results/<input_folder_name>` | writes `result/` under here |
| `WEIGHT_PATH` | `$MODEL_DIR/release_model/gen_best.pth` | checkpoint; override to point elsewhere |
| `CKPT_DIR` / `EPOCH` / `WEIGHT_FILE` | `…/release_model` / `best` / `gen_best.pth` | compose `WEIGHT_PATH` if `WEIGHT_PATH` unset |
| `MODEL_NAME` | `RetouchFormer` | `model.<NAME>` module providing `InpaintGenerator()` |
| `RESIZE_MODE` | `square` | `square` = Resize(512)+CenterCrop(512)（推荐，wild 图安全）；`smallest` = 仅 Resize(512)（复现官方 wildDataset，仅方形 512 输入安全） |
| `SIZE` | `512` | ⚠ 模型写死 512，改了几乎必崩；仅占位 |
| `DEVICE` | `cuda` | falls back to CPU if CUDA unavailable (very slow) |

### Weights (01)
| var | default | note |
|---|---|---|
| `WEIGHT_URL` | _(unset)_ | if set, direct-curl the checkpoint (proxy + CA bundle + SSL fallback) |
| `WEIGHT_PATH` / `CKPT_DIR` / `EPOCH` | see 02 | where to look for / place the checkpoint |

## Outputs
- **02 inference**: `OUTPUT_DIR/result/<rel>.png`（512×512，相对目录结构保留）。
- **01 weights**: `$MODEL_DIR/release_model/gen_best.pth`（手动放置或 `WEIGHT_URL` 下载）。

## 目录布局
```
<code-dir>/
├── media_code/                      # 本仓
│   ├── proxy.env                    # 代理 + 覆盖项, gitignored
│   └── retouchformer/               # 编排脚本(本目录)
├── RetouchFormer/                   # 官方代码(自动 clone 到 ../RetouchFormer)
│   ├── model/ core/ op/             # 模型/数据/算子
│   ├── img_retouching.py            # 官方推理脚本(本目录 02 是其增强版)
│   └── datasets/test/               # 官方测试样本夹(默认 INPUT_DIR)
└── ../../model/RetouchFormer/       # 权重(在 <code-dir> 上一级, 各算法共享)
    └── release_model/gen_best.pth   # 发布 checkpoint (Baidu 手动下)
```
默认：官方代码 `../RetouchFormer`、权重 `../../model/RetouchFormer`（相对本目录）；用 `RETOUCH_DIR` / `MODEL_DIR` 覆盖。

## 可能遇到的问题

**1. clone 本仓或官方仓报错**
- `SSL certificate problem`：公开仓免认证，加 `-c http.sslVerify=false`（`run_all.sh` 已对官方仓做兜底）。
- `Failed to connect to github.com port 443`（连不上，非 SSL）：git 没走代理。设全局代理（密码特殊字符必须 URL 编码）：
  ```bash
  git config --global https.proxy http://USER:PASS@proxyhk.huawei.com:8080
  git config --global  http.proxy http://USER:PASS@proxyhk.huawei.com:8080
  ```
  > 这和 `proxy.env` 里的 `http_proxy`/`https_proxy` 环境变量是**两套**：环境变量给 curl/pip 用，`git config http.proxy` 给 git 本身用；两个都设最稳。
- `No route to host` 连代理都连不上：根因常是 docker 网桥网段和代理 IP 冲突。排查 + 修：
  ```bash
  getent hosts proxyhk.huawei.com        # 取代理 IP
  ip route | grep <前两段>               # 命中 docker 网桥即冲突
  sudo ip route add <代理IP> via <默认网关> dev <物理网卡>
  ```

**2. `pip install -r requirements_inference.txt` 报 `SSL:CERTIFICATE_VERIFY_FAILED` / 超时**
```bash
pip config set global.trusted-host "pypi.org pypi.python.org files.pythonhosted.org"
INSTALL_DEPS=1 bash retouchformer/00_setup_env.sh
```
仍超时加 `--timeout 600 --retries 10`（`00_setup_env.sh` 已带）。本目录用的是**精简推理依赖**（不含官方 requirements.txt 里的 `transformers==4.43.0.dev0` dev pin 和 deepspeed/gradio/fschat/mpi4py 等训练才需要的重包）。

**3. `bash 01_download_models.sh` 退出并打印 Baidu 说明**
这是预期行为——官方只通过百度网盘发布权重，无 HTTP 镜像可脚本化下载。按打印的步骤下 `gen_best.pth`（提取码 `reto`）放到 `../../model/RetouchFormer/release_model/` 后重跑即可。或在 `proxy.env` 之外的命令前设 `WEIGHT_URL=https://your.host/gen_best.pth` 走内网镜像直连。

**4. 首次推理时大量编译输出 + 最终 `BuildExtension` / `Ninja` 报错**
`op/fused_act.py`、`op/upfirdn2d.py` 用 `torch.utils.cpp_extension.load` JIT 编译 stylegan2 的 CUDA 算子。需要：
- `ninja`（`requirements_inference.txt` 已含，`pip show ninja` 确认）；
- `nvcc` 在 PATH（装 CUDA toolkit，版本与 torch 匹配：torch 1.13.1→CUDA 11.7、torch 2.1→CUDA 12.1）。`00_setup_env.sh` 已检查 `nvcc` 并在缺失时告警。
- 编译产物缓存在 `~/.cache/torch_extensions/{fused,upfirdn2d}/`，首次成功后不再编译。
- 报 `CUDA_HOME not found` / `nvcc: command not found`：`export CUDA_HOME=/usr/local/cuda && export PATH=$CUDA_HOME/bin:$PATH` 后重试。

**5. 推理报 `window size` / `expected input resolution` / 维度不匹配**
你的输入是非方形图且 `RESIZE_MODE=smallest`（复现官方 `wildDataset`）。模型 VRT 的 `input_resolution=(6,64,64)`、`window_size=[6,16,16]` 要求 512×512。改 `RESIZE_MODE=square`（默认，会 CenterCrop 到 512×512）即可。FFHQR 测试集本身是 512×512，两种模式等价。

**6. 推理报 `CUDA error: no kernel image is available` / `smXX` 不匹配**
torch 1.13.1（cu117）不支持 H100（sm90）。H100 换 cu118+ 的 torch（见「首次准备」注释）。A100/4090 及更老卡用 torch 1.13.1 即可。

**7. 推理 OOM**
RetouchFormer 单图 512×512、batch=1，显存占用很小（~2-4GB），通常不会 OOM。若同时跑别的任务导致显存不足，设 `GPU=` 换一张空卡即可。

> 通用：`proxy.env`（代理凭证）在仓内 gitignored，`~/.ca-bundle.crt` 在家目录，都不入库；切勿把凭证写进脚本。

## Notes
- Official code & weights follow their own license (RetouchFormer = AAAI 2024 paper code; checkpoint via Baidu Netdisk). This folder only orchestrates; no official code is copied.
- `.gitattributes` (repo root) forces LF so Windows-pushed scripts run cleanly on Ubuntu.
- `proxy.env` (proxy creds / env overrides) is gitignored — never committed. Don't put credentials in scripts.
- SSL behind a TLS-intercepting corporate proxy: pip uses `--trusted-host`; `git`/`curl` use the CA bundle (`_env.sh` prefers `~/.ca-bundle.crt`; reuse `hypir/setup_ca_bundle.sh` to build it once).
