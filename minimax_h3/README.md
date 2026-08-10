# MiniMax-H3 runner

在 Ubuntu + NVIDIA 服务器上复现 [MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3)（MiniMax 的全模态视频+音频生成系统）。本目录只含编排脚本——官方代码自动 clone、权重从 HuggingFace 下载、推理走 **SGLang Diffusion**（官方推荐的三种框架之一）。

> ⚠️ **能本地复现的只有 H3-Base（768p）**。完整 H3 系统三模块：
> - **H3-Context-IR**（多模态指令预处理）——非开源，只能调 MiniMax API；
> - **H3-Base**（生成 768p 视频+原生立体声）——✅ 开源，本目录复现的就是它；
> - **H3-Regenerate-2K**（把 768p 重生成到 2K）——非开源，只能调 API。
>
> 即：本目录跑出来的是 **768p / 24fps / 立体声** 的视频。要 2K 需走「Full 2K Workflow」调 API（见文末）。

## 常用命令

> 假设已进入容器、`conda activate minimax_h3`、`cd media_code`；`GPU` 留空用全部可见卡，`NUM_GPUS` 控制张数。首次跑前先做下方「首次准备」。

```bash
# ── 一键（clone + 装环境 + 下权重 + 起服务 + 跑 T2VA 示例）──
INSTALL_DEPS=1 bash minimax_h3/run_all.sh            # 首次（装 SGLang）
bash minimax_h3/run_all.sh                            # 之后（跳过装包）

# ── 起服务（H3-Base 768p，长驻进程）──
# 1) FL2VA 变体（T2VA / I2VA / L2VA / FL2VA），端口 30010（前台跑，看日志）
bash minimax_h3/02_serve.sh
# 1b) 后台起 + 等就绪（run_all 用这种）
BG=1 bash minimax_h3/02_serve.sh
# 1c) 4× A100 80GB 容量配方（resident 若 OOM 就上 FSDP）
NUM_GPUS=4 ULYSSES_DEGREE=4 USE_FSDP=1 bash minimax_h3/02_serve.sh
# 1d) 4× H100 80GB 最快配方（TP2 + Ulysses2）
NUM_GPUS=4 TP_SIZE=2 ULYSSES_DEGREE=2 bash minimax_h3/02_serve.sh
# 2) Ref2VA 变体（参考图/视频/音频 -> 视频），端口 30011
MODEL_VARIANT=ref2va bash minimax_h3/02_serve.sh

# ── 发请求（服务必须已就绪）──
# T2VA 文生视频（自带默认 prompt）
TASK=t2va PROMPT="a drone shot over alpine peaks at golden hour" bash minimax_h3/03_generate.sh
# I2VA 首帧生视频（FL2VA 变体）
TASK=fl2va FIRST_FRAME=/data/imgs/first.png DURATION=8 bash minimax_h3/03_generate.sh
# Ref2VA 参考生成（Ref2VA 变体，服务在 :30011）
SERVER_URL=http://localhost:30011 TASK=ref2va \
  REF_IMAGES=/data/refs/subject.png REF_AUDIOS=/data/refs/voice.mp3 \
  PROMPT="Use <Picture 1> as the subject and <Audio 1> as the voice." \
  bash minimax_h3/03_generate.sh

# ── 复现官方三个 768p 样例（用官方 prompt + 官方 CDN 参考素材）──
bash minimax_h3/examples/run_t2va.sh                 # -> results/t2va/t2va.mp4
bash minimax_h3/examples/run_fl2va.sh                # -> results/fl2va/fl2va.mp4  (需 FL2VA 服务)
SERVER_URL=http://localhost:30011 bash minimax_h3/examples/run_ref2va.sh   # (需 Ref2VA 服务)
```

- 结果：生成视频 → `../MiniMax-H3/results/<task>/<name>.mp4`；服务日志 → `../MiniMax-H3/logs/serve_<variant>_<port>.log`。
- 服务是长驻进程，起一次能发无数请求（加载 33B 模型要几分钟，别每个请求重启）。
- 一次只能起一个变体（FL2VA / Ref2VA 权重不同），要两个变体就分起 30010 / 30011。

## 首次准备
```bash
cd <your-code-dir>            # e.g. /data_3d/<uid>/code
git -c http.sslVerify=false clone https://github.com/CrescentVelvet/media_code.git
cd media_code && cp proxy.env.example proxy.env      # 填 http_proxy / https_proxy（公司代理用）
conda create -n minimax_h3 python=3.11 -y && conda activate minimax_h3
# 若上一行报 HTTP 403 FORBIDDEN（.condarc 配了清华 TUNA 镜像且失效），加 --override-channels 直连官方源绕开：
#   conda create -n minimax_h3 python=3.11 -y --override-channels \
#     -c https://conda.anaconda.org/conda-forge -c https://repo.anaconda.com/pkgs/main && conda activate minimax_h3
# 详见下方「可能遇到的问题」第 2 条。
pip install torch --index-url https://download.pytorch.org/whl/cu124   # 先装 CUDA torch
INSTALL_DEPS=1 bash minimax_h3/00_setup_env.sh       # 装 sglang[all]（含 diffusion 支持）
HF_DISABLE_SSL=1 bash minimax_h3/01_download_models.sh  # 下 MiniMaxAI/MiniMax-H3 权重快照
```
⚠️ SGLang 自带 torch/flashinfer/cuda kernel，版本 pin 与本仓其他算法冲突，务必用专用 env（`CONDA_ENV=minimax_h3`），别装进共享 env。
⚠️ MiniMax-H3 在 HF 上是 **MiniMax H3 Community License**，可能 gated：下不动/报 401 时去 https://huggingface.co/MiniMaxAI/MiniMax-H3 接受协议、建 read token，再 `HF_TOKEN=<token> bash minimax_h3/01_download_models.sh`。

---

以下为详细参考（系统原理 / 各脚本参数 / 排错 / 目录布局）。

## 系统原理（H3-Base 在干什么）
MiniMax-H3 是全模态生成系统：吃 **文本 + 可选图片/视频/音频参考**，吐 **视频 + 原生立体声音频**（768p / 24fps / 32kHz stereo / 4–15s）。本地能跑的 H3-Base 架构：

1. **H3-Encoder**——用 Qwen3-VL-32B 全量预训练权重，取第 50 层 hidden state 喂给 Transformer。文本/图像都经它编码（音频不经）。
2. **H3-VisualVAE**——时间因果视频自编码器，空间压缩 16×、时间 4×、24 通道（f16t4d24）；潜变量再 patchify 成 `1×2×2`（有效空间下采样 32×、时间 4×）。解码器是 ViT-based。
3. **H3-AudioVAE**——左右声道各自编/解码，32kHz → 40Hz 潜序列（立体声）。
4. **H3-Omni-Transformer**——33B 单流 dense Transformer（~13B 在 AdaLN 分支，推理可预计算缓存故不用加载）。无模态专属结构（模态参数只在输入/输出/AdaLN）；用 3D MM-RoPE 联合表示 `(t,h,w)`。
5. 两个任务专用 checkpoint：**FL2VA**（首末帧 / 纯文）、**Ref2VA**（多模态参考）。都是 CFG-distilled 权重。

> 一句话：参考素材 →(各自 VAE/Encoder 编码)→ 拼成统一 packed 多模态序列 →(MM-RoPE)→ Omni-Transformer 联合预测视频+音频潜变量 →(各自 VAE 解码)→ 视频 + 立体声。

## Pipeline（推理流程详解）
对应 SGLang Diffusion server。一次生成从请求到 mp4 走 4 步：

1. **提交任务**——`POST /v1/videos`，body 含 `task`(t2va/fl2va/ref2va)、`prompt`(H3-Context-IR 格式的长描述)、`conditions`(参考素材列表)、`target`(short_edge/aspect_ratio/duration_seconds)、`seed` + 采样参数。服务返回 `{"id":"<video_id>"}`。
2. **编码参考**——服务把 `conditions` 里的图/视频/音频各自过 VAE/Encoder 编进潜空间，和文本拼成 packed 序列。
   - **FL2VA**：`conditions` 是首帧(`role=keyframe, frame_index=0`)和/或末帧(`frame_index=-1`)；无图即纯 T2VA。
   - **Ref2VA**：`conditions` 是参考素材，`role=reference`，按模态独立编号(`<Picture N>`/`<Audio N>`/`<Video N>`)。注意视频参考自带音轨（其音轨算 `<Audio 1>`，独立音频文件算 `<Audio 2>`，依此类推）。
3. **扩散去噪**——Omni-Transformer 以 packed 序列为条件，联合预测视频潜变量 + 音频潜变量（CFG-distilled，默认 50 步，`flow_shift=12` / `audio_flow_shift=3`）。
4. **解码 + 取片**——视频潜变量过 VisualVAE 解码、音频潜变量过 AudioVAE 解码（立体声），mux 成 mp4。`GET /v1/videos/<id>` 轮询 `status`；`completed` 后 `GET /v1/videos/<id>/content` 下载二进制 mp4。

### Prompt 长什么样（为什么这么长）
官方 768p 样例的 prompt 是 **H3-Context-IR 的输出格式**（因为 H3-Context-IR 非开源，官方把它的输出直接给出来让你照抄复现）。结构：
- `integrated_multimodal_description:` 逐镜头的画面+运镜+动作描述（带时间码）；
- `overall_soundscape:` 环境音/拟音描述；
- `non_diegetic_music:` 配乐描述；
- Ref2VA 还有 `subject_definitions:` / `summary:` / `retention_analysis:` / `detailed_description:`（参考素材如何用、保留哪些）。

> 想自己写短 prompt 也能跑（`03_generate.sh` 的 `PROMPT="..."`），但效果不如 H3-Context-IR 格式的长描述——官方建议接 Context-IR API 或照「Prompting Guidance」自建预处理。本目录的三个样例直接用官方给的 Context-IR 输出，可严格复现。

## Serving (02 — SGLang 起 H3-Base 服务)
`02_serve.sh` 调 `sglang serve`：把 33B Transformer + Qwen3-VL-32B 编码器分片到多卡，起一个 HTTP 服务。并行度全可配，下表是 SGLang cookbook 的 **verified 配方**映射到你的硬件：

| 硬件 | 配方 | 命令 |
|---|---|---|
| **4× A100 80GB**（你的） | 容量（最稳，推荐先试） | `NUM_GPUS=4 ULYSSES_DEGREE=4 USE_FSDP=1 bash minimax_h3/02_serve.sh` |
| 4× A100 80GB | 最快（同 H100 配方） | `NUM_GPUS=4 TP_SIZE=2 ULYSSES_DEGREE=2 bash minimax_h3/02_serve.sh` |
| 4× H100 80GB | 最快 resident | `NUM_GPUS=4 TP_SIZE=2 ULYSSES_DEGREE=2 bash ...` |
| 4× H200 141GB / 8× B200 | resident（README 默认） | `NUM_GPUS=4 ULYSSES_DEGREE=4 bash ...` |
| 2× RTX 5090 32GB | 逐层 offload（慢） | `EXTRA_SGLANG_FLAGS="--performance-mode memory --layerwise-offload-components dit,text_encoder,vae --dit-layerwise-resident-layers 20" bash ...` |

- A100 是 Ampere（H100 是 Hopper），不在官方 verified 列表，但 4× A100 80GB **单卡显存 == H100 80GB**，故 H100 的 80GB 配方适用。
- 默认 `NUM_GPUS=4 ULYSSES_DEGREE=4`（README 官方示例）在 80GB 卡上可能 OOM——A100 务必备好 `USE_FSDP=1` 或 `TP_SIZE=2 ULYSSES_DEGREE=2` 兜底。
- `MODEL_VARIANT=fl2va`(默认, :30010) / `ref2va`(:30011)；`PORT` 可覆盖。
- `BG=1` 后台起 + 轮询 `/health` 等就绪（加载 33B 要几分钟）；`BG=0`(默认) 前台跑看日志。
- `GPU=0,1,2,3` 限制可见卡；多卡 serve 一般留空 + 用 `--num-gpus`。
- `EXTRA_SGLANG_FLAGS` 透传任意 SGLang 参数（如 `--quantization fp8`、offload 选项）。

## Generate (03 — 发请求 + 轮询 + 下载)
`03_generate.sh` 调 `generate.py`：根据 `TASK` 拼请求 body、提交、轮询、下载，打印耗时。任务→conditions 映射：
```bash
# T2VA（无 conditions）
TASK=t2va PROMPT="..." DURATION=10 ASPECT_RATIO=16:9 SEED=0 bash minimax_h3/03_generate.sh
# I2VA 首帧（FL2VA 权重）
TASK=fl2va FIRST_FRAME=/data/first.png DURATION=8 bash minimax_h3/03_generate.sh
# FL2VA 首末帧
TASK=fl2va FIRST_FRAME=/data/first.png LAST_FRAME=/data/last.png DURATION=8 bash minimax_h3/03_generate.sh
# Ref2VA 参考图+音频（Ref2VA 权重）
SERVER_URL=http://localhost:30011 TASK=ref2va \
  REF_IMAGES=/data/subject.png REF_AUDIOS=/data/voice.mp3 \
  PROMPT="Use <Picture 1> as the subject and <Audio 1> as the voice." bash minimax_h3/03_generate.sh
# Ref2VA 多参料（逗号分隔；视频可带 start_time_seconds）
SERVER_URL=http://localhost:30011 TASK=ref2va \
  REF_IMAGES=/data/a.png,/data/b.png REF_VIDEOS=/data/v1.mp4 REF_VIDEO_STARTS=0,0 \
  PROMPT="Combine <Picture 1>, <Picture 2>, <Video 1> ..." bash minimax_h3/03_generate.sh
```
- 本地路径自动转 `file://`（服务进程要能读该路径）；http(s) URL 原样传（官方样例用 CDN URL）。
- Ref2VA 的 condition **顺序**会影响模态编号。官方 ref2va 样例顺序是「视频先、音频后」（视频自带音轨算 `<Audio 1>`、音频文件算 `<Audio 2>`），故用 `CONDITIONS_FILE=examples/ref2va_conditions.json` 传 verbatim JSON 数组，绕开 env 构造的固定顺序，保证严格复现。
- `NUM_INFERENCE_STEPS=50 FLOW_SHIFT=12.0 AUDIO_FLOW_SHIFT=3.0`（SGLang cookbook 默认）；`NUM_OUTPUTS=N` 每次出 N 段。
- `POLL_INTERVAL=10` `TIMEOUT_MINS=30` 控制轮询。

## 复现官方三个 768p 样例
`examples/` 下三个脚本各跑一个官方样例（prompt 直接抄自 `scripts/readme/reproducible-768p-*-request.sh`，参考素材用官方 CDN URL）：
```bash
# FL2VA 服务先起好（bash minimax_h3/02_serve.sh，:30010）
bash minimax_h3/examples/run_t2va.sh     # 文生视频，10s 16:9，星舰舰长 -> t2va.mp4
bash minimax_h3/examples/run_fl2va.sh    # 首帧生视频，8s auto，拉面/家庭 -> fl2va.mp4
# Ref2VA 服务先起好（MODEL_VARIANT=ref2va bash minimax_h3/02_serve.sh，:30011）
bash minimax_h3/examples/run_ref2va.sh   # 参考视频+音频，5s auto，粉西装男 -> ref2va.mp4
```
对照官方结果：仓库 `assets/t2va.mp4` / `fl2va.mp4` / `ref2va.mp4`（在 https://github.com/MiniMax-AI/MiniMax-H3 的 assets 下）。

## Full 2K Workflow（调 MiniMax API，非开源部分）
要 2K 输出需把本地 768p 结果喂回 MiniMax API 的 H3-Regenerate-2K，并用 H3-Context-IR API 预处理自由 prompt。本目录不含这部分（要 API Token），步骤：
1. 本地起 H3-Base 服务（本目录 02）；
2. 调 `POST /video-generation-v2-h3-context-ir`（MiniMax API）把你的自由 prompt 转成 H3-Context-IR 格式长描述；
3. 用长描述调本地服务出 768p；
4. 把 768p 调 `POST /video-generation-v2-regeneration`（API）重生成到 2K。
API 文档：Global `platform.minimax.io` / CN `platform.minimaxi.com`（`/video-generation-v2-create`、`-h3-context-ir`、`-regeneration`）。

## 可能遇到的问题

公司代理做 HTTPS 中间人解密，按阶段列常见报错与修法（命令在服务器上、conda 环境已激活时执行）。

**1. clone/pull 本仓或官方仓报错**
- `SSL certificate problem`：公开仓加 `-c http.sslVerify=false`（`run_all.sh` 已对官方仓做兜底）。
- `Failed to connect to github.com port 443`（连不上，非 SSL）：git 没走代理。设全局代理（密码特殊字符必须 URL 编码）：
  ```bash
  git config --global https.proxy http://USER:PASS@proxyhk.huawei.com:8080
  git config --global http.proxy  http://USER:PASS@proxyhk.huawei.com:8080
  ```
- `No route to host` 连代理都不通：多为 docker 网桥网段和代理 IP 冲突。查 `getent hosts <proxy>` + `ip route | grep <网段>`，加主机路由 `sudo ip route add <代理IP> via <默认网关> dev <物理网卡>`，或改 `/etc/docker/daemon.json` 的 `default-address-pools` 给 docker 分不冲突的子网。

**2. `conda create` 报 `HTTP 403 FORBIDDEN for channel conda-forge`**
`~/.condarc` 把 `custom_channels` / `default_channels` 指向了清华 TUNA 镜像（`mirrors.tuna.tsinghua.edu.cn/anaconda/...`），该镜像偶发限流/抽风返回 403。`conda config --show channels` 只显示逻辑通道名（`conda-forge` / `defaults`），镜像重定向藏在 `custom_channels` / `default_channels` 里，要用 `conda config --show-sources` 才看得到（并标出是哪个 `.condarc` 文件设的）。两种修法：
- 临时绕开（只影响当条命令，推荐先试）——直连官方源，公司代理下 `conda.anaconda.org` / `repo.anaconda.com` 走 `http_proxy`/`https_proxy`（和 clone 本仓同一条路）：
  ```bash
  conda create -n minimax_h3 python=3.11 -y --override-channels \
    -c https://conda.anaconda.org/conda-forge -c https://repo.anaconda.com/pkgs/main
  ```
  若报 SSL（代理根 CA 不被信任）：先 `bash minimax_h3/setup_ca_bundle.sh` 建 `~/.ca-bundle.crt` 再重试。官方源也被代理挡时，把上面两个 `-c` 换成 BFSU 同源镜像（`https://mirrors.bfsu.edu.cn/anaconda/cloud/conda-forge`、`https://mirrors.bfsu.edu.cn/anaconda/pkgs/main`）。
- 永久修 `~/.condarc`（影响所有 conda 命令）：编辑 `~/.condarc`，把 TUNA 域名（`mirrors.tuna.tsinghua.edu.cn/anaconda`）整体替换成 BFSU（`mirrors.bfsu.edu.cn/anaconda`，同源更稳），或直接删掉 `custom_channels` / `default_channels` 两段回退到 conda 官方源。

**3. `pip install sglang[all]` 报 SSL/超时**
```bash
pip config set global.trusted-host "pypi.org pypi.python.org files.pythonhosted.org download.pytorch.org"
INSTALL_DEPS=1 bash minimax_h3/00_setup_env.sh
```
torch 大文件超时：先单独 `pip install --timeout 600 --retries 10 torch --index-url https://download.pytorch.org/whl/cu124`，再 `INSTALL_DEPS=1`。
> flashinfer（SGLang 默认注意力后端依赖）走自建下载源，代理下易失败：先 `bash minimax_h3/setup_ca_bundle.sh` 建 CA 包；仍失败可 `pip install flashinfer -f https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python` 或换 `EXTRA_SGLANG_FLAGS="--attention-backend triton"`。

**4. `hf download` 报 `CAS service error : ReqwestMiddleware`**
HF 的 Xet/Rust 通道不认代理。`_env.sh` 已设 `HF_HUB_DISABLE_XET=1`；仍报就 `pip uninstall -y hf_xet` 后重跑 `01`。

**5. `hf download` 报 `SSLCertVerificationError`**
代理根 CA 不在系统证书包。先 `bash minimax_h3/setup_ca_bundle.sh`（抓代理证书链→`~/.ca-bundle.crt` 并自检）；自检 `[OK]` 重跑 `01`，`[FAIL]` 把公司根 CA 追加到 `~/.ca-bundle.crt`；仍不行 `HF_DISABLE_SSL=1 bash minimax_h3/01_download_models.sh`（走 SSL 免校验兜底下载器 `_hf_download.py`）。

**6. `hf download` 报 401 / `repository not found`（MiniMax-H3 gated）**
HF 上 MiniMax-H3 是 Community License，可能需接受协议。去 https://huggingface.co/MiniMaxAI/MiniMax-H3 点 Accept，建 read token，再 `HF_TOKEN=<token> bash minimax_h3/01_download_models.sh`（脚本会透传给 `hf download` 和兜底下载器）。

**7. serve 报 OOM / CUDA out of memory（4× A100 80GB）**
默认 `NUM_GPUS=4 ULYSSES_DEGREE=4`（resident）在 80GB 卡上可能不够。按顺序试：
```bash
# a) FSDP 容量配方（verified on H100 80GB，同显存）
NUM_GPUS=4 ULYSSES_DEGREE=4 USE_FSDP=1 bash minimax_h3/02_serve.sh
# b) TP2 + Ulysses2（降单卡峰值显存）
NUM_GPUS=4 TP_SIZE=2 ULYSSES_DEGREE=2 bash minimax_h3/02_serve.sh
# c) offload（极慢但能塞下，2× RTX5090 才需要）
NUM_GPUS=4 EXTRA_SGLANG_FLAGS="--performance-mode memory --layerwise-offload-components dit,text_encoder,vae --dit-layerwise-resident-layers 20" bash minimax_h3/02_serve.sh
# 还可加 export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**8. serve 起不来：`sglang: command not found` / `import sglang` 失败**
没装 SGLang：`INSTALL_DEPS=1 bash minimax_h3/00_setup_env.sh`。装了还是 `command not found` 多半没 `conda activate minimax_h3`（脚本默认沿用当前 env）。`[diffusion]` extra 没带上时（模型类找不到）补 `pip install "sglang[diffusion]"`。

**9. `02_serve.sh` 后台模式一直 `not ready`**
看 `../MiniMax-H3/logs/serve_<variant>_<port>.log` 末尾：常见是权重路径错（`model_index.json` 缺 → 重跑 `01`）、CUDA/torch 不匹配、或多卡初始化卡住。脚本会在进程死掉时自动 `tail -n 40` 报错。健康检查超时可 `HEALTH_TIMEOUT_MINS=60 bash minimax_h3/02_serve.sh` 放宽。

**10. generate 报 `connection refused` / `submit failed HTTP 5xx`**
服务没起好或挂在别的端口。`curl -s http://localhost:30010/health` 验证；FL2VA 用 30010、Ref2VA 用 30011，`SERVER_URL` 要对上。5xx 多为请求体不合法（task/conditions 拼错）——看服务日志里的 traceback。

**11. Ref2VA 用本地参料，`file://` 路径服务读不到**
本地直装 SGLang 时 `file://` 用服务进程能读的绝对路径即可；若 SGLang 跑在 **docker** 里，容器看不到宿主机路径——需 `-v /data/minimax-h3:/data/minimax-h3:ro` 挂卷，参考素材放挂载目录下并用 `file:///data/minimax-h3/xxx`。最省事：把参料传到公网 URL，用 http URL。

**12. 跑 `.sh` 报 `syntax error near unexpected token ('`（CRLF 行尾）**
Windows→服务器用 scp/zip 等非 git 方式同步带过去。本仓 `.gitattributes` 强制 LF，但只有 `git checkout/pull` 才落 LF，非 git 传输不会转。
```bash
file minimax_h3/02_serve.sh           # 出现 "CRLF line terminators" 即中招
find minimax_h3 -name '*.sh' -exec sed -i 's/\r$//' {} +    # 一次性修所有 .sh
# 或 git checkout -- minimax_h3/
```

## Config (env vars, all optional)
| var | default | note |
|---|---|---|
| `CONDA_ENV` | 当前 env | 专用 `minimax_h3` 推荐（SGLang pin 与其他算法冲突） |
| `GPU` | _(unset)_ | 限制可见卡（`0,1,2,3`）；多卡 serve 一般留空用 `--num-gpus` |
| `MINIMAX_H3_DIR` | `../MiniMax-H3` | GitHub 参考仓（scripts/skills，serve 不依赖它） |
| `MODEL_DIR` / `MODEL_PATH` | `../../model/MiniMax-H3` | HF 权重快照（SGLang `--model-path` 指它） |
| `MINIMAX_H3_REPO` | 官方 GitHub URL | clone 源 |
| `INSTALL_DEPS` | `0`（run_all: `1`） | `1` = `pip install "sglang[all]"` |
| `HF_HUB_DISABLE_XET` | `1` | 关 HF Xet/Rust 通道（代理不友好） |
| `HF_DISABLE_SSL` | `0` | `1` = SSL 免校验下载权重 |
| `HF_TOKEN` | _(unset)_ | gated 仓才需（先去 HF 页面接受协议） |
| `DOWNLOAD_REF2VA` | `0` | `1` = 额外下 Ref2VA/（默认只下 FL2VA/） |

### Serve (02)
| var | default | note |
|---|---|---|
| `MODEL_VARIANT` | `fl2va` | `fl2va` \| `ref2va` |
| `HOST` / `PORT` | `0.0.0.0` / `30010`(fl2va) `30011`(ref2va) | |
| `NUM_GPUS` | `4` | 张数 |
| `ULYSSES_DEGREE` | `=NUM_GPUS` | Ulysses 注意力并行度 |
| `TP_SIZE` | _(unset)_ | 张量并行；H100/A100 80GB 最快配 `2` |
| `USE_FSDP` | `0` | `1` = `--use-fsdp-inference true`（80GB 卡容量配方） |
| `PERFORMANCE_MODE` | `speed` | `speed` \| `memory`（offload 用 memory） |
| `EXTRA_SGLANG_FLAGS` | _(unset)_ | 透传任意 `sglang serve` 参数 |
| `BG` | `0` | `1` = 后台起 + 轮询 `/health` 等就绪 |
| `HEALTH_TIMEOUT_MINS` | `30` | 后台模式等就绪超时 |
| `LOG_DIR` / `LOG_FILE` | `../MiniMax-H3/logs/...` | 服务日志 |

### Generate (03)
| var | default | note |
|---|---|---|
| `SERVER_URL` | `http://localhost:$PORT` | 也可 `host:port`；Ref2VA 改 `:30011` |
| `TASK` | `t2va` | `t2va` \| `fl2va` \| `ref2va` |
| `PROMPT` / `PROMPT_FILE` | _(t2va 内置默认)_ | inline 优先；文件次之 |
| `DURATION` | `5` | 4–15 秒 |
| `ASPECT_RATIO` | t2va=`16:9` / fl2va·ref2va=`auto` | `16:9` `9:16` `1:1` `auto` |
| `SHORT_EDGE` | `768` | 短边像素 |
| `SEED` | `0` | |
| `FIRST_FRAME` / `LAST_FRAME` | _(unset)_ | FL2VA 首末帧（路径或 URL） |
| `REF_IMAGES` / `REF_AUDIOS` / `REF_VIDEOS` | _(unset)_ | Ref2VA 参料（逗号分隔列表） |
| `REF_VIDEO_STARTS` | _(unset)_ | 视频起始秒（逗号分隔，对应 REF_VIDEOS） |
| `VIDEO_AS_AUDIO_REF` | `0` | `1` = 视频参料带音轨（`type=video_audio`） |
| `CONDITIONS_FILE` | _(unset)_ | JSON 数组，verbatim 覆盖 REF_* 构造（复现官方顺序用） |
| `NUM_INFERENCE_STEPS` / `FLOW_SHIFT` / `AUDIO_FLOW_SHIFT` | `50` / `12.0` / `3.0` | 采样参数（cookbook 默认） |
| `NUM_OUTPUTS` | `1` | 每次出几段 |
| `OUTPUT_DIR` / `OUTPUT_NAME` | `../MiniMax-H3/results/<task>` / `<task>_seed<seed>.mp4` | |
| `POLL_INTERVAL` / `TIMEOUT_MINS` | `10` / `30` | 轮询间隔 / 超时 |

## Outputs
- **02 serve**: 日志 `../MiniMax-H3/logs/serve_<variant>_<port>.log` + PID 文件 `serve_<variant>_<port>.pid`（BG 模式）。
- **03 generate**: 视频 `../MiniMax-H3/results/<task>/<name>.mp4`（含原生立体声）。

## 目录布局
```
<code-dir>/
├── media_code/                  # 本仓
│   ├── proxy.env                # 代理 + 覆盖项, gitignored
│   └── minimax_h3/             # 编排脚本(本目录)
├── MiniMax-H3/                  # GitHub 参考仓(自动 clone 到 ../MiniMax-H3；含 scripts/skills)
│   ├── results/                # 生成视频输出
│   └── logs/                    # 服务日志
└── ../../model/MiniMax-H3/      # HF 权重快照(<code-dir> 上一级, 各算法共享)
    ├── model_index.json         # 仓库级公共入口(SGLang 读)
    ├── FL2VA/                   # FL2VA 任务族(transformer/text_encoder/tokenizer/processor/visual_vae/audio_vae)
    └── Ref2VA/                  # Ref2VA 任务族(DOWNLOAD_REF2VA=1 才下)
```
默认：参考仓 `../MiniMax-H3`、权重 `../../model/MiniMax-H3`（相对本目录）；用 `MINIMAX_H3_DIR` / `MODEL_DIR` 覆盖。SGLang 服务**只依赖 HF 权重快照**，GitHub 参考仓纯为方便看 scripts/skills。

## Notes
- 官方代码与权重遵循其自有 license（MiniMax H3 Community License，非商用——见 HF 仓库 LICENSE）。本目录只做编排，不复制官方代码。
- `.gitattributes`（仓根）强制 LF，Windows 推上来的脚本在 Ubuntu 上干净运行。
- `proxy.env`（代理凭证）gitignored，不入库；切勿把凭证写进脚本。
- 公司 TLS 拦截代理下：pip 用 `--trusted-host`；`hf`/`git` 用 CA bundle（`_env.sh` 优先 `~/.ca-bundle.crt`，由 `setup_ca_bundle.sh` 构建）。
