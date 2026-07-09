# FLUX.1 runner

在 Ubuntu + NVIDIA 服务器上跑 [FLUX.1](https://github.com/black-forest-labs/flux) 的**文生图（text-to-image）批量推理**——给一条或多条文本提示词，扩散模型逐条生成 PNG 图片。本目录只含编排脚本——权重从 HuggingFace 下载，模型经 `diffusers` 的 `FluxPipeline` 加载，**无需 clone 官方代码仓**（FLUX.1 的 HF 快照本身就是完整的 diffusers 格式）。

默认用 **FLUX.1-schnell**（Apache-2.0、公开仓、无需 token、4 步出图）；换 **FLUX.1-dev**（ gated、需 token、28 步、质量更高）只改 `HF_REPO_ID` + 传 `HF_TOKEN`。

## 常用命令

> 假设已进入容器并 `conda activate doll`（或专用 env `flux1`）；`GPU=0` 按需换卡；路径取各脚本默认值（可改）。首次跑前先做下方「首次准备」。

```bash
# ── 一键流程（首次：INSTALL_DEDS=1 装依赖 + 下权重 + 出一张样图） ──
INSTALL_DEPS=1 bash flux1/run_all.sh

# ── 推理(02) ──
# 1) 单条提示词出一张图（默认 PROMPT 即熊猫那张）
GPU=0 PROMPT="a cyberpunk city at night, neon reflections, ultra detailed" bash flux1/02_run_inference.sh
# 2) 批量：一个文件每行一条提示词 -> 逐条出图
GPU=0 PROMPTS_FILE=/data/prompts.txt bash flux1/02_run_inference.sh
# 3) 换 gated 高质量 FLUX.1-dev（先接受许可 + 传 token 下载，再推理用 dev 的步数/cfg）：
HF_REPO_ID=black-forest-labs/FLUX.1-dev HF_TOKEN=<your_token> bash flux1/01_download_models.sh
GPU=0 HF_REPO_ID=black-forest-labs/FLUX.1-dev NUM_INFERENCE_STEPS=28 GUIDANCE_SCALE=3.5 MAX_SEQUENCE_LENGTH=512 PROMPT="..." bash flux1/02_run_inference.sh
# 4) 改尺寸 / 每条提示词出多张：
GPU=0 PROMPT="..." WIDTH=768 HEIGHT=1024 NUM_IMAGES_PER_PROMPT=4 bash flux1/02_run_inference.sh
# 5) 显存不够：顺序 offload（最省显存）/ VAE tiling / fp16
GPU=0 PROMPT="..." OFFLOAD=sequential VAE_TILING=1 DTYPE=fp16 bash flux1/02_run_inference.sh
# 6) 全显存驻留（最快，需 ~24GB 单卡 bf16）：
GPU=0 PROMPT="..." OFFLOAD=none bash flux1/02_run_inference.sh
```

- 结果：`../FLUX1/results/prompt/result/<0001>_<提示词slug>.png`（生成的图）+ `.../prompt/<0001>_<slug>.txt`（用到的提示词）。
- `GPU=N` 钉单卡。**FLUX.1 单卡跑，不自动多卡切分**——显存吃紧用 `OFFLOAD=model`(默认)/`sequential`，别指望多卡 sharding。
- 默认 `FLUX.1-schnell`（4 步、cfg=0、max_seq_len=256）；`dev` 的推荐值是 `NUM_INFERENCE_STEPS=28 GUIDANCE_SCALE=3.5 MAX_SEQUENCE_LENGTH=512`。

## 首次准备
```bash
cd <your-code-dir>            # e.g. /data_3d/<uid>/code
git -c http.sslVerify=false clone https://github.com/CrescentVelvet/media_code.git
cd media_code && cp proxy.env.example proxy.env   # 填 http_proxy / https_proxy（公司代理用；自用网络可跳过）
conda activate doll                                  # 或专用 env：conda create -n flux1 python=3.10 -y && conda activate flux1
INSTALL_DEPS=1 bash flux1/00_setup_env.sh           # 装 diffusers(>=0.30) + transformers + accelerate + Pillow + sentencepiece
bash flux1/01_download_models.sh                    # 下 black-forest-labs/FLUX.1-schnell -> ../../model/FLUX1/  (公开仓，免 token)
```
> 用 **FLUX.1-dev**（gated）：1) 在 https://huggingface.co/black-forest-labs/FLUX.1-dev 接受许可；2) 建 read token；3) `HF_REPO_ID=black-forest-labs/FLUX.1-dev HF_TOKEN=<token> bash flux1/01_download_models.sh`。`01` 对 dev 仓无 token 会直接报错并给出指引。
> FLUX.1 需要 `diffusers>=0.30`（Flux 支持在此版本合入）。`doll` env（hunyuanvideo-1.5 已装 diffusers==0.35.0）通常满足；若偏旧用专用 env（`CONDA_ENV=flux1`）。

---

以下为详细参考（流程原理 / 各脚本参数 / 排错 / 目录布局）。

## Inference (02 — 更多用法)
`02_run_inference.sh` 调 `run_inference.py`：**FluxPipeline 只加载一次**，读提示词（`PROMPTS_FILE` 每行一条，或单条 `PROMPT`），逐条 `pipe(prompt, steps, guidance, height, width, max_sequence_length, generator)` 生成，存 `result/<idx>_<slug>.png`，并打印每条耗时与汇总（avg/min/max/total）。常见覆盖示例：
```bash
# 长/短边不同（竖图 1024x768）：
GPU=0 PROMPT="a tall lighthouse on a cliff" WIDTH=768 HEIGHT=1024 bash flux1/02_run_inference.sh
# 固定种子复现 + 每条出 3 张变体：
GPU=0 PROMPTS_FILE=/data/prompts.txt SEED=42 NUM_IMAGES_PER_PROMPT=3 bash flux1/02_run_inference.sh
# flash-attention（更快更省显存，需 pip install flash-attn --no-build-isolation；进阶）：
GPU=0 PROMPT="..." ATTN_IMPL=flash_attention_2 bash flux1/02_run_inference.sh
# 输出到别处：
GPU=0 PROMPT="..." OUTPUT_DIR=/data/out bash flux1/02_run_inference.sh
```

## Pipeline（推理流程详解）
一条提示词从文本到出图，经以下几步（对应 `run_inference.py`）：

1. **加载 pipeline 一次** — `FluxPipeline.from_pretrained(MODEL_PATH, torch_dtype=bf16)`。这条管线自带：T5 (`google/t5-v1_1-xxl`) + CLIP (`openai/clip-vit-large-patch14`) 两个文本编码器、`FluxTransformer2DModel`（12B 主体）、`AutoencoderKL`（VAE）、流匹配 scheduler。可选 `ATTN_IMPL` 单独加载 transformer 指定注意力（失败自动回退 sdpa）。
2. **显存策略（互斥）** — `OFFLOAD=model`(默认) `enable_model_cpu_offload()`：模块按需搬上 GPU，约 12GB 即可跑；`sequential` 更细粒度、更省但更慢；`none` 整管线上 GPU（dev 约 24GB bf16，最快）。另 `VAE_SLICING/TILING` 降 VAE 解码显存。**注意：diffusers FluxPipeline 不做多卡 sharding**，`GPU=N` 只选哪张卡，不跨卡切模型。
3. **编码文本** — T5+CLIP 把 prompt 编成文本嵌入（`max_sequence_length` 限制 T5 截断长度：schnell 256 / dev 512）。
4. **流匹配采样** — `pipe(num_inference_steps=..., guidance_scale=..., generator=...)`。schnell 4 步、`guidance=0`（不做 CFG，快）；dev 28 步、`guidance=3.5`（带 CFG，质量高）。`generator=torch.Generator("cpu").manual_seed(SEED+i)` 保证逐条可复现（每条用不同但确定的种子）。
5. **VAE 解码 + 存图** — 潜变量 → 像素 PNG，按 `<idx>_<slug>.png` 存到 `result/`，提示词存 `prompt/<idx>_<slug>.txt`。

> 总结一句：**提示词 → T5/CLIP 编码 → FluxTransformer 流匹配采样(steps×) → VAE 解码 → 存 PNG**。管线只加载一次，循环逐条出图。

> schnell vs dev：schnell 是 4 步蒸馏版（Apache-2.0、公开、出图快、cfg=0）；dev 是 28 步原版（gated、质量更好、cfg=3.5）。同一条 prompt 两者都能出，只是步数/cfg/许可不同。

## Config (env vars, all optional)
| var | default | note |
|---|---|---|
| `CONDA_ENV` | `doll` | conda env（专用 `flux1` 推荐，若怕 diffusers 升级影响其他算法） |
| `GPU` | _(unset)_ | 物理卡号，如 `GPU=0`（Flux 单卡跑，不跨卡切分） |
| `MODEL_DIR` | `../../model/FLUX1` | 权重根目录 |
| `HF_REPO_ID` | `black-forest-labs/FLUX.1-schnell` | HF 仓库 ID（下权重 01 与推理 02 传同一个） |
| `MODEL_PATH` | `$MODEL_DIR/<repo_basename>` | 本地快照目录（自动由 HF_REPO_ID 推导） |
| `INSTALL_DEPS` | `0` (run_all: `1`) | `1` = 装 diffusers/transformers/accelerate/Pillow/sentencepiece |
| `HF_HUB_DISABLE_XET` | `1` | 关 HF Xet/Rust 通道（代理不友好） |
| `HF_DISABLE_SSL` | `0` | `1` = 关 SSL 校验下权重（代理 MITM 证书不通时） |
| `HF_TOKEN` | _(unset)_ | **dev 必填**（gated）；schnell 无需 |
| `INCLUDE_PATTERNS` | _(空=全下)_ | 01 下权重时的 glob 过滤，如 `transformer/*,*.json,*.txt` |

### Inference (02)
| var | default | note |
|---|---|---|
| `PROMPT` | `A cinematic shot of a panda…` | 单条提示词（PROMPTS_FILE 未设时用） |
| `PROMPTS_FILE` | _(unset)_ | 提示词文件，每行一条（`#` 开头跳过）；设了就批量 |
| `OUTPUT_DIR` | `../FLUX1/results/prompt` | 写 `result/` + `prompt/` |
| `NUM_INFERENCE_STEPS` | `4` | schnell=4；dev=28 |
| `GUIDANCE_SCALE` | `0.0` | schnell=0.0（无 CFG）；dev=3.5 |
| `HEIGHT` / `WIDTH` | `1024` / `1024` | 出图尺寸 |
| `MAX_SEQUENCE_LENGTH` | `256` | T5 截断长度；schnell=256，dev=512 |
| `NUM_IMAGES_PER_PROMPT` | `1` | 每条提示词出几张 |
| `SEED` | `231` | 逐条用 `SEED+i`，可复现 |
| `DTYPE` | `bf16` | `bf16`(推荐) \| `fp16` \| `fp32` |
| `OFFLOAD` | `model` | `model`(~12GB) \| `sequential`(最省最慢) \| `none`(整管线上GPU，最快) |
| `VAE_SLICING` / `VAE_TILING` | `1` / `0` | 降 VAE 解码显存；tiling 用于大图/极低显存 |
| `ATTN_IMPL` | _(空=sdpa)_ | `flash_attention_2` \| `eager`（进阶；需对应 attn 库） |

## Outputs
- **02 inference**: `OUTPUT_DIR/result/<idx>_<slug>[_vN].png`（生成的图）+ `OUTPUT_DIR/prompt/<idx>_<slug>[_vN].txt`（用到的提示词）。

## 目录布局
```
<code-dir>/
├── media_code/                  # 本仓
│   ├── proxy.env                # 代理 + 覆盖项, gitignored
│   └── flux1/                   # 编排脚本(本目录)
└── ../../model/FLUX1/           # 权重(在 <code-dir> 上一级, 各算法共享)
    └── FLUX.1-schnell/          # HF 快照(model_index.json + transformer/ + vae/ + text_encoder/ + ...)
```
默认：权重 `../../model/FLUX1`、输出 `../FLUX1/results/`（相对本目录）；用 `MODEL_DIR` / `OUTPUT_DIR` 覆盖。复用现有 conda env（默认 `doll`）；FLUX.1 的 diffusers 版本要求 `>=0.30`——若与其他算法冲突，用专用 env（`CONDA_ENV=flux1`）。

## 可能遇到的问题

公司代理做 HTTPS 中间人解密，下面列常见报错与修法（命令在服务器上、conda 环境已激活时执行）。

**1. `hf download` 报 `SSLCertVerificationError` / `CAS service error : ReqwestMiddleware`**
代理根 CA 不在系统证书包，或 HF 的 Xet/Rust 通道不认代理。`_env.sh` 已设 `HF_HUB_DISABLE_XET=1`；仍报则建 CA 包：
```bash
bash flux1/setup_ca_bundle.sh     # 抓代理证书链 -> ~/.ca-bundle.crt，并自检
bash flux1/01_download_models.sh
```
- 自检 `[OK]` → 直接重跑 `01`。
- 自检 `[FAIL]` → 把公司根 CA 追加到 `~/.ca-bundle.crt` 后重跑。
- 仍报 SSL（CDN 端点用不同 MITM 证书）→ `01` 自动回退到禁用 SSL 校验的下载器（`_hf_download.py`）；或直接 `HF_DISABLE_SSL=1 bash flux1/01_download_models.sh`。

**2. 下载 dev 报 `repository not found` / 401 / GATED**
`FLUX.1-dev` 是 gated 仓，HF 对无 token 请求返回 "not found" 实为 401。`01` 对 dev 仓无 token 会直接报错指路。修法：1) 在 https://huggingface.co/black-forest-labs/FLUX.1-dev 接受许可；2) 建 read token；3) `HF_REPO_ID=black-forest-labs/FLUX.1-dev HF_TOKEN=<token> bash flux1/01_download_models.sh`（脚本自动透传 token 给 `hf download` 和 SSL 兜底下载器）。或用默认 schnell（公开免 token）。

**3. 推理报 `ImportError: diffusers` / `FluxPipeline` 不存在 / `cannot import name 'FluxTransformer2DModel'`**
`diffusers` 太旧（<0.30，Flux 支持未合入）。升级：
```bash
pip install -U "diffusers>=0.30"
```
或在专用 env 跑：`CONDA_ENV=flux1 INSTALL_DEPS=1 bash flux1/00_setup_env.sh`。

**4. 推理 OOM（显存不足）**
- 默认已开 `OFFLOAD=model`（约 12GB 可跑）。仍不够：`OFFLOAD=sequential VAE_TILING=1`（最省，最慢）。
- `DTYPE=fp16`（注意 Flux 官方推荐 bf16，fp16 偶有数值问题，作为兜底）。
- 降尺寸 `WIDTH=512 HEIGHT=512`、降步数 `NUM_INFERENCE_STEPS=4`。
- 仍紧张：`export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
- 想最快且显存够（≥24GB）：`OFFLOAD=none`。

**5. 出图慢 / 想多卡**
FluxPipeline **不自动多卡切分**。单卡提速：装 flash-attn 后 `ATTN_IMPL=flash_attention_2`；或显存够时 `OFFLOAD=none`。schnell 本就 4 步很快；dev 28 步较慢，可用 schnell 先试效果。

**6. 出图全黑 / 噪点 / 比例错乱**
多为尺寸非 8 的倍数或 fp16 数值溢出。Flux VAE 要求 H/W 是 8（patch 后是 2）的倍数——脚本默认 1024 满足；自定义尺寸确保 `%8==0`。fp16 溢出改回 `DTYPE=bf16`。

> 通用：`proxy.env`（代理凭证）在仓内 gitignored，`~/.ca-bundle.crt` 在家目录，都不入库；切勿把凭证写进脚本。

## Notes
- 模型权重遵循其自身许可：FLUX.1-schnell = Apache-2.0（可商用）；FLUX.1-dev = **非商用** + gated（需接受许可）。本目录只编排，不复制任何官方代码。用 dev 生成的图受其许可约束。
- `.gitattributes`（仓根）强制 LF，Windows 推上去的脚本在 Ubuntu 上也能干净运行。
- `proxy.env`（代理凭证 / env 覆盖）gitignored，绝不入库。别把凭证写进脚本。
- 公司 TLS 拦截代理下：pip 用 `--trusted-host`；`hf`/`git` 用 CA 包（`_env.sh` 优先 `~/.ca-bundle.crt`，由 `setup_ca_bundle.sh` 构建）。
