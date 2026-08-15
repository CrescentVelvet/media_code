# FLUX.2 — 文生图 + 图像编辑（text-to-image & editing）批量推理

在 Ubuntu + NVIDIA 服务器上跑 [FLUX.2](https://github.com/black-forest-labs/flux2) 的**文生图**与**图像编辑/多参考**批量推理——给一条或多条文本提示词（可选配 1~N 张参考图），扩散模型逐条生成 PNG。**无需 clone 官方代码仓**：FLUX.2 的 HuggingFace 快照本身就是完整的 diffusers 格式（transformer + VAE + 文本编码器 + scheduler），经 `Flux2Pipeline` / `Flux2KleinPipeline` 直接加载。权重从 HuggingFace 下载到 `../../model/FLUX2/`。

支持两个模型（`MODEL_TYPE` 切换），同一脚本同一管线：

| 模型 | 管线类 | 文本编码器 | 蒸馏 | 默认步数 | 默认 cfg | 显存 |
|---|---|---|---|---|---|---|
| **FLUX.2-klein-9B**（默认，快） | `Flux2KleinPipeline` | Qwen3 (8B) | 步数+引导双蒸馏 | 4 | 1.0（引导已 baked，忽略） | 消费级 GPU 即可（`OFFLOAD=model`） |
| **FLUX.2-dev**（最高质量） | `Flux2Pipeline` | Mistral3 (24B) | 仅引导蒸馏 | 50（28 亦可） | 4.0 | H100 级（`OFFLOAD=model`）或消费级（`QUANT=4bit`） |

两者**同时支持**：文生图 ✅ | 单参考编辑 ✅ | 多参考编辑 ✅。FLUX.2 抛弃了 FLUX.1 的 T5+CLIP，改用 LLM 作文本编码器（dev=Mistral3、klein=Qwen3），故无需 `sentencepiece`。

## 常用命令
> 假设已进入容器；首次跑前先做下方「首次准备」。
> **铁律：每条命令都必须显式写出模型地址、输入路径、输出路径，不能全靠脚本里的默认值。** 用具体路径，不要用 `...` 占位。

```bash
# ── 一键（首次：INSTALL_DEPS=1 建 env + 升级 diffusers + 下权重 + 出一张样图） ──
# 默认 FLUX.2-klein-9B（快、4 步、消费级 GPU 能跑）
GPU=0 \
  MODEL_TYPE=klein \
  MODEL_DIR=../../model/FLUX2 \
  OUTPUT_DIR=../FLUX2/results/prompt \
  PROMPT="a cyberpunk city at night, neon reflections, ultra detailed" \
  INSTALL_DEPS=1 bash flux2/run_all.sh

# ── 分步 ──
# 1) 下权重 -> ../../model/FLUX2/FLUX.2-klein-9B （两个仓都 gated，需 HF_TOKEN）
HF_TOKEN=<your_token> \
  MODEL_DIR=../../model/FLUX2 \
  MODELS_TO_DOWNLOAD="klein" \
  bash flux2/01_download_models.sh
#    也要 dev：MODELS_TO_DOWNLOAD="dev klein" （或单独 "dev"）

# 2) 推理：单条提示词出一张图（klein） -> ../FLUX2/results/prompt/result/
GPU=0 \
  MODEL_TYPE=klein \
  MODEL_PATH=../../model/FLUX2/FLUX.2-klein-9B \
  OUTPUT_DIR=../FLUX2/results/prompt \
  PROMPT="a cyberpunk city at night, neon reflections, ultra detailed" \
  bash flux2/02_run_inference.sh

# 2b) 批量：一个文件每行一条提示词 -> 逐条出图
GPU=0 \
  MODEL_TYPE=klein \
  MODEL_PATH=../../model/FLUX2/FLUX.2-klein-9B \
  OUTPUT_DIR=../FLUX2/results/prompt \
  PROMPTS_FILE=../data/prompts.txt \
  bash flux2/02_run_inference.sh

# 2c) 图像编辑 / 多参考：1~N 张参考图 + 提示词 -> 编辑/合成出图
#     （单参考=编辑，多参考=把多张图的特征融合/组合）
GPU=0 \
  MODEL_TYPE=klein \
  MODEL_PATH=../../model/FLUX2/FLUX.2-klein-9B \
  OUTPUT_DIR=../FLUX2/results/edit \
  INPUT_IMAGES="../data/cat.jpg,../data/dog.jpg" \
  PROMPT="a cat wearing sunglasses" \
  bash flux2/02_run_inference.sh

# 2d) 换 FLUX.2-dev（最高质量；需 H100 级，或消费级加 QUANT=4bit）
GPU=0 \
  MODEL_TYPE=dev \
  MODEL_PATH=../../model/FLUX2/FLUX.2-dev \
  OUTPUT_DIR=../FLUX2/results/prompt_dev \
  PROMPT="a cyberpunk city at night, neon reflections, ultra detailed" \
  NUM_INFERENCE_STEPS=50 GUIDANCE_SCALE=4.0 \
  bash flux2/02_run_inference.sh

# 2e) dev 在消费级 GPU（如 RTX 4090）跑：4-bit 量化（需 bitsandbytes）
#     或更省事：MODEL_PATH 指向预量化仓 diffusers/FLUX.2-dev-bnb-4bit（无需 QUANT）
GPU=0 \
  MODEL_TYPE=dev \
  MODEL_PATH=../../model/FLUX2/FLUX.2-dev \
  OUTPUT_DIR=../FLUX2/results/prompt_dev_4bit \
  QUANT=4bit OFFLOAD=model \
  PROMPT="a cyberpunk city at night, neon reflections, ultra detailed" \
  bash flux2/02_run_inference.sh

# 2f) 改尺寸 / 每条提示词出多张变体
GPU=0 \
  MODEL_TYPE=klein \
  MODEL_PATH=../../model/FLUX2/FLUX.2-klein-9B \
  OUTPUT_DIR=../FLUX2/results/prompt \
  PROMPT="a tall lighthouse on a cliff" WIDTH=1360 HEIGHT=768 NUM_IMAGES_PER_PROMPT=4 \
  bash flux2/02_run_inference.sh

# 2g) 全显存驻留（最快，需显存能装下整管线）
GPU=0 \
  MODEL_TYPE=klein \
  MODEL_PATH=../../model/FLUX2/FLUX.2-klein-9B \
  OUTPUT_DIR=../FLUX2/results/prompt \
  PROMPT="..." OFFLOAD=none \
  bash flux2/02_run_inference.sh
```

- 结果：`../FLUX2/results/prompt/result/<0001>_<slug>.png`（生成的图）+ `.../prompt/<0001>_<slug>.txt`（用到的提示词）。
- `GPU=N` 钉单卡。**FLUX.2 单卡跑，管线不自动多卡 sharding**——显存吃紧用 `OFFLOAD=model`(默认)/`sequential`，别指望多卡切分。
- 默认 `klein`（4 步、cfg=1.0、引导已蒸馏故 cfg 被忽略）；`dev` 推荐值 `NUM_INFERENCE_STEPS=50 GUIDANCE_SCALE=4.0`（28 步是质量/速度折中）。
- 编辑模式：`INPUT_IMAGES` 逗号分隔 1~N 张参考图，配 `PROMPT` 描述编辑/合成指令；不设 `INPUT_IMAGES` 即纯文生图。

## 首次准备
```bash
# clone 本仓 + proxy.env
cd <your-code-dir>            # e.g. /data_3d/<uid>/code
git -c http.sslVerify=false clone https://github.com/CrescentVelvet/media_code.git
cd media_code && cp proxy.env.example proxy.env
# ⚠️ 确认 proxy.env 中 http_proxy / https_proxy 已取消注释（公司代理用；自用网络可跳过）

# 建 env（克隆共享 doll env -> flux2，复用其 CUDA torch；再升级 diffusers 到 0.39）
INSTALL_DEPS=1 bash flux2/00_setup_env.sh
#   doll env 须已存在（含 CUDA torch）。克隆到 flux2 是为了不污染 doll（其他算法仍用 diffusers 0.35）。
#   升级内容：diffusers==0.39.0（Flux2Pipeline 在此版本合入）+ transformers + accelerate。
#   bitsandbytes（4-bit 量化）默认不装；要 QUANT=4bit 时再 `pip install bitsandbytes`。

# 下权重 -> ../../model/FLUX2/FLUX.2-klein-9B （gated，先在 HF 页面接受许可 + 建 read token）
HF_TOKEN=<your_token> \
  MODEL_DIR=../../model/FLUX2 \
  MODELS_TO_DOWNLOAD="klein" \
  bash flux2/01_download_models.sh
```
> 两个仓都 gated（FLUX Non-Commercial License）：1) 在 https://huggingface.co/black-forest-labs/FLUX.2-klein-9B 与 https://huggingface.co/black-forest-labs/FLUX.2-dev 接受许可；2) 建 read token；3) `HF_TOKEN=<token> ... bash flux2/01_download_models.sh`。`01` 对无 token 会直接报错指路。
> 若你**已手动下好权重**（diffusers 格式，含 `model_index.json` + 子目录），跳过 `01`，直接 `02` 并设 `MODEL_PATH` 指向快照目录即可。
> FLUX.2 需要 `diffusers>=0.39`（`Flux2Pipeline`/`Flux2KleinPipeline` 在此版本合入；doll 的 0.35 没有）。专用 `flux2` env 已满足。

权重目录布局：
```
$MODEL_DIR/                          # = ../../model/FLUX2
├── FLUX.2-klein-9B/                 # HF 快照（model_index.json + transformer/ + vae/ + text_encoder/ + tokenizer/ + scheduler/）
└── FLUX.2-dev/                      # 同上（text_encoder/ = Mistral3）
```

---

以下为详细参考。

## Pipeline（推理流程详解）
```
提示词(可选+参考图) -> [01] 下 HF 快照 -> [02] 管线加载一次 -> 文本编码 -> 流匹配采样 -> VAE 解码 -> 存 PNG
```

一条提示词从文本到出图，经以下几步（对应 `run_inference.py`）：

1. **加载管线一次** — 按 `MODEL_TYPE` 选类：`dev`→`Flux2Pipeline`（文本编码器 Mistral3-24B）、`klein`→`Flux2KleinPipeline`（文本编码器 Qwen3-8B）。`from_pretrained(MODEL_PATH, torch_dtype=bf16)` 一次载入 transformer（`Flux2Transformer2DModel`，dev=32B / klein=9B）+ VAE（`AutoencoderKLFlux2`，改进版）+ 文本编码器 + tokenizer + scheduler。可选 `QUANT=4bit/8bit` 用 `BitsAndBytesConfig` 把 transformer + 文本编码器在加载时量化（dev 上消费级 GPU 的关键；镜像官方 4-bit 指南）。
2. **显存策略（互斥）** — `OFFLOAD=model`(默认) `enable_model_cpu_offload()`：模块按需上 GPU，dev（32B+24B）H100 也得靠它（文本编码完搬回 CPU 再跑 transformer）；`sequential` 更细粒度、更省但更慢；`none` 整管线上 GPU（klein 在 ≥40GB 卡可全驻留，最快）。**管线不做多卡 sharding**，`GPU=N` 只选哪张卡。
3. **文本编码** — dev 用 Mistral3（取第 10/20/30 层 hidden states 拼接）、klein 用 Qwen3（取第 9/18/27 层），把 prompt 编成文本嵌入；`max_sequence_length=512`。二者都用各自 LLM 的 chat template（Mistral3 带 system message；Qwen3 关闭 thinking）。
4. **流匹配采样** — `pipe(num_inference_steps=..., guidance_scale=..., generator=...)`。klein 4 步（步数+引导双蒸馏，`guidance` 被忽略）、`guidance=1.0`（no-op）；dev 50 步、`guidance=4.0`（引导蒸馏，单次 forward 近似 CFG）。`generator=torch.Generator("cuda").manual_seed(SEED+i)` 逐条可复现。若有参考图，参考图的 VAE token 与文本/噪声 token 一起做注意力（多参考编辑）。
5. **VAE 解码 + 存图** — 潜变量 → 像素 PNG，按 `<idx>_<slug>.png` 存 `result/`，提示词存 `prompt/<idx>_<slug>.txt`。

> 总结一句：**提示词(+参考图) → LLM 文本编码 → Flux2Transformer 流匹配采样(steps×) → VAE 解码 → 存 PNG**。管线只加载一次，循环逐条出图。
> dev vs klein：dev 32B+Mistral3 24B，质量顶、需 H100 级或量化；klein 9B+Qwen3 8B，4 步蒸馏、亚秒级、消费级 GPU 能跑。两者同管同脚本，只换 `MODEL_TYPE`。

## Config (env vars)
| var | default | note |
|---|---|---|
| `CONDA_ENV` | `flux2` | 专用 env（`00` 克隆自 `doll` 并升级 diffusers）；`conda activate flux2` 一次后免传 |
| `GPU` | _(unset)_ | 物理卡号，如 `GPU=0`（Flux2 单卡跑，不跨卡切分） |
| `MODEL_DIR` | `../../model/FLUX2` | 权重根目录 |
| `MODEL_TYPE` | `klein` | `klein`(Flux2KleinPipeline/Qwen3) \| `dev`(Flux2Pipeline/Mistral3) |
| `MODEL_PATH` | `$MODEL_DIR/FLUX.2-klein-9B` 或 `.../FLUX.2-dev` | 本地快照目录（02 用，按 MODEL_TYPE 推导） |
| `INSTALL_DEPS` | `0` (run_all: `1`) | `1` = 克隆 doll->flux2 + 升级 diffusers/transformers/accelerate |
| `MODELS_TO_DOWNLOAD` | `dev klein` | 01 下哪几个（`dev` / `klein` / 空格分隔） |
| `HF_HUB_DISABLE_XET` | `1` | 关 HF Xet/Rust 通道（代理不友好） |
| `HF_DISABLE_SSL` | `0` | `1` = 关 SSL 校验下权重（代理 MITM 证书不通时） |
| `INCLUDE_PATTERNS` | _(空=全下)_ | 01 下权重时的 glob 过滤 |

### Inference (02)
| var | default | note |
|---|---|---|
| `PROMPT` | `A cinematic shot of a panda…` | 单条提示词（PROMPTS_FILE 未设时用） |
| `PROMPTS_FILE` | _(unset)_ | 提示词文件，每行一条（`#` 开头跳过）；设了就批量 |
| `INPUT_IMAGES` | _(unset)_ | 参考图路径，逗号分隔（编辑/多参考）；不设=纯文生图 |
| `OUTPUT_DIR` | `../FLUX2/results/prompt` | 写 `result/` + `prompt/`（编辑时 label=edit） |
| `NUM_INFERENCE_STEPS` | klein=4 / dev=50 | klein 双蒸馏 4 步；dev 50（28 折中） |
| `GUIDANCE_SCALE` | klein=1.0 / dev=4.0 | klein 引导已 baked（忽略）；dev 4.0 |
| `HEIGHT` / `WIDTH` | `1024` / `1024` | 出图尺寸（须是 16 的倍数，VAE 下采样 16×） |
| `MAX_SEQUENCE_LENGTH` | `512` | LLM 文本截断长度 |
| `NUM_IMAGES_PER_PROMPT` | `1` | 每条提示词出几张 |
| `SEED` | `231` | 逐条用 `SEED+i`，可复现 |
| `DTYPE` | `bf16` | `bf16`(推荐) \| `fp16` \| `fp32` |
| `OFFLOAD` | `model` | `model`(按需上GPU) \| `sequential`(最省最慢) \| `none`(整管线上 GPU，最快) |
| `QUANT` | _(空)_ | `4bit` \| `8bit`（dev 在消费级 GPU；需 bitsandbytes。或直接用预量化仓 `diffusers/FLUX.2-dev-bnb-4bit`） |

### Outputs
- **02 inference**: `OUTPUT_DIR/result/<idx>_<slug>[_vN].png`（生成的图）+ `OUTPUT_DIR/prompt/<idx>_<slug>[_vN].txt`（用到的提示词）。

## 可能遇到的问题

公司代理做 HTTPS 中间人解密，下面列常见报错与修法（命令在服务器上、conda 环境已激活时执行）。

**1. `hf download` 报 `SSLCertVerificationError` / `CAS service error : ReqwestMiddleware`**
代理根 CA 不在系统证书包，或 HF 的 Xet/Rust 通道不认代理。`_env.sh` 已设 `HF_HUB_DISABLE_XET=1`；仍报则建 CA 包：
```bash
bash flux2/setup_ca_bundle.sh     # 抓代理证书链 -> ~/.ca-bundle.crt，并自检
HF_TOKEN=<token> MODELS_TO_DOWNLOAD="klein" MODEL_DIR=../../model/FLUX2 bash flux2/01_download_models.sh
```
- 自检 `[OK]` → 直接重跑 `01`。
- 仍报 SSL（CDN 端点用不同 MITM 证书）→ `01` 自动回退到禁用 SSL 校验的下载器（`_hf_download.py`）；或直接 `HF_DISABLE_SSL=1 bash flux2/01_download_models.sh`。

**2. 下载报 `repository not found` / 401 / GATED**
FLUX.2-dev 与 FLUX.2-klein-9B 都是 gated 仓，HF 对无 token 请求返回 "not found" 实为 401。`01` 对无 token 会直接报错指路。修法：1) 在两个 HF 页面接受许可；2) 建 read token；3) `HF_TOKEN=<token> MODELS_TO_DOWNLOAD="klein" MODEL_DIR=../../model/FLUX2 bash flux2/01_download_models.sh`。

**3. 推理报 `ImportError: cannot import name 'Flux2Pipeline'` / `Flux2KleinPipeline`**
`diffusers` 太旧（<0.39，Flux2 管线未合入）。在专用 env 升级：
```bash
conda activate flux2
pip install -U "diffusers==0.39.0" transformers accelerate
# 或重跑 setup: INSTALL_DEPS=1 bash flux2/00_setup_env.sh
```

**4. 推理 OOM（显存不足）**
- 默认已开 `OFFLOAD=model`。仍不够：`OFFLOAD=sequential`（最省，最慢）。
- dev 在消费级 GPU：`QUANT=4bit`（需 `pip install bitsandbytes`），或更省事把 `MODEL_PATH` 指向预量化仓 `diffusers/FLUX.2-dev-bnb-4bit`（无需 QUANT，权重本就是 4-bit）。
- 降尺寸（须 16 的倍数，如 `512x512`）、dev 降到 `NUM_INFERENCE_STEPS=28`。
- 仍紧张：`export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
- 想最快且显存够：`OFFLOAD=none`（klein 在 ≥40GB 卡、dev 在 H200/B200）。

**5. dev 报加载 Mistral3 文本编码器相关错误 / transformers 版本太旧**
dev 用 `Mistral3ForConditionalGeneration`、klein 用 `Qwen3ForCausalLM`，需较新的 `transformers`（≥4.50，建议最新）。升级：`pip install -U transformers`。

**6. `QUANT=4bit` 报 `No module named 'bitsandbytes'`**
4-bit 在线量化需 bitsandbytes。`pip install bitsandbytes`；或改用预量化仓（`MODEL_PATH=.../FLUX.2-dev-bnb-4bit` 且不设 `QUANT`）。

**7. 出图全黑 / 噪点 / 比例错乱**
多为尺寸非 16 的倍数或 fp16 数值溢出。FLUX.2 VAE 下采样 16×（且 latent 被 2×2 patch），故 H/W 须是 16 的倍数——默认 1024 满足；自定义确保 `%16==0`。fp16 溢出改回 `DTYPE=bf16`。

> 通用：`proxy.env`（代理凭证）在仓内 gitignored，`~/.ca-bundle.crt` 在家目录，都不入库；切勿把凭证写进脚本。

## 目录布局
```
<code-dir>/
├── media_code/                  # 本仓
│   ├── proxy.env                # 代理 + 覆盖项, gitignored
│   └── flux2/                   # 编排脚本(本目录)
└── ../model/FLUX2/              # 权重(在 <code-dir> 上一级, 各算法共享)
    ├── FLUX.2-klein-9B/        # HF 快照(model_index.json + transformer/ + vae/ + text_encoder/ + ...)
    └── FLUX.2-dev/             # 同上(text_encoder/ = Mistral3)
```
默认：权重 `../../model/FLUX2`、输出 `../FLUX2/results/`（相对本目录）；用 `MODEL_DIR` / `OUTPUT_DIR` 覆盖。专用 conda env `flux2`（`00` 克隆自 `doll`，升级 diffusers 到 0.39）；FLUX.2 的 diffusers 版本要求 `>=0.39`。

## Notes
- 模型权重遵循其自身许可：FLUX.2-dev 与 FLUX.2-klein-9B 均为 **FLUX Non-Commercial License**（非商用 + gated，需接受许可）；klein-4B 才是 Apache-2.0（本目录未含）。本目录只编排，不复制任何官方代码。用 dev/klein-9B 生成的图受其许可约束。
- `.gitattributes`（仓根）强制 LF，Windows 推上去的脚本在 Ubuntu 上也能干净运行。
- `proxy.env`（代理凭证 / env 覆盖）gitignored，绝不入库。别把凭证写进脚本。
- 公司 TLS 拦截代理下：pip 用 `--trusted-host`；`hf`/`git` 用 CA 包（`_env.sh` 优先 `~/.ca-bundle.crt`，由 `setup_ca_bundle.sh` 构建）。
