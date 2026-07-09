# Qwen3-VL runner

在 Ubuntu + NVIDIA 服务器上跑 [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) 的**图生文（image-to-text）批量推理**——给一文件夹图片，VLM 逐图「看图说话 / 视觉问答」，输出每图一段文本。本目录只含编排脚本——权重从 HuggingFace 下载，模型经 `transformers` + `qwen-vl-utils` 加载，**无需 clone 官方代码仓**（Qwen3-VL 的 HF 快照本身就是完整的）。

## 常用命令

> 假设已进入容器并 `conda activate doll`（或专用 env `qwen3vl`）；`GPU=0` 按需换卡；路径取各脚本默认值（可改）。首次跑前先做下方「首次准备」。

```bash
# ── 一键流程（首次：INSTALL_DEPS=1 装依赖 + 下权重 + 推理） ──
# 把样例图片放到 ../Qwen3-VL/examples/images/（或用 IMAGE_DIR 指你自己的夹）
INSTALL_DEPS=1 bash qwen3vl/run_all.sh

# ── 推理(02) ──
# 1) 图生文：逐图详细描述（默认 PROMPT 即「详细描述这张图…」）
GPU=0 IMAGE_DIR=/data/images bash qwen3vl/02_run_inference.sh
# 2) 视觉问答(VQA)：换一个问题
GPU=0 IMAGE_DIR=/data/images PROMPT="What objects are in this image? Answer briefly." bash qwen3vl/02_run_inference.sh
# 3) 每图问不同的问题：TXT_DIR 与 IMAGE_DIR 同构、每图一个 .txt（同名）
GPU=0 IMAGE_DIR=/data/images TXT_DIR=/data/questions bash qwen3vl/02_run_inference.sh
# 4) 换大模型(MoE 30B，跨 2 卡 sharding)：
GPU=0,1 HF_REPO_ID=Qwen/Qwen3-VL-30B-A3B-Instruct IMAGE_DIR=/data/images bash qwen3vl/02_run_inference.sh
#    （先下载：HF_REPO_ID=Qwen/Qwen3-VL-30B-A3B-Instruct bash qwen3vl/01_download_models.sh）
# 5) 思考模型(thinking 变体，输出含思考块；STRIP_THINKING=1 只留最终答案)：
GPU=0 HF_REPO_ID=Qwen/Qwen3-VL-7B-Thinking IMAGE_DIR=/data/images THINKING=1 STRIP_THINKING=1 MAX_NEW_TOKENS=2048 bash qwen3vl/02_run_inference.sh
# 6) 显存不够：4-bit 量化(需先 pip install bitsandbytes) / 纯贪婪解码省 token
GPU=0 IMAGE_DIR=/data/images LOAD_IN_4BIT=1 DO_SAMPLE=false MAX_NEW_TOKENS=256 bash qwen3vl/02_run_inference.sh
```

- 结果：`../Qwen3-VL/results/<输入夹名>/result/<同名相对路径>.txt`（模型生成的文本）+ `.../prompt/<同名>.txt`（每图用到的提问）。
- `GPU=N` 钉单卡；`GPU=0,1` 让 `device_map="auto"` 把大模型切到多卡。不设 `GPU` = 用全部可见卡。
- 默认模型 `Qwen/Qwen3-VL-7B-Instruct`（公开仓，无需 token）；换模型只改 `HF_REPO_ID`（下载 01 与推理 02 都要传同一个）。

## 首次准备
```bash
cd <your-code-dir>            # e.g. /data_3d/<uid>/code
git -c http.sslVerify=false clone https://github.com/CrescentVelvet/media_code.git
cd media_code && cp proxy.env.example proxy.env   # 填 http_proxy / https_proxy（公司代理用；自用网络可跳过）
conda activate doll                                  # 或专用 env：conda create -n qwen3vl python=3.10 -y && conda activate qwen3vl
INSTALL_DEPS=1 bash qwen3vl/00_setup_env.sh         # 装 transformers(>=4.55) + qwen-vl-utils + accelerate
bash qwen3vl/01_download_models.sh                  # 下 Qwen/Qwen3-VL-7B-Instruct -> ../../model/Qwen3-VL/
```
> Qwen3-VL 的 `transformers` 版本要求较新（Qwen3-VL 支持只在很新的版本里合入）。`doll` env 里若 `transformers` 偏旧，`00_setup_env.sh` 会 `pip install -U transformers`；若怕影响其他算法，用专用 env（`CONDA_ENV=qwen3vl`）。

---

以下为详细参考（流程原理 / 各脚本参数 / 排错 / 目录布局）。

## Inference (02 — 更多用法)
`02_run_inference.sh` 调 `run_inference.py`：**模型+processor 只加载一次**，递归遍历 `IMAGE_DIR` 全部图片，逐图构建 chat message → `apply_chat_template` → `process_vision_info` 取图 → `model.generate` → 砍掉 prompt token 只解码新生成部分 → 存 `result/<rel>.txt`，并打印每图耗时与汇总（avg/min/max/total）。常见覆盖示例：
```bash
# 默认 PROMPT 改成一句话描述（短 caption，省 token）：
GPU=0 IMAGE_DIR=/data/images PROMPT="Write a short one-sentence caption for this image." MAX_NEW_TOKENS=64 bash qwen3vl/02_run_inference.sh
# 输出到别处 / 只用单卡 fp16：
GPU=0 IMAGE_DIR=/data/images OUTPUT_DIR=/data/out DTYPE=fp16 bash qwen3vl/02_run_inference.sh
# 贪婪解码（确定性，temperature/top_p/top_k 失效）：
GPU=0 IMAGE_DIR=/data/images DO_SAMPLE=false bash qwen3vl/02_run_inference.sh
# flash-attention（更快更省显存，需 pip install flash-attn --no-build-isolation）：
GPU=0 IMAGE_DIR=/data/images ATTN_IMPL=flash_attention_2 bash qwen3vl/02_run_inference.sh
```

## Pipeline（推理流程详解）
一张图片从输入到输出文本，经以下几步（对应 `run_inference.py`）：

1. **加载模型一次** — `AutoModelForImageTextToText.from_pretrained(MODEL_PATH, torch_dtype=bf16, device_map="auto")` + `AutoProcessor.from_pretrained(...)`。用 `Auto*` 类自动识别 Qwen3-VL 的模型类（不硬编码类名，7B/30B-A3B/instruct/thinking 都通用）。`device_map="auto"` 按 `CUDA_VISIBLE_DEVICES`（即 `GPU=`）把模型切到可见卡；`accelerate` 自动层间搬运。
2. **构建 chat message** — 每图一个 `{"role":"user","content":[{"type":"image","image":<path>},{"type":"text","text":<PROMPT>}]}`。`PROMPT` 取全局 `PROMPT`，或传 `TXT_DIR` 按同名 `.txt` 逐图读不同问题（VQA）。
3. **套 chat 模板** — `processor.apply_chat_template(messages, add_generation_prompt=True)` 拼成模型输入串（含 `<|im_start|>user …<|im_end|><|im_start|>assistant`）。思考模型额外传 `enable_thinking=True`（旧版 transformers 不支持该参数会自动回退）。
4. **取图 + 跑 processor** — `qwen_vl_utils.process_vision_info(messages)` 把消息里的 `{"type":"image"}` 解析成 PIL 图像张量；再 `processor(text=[...], images=..., return_tensors="pt")` 把文本+图像 token 化、对齐成模型输入字典。`run_inference.py` 兼容新旧两版 `process_vision_info` 签名（新版多返回 `video_kwargs`，图像推理用不到但会透传）。
5. **生成** — `model.generate(**inputs, max_new_tokens=..., do_sample=..., temperature=..., top_p=..., top_k=..., repetition_penalty=...)`。`DO_SAMPLE=false` 走贪婪（确定性）；默认 `true` 采样（Qwen 官方推荐 temp=0.7/top_p=0.8/top_k=20/rep=1.05）。
6. **砍 prompt、解码** — `generated_ids[:, inputs.input_ids.shape[1]:]` 只取**新生成**的 token（processor 已把图像占位符展开到 input_ids，故切片对齐），`processor.batch_decode(..., skip_special_tokens=True)` 得到文本。`STRIP_THINKING=1` 时正则去掉 `iland…tuttoc` 思考块只留最终答案。写入 `result/<rel>.txt`。

> 总结一句：**图片+提问 → chat 模板 → process_vision_info 取图 → processor token 化 → generate → 砍 prompt 解码 → 存 txt**。模型只加载一次，循环逐图出文。

## Config (env vars, all optional)
| var | default | note |
|---|---|---|
| `CONDA_ENV` | `doll` | conda env（专用 `qwen3vl` 推荐，若怕 transformers 升级影响其他算法） |
| `GPU` | _(unset)_ | 物理卡号，如 `GPU=0`；`GPU=0,1` 让 device_map 切多卡；不设=全部可见卡 |
| `MODEL_DIR` | `../../model/Qwen3-VL` | 权重根目录 |
| `HF_REPO_ID` | `Qwen/Qwen3-VL-7B-Instruct` | HF 仓库 ID（下权重 01 与推理 02 传同一个） |
| `MODEL_PATH` | `$MODEL_DIR/<repo_basename>` | 本地快照目录（自动由 HF_REPO_ID 推导） |
| `INSTALL_DEPS` | `0` (run_all: `1`) | `1` = 装 transformers/qwen-vl-utils/accelerate |
| `HF_HUB_DISABLE_XET` | `1` | 关 HF Xet/Rust 通道（代理不友好） |
| `HF_DISABLE_SSL` | `0` | `1` = 关 SSL 校验下权重（代理 MITM 证书不通时） |
| `HF_TOKEN` | _(unset)_ | Qwen3-VL 公开仓无需；仅当你换 gated 仓时才要 |
| `INCLUDE_PATTERNS` | _(空=全下)_ | 01 下权重时的 glob 过滤，如 `*.json,*.txt,*.safetensors` |

### Inference (02)
| var | default | note |
|---|---|---|
| `IMAGE_DIR` | `../Qwen3-VL/examples/images` | 输入图像夹（递归遍历） |
| `TXT_DIR` | _(unset)_ | 每图提问夹（与 IMAGE_DIR 同构、同名 .txt）；不设=全部用全局 PROMPT |
| `OUTPUT_DIR` | `../Qwen3-VL/results/<image_dir_name>` | 写 `result/` + `prompt/` |
| `PROMPT` | `Describe this image in detail…` | 全局提问（TXT_DIR 未设时用） |
| `MAX_NEW_TOKENS` | `512` | 思考模型建议 ≥2048 |
| `DO_SAMPLE` | `true` | `false` = 贪婪（确定性，temperature 等失效） |
| `TEMPERATURE` / `TOP_P` / `TOP_K` | `0.7` / `0.8` / `20` | 采样参数（Qwen 官方推荐值） |
| `REPETITION_PENALTY` | `1.05` | |
| `SEED` | `231` | |
| `DTYPE` | `bf16` | `bf16` \| `fp16` \| `fp32` |
| `DEVICE_MAP` | `auto` | `auto`(按可见卡切) \| `cpu` \| 自定义 device map |
| `ATTN_IMPL` | _(空=配置默认)_ | `sdpa` \| `flash_attention_2` \| `eager` |
| `LOAD_IN_4BIT` / `LOAD_IN_8BIT` | `0` / `0` | `1` = bitsandbytes 量化（需 `pip install bitsandbytes`） |
| `THINKING` | `0` | `1` = 思考模型（传 enable_thinking=True） |
| `STRIP_THINKING` | `0` | `1` = 去掉 `iland…tuttoc` 块只留最终答案 |

## Outputs
- **02 inference**: `OUTPUT_DIR/result/<rel>.txt`（模型生成的文本）+ `OUTPUT_DIR/prompt/<rel>.txt`（每图用到的提问）。

## 目录布局
```
<code-dir>/
├── media_code/                  # 本仓
│   ├── proxy.env                # 代理 + 覆盖项, gitignored
│   └── qwen3vl/                 # 编排脚本(本目录)
└── ../../model/Qwen3-VL/        # 权重(在 <code-dir> 上一级, 各算法共享)
    └── Qwen3-VL-7B-Instruct/    # HF 快照(config + *.safetensors + processor 资产)
```
默认：权重 `../../model/Qwen3-VL`、输出 `../Qwen3-VL/results/`（相对本目录）；用 `MODEL_DIR` / `OUTPUT_DIR` 覆盖。复用现有 conda env（默认 `doll`）；Qwen3-VL 的 transformers 版本要求较新——若与其他算法冲突，用专用 env（`CONDA_ENV=qwen3vl`）。

## 可能遇到的问题

公司代理做 HTTPS 中间人解密，下面列常见报错与修法（命令在服务器上、conda 环境已激活时执行）。

**1. `hf download` 报 `SSLCertVerificationError` / `CAS service error : ReqwestMiddleware`**
代理根 CA 不在系统证书包，或 HF 的 Xet/Rust 通道不认代理。`_env.sh` 已设 `HF_HUB_DISABLE_XET=1`；仍报则建 CA 包：
```bash
bash qwen3vl/setup_ca_bundle.sh    # 抓代理证书链 -> ~/.ca-bundle.crt，并自检
bash qwen3vl/01_download_models.sh
```
- 自检 `[OK]` → 直接重跑 `01`。
- 自检 `[FAIL]` → 把公司根 CA 追加到 `~/.ca-bundle.crt` 后重跑。
- 仍报 SSL（CDN 端点用不同 MITM 证书）→ `01` 自动回退到禁用 SSL 校验的下载器（`_hf_download.py`）；或直接 `HF_DISABLE_SSL=1 bash qwen3vl/01_download_models.sh`。

**2. `hf download` 报 `repository not found` / 401**
公开仓 `Qwen/Qwen3-VL-7B-Instruct` 无需 token。若你换成了 gated 仓（HF 对未认证账号返回 "not found" 实为 401）：1) 在该仓库页面接受许可证；2) 建 read token；3) `HF_TOKEN=<token> bash qwen3vl/01_download_models.sh`（脚本自动透传给 `hf download` 和 SSL 兜底下载器）。

**3. 推理报 `KeyError` / 找不到 Qwen3-VL 模型类 / `AutoModelForImageTextToText` 不识别**
`transformers` 太旧，Qwen3-VL 支持未合入。升级：
```bash
pip install -U "transformers>=4.55"
# 仍不行就装最新：pip install -U transformers
```
或在专用 env 跑：`CONDA_ENV=qwen3vl INSTALL_DEPS=1 bash qwen3vl/00_setup_env.sh`。

**4. 推理报 `ImportError: qwen_vl_utils`**
没装 qwen-vl-utils：`pip install qwen-vl-utils`（或 `INSTALL_DEPS=1 bash qwen3vl/00_setup_env.sh`）。

**5. 推理 OOM（显存不足）**
- 量化：`LOAD_IN_4BIT=1 bash qwen3vl/02_run_inference.sh`（先 `pip install bitsandbytes`）。
- 多卡切分：`GPU=0,1 bash qwen3vl/02_run_inference.sh`（device_map=auto 自动层切）。
- 省 token：`MAX_NEW_TOKENS=256`；省显存：`DTYPE=fp16`、`ATTN_IMPL=sdpa`（或装好 flash-attn 后 `ATTN_IMPL=flash_attention_2`）。
- 仍紧张：`export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

**6. 思考模型输出只有思考块、没有最终答案**
`MAX_NEW_TOKENS` 太小，思考还没结束就被截断。调大：`THINKING=1 MAX_NEW_TOKENS=2048 STRIP_THINKING=1 bash qwen3vl/02_run_inference.sh`。

**7. 生成内容重复 / 跑题**
默认采样参数(temp=0.7/rep=1.05)偶有重复；要确定性用 `DO_SAMPLE=false`（贪婪），或调高 `REPETITION_PENALTY`。

> 通用：`proxy.env`（代理凭证）在仓内 gitignored，`~/.ca-bundle.crt` 在家目录，都不入库；切勿把凭证写进脚本。

## Notes
- 模型权重遵循其自身许可（Qwen3-VL 见 [QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)）。本目录只编排，不复制任何官方代码。
- `.gitattributes`（仓根）强制 LF，Windows 推上去的脚本在 Ubuntu 上也能干净运行。
- `proxy.env`（代理凭证 / env 覆盖）gitignored，绝不入库。别把凭证写进脚本。
- 公司 TLS 拦截代理下：pip 用 `--trusted-host`；`hf`/`git` 用 CA 包（`_env.sh` 优先 `~/.ca-bundle.crt`，由 `setup_ca_bundle.sh` 构建）。
