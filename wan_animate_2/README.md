# Wan-Animate-2 — 端到端角色动画（参考图 + 驱动视频 -> 动画视频）

在 Ubuntu + NVIDIA 服务器上跑 [Wan-Animate-2](https://github.com/Wan-Video/Wan-Animate-2)（阿里 Wan 团队，2026.08）的**角色动画推理**：输入一张参考图 + 一段驱动视频 + 文本描述，端到端生成保持身份、复刻动作的动画视频。本目录只含编排脚本——官方代码自动 `git clone --recursive` 到 `../Wan-Animate-2`，权重从本地 `$MODEL_DIR` 加载（`01` 下载并符号链接到 `../Wan-Animate-2/ckpts`，YAML 的 `../ckpts/...` 相对路径原样生效）。

Wan-Animate-2 直接把驱动视频喂进 Diffusion Transformer，**去掉了中间动作提取器**（无需 SMPL / 运动点），高保真动作 + 强身份保持；外加文本驱动视角控制。本仓支持两个变体：**Base**（40 步）与 **Distillation/Lite**（10 步实时版）。

## 常用命令

> 假设已进入容器并 `conda activate wan_animate_2`；`GPU=0,1` 按需换卡。首次跑前先做下方「首次准备」。

```bash
# ── 一键(00+01+02) ──
GPU=0,1 INSTALL_DEPS=1 bash wan_animate_2/run_all.sh

# ── 推理(02) ──
# 1) Base 变体 —— 官方 demo1（reference.png + template.mp4 + 默认猫 prompt）
GPU=0,1 bash wan_animate_2/02_run_inference.sh
# 2) Distillation 变体 —— 10 步快速版
GPU=0,1 MODEL_VARIANT=distillation bash wan_animate_2/02_run_inference.sh
# 3) 自定义参考图 / 驱动视频 / prompt
GPU=0,1 REFER_IMAGE=/path/to/char.png REFER_VIDEO=/path/to/driver.mp4 \
  PROMPT="人物外观描述：... 背景描述：..." OUTPUT_NAME=my_anim \
  bash wan_animate_2/02_run_inference.sh
# 4) 单卡(低分辨率，sp_size=1) —— 注意 Base 默认较重，建议 Distillation + 小分辨率
GPU=0 MODEL_VARIANT=distillation WIDTH=480 HEIGHT=640 CLIP_LEN=49 \
  bash wan_animate_2/02_run_inference.sh
# 5) 720P（需 8 卡，官方默认配置）
GPU=0,1,2,3,4,5,6,7 WIDTH=720 HEIGHT=1280 CLIP_LEN=81 \
  bash wan_animate_2/02_run_inference.sh
```

- 结果：`../wan_animate_2_results/<name>.mp4`。
- prompt 建议**先用 LLM（如 Qwen3.7-Plus）按官方模板给参考图生成中文描述**（只描述人物外观 + 背景，不描述动作），见下方「Prompt」。
- 默认分辨率 `640×800 / clip_len=81`；官方 720P（`720×1280`）需 8 卡，480P 需 2 卡。

## 首次准备
```bash
cd <your-code-dir>            # e.g. /data_3d/<uid>/code
git clone <本仓> media_code && cd media_code && cp proxy.env.example proxy.env   # 填 http_proxy / https_proxy
# conda env + torch（cu126）
conda create -n wan_animate_2 python=3.11 -y && conda activate wan_animate_2
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
# clone 官方仓 + 装依赖（requirements.txt 去掉 torch 行 + flash-attn + pip install -e .）
INSTALL_DEPS=1 bash wan_animate_2/00_setup_env.sh
# 下权重（默认 modelscope；可 DOWNLOAD_SOURCE=HuggingFace）
bash wan_animate_2/01_download_models.sh
```
权重会下到 `$MODEL_DIR/Wan-AI/Wan2.2-Animate-2-14B/`（即官方 `ckpts/` 目录树），再符号链接到 `../Wan-Animate-2/ckpts`，YAML 的 `../ckpts/...` 相对路径无需改动：
```
$MODEL_DIR/Wan-AI/Wan2.2-Animate-2-14B/
  wan_animate_2/
    wan_animate_2_bf16.safetensors               # Base 主模型 (DiT)
    wan_animate_2_bf16_distillation.safetensors  # Distillation 主模型
  videomodel/Wan-AI/
    models_t5_umt5-xxl-enc-bf16.pth              # UMT5-XXL 文本编码器
    umt5-xxl/                                     # T5 tokenizer
    vae.pth                                       # VAE
    models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth  # CLIP
    xlm-roberta-large/                           # CLIP tokenizer
```
下载命令（如需补下）：
```bash
modelscope download --model Wan-AI/Wan2.2-Animate-2-14B --local_dir $MODEL_DIR/Wan-AI/Wan2.2-Animate-2-14B
# 或
huggingface-cli download Wan-AI/Wan2.2-Animate-2-14B --local-dir $MODEL_DIR/Wan-AI/Wan2.2-Animate-2-14B
```

---

以下为详细参考（流程原理 / 各脚本参数 / 排错 / 目录布局）。

## Pipeline（流程详解）
```
参考图(reference.png) ──┐
驱动视频(template.mp4) ─┼─► [YAML] WanAnimate2MPIPipeline ─► 动画视频(<name>.mp4)
中文 prompt ────────────┘        (DiT 14B + UMT5 + VAE + CLIP)
```
1. `02_run_inference.sh` 选 YAML（base/distillation），设默认 step/guide_scale。
2. `run_inference.py` 把 YAML 里的 `sp_size`/`sharding_size` patch 成可见 GPU 数（官方默认 8/8），生成 `_patched_*.yaml`（与原 YAML 同目录，保证 `../ckpts/...` 解析不变）。
3. `from core import build_object_from_config_file` 加载 pipeline，调 `pipeline(refer_img_path=, tpl_video_path=, output_path=, width, height, fps, seed, clip_len, sample_guide_scale, step, prompt, prompt_ref=)`。
4. 把 pipeline 写出的 mp4 移到 `$OUTPUT_DIR/$OUTPUT_NAME.mp4`，清理临时 session 目录。

## Prompt（文本描述生成）
推理前**必须**用 LLM 给参考图生成中文描述，作为 `PROMPT`。官方模板：
```
用中文客观描述图片中的内容，包括以下要点：人物外观描述，不描述动作行为。 背景描述，忽略主观评价和情绪推测。 下面给出描述范例，必须遵循这个范式，不要输出额外的符号： 人物外观描述：穿着一件浅蓝色的校服衬衫，领口和袖口有白色边饰。胸前有一个圆形徽章。 背景描述：背景为明亮、整洁的教室或办公室，氛围安静有序。
```
`PROMPT_REF`（默认「人物动作的参考视频」）是视角控制参考文本。

## Config (env vars)
### 通用
| var | default | note |
|---|---|---|
| `CONDA_ENV` | 当前 env | 建议 `wan_animate_2`（python 3.11 + torch 2.7 cu126） |
| `GPU` | _(unset=全部可见卡)_ | 物理卡号，逗号分隔；`GPU=0,1` → sp_size=2 |
| `OFFICIAL_DIR` | `../Wan-Animate-2` | 官方代码路径（sibling of media_code） |
| `MODEL_DIR` | `../../model` | 权重根（各算法共享） |
| `CKPTS_DIR` | `$MODEL_DIR/Wan-AI/Wan2.2-Animate-2-14B` | HF/ModelScope 下载落地处 |
| `OFFICIAL_REPO` | 官方 GitHub URL | clone 源（带 `--recursive`） |
| `MODEL_REPO` | `Wan-AI/Wan2.2-Animate-2-14B` | HF/ModelScope 模型 repo |
| `DOWNLOAD_SOURCE` | `modelscope` | `modelscope` / `HuggingFace` |
| `INSTALL_DEPS` | `0` (run_all: `1`) | `1` → 装 requirements + flash-attn + `pip install -e .` |

### 推理 (02)
| var | default | note |
|---|---|---|
| `MODEL_VARIANT` | `base` | `base`(40步) / `distillation`(10步实时版) |
| `PROMPT` | 官方猫 prompt | 参考图的中文外观+背景描述（先用 LLM 生成） |
| `PROMPT_REF` | `人物动作的参考视频` | 视角控制参考文本 |
| `REFER_IMAGE` | 官方 `examples/demo1/reference.png` | 参考图 |
| `REFER_VIDEO` | 官方 `examples/demo1/template.mp4` | 驱动视频（缺失时自动 glob `*.mp4/*.mov`） |
| `OUTPUT_DIR` | `../wan_animate_2_results` | 输出目录 |
| `OUTPUT_NAME` | `animate` | 输出文件名（不含扩展名） |
| `WIDTH` / `HEIGHT` | `640` / `800` | 分辨率；720P=`720×1280`(需8卡) |
| `CLIP_LEN` | `81` | 生成帧数 |
| `FPS` | `24` | 输出帧率 |
| `STEP` | base=`40` / distill=`10` | 采样步数 |
| `SAMPLE_GUIDE_SCALE` | base=`3.0` / distill=`1.0` | CFG 引导强度（distill=1.0 即无 CFG） |
| `SEED` | `-1` | 随机种子（-1 = 随机） |

## 可能遇到的问题

**1. `import core` 报 ModuleNotFoundError**
官方包没装。跑 `INSTALL_DEPS=1 bash wan_animate_2/00_setup_env.sh`（会 `pip install -e ../Wan-Animate-2`）。

**2. 模型文件找不到 / `../ckpts/...` 路径报错**
`01_download_models.sh` 负责下载 + 符号链接。确认 `../Wan-Animate-2/ckpts` 是指向 `$CKPTS_DIR` 的软链，且 `$CKPTS_DIR` 下有 `wan_animate_2/` 与 `videomodel/Wan-AI/`。重跑 `bash wan_animate_2/01_download_models.sh`。

**3. 推理 OOM（显存不足）**
- 降分辨率/帧数：`WIDTH=480 HEIGHT=640 CLIP_LEN=49`。
- 换 Distillation 变体（10 步、更快、显存更友好）：`MODEL_VARIANT=distillation`。
- 8 卡 720P 是官方默认；卡少就用 480P 或更小。本仓默认 `sp_size=sharding_size=可见卡数`，会自动适配。

**4. 多卡 / `sp_size` 报错**
本仓 `run_inference.py` 已把 YAML 的 `sp_size`/`sharding_size` patch 成可见 GPU 数。如果 `GPU=0,1`（2 卡）则 =2。若 pipeline 要求 sp_size 整除某些维度，调小 `WIDTH/HEIGHT/CLIP_LEN`。单卡（`GPU=0`）= sp_size=1，注意 Base 默认 640×800×81 可能仍重，建议 Distillation + 小分辨率。

**5. `flash-attn` 装不上**
必须 `--no-build-isolation`（否则找不到已装 torch）：`pip install flash-attn --no-build-isolation`。公司代理封 `download.pytorch.org` 时，torch 已按「首次准备」从 pytorch index 装好；flash-attn 走 PyPI 镜像。

**6. modelscope / pip / git 下载报 SSL 证书错误**
公司代理 HTTPS 中间人解密。先建 CA 包（`bash hypir/setup_ca_bundle.sh`），`_env.sh` 会自动用 `~/.ca-bundle.crt`。

**7. 跑 `.sh` 报 `syntax error near unexpected token '('`（CRLF 行尾）**
`.gitattributes` 强制 LF，git 同步的文件没问题；手动 scp 的需：`find wan_animate_2 -name '*.sh' -exec sed -i 's/\r$//' {} +`。

## 目录布局
```
<code-dir>/
├── media_code/                    # 本仓
│   ├── proxy.env                  # 代理 + 覆盖项, gitignored
│   └── wan_animate_2/             # 编排脚本(本目录)
├── Wan-Animate-2/                 # 官方代码(自动 git clone --recursive)
│   ├── infer/                     # YAML + demo 脚本
│   ├── examples/demo1/            # reference.png / template.mp4
│   └── ckpts -> $MODEL_DIR/.../Wan2.2-Animate-2-14B   # 01 建的符号链接
└── ../../model/                   # 权重(在 <code-dir> 上一级, 各算法共享)
    └── Wan-AI/Wan2.2-Animate-2-14B/
        ├── wan_animate_2/         # DiT 主模型(base + distillation)
        └── videomodel/Wan-AI/      # T5 + VAE + CLIP
```
默认：官方代码 `../Wan-Animate-2`、权重 `../../model`（相对本目录）；用 `OFFICIAL_DIR` / `MODEL_DIR` 覆盖。

## Notes
- Official code & weights follow their own license (Wan-Animate-2 = Apache 2.0). This folder only orchestrates; no official code is copied.
- `.gitattributes` (repo root) forces LF so Windows-pushed scripts run cleanly on Ubuntu.
- `proxy.env` (proxy creds / env overrides) is gitignored — never committed.
