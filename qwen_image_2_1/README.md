# Qwen-Image-2.1 — 统一文生图 + 图像编辑（原生 RGBA 透明）

阿里 Qwen 团队 2026-09-20 开源的统一图像生成/编辑模型：7B Single-Stream DiT（32 层）+ Qwen3-VL 8B 文本编码器 + 64 通道 RGBA VAE，全栈 bf16 ≈ 33 GiB。支持原生透明图生成/编辑、最多 10 张参考图编辑。无官方推理仓（纯 diffusers 生态，`QwenImage21Pipeline` 需 git 版 diffusers）。

> ⚠️ **许可证：Qwen Research License，仅限非商用**（前代 Qwen-Image 是 Apache 2.0，本代收紧）。生成图片的权利归用户。

权重：`D:\wheel\qwen_image21_ms`（WSL: `/mnt/d/wheel/qwen_image21_ms`），ModelScope/HF 同源（`Qwen/Qwen-Image-2.1`）。迅雷批量下载清单见 `download_urls_hf.txt` / `download_urls_modelscope.txt`。

## 常用命令
> 假设 cwd = `media_code/`（WSL: `cd /mnt/c/code/media_code`）。WSL 从零复现见 [README_wsl.md](README_wsl.md)。

```bash
# 一键（校验权重 + t2i 冒烟）
GPU=0 MODEL_PATH=/mnt/d/wheel/qwen_image21_ms \
  RESULTS_DIR=/mnt/d/output/qwen_image_2_1_results \
  bash qwen_image_2_1/run_all.sh

# 分步
# 1) 校验权重（25 个文件字节级比对官方清单）
MODEL_PATH=/mnt/d/wheel/qwen_image21_ms \
  bash qwen_image_2_1/01_check_models.sh

# 2) 文生图
GPU=0 MODE=t2i \
  PROMPT="A neon shop sign that reads \"QWEN IMAGE 2.1\", rainy night" \
  ASPECT=1:1 STEPS=40 SEED=42 \
  MODEL_PATH=/mnt/d/wheel/qwen_image21_ms \
  RESULTS_DIR=/mnt/d/output/qwen_image_2_1_results \
  bash qwen_image_2_1/02_run_inference.sh

# 3) 图像编辑（IMAGES 逗号分隔，多图=多参考图）
GPU=0 MODE=edit \
  IMAGES=/mnt/d/dataset/sample/input.png \
  PROMPT="Change the background to a sunset beach" \
  MODEL_PATH=/mnt/d/wheel/qwen_image21_ms \
  RESULTS_DIR=/mnt/d/output/qwen_image_2_1_results \
  bash qwen_image_2_1/02_run_inference.sh

# 4) 透明图（RGBA）生成 —— prompt 需包含官方约定句式
GPU=0 MODE=rgba \
  PROMPT="This is an RGBA image with transparency. A cute cartoon dragon sticker. The image has alpha channel and the background is transparent." \
  ASPECT=1:1 \
  MODEL_PATH=/mnt/d/wheel/qwen_image21_ms \
  RESULTS_DIR=/mnt/d/output/qwen_image_2_1_results \
  bash qwen_image_2_1/02_run_inference.sh
```

## 首次准备
```bash
# WSL 本机：建 env + 装依赖（torch 2.8 复用 /mnt/d/wheel 本地 wheel）
bash qwen_image_2_1/00a_setup_env.sh
```

权重目录布局（`$MODEL_PATH`，diffusers 结构）：
```
qwen_image21_ms/
├── model_index.json, configuration.json
├── scheduler/scheduler_config.json
├── transformer/    config.json + *.index.json + 2 分片 (9.28+3.97 GiB)   # 7B DiT
├── text_encoder/   config.json + generation_config.json + *.index.json + 4 分片 (~16.2 GiB)  # Qwen3-VL 8B
├── vae/            config.json + diffusion_pytorch_model.safetensors (1.26 GiB)  # 64ch RGBA
└── processor/      tokenizer.json 等 9 个文件
```

---

以下为详细参考。

## Config (env vars)
| var | default | note |
|---|---|---|
| `MODEL_PATH` | `/mnt/d/wheel/qwen_image21_ms` | 模型目录 |
| `MODE` | `t2i` | `t2i` / `edit` / `rgba` |
| `PROMPT` | 官方示例 prompt | rgba 需含官方约定句式（见上方命令） |
| `IMAGES` | 空 | edit 必填，逗号分隔多图（最多 10 张） |
| `ASPECT` | `1:1` | `1:1`/`4:3`/`3:4`/`3:2`/`2:3`/`16:9`/`9:16`，覆盖 WIDTH/HEIGHT |
| `WIDTH`/`HEIGHT` | 2048/2048 | edit 模式忽略（跟随输入图） |
| `STEPS` | 40 | 官方示例值 |
| `SEED` | 42 | |
| `GUIDANCE_SCALE` | 空（pipeline 默认） | 可选覆盖 |
| `CPU_OFFLOAD` | 1 | 24G 显存必须开（全栈 bf16 ~33G）；0=整模上卡（需 ≥40G） |
| `RESULTS_DIR` | `~/output/qwen_image_2_1_results` | |
| `OUTPUT` | `$RESULTS_DIR/<mode>_<时间戳>.png` | |

## 目录布局
```
media_code/qwen_image_2_1/     # 本目录（编排脚本，无官方代码可 clone——纯 diffusers）
/mnt/d/wheel/qwen_image21_ms/  # 权重（只读）
~/output/qwen_image_2_1_results/  # 默认输出（建议命令里显式指到 /mnt/d/output/）
```

## Notes
- RTX 3090 24GB：必须 `CPU_OFFLOAD=1`（默认已开）；第三方实测 RTX 4090 编辑 1MP 约 5s、2K 生成约 1min 量级【官方未给硬件基准，仅供参考】。
- `transformers>=5.17` 是硬要求；`diffusers` 必须 git 版（`QwenImage21Pipeline` 尚未进 release）。
- edit 模式多参考图传 list 的接口形态按官方宣传整理，实测以 diffusers git 版为准【待验证】。
