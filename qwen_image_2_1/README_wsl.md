# Qwen-Image-2.1 (WSL Ubuntu 24.04) — 本地复现指南

本文件是 [`README.md`](README.md) 的 WSL 本地复现版。照着做即可从零跑完全流程。

## 与服务器版的核心差异
| | 服务器 | WSL |
|---|---|---|
| conda env | clone doll | `00a` 从零建 `qwen_image_2_1`（python 3.11） |
| pip 源 | 默认 PyPI | 阿里镜像 + `--find-links /mnt/d/wheel`（本地 wheel 优先） |
| 权重 | `$REPO_DIR/../../model/` | `D:\wheel\qwen_image21_ms`（迅雷/ModelScope 下载，只读） |
| diffusers | pip release | git 源码 clone 到 `~/repos/diffusers` + `pip install -e` |

## Windows 路径 → WSL 路径
| Windows | WSL |
|---|---|
| D:\wheel\qwen_image21_ms | /mnt/d/wheel/qwen_image21_ms |
| D:\output\qwen_image_2_1_results | /mnt/d/output/qwen_image_2_1_results |

## 前提条件
- WSL2 Ubuntu 24.04 + NVIDIA Windows 驱动（CUDA 透传）+ `~/miniconda3`
- GPU：RTX 3090 24GB（必须开 `CPU_OFFLOAD=1`，默认已开）
- 权重已就位（见下）

## 首次准备
```bash
cd /mnt/c/code/media_code

# ── 0) 建 env + 装依赖（约 5-10 分钟，torch 复用 /mnt/d/wheel 本地 wheel）──
bash qwen_image_2_1/00a_setup_env.sh

# ── 0b) 权重：已在 D:\wheel\qwen_image21_ms（2026-10-23 落盘并校验）──
#   如需重下：本目录 download_urls_hf.txt / download_urls_modelscope.txt 迅雷批量导入。
#   注意三个 config.json 重名（transformer/text_encoder/vae），下完按目录归位。
MODEL_PATH=/mnt/d/wheel/qwen_image21_ms \
  bash qwen_image_2_1/01_check_models.sh
```

## 全流程命令
```bash
cd /mnt/c/code/media_code

# ── 1) 文生图冒烟（2048x2048，40 步，固定种子）──
#    输入：无
#    输出：/mnt/d/output/qwen_image_2_1_results/t2i_*.png
GPU=0 MODE=t2i \
  PROMPT="A neon shop sign that reads \"QWEN IMAGE 2.1\", rainy night, reflections on wet pavement" \
  ASPECT=1:1 STEPS=40 SEED=42 \
  MODEL_PATH=/mnt/d/wheel/qwen_image21_ms \
  RESULTS_DIR=/mnt/d/output/qwen_image_2_1_results \
  bash qwen_image_2_1/02_run_inference.sh

# ── 2) 图像编辑（可选；IMAGES 逗号分隔，最多 10 张参考图）──
# GPU=0 MODE=edit \
#   IMAGES=/mnt/d/dataset/sample/input.png \
#   PROMPT="Change the background to a sunset beach" \
#   MODEL_PATH=/mnt/d/wheel/qwen_image21_ms \
#   RESULTS_DIR=/mnt/d/output/qwen_image_2_1_results \
#   bash qwen_image_2_1/02_run_inference.sh

# ── 3) RGBA 透明图（可选；prompt 必须含官方约定句式）──
# GPU=0 MODE=rgba \
#   PROMPT="This is an RGBA image with transparency. A cute cartoon dragon sticker. The image has alpha channel and the background is transparent." \
#   ASPECT=1:1 \
#   MODEL_PATH=/mnt/d/wheel/qwen_image21_ms \
#   RESULTS_DIR=/mnt/d/output/qwen_image_2_1_results \
#   bash qwen_image_2_1/02_run_inference.sh
```

## 可能遇到的问题（WSL 专属）
- **conda ToS 报错（conda 26.x）**：`00a` 已自动 accept；手动补：
  `conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main`（和 `pkgs/r`）
- **git clone diffusers 挂 libffi**：脚本已用 `LD_LIBRARY_PATH= git clone` 绕开；手动操作同加此前缀
- **OOM / 加载即爆显存**：确认 `CPU_OFFLOAD=1`（默认）；整模上卡需 ≥40G 显存
- **/mnt/d 读权重慢**：9p 文件系统，33G 权重首轮加载约几分钟，属正常；想快可拷到 `~/model/`（Linux fs）
- **vhdx 空间**：pip 缓存在装完已自动 purge；输出图直接写 `/mnt/d/output/` 不占 vhdx
