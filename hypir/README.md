# HYPIR runner

在 Ubuntu + NVIDIA 服务器上跑 [HYPIR](https://github.com/XPixelGroup/HYPIR)（SIGGRAPH 2025 图像复原）的**推理 / 数据集构建 / LoRA 训练**。本目录只含编排脚本——官方代码自动 clone、权重从 HuggingFace 下载。

> 📄 详细文档：原理详解与排错手册见 [NOTES.md](NOTES.md)；A/B/C/D 美颜对比实验、扫参实验与调参速查见 [EXPERIMENTS.md](EXPERIMENTS.md)。

## 常用命令

> 假设已进入容器并 `conda activate hypir`；路径取各脚本默认值（可改）；`GPU=0` 按需换卡。首次跑前先做下方「首次准备」。

```bash
# ── 配对路径(真实 LQ+HQ，03b/04b) ──
# ♻️1) 构建配对数据集(按同名文件配对 HQ/LQ -> parquet)
HQ_DIR=../HYPIR/dataset/ppr10k_faces_20260703/hq LQ_DIR=../HYPIR/dataset/ppr10k_faces_20260703/lq bash hypir/03b_build_paired_dataset.sh
# 🚀2) 开始训练(暖启动, 默认后台, 日志见提示)
GPU=0 BG=0 bash hypir/04b_train_paired.sh
# 🚀3) 继续上次 LoRA 训练(RESUME 指向 checkpoint 目录)
GPU=0 BG=0 RESUME=../HYPIR/experiments/ppr10k_faces_paired/checkpoint-65000 bash hypir/04b_train_paired.sh

# ── 合成退化路径(只输入 HQ, 在线合成 LQ, 03c/04c) ──
# ♻️4) 构建 HQ-only 数据集(LQ 训练时在线合成, 不存盘)
HQ_DIR=../HYPIR/dataset/ppr10k_faces_20260703/hq bash hypir/03c_build_synthetic_dataset.sh
# 🚀5) 开始训练(暖启动 + 在线退化; HQ>512 用 CROP_TYPE=random 在线裁 512 patch)
GPU=0 HQ_DIR=../HYPIR/dataset/ppr10k_faces_20260703/hq CROP_TYPE=random BG=0 BATCH_SIZE=8 HF_HUB_OFFLINE=1 bash hypir/04c_train_synthetic.sh
# 🚀5b) 换别的数据集训(只改 HQ_DIR + OUTPUT_DIR，别和旧实验混；guojia_datas 是 HQ 文件夹，可含子目录)
GPU=0 HQ_DIR=../HYPIR/dataset/guojia_datas_20260708 OUTPUT_DIR=../HYPIR/experiments/guojia_datas CROP_TYPE=random BG=0 BATCH_SIZE=8 HF_HUB_OFFLINE=1 bash hypir/04c_train_synthetic.sh

# ── 美颜退化路径(只输入原始图像, 合成高斯模糊退化 LQ 和美颜增强 HQ, 03d→03e/04b) ──
# 🔍03d) 抽样看效果
GPU=0 SKIP_PARQUET=1 SAVE_COMPARE=1 INPUT_DIR=../HYPIR/input/test_faces_hq OUTPUT_DIR=../../output/hypir_test_results/美颜退化数据预览 bash hypir/03d_build_beauty_dataset.sh
# ♻️03d) 构建全量数据集（多卡）
GPU=0,1,2 NPROC=3 INPUT_DIR=../HYPIR/dataset/guojia_datas_20260708 bash hypir/03d_build_beauty_dataset.sh
# ♻️03d) C 二次美颜数据集(BEAUTY_PASSES=2 一次产出 A/B/C 三套 parquet, A/B 不变) — 实验设计见 EXPERIMENTS.md
GPU=0,1,2 NPROC=3 BEAUTY_PASSES=2 INPUT_DIR=../HYPIR/dataset/guojia_datas_20260708 bash hypir/03d_build_beauty_dataset.sh
# ♻️03e) D 去红润美颜数据集(RetouchFormer + wavelet 融合; DECOLOR_MODE=high_freq_dc 去红, 与 03d 并列不依赖)
GPU=0,1,2 NPROC=3 INPUT_DIR=../HYPIR/dataset/guojia_datas_20260708 bash hypir/03e_decolor_beauty_dataset.sh

# ── A/B/C/D 依次训练(各自 OUTPUT_DIR 分开; 实验设计见 EXPERIMENTS.md)：
# 🚀04b) A 基线(只高斯模糊，预期会长痘变丑)：
GPU=0,1,2 N_TRAIN_GPU=3 BG=0 PARQUET_PATH=../HYPIR/dataset/beauty_guojia_datas_20260708/rest.parquet OUTPUT_DIR=../HYPIR/experiments/rest bash hypir/04b_train_paired.sh
# 🚀04b) B 复原+美颜(LQ 同样模糊、HQ 换美颜版 1pass，预期修掉长痘、又不毁脸)：
GPU=0,1,2 N_TRAIN_GPU=3 BG=0 PARQUET_PATH=../HYPIR/dataset/beauty_guojia_datas_20260708/rest_beauty.parquet OUTPUT_DIR=../HYPIR/experiments/beauty bash hypir/04b_train_paired.sh
# 🚀04b) C 二次美颜(LQ 同样模糊、HQ 换迭代美颜版 N pass，预期美颜最强、但可能过磨失结构)：
GPU=0,1,2 N_TRAIN_GPU=3 BG=0 PARQUET_PATH=../HYPIR/dataset/beauty_guojia_datas_20260708/rest_beauty_strong.parquet OUTPUT_DIR=../HYPIR/experiments/beauty_strong bash hypir/04b_train_paired.sh
# 🚀04b) D 去红润美颜(LQ 同样模糊、HQ 换去红润美颜版 wavelet 融合，预期红润减弱、磨皮保留)：
GPU=0,1,2 N_TRAIN_GPU=3 BG=0 PARQUET_PATH=../HYPIR/dataset/beauty_decolor_guojia_datas_20260708/rest_beauty_decolor.parquet OUTPUT_DIR=../HYPIR/experiments/beauty_decolor bash hypir/04b_train_paired.sh

# ── 推理(02/06) ──
# 💡02) 测试原生(发布)模型 —— 指定输入路径
GPU=0 LQ_DIR=../HYPIR/input/test_faces UPSCALE=4 bash hypir/02_run_inference.sh
# 💡02) 测试自己训的 LoRA —— 指定输入路径 + 训练权重(A/B/C/D 各换 WEIGHT_PATH + OUTPUT_DIR 即可对比)
GPU=0 LQ_DIR=../HYPIR/input/test_faces UPSCALE=4 WEIGHT_PATH=../HYPIR/experiments/ppr10k_faces_paired/checkpoint-65000/ema_state_dict.pth bash hypir/02_run_inference.sh
# 💡06) 预览合成退化效果(HQ -> LQ，看 04c 训练时在线合成的退化长啥样)
GPU=0 HQ_DIR=../HYPIR/input/test_faces_hq NUM_PER_IMAGE=4 bash hypir/06_preview_degradation.sh
# 💡02) 测试外插视角优化效果(在输入没有的角度上渲染图像会很模糊，考虑用HYPIR进行去噪增强)
GPU=0 LQ_DIR=../../output/hypir_test_results/input UPSCALE=4 WEIGHT_PATH=../HYPIR/experiments/beauty_ppr50k_20260721/checkpoint-1000/ema_state_dict.pth OUTPUT_DIR=../../output/hypir_test_results/output bash hypir/02_run_inference.sh

# ── 模型打包与自测(服务器) ──
conda activate hypir
CUDA_VISIBLE_DEVICES=7 python3 ../Reconstruction/enh_model_打包模型.py
conda activate 3dgsr
CUDA_VISIBLE_DEVICES=7 python ../Reconstruction/reconstruction_all.py
conda activate xcodec
CUDA_VISIBLE_DEVICES=7 python3 ../Reconstruction/uwa_video_封装视频.py
CUDA_VISIBLE_DEVICES=7 python3 ../Reconstruction/uwa_video_封装视频.py --only=test_human_zhaocheng
重建模型在 ../Reconstruction/output/B003
封装视频在 ../../output/uwa_format_mp4

# ── 并行训练扫参(04d) —— LR/loss 权重/真实退化配对扫最佳，布局见 EXPERIMENTS.md ──
# GPU=0 PARQUET_PATH=.../rest_beauty.parquet bash hypir/04d_train_sweep.sh 5e-6
```

- 结果：训练 → `../HYPIR/experiments/<exp>/checkpoint-*/`（`<exp>` = `OUTPUT_DIR` 的名字）；推理 → `../HYPIR/results/<输入夹名>/result/*.png`。
- 想要定量指标（PSNR/SSIM/LPIPS + LQ|result|HQ 对比图）用 `GPU=0 bash hypir/05_eval.sh`。
- prompt 默认空 caption；要逐图描述就传 `TXT_DIR`（与 `LQ_DIR` 同构、每图一个 `.txt`）。
- 两条训练路径区别：04b 用真实配对 LQ（不退化）；04c 只给 HQ、LQ 在线合成。同一份 HQ 都可试，对比真实 vs 合成退化。
- `#5 用 BATCH_SIZE=8` 的原因：`queue_size`(256) 须是 `batch_size`(每卡) 的倍数（详见 NOTES.md 排错 #11）。
- 04d 并行扫参（LR/loss 权重/真实退化配对）与训练调参速查见 [EXPERIMENTS.md](EXPERIMENTS.md)。

## 首次准备
```bash
cd <your-code-dir>            # e.g. /data_3d/<uid>/code
git -c http.sslVerify=false clone https://github.com/CrescentVelvet/media_code.git
cd media_code && cp proxy.env.example proxy.env   # 填 http_proxy / https_proxy
conda create -n hypir python=3.10 -y && conda activate hypir
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
INSTALL_DEPS=1 bash hypir/00_setup_env.sh          # 装官方 requirements.txt
HF_DISABLE_SSL=1 bash hypir/01_download_models.sh  # 下 SD2-base + HYPIR_sd2.pth
```
⚠️ HYPIR 的 `diffusers/transformers/peft` 版本 pin 与本仓其他算法冲突，务必用专用 env（`CONDA_ENV=hypir`），别装进共享的 `doll`。
⚠️ 训练默认**暖启动**（从发布 `HYPIR_sd2.pth` 继续练）；机制与自检方法见 [NOTES.md](NOTES.md)「暖启动机制」——**别把 clone 里的 `sd2.py` 还原成官方版**，否则暖启动静默失效。

## Config (env vars, all optional)
| var | default | note |
|---|---|---|
| `CONDA_ENV` | `doll` | conda env to activate (dedicated `hypir` recommended — pins conflict with other algos) |
| `GPU` | _(unset)_ | physical GPU id to pin, e.g. `GPU=0`; leave unset for multi-GPU training |
| `HYPIR_DIR` | `../HYPIR` | official code path |
| `MODEL_DIR` | `../../model/HYPIR` | weights path (sd2_base/ + HYPIR_sd2.pth) |
| `HYPIR_REPO` | official GitHub URL | clone source |
| `INSTALL_DEPS` | `0` (run_all: `1`) | set `1` to `pip install -r requirements.txt` |
| `SKIP_TORCH` | `0` | `1` = filter torch/torchvision pins out of requirements (keep existing torch) |
| `HF_HUB_DISABLE_XET` | `1` | disable HF Xet/CAS Rust path (proxy-unfriendly) |
| `HF_DISABLE_SSL` | `0` | set `1` to download weights with SSL verification disabled |
| `HF_TOKEN` | _(unset)_ | only needed if you point `HF_BASE_REPO` at a gated repo; the default mirror is public |
| `HF_BASE_REPO` | `Manojb/stable-diffusion-2-1-base` | base diffusers model repo (public mirror; original `stabilityai/...` was removed) |
| `HF_LORA_REPO` | `lxq007/HYPIR` | LoRA weights repo |
| `LORA_FILE` | `HYPIR_sd2.pth` | LoRA file name inside HF_LORA_REPO |
| `BASE_MODEL_DIR` | `$MODEL_DIR/sd2_base` | local base model dir (passed as `--base_model_path`) |
| `WEIGHT_PATH` | `$MODEL_DIR/HYPIR_sd2.pth` | LoRA weight file (passed as `--weight_path`) |

### Inference (02)
| var | default | note |
|---|---|---|
| `LQ_DIR` | `../HYPIR/examples/lq` | folder of low-quality images (walked recursively) |
| `TXT_DIR` | `../HYPIR/examples/prompt` | matching prompt folder (.txt per image); empty → `--captioner empty` |
| `OUTPUT_DIR` | `../HYPIR/results/<lq_folder_name>` | writes `result/` + `prompt/` under here |
| `SCALE_BY` | `factor` | `factor` \| `longest_side` |
| `UPSCALE` | `4` | upscaling factor (when `SCALE_BY=factor`) |
| `TARGET_LONGEST_SIDE` | _(unset)_ | required when `SCALE_BY=longest_side` |
| `PATCH_SIZE` / `STRIDE` | `512` / `256` | tiled processing size / stride |
| `SEED` | `231` | |
| `LORA_RANK` / `LORA_MODULES` | `256` / official list | only override if you trained a different config |

### Dataset (03)
| var | default | note |
|---|---|---|
| `DATA_DIR` | _(required)_ | folder of high-quality images |
| `PARQUET_OUT` | `$DATA_DIR/hypir_train.parquet` | output parquet path |
| `PROMPT` | `""` | caption for every image (null-text training) |
| `CROP` | `1` | `1` = slice into 512 patches first; `0` = use images as-is |
| `CROP_SIZE` / `CROP_STRIDE` | `512` / `=CROP_SIZE` | patch size / stride (smaller ⇒ overlap) |
| `CROP_OUT` | `<parquet_dir>/patches` | where patches are saved |

### Training (04)
| var | default | note |
|---|---|---|
| `PARQUET_PATH` | _(required)_ | parquet from 03 (or set `DATA_DIR` to build it first) |
| `OUTPUT_DIR` | `../HYPIR/experiments/exp1` | checkpoints + logs dir |
| `CROP_TYPE` | `none` | `none`\|`random`\|`center` (use `random` if GTs aren't 512×512 and not pre-cropped) |
| `OUT_SIZE` | `512` | must match your GT image size when `CROP_TYPE=none` |
| `MAX_TRAIN_STEPS` | `30000` | |
| `BATCH_SIZE` | `6` | per-GPU |
| `LR_G` / `LR_D` | `1e-5` / `1e-5` | generator / discriminator learning rates |
| `GRAD_ACCUM` | `1` | gradient accumulation steps |
| `SEED` | `231` | |
| `CHECKPOINTING_STEPS` | `500` | save a checkpoint every N steps |
| `LOG_IMAGE_STEPS` / `LOG_GRAD_STEPS` | `100` / `100` | |
| `N_TRAIN_GPU` | _(unset)_ | `>1` → `accelerate launch --num_processes N` (run `accelerate config` first) |
| `MIXED_PRECISION` | _(unset; config=bf16)_ | override accelerate mixed precision |
| `RESUME` | _(unset)_ | `checkpoint-N` dir to resume from |

### Evaluation (05)
| var | default | note |
|---|---|---|
| `TRAIN_DIR` | `../HYPIR/experiments/ppr10k_faces_paired` | 训练产物目录（checkpoint-*/ 在此） |
| `CKPT_STEP` | `65000` | 评测哪个 checkpoint；`WEIGHT_PATH` 默认 `$TRAIN_DIR/checkpoint-$CKPT_STEP/state_dict.pth` |
| `WEIGHT_PATH` | _(see above)_ | 训练好的 LoRA；覆写即评任意权重 |
| `TEST_LQ_DIR` | `.../ppr10k_faces_20260703/lq` | 测试 LQ 图像夹（按同名文件与 HQ 配对） |
| `TEST_HQ_DIR` | `.../ppr10k_faces_20260703/hq` | 测试 HQ 图像夹；设空则只复原不算指标 |
| `EVAL_DIR` | `$TRAIN_DIR/eval_ckpt$CKPT_STEP` | 评测输出（result/ compare/ metrics.csv） |
| `UPSCALE` | `1` | 人脸配对是 512→512 复原；做超分改大 |
| `EVAL_LIMIT` | `50` | 评测张数；`0`=全部 |
| `SAVE_COMPARE` | `1` | `1`=存 LQ\|result\|HQ 三联对比图 |
| `SCALE_BY` / `PATCH_SIZE` / `STRIDE` / `SEED` | `factor` / `512` / `256` / `231` | 同 02 |

## Outputs
- **02 inference**: `OUTPUT_DIR/result/<rel>.png` (restored) + `OUTPUT_DIR/prompt/<rel>.txt`.
- **03 dataset**: `PARQUET_OUT` (+ `patches/*.png` when `CROP=1`).
- **04 training**: `OUTPUT_DIR/checkpoint-<step>/{state_dict.pth, ema_state_dict.pth, ...}`. Point `02`'s `WEIGHT_PATH` at `state_dict.pth` to run your fine-tuned model.
- **05 eval**: `EVAL_DIR/{result,compare}/<rel>.png` + `EVAL_DIR/metrics.csv`（逐图 PSNR/SSIM/LPIPS，含 bicubic 基线）.

## 目录布局
```
<code-dir>/
├── media_code/                  # 本仓
│   ├── proxy.env                # 代理 + 覆盖项, gitignored
│   └── hypir/                   # 编排脚本(本目录)
├── HYPIR/                       # 官方代码(自动 clone 到 ../HYPIR)
└── ../../model/HYPIR/           # 权重(在 <code-dir> 上一级, 各算法共享)
    ├── sd2_base/                # Manojb/stable-diffusion-2-1-base (公开镜像; diffusers: scheduler/tokenizer/text_encoder/unet/vae)
    └── HYPIR_sd2.pth            # 发布 LoRA (lxq007/HYPIR)
```
默认：官方代码 `../HYPIR`、权重 `../../model/HYPIR`（相对本目录）；用 `HYPIR_DIR` / `MODEL_DIR` 覆盖。复用现有 conda env（默认 `doll`），但 HYPIR 的依赖 pin 与其他算法冲突——建议专用 env（`CONDA_ENV=hypir`），`SKIP_TORCH=1` 可不动现有 torch。

## Notes
- Official code & weights follow their own license (HYPIR = non-commercial use only — see the repo). This folder only orchestrates; no official code is copied.
- `.gitattributes` (repo root) forces LF so Windows-pushed scripts run cleanly on Ubuntu.
- `proxy.env` (proxy creds / env overrides) is gitignored — never committed. Don't put credentials in scripts.
- SSL behind a TLS-intercepting corporate proxy: pip uses `--trusted-host`; `hf`/`git` use the CA bundle (`_env.sh` prefers `~/.ca-bundle.crt`, built by `setup_ca_bundle.sh`).
