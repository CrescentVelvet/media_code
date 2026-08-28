# HYPIR runner

在 Ubuntu + NVIDIA 服务器上跑 [HYPIR](https://github.com/XPixelGroup/HYPIR)（SIGGRAPH 2025 图像复原）的**推理 / 数据集构建 / LoRA 训练**。本目录只含编排脚本——官方代码自动 clone、权重从 HuggingFace 下载。

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

# 03d) C 二次美颜数据集(BEAUTY_PASSES=2, RetouchFormer 跑两次 -> hq_beauty_strong)
#      注：BEAUTY_PASSES=2 会一次性产出 A/B/C 三套 parquet(rest / rest_beauty / rest_beauty_strong)，
#      A/B 产物与 BEAUTY_PASSES=1 完全一致；故要么单独跑这条跑 C、要么 A/B 也直接用这条。
# 🔍03d) 抽样看效果
GPU=0 SKIP_PARQUET=1 SAVE_COMPARE=1 BEAUTY_PASSES=2 INPUT_DIR=../HYPIR/input/test_faces_hq OUTPUT_DIR=../../output/hypir_test_results/二次美颜数据预览 bash hypir/03d_build_beauty_dataset.sh
# ♻️03d) 构建全量数据集（多卡）
GPU=0,1,2 NPROC=3 BEAUTY_PASSES=2 INPUT_DIR=../HYPIR/dataset/guojia_datas_20260708 bash hypir/03d_build_beauty_dataset.sh

# 03e) D 去红润美颜数据集(独立跑 RetouchFormer + wavelet 融合: 美颜高频+原图低频 -> hq_beauty_decolor；
#      人脸红润在高频，美颜把皮肤推向暖色统计；DECOLOR_MODE=wavelet(默认,只换低频磨皮,去不掉高频红) | high_freq_dc(高频减DC去红)
#      与 03d 并列(非依次)：自己跑 RetouchFormer + 高斯模糊产 lq_gauss(同 seed 与 03d 逐像素一致)，不依赖 03d。训练/推理脚本均不改)
# 🔍03e) 抽样看效果 (compare/ 下 [LQ|orig|beauty|decolor] 四联图，确认去红润+磨皮保留)
GPU=0 SKIP_PARQUET=1 SAVE_COMPARE=1 DECOLOR_MODE=high_freq_dc INPUT_DIR=../HYPIR/input/test_faces_hq OUTPUT_DIR=../../output/hypir_test_results/去红润美颜数据预览 bash hypir/03e_decolor_beauty_dataset.sh
# ♻️03e) 构建全量数据集 parquet(独立产 hq_beauty_decolor/+lq_gauss/，不依赖 03d)
GPU=0,1,2 NPROC=3 INPUT_DIR=../HYPIR/dataset/guojia_datas_20260708 bash hypir/03e_decolor_beauty_dataset.sh

# 04b) A/B/C/D 依次训练(无需先 accelerate config，换卡就改 GPU=列表、N_TRAIN_GPU 对齐)：
# 🚀04b) A 基线(只高斯模糊，预期会长痘变丑)：
GPU=0,1,2 N_TRAIN_GPU=3 BG=0 PARQUET_PATH=../HYPIR/dataset/beauty_guojia_datas_20260708/rest.parquet OUTPUT_DIR=../HYPIR/experiments/rest bash hypir/04b_train_paired.sh
# 🚀04b) B 复原+美颜(LQ 同样模糊、HQ 换美颜版 1pass，预期修掉长痘、又不毁脸)：
GPU=0,1,2 N_TRAIN_GPU=3 BG=0 PARQUET_PATH=../HYPIR/dataset/beauty_guojia_datas_20260708/rest_beauty.parquet  OUTPUT_DIR=../HYPIR/experiments/beauty bash hypir/04b_train_paired.sh
# 🚀04b) C 二次美颜(LQ 同样模糊、HQ 换迭代美颜版 N pass，预期美颜最强、但可能过磨失结构)：
GPU=0,1,2 N_TRAIN_GPU=3 BG=0 PARQUET_PATH=../HYPIR/dataset/beauty_guojia_datas_20260708/rest_beauty_strong.parquet OUTPUT_DIR=../HYPIR/experiments/beauty_strong bash hypir/04b_train_paired.sh
# 🚀04b) D 去红润美颜(LQ 同样模糊、HQ 换去红润美颜版 wavelet 融合，预期红润减弱、磨皮保留)：
GPU=0,1,2 N_TRAIN_GPU=3 BG=0 PARQUET_PATH=../HYPIR/dataset/beauty_decolor_guojia_datas_20260708/rest_beauty_decolor.parquet OUTPUT_DIR=../HYPIR/experiments/beauty_decolor bash hypir/04b_train_paired.sh

# ── 推理(02/06) ──
# 💡02) 测试原生(发布)模型 —— 指定输入路径
GPU=0 LQ_DIR=../HYPIR/input/test_faces UPSCALE=4 bash hypir/02_run_inference.sh
# 💡02) 测试自己训的 LoRA —— 指定输入路径 + 训练权重(小数据集在 experiments/ppr10k_faces_paired; 大数据集在 experiments/synthetic_exp1/; 退化数据集在experiments/guojia_datas_20260708/; 美颜数据集在 experiments/rest(A) / beauty(B) / beauty_strong(C) / beauty_decolor(D))
GPU=0 LQ_DIR=../HYPIR/input/test_faces UPSCALE=4 WEIGHT_PATH=../HYPIR/experiments/ppr10k_faces_paired/checkpoint-65000/ema_state_dict.pth bash hypir/02_run_inference.sh
# 💡06) 预览合成退化效果(HQ -> LQ，看 04c 训练时在线合成的退化长啥样)
GPU=0 HQ_DIR=../HYPIR/input/test_faces_hq NUM_PER_IMAGE=4 bash hypir/06_preview_degradation.sh
# 💡02) 测试外插视角优化效果(在输入没有的角度上渲染图像会很模糊，考虑用HYPIR进行去噪增强)
GPU=0 LQ_DIR=../../output/hypir_test_results/input UPSCALE=4 WEIGHT_PATH=../HYPIR/experiments/beauty_ppr50k_20260721/checkpoint-1000/ema_state_dict.pth OUTPUT_DIR=../../output/hypir_test_results/output bash hypir/02_run_inference.sh

# ── 模型打包与自测(服务器) ──
conda activate hypir
python3 ../Reconstruction/enh_model_打包模型.py
conda activate 3dgsr
CUDA_VISIBLE_DEVICE=0 python ../Reconstruction/reconstruction_all.py
conda activate xcodec
python3 ../Reconstruction/uwa_video_封装视频.py
python3 ../Reconstruction/uwa_video_封装视频.py --only=test_human_zhaocheng
重建模型在 ../Reconstruction/output/B003
封装视频在 ../../output/uwa_format_mp4


# # ── 并行训练(04d) ── 越练越模糊是 L2 坍缩；扫 LR_G + loss 权重 + 真实退化配对 找最佳
# #   (04d 默认 steps=30000 / ckpt_every=100 / LR_D=LR_G / 后台；逐 ckpt 评找峰值，常在早期)
# # 复原+美颜：多个 LR 并行（各占一卡）：
# GPU=0 PARQUET_PATH=../HYPIR/dataset/beauty_guojia_datas_20260708/rest_beauty.parquet bash hypir/04d_train_sweep.sh 5e-6
# GPU=1 PARQUET_PATH=../HYPIR/dataset/beauty_guojia_datas_20260708/rest_beauty.parquet bash hypir/04d_train_sweep.sh 2e-6
# GPU=2 PARQUET_PATH=../HYPIR/dataset/beauty_guojia_datas_20260708/rest_beauty.parquet bash hypir/04d_train_sweep.sh 1e-5
# GPU=3 PARQUET_PATH=../HYPIR/dataset/beauty_guojia_datas_20260708/rest_beauty.parquet LR_D=1e-5 SWEEP_TAG=disc1e5 bash hypir/04d_train_sweep.sh 5e-6
# # 加强美颜(C 组)：把 PARQUET_PATH 换成 rest_beauty_strong.parquet 即可对 C 组扫同样 LR/loss：
# GPU=0 PARQUET_PATH=../HYPIR/dataset/beauty_guojia_datas_20260708/rest_beauty_strong.parquet SWEEP_TAG=strong_5e6 bash hypir/04d_train_sweep.sh 5e-6
# GPU=1 PARQUET_PATH=../HYPIR/dataset/beauty_guojia_datas_20260708/rest_beauty_strong.parquet SWEEP_TAG=strong_2e6 bash hypir/04d_train_sweep.sh 2e-6
# # 真实退化配对(03b 的 360p 相机 LQ + RAW HQ——发布模型擅长的配方，纯高斯模糊 LQ 之外的对照)：
# #   先建 parquet(03b)，再扫 LR：
# HQ_DIR=../HYPIR/dataset/ppr10k_faces_20260703/hq LQ_DIR=../HYPIR/dataset/ppr10k_faces_20260703/lq bash hypir/03b_build_paired_dataset.sh   # -> .../ppr10k_faces_20260703/hypir_paired.parquet
# GPU=4 PARQUET_PATH=../HYPIR/dataset/ppr10k_faces_20260703/hypir_paired.parquet bash hypir/04d_train_sweep.sh 5e-6
# GPU=5 PARQUET_PATH=../HYPIR/dataset/ppr10k_faces_20260703/hypir_paired.parquet bash hypir/04d_train_sweep.sh 2e-6
# # loss 权重调节(抗 L2 坍缩：up GAN、down L2；LAMBDA_GAN/LAMBDA_LPIPS/LAMBDA_L2 透传 04b，SWEEP_TAG 标注实验名)：
# GPU=6 PARQUET_PATH=../HYPIR/dataset/beauty_guojia_datas_20260708/rest_beauty.parquet LAMBDA_GAN=2 LAMBDA_L2=0.5 SWEEP_TAG=gan2_l2p5 bash hypir/04d_train_sweep.sh 5e-6
# GPU=7 PARQUET_PATH=../HYPIR/dataset/beauty_guojia_datas_20260708/rest_beauty.parquet LAMBDA_GAN=1 LAMBDA_LPIPS=10 SWEEP_TAG=gan1_lp10 bash hypir/04d_train_sweep.sh 5e-6
# GPU=8 PARQUET_PATH=../HYPIR/dataset/beauty_guojia_datas_20260708/rest_beauty.parquet LAMBDA_L2=0.3 SWEEP_TAG=l2p3 bash hypir/04d_train_sweep.sh 5e-6
```

- 结果：训练 → `../HYPIR/experiments/<exp>/checkpoint-*/`（`<exp>` = `OUTPUT_DIR` 的名字，如 04b=`ppr10k_faces_paired`、04c 默认=`synthetic_exp1`、换数据集时=`guojia_datas`）；推理 → `../HYPIR/results/<输入夹名>/result/*.png`。
- 想要定量指标（PSNR/SSIM/LPIPS + LQ|result|HQ 对比图）用 `GPU=0 bash hypir/05_eval.sh`。
- prompt 默认空 caption；要逐图描述就传 `TXT_DIR`（与 `LQ_DIR` 同构、每图一个 `.txt`）。
- 两条训练路径区别：04b 用真实配对 LQ（不退化）；04c 只给 HQ、LQ 在线合成（HYPIR 默认退化 blur/sinc/noise/jpeg）。同一份 HQ 都可试，对比真实 vs 合成退化。
- `#5 用 BATCH_SIZE=8` 的原因：官方 `sd2_train.yaml` 的 `RealESRGANBatchTransform` 要求 `queue_size`(默认 256) 是 `batch_size`(每卡) 的倍数——`256%6≠0` 会报错，`256%8=0` OK。想用 bs=6 就改 `queue_size=252`(=6×42)（在 04c/04_train.sh heredoc 加 `cfg.data_config.train.batch_transform.params.queue_size=252`），别动 bs。

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

---

以下为详细参考（流程原理 / 各脚本参数 / 排错 / 目录布局）。

## Pipeline（推理流程详解）
对应官方代码 `HYPIR/enhancer/base.py::enhance` + `HYPIR/enhancer/sd2.py::forward_generator` + `HYPIR/utils/common.py`。一张 LQ 图像从输入到输出经过 5 步：

1. **上采样（bicubic 插值）** — `F.interpolate(lq, scale_factor=upscale, mode="bicubic")`。先把 LQ 双三次插值放大到目标分辨率（`scale_by=factor` 时按固定倍数，`longest_side` 时按长边到固定尺寸）。这一步的输出同时存为 `ref`（参考图），供第 5 步小波融合用。若短边 ≤ `patch_size`(512)，还会把短边 resize 到至少 512，保证 VAE 有足够大的画布（此 resize 只作用于送入 VAE 的 `lq`，不改动 `ref`）。
2. **VAE 编码（分块，patch=512）** — `lq` 归一化到 `[-1,1]`、pad 到 8 的倍数（`vae_scale_factor=8`），再用 `make_tiled_fn` 分块送入 `vae.encode(...).latent_dist.sample()`。tile 在**像素空间**大小为 `patch_size`(512)、stride=`stride`(256)，下采样到**潜空间** 64（512/8）。重叠区用高斯权重平滑拼接，避免接缝。注意 `.sample()` 从 VAE 编码分布里采样，带轻微随机性——这就是推理要设 `--seed` 的原因。
3. **UNet 一步去噪（LoRA + SD2，t=200）** — 见下方「一步去噪」。加载了 LoRA 的 SD2 UNet 以 LQ 潜变量为输入、在 timestep=200 做一次前向 + 一次 DDPM 反推，得到复原的 HR 潜变量。同样分块（潜空间 64 / stride 32）。
4. **VAE 解码（分块）** — `make_tiled_fn(vae.decode(tile).sample, scale_type="up", scale=8, channel=3)` 把复原潜变量解码回像素空间（潜 64 → 像素 512 的 tile）。然后裁掉 padding、`(x+1)/2` 回到 `[0,1]`、bicubic 缩回 `ref` 的尺寸 `(h0,w0)`。
5. **小波融合** — `wavelet_reconstruction(x, ref)`：把第 4 步解码输出（`content`）与第 1 步上采样参考图（`style=ref`）做多尺度融合。见下方「小波融合」。

### 一步去噪（one-step denoising）怎么回事
标准 DDPM 生成要迭代 ~1000 步（DDIM ~50 步）逐步去噪。**HYPIR 不迭代，只走一步**，关键在于把「LQ 潜变量」直接当作部分加噪的 `x_t`：

```
z_in = z_lq * vae.scaling_factor          # LQ 潜变量当 x_t（不再加随机噪声！）
eps  = UNet_lora(z_in, t=200, text_embed)  # UNet 预测"把 x_0 变成 x_t 的噪声"
z0   = scheduler.step(eps, coeff_t=200, z_in).pred_original_sample   # 一步反推 x_0
```

- DDPM 有闭式关系：`x_0 = (x_t − √(1−ᾱ_t)·eps) / √(ᾱ_t)`。给定 `x_t`（=LQ 潜变量）和 UNet 预测的噪声 `eps`，**一次代数运算**就能解出估计的干净潜变量 `x_0`，不需要迭代。
- 为什么一步就够？因为 LoRA 是**专门为 t=200 这一步训练**的：训练时同样把 GT 的 LQ 潜变量当 `x_t`、UNet 预测 `eps`、一步反推 `x_0`，再用 `x_0` 解码出的图像与 GT HR 算 L2+LPIPS+GAN 损失（见 `HYPIR/trainer/base.py::optimize_generator`）。也就是说 LoRA 学到的就是「在 t=200 这一步、从 LQ 潜变量一步反推出 HR 潜变量」这个映射，推理时自然一步即出。
- `model_t=200` 是喂给 UNet 的时间步标签（UNet 以为自己正在 t=200 去噪）；`coeff_t=200` 是反推 `x_0` 时用的噪声调度系数所在时间步。两者在此都取 200。
- 这就是论文标题里的 "score prior"：扩散 UNet 的噪声预测（score）在单个时间步上提供了一个强先验，把退化图直接拉回干净图——快（一次前向）且借用了 SD2 在海量图像上学到的先验。

### 小波融合（wavelet reconstruction）怎么回事
解码出的 HR 图像纹理清晰，但扩散模型可能引入色偏/结构幻觉；而上采样 LQ 颜色和整体结构是可靠的，只是模糊（缺高频）。小波融合取两者之长：

- `wavelet_decomposition(img, levels=5)` 用逐级放大的高斯模糊（dilation 半径 1,2,4,8,16）把图像拆成 **低频**（`low`：颜色、光照、大尺度结构）和 **高频**（`high`：边缘、纹理、细节）。
- `wavelet_reconstruction(content=x_decoded, style=ref)` 的实际计算是：
  ```
  result = content_high + style_low
         = 解码输出的高频 + 上采样LQ的低频
  ```
  即**高频（纹理/边缘）来自扩散解码输出，低频（颜色/结构）来自上采样 LQ**。
- 效果：最终图保留 LQ 原本正确的颜色与构图（低频），同时注入扩散模型生成的锐利纹理（高频），避免色偏和结构漂移，又实现了超分。这是 SUPIR/CCSR 一脉相承的经典 trick。

> 总结一句：**上采样定颜色结构 → VAE 编码进潜空间 → LoRA-UNet 一步反推干净潜变量 → VAE 解码回像素 → 与上采样原图小波融合（取扩散的高频 + LQ 的低频）**。

### 合成退化流程（训练时 HQ→LQ，03c/04c 与 06 预览共用）
官方 `RealESRGANDataset` + `RealESRGANBatchTransform` 把 HQ 合成退化成 LQ（HYPIR 发布模型本身的训练方式；03c/04c 在线跑、06 离线预览复用同一套，参数取自 `configs/sd2_train.yaml`，无复制）。

**1) 取 HQ + 生成核**（`RealESRGANDataset.__getitem__`）
- 加载 HQ；`crop_type=none` 时 resize 到 `out_size`(512)（`random` 则裁 512 patch）。
- 随机生成 3 个核：`kernel1`/`kernel2`（模糊核：iso/aniso/generalized/plateau，或 sinc 低通，按 `kernel_prob`/`sinc_prob`）、`sinc_kernel`（最终 sinc 滤波核）。
- 可选 flip/rot 增强（06 预览/确定性时关掉）。
- 返回 `{hq, kernel1, kernel2, sinc_kernel, txt}`。

**2) 两阶段退化**（`RealESRGANBatchTransform.__call__`，对 `hq` 操作，参数全在 config 的 `batch_transform.params`）
- `GT = USMSharp(hq)`：先 USM 锐化 HQ 当 GT（匹配发布模型训练时的 GT 预处理）。
- **第一阶段**：`filter2D(hq, kernel1)` 模糊 → 随机 resize（up/down/keep，scale 取自 `resize_range`，mode 随机 area/bilinear/bicubic）→ 加噪（按 `gaussian_noise_prob` 选高斯或泊松，强度 `noise_range`/`poisson_scale_range`，可灰噪）→ JPEG（质量取自 `jpeg_range`）。
- **第二阶段**：按 `second_blur_prob` 可能再 `filter2D(kernel2)` 模糊 → 按 `stage2_scale`(≈4) 下采样 → 随机 resize2 → 加噪2 → **resize-back + sinc 滤波** 与 **JPEG2** 顺序随机（各 0.5，避免扭曲条纹）→ 若 `resize_back=true` 缩回原尺寸(512)。
- `LQ = clamp(round(out·255))/255`（量化到 8-bit）。

**3) 训练池**（`queue_size>0` 时，仅训练用）
- 把 `{GT,LQ,txt}` 入队（池容量 `queue_size`=256）；满后随机抽一个**缓存样本**返回（增 batch 内多样性），故 `queue_size` 须被 `batch_size`(每卡) 整除（`256%6` 报错 → 用 `BATCH_SIZE=8` 或改 `queue_size=252`，见排错 #11）。
- **06 预览脚本设 `queue_size=0`**：跳过池，直接返回**当前 HQ 的 LQ**（否则满 256 后返回的是别的图的缓存，看不到当前 HQ 的退化）。

> 一句话：`HQ →(USM锐化)→ GT`；`HQ →(两阶段 blur/resize/noise/jpeg/sinc)→ LQ`。训练时 LQ 现场合、每 epoch 随机刷新；06 离线跑同一套、`queue_size=0` 看当前图。

> ⚠️ **本仓 clone 的 `HYPIR/dataset/batch_transform.py` 是修改版**（只高斯模糊）：`__call__` 不走上面官方那套两阶段 blur/resize/noise/jpeg/sinc，而是简化为——
> ```python
> # 修改版 __call__（你的 clone 实际跑的）
> hq = USMSharp(batch[hq_key])                 # GT = USM 锐化后的 HQ
> kernel = random.randint(1,5)*2+1             # 3/5/7/9/11
> num = random.randint(1,5)                    # 重复 1-5 次
> lq = hq.clone()
> for _ in range(num):
>     lq = torchvision.transforms.GaussianBlur(kernel_size=kernel, sigma=(1.0,2.0))(lq)
> # 没有 resize / noise(gauss/poisson) / JPEG / sinc / 第二阶段
> ```
> 所以上面官方流程的**第一阶段后半（resize/noise/JPEG）和整个第二阶段（sinc/JPEG2/resize-back）在你环境里不发生**——LQ 只是「HQ 被 1-5 次随机高斯模糊」。`queue_size` 训练池保留（故 `256%6` 仍报错，见排错 #11）。**以你 clone 的 `batch_transform.py` 实际代码为准**；06 预览和 04c 训练复用的就是这版，看到/用到的是纯高斯模糊退化。想恢复完整官方退化就把 `batch_transform.py` 还原成 GitHub 版。

## Paired face fine-tuning on REAL LQ+HQ (03b / 04b) — `crop_faces_paired.py` output → HYPIR

The official `03`/`04` path trains on **HQ only** and *synthesizes* LQ via
RealESRGAN degradation. To train on **real** degradation (e.g. the 360p camera
LQ + RAW-decoded HQ face pairs from `face_crop/crop_faces_paired.py`), use the
paired plug-ins — **no official file is modified**: the derived config points
`target:` at plug-in classes in `paired_face_plugin.py` (importable via
PYTHONPATH), and `train_paired.py` subclasses `SD2Trainer` to warm-start the
LoRA from the released `HYPIR_sd2.pth`.

> 中文说明：官方 `03/04` 只用 HQ、LQ 是现场用 RealESRGAN 合成退化的（盲复原配方）。
> 你用 `face_crop/crop_faces_paired.py` 建的是「真实 360p 相机 LQ + RAW 解码 HQ」配对，
> 想直接喂真实退化，就用这套配对插件。**不改任何官方文件**：填好的配置把 `target:`
> 指向 `paired_face_plugin.py` 里的插件类（靠 `04b_train_paired.sh` 设的 `PYTHONPATH`
> 导入），`train_paired.py` 再子类化 `SD2Trainer` 实现从发布权重暖启动。
>
> 核心思想：7k 张配对从零训 LoRA 不够（发布模型是 bs1024 大数据训的），所以默认
> **暖启动**——把发布的 `HYPIR_sd2.pth` 当 LoRA 初始化，再在你的人脸数据上继续练，
> 等于把通用复原模型「适配」到人脸域。数据流见下图，训练循环(一步去噪+L2/LPIPS/GAN)
> 与官方完全一致，只是 LQ 来自真实配对而非合成。

```
HQ folder (.../hq/<stem>_faceN.png)  ┐  paired by filename
LQ folder (.../lq/<stem>_faceN.png)  ┘
   └─ build_paired_dataset.py     -> parquet(hq_path, lq_path, prompt)
        └─ PairedFaceDataset        : load HQ+LQ, resize/paired-crop to 512,
                                      same flip/rot -> {hq, lq, txt}
        └─ PairedFaceBatchTransform : USM-sharpen HQ, rename -> {GT, LQ, txt}
        └─ FineTuneSD2Trainer       : warm-start LoRA from HYPIR_sd2.pth, then
                                      the unchanged one-step + L2/LPIPS/GAN loop
```

## 配对人脸微调全流程

这篇给第一次用的人：从「数据已上传」到「跑完训练、看到还原结果」，每步都有可复制的命令。命令都在服务器上执行。

### 你需要先有的
- 一台 Ubuntu + NVIDIA GPU 服务器（训练建议 A100 / H100 / 4090；只推理 T4 也行）。
- 已用 `face_crop/crop_faces_paired.py` 建好的**配对人脸数据集**（`hq/` 与 `lq/` 同名 PNG），并上传到默认路径：
  `../HYPIR/dataset/ppr10k_faces_20260703/{hq,lq}`
- 服务器上已装好 `conda` 和 `git`。

### 训练到底在干什么（一句话版）
把「模糊的 360p LQ 人脸」和「清晰的 HQ 人脸」**成对**喂给模型，让它学会把模糊脸还原成清晰脸。我们不是从零训练，而是在官方已发布的 `HYPIR_sd2.pth` 上**继续练（暖启动）**，让它专精你的人脸数据——所以 7 千张就够、收敛也快。

### 调参速查（按需改）
- 显存不够：`BATCH_SIZE=4`，或 `GRAD_ACCUM=2`（等效翻倍 batch）。
- 想更稳、别把发布模型「练歪」：`LR_G=5e-6 LR_D=5e-6`。
- 想多练/少练：`MAX_TRAIN_STEPS=30000`（更多）或 `10000`（更快）。
- 多卡：先跑一次 `accelerate config` 选 multi-GPU，再 `N_TRAIN_GPU=8 bash hypir/04b_train_paired.sh`（此时**不要**设 `GPU=`）。
- 跑报错了：看下面「可能遇到的问题」对应条目。

## Synthetic degradation (03c / 04c — 只输入 HQ，在线退化 + 暖启动)
`03c_build_synthetic_dataset.sh` 是 `03b`(真实配对) 的**合成退化**对照：只给 HQ 文件夹，产出 parquet(`image_path` + `prompt`)；训练时由官方 `RealESRGANDataset` + `RealESRGANBatchTransform` **在线合成 LQ**（HYPIR 默认退化：blur/sinc/noise/jpeg 两阶段，每 epoch 随机刷新）——不存 LQ 文件、增强更多样、省盘。这是 HYPIR 发布模型本身的训练方式。`04c_train_synthetic.sh` 在此基础上**暖启动发布 LoRA**（用官方 `sd2_train.yaml` + `train_paired.py`/`FineTuneSD2Trainer`，你改过的 `sd2.py` 会 `torch.load(config.weight_path)` 真正加载）。
命令见上文「常用命令」#4/#5（建 HQ-only parquet → 04c 在线退化训练，默认暖启动 `HYPIR_sd2.pth`）。
- RealESRGANDataset 需要 HQ ≥ 512：`CROP_TYPE=random`（在线裁 512 patch，HQ>512 时推荐）或 `CROP_TYPE=none`（HQ 需预 resize 到 512）。HQ < 512 时先 resize 上采样到 ≥512。
- **04c = 暖启动 + 在线退化**（默认从 `$MODEL_DIR/HYPIR_sd2.pth` 暖启动）。想**从零训**（官方默认）就用 `04_train.sh`：`PARQUET_PATH=... CROP_TYPE=random bash hypir/04_train.sh`。
- 与 `04b_train_paired` 区别：04b_train_paired 用真实 LQ（`sd2_train_paired.yaml`，不退化）；04c 用合成 LQ（官方 `sd2_train.yaml`，在线退化）。同一份 HQ 两种方式都试，对比真实 vs 合成退化的效果。

## 美颜增强 (03d/03e → 04b — RetouchFormer 美颜蒸馏 + A/B/C/D 对比)
`03d_build_beauty_dataset.sh` 是 `03c`(在线高斯模糊退化) 的**美颜对照**：只输入一个「原图人脸」文件夹，用 [RetouchFormer](../retouchformer/README.md) 对每张美颜，每图同时保存多张像素级对齐的 512×512 PNG（同源 src 张量派生，任意尺寸/宽高比输入都安全——模型 VRT 写死 512×512，非方形会被 CenterCrop，`hq_orig`/`lq_gauss` 存的正是这个 crop）：
- `hq_orig/<name>.png` = 原图对齐 crop（**A 组复原实验的 HQ 目标**）
- `hq_beauty/<name>.png` = RetouchFormer 美颜结果，1 pass（**B 组美颜实验的 HQ 目标**）
- `hq_beauty_strong/<name>.png` = RetouchFormer 迭代美颜 `BEAUTY_PASSES` 次（**C 组加强美颜实验的 HQ 目标**；仅 `BEAUTY_PASSES>=2` 产出，默认 1 不产）
- `lq_gauss/<name>.png` = 同一 crop 的高斯模糊退化（**A/B/C/D 四组共用的 LQ 输入**，复现简化版 `batch_transform.py` 的随机 kernel 3/5/7/9/11、sigma 1-2、重复 1-5 次，每图一个 FIXED seeded 实现，离线非每 epoch 重随机）

> **去红润美颜（03e，D 组的 `hq_beauty_decolor`）**：`03e_decolor_beauty_dataset.sh` 与 03d **并列**（非依次）——自己跑 RetouchFormer 得 `beauty`（红润），再做小波融合取美颜高频（磨皮纹理）+ 原图低频（原色调）。RetouchFormer 的红润主要在 **beauty 的高频**（per-channel DC 偏移——美颜把皮肤推向"健康暖色"统计，写进高频），不在低频：故只换低频（`DECOLOR_MODE=wavelet`，默认原行为）往往去不掉红；设 `DECOLOR_MODE=high_freq_dc` 减掉 `beauty_high` 的 per-channel 空间 DC 再加 `orig_low`，保留磨皮结构（高频纹理）只剥掉红色调常数项。若 `high_freq_dc` 仍去不掉（红是局部非全局 DC），升级全局色调校正。`hq_orig`/`hq_beauty` 是内存中间量不存盘；产出 `hq_beauty_decolor/` + `lq_gauss/` + `rest_beauty_decolor.parquet`。RetouchFormer 加载/transform/blur 逻辑 copy 自 `build_beauty_dataset.py`（并列脚本，重复正常）；wavelet 三函数 verbatim copy 自 `HYPIR/utils/common.py:32-80`（仅依赖 torch）。**训练/推理脚本均不改**：HQ 目标去红润，LoRA 学到的还原也去红润——换 `PARQUET_PATH`/`WEIGHT_PATH` 即可。

> 迭代美颜（C 组的 `hq_beauty_strong`）：RetouchFormer 输入输出都是 `[1,3,512,512]` in `[-1,1]`，把上一轮输出 `clamp(-1,1)` 后喂回去即叠加一次美颜——更磨皮、更去瑕疵。pass 1 永远存 `hq_beauty`；pass 2..N 的最终结果存 `hq_beauty_strong`（中间 pass 不留）。`BEAUTY_PASSES>2` 继续叠加但收益递减，**2 通常是甜点**。

**为什么多套（多张 parquet）而不是一套**：现有 03c/04c 在线退化（LQ=高斯模糊, HQ=原图）虽能复原模糊，但会「**长痘变丑**」——模型过度增强、凭空 invent 皮肤瑕疵。把 HQ 目标换成 RetouchFormer 的美颜版（已去瑕疵+磨皮保结构）而 LQ 保持同样的模糊，模型仍学去模糊（增强不失）但目标变成干净光滑皮肤，故不再 invent 瑕疵。同时建多套做 A/B/C/D 对比（A/B/C/D **共用同一 `lq_gauss`**，单变量对比只 HQ 目标不同）：
| 组 | parquet | HQ 目标 | 预期 |
|---|---|---|---|
| **A 基线** | `rest.parquet` | `hq_orig`（原图） | 复原模糊但会「长痘变丑」（= 现有 03c 风格） |
| **B 美颜** | `rest_beauty.parquet` | `hq_beauty`（美颜 1 pass） | 修掉长痘、又不毁脸 |
| **C 加强美颜** | `rest_beauty_strong.parquet` | `hq_beauty_strong`（迭代 `BEAUTY_PASSES` 次） | 美颜最强，但可能过磨失结构（需对比 B 找甜点） |
| **D 去红润美颜** | `rest_beauty_decolor.parquet` | `hq_beauty_decolor`（wavelet 融合：美颜高频+原图低频） | 红润减弱、磨皮保留（修掉 B 的红润副作用） |

> C 组的 `rest_beauty_strong.parquet` 仅在 `BEAUTY_PASSES>=2` 时产出。默认 `BEAUTY_PASSES=1` 只产 A/B 两张；要跑 C 组就设 `BEAUTY_PASSES=2`（一次产出 A/B/C 三张，A/B 产物不变、向后兼容）。
>
> D 组的 `hq_beauty_decolor` + `lq_gauss` + `rest_beauty_decolor.parquet` 由 `03e` **独立**产出（输入原始人脸，自己跑 RetouchFormer + wavelet + 高斯模糊，不依赖 03d）。D 与 A/B/C 并列、各自产 `lq_gauss`（同 `BLUR_SEED`=231 + 同图顺序 → 逐像素一致，单变量对比「HQ 目标颜色」对红润的影响）。

各喂 04b 训一个（`OUTPUT_DIR` 分开），再用 `05_eval.sh` / `02_run_inference.sh` 评测算指标，对比哪组不毁脸、美颜多强。

**conda env**：不强制——默认沿用当前已激活 env，缺包（torch/torchvision/PIL、Phase B 的 polars）就 pip 兜底装；想强制专 env 就设 `RETOUCH_CONDA_ENV`（Phase A）/ `HYPIR_CONDA_ENV`（Phase B）。官方推荐 Phase A 用 `retouchformer` env（python3.8 + torch1.13.1，含 stylegan2 CUDA 算子），但别的 env 也能跑（op/ 非 Linux 或 torch 版本不匹配会回退纯 PyTorch，慢但能出图）。**前置**：放好 `gen_best.pth`（`retouchformer/01_download_models.sh`，百度网盘手动下，提取码 `reto`）。

### 构建数据集
命令见上文「常用命令」#03d（抽样 / 单卡 / 多卡 / C 加强美颜）+ #03e（D 去红润美颜）。03d 产出在默认 `beauty_<input>/` 下：`hq_orig/` + `hq_beauty/` +（`BEAUTY_PASSES>=2` 时）`hq_beauty_strong/` + `lq_gauss/` + `compare/`（SAVE_COMPARE=1）+ `rest.parquet` + `rest_beauty.parquet` +（`BEAUTY_PASSES>=2` 时）`rest_beauty_strong.parquet`。03e **独立**产出在默认 `beauty_decolor_<input>/` 下：`hq_beauty_decolor/` + `lq_gauss/` + `compare/`（SAVE_COMPARE=1，[LQ|orig|beauty|decolor]）+ `rest_beauty_decolor.parquet`。常用参数：
- `BEAUTY_PASSES=2` 跑 C 组加强美颜（迭代美颜，产 `hq_beauty_strong` + `rest_beauty_strong.parquet`）；默认 `1` 只产 A/B。一次 `=2` 会同时产出 A/B/C 三套，A/B 产物不变。
- `SAVE_COMPARE=1` 额外存 `compare/<name>.png` = `[LQ模糊 | 原图 | 美颜(1pass)` +（`BEAUTY_PASSES>=2`）` | 加强美颜(Npass)]` 横拼，一眼核对各张对齐 + 模糊度 + 美颜强度。
- 想抽样核对或做 PPT 展示：加 `SAVE_COMPARE=1 SKIP_PARQUET=1`（只出图、不建 parquet，只跑 Phase A 用当前 env 即可）——产出的 `compare/<name>.png` 把「模糊退化 → 原图 → 美颜 → 加强美颜」各态并排，对齐 + 模糊度 + 美颜强度一目了然，直接可下到 PPT；要单独摆版就用 `hq_orig/` + `hq_beauty/` + `hq_beauty_strong/` + `lq_gauss/` 各张原图。输入用任意人脸小夹即可。
- 只要 `hq_orig`+`hq_beauty[+strong]`、不要模糊 LQ（则也不配对、不建 parquet）：加 `SKIP_BLUR=1`。
- 高斯模糊随机种子复现：`BLUR_SEED=231`（默认，与 HYPIR 的 `SEED` 同值）。
- **03e 专属** `DECOLOR_MODE`：`wavelet`（默认，只换低频，去不掉高频红，作对照）/ `high_freq_dc`（减 `beauty_high` 的 per-channel 空间 DC 去红，保留磨皮结构）。红润在 beauty 高频时必须设 `high_freq_dc`；若仍去不掉（红是局部非全局 DC），升级全局色调校正。
- 模糊作用于 raw 对齐 crop（非 `USM(orig)`），与 03c 的 `LQ=blur(USM(orig))` 略有偏差；但 A/B/C 共用同一 `lq_gauss`，对比仍是单变量。NB：`lq_gauss` 是离线固定模糊（每图一个实现），不像 03c/04c 每 epoch 在线重随机——故 A 是「略少增强的 03c 基线」，但 A vs B vs C 是干净的单变量实验（只 HQ 目标不同）。

#### 去红润诊断（`decolor_probe.py`）
`decolor_probe.py` 是一次性诊断脚本，跑一张图量化红润来源、对比各去红方案，用于决定 03e 该用哪个 `DECOLOR_MODE`。它输出：beauty vs src 的 per-channel mean + ΔR-B（量化红润）、`beauty_high` 的 per-channel DC（判断红是全局 DC 还是局部）、方案 C（attention mask 像素融合：`mask·beauty + (1-mask)·src`，mask 取自 RetouchFormer forward 第 164-166 行返回的 `attention_list`，降维用 channel-max、取最细尺度 resize 到 512）的 soft/hard 变体 ΔR-B，并产一张 `[src|beauty|mask|C_soft|C_hard|high_freq_dc]` 横拼对比图。看三个数即可决策：`>0.5 blemish area (max)` 小=mask 方向对、≈1=反了用 `1-mask`；`C_soft ΔR-B` 接近 src(0)=方案 C 去红成功→整合进 03e 加 `DECOLOR_MODE=attention_mask`；仍红=瑕疵区也带红→方案 C + 色调校正组合。需在 `hypir` env 跑（脚本 import HYPIR SD2Enhancer 做 LoRA 推理；`retouchformer` env 无 diffusers 会崩）：
```bash
conda activate hypir
GPU=0 INPUT_DIR=../HYPIR/input/test_faces_hq \
  OUTPUT_DIR=../../output/hypir_test_results/probe \
  python hypir/decolor_probe.py
GPU=0 LORA_PATHS=../HYPIR/experiments/beauty_ppr50k_20260721/checkpoint-1000/ema_state_dict.pth,../HYPIR/experiments/beauty_strong_20260814/checkpoint-1000/ema_state_dict.pth,../HYPIR/experiments/beauty_decolor_20260828/checkpoint-1000/ema_state_dict.pth \
  INPUT_DIR=../../output/hypir_test_results/input \
  OUTPUT_DIR=../../output/hypir_test_results/probe \
  python hypir/decolor_probe.py
```

### A/B/C/D 对比训练
命令见上文「常用命令」#04b（A 基线 / B 美颜 / C 加强美颜 / D 去红润美颜，各自 `OUTPUT_DIR` 分开，默认暖启动 `HYPIR_sd2.pth`）。
- 四条都默认从 `$MODEL_DIR/HYPIR_sd2.pth` 暖启动（暖启动机制见下文「⚠️ 暖启动机制」；勿把 clone 里 `sd2.py` 还原成官方版，否则暖启动静默失效）。
- 训完用 `05_eval.sh` 算 PSNR/SSIM/LPIPS + 三联对比图，或 `02_run_inference.sh` 各跑一组测试图肉眼对比：
  ```bash
  # A vs B vs C vs D 各推理一组测试图(OUTPUT_DIR 分开)
  GPU=0 LQ_DIR=.../test_faces \
    WEIGHT_PATH=../HYPIR/experiments/rest/checkpoint-N/state_dict.pth \
    OUTPUT_DIR=../HYPIR/results/beauty_A bash hypir/02_run_inference.sh
  GPU=0 LQ_DIR=.../test_faces \
    WEIGHT_PATH=../HYPIR/experiments/beauty/checkpoint-N/state_dict.pth \
    OUTPUT_DIR=../HYPIR/results/beauty_B bash hypir/02_run_inference.sh
  GPU=0 LQ_DIR=.../test_faces \
    WEIGHT_PATH=../HYPIR/experiments/beauty_strong/checkpoint-N/state_dict.pth \
    OUTPUT_DIR=../HYPIR/results/beauty_C bash hypir/02_run_inference.sh
  GPU=0 LQ_DIR=.../test_faces \
    WEIGHT_PATH=../HYPIR/experiments/beauty_decolor/checkpoint-N/state_dict.pth \
    OUTPUT_DIR=../HYPIR/results/beauty_D bash hypir/02_run_inference.sh
  ```
- 四条路径 LQ/HQ 取舍对比：04b 真实配对 LQ=真实退化(360p 相机)、HQ=原图；03c/04c 在线 LQ=每 epoch 重随机高斯模糊、HQ=原图；03d 离线 LQ=固定高斯模糊、HQ=原图**或**美颜版（1 pass = B，迭代 N pass = C，与 A 做单变量对比）；03e 与 03d 并列、独立跑 RetouchFormer + wavelet 融合得 D 的 HQ（自己产 lq_gauss，同 seed 与 03d 逐像素一致）。

## 暖启动机制
**背景**：官方 GitHub 的 `HYPIR/trainer/sd2.py` 里 `SD2Trainer.init_generator` 只做 `init_lora_weights="gaussian"`（随机初始化），**完全不加载 LoRA 权重** —— 即官方默认是从零训，没有暖启动。

**改动**：本仓库 clone 的 `HYPIR/HYPIR/trainer/sd2.py` 被改过，在 `init_generator` 里加了加载逻辑（`grep -n "weight_path\|torch.load" sd2.py` 可见第 57-58 行）：
```python
print(f"Load model weights from {self.config.weight_path}")
state_dict = torch.load(self.config.weight_path, map_location="cpu", weights_only=False)
# ... load_state_dict ...
```
**这是暖启动能生效的唯一原因**。没有这两行，下面的链路全断。

**暖启动链路**（04c / 04b 都走这条）：
```
config.lora_weight_path = .../HYPIR_sd2.pth        # 04c/04b 在填好的 config 里设
  ↓ FineTuneSD2Trainer.init_generator (paired_face_plugin.py)
config.weight_path = config.lora_weight_path        # 官方 config 无 weight_path 字段 → hasattr=False → 赋值
  ↓ super().init_generator()  (你改过的 sd2.py)
torch.load(config.weight_path) → load_state_dict    # 真正把发布 LoRA 载入 UNet
  → 暖启动成功（checkpoint-* 是暖启动来的，不是从零）
```

**勿改错清单**：
- 🔴 **别把 clone 里的 `sd2.py` 还原成官方版**。一旦丢了第 57-58 行的 `torch.load`，`FineTuneSD2Trainer` 设的 `weight_path` 无人读 → 暖启动静默失效，变回从零训（不会报错，但 checkpoint 质量退化，且很难察觉）。
- 🔴 **别删 `FineTuneSD2Trainer` 里的 `config.weight_path = lora_wp` 映射**。否则 `sd2.py` 读 `config.weight_path` 时缺键 → `Missing key weight_path` 直接崩。
- 🟡 改过的 `sd2.py` 是**无条件** `torch.load(config.weight_path)`（grep 未见 `if` guard）。所以 `lora_weight_path` 留空（想从零训）会把 `weight_path=None` 传进去 → `torch.load(None)` 崩。**04c / 04b 务必设 `lora_weight_path`**；从零训用 `04_train.sh`（官方 `train.py`），但跑前先 `grep` 确认 `sd2.py` 对缺 `weight_path` 的处理，否则同样崩。

**定期自检**（确认暖启动仍生效）：
```bash
grep -n "weight_path\|torch.load" $HYPIR_DIR/HYPIR/trainer/sd2.py
# 应见 init_generator 里那两行 torch.load(self.config.weight_path)；没有 = 暖启动已坏
```

## 可能遇到的问题

公司代理做 HTTPS 中间人解密，下面按流水线阶段列出常见报错与修法（命令在服务器上、conda 环境已激活时执行）。

**1. clone/pull 本仓或官方仓报错**
- 报 `SSL certificate problem` / 认证：公开仓免认证，加 `-c http.sslVerify=false`（`run_all.sh` 已对官方仓做 `sslVerify=false` 兜底）。
- 报 `Failed to connect to github.com port 443`（连不上，**非 SSL**）：git 没走代理。设全局代理（带认证的把用户名密码写进 URL，**密码特殊字符必须 URL 编码**，否则 git 解析错/连不上）：
  ```bash
  git config --global https.proxy http://USER:PASS@proxyhk.huawei.com:8080
  git config --global http.proxy  http://USER:PASS@proxyhk.huawei.com:8080   # 顺手也设 http
  # 取消：git config --global --unset https.proxy ; git config --global --unset http.proxy
  ```
  ⚠️ 密码特殊字符 URL 编码：`*`→`%2A`、`+`→`%2B`、`@`→`%40`、`:`→`%3A`、`#`→`%23`、`&`→`%26`、`=`→`%3D`。例：密码 `p*ss+word` 写成 `p%2Ass%2Bword`。
  > 这和 `_env.sh` 里的 `http_proxy`/`https_proxy` 环境变量是**两套**：环境变量给 curl/hf/pip 用，`git config http.proxy` 给 git 本身用；两个都设最稳。
- 报 `Failed to connect to proxyhk.huawei.com port 8080: No route to host`（代理都连不上 → git/pip/hf 全挂）：根因是 **docker 网桥网段和代理 IP 冲突**——`docker-compose` 建 network 时分的子网（如 `172.18.0.0/16`，网桥 `br-407a71493298`）把 `proxyhk.huawei.com` 解析到的 `172.18.100.92` 包进去了，内核把去代理的流量送进 docker 网桥而非物理网卡。排查 + 修：
  ```bash
  # 1) 查代理 IP + 看它落进哪个网桥网段（命中即冲突）：
  getent hosts proxyhk.huawei.com        # 例: 172.18.100.92
  ip route | grep 172.18                # 命中 172.18.0.0/16 dev br-xxxxx → 冲突
  # 2) 加一条主机路由，把代理 IP 强制走物理网卡（ens1f0 换成你的网卡，网关用默认网关）：
  ip route show default                  # 取默认网关，例 10.x.x.1
  sudo ip route add 172.18.100.92 via 10.x.x.1 dev ens1f0
  # 3) 验证：应能连代理了
  curl -x http://USER:PASS@proxyhk.huawei.com:8080 https://github.com -I   # 200/302 即通
  ```
  持久化（重启不丢）：写到 `/etc/network/if-up.d/` 脚本或 `nmcli`；治本是在 `/etc/docker/daemon.json` 配 `default-address-pools` 给 docker 分不与代理冲突的子网（如 `10.250.0.0/16`），再 `systemctl restart docker`。
  > 同机其他容器/虚拟网桥也可能占 `172.17.0.0/16`、`172.19.0.0/16` 等；只要代理 IP 落进任一 docker 网段就中招。

**2. `pip install -r requirements.txt` 报 `SSL:CERTIFICATE_VERIFY_FAILED` / 超时**
```bash
pip config set global.trusted-host "pypi.org pypi.python.org files.pythonhosted.org download.pytorch.org"
INSTALL_DEPS=1 bash hypir/00_setup_env.sh
```
仍超时（大文件 torch==2.6.0）：加 `--timeout 600 --retries 10`，或先单独装 torch：`pip install --timeout 600 --retries 10 torch==2.6.0` 再重跑。不想动现有 torch 就 `SKIP_TORCH=1 INSTALL_DEPS=1 bash hypir/00_setup_env.sh`。

**3. `hf download` 报 `CAS service error : ReqwestMiddleware`**
HF 的 Xet/Rust 通道不认代理。`_env.sh` 已设 `HF_HUB_DISABLE_XET=1`；仍报则彻底卸载：
```bash
pip uninstall -y hf_xet
bash hypir/01_download_models.sh
```

**4. `hf download` 报 `SSLCertVerificationError`**
代理根 CA 不在系统证书包。先一次性建包，`_env.sh` 会自动用 `~/.ca-bundle.crt`：
```bash
bash hypir/setup_ca_bundle.sh    # 抓代理证书链 -> ~/.ca-bundle.crt，并自检
bash hypir/01_download_models.sh
```
- 自检 `[OK]` → 直接重跑 `01`。
- 自检 `[FAIL]` → 把公司根 CA 追加到 `~/.ca-bundle.crt` 后重跑（公司根 CA 常见于 `/usr/local/share/ca-certificates/`，脚本已自动并入）。
- 仍报 SSL（CDN 端点用了不同的 MITM 证书）→ `01` 自动回退到禁用 SSL 校验的下载器（`_hf_download.py`）；或直接 `HF_DISABLE_SSL=1 bash hypir/01_download_models.sh`。

**5. `hf download` 报 `repository not found for url .../stable-diffusion-2-1-base`**
原始 `stabilityai/stable-diffusion-2-1-base` 已从 HuggingFace 下架。脚本默认改用公开镜像 `Manojb/stable-diffusion-2-1-base`（完整 diffusers 格式，非 gated，无需 token）。若你仍指向旧的 `stabilityai/...`，会报此错；改成默认源即可：
```bash
unset HF_BASE_REPO   # 用默认 Manojb/stable-diffusion-2-1-base
bash hypir/01_download_models.sh
```
若你换用的镜像确为 gated 仓库（HF 对未认证账号返回 "not found" 实为 401），则需：1) 在该仓库页面接受许可证；2) 建 read token；3) `HF_TOKEN=<token> bash hypir/01_download_models.sh`（脚本会自动把 token 透传给 `hf download` 和 SSL 兜底下载器）。

**6. 推理 找不到diffusion_pytorch_model.bin**
是HYPIR/HYPIR/enhancer/sd2.py里强制要求bin格式，但下载到模型是safetensor格式，修改use_safetensors=True即可。

**7. 推理 OOM（显存不足）**
降 `PATCH_SIZE`（512→256，并相应降 `STRIDE` 到 128），或降 `UPSCALE`。免费 T4 可跑默认 512 patch（见官方 colab）。仍紧张时设 `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

**8. 训练时 `open_clip` / `lpips` 下载权重失败**
- 判别器 `ImageOpenCLIPConvNext` 在初始化时下载 `convnext_xxlarge`（laion2b_s34b_b82k_augreg_soup，open_clip 走 HuggingFace）。`_env.sh` 的 CA bundle + `HF_HUB_DISABLE_XET` 通常能覆盖；仍失败时手动放到 open_clip 缓存或 `HF_DISABLE_SSL=1` 后重试。
- **首次成功下载后，设 `HF_HUB_OFFLINE=1` 强制走本地缓存**（写进 `proxy.env` 或运行命令前），避免每次训练反复下载、代理下反复失败。前提是 `~/.cache/huggingface/` 里已有 convnext 权重（先联网成功跑一次再开离线）。
- `lpips.LPIPS(net="vgg")` 从作者 URL 下载 VGG 权重（走 `torch.hub`）。代理下若失败：先 `bash hypir/setup_ca_bundle.sh`，或预先把 `vgg.pth` 放进 `~/.cache/torch/hub/checkpoints/`。

**9. 训练报 `assert image.height == self.out_size`**
你的 GT 不是 512×512 且 `crop_type=none`。要么用 `03_build_dataset.sh` 的 `CROP=1` 预切 512 patch，要么 `CROP_TYPE=random bash hypir/04_train.sh`。

**10. 训练多卡 `accelerate launch` 只用一卡**
没配 accelerate 多进程。先 `accelerate config`（选 multi-GPU），再 `N_TRAIN_GPU=8 bash hypir/04_train.sh`（脚本会加 `--num_processes`）。单卡可忽略。

**11. 训练报 `queue_size` 不能被 `batch_size` 整除（合成退化路径 04c/04 才有）**
官方 `sd2_train.yaml` 的 `RealESRGANBatchTransform` 有个训练池 `queue_size=256`，要求是 `batch_size`(每卡) 的倍数。`256%6≠0` → 报错。修法二选一：
- 用 `BATCH_SIZE=8`（`256%8=0`，常用命令 #5 即如此；显存够就这个）；
- 或保 `BATCH_SIZE=6`、改 `queue_size=252`(=6×42)：在 04c/04_train.sh 的 heredoc 加 `cfg.data_config.train.batch_transform.params.queue_size=252`。
> 注：配对路径 04b 用 `PairedFaceBatchTransform`（无 queue_size），不受此限制。

**12. 跑 `.sh` 报 `syntax error near unexpected token ('`（CRLF 行尾）**
脚本被 CRLF 行尾污染（Windows→服务器用 scp/rsync/zip/复制等非 git 方式同步时带过去）。bash 遇到每行末尾的 `\r` 把引号上下文搞坏，于是 `echo "...(Baidu manual step)..."` 里的 `(` 被当成未引用的子 shell 起始符 → 解析报错（常报在某条 echo 行，如 `03d` 第 105 行）。本仓 `.gitattributes` 强制 LF，但只有 `git checkout/pull` 才会在检出时落 LF，非 git 传输方式不会自动转。
```bash
file hypir/03d_build_beauty_dataset.sh           # 出现 "CRLF line terminators" 即中招
grep -c $'\r' hypir/03d_build_beauty_dataset.sh  # 非 0 即有 \r
# 修法（任选）：
sed -i 's/\r$//' hypir/03d_build_beauty_dataset.sh                    # 剥掉 \r
find hypir retouchformer -name '*.sh' -exec sed -i 's/\r$//' {} +    # 一次性修所有 .sh
dos2unix hypir/03d_build_beauty_dataset.sh                           # 有 dos2unix 的话
git checkout -- hypir/03d_build_beauty_dataset.sh                    # git 同步的：.gitattributes 还原 LF
```
`.py` 不受影响（Python 词法器能吃 CRLF），但想统一也可一并 `sed`。预防：用 `git pull` 同步（git 按 `.gitattributes` 落 LF）；用 scp/rsync 的话传完跑一下上面的 `sed`；或 Windows 上 `git config --global core.autocrlf false` 再提交/同步。

**13. 跑 03d 报 `OSError: image file is truncated (N bytes not processed)`**
源夹里有损坏/截断图（下载不完整、传输中断等）。`build_beauty_dataset.py` 已把整张图处理包进 try/except——遇损坏图会打印 `! failed (skipped): ...` 并删掉本图半成品（不中断、不产生半对），汇总行有 `skipped=N` 计数，故**直接忽略即可**，03d 会自动跳过继续。想一次性清掉源夹里的坏图（之后重跑就没的跳了）：
```bash
INPUT_DIR=../HYPIR/dataset/guojia_datas_20260708 \
  python hypir/scan_corrupt_images.py                 # 只列
INPUT_DIR=../HYPIR/dataset/guojia_datas_20260708 DELETE=1 \
  python hypir/scan_corrupt_images.py                 # 列 + 原地删
```
扫描器和 build 脚本用同一套 `Image.open().convert("RGB").load()` 判定，一致；故意没设 `LOAD_TRUNCATED_IMAGES`（要它抛、好检测坏图）。

> 通用：`proxy.env`（代理凭证）在仓内 gitignored，`~/.ca-bundle.crt` 在家目录，都不入库；切勿把凭证写进脚本。

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
- **05 eval**: `EVAL_DIR/{result,compare}/<rel>.png` + `EVAL_DIR/metrics.csv`（逐图 PSNR/SSIM/LPIPS，含 bicubic 基线）。

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
