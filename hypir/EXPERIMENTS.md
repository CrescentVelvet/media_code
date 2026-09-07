# HYPIR — 实验设计与结果归档（EXPERIMENTS）

> 本文件从 README.md 迁入：A/B/C/D 美颜对比实验设计、去红润诊断、04d 扫参实验、训练调参速查。
> 运行命令与参数速查看 [README.md](README.md)；原理与排错看 [NOTES.md](NOTES.md)。
> 新实验结论按日期倒序追加到本文件头部。

## 美颜增强 A/B/C/D 对比实验（03d/03e → 04b）

### 实验动机：为什么「长痘变丑」

现有 03c/04c 在线退化（LQ=高斯模糊, HQ=原图）虽能复原模糊，但会「**长痘变丑**」——模型过度增强、凭空 invent 皮肤瑕疵。把 HQ 目标换成 RetouchFormer 的美颜版（已去瑕疵+磨皮保结构）而 LQ 保持同样的模糊，模型仍学去模糊（增强不失）但目标变成干净光滑皮肤，故不再 invent 瑕疵。

### 实验设计（单变量：只 HQ 目标不同，LQ 共用）

用 [RetouchFormer](../retouchformer/README.md) 对每张美颜，每图同时保存多张像素级对齐的 512×512 PNG（同源 src 张量派生，任意尺寸/宽高比输入都安全——模型 VRT 写死 512×512，非方形会被 CenterCrop，`hq_orig`/`lq_gauss` 存的正是这个 crop）：

| 组 | parquet | HQ 目标 | 预期 |
|---|---|---|---|
| **A 基线** | `rest.parquet` | `hq_orig`（原图） | 复原模糊但会「长痘变丑」（= 现有 03c 风格） |
| **B 美颜** | `rest_beauty.parquet` | `hq_beauty`（美颜 1 pass） | 修掉长痘、又不毁脸 |
| **C 加强美颜** | `rest_beauty_strong.parquet` | `hq_beauty_strong`（迭代 `BEAUTY_PASSES` 次） | 美颜最强，但可能过磨失结构（需对比 B 找甜点） |
| **D 去红润美颜** | `rest_beauty_decolor.parquet` | `hq_beauty_decolor`（wavelet 融合：美颜高频+原图低频） | 红润减弱、磨皮保留（修掉 B 的红润副作用） |

关键设计点：
- **A/B/C/D 共用同一 `lq_gauss`**（03d 产的），单变量对比只 HQ 目标不同。D 组由 03e **独立**产出自己的 `lq_gauss`，但同 `BLUR_SEED`=231 + 同图顺序 → 与 03d 逐像素一致，仍是干净的单变量对比（对比「HQ 目标颜色」对红润的影响）。
- C 组的 `rest_beauty_strong.parquet` 仅在 `BEAUTY_PASSES>=2` 时产出（一次 `=2` 会同时产出 A/B/C 三套，A/B 产物不变、向后兼容）。
- 各喂 04b 训一个（`OUTPUT_DIR` 分开，默认暖启动 `HYPIR_sd2.pth`），训完用 `05_eval.sh` 算指标 + `02_run_inference.sh` 各跑一组测试图肉眼对比（如 results/beauty_A~D）。
- 迭代美颜原理（C 组）：RetouchFormer 输入输出都是 `[1,3,512,512]` in `[-1,1]`，把上一轮输出 `clamp(-1,1)` 后喂回去即叠加一次美颜。pass 1 永远存 `hq_beauty`；pass 2..N 的最终结果存 `hq_beauty_strong`（中间 pass 不留）。`BEAUTY_PASSES>2` 继续叠加但收益递减，**2 通常是甜点**。

### D 组去红润：红润到底在哪（机理分析）

RetouchFormer 的红润主要在 **beauty 的高频**（per-channel DC 偏移——美颜把皮肤推向"健康暖色"统计，写进高频），不在低频：
- `DECOLOR_MODE=wavelet`（默认，只换低频磨皮）：**去不掉高频红**，作对照。
- `DECOLOR_MODE=high_freq_dc`：减掉 `beauty_high` 的 per-channel 空间 DC 再加 `orig_low`，保留磨皮结构（高频纹理）只剥掉红色调常数项。红润在 beauty 高频时必须用这个。
- 若 `high_freq_dc` 仍去不掉（红是局部非全局 DC），升级全局色调校正。

`hq_orig`/`hq_beauty` 在 03e 里是内存中间量不存盘；产出 `hq_beauty_decolor/` + `lq_gauss/` + `rest_beauty_decolor.parquet`。RetouchFormer 加载/transform/blur 逻辑 copy 自 `build_beauty_dataset.py`（并列脚本，重复正常）；wavelet 三函数 verbatim copy 自 `HYPIR/utils/common.py:32-80`（仅依赖 torch）。**训练/推理脚本均不改**：HQ 目标去红润，LoRA 学到的还原也去红润——换 `PARQUET_PATH`/`WEIGHT_PATH` 即可。

### 去红润诊断（`decolor_probe.py`）

一次性诊断脚本，跑一张图量化红润来源、对比各去红方案，用于决定 03e 该用哪个 `DECOLOR_MODE`。输出：
- beauty vs src 的 per-channel mean + ΔR-B（量化红润）
- `beauty_high` 的 per-channel DC（判断红是全局 DC 还是局部）
- 方案 C（attention mask 像素融合：`mask·beauty + (1-mask)·src`，mask 取自 RetouchFormer forward 第 164-166 行返回的 `attention_list`，降维用 channel-max、取最细尺度 resize 到 512）的 soft/hard 变体 ΔR-B
- 一张 `[src|beauty|mask|C_soft|C_hard|high_freq_dc]` 横拼对比图

**看三个数决策**：`>0.5 blemish area (max)` 小=mask 方向对、≈1=反了用 `1-mask`；`C_soft ΔR-B` 接近 src(0)=方案 C 去红成功→整合进 03e 加 `DECOLOR_MODE=attention_mask`；仍红=瑕疵区也带红→方案 C + 色调校正组合。

需在 `hypir` env 跑（脚本 import HYPIR SD2Enhancer 做 LoRA 推理；`retouchformer` env 无 diffusers 会崩）：
```bash
conda activate hypir
GPU=0 INPUT_DIR=../HYPIR/input/test_faces_hq \
OUTPUT_DIR=../../output/hypir_test_results/probe \
python hypir/decolor_probe.py

GPU=0 LORA_PATHS=../HYPIR/experiments/beauty_ppr50k_20260721/checkpoint-1000/ema_state_dict.pth,../HYPIR/experiments/beauty_strong_20260814/checkpoint-1000/ema_state_dict.pth,../HYPIR/experiments/beauty_decolor_20260828/checkpoint-1000/ema_state_dict.pth \
N_COLS=5 \
INPUT_DIR=../../output/hypir_test_results/input \
OUTPUT_DIR=../../output/hypir_test_results/probe \
python hypir/decolor_probe.py
```

## 04d 并行扫参实验（LR / loss 权重 / 真实退化配对）

**背景**：并行训练（04d）时发现「越练越模糊」是 **L2 坍缩**；用 04d 多卡各占一卡并行扫 `LR_G` + loss 权重 + 真实退化配对找最佳。04d 默认 steps=30000 / ckpt_every=100 / LR_D=LR_G / 后台；**逐 ckpt 评找峰值，常在早期**。

典型扫参布局（各实验一卡）：

```bash
# 复原+美颜：多个 LR 并行（各占一卡）：
GPU=0 PARQUET_PATH=../HYPIR/dataset/beauty_guojia_datas_20260708/rest_beauty.parquet bash hypir/04d_train_sweep.sh 5e-6
GPU=1 PARQUET_PATH=../HYPIR/dataset/beauty_guojia_datas_20260708/rest_beauty.parquet bash hypir/04d_train_sweep.sh 2e-6
GPU=2 PARQUET_PATH=../HYPIR/dataset/beauty_guojia_datas_20260708/rest_beauty.parquet bash hypir/04d_train_sweep.sh 1e-5
GPU=3 PARQUET_PATH=../HYPIR/dataset/beauty_guojia_datas_20260708/rest_beauty.parquet LR_D=1e-5 SWEEP_TAG=disc1e5 bash hypir/04d_train_sweep.sh 5e-6
# 加强美颜(C 组)：把 PARQUET_PATH 换成 rest_beauty_strong.parquet 即可对 C 组扫同样 LR/loss：
GPU=0 PARQUET_PATH=../HYPIR/dataset/beauty_guojia_datas_20260708/rest_beauty_strong.parquet SWEEP_TAG=strong_5e6 bash hypir/04d_train_sweep.sh 5e-6
GPU=1 PARQUET_PATH=../HYPIR/dataset/beauty_guojia_datas_20260708/rest_beauty_strong.parquet SWEEP_TAG=strong_2e6 bash hypir/04d_train_sweep.sh 2e-6
# 真实退化配对(03b 的 360p 相机 LQ + RAW HQ——发布模型擅长的配方，纯高斯模糊 LQ 之外的对照)：
#   先建 parquet(03b)，再扫 LR：
# HQ_DIR=../HYPIR/dataset/ppr10k_faces_20260703/hq LQ_DIR=../HYPIR/dataset/ppr10k_faces_20260703/lq bash hypir/03b_build_paired_dataset.sh   # -> .../ppr10k_faces_20260703/hypir_paired.parquet
# GPU=4 PARQUET_PATH=../HYPIR/dataset/ppr10k_faces_20260703/hypir_paired.parquet bash hypir/04d_train_sweep.sh 5e-6
# GPU=5 PARQUET_PATH=../HYPIR/dataset/ppr10k_faces_20260703/hypir_paired.parquet bash hypir/04d_train_sweep.sh 2e-6
# loss 权重调节(抗 L2 坍缩：up GAN、down L2；LAMBDA_GAN/LAMBDA_LPIPS/LAMBDA_L2 透传 04b，SWEEP_TAG 标注实验名)：
# GPU=6 PARQUET_PATH=../HYPIR/dataset/beauty_guojia_datas_20260708/rest_beauty.parquet LAMBDA_GAN=2 LAMBDA_L2=0.5 SWEEP_TAG=gan2_l2p5 bash hypir/04d_train_sweep.sh 5e-6
# GPU=7 PARQUET_PATH=../HYPIR/dataset/beauty_guojia_datas_20260708/rest_beauty.parquet LAMBDA_GAN=1 LAMBDA_LPIPS=10 SWEEP_TAG=gan1_lp10 bash hypir/04d_train_sweep.sh 5e-6
# GPU=8 PARQUET_PATH=../HYPIR/dataset/beauty_guojia_datas_20260708/rest_beauty.parquet LAMBDA_L2=0.3 SWEEP_TAG=l2p3 bash hypir/04d_train_sweep.sh 5e-6
```

## 训练调参速查（04b/04c 按需改）

- 显存不够：`BATCH_SIZE=4`，或 `GRAD_ACCUM=2`（等效翻倍 batch）。
- 想更稳、别把发布模型「练歪」：`LR_G=5e-6 LR_D=5e-6`。
- 想多练/少练：`MAX_TRAIN_STEPS=30000`（更多）或 `10000`（更快）。
- 多卡：先跑一次 `accelerate config` 选 multi-GPU，再 `N_TRAIN_GPU=8 bash hypir/04b_train_paired.sh`（此时**不要**设 `GPU=`）。
- 跑报错了：看 [NOTES.md](NOTES.md)「可能遇到的问题」对应条目。

## 四条训练路径 LQ/HQ 取舍对比

| 路径 | LQ | HQ |
|---|---|---|
| 04b 真实配对 | 真实退化(360p 相机) | 原图 |
| 03c/04c 在线退化 | 每 epoch 重随机高斯模糊 | 原图 |
| 03d 离线退化 | 固定高斯模糊（每图一个 seeded 实现） | 原图**或**美颜版（1 pass = B，迭代 N pass = C） |
| 03e 去红润 | 独立产 lq_gauss（同 seed 与 03d 逐像素一致） | wavelet 融合去红润美颜（D） |

> NB：03d 的 `lq_gauss` 是离线固定模糊，不像 03c/04c 每 epoch 在线重随机——故 A 是「略少增强的 03c 基线」，但 A vs B vs C 是干净的单变量实验（只 HQ 目标不同）。模糊作用于 raw 对齐 crop（非 `USM(orig)`），与 03c 的 `LQ=blur(USM(orig))` 略有偏差；但 A/B/C 共用同一 `lq_gauss`，对比仍是单变量。
