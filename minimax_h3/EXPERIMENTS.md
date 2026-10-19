# EXPERIMENTS.md — MiniMax-H3 复现实验与排障结论

> 分工：`README.md` / `README_wsl.md` = 怎么跑；`NOTES.md` = 原理与排错手册；本文件 = 实验结论与数据。
> 按日期**倒序**追加，新结论写在最上面。

## 0. 基线环境与对照

| 项 | 本机（WSL2） | 服务器（对照） |
|---|---|---|
| GPU | RTX 3090 24 GB（sm_86，驱动 595.95） | 单卡 80 GB |
| 内存 | 56 GB RAM + 128 GB swap | 远大于本机 |
| 系统 | WSL2 Ubuntu 24.04 | 原生 Linux |
| diffusers | **0.40.0** | **0.40.0.dev0** |
| transformers | 5.15.1 | 5.12.1 |
| torch | 2.6.0+cu124 | 2.9.1 |
| torchao | 0.16.0 | 0.15.0 |
| 权重 | `/mnt/d/wheel/minimaxh3_ms`（ModelScope 下载） | HF 官方 |
| 脚本 | 同一套 `06x` diffusers 脚本 | 同一套 |

**服务器结论（峰哥实测）**：全量 bf16 与 int8 量化两支**都正常**，无马赛克。
本机同脚本同权重**全出马赛克**。这条对照是整轮排查的起点——它把「模型 / pipeline 代码 /
量化 / LoRA / scheduler」一次性排除，只剩「本机这份权重」和「本机环境」两个变量。

---

## 1. 结论速查

| 项 | 结论 | 关键证据 |
|---|---|---|
| 本机复现是否成功 | ✅ 成功（2026-10-08 起） | 09b 同参复跑，画面正常（§6） |
| 马赛克根因 | ✅ **权重损坏**：`transformer/00001–00005` 五片 | sha256 与 HF 不符；替换后正常（§5、§3） |
| 与代码/量化/offload 的关系 | ✅ 全部无关 | §4 排除矩阵 |
| 24 GB 单卡可用配方 | int8 weight-only + group offload（06c/06d/09b） | 峰值 VRAM 15.4 GB（09b 实测） |
| bf16 全量单卡 | ❌ 不可行（权重 ≈124 GB vs RAM 56 GB） | 06b 靠 ComponentsManager auto offload 才在 80 GB 上用 |
| Turbo 4 步 LoRA | ✅ 可用；768p 配方必须 `VIDEO_SHIFT=6` | orbit 首跑 4 步出片（§7） |
| 人物静止 + 相机 360° | ⚠️ 能转一圈并闭环，但 v1 prompt 出现「钟摆半圈往返」 | §7、§2 |
| Ref2VA（真多参考图合成） | ❌ 未复现：本地无 `transformer_ref` 权重 | `ls /mnt/d/wheel/minimaxh3_ms/` |
| V2V / 局部重绘 | ❌ 不存在：官方只发布 t2va / fl2va / ref2va 三种 workflow | `diffusers/modular_pipelines/minimax_h3/` |

---

## 2. 2026-10-09 环绕 prompt v2（**未真机验证**）

峰哥实测 v1 产物：相机转半圈后**原路返回**（钟摆轨迹），不是整圈。

判读与改动：
1. FL2VA 的「首尾帧同图」约束下，模型倾向以**最短路径**闭合 —— 转到背面再倒回来，
   比「继续向前走完整圈」更容易满足首尾一致。v1 结尾措辞 `settles back at the starting
   front view`（回到起点）在语义上鼓励了往返。
2. v2 加**单向单调方位角**硬约束（`always rotating in the same direction` /
   `orbital angle increases monotonically` / `never reverses`），结尾改为「经正后方继续前进，
   完成整圈后**到达**起点」。
3. 背景：v1 的 `photography studio` 被**字面执行**（转侧面/背面时露出柔光箱与幕布边缘），
   v2 改为 `seamless infinite light-gray cyclorama, completely empty`。

状态：**只做了语法与 DRY_RUN 校验，尚未真机跑**。若 v2 仍钟摆，结构性兜底方案是分两段生成
（正面→背面 / 背面→正面），用一帧背面图作中间锚点。

---

## 3. 2026-10-08 修复后终验（✅ 通过）

### E-06a `09g_realscale_probe.py`（64×64 探针，快速判活）

配置：真权重、int8 + leaf-level group offload、`W=H=64`、`NUM_FRAMES=124`、
`NIS=51`（50 步去噪）、`shift=12`、`ARMS=default`。加载 ≈18–20 min，去噪 ≈3 min。

| 状态 | z std | 帧间相关（皮尔逊） | 判读 |
|---|---|---|---|
| 修复前（坏权重） | — | 脚本打印 0.038（**该数值来自 09d，已作废，见 §9**） | 无明显结构 |
| 修复后 | **3.7596**（该 .pt 无 `latents_std` 元数据，无法算比值） | **+0.5310** | 有结构，但 64×64 只有 4×4 latent 格 |

⚠️ 64×64 远低于训练分辨率（768p 预算），latent 空间格只有 4×4，帧间一致性天然偏低。
**这个尺寸只能判「活/死」，不能判画质** —— 必须回正常分辨率复核。

### E-06b `09b_diag_baseline.py`（全分辨率终审，同参对照）

配置：`960×544 / 124 帧 / 50 步（NIS=51）/ seed=42 / t2va 狐狸 prompt`，
int8 + `enable_group_offload(offload_type="leaf_level")`，VAE 也 offload，CPU generator。
**与当初产出马赛克的那次 run 逐参数一致**，是干净对照。

| 指标 | 坏权重那轮 | 修复后（本次） | 基准 |
|---|---|---|---|
| 生成耗时 | ≈37 min | **71.1 min**（与其它负载并发） | — |
| 峰值 VRAM | — | **15.4 GB** | 24 GB 卡 |
| raw latent `z` std | 3.4714 | **2.2585** | `RMS(latents_std)=1.9250` |
| std / 基准 | 1.80× | **1.17×** | 分布内 |
| 与 seed42 重放初始噪声的残差相关 | +0.44 | **+0.0853** | 收敛应 <0.1 |
| 帧间相关（皮尔逊） | +0.796 | **+0.8066** | ⚠️ 不能区分，见 §9 |
| 肉眼 | 细密 RGB 雪花（电视雪花） | **红狐狸在积雪松林行走，六帧步态连贯** | 通过 |

产物：`/mnt/d/output/minimaxh3_rotate_results/diag_baseline/baseline_t2va_seed42.mp4`
（960×544 / 124 帧 / 24 fps / 5.17 s / AAC 双声道）。
同目录 `latent_decode1.pt`（7.2 MB）**已被本轮覆盖**，现存的是修复后版本；
坏时代那份已不存在，只能引用日志里的统计数字。

**判读**：z std 从 1.80× 回到 1.17×、残差噪声相关从 0.44 掉到 0.085、画面由雪花变为正常视频
—— 三项独立证据同向，修复坐实。

---

## 4. 2026-09-29 排除实验批（定位过程中的关键否定）

目的：服务器同脚本同权重干净 → 逐个证伪本机侧的每一层嫌疑。**全部为实测，非推理。**

| 嫌疑 | 实验 | 结论 |
|---|---|---|
| offload 机制 | 官方 tiny fixture（0.16 MB）四配方 A/B：常驻 / `block_level` / `block_level+use_stream` / `ComponentsManager` auto offload | `z` 与 `x` 的 **max\|Δ\| = 0.0000e+00，cos = 1.000000**；onload 时刻 block 权重与 CPU 母本差 **0.000e+00（0/12 不一致）** |
| torchao int8 数值 | 512×256 Linear，per-channel 尺度 4× 差异；手写 int8 / torchao v2 / v1 对比 | v2/v1 相对误差 **5.8e-3**，逐通道 0.0039–0.0086，**0/256 通道 >5%** |
| C++ 扩展缺失 | `torch.ops.torchao._weight_int8pack_mm` 不存在、`intmm` 可导入 | 只是走后备 kernel，**数值正确**；"Skipping import of cpp extensions" 警告无害 |
| scheduler 约定 | 不加载权重，直读 `MiniMaxH3Scheduler` | sigma 网格是 **shift 变换后**的（非 `linspace(1,0,N)`）；`set_timesteps(51)` → 51 sigmas / 50 timesteps，尾部 `0.4337→0.3333→0.1967→0.0000`；快照显式 `shift=12.0`，类默认也是 12.0 → **无漂移** |
| CFG / guider | 读 `denoise.py:55-56` | H3 是 **guidance-distilled：无 unconditional pass、无 guider**，CFG 类 bug 不可能存在 |
| text encoder（09f） | 真权重 int8 + leaf offload 跑 hidden states，再离线复核 | 4 层 × 3 prompt 的 NaN/Inf 全 0；hs[50] 与 hs[-1] 相对差 91–239（**没取错层**）；token 未塌缩；剔除 3 个 massive 通道后 prompt 可分（cos 0.17 / 0.44 / 0.81）→ **洗清** |
| VAE | 09a roundtrip（encode→decode，不跑去噪） | ✅ 完美，`raw` 版与 `norm` 版行为符合预期 → 归一化约定正确 |
| LoRA alpha | 对外部官方 `inference_minimax_h3.py` 定版 | `DEFAULT_LORA_ALPHA=8`，仅 `4step_v1.0_768p` 显式 128；**历史出噪那几次 alpha 全是对的**，与马赛克无关 |
| 配置漂移 | 本地 26 个 json（scheduler/transformer/vae/text_encoder/tokenizer/index）与 hf-mirror 逐字节比对 | **全一致**（唯一 DIFF = 自造的 `modular_model_index.json` 缺尾行换行，无害；该文件本身是 `model_index.json` 原样复制，2936 字节） |

### 4.1 排除矩阵（出片形态 × 配方）

| 产物 | 脚本/配方 | 量化 | LoRA | offload 路径 | 分辨率 | 形态 |
|---|---|---|---|---|---|---|
| `results_int8/diag_int8_12step` | 06c，12 步 | int8 | 无 | 手动 block + **VAE 常驻** | 480×256 | 粗块马赛克 |
| `results_int8turbo/t2va_int8turbo_seed42` | 06d，4 步 | int8 | Turbo v1.0_768p | 手动 block | 1344×736 | 细密雪花 |
| `results_int8turbo/rotate_360` | 06d，4 步 FL2VA | int8 | Turbo | 手动 block | 864×1152 | 细密雪花 |
| `results_turbo/rotate_360_turbo_544p` | 06b，4 步 | **bf16 无量化** | Turbo v0.1 | **ComponentsManager auto** | 608×832 | 细密雪花 |
| `diag_baseline/baseline_t2va_seed42`（坏） | 09b，50 步 | int8 | 无 | leaf-level | 960×544 | 细密雪花 |
| `diag_vae/vae_roundtrip_*` | 09a，纯 VAE | — | — | — | 960×544 | ✅ 完美 |
| `diag_baseline/…`（修复后） | 09b 同上 | int8 | 无 | leaf-level | 960×544 | ✅ 正常 |

一次排除：**int8**（06b 是 bf16 也坏）、**LoRA**（06c/09b 无 LoRA 也坏）、
**VAE**（09a 干净、09c 用常驻 VAE 解码同一 latent 也救不回）、
**offload 实现**（CM auto 与手动 block 都坏）、
**分辨率**（480×256 到 1344×736 全坏，排除 tiling / 长序列专属问题）。

---

## 5. 2026-10-08 定位：32 个权重分片全量 sha256

方法：以 HF LFS `oid` 为期望值，逐片 `sha256sum`，脚本**打印实际读取字节数**
（`hash_local2.sh` / `verify_transformer_5.sh`，都在 `%TEMP%\bmh3_diag\`）。

**结果：14 个 transformer 分片里前 5 片损坏（连续前缀，合计 ≈23.7 GB）**

| 分片 | 本地 sha256（前 8 位） | HF 期望（前 8 位） | 判定 |
|---|---|---|---|
| `transformer/…-00001-of-00014` | `5aa15fb0` | `2d847200` | ❌ 坏 |
| `…-00002-of-00014` | `2e7def5f` | `2c4d362e` | ❌ 坏 |
| `…-00003-of-00014` | `6e7f8409` | `949c5aaf` | ❌ 坏 |
| `…-00004-of-00014` | `a0dae3c0` | `eef76167` | ❌ 坏 |
| `…-00005-of-00014` | `ae3027c9` | `43fdf42d` | ❌ 坏 |
| `…-00012-of-00014` | `48fa2bd8` | `48fa2bd8` | ✅ 干净（曾误报，见 §9.4） |
| 其余 26 片（transformer 6–14 / text_encoder 全 14 / vae 3 / audio_vae 1） | — | — | ✅ 一致 |

**坏片的形态（决定它为什么难查）**
- 文件**大小精确正确**（00003 = 4,933,368,192 B）
- safetensors 头**完全合法**（4984 字节头、43 个张量、payload 偏移与文件大小严丝合缝）
- 加载**不报任何错**，只是张量数值错 → 去噪直接产出垃圾

00003 用三种独立读法在内存空闲下复现（`dd iflag=direct` 绕页缓存 / `dd` 普通 /
Python 分块逐块释放），三次哈希一致 `6e7f8409…` —— 排除「内存压力造成的测量假象」。

成因推断：坏片是**连续前缀**，符合「下载工具断点续传时看文件大小够就跳过，
而这几片在首次下载（疑似断电）中已写坏」的形态。**未实锤。**

修复：峰哥用迅雷重下 5 片放回原位（mtime 13:32–14:09），逐片复验 sha256 全与上游一致。

---

## 6. 2026-08-31 ~ 09-02 复现阶段（坏权重时期）

| 日期 | 实验 | 配置 | 结果 |
|---|---|---|---|
| 08-31 | E-01 三配方复现 | 06c / 06d / 06b（见 §4.1） | 全部马赛克；**bf16 也坏**这一点首次把量化排除 |
| 09-01 | E-02 VAE 自检（09a） | 合成图案 encode→decode；norm / raw 两版 | ✅ 完美；确认 `vae.decode` 期望 **raw** latent，归一化只应做一次 |
| 09-02 | E-03 官方消费卡配方（09b） | int8 + leaf offload + CPU generator + 50 步 | 仍马赛克；`z std=3.4714`（1.80×）；各通道 DC z 分数 +81…+519；与初始噪声残差 +0.44 → **去噪未收敛** |
| 09-02 | E-04 09e（ComponentsManager 配方对照） | 只把 offload 路径换成 CM auto | 无效；**已标注作废**（配方 A/B 混搭错误），保留供考古 |

E-03 的价值在于确立了之后一直沿用的判据（std 比值 + 残差噪声相关），
但**当时无法指认凶手**：0.44 的残差相关不能区分「权重坏」与「数值坏」，
只能证明「模型的速度场没在干活」。

---

## 7. 2026-10-08 首条「静止人物 + 360° 环绕」真机验证（✅ 通过，有瑕疵）

脚本 `10_orbit_datagen.py`（FL2VA 首尾同图闭合 + 枚举式静止 prompt；turbo 4 步）。

| 项 | 值 |
|---|---|
| 输入 | `examples/681533632532078.jpg`（实为**三人餐桌场景**，比预期的单人肖像更难） |
| 输出 | 864×1152 竖幅 / 124 帧 / 24 fps / 5.17 s / AAC 双声道 |
| 配方 | turbo 4-step v1.0_768p，`VIDEO_SHIFT=6`，`LORA_ALPHA=auto→128`，闭环 ON |
| 耗时 | 加载 22.6 min（D 盘 HDD）+ 生成 1558 s（26 min，**与另一训练并发被拖慢**；独占预计 8–10 min） |
| `loop_err`（首尾灰度差） | 5.31 —— 首尾视角基本重合 |
| `motion_index`（中心区逐帧差） | 23.04 —— **无校准基线**，不能单独当判据（§9.3） |
| 画面 | f000→f123 完整绕场约 360°（正→右→背→左→正）；三人姿势全程锁定 |

瑕疵：转到侧面/背面时露出**摄影棚柔光箱与幕布边缘**（prompt 写了 `photography studio`，
模型就字面给了摄影棚）。已在 v2 修正（§2）。

---

## 8. 待验证 / 未复现

| 项 | 状态 | 说明 |
|---|---|---|
| orbit prompt v2 | 待真机 | §2；判据：是否整圈单向 + 背景是否干净 |
| Ref2VA 真多参考 | 未复现 | 缺 `transformer_ref` 权重（未下载，约 20 GB+）；当前多图 = 每图各出一条 |
| 06 / 06a（两卡分拆、单卡常驻 int8） | 未在本机跑 | 3090 单卡无意义；属服务器路径 |
| 07 FlashVSR 超分 | 未跑 | 依赖 Block-Sparse-Attention + diffsynth，未安装 |
| 08 Context-IR（hosted API 转长 prompt） | 未跑 | 需 MiniMax API token |
| 544p / 768p 画质对比 | 未做 | 只跑了 544p（09b）与 768p 预算（orbit） |
| shift 消融（6 / 12） | 未做 | turbo 768p 用 6、基座用 12，未交叉验证 |

---

## 9. ⚠️ 指标口径勘误（两次误判的教训）

**背景**：定位过程中我两次凭单一指标下过错误结论，都被后续实测推翻。数字本身没算错，
**错在指标不具备判别力**。

### 9.1 「帧间相关」不能区分好坏

| 状态 | 帧间相关（皮尔逊） |
|---|---|
| 坏权重（当时对 `latent_decode1.pt` 复核） | **+0.796** |
| 修复后（2026-10-19 复测现存文件） | **+0.8066** |

两者几乎相同。原因：坏 latent 有强空间结构和逐通道 DC 偏移，帧间当然相关；
真正坏掉的是「去噪没收敛」，不是「没有结构」。

→ `09d_analyze_latent.py` 曾报「帧间相关 0.038」并据判「零结构噪声」，**是错的**，
该文件已加作废标记；`09g` docstring 里把 0.038 当坏例参照值，**同样不可引用**。
（坏时代那份文件已被覆盖，0.038 的算法细节无法再复现，只能确定它与 0.796 不是同一口径。）

### 9.2 「逐通道 DC z 分数」也不能当判据

修复后的正常 latent：**23/24 个通道 |z|>3**，范围 −551.7 … +320.7。
说明这个统计量对**自然视频 latent 本来就全越界**（视频 latent 的通道均值本就不为零）。

### 9.3 可用的判据（按可靠性排序）

| 判据 | 坏 | 好 | 备注 |
|---|---|---|---|
| `z` std / `RMS(latents_std)=1.925` | 1.80× | **1.17×** | 最稳；需 `latents_std` 元数据 |
| 与同 seed 重放初始噪声的残差相关 | 0.44 | **0.085** | 需知生成时的 generator 契约（CPU + seed） |
| 画面肉眼 | 雪花 | 正常 | 最终裁决，永远要看 |
| 帧间相关 / DC z 分数 / `motion_index` 绝对值 | — | — | **不能单独用**，只能辅助 |

### 9.4 假警报：内存枯竭制造 HASHBAD

内存被并发任务吃到 1.6 GB / 56 GB 时，drvfs 读 `/mnt/d` 报 `Cannot allocate memory`，
`sha256sum` 把短读当 EOF，输出一个**截断流的哈希**，看起来像「文件坏了」。

方法论（已固化进脚本）：
- **「OK」永远可信** —— 要读到完整文件且哈希恰好等于上游期望值，错数据凑不出来。
- **「HASHBAD」必须复验** —— 内存空闲下带「实际读取字节数」重算；字节数不满即读取失败，不是损坏。
- 本次因此先误报 00012、后又漏报 00001/00002/00004/00005（另一份日志把它们读成 OK），
  最终靠内存空闲下的独立复验才定案。

---

## 10. 产物索引

路径前缀 `/mnt/d/output/minimaxh3_rotate_results/`：

| 路径 | 内容 |
|---|---|
| `results_int8/diag_int8_12step.mp4` | 06c 12 步，480×256，粗块马赛克（坏权重时期） |
| `results_int8turbo/t2va_int8turbo_seed42.mp4` | 06d 4 步 t2va，1344×736，雪花 |
| `results_int8turbo/rotate_360.mp4` | 06d 4 步 FL2VA 环绕，864×1152，雪花 |
| `results_turbo/rotate_360_turbo_544p.mp4` | 06b bf16 + CM auto offload，608×832，雪花 |
| `diag_vae/vae_roundtrip_{raw,norm}.mp4` | 09a VAE 自检，✅ 干净 |
| `diag_baseline/baseline_t2va_seed42.mp4` | **09b 修复后终审，✅ 正常**（同目录 `latent_decode1.pt`） |
| `diag_probe/probe_default_64x64.pt` | 09g 探针 latent（修复后） |
| `diag_textenc/textenc_embeds.pt` | 09f text encoder hidden states（5.7 MB） |
| `orbit_datagen/681533632532078_orbit_seed42.mp4` + `_orbit_seed42/`（124 张 jpg） | 首条环绕视频 + 逐帧序列 |
| `orbit_datagen/manifest.jsonl` | 每条产物的 `motion_index` / `loop_err` / 参数 |
| `orbit_datagen/prompt_used.txt` | 实际使用的 prompt（可复查） |
