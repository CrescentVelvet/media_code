# 四种方法：官方训练数据与训练配置明细

> 配套阅读：[训练、精调与评测指南](TRAINING_GUIDE_3DGS.md) ｜ [训练数据集与手动下载清单](TRAINING_DATASETS.md) ｜ [ZipSplat 重训准备手册](ZIPSPLAT_DATA_PREP.md)（环境 / backbone 权重 / 启动 / 数据 / 参数 / 产物 / 续训 / 评测）
>
> 核对日期：2026-11-27。依据论文原文（arXiv HTML 全文）、ZipSplat 官方配置与源码、本仓脚本。**未执行任何官方训练，也未下载训练集。**

## 0. 怎么读这几张表

同一句“官方训练用了什么”其实有四种不同强度的事实，本文强制分列：

| 口径 | 含义 | 典型陷阱 |
|---|---|---|
| **论文报告** | 正文/附录写了数字 | 把消融配置当最终配方；把评测协议当训练采样 |
| **官方代码** | 配置/源码可读到 | 把 252 配置套到 504 发布模型；把配置注释当实测 |
| **可算术展开** | 由公开的 batch×views 直接推出 | 把“带重复的访问量”当“唯一图像数” |
| **未公开** | 论文和代码都没有 | 用总数据规模×猜测帧数反推精确值 |

三条硬边界，贯穿全文：

1. **“数据集官方发布含 Depth/mask”≠“该方法训练读取了它”**。见第 9 节逐项核对。
2. **“场景数/条目数”≠“图像数”**。Table 1 和 SA Table 4 都是条目数，不能相乘当渲染张数。
3. **“16 卡 wall-clock”≠“单卡耗时”**，也不能反推单卡 ETA。

## 1. 总览：七个训练阶段

| # | 模型 / 阶段 | 训练集（论文口径） | steps | 显卡 | wall-clock | 训练 batch |
|---|---|---|---:|---|---|---|
| 1 | **DA3 几何主干**（Giant） | Table 1 的 22 个数据集条目 | 200K | 128×H100 | ≈10 天 | 动态（按 token 守恒），数值未公开 |
| 2 | **DA3 GS-DPT / 前馈 NVS** | DL3DV **10,015** scenes | 200K | 8×A100 | **未公开** | `batch size 1`（per-GPU/global 未说明） |
| 3 | **DA3-Metric** | 逐名 13 个（正文称 14） | 160K iters | **未公开** | **未公开** | 64（per-GPU/global 未说明） |
| 4 | **ZipSplat 252**（论文） | RE10K + DL3DV，各 50% | 450K | 16×GH200 | **未公开** | 每卡 24 samples → 名义 global 384 |
| 5 | **ZipSplat 504**（第二阶段） | 10 个公开数据集，见 3.1 | 100K | **未在配置标注** | **未公开** | 配置 `batch_size: 384`，实际按图像预算动态 |
| 6 | **InfiniSplat v2** Stage I + II | Hypersim / Raw3D / ScanNet++ = 60/10/30 | 25K + 5K | 16×H20 | ≈48 h + ≈13 h | global 16（每卡 1 样本） |
| 7 | **SplatAnything**（属性训练） | 14 个数据集，见 5.1 | 20K iters | 64×H20 | **未公开** | **未公开** |

两处必须点破的常见误传：

- **“DA3 GS-DPT = 50K steps / 64×H100”在 DA3 v1 全文中没有依据**。128×H100、约 10 天属于几何主干；前馈 NVS 对照是 8×A100、200K steps。
- **ZipSplat 的“16 卡 × 24 samples = 384”是论文的样本口径**，不等于当前代码每 step 恒定的真实张数（7.1）。

## 2. DA3：三个阶段不能互相代入

| 项 | 几何主干（DA3-Giant） | GS-DPT / 前馈 NVS | DA3-Metric |
|---|---|---|---|
| 训练数据 | Table 1 的 22 个条目（见 2.1） | **DL3DV 10,015 scenes**，COLMAP 相机 | 13 个逐名数据集（深度/位姿/相机） |
| 真实/仿真 | 混合：合成 + LiDAR 实采 + COLMAP 重建 | **真实**（DL3DV 为实拍视频） | 混合：室内/驾驶/街景，实采为主 |
| 输入 | RGB（+可选相机条件） | RGB（12 个 context） | RGB |
| Depth 来源 | 真实稀疏/噪声深度，经合成数据训练的 teacher 补相对深度，再 RANSAC scale-shift 对齐 | **网络预测**（Gaussian 中心由预测深度反投影），非 GT 输入 | teacher 预测与 metric GT 对齐作标签，20% 概率回退原始 GT |
| Mask | teacher 训练有 sky/object mask loss，**不是** GS-DPT 的输入 mask | **未报告** mask 规则 | 损失含 sky-mask 项 |
| 训练分辨率 | 从 8 档随机采样（2.2） | **270×480** | 基准 504，8 种宽高比 |
| 每样本视角 | 504×504 时 views 在 [2,18] 均匀取 | **12 个 context**（farthest point sampling） | 未按视角数描述 |
| steps | 200K（8K warmup，peak LR 2e-4） | 200K | 160K iters |
| 显卡 | 128×H100 | **8×A100** | 未公开 |
| wall-clock | ≈10 天（Giant） | **未公开** | **未公开** |

其它必须分开的 DA3 配方（都不能当最终 GS-DPT）：

| 配置 | 用途 | 关键数字 |
|---|---|---|
| ViT-L 消融 | 主干规模/视角数消融 | 最多 10 views、32×H100、约 4 天 |
| §7.2.1 特定消融 | 视角尺寸 / batch 扫描 | view size 10、batch 128、120K steps |
| pixelSplat 基线 | 对比方法（epipolar attention 慢） | 100K steps，其余同 8×A100 |

### 2.1 DA3 主干的 22 个数据集条目（Table 1，单位是 scene/对象条目，不是图片）

| 类型 | 数据集 | #Scenes | 数据集 | #Scenes |
|---|---|---:|---|---:|
| 合成 | AriaDigitalTwin | 237 | AriaSyntheticENV | 99,950 |
| 合成 | HyperSim | 344 | MegaSynth | 6,049 |
| 合成 | MvsSynth | 121 | Objaverse | 505,557 |
| 合成 | OmniObject3D | 5,885 | OmniWorld | 1,039 |
| 合成 | PointOdyssey | 44 | Replica VMAP | 17 |
| 合成 | SceneNet RGB-D | 16,866 | TartanAir | 355 |
| 合成 | Trellis | 557,408 | vKITTI2 | 50 |
| LiDAR 实采 | ARKitScenes | 4,388 | ScanNet++ | 230 |
| LiDAR 实采 | WildRGB-D | 23,050 | — | — |
| COLMAP 重建 | BlendedMVS | 503 | CO3Dv2 | 30,616 |
| COLMAP 重建 | DL3DV | **6,379** | MapFree | 921 |
| COLMAP 重建 | MegaDepth | 268 | — | — |

三条不能做的换算：

1. 加总这 22 项没有意义 —— 合成条目按资产/场景计，重建条目按序列计，单位不一致。
2. 这里的 DL3DV 是 **6,379**，而 GS-DPT 用的是 **10,015**，是不同的阶段与筛选口径，**不可互换**。
3. ScanNet++ 在主干训练中做了 scene-level 训练/测试互斥；这不能推广成“所有数据集都按 scene 隔离”。

### 2.2 DA3 主干的分辨率与 batch

- 从 8 档随机采样：`504×504`、`504×378`、`504×336`、`504×280`、`336×504`、`896×504`、`756×504`、`672×504`。
- 仅在 `504×504` 时，views 在 `[2,18]` 均匀取值；其它档的 view 分布未给出。
- batch 随分辨率**动态调整**，目标是每 step token 数近似恒定；**具体数值未公开**，因此不能反推“每步多少张图”。
- 训练 120K steps 后，深度监督由 GT 切换到 teacher labels；pose conditioning 概率 0.2。

## 3. ZipSplat：论文 252 与发布 504 是两套配方

| 项 | 论文 252 模型 | 发布 504 模型（第二阶段） |
|---|---|---|
| 数据集 | RE10K + DL3DV，**各 50%** | **10 个数据集**，见 3.1 |
| 输入分辨率 | **252×252** 方形 | 长边 **504**，7 种宽高比 |
| 每 scene token 高斯数 | 论文正文 **G=32** | **G=16**（从 `zipsplat-g16` 初始化） |
| context 数 | 2 → 24，**前半段线性增长**（约前 225K steps） | **step 0 即全范围 2–24**（`view_schedule: 0`） |
| target 数 | **4 / 样本** | **4 / 样本** |
| steps | **450K** | **100K** |
| 显卡 | **16×GH200**，每卡 24 samples | 配置未标注卡数 |
| 名义 global batch | **384**（16×24） | 配置 `batch_size: 384`，实际动态 |
| 学习率 | 3e-4，5% warmup + cosine，wd 0.05 | **1e-4** |
| 其它 | DA3 backbone lr = 0.1× 基础 lr，**非冻结** | DA3 trunk 开 activation checkpointing，SH dropout 0.5 |
| wall-clock | **未公开** | **未公开** |
| 峰值显存 | **未公开** | **未公开** |

**504 发布模型 = 两阶段串起来**：先按 252 配方训练 **G=16** 模型（RE10K+DL3DV，450K steps），再以 `zipsplat-g16/checkpoint_best.tar` 初始化，跑上表第二列。只下 504 数据、不准备第一阶段 checkpoint，复现不出发布权重。

### 3.1 504 第二阶段的 10 个数据集与抽取权重

| 数据集 / child | 权重 | 数据集 / child | 权重 |
|---|---:|---|---:|
| RE10K | 0.05 | DL3DV | **0.32** |
| ScanNet++ DSLR | 0.10 | ScanNet++ iPhone | 0.08 |
| ARKitScenes 横屏 | 0.07 | ARKitScenes 竖屏 | 0.06 |
| Hypersim | 0.05 | UnrealStereo4K | 0.02 |
| BlendedMVS | 0.02 | CO3D | 0.10 |
| WildRGB-D | 0.08 | G-Objaverse | 0.05 |

合计 1.0。ScanNet++ 两设备、ARKitScenes 两方向是 **child**，所以是“10 个数据集 / 12 个 child”。

**这是抽 scene 的权重，不是图像数、存储量或训练时长占比。** 小数据集通过重复 shard 实现 oversampling，因此不能由 100K steps 反推每个唯一 scene 被访问几次。

### 3.2 视角采样器（按 child 配置）

| 采样器 | 用于 | 规则要点 |
|---|---|---|
| `bounded` | RE10K、DL3DV | 按帧距约束取 context：RE10K `min=35 / gap×45 / max_consecutive=90`；DL3DV `min=8 / init_min=5 / gap×11 / max_consecutive=22`；两者 `max_pose_jump_ratio=10.0` |
| `rank` | ScanNet++（两设备）、ARKitScenes（两方向）、Hypersim、UnrealStereo4K | 随机 reference；按预计算共视排序；较远 context 从 overlap `[0.2,0.8]` 抽，其余 context 与 target 从 overlap `≥0.2` 的池抽；不足时按最共视补齐 |
| `random` | BlendedMVS、CO3D、WildRGB-D、G-Objaverse | 随机取视角 |

补充约束：

- 同一样本内 context 与 target 索引**不重复**，scene 至少要有 `context + 4` 张图。
- `random_reference_view=true` 在训练时会 shuffle context，影响 first-camera reference。
- `rank` 的 target 规则**不等于** SplatAnything 的“大角度/遮挡感知对称共视”策略，不要互相套用。

## 4. InfiniSplat v2：两阶段共用同一数据配方

| 项 | Stage I | Stage II |
|---|---|---|
| 数据混合 | Hypersim **60%** / Raw3D **10%** / ScanNet++ **30%** | 同左 |
| 训练/验证 | 场景不相交 | 同左 |
| 每样本视角 | **4–12 input + 4 target** | 同左 |
| 输入长边 | N≤8 → **1022**；N=9/10 → **756**；N=11/12 → **518** | 同左 |
| 冻结/训练 | 冻结几何分支；训属性 backbone 最后 10 个 Transformer blocks + Gaussian head | 冻结 reconstructor；训 DPT feature projection + learned refiner |
| steps | **25K**（2K warmup） | **5K**（无 warmup） |
| 峰值 LR | 1e-6 / 2e-4 | 2e-4 → cosine 至 0.1× |
| 显卡 | **16×H20**，global batch 16（每卡 1 样本，无梯度累积） | 同左 |
| wall-clock | **≈48 h** | **≈13 h**（含 1–3 次 unroll/refine 渲染） |
| 损失 | input+target 渲染的等权 **masked L1 + LPIPS** | 同左 |

两阶段相加约 61 h，是 **16 卡并行 wall-clock**；论文没有另报含数据加载与评估的端到端总时长。本仓 `14_finetune_gs.py` 的 3000 步单场景属性精调**不属于**这 30K 网络训练。

### 4.1 InfiniSplat v2 的 target 采样与 mask 判定（论文给了细则）

- **target 采样**：把非 input 视角按“与最近 input 光轴的夹角”排序，前一半为 large-angle pool、后一半为 small-angle pool；以 **0.7 / 0.3** 概率选池，再无放回均匀抽 4 个 target；池内至少 4 个候选。
- **监督 mask** 需同时满足：
  1. target GT depth 有限且 `> 1e-6`；
  2. 原数据 validity mask 通过；
  3. 至少一个 input 视角满足投影有效且 `|z_proj − d_src| ≤ max(0.03, 0.01·d_src)`。
- 因此 ISv2 的 Depth 是**可见性/监督筛选依据**，不是网络输入通道；这一点与 ZipSplat 的 `depth_mask` 语义不同（第 9 节）。

## 5. SplatAnything：数据最杂，公开的训练配置最少

| 项 | 内容 |
|---|---|
| steps | **20,000 iterations**（500-step warmup） |
| 显卡 | **64×H20** |
| 每样本输入视角 | **4–32** |
| 训练分辨率 | **动态 504–1024**（只写“input resolution”，未说长边/短边） |
| 学习率 | Gaussian Backbone **5e-6**；GSHead **2e-5** |
| 损失 | target RGB 的 **MSE + LPIPS**，权重均为 1 |
| 初始化 | Backbone 从冻结的 WorldMirror 2.0 几何模型初始化并微调；GSHead 从头训练 |
| 训练 batch / 每样本 target 数 | **未公开** |
| wall-clock / 峰值显存 | **未公开** |
| mask 监督 | **未报告** |

### 5.1 14 个训练数据集（附录 Table 4）

| 归类 | 数据集 | #Scenes | 原始分辨率 |
|---|---|---:|---|
| Synthetic | HyperSim | 364 | 768×1024 |
| Synthetic | UnrealStereo4K | 18 | 1080×1920 |
| Synthetic | OmniObject3D | 5,914 | 800×800 |
| Synthetic | Rawmantic3D | 30 | 1080×1920 |
| Synthetic | OmniWorld | 10,464 | 720×1280 |
| Synthetic | GTA-SfM | 200 | 480×640 |
| Synthetic | MatrixCity | 8 | 1080×1920 |
| Synthetic | TartanGround | 635 | 640×640 |
| Real-world | ScanNet++ | 936 | 696×1035 |
| Real-world | Matterport3D | 90 | 1024×1280 |
| Real-world | ARKitScenes | 2,501 | 480×640 |
| Real-world | Taskonomy | 537 | 512×512 |
| Real-world | MapFree | 460 | 720×540 |
| Real-world | BlendedMVS | 502 | 384×504 |

表内算术和是合成 17,633、真实 5,026、合计 22,659 个 **“#Scenes” 条目**。三条限制：

1. 单位异质 —— 物体集按物体、TartanGround 按轨迹/序列、MatrixCity 的 8 不是 8 张地图；加总不等于去重后的独立场景数。
2. 这张表的分辨率是**原始数据属性**，不是训练处理分辨率。
3. OmniObject3D 底层是真实扫描物体，但训练视角由资产渲染，论文归为 Synthetic；保留论文口径并加注。

`Raw3D` / `Rawmantic3D` 的口径差异要注意：本仓数据清单按“真实三维训练数据”描述，而 SA Table 4 把它放在 **Synthetic** 下。两者指同一批 Rawmantic 数据，差异来自“资产是否来自真实扫描”与“训练视角是否渲染”的判定标准，引用时应写明是哪一种口径。

## 6. 训练/验证/测试划分

| 阶段 | 划分方式 | 是否场景隔离 | 备注 |
|---|---|---|---|
| DA3 几何主干 | Table 1 各集按自身 protocol | ScanNet++ 明确 scene-level 互斥 | 其它集划分比例未报 |
| DA3 GS-DPT | 10,015 train scenes；**140 个 DL3DV benchmark scenes 完全排除** | 是 | 这 140 是评测集，不能算进训练规模 |
| DA3-Metric | 未报 | 未报 | 逐名只列 13 个，正文称 14 个 |
| ZipSplat | 读 `data/<dataset>/<HxW>/train-scenes/` 与 `test-scenes/` 两套 shard | 是 | 504 配置的 `val_samples: 512` 用固定 seed 取固定验证子集；benchmark 另有独立固定的 context/target indices |
| InfiniSplat v2 | train / validation 场景不相交 | 是 | 具体场景数未报 |
| SplatAnything | validation 用固定确定性配置 | 部分 | 各集划分规模与比例未报 |

一个常见混写：**ZipSplat 训练时的随机采样 ≠ 论文评测的视角协议**。评测使用写死的 context/target 索引，训练使用 `bounded`/`rank`/`random` 采样器，两者不能互相引用。

## 7. 一个 optimizer step 到底输入多少张图

| 阶段 | 每样本图像数 | 全局每 step | 性质 |
|---|---|---|---|
| DA3 GS-DPT | 12 context + target（数未报） | **未知** | `batch size 1` 未说明 per-GPU 还是 global |
| ZipSplat 252（论文口径） | 2–24 context **+ 4 target** = **6–28** | context 768–9,216；target 1,536；合计 **2,304–10,752** | 算术边界，非实测 |
| ZipSplat 504（当前代码） | 同上 | **不固定** | 见 7.1 动态 batching |
| InfiniSplat v2 | **N + 4 = 8–16**（N = 4–12） | source 64–192；target **64**；合计 **128–256** | 由 global 16 直接推出 |
| SplatAnything | 4–32 input + target（数未报） | **未知** | batch 与 target 数都未公开 |

### 7.1 ZipSplat 为什么不能写“每步固定 384×(N+4)”

当前 `multi_view_dataset.py` 每个 batch 先随机定 context 数，再按每卡图像预算算 scene 数：

```python
vc = rng.randint(self._image_num_range[0], self._image_num_range[1])   # 2..24
ar = float(rng.choice(list(self.conf.aspect_ratios)))
bs = max(1, self.max_img_per_gpu // vc)
```

再加 `composed.py` 的“一个 batch 只含一个 child、跨数据集即截断”，实际每 step 的 scene 数随视角数与数据集边界浮动。因此：

- **论文**：每卡 24 samples、global 384、每样本 4 target；
- **代码**：按每卡图像预算动态决定 scene 数；
- **不能**把 `384×(N+4)` 当成代码每步永恒不变的真实张数。

InfiniSplat v2 的 128–256 张/step 则可放心使用，因为论文明确给了 `global batch 16`、`每卡 1 样本`、`无梯度累积`、`4–12 input + 4 target`，没有动态项。

### 7.2 把 steps 乘开：累计张次与访问遍数

第 7 节给的是“一步多少张”。乘上 steps 就是整轮训练看到多少**张次**（含重复）—— 这是唯一能算的规模量，但它不是唯一图像数。

| 阶段 | 唯一图像池（量级） | 全局每 step 张次 | steps | 累计张次 | 折算访问遍数 |
|---|---|---|---:|---:|---:|
| DA3 主干 | 22 个条目，量级未报 | 动态（token 守恒） | 200K | 不可算 | — |
| DA3 GS-DPT | DL3DV，百万帧级 | 12 + target（batch 归属歧义） | 200K | 240 万–1,920 万 | ≈2–19 |
| ZipSplat 252 | ≈1,000 万帧（§10.1 估算） | 2,304–10,752 | 450K | **10–48 亿** | ≈100–480 |
| ZipSplat 504 | ≈1,500 万帧 | 名义同左，代码动态 | 100K | 2.3–10.8 亿 | ≈15–70 |
| InfiniSplat v2 | 十万帧级 | 128–256 | 30K | 384–768 万 | ≈40–80 |
| SplatAnything | 22,659 条目，百万帧级 | 未知 | 20K | 不可算 | — |

算法与三条边界：

1. 张次 = 每 step 全局张数 × steps；区间两端取第 7 节的上下限，是算术边界，不是实测统计。
2. DA3 GS-DPT 的 240 万–1,920 万取决于 `batch size 1` 是 per-GPU 还是 global（8×A100 差 8 倍）。
3. 访问遍数 = 累计张次 ÷ 唯一图像池；池子取 §10.1 的量级估算，因此这一列是**量级**，不是精确倍数。

结论：**“训练用了多少万张图”在这四套配方里没有唯一答案。** 以 ZipSplat 252 为例，池子约 1,000 万帧、模型实际看了 10–48 亿张次，同一帧平均被访问 100–480 次；InfiniSplat v2 池子只有十万帧级，访问 40–80 遍。两者差近一个数量级，与 §3.1 里小数据集重复 shard 做 oversampling 的设计一致。

## 8. resize / crop / 分辨率处理

| 阶段 | 目标尺寸 | 处理顺序与插值 |
|---|---|---|
| DA3 GS-DPT | **270×480** | 论文未给 resize/crop 算法与插值 |
| DA3 主干 | 8 档随机（见 2.2） | 未给算法；batch 随分辨率动态调整以守恒 token |
| DA3-Metric | 基准 504，8 种宽高比（1:1、1:2、16:9、9:16、3:4、1:1.5、1.5:1、1:1.8） | 5% 概率旋转 90°/270°；canonical focal 300；crop/resize 细节未报 |
| ZipSplat 252 | **252×252**，`edge_divisible_by=14` | resize-to-cover → center crop → 必要时 crop 到 14 的倍数；RGB bilinear + antialias |
| ZipSplat 504 | 长边 504，7 种 `W×H` | 见 8.1，顺序严格 |
| InfiniSplat v2 | 长边 1022 / 756 / 518 | **只报长边**；短边、插值、是否 crop 均未报 |
| SplatAnything | 动态 504–1024 | 未报是否长边、宽高组合、插值或 crop |

### 8.1 ZipSplat 504 的真实处理链

1. **per-scene center zoom crop**：zoom 从 `[0.8, 1.0]` 采样，同一样本所有视角共用同一 zoom；
2. zoom 被 clamp，保证不因裁剪而 upscale；
3. resize-to-cover 到指定长边与宽高比；
4. center crop 到目标宽高比；
5. 必要时再 center crop 到 14 的倍数。

七种 `W×H`：`504×280`、`504×336`、`504×378`、`504×504`、`378×504`、`336×504`、`280×504`。每个 child 只列其原始图像在不放大时能支持的宽高比。

同步要求：相机内参必须随图像一起做 `3×3` 变换；Depth 在该路径用 nearest，mask 跟随 RGB 的空间变换。默认颜色增广为 identity。

## 9. Depth 与 mask：四种状态必须分开

“有没有 Depth/mask”要拆成四问：**原始数据有没有 → 转换后 shard 有没有 → 训练代码读不读 → 损失用不用**。

| 阶段 / 数据 | 原始发布含 Depth | 原始发布含 mask | 训练实际读取 | 真实用途 |
|---|---|---|---|---|
| DA3 主干（22 集） | 部分（LiDAR 稀疏、重建派生） | 部分 | RGB | 真实深度经 teacher 补相对深度并 RANSAC scale-shift 对齐后作监督；teacher 自身有 sky/object mask loss |
| DA3 GS-DPT（DL3DV） | **无可靠逐帧 GT** | 无 | RGB | 渲染新视角的 MSE+LPIPS；observed views 加 scale-shift-invariant depth loss；**GT depth 不作输入** |
| DA3-Metric | **有 metric GT** | — | RGB | depth + gradient + sky-mask 三项损失；20% 概率用原始 GT；stereo 用 FoundationStereo 预测当标签 |
| ZipSplat：RE10K / DL3DV | **无** | 无 | RGB + DA3 `extract_depth` 补的 depth | `depth_mask` 是 `depth>0` 的**有效区**，不是前景；rendered-depth L1 默认权重 0；GeoCE 用 depth + 相机 + validity |
| ZipSplat：ScanNet++ / ARKitScenes / Hypersim / UnrealStereo4K | 原生 RGB-D | 部分 | 同上 | 同上 |
| ZipSplat：CO3D / WildRGB-D | 部分 | **发布含前景 mask** | **`MultiViewDataset._load_view()` 未读取 mask** | 无前景 mask 监督 |
| InfiniSplat v2 | Hypersim / ScanNet++ 有 GT | 有 validity mask | RGB + GT pose（训练时） | GT depth 仅用于 visibility/supervision mask 与 target 采样；损失是 masked L1 + LPIPS |
| SplatAnything（14 集） | 部分有 GT | 部分有 | RGB（+ 可选内参/外参） | WorldMirror 预测 depth 反投影 Gaussian 中心；**未报告直接 depth loss**；mask 用途未报告 |

三条结论：

1. **ZipSplat 的 `depth_mask` ≠ 人物/物体前景 mask。**`ImagePreprocessor` 虽然支持传语义/二值 mask，但当前数据加载路径没有把 scene 的 mask 传进去。
2. **ISv2 的 GT depth 是“筛选器”，不是输入通道。** 它的 mask 判定含 `|z_proj − d_src| ≤ max(0.03, 0.01·d_src)` 的几何一致性检验。
3. **SA 的 WorldMirror 是冻结教师。** 教师的预训练数据不能并入这 14 个学生训练集；也不能因为教师输出 depth 就写“学生有 depth 监督”。

## 10. “渲染了多少图像”——多数算不出来

| 阶段 | 官方是否给出图像总数 | 为什么不能算 |
|---|---|---|
| DA3 主干 | **否** | Table 1 是 scene/对象条目，不是图片；各集采样权重也未给 |
| DA3 GS-DPT | **否** | 只给 10,015 scenes，未给每 scene 抽帧数与 target 数 |
| ZipSplat 252 / 504 | **否** | 未给每 scene 唯一帧使用数；小数据集还重复 shard |
| InfiniSplat v2 | **否** | 未给每 scene 抽帧数与总 scene 数 |
| SplatAnything | **否** | 未给 batch、每样本 target 数、各集抽帧数 |

可以公开引用的替代量只有一个：**每 step 的图像张数区间**（第 7 节），以及 **steps × batch × views 的访问量**。后者含重复，不等于唯一图像数。已有数据规模也不能替代：DL3DV 的 51.2M 帧是完整发布量，Hypersim 的 74,619 张是公开发布量，都不是某方法的实际读取量。

### 10.1 反过来算：自己跑要准备多少张图

论文没给唯一图像数，但采样器有一条**硬门槛**：一个 scene 至少要凑够 `max_context + target` 张不同视角，凑不够整条样本直接被丢弃。这是准备数据时唯一不能违反的约束，比“总共有多少张”更要紧。

| 方法 | 需要几个 scene / 序列 | 每序列**最少**张数 | 总量级 | 下载体积 |
|---|---|---|---|---|
| DA3-GS（本仓单场景） | 1 | 30（实测 125 视角质量更好） | 百张 | < 5 GB |
| DA3 GS-DPT | 10,015 个 DL3DV scene | ≥12（12 context；target 数未报） | 百万帧级 | DL3DV 数 TB |
| ZipSplat 252 | RE10K 约 80K clip + DL3DV 10,510 | **28**（24 + 4） | ≈1,000 万帧 | ≈2.8 TB |
| ZipSplat 504 | 上述 + 8 个数据集 | **28** | ≈1,500 万帧 | ≈10 TB |
| InfiniSplat v2 | Hypersim 461 + ScanNet++ 1,000 + Raw3D 30 | 16（12 + 4） | 十万帧级 | 数 TB |
| SplatAnything | 22,659 个条目 | ≥32（32 input；target 数未报） | 百万帧级 | 数 TB |

判断某个数据集够不够用，看它的**平均每序列帧数**。下面三个是能直接从公开数字算出来的，其余数据集的帧数分布未公开，只能下完数据后统计 shard 里的 `num_views`：

| 数据集 | 平均每 scene/clip 帧数 | 对 28 张门槛的余量 |
|---|---|---|
| RE10K | 约 125（10M 帧 ÷ 80K clip） | 充足 |
| Hypersim | 162（74,619 ÷ 461） | 充足 |
| G-Objaverse | **40 视角 / 对象**（脚本写死） | **仅剩 12 帧**，不能再裁视角 |

## 11. 未公开项清单（不要靠乘法补）

| 项 | DA3 GS-DPT | DA3 主干 | ZipSplat 252 | ZipSplat 504 | ISv2 | SplatAnything |
|---|---|---|---|---|---|---|
| 训练图像总数 | 未公开 | 未公开 | 未公开 | 未公开 | 未公开 | 未公开 |
| 每样本 target 数 | 未公开 | — | **4** | **4** | **4** | 未公开 |
| batch 的 per-GPU/global 归属 | 歧义 | 未公开 | 明确 24/卡 | 配置 384，实际动态 | **明确 global 16** | 未公开 |
| 训练 wall-clock | 未公开 | ≈10 天 | 未公开 | 未公开 | 48 h + 13 h | 未公开 |
| 训练峰值显存 | 未公开 | 未公开 | 未公开 | 未公开 | 未公开 | 未公开 |
| resize/crop 算法 | 未公开 | 未公开 | 已核准 | 已核准 | 部分（仅长边） | 未公开 |

## 12. 人像数据的边界

四套官方配方里**没有专门的真人身份、人头或三图人像训练集**：

- **G-Objaverse `Human-Shape`**：ZipSplat 504 的 G-Objaverse 权重 0.05，类别注释约 41,557 个资产、每对象通常 40 视角；这是**静态资产的渲染**，不是带身份划分的真人数据集，41,557 也不是 ZipSplat 实际人物样本数。
- **OmniWorld 的 human 子集**：不能自动并入 SA 的 10,464 个合成场景，也不能据此宣称有人像身份训练。
- **本仓 UE53**：27 身份 / 53 sequence，train 16/32、val 2/4、test 2/4、hard-only 7/13；非公开、非官方 SA 训练集，正式训练尚未过审核门。

需要人像能力时，正确做法是按身份/sequence 重新划分并留独立 target，而不是从这些场景数据里“间接学到”。

## 13. 证据来源

- 论文全文：DA3 `arXiv:2511.10647v1`、InfiniSplat v2 `arXiv:2610.20341v1`、SplatAnything `arXiv:2611.08264v2`、ZipSplat `arXiv:2606.05102`。
- DA3 关键定位：主干资源 L843-845 / L1551；GS-DPT 训练配置 L2339；DL3DV 10,015 与 140 排除 L1017；Table 1 L652-834；Metric L954-967。
- ISv2 关键定位：数据混合 L723-727 / L1982-1989；视角与分辨率 L1985-1989、L2190-2192；target 采样 L1993-2001；mask L2010-2028；资源 L2090-2103。
- SA 关键定位：训练设置 L715-719；Table 4 L1763-1839；采样 L1851-1862；scale 对齐 L2002-2020；推理协议 L3749-3752。
- ZipSplat 官方配置：`~/repos/ZipSplat/splatfactory/configs/{zipsplat.yaml, zipsplat_ft504.yaml}`、`configs/data/{re10k_dl3dv_252.yaml, multi_dataset_504.yaml}`；源码 `datasets/multi_view_dataset.py`、`datasets/composed.py`、`datasets/view_sampler/rank.py`、`utils/image.py`、`trainer.py`。

本文只做信息核对，未执行官方训练，也未下载训练集；表中“未公开”均为已核对后的结论，不是尚未查找。

