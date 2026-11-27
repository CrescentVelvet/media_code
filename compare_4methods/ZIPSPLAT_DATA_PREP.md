# ZipSplat 重训：数据准备手册（手动下载版）

> 配套：[官方训练数据与训练配置明细](OFFICIAL_TRAINING_DETAILS.md) ｜ [训练指南](TRAINING_GUIDE_3DGS.md) ｜ [数据集总清单](TRAINING_DATASETS.md)
>
> 核对日期：2026-11-27。命令与目录均来自官方仓 `splatfactory/datasets/README.md`、各 `download.py`/`convert.py` 与 `configs/data/multi_dataset_504.yaml`。**本文未执行下载**，体积为发布规模近似值。

## 0. 先分清三种视角

一次训练样本（sample）= **一个场景 + 一组抽出来的视角**，视角被分成两组：

| 术语 | 在本仓/论文里的叫法 | 是否进网络编码器 | 作用 |
|---|---|---|---|
| **context**（上下文视角） | 也叫 input / source / 输入视角 | **是** | 网络只看这些图，预测深度、相机和 3D 高斯；它们定义坐标系与基线 |
| **target**（目标视角） | 监督视角 / novel view | **否** | 只拿 GT 图和新视角相机算渲染损失，不参与建表示 |
| **reference view** | 第一个 context | 是 | 固定为世界坐标原点（GSN 归一化的基准），因此 `random_reference_view` 会改变尺度与朝向后处理 |

几个容易误解的点：

1. **context 不是“历史帧”**。ZipSplat 是前馈模型，没有时序状态；每个 step 从同一场景随机抽 N 个视角当 context，N 在 2–24 之间随机（504 配置从 step 0 就是全范围）。
2. **target 也不是“下一帧”**。它是**留出的其它视角**，只用来算 `MSE + LPIPS`。RE10K/DL3DV 的 target 通过帧距约束选（`bounded`），ScanNet++ 等通过共视度选（`rank`），物体类随机选（`random`）。
3. **context 与 target 索引在同一样本内不重复**，且都来自同一场景。所以一个 scene 至少要 `N + 4` 张图，否则该 scene 被跳过。
4. **一个 batch 只含一个数据集（child）**，跨数据集会截断当前 batch —— 这就是“多数据集混合”实现上不等于把十份数据混成一锅的原因。

用一张图概括一个 step：

```text
sample（一个 scene）
├── context: N 张 (N ~ U[2,24])  ──► encoder ──► depth/camera ──► Gaussians
│                                                    │
└── target: 4 张 (GT RGB + 相机) ────────────────────┴──► render ──► MSE + LPIPS
```

所以“一个 step 输入多少张图”= `Σ_samples (N_i + 4)`；论文口径与代码动态 batch 的差别见明细文档第 7 节。

## 1. 开跑前必须准备好的东西

| 类别 | 具体内容 | 说明 |
|---|---|---|
| 代码 | 官方仓 `cvg/ZipSplat` + `splatfactory` 环境（Python 3.12、torch、gsplat） | 与本仓轻量推理 env 分开 |
| 权重 | **DA3 `depth-anything/DA3NESTED-GIANT-LARGE`** | `extract_depth.py` 写死这个名字；RE10K / DL3DV 补深度必需，要 GPU |
| 包 | `depth_anything_3`（pip） | 只补深度时需要 |
| 显卡 | 补深度、precompute rankings 都要 GPU | 单卡可跑，`torchrun` 可多卡 |
| 磁盘 | raw + shard + depth 回写 ≈ 原始 2–3 倍 | shard 是 tar，深度以 WebP 存，不能按 raw 大小估算 |
| 授权 | DL3DV / ScanNet++ / Matterport 类需申请或签协议 | 脚本只从官方 host 取，不绕过条款 |

## 2. 三个层级，别一上来就全量

| 层级 | 数据 | 体积量级 | 能验证什么 |
|---|---|---|---|
| **T0 冒烟** | DL3DV `--sample`（50 scenes）+ Hypersim `--max-scenes 10` | 几十 GB | 转换、补深度、rankings、训练循环是否通（各自验一遍转换，训练只挂一个 child） |
| **T1 复现 252** | RE10K 全量 + DL3DV 960P（至少一个子集） | RE10K 未知 + DL3DV ≈2.8 TB（960P 全量） | 论文 252 配方（450K steps）能否复现趋势 |
| **T2 精调 504** | 十个数据集全量 | **≈10 TB 级** | 发布 504 模型的第二阶段 |

T0 要自己写一份只含一个 child 的小 yaml（把 `re10k_dl3dv_252.yaml` 复制一份，删掉一个 child、`batch_size` 改 4–8、`num_workers` 改 2）；官方配置没有为小样准备开关，直接改 `batch_size` 跑全量数据会一直读盘。

## 3. 手动下载清单

以下命令都在**官方仓根目录**执行（`cd ~/repos/ZipSplat`），前缀统一为 `python -m splatfactory.datasets.scripts.`，表里用 `<ds>` 简写。每个数据集都是“先 download 原始包、再 convert 成 shard”。

### 3.1 T1：252 配方只要这两个

| 数据集 | 官方入口 | 下载命令 | shard 目录（配置写死） | 体积 | 备注 |
|---|---|---|---|---|---|
| **RE10K** | [项目页](https://google.github.io/realestate10k/)；脚本从 `http://schadenfreude.csail.mit.edu:8000/re10k.zip` 取 360×640 的 `.torch` 包 | `python -m splatfactory.datasets.scripts.re10k.download` → `... .re10k.convert --num-workers 8` | `data/re10k/360x640` | 以站点为准（未取到 Content-Length） | 无需补深度（转换后由 DA3 补）；`--split all` 默认同时出 train/test |
| **DL3DV** | [GitHub](https://github.com/DL3DV-10K/Dataset)；HF 数据集：[960P](https://huggingface.co/datasets/DL3DV/DL3DV-ALL-960P)（另有 [480P](https://huggingface.co/datasets/DL3DV/DL3DV-ALL-480P) / 2K / 4K） | `... .dl3dv.download --subset 1K --resolution 960P` → `.dl3dv.convert --resolution 960P --split all --num-workers 8` | `data/dl3dv/540x960` | 960P 全量 ≈2.8 TB | **gated**：先在 HF 页面同意条款并 `huggingface-cli login`；**必须 960P**，480P 出的是 270×480，目录名对不上 |

小样验证用：`... .dl3dv.download --sample`（1K 子集前 50 个 scene）。它仍然走 960P 目录，只是数量少。

### 3.2 T2：504 精调追加的八个

| 数据集 | 官方入口 | 下载命令 | shard 目录 | 体积（近似） | 授权 / 前置 |
|---|---|---|---|---|---|
| **ScanNet++** | [官网](https://kaldir.vc.in.tum.de/scannetpp/)（需签署协议） | **无下载脚本**，用 [官方 toolkit](https://github.com/scannetpp/scannetpp) 拉到 `data/scannetpp_v2/`，再 `.scannetpp.convert --device dslr --max-side 1080 --num-workers 12`（iPhone 另跑一次 `--device iphone`） | `data/scannetpp_v2/720x1080/{dslr,iphone}` | 1,000+ 场景，体积以协议页为准 | 需注册与签署；转换要 `open3d`（自己渲染 mesh 深度） |
| **ARKitScenes** | [GitHub](https://github.com/apple/ARKitScenes)；资源域 `docs-assets.developer.apple.com/.../arkitscenes/v1` | `... .arkitscenes.download --max-scenes 20`（全量去掉该参数）→ `.arkitscenes.convert --num-workers 8` | `data/arkitscenes/{480x640,640x480}`（横/竖两棵树） | raw 数 TB（5,047 videos） | 无 gating；转换按 Apple sky_direction 转正，横竖分树 |
| **Hypersim** | [GitHub](https://github.com/apple/ml-hypersim)；资源域 `.../hypersim/v1/scenes` | `... .hypersim.download --max-scenes 10` → `.hypersim.convert --num-workers 8` | `data/ml-hypersim/768x1024` | 461 场景 ≈1.9 TB | 无 gating；自带深度 |
| **UnrealStereo4K** | [SMD-Nets](https://github.com/fabiotosi92/SMD-Nets)；HF 数据集 [fabiotosi92/UnrealStereo4K](https://huggingface.co/datasets/fabiotosi92/UnrealStereo4K) | `... .unrealstereo4k.download --sample`（8 环境/80 chunks）→ `.unrealstereo4k.convert` | `data/unrealstereo4k/540x960` | 较小（80 chunks） | 无 gating；帧间曝光会闪，权重只给 0.02 |
| **BlendedMVS** | [GitHub](https://github.com/YoYo000/BlendedMVS) / [v1.0.0 Release](https://github.com/YoYo000/BlendedMVS/releases/tag/v1.0.0) | `... .blendedmvs.download --num-workers 4` → `.blendedmvs.convert` | `data/blendedmvs/576x768` | 低分辨率 113 场景 ≈30 GB | 无 gating；唯一航拍/大尺度内容 |
| **CO3D** | [GitHub](https://github.com/facebookresearch/co3d)；`links.json` 给出分片地址 | `... .co3d.download --single-sequence`（≈8.9 GB）或 `--categories ... --max-chunks 2` → `.co3d.convert` | `data/co3d/720s` | 完整 ≈5.5 TB | 非商业条款；50 类物体环绕 |
| **WildRGB-D** | [GitHub](https://github.com/wildrgbd/wildrgbd)；HF 数据集 [hongchi/wildrgbd](https://huggingface.co/datasets/hongchi/wildrgbd) | `... .wildrgbd.download --max-categories 4` → `.wildrgbd.convert` | `data/wildrgbd/640x480` | ≈3.37 TB 压缩 | 非商业条款 |
| **G-Objaverse** | [richdreamer 数据页](https://github.com/modelscope/richdreamer/tree/main/dataset/gobjaverse)；tar 源 `virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d` | `... .gobjaverse.download --max-per-category 10`（全量去掉该参数）→ `.gobjaverse.convert` | `data/gobjaverse/512x512` | 约 280K 对象、40 视角/对象 ≈6.5 TB | 继承底层 Objaverse 资产许可；**`Human-Shape` 是静态人物形体资产，不是真人集** |

504 配置不涉及上表之外的数据集：十个 child 就是这 8 个数据集（ScanNet++、ARKitScenes 各拆两棵树）加 RE10K 与 DL3DV。

## 4. 转完之后按这个顺序处理

```bash
cd ~/repos/ZipSplat
M="python -m splatfactory.datasets.scripts"

# 1) 补深度：只有 RE10K / DL3DV 需要（其它数据集自带 depth）
$M.extract_depth data/re10k/360x640/train-scenes
torchrun --nproc_per_node=4 -m splatfactory.datasets.scripts.extract_depth \
    data/dl3dv/540x960/train-scenes --max-views 150

# 2) 共视排序：只有 rank 采样器的四个数据集需要
$M.precompute_rankings data/scannetpp_v2/720x1080/dslr/train-scenes \
                        data/scannetpp_v2/720x1080/dslr/test-scenes
$M.precompute_rankings data/arkitscenes/480x640/train-scenes \
                        data/arkitscenes/480x640/test-scenes
$M.precompute_rankings data/ml-hypersim/768x1024/train-scenes \
                        data/ml-hypersim/768x1024/test-scenes
$M.precompute_rankings data/unrealstereo4k/540x960/train-scenes \
                        data/unrealstereo4k/540x960/test-scenes

# 3) 可选：先缩一次，省训练时的 CPU/IO
$M.resize data/dl3dv/540x960/train-scenes \
    --output data/dl3dv/252x252/train-scenes --size 252 --num-workers 32
```

三点约束：

- `extract_depth` 是**原地重写 shard**（临时文件 + 原子替换），会调 DA3 GIANT-LARGE，要有 GPU 与足够时间；`--max-views` 把长序列轮转切块，保证同一场景的深度尺度一致。
- `precompute_rankings` 也写回 shard，已算过的场景会跳过，重跑是 no-op；它只读 depth/camera/pose，所以**必须放在补深度之后**。
- 若做了第 3 步预缩，记得把 yaml 里的 `dataset_dir` 改到 `252x252` 那份，否则仍是训练时在线缩放。

## 5. 开工前的检查清单

| 检查项 | 怎么查 | 不通过会怎样 |
|---|---|---|
| 目录名与实际分辨率一致 | 随便读一个 scene 的图像 shape，确认是 `360x640` / `540x960` 等 | 配置直接读不到，或 aspect 预设全部被过滤 |
| train / test 两棵树都在 | `ls data/<ds>/<HxW>/` 应有 `train-scenes/` `test-scenes/` | `val_samples: 512` 取不到验证集 |
| depth 已补且有效 | 抽一个 scene 看 `depths` 非全零、`depth_ranges` 合理 | 该视角 `depth_mask` 全 false，几何监督失效 |
| rankings 已写 | rank 数据集 shard 内应有 `*.rankings.npy` 等四个成员 | rank 采样器无候选，样本被跳过 |
| yaml 的 `dataset_dir` 对得上 | 逐 child 核 `dataset_dir` / `train_shard_dir` | 静默回退或直接报错 |
| 252 与 504 不混用 | 252 配置写的是 `resize: 252` + 360x640/540x960 源 | 把 504 目录塞进 252 配置会得到错误视野与分辨率 |

## 6. 六个容易踩的坑

1. **DL3DV 默认 480P**。不加 `--resolution 960P` 会下 270×480，而配置要 `data/dl3dv/540x960`，目录名对不上不是“改个名”能修的。
2. **DL3DV 是 gated**。脚本会先 `verify_hf_access`，没在 HF 页面同意条款就直接退出，报错里给了页面链接。
3. **ScanNet++ 没有下载脚本**。它是签协议分发的，必须用官方 toolkit 自己拉，再 `convert`；别找第三方转存。
4. **G-Objaverse 的 `Human-Shape` 不是真人数据**。它是静态人物形体资产的多视角渲染，学不到身份与真人皮肤细节；要人像得自建身份/sequence 划分。
5. **小数据集的权重不是数据量**。0.02 的 UnrealStereo4K 靠重复 shard 被过度采样，`权重 × 总样本` 不能当“访问了多少唯一场景”。
6. **全量 504 是 TB 级投入**。先跑 T0 冒烟再决定：转换链、补深度、rankings 任一环节错了，全量重来代价很高。

## 7. 与本仓的关系

本仓 `zipsplat/` 只用发布权重做推理、TokenBA 和导出，**不需要上面任何训练数据**。这套清单只在你要自己重训或微调 `splatfactory` 时才用得上；跑通 T0 也不等于复现了官方指标。

