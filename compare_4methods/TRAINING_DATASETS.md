# 四种方法：训练数据集与手动下载清单

> 配套阅读：[训练、精调与评测指南](TRAINING_GUIDE_3DGS.md)。
>
> 核对日期：2026-11-27。优先依据论文、ZipSplat 官方配置/下载脚本及数据集官方页面；容量均为近似值，不代表解压后仍相同。

## 1. 先分清“训练”与“本仓运行”

| 方法 | 本仓运行是否下载这些训练集 | 官方训练数据口径 |
|---|---|---|
| DA3-GS | **否**。本仓 `da3_human/04` 只需当前场景图片，用 DA3 初始化后训练原版 3DGS | DA3 前馈 NVS 模型（GS-DPT）用 **10,015 个 DL3DV 场景**；140 个 benchmark 场景严格排除。DA3 几何主干另有更大训练配方，不能与 GS-DPT 混写 |
| ZipSplat | 用发布权重推理不需要；重训官方 `splatfactory` 才需要 | 252：RE10K + DL3DV；504 第二阶段：10 个公开数据集，见下表 |
| InfiniSplat v2 | **否**。本仓是前馈推理及已有 PLY 的单场景属性精调 | Stage I/II 均为 Hypersim / Raw3D / ScanNet++，采样比 **60% / 10% / 30%**；公开包无对应训练器 |
| SplatAnything | 推理不需要；`09/40` 是本仓自研探针/人物域实验 | 官方论文附录 Table 4 给出 14 个训练集及实际处理规模；公开包无训练器，WorldMirror 几何模型在相关实验中冻结 |

**“论文使用场景数”与“官方发布总规模”是两列，不能互换。** 同名数据集被不同方法使用时，采样、划分、处理和分辨率也不同。`Raw3D` / `Rawmantic3D` 指论文引用的 Rawmantic “3D Training Field”；其完整公开下载尚未核实，故保留在清单中但不标为可直接下载。

主要论文依据：[DA3 §5.3](https://arxiv.org/html/2511.10647)、[InfiniSplat v2 附录 A.1](https://arxiv.org/html/2610.20341)、[SplatAnything 附录 Table 4](https://arxiv.org/html/2611.08264)。

## 2. 人像 / 人物相关数据（优先）

### 2.1 最重要的边界

四种方法的已核对训练配方中，**没有专门的真人身份、人头或三图人像数据集**。若目标是让模型学会少视角人像重建，下载几 TB 室内/街景数据并不能替代人物数据；需要另行按身份/sequence 划分，先留独立 target，再设计人物域实验。

| 数据 | 与四种方法的关系 | 能下载什么 | 不应误解成什么 |
|---|---|---|---|
| **G-Objaverse `Human-Shape`** | ZipSplat 504 第二阶段：整个 G-Objaverse 的 scene 采样权重为 **0.05**；默认下载类别包含 `Human-Shape` | 静态 Objaverse 资产的多视角渲染；类别索引注释为 **41,557 个 Human-Shape 资产**；每对象通常 40 视角，整包约 6.5 TB | **不是**真人人像/人脸身份集；41,557 不是 ZipSplat 实际有效人物训练场景数，也不证明人物域提升 |
| **本仓 UE53** | SplatAnything `40` 自研人物域短训协议 | 不可公开下载：27 个身份、53 个自有 UE sequence；train 16/32、val 2/4、test 2/4、hard-only 7/13（身份/sequence） | 不是官方 SplatAnything 训练集；审核、真实训练和质量收益仍未完成 |
| **OmniWorld 全量发布中的人类域** | SplatAnything 使用 OmniWorld 合成数据 **10,464 场景**；DA3 几何主干使用 1,039 场景 | OmniWorld 总库还整合 human/robot/internet 数据；应只选择含 RGB、深度和位姿的合适子集 | 不能把全库的 human 子集自动算入 SplatAnything 的 10,464 个合成训练场景，也不能据此宣称有人像身份训练 |
| **海思采集、物体三图、佳佳抱枕等** | 本仓推理/评测/单场景实验数据 | 以本仓现有文件为准 | 不是任何一篇论文的公开训练集 |

`Human-Shape` 的类别说明与入口见 [G-Objaverse 数据页](https://github.com/modelscope/richdreamer/tree/main/dataset/gobjaverse)。渲染包的公开不等于每个底层 Objaverse 资产都允许商用或再分发；应保留对象 ID 并逐项检查原始资产许可。

### 2.2 只下载 10 个 Human-Shape 资产验证链路

在官方 ZipSplat 仓根目录执行；这只是小样本，不是完整人物训练：

```bash
cd ~/repos/ZipSplat
python -m splatfactory.datasets.scripts.gobjaverse.download \
  --categories Human-Shape --max-per-category 10 --num-workers 4
python -m splatfactory.datasets.scripts.gobjaverse.convert --num-workers 8
```

若要做真人肖像研究，应另找许可清晰的人像/人头数据，并确认多视角、相机、身份划分及隐私条款；本文不把它包装成上述模型的官方数据配方。

## 3. 场景数据：训练使用与官方下载

### 3.1 一页总表

缩写：`Z` = ZipSplat，`I` = InfiniSplat v2，`S` = SplatAnything，`D` = DA3。数字优先写该方法论文/配置的实际口径；`—` 表示没有可靠统一数值。

| 数据集 | 方法及训练使用规模 | 官方发布规模 / 模态 | 下载、容量与许可提示 |
|---|---|---|---|
| [DL3DV](https://github.com/DL3DV-10K/Dataset) | **D/GS-DPT：10,015**；Z：252 的 50%，504 的 32% | 10,510 视频/场景、51.2M 帧；RGB + COLMAP 位姿 | 官方 480P 约 730 GB、960P 2.8 TB、2K 11 TB、4K 44 TB；HF 逐数据集同意条款、登录下载；Z 转换目录要求 540×960，见 §5 |
| [RE10K](https://google.github.io/realestate10k/) | Z：252 的 50%，504 的 5%；S 未列为训练集 | 原始约 80K clips / 10M 帧；YouTube 房地产视频、相机 | Z 用 360×640 预处理 `.torch` 包，非原始视频；后续仍需补深度；按原页面条款使用 |
| [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/) | I：30%，训练/验证不重叠；S：936，696×1035；Z：DSLR 10% + iPhone 8%；D 主干：230 | 1000+ 室内场景；DSLR/iPhone RGB、深度、位姿、激光 mesh | 注册、签署协议，用官方 toolkit；Z 需要 mesh 渲染深度、相机与 rankings，见 §5 |
| [ARKitScenes](https://github.com/apple/ARKitScenes) | S：2,501，480×640；Z：横屏 7% + 竖屏 6%；D 主干：4,388 | 5,047 captures、1,661 唯一室内场景；VGA/高分辨率 RGB-D、位姿 | Z 下载 VGA wide、内参、低分辨率 depth、trajectory；raw 数 TB，脚本估计每 10 视频约 2–4 GB；遵守官方协议 |
| [Hypersim](https://github.com/apple/ml-hypersim) | I：60%；S：364，768×1024；Z：5%；D 主干：344 | 461 合成室内场景，公开 74,619 张图（原生成 77,400，过滤人物/显著品牌等）；RGB、深度、法线、位姿等 | Z 发布包约 1.9 TB，可按场景下载；图像包 **CC BY-SA 3.0**；原始 Evermotion 三角网格资产需另购 |
| [UnrealStereo4K](https://github.com/fabiotosi92/SMD-Nets) | S：18，1080×1920；Z：只用 8 个环境 / 80 chunks，权重 2% | 原发布 9 环境：00000–00007 各 1000 双目对，00008 为 200 对域外测试；RGB、视差/深度 | Z 用 540×960 Q 版，约 4 GB/环境，不含相机；脚本从 4K 大包 Range 取相机，勿把 Z 的 8 与 S 的 18 混写 |
| [Matterport3D](https://niessner.github.io/Matterport/) | S：90，1024×1280 | 90 建筑、10,800 全景、194,400 RGB-D；位姿、网格、2D/3D 语义 | 填签官方 Terms of Use 后邮件申请；下载容量和商用权以协议为准，不把代码许可当数据许可 |
| [Taskonomy](https://github.com/StanfordVL/taskonomy/tree/master/data) | S：537，512×512 | 537 建筑、460 万+ 室内图；RGB、深度、法线、关键点等 | 完整约 **11.16 TB**；推荐 Omnidata tools 按 component 下载，先选 tiny 验证；`--agree` 表示接受其条款 |
| [MapFree](https://www.nianticspatial.com/research/map-free/dataset) | S：460，720×540；D 主干：921 | 655 地点 = 460 train / 65 val / 130 test；RGB、内参、位姿，train 有两 scans | 样例 0.5 GB；train 110 GB、COLMAP models 30 GB、val 8 GB、test 15 GB；**仅非商业用途**，接受专用协议 |
| [Raw3D / Rawmantic3D](https://raw3d.xyz/dataset) | I：10%；S：30，1080×1920 | 论文引用的真实三维训练数据；官方总量/模态清单未可靠取得 | 官方场景展示页不等于完整数据下载；**完整包公开下载、容量、许可待核实**，需联系官方，不用第三方转存代替 |
| [GTA-SfM](https://github.com/HKUST-Aerial-Robotics/Flow-Motion-Depth/tree/master/extracted_dataset) | S：200，480×640 | GTA5 提取的 RGB 序列、深度、相机；train/test 地理区域不重叠 | 官方 README 提供百度云 train/test 与 sample，Dropbox 已失效；容量、完整可用性、数据许可需按入口核验，非 MIT 自动推断 |
| [MatrixCity](https://github.com/city-super/MatrixCity) | S：8，1080×1920 | Small City 2.7 km²、Big City 25.3 km²；航拍/街景 RGB、位姿、深度、法线等 | HF / OpenXLab / 百度云（官方 README）；可按地图与模态选包；仓代码 Apache-2.0，不自动代表所有数据资产许可 |
| [TartanGround](https://huggingface.co/datasets/theairlabcmu/TartanGround) | S：635，640×640 | 63 合成环境、878 轨迹、144 万+ 样本；多向 RGB、深度、位姿、光流、LiDAR 等 | HF 卡标注 **CC BY 4.0**；全模态体量大，先按环境、轨迹和 RGB/depth/pose 选文件，勿默认整库下载 |
| [OmniWorld](https://github.com/yangzhou24/OmniWorld) | S：10,464，720×1280；D 主干：1,039 | 总库 600K+ sequences / 300M+ frames；其中 OmniWorld-Game 96K clips、18M+ 帧、720P，RGB/深度/位姿/文本/光流/mask | [HF](https://huggingface.co/datasets/InternRobotics/OmniWorld) / [ModelScope](https://modelscope.cn/datasets/InternRobotics/OmniWorld) 按子集下载；**CC BY-NC-SA 4.0 + 游戏资产附加限制**；SA 子集划分需另获，不是全库等同训练集 |

说明：

- InfiniSplat v2 每样本 4–12 输入 + 4 target；`N≤8` 长边 1022，`N=9/10` 为 756，`N=11/12` 为 518。这里数据集的规模是训练来源，不表示本仓能运行官方 Stage I/II。
- SplatAnything 表中数字是论文实际使用 `# Scenes` 和处理分辨率 H×W，不是官方总规模；官方训练 20K iterations、64×H20，不代表公开代码可复现。
- ZipSplat 504 的百分数是**抽 scene 的权重**，不是容量、样本量或耗时占比。ScanNet++ 两设备、ARKitScenes 两方向是 child，合计仍十个数据集。

## 4. 物体数据：训练使用与官方下载

| 数据集 | 方法及训练使用规模 | 官方发布规模 / 模态 | 下载、容量与许可提示 |
|---|---|---|---|
| [G-Objaverse](https://github.com/modelscope/richdreamer/tree/main/dataset/gobjaverse) | Z 504：5%，默认除 Poor-quality 外的类别；含 Human-Shape | 脚本口径约 280K Objaverse 对象、40 视角/对象；官方压缩 tar 镜像有“约 260K 文件”口径 | 约 6.5 TB，平均约 25 MB/对象；可按类别/`max-per-category` 小样下载；保留底层资产各自许可 |
| [CO3Dv2](https://github.com/facebookresearch/co3d) | Z 504：10% | 51 类常见物体，视频、mask、COLMAP 相机/点云 | 完整约 **5.5 TB**；single-sequence 子集约 **8.9 GB**，仅适合链路验证；官方页列 **CC BY-NC 4.0**，非商业 |
| [WildRGB-D](https://github.com/wildrgbd/wildrgbd) | Z 504：8%；D 主干：23,050 | 46 类真实物体；RGB、深度、mask、内参、OpenCV c2w | 官方约 3.37 TB 压缩 / 4 TB 解压；Z 脚本写约 3.7 TB，是不同近似口径；可按类别 zip 下载，核验许可 |
| [OmniObject3D](https://www.omniobject3d.net/) | S：5,914，800×800；D 主干：5,885 | 约 6,000 真实扫描物体、190 类；textured mesh、点云、多视角渲染和真实视频 | 从官方页跳 OpenXLab，按类别/模态选包；S 的 5,914 不是官方总物体数；下载前核对使用协议 |
| [BlendedMVS](https://github.com/YoYo000/BlendedMVS) | Z 504：2%，低分辨率 113 场景；S：502，384×504；D 主干：503 | 重建物体/场景多视角图、深度与相机；低分辨率版本 768×576 | Z 低分辨率 release 16 分卷、约 30 GB，可保留 zip 供转换器读取；完整/其它版本不等同此训练子集 |

SA 的 `OmniObject3D` 虽由真实扫描资产渲染训练视角，论文 Table 4 将其列为 **Synthetic**；本表按内容把它放物体数据，不擅自改论文类型。CO3D 与 WildRGB-D 含真实世界视频/RGB-D，但其训练视角仍是围绕物体采集，不是人像身份数据。

## 5. ZipSplat：从手动下载到可训练 shards

### 5.1 官方转换目录与流程

在 `~/repos/ZipSplat` 下执行；先建独立训练环境并查看[官方数据 README](https://github.com/cvg/ZipSplat/tree/main/splatfactory/datasets)。通用流程：

```bash
cd ~/repos/ZipSplat
python -m splatfactory.datasets.scripts.<dataset>.download
python -m splatfactory.datasets.scripts.<dataset>.convert --num-workers 8
```

| 数据 | 转换输入/输出根 | 额外准备 |
|---|---|---|
| RE10K | `data/re10k/360x640` | 缺原始深度：转换后 `extract_depth` |
| DL3DV | `data/dl3dv/540x960` | HF 授权；转换器按实际图像尺寸生成目录，**960P 才对应 540×960**；默认 480P 需另选 `--target-size 252` 等目标输出；转换后补深度 |
| ScanNet++ | `data/scannetpp_v2/720x1080/{dslr,iphone}` | 无自动下载；官方 toolkit 下 DSLR `resized_images`、iPhone `rgb.mkv`、nerfstudio `transforms.json`、`mesh_aligned_0.05.ply`；open3d 渲染深度 + rankings |
| ARKitScenes | `data/arkitscenes/{480x640,640x480}` | 两种方向分开转换；rankings |
| Hypersim | `data/ml-hypersim/768x1024` | 可 `--max-scenes 10`；rankings |
| UnrealStereo4K | `data/unrealstereo4k/540x960` | 相机来自原始包，Q 图本身无相机；rankings |
| BlendedMVS | `data/blendedmvs/576x768` | 可由 zip 转换 |
| CO3D | `data/co3d/720s` | 可先 single-sequence / 指定类别 |
| WildRGB-D | `data/wildrgbd/640x480` | 按类别下载，分卷合并/校验 |
| G-Objaverse | `data/gobjaverse/512x512` | 可按类别与对象数小样下载 |

每个数据转换成 `data/<dataset>/<HxW>/{train,test}-scenes/` 下的场景 tar shards，包含 images、cameras、poses、depth 等；**不能只准备 RGB**。即使配置 rendered-depth L1 权重为 0，GeoCE 等仍需要深度与位姿。

### 5.2 实用的小样命令

```bash
cd ~/repos/ZipSplat

# DL3DV：1K 子集前 50 场景；960P 对应 540×960，先登录且网页同意条款
huggingface-cli login
python -m splatfactory.datasets.scripts.dl3dv.download \
  --sample --resolution 960P --num-workers 4

# 明确选择分辨率的小批下载示例；并非完整 504 复现
python -m splatfactory.datasets.scripts.dl3dv.download \
  --subset 1K --resolution 960P --num-workers 4 --extract

python -m splatfactory.datasets.scripts.hypersim.download --max-scenes 10
python -m splatfactory.datasets.scripts.co3d.download --single-sequence
python -m splatfactory.datasets.scripts.co3d.download \
  --categories hydrant teddybear bench --max-chunks 2 --num-workers 4
python -m splatfactory.datasets.scripts.wildrgbd.download --categories apple car
```

`--sample`、`--max-scenes`、`--single-sequence` 只用于测试下载/转换，不能宣称复现全量训练。先跑一个场景，确认文件模态、许可与 shard 内容，再扩量。

DL3DV 不需要先解压，转换器直接读 zip。小样链路为：

```bash
cd ~/repos/ZipSplat
python -m splatfactory.datasets.scripts.dl3dv.convert \
  --resolution 960P --num-workers 8
python -m splatfactory.datasets.scripts.extract_depth \
  data/dl3dv/540x960/train-scenes
```

若用默认 480P，则转换器会按实际图像尺寸输出（通常 270×480），不会自动得到 `540x960`；此时可另做 `--target-size 252` 输出用于对应 252 配置，而不能只改目录名冒充 960P。

### 5.3 深度、rankings 与降采样顺序

```bash
cd ~/repos/ZipSplat

# RE10K / DL3DV：只为缺深度 scenes 补 DA3 深度；需要 GPU 和 depth-anything-3
python -m splatfactory.datasets.scripts.extract_depth \
  data/re10k/360x640/train-scenes

# ScanNet++ / ARKitScenes / Hypersim / UnrealStereo4K：供 covisibility 采样
python -m splatfactory.datasets.scripts.precompute_rankings \
  data/ml-hypersim/768x1024/train-scenes

# 为节省训练 CPU/IO，预生成 252 版本；先保留原始 shards
python -m splatfactory.datasets.scripts.resize \
  data/dl3dv/540x960/train-scenes \
  --output data/dl3dv/252x252/train-scenes --size 252 --num-workers 32
```

其它数据集把路径替换成表中的实际目录。补深度和 rankings 的更多参数以脚本 `--help` 为准；不要未查参数就给所有数据集套同一条命令。

### 5.4 504 的十个数据集与采样权重

| 数据 | 权重 | 数据 | 权重 |
|---|---:|---|---:|
| RE10K | 0.05 | DL3DV | 0.32 |
| ScanNet++ DSLR | 0.10 | ScanNet++ iPhone | 0.08 |
| ARKitScenes 横屏 | 0.07 | ARKitScenes 竖屏 | 0.06 |
| Hypersim | 0.05 | UnrealStereo4K | 0.02 |
| BlendedMVS | 0.02 | CO3D | 0.10 |
| WildRGB-D | 0.08 | G-Objaverse | 0.05 |

权重和为 1。504 发布模型是**两阶段**：先 252 配方训练 G=16 模型 450K steps，再以它为初始化，在十数据集、长边 504、多宽高比下训练 100K steps；每样本 2–24 context + 4 target。只下载 504 数据而不准备第一阶段 checkpoint，不能复现发布模型。训练命令、损失、checkpoint 和输出见[训练指南](TRAINING_GUIDE_3DGS.md#3-zipsplat唯一有完整官方跨场景训练器的一项)。

## 6. DA3 主干数据：作为背景单列，不冒充 DA3-GS 输入

本仓的 `04` 单场景 3DGS 不需要以下数据；GS-DPT 也仅使用前述 DL3DV。若研究的是 **DA3 几何网络主干本身的预训练**，论文 Table 1 才列出以下 22 项及论文使用场景数：

| 数据 | #Scenes | 数据 | #Scenes |
|---|---:|---|---:|
| AriaDigitalTwin | 237 | AriaSyntheticENV | 99,950 |
| ARKitScenes | 4,388 | BlendedMVS | 503 |
| CO3Dv2 | 30,616 | DL3DV | 6,379 |
| HyperSim | 344 | MapFree | 921 |
| MegaDepth | 268 | MegaSynth | 6,049 |
| MvsSynth | 121 | Objaverse | 505,557 |
| OmniObject | 5,885 | OmniWorld | 1,039 |
| PointOdyssey | 44 | ReplicaVMAP | 17 |
| ScanNet++ | 230 | SceneNet RGB-D | 16,866 |
| TartanAir | 355 | Trellis | 557,408 |
| vKITTI2 | 50 | WildRGB-D | 23,050 |

该表没有独立列出 teacher 的完整训练配方；teacher、monocular student、metric model 又是不同模型/阶段，不能从 DA3 论文任意抓一个数据名就算作 DA3-GS 训练集。复现主干还需论文预处理、教师标签、200K steps、128×H100 等训练条件；本仓并未提供这套训练器。

## 7. 哪些数据没有列进训练清单

- **仅评测/基准**：DA3 的 HiRoom、ETH3D、DTU、7Scenes、Tanks & Temples 与 NVS MegaDepth 子集；InfiniSplat v2 结果中的 DL3DV、ETH3D、NRGBD、Mip-NeRF 360、RE10K、DTU；SplatAnything 的 ScanNet、NRGBD、DL3DV、Tanks & Temples、Mip-NeRF 360、ETH3D、DTU。这些不自动成为对应网络的训练集。
- **本仓自有/自研**：UE53、海思采集、物体三图、佳佳抱枕及单场景 `my_scene`。它们可做独立实验，但不是公开官方训练数据。
- **上游冻结模型的预训练数据**：例如 SplatAnything 的 WorldMirror 2.0 几何教师，不能自动合并成学生属性训练数据表。
- **小样与发布全集**：CO3D single-sequence、DL3DV sample、Human-Shape 十资产等只是链路验证，不是正式全量复现。

## 8. 手动下载前的实际建议

1. **只跑本仓四种方法**：先不下载这些大训练集，按训练指南用当前场景图片跑通推理/单场景优化。
2. **要 ZipSplat 小样训练**：优先 RE10K + DL3DV 小批，先核相机、深度、shard 和 train/test 分离，再扩数据；小样结果不能外推官方配方。
3. **按数据域下载，不盲目全量**：G-Objaverse 可按类，CO3D/WildRGB-D 可按类，Hypersim 可按 scene；Taskonomy/OmniWorld/TartanGround 应按子集和模态选文件。
4. **预留额外空间**：压缩包、解压 raw（若选择解压）、转换 shards、DA3 深度、rankings 与 252 resize 可能同时存在；表中容量不能当作最终占用。先小规模实测放大系数。
5. **许可单独核对**：公开可下载不等于可商用/可再分发。ScanNet++、Matterport3D、DL3DV 需要授权；MapFree、CO3D、OmniWorld 有非商业限制；G-Objaverse 还继承底层资产许可。官方更新时以最新条款为准。
6. **不要从此文档链接转存再发布数据**：它们是官方项目/下载入口，不是镜像授权；尤其 DL3DV 明确有单独访问条款。

### ZipSplat 原始配置/脚本证据

- `~/repos/ZipSplat/splatfactory/configs/data/re10k_dl3dv_252.yaml`
- `~/repos/ZipSplat/splatfactory/configs/data/multi_dataset_504.yaml`
- `~/repos/ZipSplat/splatfactory/datasets/README.md`
- `~/repos/ZipSplat/splatfactory/datasets/scripts/<dataset>/{download,convert}.py`

本文只整理数据与下载路径，未执行大规模下载，也未因此证明任一官方训练已在本机复现。