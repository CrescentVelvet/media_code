# 前馈人像重建：数据集分级清单（S / A / B）

> 四版，2026-11-03。目标：为"少图/单图 → 人像 3DGS 前馈重建"找训练数据。
> 一版只有人体数据集；二版并入 Depth Pro（ICLR 2025, arXiv 2410.02073 v2）附录 Table 15/17 的 19 个训练集；
> 三版加入 8 个模型/材质/HDRI 资产站。**四版改为按"对人像前馈重建的有用程度"统一分为 S / A / B 三级**，
> 不再按数据来源分节 —— 来源相同的两个数据集，价值可能差两个数量级（例：BEDLAM 与 Synscapes 同出 Depth Pro 清单）。
> 每张表统一 **规模 / 相机 / 标注 / 体量 / 门槛 / 下载** 六列。数值来自官方页面；无法核实的标【待核实】。

## 0. 总表（勾选台账）

用法：`☐` → `☑` 表示已下载入库；「存储路径」填相对 `media_code/` 的路径（WSL 内数据写 `/mnt/d/...` 或 `~/output/...`）；「负责人」写人名。
**行序 = 建议投入优先级**（级别内按性价比排），详细参数见第 2~4 节。

| # | 级 | 数据集 | 一句话用途 | 体量 | 门槛 | 已下载 | 存储路径 | 负责人 |
|---|---|---|---|---|---|---|---|---|
| 1 | **S** | HuMMan | 1,000 身份 × 10 视角 RGB-D，跨身份泛化主力 | 724.7 GB（子集） | 登录即下 ✅ | ☐ | | |
| 2 | **S** | SynBody | 10k 合成身份，视角/位姿/深度全可控，补长尾 | 379.2 GB | HF 直下 ✅ | ☐ | | |
| 3 | **S** | GeneBody | 100 人 × 48 视角真实数据，逐帧 SMPL-X | 数 TB（Test10 免费） | Test10 直下；Train40 签协议 | ☐ | | |
| 4 | **S** | DNA-Rendering | 500 演员 × 60 视角 4K + SMPL-X，保真度上限 | 192 GB（Part1）～10.8 TB | 签 PDF + 邮件，每周五批 ⚠️ | ☐ | | |
| 5 | **S** | RenBody | 1,500 ID × 60 视角 4K，规模最大 | TB 级【待核实】 | OpenDataLab 申请 | ☐ | | |
| 6 | **S** | MVP-Human | 400 身份 × 8 视角同姿态扫描，最贴前馈任务定义 | 【待核实】 | Google 表单 ⚠️ | ☐ | | |
| 7 | A1 | THUman4.0 | 3 人 × 24 视角，零审批跑通 + 评测基准 | 119 GB | 直下 ✅ | ☐ | | |
| 8 | A1 | AvatarReX | 4 人 × 16 视角，验跨身份与伪 GT 流程 | 60 GB | 直下 ✅ | ☐ | | |
| 9 | A1 | ActorsHQ | 8 人 × 160 台 12MP，测高分辨率上限 | TB 级 | PI 签 release form ⚠️ | ☐ | | |
| 10 | A1 | ZJU-MoCap | 9 人 × 23 相机，稀疏视角对照组 | 【待核实】 | 签协议 + 邮件 ⚠️ | ☐ | | |
| 11 | A2 | BEDLAM | 10,450 段合成人体，SMPL-X + 深度（**单目**） | 数 TB（有 6fps 子集） | MPI 注册 ⚠️ | ☐ | | |
| 12 | A2 | Dynamic Replica | 1,451 段合成动态场景，带 SMPL + 深度 | 1.8 TB / test 328 GB | 签协议拿 links.json ⚠️ | ☐ | | |
| 13 | A2 | iPER | 30 人单目视频，SMPL，姿态多样性 | 【待核实】 | 学术直下 ✅ | ☐ | | |
| 14 | A2 | GTA-Human | 20K 单目视频 1.4M 帧，量大便宜 | 【待核实】 | 学术直下 ✅ | ☐ | | |
| 15 | A3 | X-Humans | 20 人扫描 **+ 已渲染 RGB-D + 相机参数**，省自渲染 | 【待核实】 | Google Drive 申请 | ☐ | | |
| 16 | A3 | THUman2.0/2.1 | 2,500 扫描，**唯一规模化背面来源** | 【待核实】 | 网格填表换密码（SMPL-X fitting 免申请）⚠️ | ☐ | | |
| 17 | A3 | THuman 3.0 | 20 组服装扫描，衣物褶皱细节 | 【待核实】 | 非商用 | ☐ | | |
| 18 | A3 | MultiHuman | 453 个 1-3 人扫描，**唯一多人遮挡** | 【待核实】 | 非商用 | ☐ | | |
| 19 | A3 | THuman 原始版 | 200 人 × 30 姿态，mesh 与 SMPL 配对 | 【待核实】 | 非商用 | ☐ | | |
| 20 | A3 | 4D-DRESS | 64 套服装 4D 扫描，动态衣物几何 | 【待核实】 | 机构邮箱注册 ⚠️ | ☐ | | |
| 21 | A4 | MVS-Synth | 120 序列 × 100 帧，内外参最干净，**先打通 dataloader** | 9 GB（540p） | 直下 ✅ | ☐ | | |
| 22 | A4 | CO3Dv2 | 19K 环绕视频，真实域多视角 | 8.9 GB（单序列）～5.5 TB | 直下 ✅ | ☐ | | |
| 23 | A4 | BlendedMVS | 113 场景 MVS，**CC BY 4.0 可商用** | 27.5 GB（低清） | 直下 ✅ | ☐ | | |
| 24 | A4 | DL3DV-10K | 10,510 场景 51M 帧，backbone 预训练主力 | 730 GB(480P)～2.8 TB(960P) | HF 申请 | ☐ | | |
| 25 | A4 | RealEstate10K | 10K 视频 SfM 位姿，室内漫游 | 【待核实】 | 学术直下 ✅ | ☐ | | |
| 26 | A4 | WildRGBD | 8,000+ 物体级 RGB-D 环绕 | 【待核实】 | 学术直下 ✅ | ☐ | | |
| 27 | A4 | MVImgNet | 219K 视频 238 类，物体环绕量大 | 【待核实】 | 学术申请 | ☐ | | |
| 28 | A4 | EDEN | 合成花园多视角，位姿精确无噪声 | 38 GB（样本）～1.4 TB | 直下 ✅ | ☐ | | |
| 29 | B1 | Hypersim | 室内几何 + 逐图内外参，深度/法线辅助头 | 1.9 TB（可只取子集） | 脚本直下 ✅ | ☐ | | |
| 30 | B1 | TartanAir | 百万图逐帧位姿，**可商用** | ~3 TB（按环境取） | 直下 ✅ | ☐ | | |
| 31 | B1 | Virtual KITTI 2 | 街景深度，轻量辅助 | 15 GB（RGB+深度） | 直下 ✅ | ☐ | | |
| 32 | B1 | UrbanSyn | 街景深度 EXR，**可商用** | 10-40 GB | 直下 ✅ | ☐ | | |
| 33 | B1 | Synscapes | 25K 街景深度，**禁再分发** | 337 GB | 直下 ✅ | ☐ | | |
| 34 | B1 | UnrealStereo4K | 4K 合成立体，边缘锐度 | 570 GB（1/4 版在 HF） | 直下 ✅ | ☐ | | |
| 35 | B1 | 3D Ken Burns | 合成视差深度 | ~200 GB | 直下 ✅ | ☐ | | |
| 36 | B1 | IRS | 10 万室内立体样本 | 【待核实】 | OneDrive ✅ | ☐ | | |
| 37 | B1 | HRWSI | 21K 真实立体对 | 7.6 GB | 直下 ✅ | ☐ | | |
| 38 | B1 | SAIL-VOS 3D | GTA-V 帧，**须自购游戏** | 【待核实】 | 邮件联系作者 ❌ | ☐ | | |
| 39 | B2 | ScanNet++ | 1,000+ 室内场景，iPhone 视角最贴"手机拍人" | TB 级 | 注册 + token | ☐ | | |
| 40 | B2 | ARKitScenes | 1,661 室内场景 iPhone RGB-D | ~1.5 TB | 同意许可后脚本下 | ☐ | | |
| 41 | B2 | ScanNet v2 | 707 场景 RGB-D，室内背景增广 | ~1.3 TB | 签协议 | ☐ | | |
| 42 | B2 | DIML Indoor | 1,500 帧双目 RGB-D | 2.8 GB（sample） | 链接可疑 ⚠️ | ☐ | | |
| 43 | ❌ | SmartPortraits | 真实人像 + 深度（本可是 S 级唯一） | 【待核实】 | **已失效**，无替代品 | ☐ | | |

**合计**：S 级 6 项约 **1.5 TB 起**（不含 DNA-Rendering 全量 10.8 TB）；A1 跑通三件套（7+8+21）**188 GB / 零审批**；B 级全部只做辅助，按需再取，不必先下。
**当前进度**：0 / 43 已入库。

## 1. 分级判据（不同意判据就不用看表）

按顺序筛四条：

| # | 判据 | 达标线 | 不达标的后果 |
|---|---|---|---|
| ① | **画面里有人** | 人体是主体，或带大量人体级标注 | 无人体 → **最高 B**（只能教几何，教不了人） |
| ② | **同帧多视角 + 标定内外参** | 同一时刻 ≥4 个视角，内参外参可读 | 单目/立体对（≤3 视角）→ **最高 A**；扫描型无相机 → **A**（需自渲染） |
| ③ | **跨身份数** | ≥100 个不同身份 | <100 → **只到 A**（能跑通、能评测，无法支撑泛化） |
| ④ | **GT 可用** | 逐帧 SMPL-X / mesh / 深度直接给 | 缺 → 不降级，但要自造伪 GT（效率损失，质量风险） |

**等级定义**

- **S｜主训练集**：①②③基本全满足。只有这级能同时教"人长什么样"和"多视角怎么一致"，可直接承担主监督。
- **A｜环节级主力**：某一环不可替代，但**不能单独撑起主训练**。四种缺口各成一组：
  A1 多人视角但身份 <100（跑通与评测）｜A2 有人体但视角 ≤3（单图分支与先验）｜
  A3 扫描型无相机（背面补全与几何 GT）｜A4 无人体但多视角（backbone 预训练）。
- **B｜辅助与兜底**：无人体且视角 ≤2，只提供深度/法线/尺度先验，或已经拿不到。

导览：

```
S 级  6 个   DNA-Rendering / RenBody / HuMMan / GeneBody / SynBody / MVP-Human
A 级 22 个   A1 身份少的多视角人体 4｜A2 人体但视角≤3 4｜A3 扫描型 6｜A4 通用多视角 8
B 级 14 个   单目立体深度 10｜真实室内 RGB-D 4（另有 1 个已失效，不计入）
```

## 2. S 级：主训练集（6 个）

判据 ①②③ 全中。共同特征：**同帧 ≥8 视角 + 标定 + 身份 ≥100**。这是唯一能训出跨身份多视角前馈模型的池子。

| 数据集 | 规模 | 相机 | 标注 | 体量 | 门槛 | 下载 |
|---|---|---|---|---|---|---|
| **DNA-Rendering** | 500 演员 / 527 套服装 / 269 日常 + 153 特殊动作 / 25,320 序列 / 67.5M 帧 | **60 同步 RGB**（最高 4096×3000 @15fps）+ 8 Kinect | 标定、matting、2D/3D keypoints、**SMPL-X** | Part1 ~192 GB；全量 ~10.8 TB | 签 License PDF + 机构邮箱，**每周五批**（`luohuiwen@pjlab.org.cn`，抄送 wuwenyan0503 / linjunyi9335） | https://dna-rendering.github.io/ ；代码 https://github.com/DNA-Rendering/DNA-Rendering |
| **RenBody** | 1,500 ID / 5,000+ 序列 | **60 RGB（3000×4096）+ 8 Kinect** | 标定、前后景分割、参数化模型、手工标签 | TB 级【待核实】 | OpenDataLab 申请；CC BY-NC-SA 4.0 | https://opendatalab.com/OpenXD-RenBody |
| **HuMMan** | 1,000 人 / 500 动作 / 400K 序列 / 60M 帧 | **10 台 Azure Kinect**（硬件同步 + 两级标定）+ iPhone | RGB-D、点云、2D/3D keypoints、SMPL、纹理网格 | 子集 **724.7 GB**；全量数 TB | CC BY-NC-SA 4.0，登录可下 ✅ | https://caizhongang.github.io/projects/HuMMan/ ；https://opendatalab.com/OpenXD-HuMMan |
| **GeneBody** | 100 人 / 370 序列 / 2.95M 帧 | **48 视角** | **逐帧 SMPL-X**、前景分割 | 数 TB【待核实】 | Test10 直下；Train40 需签协议 + 机构邮箱换密码（非商用，SenseTime + 上海 AI Lab） | Test10：https://github.com/generalizable-neural-performer/gnr/blob/main/docs/Dataset.md ；全量 https://opendatalab.com/GeneBody |
| **SynBody**（合成） | 10,000 人体模型 / 1,187 动作 / 1.2M 图 | 虚拟多视角（UE5 + XRFeitoria，视角数未标） | SMPL/SMPLX、分层服装、深度/法线/光流/分割 | **379.2 GB** | CC BY-NC-SA 4.0，**直接下** ✅ | https://synbody.github.io/ ；HF https://huggingface.co/datasets/caizhongang/SynBody |
| **MVP-Human** | 400 人 × 15 姿态 × 8 视角 = 6,000 扫描 / 48K 图 | **8 视角静态扫描** | 3D 扫描 + LBS 权重 + SMPL fit | 【待核实】 | Google 表单申请 ⚠️ | 表单 https://docs.google.com/forms/d/e/1FAIpQLSfN2ISu7B1CaO2QvLI1UMZ4YQL-RWZ01jbYtgSSFYc04VSiow/viewform ；代码 https://github.com/TingtingLiao/MVPHuman |

> **与三版的差异**：MVP-Human 从 A 提到 S。理由是它 400 个身份、8 视角同为**同一姿态的同步多视角**，正好命中前馈任务定义，且它就是为"多帧输入"设计的。降 A 的唯一原问题是"静态扫描无动态"，但你要的是静态人像重建，动态不是必要条件。

**S 级内部排序（按投入回报）**

1. **HuMMan** — 唯一"1000 身份 + 登录即下 + 带 RGB-D 与 SMPL"的，最先该拿。
2. **SynBody** — 10k 身份、视角/位姿/深度全可控，直连 HF，补长尾最快。
3. **GeneBody** — 48 视角真实数据，先拿免费 Test10 验域差，再决定要不要签 Train40。
4. **DNA-Rendering** — 保真度最高（60 视角 4K + SMPL-X），但每周五批、全量 10.8 TB，属于"排队 + 分期"。
5. **RenBody** — 规模最大，但公开信息最少，得先确认实际可下的子集体量。
6. **MVP-Human** — 结构最贴合任务，但申请表单的通过率未知，先提交不阻塞。

## 3. A 级：环节级主力（22 个）

### 3.1 A1｜多视角人体，但身份 <100（跑通与评测）

缺 ③。价值在**零审批先跑通**、**做评测基准**、**测高分辨率上限**。

| 数据集 | 规模 | 相机 | 标注 | 体量 | 门槛 | 下载 |
|---|---|---|---|---|---|---|
| **THUman4.0** | 3 人 / 2,500 + 5,060 + 3,110 帧 | **24 标定 RGB**，1330×1150 @30fps | 多视角图、mask、`calibration.json`、`smpl_params.npz` | **~119 GB**（25+58+36） | 非商用，**直下** ✅ | https://github.com/ZhengZerong/THUman-dynamic-Dataset |
| **AvatarReX** | 4 人 / 1,801-2,001 帧 | **16 标定 RGB**，1500×2048 @30fps | 图、mask(pha)、`calibration_full.json`、SMPL-X | **~60 GB**（21+12+11+16） | 非商用，**直下** ✅ | https://github.com/0iui0/AnimatableGaussians/blob/master/AVATARREX_DATASET.md |
| **ActorsHQ** | 8 人 × 2 段 = 16 序列 / **39,765 帧** | **160 台** Ximea 12MP @25fps，快门 650us，420 颗 LED | 逐帧 mesh（50 万顶点）、`calibration.csv`、occupancy grid、SMPL-X | 全分辨率极大（TB 级） | **PI/组长**签 release form + 单位邮箱发 `dataset@actors-hq.com` ⚠️ | https://actors-hq.com/ |
| **ZJU-MoCap** | 9-10 人 / 23 相机（部分子集 4 相机） | 23 同步相机，1024×1024 | 逐帧 SMPL、mask | 【待核实】 | 签协议 + 邮件 ⚠️ | https://github.com/zju3dv/EasyMocap/blob/master/doc/zju_mocap.md |

**用法**：THUman4.0 + AvatarReX 是"第 0 步"（合计 179 GB、零审批、两天可跑通 dataloader）；ActorsHQ 只用来测"160 视角 12MP 时模型还撑不撑得住"，ZJU-MoCap 只在需要 4 相机稀疏设定时用。

### 3.2 A2｜有人体，但视角 ≤3（单图分支与先验）

缺 ②。不能教多视角一致性，但**带逐帧 SMPL-X 的合成人体**是单图分支唯一可用的强监督。

| 数据集 | 规模 | 相机 | 标注 | 体量 | 门槛 | 下载 |
|---|---|---|---|---|---|---|
| **BEDLAM** | **10,450 段**人体序列，1280×720 @30fps | 单目（合成相机轨迹） | **SMPL-X GT(CSV)**、深度 EXR、人体/服装/头发 mask、服装资产 | 数 TB【待核实】（有 6fps 子集先跑） | 自定义非商用，**MPI 注册 + 接受许可** | https://bedlam.is.tue.mpg.de/download.php |
| **Dynamic Replica** | 1,451 序列（合成动态场景） | **2 组立体 + 1 移动视角**，带位姿 | 深度、mask、**SMPL 人体**、3D mesh、相机 | train **1.8 TB** / test **328 GB** | CC BY-NC 4.0，**签协议拿 links.json** | https://github.com/facebookresearch/dynamic_replica |
| **iPER** | 30 人 / 103 套衣服 / 206 视频 / 241K 帧 | 单目视频 | SMPL、背景/前景 | 【待核实】 | 学术，OneDrive/百度 ✅ | https://svip-lab.github.io/dataset/iPER_dataset.html |
| **GTA-Human** | 20K 视频 / 1.4M 帧 | 游戏渲染单目 | SMPL、分割 | 【待核实】 | 学术 ✅ | https://caizhongang.github.io/projects/GTA-Human/ |

**用法**：BEDLAM 是 Depth Pro 19 个训练集里唯一"有人体 + 逐帧 SMPL-X + 深度"的，但**单目**，只能训单图分支或做伪多视角增广；Dynamic Replica 视角数（2-3）不够环绕，但带 SMPL 和深度，做动态一致性辅助损失可以。

### 3.3 A3｜扫描型，无真实相机（背面补全与几何 GT）

缺 ②。**位姿深度完美、可任意采样视角（含背面）**，但贴图已烘焙光照、无真实噪声与遮挡 → 不能当主训练集，是**唯一的背面来源**。

| 数据集 | 规模 | 相机 | 标注 | 体量 | 门槛 | 下载 |
|---|---|---|---|---|---|---|
| **THUman2.0 / 2.1** | 500 → **2,500** 个高质量扫描 | ❌ 无真实相机，需渲染（ICON 脚本出 RGB/Normal/Depth） | obj 网格 + 贴图、**SMPL-X fitting 参数+mesh（免申请，Google Drive 直下）** | 【待核实】 | 原始网格为加密压缩：填 `THUman2.1_Agreement.pdf` 发 liuyebin@mail.tsinghua.edu.cn（抄 ytrock@126.com）换密码 ⚠️ | https://github.com/ytrock/THuman2.0-Dataset |
| **X-Humans** | 20 人 / 233 序列 / 35,427 帧 | 扫描 + **已渲染 RGB-D（附 `cameras.npz` 内外参）** | 贴图 mesh、深度、SMPL/SMPL-X、表情/手部参数 | 【待核实】 | Google Drive 申请 | https://github.com/Skype-line/X-Avatar/blob/main/data_process/README.md |
| **4D-DRESS** | 64 套服装 / 520+ 序列 / 78K 扫描帧 | 4D 扫描（部分含 `Capture/images` + `cameras.pkl`） | 顶点级服装语义、garment mesh、SMPL(-X) | 【待核实】 | 机构邮箱注册 + 详细用途（拒绝"research"一词） ⚠️ | https://4d-dress.ait.ethz.ch/ |
| **THuman 3.0** | 20 组服装 × 15-35 扫描 | ❌ 无真实相机 | 贴图 mesh | 【待核实】 | 与 THUman2.x 同源，非商用 | https://github.com/fwbx529/THuman3.0-Dataset |
| **MultiHuman** | 453 个 1-3 人扫描 | ❌ 无真实相机 | 贴图 mesh、多人接触 | 【待核实】 | 非商用 | https://github.com/y-zheng18/MultiHuman-Dataset/ |
| **THuman 原始版** | 200 人 × 30 姿态 = 6k 扫描 | 单 RGB-D + SMPL | mesh、SMPL | 【待核实】 | 非商用 | https://github.com/ZhengZerong/DeepHuman/tree/master/THUmanDataset |

**用法**：X-Humans 是这批里唯一**已经把渲染结果和相机参数一起给出**的，省掉自己搭渲染器；MultiHuman 是唯一能造"多人互相遮挡"的；THUman2.x 的 2,500 扫描是补背面和做评测 GT 的主力。

### 3.4 A4｜无人体，但多视角（backbone 预训练）

缺 ①。**对人像本身零贡献**，但"多视角一致性"这个能力可以在这里预训练，再到 S/A1 上 finetune。DepthSplat / VGGT 一类工作都走这条路。

| 数据集 | 规模 | 相机 | 标注 | 体量 | 门槛 | 下载 |
|---|---|---|---|---|---|---|
| **DL3DV-10K** | 10,510 场景 / **51.2M 帧** / 65 类 POI | 真实拍摄，COLMAP 位姿 | 位姿、场景属性标签、3DGS/Nerfstudio 格式 | 480P **730 GB**｜960P **2.8 TB**｜2K **11 TB**｜4K **44 TB** | HF 申请访问即同意条款；可分片脚本下 | https://huggingface.co/datasets/DL3DV/DL3DV-ALL-960P ；脚本 https://github.com/DL3DV-10K/Dataset |
| **CO3Dv2** | 50 类 / ~19K 视频 / 1.5M 图 | 真实环绕拍摄，逐帧位姿 + 内参 | 前景 mask、点云、相机 | 全量 **5.5 TB**；单序列子集 **8.9 GB** | CC BY-NC 4.0 ✅ | https://github.com/facebookresearch/co3d |
| **MVImgNet** | 219,188 视频 / 6.5M 帧 / 238 类 | 真实环绕拍摄，相机参数 | 物体 mask、相机参数、点云 | 【待核实】 | 学术申请【待核实】 | https://gaplab.cuhk.edu.cn/projects/MVImgNet/ |
| **RealEstate10K** | 10K 视频 / ~80K 片段 | 真实，YouTube SfM 位姿 | 相机内参 + 逐帧位姿 | 【待核实】 | 学术 ✅ | https://google.github.io/realestate10k/download.html |
| **WildRGBD** | 8,000+ 物体级序列【待核实】 | 真实 RGB-D 环绕 | 深度、位姿、mask | 【待核实】 | 学术 ✅ | https://github.com/facebookresearch/WildRGBD |
| **BlendedMVS** | 113 场景 / ~17K 图 | **多视角 MVS**，COLMAP 位姿 | 深度、点云、mesh | 低清 **27.5 GB** / 高清 **156 GB** / MVG 全套 ~190 GB | CC BY 4.0（**可商用**） | https://github.com/YoYo000/BlendedMVS |
| **MVS-Synth** | 120 序列 × 100 帧 | **多视角**，JSON 内外参 | 深度 EXR、位姿（注意 R 行列式 **-1**） | 1080p **34 GB** / 720p **16 GB** / 540p **9 GB** | 仅研究与教育 | https://huggingface.co/datasets/phuang17/MVS-Synth |
| **EDEN** | 合成花园场景（多视角渲染） | **虚拟多视角**，位姿精确 | 深度、法线、分割 | 全量 **~1.4 TB**；样本集 **~38 GB** | 页面未标许可【待核实】 | https://lhoangan.github.io/eden/ |

**用法**：先拿 MVS-Synth 540p（9 GB）打通 dataloader —— 它的 JSON 内外参最干净；再上 CO3Dv2 单序列子集（8.9 GB）验真实域；真要预训练 backbone 才动 DL3DV 960P（2.8 TB）。BlendedMVS 是这里唯一 CC BY 4.0 可商用的多视角数据。

## 4. B 级：辅助与兜底（14 个 + 1 个失效）

缺 ①（无人体）且缺 ②（≤2 视角）。**只提供深度/法线/尺度先验，或补室内背景分布**；不要指望它们教人像。

### 4.1 B1｜单目 / 立体深度（辅助损失）

| 数据集 | 规模 | 相机 | 标注 | 体量 | 门槛 | 下载 |
|---|---|---|---|---|---|---|
| **Hypersim** | 366 场景 / 77,400 图，1024×768 | 虚拟相机，**逐图内参+外参** | 深度、法线、instance/semantic、光照参数 | **1.9 TB** | CC BY-SA 3.0（**传染性**） | https://github.com/apple/ml-hypersim |
| **TartanAir** | 30+ 环境，100 万+ 图 | 双目针孔/鱼眼，**逐帧位姿** | 深度、分割、光流、LiDAR、IMU | **全量 ~3 TB**（可按环境取） | CC BY 4.0（**可商用**） | https://tartanair.org/ ；旧工具 https://github.com/castacks/tartanair_tools |
| **UnrealStereo4K** | 8 场景 / 4K 合成立体 | 立体对（UE4 渲染） | 视差/深度、位姿 | 8 场景各 67-76 GB ≈ **570 GB**；1/4 分辨率版在 HF | 自定义（学术） | https://github.com/fabiotosi92/SMD-Nets |
| **Synscapes** | 25,000 张，1440×720 | 单目车载（附内外参） | 语义/实例分割、深度、3D 场景 | **337 GB**（41 个 5 GB 分片） | 自定义：**非商用、禁止再分发** | https://synscapes.on.liu.se/download.html |
| **Virtual KITTI 2** | 5 序列 / 21,260 帧，1242×375 | 单目车载（虚拟），逐帧位姿 | 深度、语义/实例、光流 | RGB **7.0 GB** + 深度 **7.6 GB**（全模态 ~100 GB） | CC BY-NC-SA 3.0 | https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/ |
| **UrbanSyn** | 7,534 张，2048×1024 | 单目单帧，**未提供多视角** | 语义、深度 EXR、panoptic、2D bbox | 【待核实】（7.5k 图 + EXR，量级 10-40 GB） | CC BY-SA 4.0（**可商用**） | https://huggingface.co/datasets/UrbanSyn/UrbanSyn |
| **3D Ken Burns** | 分场景合成深度 | 单目 + 合成视差 | 深度 | **~200 GB** | CC BY-NC-SA 4.0 | https://github.com/sniklaus/3d-ken-burns |
| **IRS** | 室内合成立体样本 103,316【待核实】 | 立体对 | 视差/深度、位姿 | 【待核实】 | 未明示（学术） | https://github.com/HKBU-HPML/IRS |
| **HRWSI** | ~21K 立体对 | 立体对（网络采集） | 视差 | **7.6 GB** | 自定义（学术） | https://github.com/kexianhust/RealSR-Ranking |
| **SAIL-VOS 3D** | 237,611 帧（GTA-V） | 单目游戏渲染 | 实例 mask、SMPL、深度 | 【待核实】 | ❌ 非商用 + **须自购 GTA V** + 邮件联系作者 | https://sailvos.web.illinois.edu/ |

### 4.2 B2｜真实室内 RGB-D（背景与遮挡域增广）

| 数据集 | 规模 | 相机 | 标注 | 体量 | 门槛 | 下载 |
|---|---|---|---|---|---|---|
| **ARKitScenes** | 5,047 段 / 1,661 室内场景 | iPhone RGB-D + IMU，**逐帧位姿** | 深度、3D 标注（家具/窗/门）、平面 | TB 级（raw ~1.5 TB）【待核实】 | Apple 自定义许可（分商用/非商用），同意后脚本下 | https://github.com/apple/ARKitScenes/blob/main/DOWNLOAD.md |
| **ScanNet v2** | 707 场景 / 1,513 扫描 / 2.5M 视图 | RGB-D 手持，COLMAP 位姿 | 深度、语义/实例、CAD 模型 | **~1.3 TB** | 签协议（官网表单） | https://github.com/ScanNet/ScanNet |
| **ScanNet++** | 1,000+ 室内场景（v2），DSLR 33MP + iPhone | 真实，亚毫米激光扫描配准 | 位姿、深度、语义（long-tail）、3DGS demo | TB 级【待核实】 | 注册 + 提交申请 + 个人 token | https://kaldir.vc.in.tum.de/scannetpp/ |
| **DIML Indoor** | sample 1,500 帧 / 2.8 GB | 双目 RGB-D 手持 | 深度、位姿 | 样本 2.8 GB；全量更大 | 非商用；**Naver 短链状态可疑** | https://dimlrgbd.github.io/ |

**B2 用法**：人像前馈绕不开背景与遮挡——训练时把人体合成到这些室内场景里，比单纯抠人贴纯色背景的泛化好得多。ScanNet++ 与 ARKitScenes 的 iPhone 视角分布最接近"手机拍人"的真实输入。

### 4.3 已失效（占位，别再花时间）

| 数据集 | 原定位 | 现状 |
|---|---|---|
| **SmartPortraits** | 1,000 序列 / 200 人 **真实人像 + 深度**（本可是 S 级唯一的"真人像深度"） | ❌ toolkit 仓库已 archive，下载链接写着 "TODO"；https://augmentedperception.github.io/smartportraits/ |

**这是整份清单里最可惜的一条**：真实人像 + 逐像素深度这个组合，目前公开数据里没有替代品。要么自采，要么用 BEDLAM → real 的域适应。

## 5. 为什么 Depth Pro 那 19 个几乎全落在 B

它的训练协议是"单图度量深度"，判据 ①②③ 全部不达标：

| Depth Pro 分组 | 数量 | 落到哪一级 | 原因 |
|---|---|---|---|
| 合成带度量深度 | 9 | B（BEDLAM 除外，A2） | 街景/室内，无人体，单目 |
| 真实 RGB-D | 4 | B2（SmartPortraits 失效） | 室内场景，无人体 |
| 立体 / 非度量 | 4 | A4（MVS-Synth、EDEN）+ B | 只有这两个是真多视角 |
| 仅结构相似 | 2 | A4（BlendedMVS）+ B | 只有 BlendedMVS 是真多视角 |

**唯一进 A 的两个人视角数据集**：BEDLAM（A2）、Dynamic Replica（A2）；**唯一进 S 的没有**。
所以这份清单的正确用法是：**给 backbone 加深度/法线辅助头 + 补室内背景分布**，不要当人像主训练集。

另外，它的训练协议（Stage 1 250ep 全部标注集 MAE + SSI-MAGE → Stage 2 100ep 加一阶/二阶导数损失提边缘）值得抄；焦距头单独训练，论文只说"用了多个数据集 + 仅焦距监督的大规模图像集"，未逐项列明【待核实】。

## 6. 训练硬约束（写 DESIGN 前先看）

1. **不要跨姿态混帧**。一个样本 = 同一帧的 N 个视角。跨帧混训会把两种姿态塞进同一空间——就是 som_human 里"垂手 vs 抬手 → 幽灵/半透明"的同一机理。
2. **输入视角数决定任务难度**。单图 → 必须带 SMPL-X/MHR 强先验（A2/A3 上场）；4-8 图 → 主流设定（S 级全部可用）；>12 图 → 退化成 MVS，学不到"补全"。
3. **身份数比帧数重要**。这就是 ③ 判据的来源：THUman4.0 有 2,500+5,060+3,110 帧但只有 3 个人，只能评测；SynBody 10,000 身份才是泛化来源。
4. **GT 两条路**。①数据集自带（DNA-Rendering、HuMMan、ActorsHQ、GeneBody、BEDLAM、MVP-Human）；②现成 3DGS 逐场景优化当伪 GT（THUman4.0、AvatarReX 走这条，注意伪 GT 会把过平滑一起学进去——正是 som_human 已经踩过的坑）。
5. **评估口径**。mask 内 FG PSNR 抓不到幽灵/透明（幽灵落在 GT mask 之外），必须配 mask 外泄漏指标或 alpha 渲染。
6. **许可红线**。S/A 级人体数据**全部非商用**（含合成的 SynBody）；Hypersim 的 CC BY-SA 3.0 有传染性，Synscapes 禁止再分发；真正可商用的只有 TartanAir / BlendedMVS / UrbanSyn。商用模型需另找付费扫描资产（如 RenderPeople）。

## 7. 落地顺序（按级别排）

```
第 0 步 · 跑通（~190 GB，零审批，1-2 天）
  A1  THUman4.0          119 GB   24 视角 × 3 身份，先验 4-8 图输入 → 3DGS
  A1  AvatarReX           60 GB   16 视角 × 4 身份，验证跨身份
  A4  MVS-Synth 540p       9 GB   JSON 内外参最干净，打通 dataloader

第 1 步 · 上主训练集（~1.1 TB，登录/直下）
  S   HuMMan             724 GB   1000 身份，练跨身份泛化（10 视角 RGB-D）
  S   SynBody            379 GB   10k 身份，视角/位姿/深度可控，补长尾
  S   GeneBody Test10      ~?     48 视角真实数据，先验域差再决定签 Train40

第 2 步 · 补两个缺口
  A3  X-Humans             ~?     唯一已渲染好并给相机的扫描数据 → 背面
  A3  THUman2.1           ~?     2,500 扫描，SMPL-X fitting 免申请 → 几何 GT
  A2  BEDLAM 6fps          ~?     唯一"人体 + SMPL-X + 深度"，单图分支
  B2  ScanNet++ / ARKitScenes     真实室内背景与遮挡增广

第 3 步 · 排队与预训练（越早越好）
  S   DNA-Rendering Part1 192 GB   60 视角 4K + SMPL-X，每周五批
  S   RenBody / MVP-Human          规模最大 / 结构最贴，先递申请
  A4  DL3DV 960P          2.8 TB   只在要预训练 backbone 时动用
  A1  ActorsHQ                    160 台 12MP，测分辨率上限
```

## 8. 资产库：把 A3 扩增成 S 级规模的杠杆

**先定性**：这 8 个站（+2 个物体库）提供的是**资产**，不是数据，**不参与 S/A/B 分级**。
它们的定位很清楚：**S 级人体数据负责"人"，资产站负责"人所在的世界"**；而 A3 扫描型数据一旦配上资产站，就能渲染出"带环境、带随机光照的多视角序列"——这是把 A3 抬到接近 S 可用度的唯一低成本手段。

```
输入：S/A1 的「同一帧 N 视角 RGB + mask + SMPL-X」，或 A3 的「扫描 mesh + 贴图」
  ↓ 抠人 → 带 alpha 的人体层
  ↓ Blender + BlenderProc：
      环境 = ambientCG 材质搭的 box 房间 或 Poly Haven HDRI 穹顶
      光照 = 随机 HDRI 旋转 + 强度扰动（关键：A3 贴图里已烘焙光照，必须重打）
      相机 = 环绕球面采样（方位全覆盖，仰角 -20°~+40°）
      干扰 = 随机道具（Objaverse/GSO）+ 遮挡 + 运动模糊 + 噪声
  ↓ 渲染：RGB + 深度 + 法线 + mask
输出：可无限扩增的多视角序列
```

### 8.1 通用资产站

| 站点 | 内容 | 规模 | 许可 | 能否商用 | 下载 / API |
|---|---|---|---|---|---|
| **Poly Haven** | HDRI / 无缝 PBR 材质 / 模型 | 三类持续更新（官方未标总量） | **全部 CC0** | ✅ | 免注册直下；API `api.polyhaven.com` |
| **ambientCG** | PBR 材质 / HDRI / 少量模型 / Substance | 数千套 | **全部 CC0** | ✅ | 免注册直下；有 API 与批量下载 |
| **3DTextures.me** | 无缝 PBR 纹理 | 数百套 | **全部 CC0** | ✅ | 逐张下（全量打包需赞助） |
| **shareTextures** | 纹理 / 模型 / atlas | **1,600+ 纹理、250+ 模型、50+ atlas** | **CC0** | ✅ | 免注册直下，最高 4K |
| **Blend Swap** | .blend 工程（模型+材质+灯光+节点） | 社区数万 | 逐资源：CC0 / CC-BY / CC-BY-SA / CC-BY-NC | 仅 CC0、CC-BY | 免费注册，网页下 |
| **Sketchfab** | 模型（含 3D 扫描、人体、道具） | 百万级；可下载需作者开启 | 逐模型：CC0 / BY / BY-SA / BY-ND / BY-NC；商店授权已迁 Fab | 仅 CC0~BY | 需账号；Data API v3（OAuth），ToS 禁爬 |
| **RenderHub** | 市场：模型 / 纹理 / HDRI / 扫描 | 免费区数千 | 两档：Personal Use Only / Extended Use，**均禁转售再分发** | 仅 Extended Use | 注册下载，付费为主 |
| **freePBR** | PBR 材质 + 少量模型 | 600+ 套 | ⚠️ **许可页已 404**，免费单张条款含糊；$19 全库包明示含 commercial rights | 仅买包稳妥 | 免注册逐张下 |

### 8.2 真正有"量"的物体库

| 资产库 | 内容 | 规模 | 许可 | 下载 |
|---|---|---|---|---|
| **Objaverse** | 通用 3D 物体（含大量扫描与生成体） | **~800K 物体**；XL 版号称 10M+ | **逐物体混合** CC-BY / CC0，必须按元数据过滤 | HF `allenai/objaverse` |
| **Google Scanned Objects (GSO)** | 真实扫描家居物品 | 1,030 个 | CC BY 4.0 | https://app.gazebosim.org/GoogleResearch/fuel/collections/Scanned%20Objects%20by%20Google%20Research |

### 8.3 三条使用纪律

1. **训练数据只碰 CC0 四站**：Poly Haven / ambientCG / 3DTextures.me / shareTextures —— 无权利保留，可进训练集、可随模型发布。
2. **CC-BY 留署名清单**：Blend Swap、Sketchfab、GSO、Objaverse 的 CC-BY 资产入训时保留 `asset_id → 作者 → 许可 → URL` 的 CSV。
3. **"可商用" ≠ "可训练"**。RenderHub Extended Use、freePBR 单张、Sketchfab 商店授权都只授予"用于创作"，**没有一条明示允许 AI 训练**；训练模型再发权重，实质是在分发"从这些资产学到的能力"。保守做法：训练集只用 CC0 / CC-BY，商用资产只做定性演示。

---
附：一版调研（各数据集门槛、申请邮件要点）见 `.workbuddy/memory/2026-11-03.md`。
