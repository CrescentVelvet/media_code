# som_human — SoM 式低秩运动基替换形变 MLP（手部微动方案 Phase 1）

把 Deformable-3D-Gaussians 的**逐帧独立形变 MLP** 替换为 **Shape of Motion 式的
逐帧 SE(3) 运动基凸组合**（B 条全局基 + 空间 MLP 系数场），治疗单目动态人体
重建中手部「重影 / 撕裂 / 漂浮」的**跨帧不一致**问题（成因③）。

方案设计与选型依据：`media_paper/3D-20261012-手部微动重建解决方案.html`
（分层混合架构 Phase 1）；方法来源：`media_paper/3D-20240718-Shape-of-Motion.html`。
渲染 / 光栅化 / 训练循环 / 数据准备全部复用 deformable_human 基建（含 COLMAP
场景与 env），本目录只有运动模型 + 训练/渲染 fork 两个新文件。

## 常用命令
> 假设 WSL 已配好 deformable_human（env + 官方仓 + 场景产物）。
> **铁律：每条命令显式写出路径，不全靠默认值。**

```bash
cd /mnt/c/code/media_code

# ── 0) 环境体检（不新建 env，复用 deformable_human）──
bash som_human/00a_setup_env.sh

# ── 1a) 数据（复用）：deformable_human 的 01 产物 ──
#    SOURCE_PATH 默认指向 ~/output/deformable_human_results/<scene>/colmap_scene

# ── 1b) 数据（新采集包）：手机 App 采集（transforms.json + image/*.heic + pcd.ply）──
#    自带位姿/内参/点云，免 COLMAP。位姿约定 = c2w OpenGL（脚本内投影自检把关）
SCENE_NAME=double_wave \
RAW_DIR="/mnt/d/dataset/手部微动样例/f5cf3ba0d4014ce79a82fcaf7ecae053-双人招手" \
  bash som_human/01_convert_capture.sh

# ── 1c) 位姿精化（手机采集场景强烈建议，治 VIO 漂移根因）──
#    01b 诊断：App VIO 位姿逐帧 inlier 摆 250 倍 → PSNR 钉死 13~14。
#    对 01 的去畸变图从零跑 COLMAP SfM（GPU SIFT+exhaustive+mapper，内参固定），
#    产出 colmap_refined/ 场景（位姿+三角化点云都换掉，图像软链复用）。
SCENE_NAME=double_wave bash som_human/01c_colmap_refine.sh
#    精化质量校验（inlier 分布/重投影误差/VIO 轨迹对齐残差 + 叠加图）：
python som_human/01d_check_refined.py \
  ~/output/som_human_results/double_wave/colmap_refined \
  ~/output/som_human_results/double_wave/colmap_scene

# ── 2) SoM 运动基训练（Phase 1 核心）──
GPU=0 \
SCENE_NAME=human_seq \
SOURCE_PATH=$HOME/output/deformable_human_results/human_seq/colmap_scene \
MODEL_PATH=$HOME/output/som_human_results/human_seq/model \
ITERATIONS=20000 NUM_BASES=10 \
EXTRA_TRAIN_ARGS='-r 2 --densify_grad_threshold 0.0004 --densify_until_iter 2000' \
  bash som_human/02_train.sh

# ── 3) 渲染 + 评测 ──
# MODE=original 出时间×视角组合渲染——目视检查重影用它（验收主指标）
GPU=0 SCENE_NAME=human_seq MODE=render RUN_METRICS=1 \
  bash som_human/03_render.sh
GPU=0 SCENE_NAME=human_seq MODE=original \
  bash som_human/03_render.sh

# ── 4) 搬运到 D 盘 ──
SCENE_NAME=human_seq bash som_human/08_move_output.sh
```

## 输入数据格式（标准约定，2026-10-13 起）

今后的输入数据**统一为华为手机采集包**（Remy 类 App 导出，样例见
`/mnt/d/dataset/手部微动样例/`）：

```
<采集包>/
├── transforms.json   # 逐帧相机参数（meta 里带 platform=Harmony）
├── image/*.heic      # 逐帧 HEIC 图片（文件名=时间戳，字典序即时间序）
└── pcd.ply           # 场景点云（binary_little_endian，float xyz + uchar rgb）
```

`transforms.json` 逐帧字段：

| 字段 | 含义 |
|---|---|
| `w, h` | 图像尺寸（实测 1440×1920 竖屏，HEIC 解码后一致、无 EXIF 旋转问题） |
| `cx, cy, fl_x, fl_y` | 针孔内参（像素），**全序列一致**（单相机） |
| `k1, k2, k3, p1, p2` | OpenCV 畸变系数 |
| `file_path` | 指向 `image/` 的相对路径 |
| `transform_matrix` | 4×4，**c2w + OpenGL 相机系**（y 上 z 朝后） |

实测约定（投影正深度率四候选判定，勿凭直觉改）：

- 位姿 = c2w OpenGL → OpenCV w2c 的转换式：**`w2c = inv(M @ diag(1,-1,-1,1))`**
  （c2w_opengl 正深度率 99%，c2w_opencv / w2c_opencv / w2c_opengl 全被否决）
- 点云含离群飞点（实测 z 到 -20 m），转换时按「到相机中心距离」分位过滤
  （`KEEP_FRAC` 默认 0.90）
- D-3DGS 光栅化不建模畸变（只认 PINHOLE），**cv2 去畸变 + ROI 裁剪是强制步骤**，
  内参同步更新

转换入口：`01_convert_capture.sh`（HEIC 解码 → 去畸变 → 整数重命名 →
COLMAP txt 三件套 + 末尾投影自检，自检不过会直接退出）。

整批跑：`bash som_human/hand_batch.sh all`（对 `RAW_ROOT` 下每个采集包串跑
01 → 01c → 02；`PAR=2` 控并发、幂等跳过已产出的阶段）。
2026-10-16 实测 10 个包（共 1382 帧）：01 用 12 min ｜ 01c 用 60 min（全部帧注册）
｜ 02（10k + `-r 2`）用 100 min，test PSNR 21.3~27.7、高斯 53~63 万/场景。

## 首次准备
训练/渲染无新增安装。前置 = deformable_human 全流程已跑通（见其 README_wsl.md）。
`00a_setup_env.sh` 只做体检：env 存在 / 官方仓在位 / fork 光栅化器可 import /
som_motion 自检。

**01c 位姿精化需要 COLMAP（独立 env，CUDA 版）**：
```bash
conda create -n colmap --override-channels -c conda-forge 'colmap=*cuda*' -y
# ⚠️ conda-forge colmap 4.2.1 配方漏声明 openimageio 且 ABI 锁 3.1，必须补钉：
conda install -n colmap --override-channels -c conda-forge 'openimageio=3.1' -y
```
01c 脚本会自动发现 `~/miniconda3/envs/colmap/bin/colmap` 并补 LD_LIBRARY_PATH。
另注意 COLMAP 4.x 要求 mapper 的 output_path 目录已存在（3.x 会自动建，脚本已处理）。

## 位姿精化（治手机 VIO 漂移，2026-10-13 实证为手机场景第一瓶颈）

**根因链**（double_wave 三轮实验 + 诊断）：

| 实验 | 配置 | test PSNR |
|---|---|---|
| v1 | 21.6 万点，20k，VIO 位姿 | 14.04 |
| v3 | 147 万点（撞护栏），@7000 | 12.98 |
| 静态消融 | WARM_UP=20000（运动模型不激活） | 13.40 |

点数 ×7、有无运动模型都无效 → 瓶颈不在容量/运动模型。`01b_check_reproj.py`
把 pcd 投回各帧：inlier 436~112750 **摆幅 250 倍** → App VIO 位姿漂移。
与 deformable_human 证伪清单「手机 VI 位姿直接替 COLMAP −4dB」互证。

**01c 精化结果**（double_wave，GPU SIFT，4m49s 全流程）：

- **160/160 帧注册**（mapper 一个子模型，无断裂），三角化点 53,635；
- 每帧 inlier 1899~4504（max/min **2.4×**，对照 VIO 的 250×）；
- 重投影误差中位 **0.92px**（p95 1.96，max 3.98）；
- VIO→COLMAP 轨迹相似对齐：scale 2.537，位置残差中位 1.59（COLMAP 单位，
  ≈0.63m VIO 单位）、旋转残差中位 47°——VIO 漂移幅度的直接量化；
- 精化点云叠加图：点坐在人/地面正确结构上（`colmap_refined/reproj_check/`）。

训练侧收益见「实验记录」节 ①：同 10k 迭代下 **13.978 → 21.265（+7.29dB）**。

## 按人独立重建（per-person，多人场景，2026-10-17）

多人采集包（双人招手、三人聚餐）整场景一起重建有三个结构性问题，都会直接
吃掉手部细节：

1. **容量竞争**：`MAX_GAUSSIANS` 是全场景共享，背景与其他人先占满，目标人的
   手只分到零头。
2. **运动基竞争**：10 条 SE(3) 基要同时表达多个人的运动，每个人的低秩表达
   都被稀释。
3. **评测被背景稀释**：PSNR 里背景占 ~90% 面积，手糊了也看不出来（hand10
   整场景 21.29dB 看着"还行"，其实是背景在撑）。

per-person 把「一个场景」拆成「每人一个独立模型」：各自独立的容量与运动基，
互不竞争；评测只在本人 mask 内算。

### 链路

```bash
# 1) 逐人 mask（SAM3 video 模式，obj_id 跨帧一致）
SCENE_NAME=hand10_shuangren_zhaoshou bash som_human/01g_person_masks.sh

# 2) 按 pid 拆场景（帧覆盖率 ≥50% 的才是真人）
SCENE_NAME=hand10_shuangren_zhaoshou bash som_human/01h_build_person_scenes.sh

# 3) 每人一个模型（FG_MASK_DIR 指向该人的 masks/）
SCENE_NAME=hand10_p02 SOURCE_PATH=<scene>/persons/p02 MODEL_PATH=<out>/model_p02 \
FG_MASK_DIR=<scene>/persons/p02/masks \
bash som_human/02_train.sh

# 4) 前景指标（mask 内 PSNR，背景不参与）
python som_human/03b_fg_metrics.py --scene <scene>/persons/p02 --model <out>/model_p02
```

### 三个必须注意的点（都是实测踩出来的）

1. **只能 video 模式**：`MASK_MODE=image` 每帧实例 ID 按检测分数排序、跨帧
   不对应 → 同一人 ID 每帧都变，拆分会把一个人拆成几十个碎片。
2. **text prompt 会把背景物体误检成 person**：hand10 实测 27 个 track，只有
   2 个是真人（160/160 帧、占画面 10.2%/10.8%），其余 25 个面积 0.01~1.3%、
   与真人 mask 的 **IoU 恒为 0**（桌椅之类的误检）。01h 因此用**帧覆盖率
   ≥50% + 平均面积 ≥0.5%** 双重过滤——不能只看面积（实测有个 1.26% 的误检
   覆盖了 36% 帧，只按面积会漏进来）。
3. **初始点必须过滤**：loss 只算 fg 后，落在背景上的 COLMAP 点**永远没有梯度**
   （既不被 densify 强化、也不会被 prune 掉），会一直占容量并在合成时浮在
   人体周围。`FG_INIT_FILTER=1` 按多视角 mask 投票过滤（实测 53,676 → 13,925
   点，25.9%）。

### 实现要点

- **面积归一化**：`Ll1` 只算 mask 内，再乘 `numel/area` 恢复到全图量级 ——
  不归一化的话 loss 按人体占比（~10%）缩小，而 `densify_grad_threshold` 是
  固定值，稠密化会整体失准。
- **随机背景**：`BG_RANDOM=1` 每步随机背景色，防止模型把背景色学进 SH；
  评测/存档用固定色（`eval_background`）。
- **mask 尺寸自动对齐**：按 `gt_image` 的 H/W 做 bilinear 缩放（`-r 2` 降采样
  后仍对齐）。
- **`max_radii2D` 必须同步重建**：DG 在 `create_from_pcd` 末尾按原点数建它，
  `training_setup` 不重建 → 过滤后不同步会在第一步 `IndexError`。

### 头身分离（设计，未实测）

同一套机制再套一层：`masks/` 换成 head mask（SAM3 `prompt=head`，或
`person mask ∩ 3DMM 头包围盒`），`FG_MASK_DIR` 指过去就是 head_gs；body_gs
用 `person mask − head mask`。两者共享同一份 COLMAP 位姿与运动基配置，
合成时直接叠加渲染（都是同一世界坐标系下的独立高斯集合）。

### 实测（hand10 双人招手，2026-10-17）

| 模型 | 点云 初始 → 最终 | 训练耗时 | FG PSNR（该人 mask 内） |
|---|---|---|---|
| 整场景（基线） | 53,676 → 608,400 | 635 s | p02: 24.492 / p07: 24.497 |
| per-person p02 | 14,594 → 103,669 | ~320 s | **24.343**（−0.15，噪声内） |
| per-person p07 | 13,189 → 115,896 | ~320 s | **25.193**（+0.70） |

- **资源**：点数与体积都降到 **~1/6**（10.4~11.6 万 vs 60.8 万），训练速度从
  ~3 it/s 提到 **~32 it/s**（初始点少了 4 倍，非渲染开销随点数线性）。
- **质量**：FG PSNR 与整场景持平/略升；**逐视角最差帧 per-person 更好**
  （p02 的 #1 帧 14.9 vs 基线 13.8，+1.1dB）。全图 PSNR 10 vs 21 不是退化，
  是背景本来就该是黑（背景误差不该进指标）。
- **⚠️ 遗留缺陷**：montage 可见人体边缘有**黑色飞溅状飞点**（手/头周围
  尤甚）。来源：①mask 边缘羽化区仍产生梯度；②双人贴身时 mask 会带对方
  的衣角；③初始过滤留了 24%（含两人重叠区的点）。后续可做：mask 腐蚀、
  训练中按 mask 周期性 prune、提高 init ratio。
- **结论**：可行，且点数/耗时显著省；视觉毛刺需下一轮工程处理，但**不是
  per-person 架构引入的**——双人招手场景整场景重建在 fg 区域本来就糊
  （worst 帧 13.8）。

## 曝光补偿（方案 A，`exposure.py`，默认关闭）

针对手机 auto-exposure 逐帧光度漂移（户外场景 GT 亮度/色温不一致系统性干扰
几何学习）。移植自服务器 Reconstruction 的 ExposureModule（BACKLOG 方案 A）：

- 每训练相机一个可学习 **4×3 齐次仿射** A（`img_adj = Aᵀ[render;1]`，恒等初始化）；
- **门控**：`iteration < EXPOSURE_FROM_ITER`（默认 = WARM_UP）前 A 冻结为恒等；
  门控开启瞬间对全部训练帧渲染一遍做 **lstsq 闭式初始化**（含 2×中位数抗外点重解）；
- 之后 A 以独立 lr（默认 1e-3）联合训练，L1/DSSIM 用同一张校正图（口径一致）；
- 正则 `0.01·Σ|A−A_eye|` 拉回恒等，防与 SH 颜色互相吸收震荡；
- test 帧没训过 A：按 fid 与相邻训练帧 **线性插值**，评测同口径 apply；
- 权重随 ckpt 存 `model_path/exposure/iteration_N/exposure.pth`，
  render_som 自动检测并应用（time/original 插帧模式不加，只看运动重影）。

⚠️ **渲染口径坑（2026-10-16 修，A 案复核时踩到）**：
`ExposureBank.load()` 与 `SoMMotionModel.load_weights()` 的默认参数都是
**「取最大迭代号」**，而 render_som 原先两处都没传 `--iteration` 对应的迭代号
→ **渲染旧 ckpt 时会套上最新迭代的运动基/曝光权重**（实测：`model_colmap_exp`
渲染 10k 时套了 20k 的运动权重，虚高 **+0.49dB**）。现均改为按 `scene.loaded_iter`
配对加载，曝光权重缺失时回退到「≤ 目标迭代的最大者」并告警。
**渲染非最新 ckpt 前务必核对日志**：`✅ [SoM] 加载运动基权重: .../iteration_N/som.pth`
的 N 是否等于目标迭代。

开启：`USE_EXPOSURE=1 bash som_human/02_train.sh`（**实测无收益，默认关**，见实验记录 ②）。

## 实验记录（按日期倒序，负结果也记录）

### 2026-10-16 ② 曝光补偿（方案 A）A/B 定论：**无收益，默认关**

`USE_EXPOSURE` 0→1，其余逐字同配（double_wave/colmap_refined，20k 计划、
warm_up 1000、-r 2、max_gauss 1.5M 撞帽、matmul 修复后代码）：

| 模型 | exposure | PSNR@10k | SSIM | LPIPS |
|---|---|---|---|---|
| model_colmap（对照） | 0 | **21.265** | **0.6748** | **0.2485** |
| model_colmap_exp | 1 | 20.953 | 0.6566 | 0.2519 |
| Δ | | **−0.31** | −0.018 | +0.003 |

- **10k 是唯一严格配对点**：对照那次训练在 14k 被杀，只落了 7k/10k ckpt。
- @20k：曝光组终点 21.633（best 21.863@19k），**无同配置对照**；且低于同期
  600k-cap 基线 21.808（cap 变量不可比，仅作指示）。两期点数均撞 1.5M cap
  （1,500,885 vs 1,502,625），配对性良好。
- 闭式初始化证明曝光漂移**真实存在**（max|A−I|=0.246，mean 0.020），但收益被
  SH 颜色吸收/噪声吞没 → **默认 `USE_EXPOSURE=0`**。
- ⚠️ 复核时先测到「+0.18dB」是**假象**：渲染 10k ckpt 时套了 20k 的运动基权重
  （见方案 A 节「渲染口径坑」，已修）。修正后渲染值与训练期协议逐位吻合
  （20.953 vs 20.954）。

### 2026-10-13 ① 位姿根因诊断 + COLMAP 精化（01c/01d）

见「位姿精化」节：VIO 位姿漂移实锤（inlier 摆 250 倍），01c 精化后 160/160
注册、重投影中位 0.92px、inlier 分布收窄到 2.4×。

**受控 A/B（同 v1 配方只换位姿，同迭代数口径）——决定性正收益**：

| 模型 | 位姿来源 | PSNR@10k | SSIM | LPIPS |
|---|---|---|---|---|
| model_v1 | App VIO（colmap_scene） | 13.978 | 0.3345 | 0.6629 |
| model_colmap | COLMAP 精化 | **21.265** | **0.6748** | **0.2485** |
| Δ | | **+7.29** | +0.340 | −0.414 |

@20k 只能单边看：v1 = 14.038（精化组 20k ckpt 不存在，训练在 14k 被杀）。
**结论：手机采集场景，位姿精化是收益最大的单一步骤，01c 应默认开启。**

## 分阶段分辨率调度（coarse-to-fine）—— ⚠️ 实测否决，默认关闭

实现了（`res_scheduler.py`，机制验证通过），但 2026-10-13 的 A/B/C 探针
（double_wave，2500 步，同场景同代码唯一变量是调度/补偿）给出负结果：

| 探针 | 配置 | 2300 步高斯数 | 2500 步耗时 | PSNR@2500 |
|---|---|---|---|---|
| A | 调度 + 阈值 ÷scale 补偿 | 418,162 | 9:05 | 13.28 |
| B | 调度无补偿 | 323,363 | 6:48 | 13.61 |
| C | 无调度（基线） | 307,182 | 6:45 | 13.56 |

**结论**：

1. **零加速**（B≈C，差 0.7% 在噪声内）：70% 迭代跑 1/4~1/2 分辨率总耗时不变。
   瓶颈不是光栅化像素数，而是随高斯数 N 线性增长的**分辨率无关开销**
   （SoM 系数场前向、Adam、稠密化记账）——参考方案的「渲染便宜 4~16 倍」
   在渲染占主导的栈里才成立，D-3DGS 系不成立。
2. **阈值补偿有害**（A 比 B 多 29% 高斯、慢 34%、PSNR 反低 0.33）：低分辨率下
   梯度分布变平，÷scale 阈值放进大量冗余小高斯。量纲推导的启发式被实测否决。

代码保留备查（`RES_SCHEDULE="0.3:4,0.7:2"` 手动开启），但**默认关闭**；
真正的速度杠杆是控制高斯数增长（densify 阈值），不是分辨率。

补充（2026-10-13 拿到对方完整文字描述后的口径对齐）：他们的实现与我们有三处
不同——①GT 降采样用 `F.interpolate(mode="area")`（我们用 bilinear+antialias）；
②补偿的是 **prune 阈值 ÷scale^0.5**（我们实测否决的是 grad 阈值 ÷scale，
不是同一个东西）；③官方建议首选「稠密化窗口放在 scale≤2 段」而非阈值补偿。
这些能改进实现质量，但**不改变零加速的物理原因**（渲染不占步时主导）。
**重启条件**（满足其一时该方案值得重估）：(a) 上 `-r 1` 全分辨率训练（像素 ×4，
光栅化份额翻转）；(b) 接入按像素计价的重 loss（如 mask 权重缓存）；(c) 方案 D
压住高斯数后光栅化占比回升。重启时按 ①②③ 对齐实现。

## 稠密化自动调度（方案 D 移植，`densify_sched.py`，默认关闭）

治「densify 阈值/窗口跨场景靠人肉猜」的病（v1 饿死 216k、v2 爆炸 ETA 10.7h
都是同一个根因）。机制（按对方文字描述忠实移植，不动官方仓）：

1. **自适应软上限**：`max_n = init_n + momentum`，momentum 初始 = init_n × mult
   （稀疏初始化 ×4 / 稠密 ×0.5；auto 规则 init_n ≥ 10 万判稠密）；
2. **需求累计（只增不减）**：每稠密化周期 `momentum = max(momentum,
   0.98×momentum + min(40000, 本轮超阈值候选数))`——需求再高也被衰减和
   40k/轮封顶限制成渐进曲线；
3. **topk 截断**：本轮目标增量 = 软上限 − 当前点数，候选（grad ≥ thr，口径与
   官方一致）超目标时只保留梯度最大的前 target 个（预过滤 grads 后调官方
   `densify_and_clone/split`，选择逻辑一字不改，`sched_densify_and_prune`）。

与 `MAX_GAUSSIANS` 的关系：调度器是软的速率控制（也接受 MAX_GAUSSIANS 作为
absolute_cap 输入），硬护栏独立兜底。**两者开启后，理论上任何场景裸跑
`02_train.sh`（不带 EXTRA_TRAIN_ARGS 降载参数）点数曲线都受控**——这正是
对方「手工配方从必须记的坑变成默认行为」的目标。

验证状态（2026-10-13）：调度器数学单测 ✅；fork 的 topk 截断在真
GaussianModel 上验证 ✅（不截断=官方行为、target=30 精确 +30、target=0 安全、
截断保留的确实是梯度 top-k）。端到端冒烟见下两节。

### 冒烟结果（2026-10-16，double_wave / colmap_refined / 20k 迭代）

| 模型 | thr/until | 调度器 | 窗口后 prune | @10k | best(19k) | @20k | 点数@20k |
|---|---|---|---|---|---|---|---|
| model_colmap_cap600k（参考） | 默认 0.0007/15000 | 关 | 无（撞帽整块跳过） | 21.18 | 21.63 | **21.81** | 629,715 |
| model_dsche_aggr（旧代码 bug） | 0.0002/10000 | 开 | **带尺寸判据** | 20.52 | — | **5.34 崩** | 9,682 |
| model_dsche_aggr（修复后，A2） | 0.0002/10000 | 开 | 只按 opacity | 20.65 | — | 21.41 | 565,493 |
| model_dsche_sched_def（裸跑验收，C2） | 默认 0.0007/15000 | 开 | 只按 opacity | 20.85 | **21.549** | 21.465 | 580,267 |
| model_dsche_noprune（C3，隔离实验） | 默认 0.0007/15000 | 开 | **关**（完全不剪） | 20.93 | 21.557 | 21.306 | 599,319 |

结论：
- **防爆 + 免调参成立**：激进（0.0002/10000）与默认（0.0007/15000）两种参数
  都稳定收敛、点数严格 ≤cap，全程无需人肉传 EXTRA_TRAIN_ARGS；B（momentum
  模式）3k 步冒烟动量单调累计、点数 53k→948k 受控，语义正确。
- **质量代价 ~0.3 dB**：调度器组（21.41 / 21.55）比手工硬护栏组（21.81）低
  0.26~0.34 dB（末三点均：参考 21.724 / C2 21.439，三次重跑噪声 ±0.045，
  差异显著）。归因见下条。
- **归因（C3 已排除「窗口后剪」这一嫌疑）**：C2（剪）vs C3（完全不剪）末三点均
  21.439 vs 21.384，差 **−0.055 dB**（@19k +0.008、@20k −0.159），落在噪声
  ±0.045 量级内 → **窗口后 opacity-only prune 对质量中性**，且 C2 少 19,052 点
  （−3.2%）。故代价来自「严格 cap 到 600k + absgrad 均值 topk 截断」这一策略
  本身：参考组撞帽后超到 629.7k 冻结（实得 5% 额外容量）、且无 topk 排序偏置。
  → **维持默认 `DENSIFY_PRUNE_AFTER=1`**（省点不亏质量）；调度器默认仍关
  （`USE_DENSIFY_SCHED=0`），作「裸跑防爆」保险丝按需开。

### 死亡螺旋（2026-10-16 冒烟事故，重要教训）

**症状**：窗口（densify_until_iter）关闭瞬间首个「只剪不增」一步删 136k 点
（599k→463k），之后每千步 ×0.685 指数衰减到 9.6k，PSNR 20.52→5.34；三次独立
重跑完全一致。

**根因**：`densification_postfix`（fork `scene/gaussian_model.py:319`）在每次
densify（clone/split 加点）后把 `max_radii2D` **整体清零**。所以窗口内
`max_radii2D > 20` 判据形同虚设（只看最近 100 步新增，≈0）；**窗口关闭后不再
densify → max_radii2D 不再清零 → 单调累积成「历史最大屏幕半径」**，首个窗口后
prune 就把 ~137k「曾以 >20px 出现过」的点（单手近景下正常存在，非无效点）全删。
删点没有补充来源 → 剩余点被迫长大补偿 → 屏幕半径进一步超标 → 再删，正反馈崩塌。

**修复**：窗口后的 prune **只按 `opacity < 0.005`，不得带尺寸判据**
（`max_radii2D>20` / `scaling>0.1·extent`）——`sched_prune_only(...,
max_screen_size=None)`。

**分量埋点实锤（2026-10-16 修复后全程取证，口径自洽）**：

| 阶段 | 日志样例（C2：窗口 [0,15000)，修复后） | vs | opa | applied |
|---|---|---|---|---|
| 窗口内 iter 1000–14000 | `iter 5000 opa=... ws=... applied=... max_screen_size=20` | **恒 0** | 数百 | ≈opa |
| 窗口关闭 iter 15000 | `iter 15000 (window closed) opa=294 vs=113867` | 113,867 | 294 | **294** |
| 窗口关闭 iter 20000 | `iter 20000 (window closed) opa=396 vs=139584` | 139,584 | 396 | **396** |

- **窗口内 vs≡0**：`densification_postfix` 每轮加点清零 `max_radii2D`，尺寸判据
  只看最近 100 步新增 → 完全失效（无害：每轮 prune 后紧跟 clone/split 补点）。
- **窗口后 vs 单调累积**：15k→20k 从 113,867 涨到 139,584（A2 更长窗口：169k→178k），
  即「历史最大屏幕半径 >20px」的点数——**修复前这一整串会被一次删掉**。
- **`applied` 只 294~422 = opa 单路**（`max_screen_size=None`）→ 修复生效的直接
  证据；修复前该位是 137,080。

### 同族地雷：硬护栏路径（2026-10-16 二次发现，默认路径上的）

**同一根因的第二处**：`n_before >= SOM_MAX_GAUSSIANS` 的硬护栏分支也是「只 prune
不 densify」，它当时传的是 `size_threshold`（=20，`iteration > opacity_reset_interval`
即 >3000 之后）。护栏路径**没有 densify 补点** → `max_radii2D` 同样不清零 →
判据退化成「历史最大屏幕半径」→ 批量误删。

> ⚠️ 这条路径是 `MAX_GAUSSIANS=600000` **默认开启**的（不是可选的调度器），
> 且旧版是「撞帽跳过整个调用」（=不 prune，21.81 那次就是这么跑的）——所以
> 2026-10-16 把语义改成「只 prune 不 densify」时引入了这个回归，此前无任何冒烟
> 覆盖（当天 D 案四组全走调度器分支）。

**低 cap A/B 实锤**（double_wave/colmap_refined，`MAX_GAUSSIANS=120000`、
调度器关、5000 步、`-r 2`；cap 在 iter 900 触发，但 `size_threshold` 到 iter>3000
才变 20）：

| | 带尺寸判据（修复前） | 去尺寸判据（修复后） |
|---|---|---|
| iter ≤3000（`max_screen_size=None`） | `applied` = opa 69~353，点数稳在 ~120k | 同左：opa 56~99，点数 ~120k |
| iter 3100 首个带判据周期 | **`vs=76,617` → `applied=86,738`**，一步 162,115→75,377（**−54%**） | `vs=57,776` 被忽略，`applied=994`（=opa） |
| iter 3100–5000 | 与 densify 交替震荡：110k↔150k，每 100~200 步删 6k~26k | 单调缓降（135,789→133,747），无震荡 |
| **@5000 test PSNR** | **8.85（训练崩）** | **19.73（正常）** |

**修复**：护栏分支同样传 `max_screen_size=None`，并补 `[DENSIFY-PRUNE] (cap)`
分量埋点 + 点数前后打印（此前这条路径是哑的）。

**结论（写进纪律）**：**任何「只剪不增」的 prune 调用一律不得带尺寸判据**。
判定方法很简单——这段代码附近没有 `densify_*` / `clone` / `split` / `postfix`
调用，就不该传 `max_screen_size`。合法使用尺寸判据的只有窗口内那一处（每轮
prune 后紧跟 clone/split 补点，自纠）。

## 单目深度先验正则（方案 E，`depth_prior.py`，默认关闭，⚠️ 首轮负结果）

**设计转折**：最初是「渲染深度 ↔ 单目深度」图像级 SSI loss，但本仓 fork
光栅化器（depth-diff-gaussian-rasterization）的 depth 是**前向-only 输出**
（backward 无深度梯度通道）→ 图像级渲染深度 loss 是摆设。故改为**点级锚定**
（MonoGS/Depth-GS 一族做法）：把形变后高斯中心投影到当前相机、采样单目深度，
加权最小二乘解 (a,b) 对齐后算 L1，只对位置求梯度（opacity 取 detach，不给
「调低透明度逃惩罚」留捷径），fg mask 加权。

**离线资产**：`01f_depth_maps.sh`（UniDepth V1 ViT-L14，gavs env）→
`<scene>/depths/<stem>.npz`（米制深度 float16）。fg mask 复用 01d（缺失退化为
纯 opacity 权重）。

**首轮 A/B（2026-10-16，double_wave/colmap_refined，20k，同代码逐字同参，
唯一变量 λ 0→0.05）——负结果**：

| 模型 | λ | ALL | FG | BG |
|---|---|---|---|---|
| model_depth_ctl（对照） | 0 | **21.872** | **23.474** | **21.670** |
| model_depth005 | 0.05 | 21.345 | 22.433 | 21.140 |
| Δ | | **−0.527** | **−1.041** | **−0.530** |

- 前景掉得比背景还狠 → 深度先验专伤它要救的部位。
- **根因（两条叠加）**：① 量级失衡——残差是绝对米制未归一化（~3.5~6.7，
  ×0.05=0.18~0.33 ≈ 光度项 ~0.09 的 2~4 倍，反客为主）；② 单目深度对手部
  不可靠（BACKLOG 风险命中），把逐点几何硬拉向错误深度。
- **过程中修掉一个 gy 垂直镜像 bug**（投影采样 y 符号错，让 `a<=0` 门限把
  损失静默失效 10k 步；用光栅化器自渲染深度逐点核对裁决 `gy=+ndc_y`）。
- **结论：默认保持 `DEPTH_LAMBDA=0`**。后续若再做：残差尺度归一化、λ 降到
  0.005 量级、或换边缘感知项（L1 on ∂d/∂x,∂d/∂y）。代码与
  `03b_fg_metrics.py`（前景专项指标工具）留仓备查。

## 与 SoM 原版的有意差异（改动都有理由，勿当 bug 修）
| 项 | SoM 原版 | 本实现 | 理由 |
|---|---|---|---|
| 系数 w | per-gaussian 可学习表 [G,B]（params.py:136） | 空间 MLP 场 softmax(coef_net(fourier(x))) | D-3DGS densify/prune 增删高斯，查表需 hook densify 记账；MLP 场对新增点天然合理，且空间光滑。代价：分组边界不如查表锐利 |
| 基初始化 | 3D track 速度 k-means + 逐类加权 Procrustes（init_utils.py） | identity 初始化 + warm_up 静态热身 | Phase 0 tracks 体检未做前没有 3D track 输入；SoM 消融显示跳初始化 EPE +35%——**若 Phase 1 验收不达标，这是首要怀疑对象**，届时补 init |
| 2D track 监督 | TAPIR tracks 加权（权重 2.0） | 暂未接入（Phase 1 纯光度 + 平滑） | tracks 基建（CoTracker3）未就绪；D-3DGS baseline 同样纯光度，对照干净 |
| 深度监督 | disparity 空间 L1（权重 0.5） | 未接入 | 同上，Phase 2 再评估 |
| 6D 混合 | 6D 加权后 Gram-Schmidt | 同款 | 继承其已知局限（朝向差异大的基之间过渡「缩短」，调研笔记 §11），需要时改 log/screw 空间混合 |

## Config (env vars)
| var | default | note |
|---|---|---|
| SCENE_NAME | human_seq | 场景名（02/03/08 共用） |
| SOURCE_PATH | $DEFORMABLE_HUMAN_RESULTS_DIR/$SCENE_NAME/colmap_scene | COLMAP 场景（deformable_human 01 产物） |
| MODEL_PATH | $RESULTS_DIR/$SCENE_NAME/model | 输出 |
| ITERATIONS | 10000 | 快速迭代；最终出片 20000（对齐 deformable_human 口径） |
| WARM_UP | 1000 | 前 N 步纯静态（基 identity 初始化，天然等价） |
| NUM_BASES | 10 | SoM 全局基 B；手部子模型（Phase 2）才需要局部 B_hand |
| SMOOTH_WEIGHT | 0.1 | 基加速度平滑（SoM 0.1；rot:transl 内权 1:2） |
| BASIS_LR | 1.6e-4 | 基 rots/transls lr（SoM configs.py:23-26） |
| COEF_LR | 1e-2 | 系数场 lr（SoM configs.py:5-11） |
| RES_SCHEDULE | ""（关闭） | 分阶段分辨率；实测零加速（见上节负结果），备查 |
| MAX_GAUSSIANS | 600000 | 高斯数硬护栏（v2 爆炸事故后引入），超帽只 prune；2026-10-15 cap600k A/B：60 万 @10k 掉 0.09dB<判定线、@20k 21.81 反超无 cap 组、2.4× 省点、~1.7× 加速 → 默认值由 150 万改 60 万 |
| USE_DENSIFY_SCHED | 0 | 方案 D 自动调度（见上节）；开启后 thr/窗口不敏感 |
| DENSIFY_MOMENTUM_MULT | auto | 调度器初始动量倍率（auto：init≥10万×0.5 否则×4） |
| DENSIFY_MAX_RATE | 1.0 | 相邻周期 next_n 相对增长上限（服务器同名常量） |
| DENSIFY_CNT_CLAMP | 3.0 | topk 分数分母 clamp（服务器 /cnt.clamp_min(3)） |
| DENSIFY_PRUNE_AFTER | 1 | densify 窗口结束后继续按期 prune（只剪不增，**只按 opacity**） |
| USE_EXPOSURE | 0 | 方案 A 曝光仿射补偿（见上节） |
| EXPOSURE_FROM_ITER | -1 | 曝光门控起点（-1 = WARM_UP） |
| EXPOSURE_LR | 1e-3 | 曝光仿射独立 lr |
| EXPOSURE_REG | 0.01 | 拉回恒等正则权重 |
| DEPTH_LAMBDA | 0 | 方案 E 深度先验权重（0=关；⚠️ 0.05 首轮负结果，见上节）。需先跑 01f_depth_maps.sh 产出 `<scene>/depths/*.npz` |
| DEPTH_FROM_ITER | -1 | 深度损失启用起点（-1 = WARM_UP，且强制 ≥ warm_up） |
| DEPTH_USE_MASK | 1 | 1=有 01d 前景 mask（`<scene>/masks/`）就加权 |
| DEPTH_MIN_POINTS | 2048 | 帧内投影点不足则该步跳过深度损失 |
| EXTRA_TRAIN_ARGS | （空） | ⚠️ 复杂背景必带降载参数（见下） |

## 目录布局
```
media_code/som_human/
├── _env.sh               # 复用 deformable_human env + 共用 DG_DIR
├── 00a_setup_env.sh      # WSL 体检（不新建 env）
├── 01_convert_capture.sh/.py  # 手机采集包（heic+位姿json+pcd）→ COLMAP 场景
├── 01b_check_reproj.py   # 位姿/内参诊断：pcd 重投影叠加 + 逐帧 inlier/色差
├── 01c_colmap_refine.sh  # COLMAP SfM 精化 VIO 位姿（GPU SIFT+mapper，内参固定）
├── 01d_check_refined.py  # 精化质量校验：inlier 分布/重投影误差/轨迹对齐残差
├── 01f_depth_maps.sh     # 方案 E 资产：为每帧生成单目深度图（UniDepth）到 <scene>/depths/
├── 01g_person_masks.sh   # per-person：SAM3 video（prompt=person）逐人 mask + manifest
├── 01h_build_person_scenes.sh/.py  # per-person：按 pid 拆独立训练场景（images/sparse 软链）
├── 02_train.sh           # SoM 运动基训练
├── 03_render.sh          # 渲染 + 评测（render/time/original）
├── 03b_fg_metrics.py     # 前景专项指标（fg/bg/all PSNR + 逐视角最差 + 三联拼图）
├── 08_move_output.sh     # WSL：产物搬到 D 盘
├── hand_batch.sh         # 批量：一批采集包跑通 01→01c→02（幂等，PAR 控并发）
├── som_motion.py         # 核心：SE(3) 基凸组合运动模型（替换 DeformModel）
├── res_scheduler.py      # 分阶段分辨率调度（coarse-to-fine，实测否决备查）
├── densify_sched.py      # 稠密化自动调度（方案 D：软上限+需求累计+topk 截断）
├── exposure.py           # 曝光仿射补偿（方案 A：逐相机 4×3 仿射 + 闭式初始化）
├── depth_prior.py        # 方案 E：点级深度锚定损失（投影采样单目深度 + lstsq 对齐 (a,b)）
├── depth_est.py          # 方案 E：单目深度估计（UniDepth 推理，供 01f 调用）
├── train_som.py          # fork 官方 train.py（SOM / SOM-RES / SOM-DENSIFY / SOM-EXPOSURE 注释标注差异）
├── render_som.py         # fork 官方 render.py（render/time/original）
└── README.md / README_wsl.md
官方仓（共用）：~/repos/Deformable-3D-Gaussians
输出：~/output/som_human_results/<scene>/
```

## 验收口径（方案 §10 的 Phase 1 子集）
1. **目视主指标**：`MODE=original` 组合渲染对比 deformable_human 20k baseline，
   重影/撕裂应显著减弱；
2. masked PSNR 不低于 D-3DGS baseline（手部 mask 评测待 Phase 2 mask 基建）；
3. 训练显存 ≤ 24GB（沿用降载参数）。
