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

## 首次准备
无新增安装。前置 = deformable_human 全流程已跑通（见其 README_wsl.md）。
`00a_setup_env.sh` 只做体检：env 存在 / 官方仓在位 / fork 光栅化器可 import /
som_motion 自检。

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
截断保留的确实是梯度 top-k）。**端到端待 v4 冒烟**（GPU 被 v3 占用中）。

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
| MAX_GAUSSIANS | 1500000 | 高斯数硬护栏（v2 爆炸事故后引入），超帽只 prune |
| USE_DENSIFY_SCHED | 0 | 方案 D 自动调度（见上节）；开启后 thr/窗口不敏感 |
| DENSIFY_MOMENTUM_MULT | auto | 调度器初始动量倍率（auto：init≥10万×0.5 否则×4） |
| EXTRA_TRAIN_ARGS | （空） | ⚠️ 复杂背景必带降载参数（见下） |

## 目录布局
```
media_code/som_human/
├── _env.sh               # 复用 deformable_human env + 共用 DG_DIR
├── 00a_setup_env.sh      # WSL 体检（不新建 env）
├── 01_convert_capture.sh/.py  # 手机采集包（heic+位姿json+pcd）→ COLMAP 场景
├── 02_train.sh           # SoM 运动基训练
├── 03_render.sh          # 渲染 + 评测（render/time/original）
├── 08_move_output.sh     # WSL：产物搬到 D 盘
├── som_motion.py         # 核心：SE(3) 基凸组合运动模型（替换 DeformModel）
├── res_scheduler.py      # 分阶段分辨率调度（coarse-to-fine，实测否决备查）
├── densify_sched.py      # 稠密化自动调度（方案 D：软上限+需求累计+topk 截断）
├── train_som.py          # fork 官方 train.py（SOM / SOM-RES / SOM-DENSIFY 注释标注差异）
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
