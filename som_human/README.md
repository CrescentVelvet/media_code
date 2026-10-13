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

# ── 1) 数据：复用 deformable_human 的 01 产物，本目录不重复 ──
#    SOURCE_PATH 默认指向 ~/output/deformable_human_results/<scene>/colmap_scene

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

## 首次准备
无新增安装。前置 = deformable_human 全流程已跑通（见其 README_wsl.md）。
`00a_setup_env.sh` 只做体检：env 存在 / 官方仓在位 / fork 光栅化器可 import /
som_motion 自检。

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
| EXTRA_TRAIN_ARGS | （空） | ⚠️ 复杂背景必带降载参数（见下） |

## 目录布局
```
media_code/som_human/
├── _env.sh               # 复用 deformable_human env + 共用 DG_DIR
├── 00a_setup_env.sh      # WSL 体检（不新建 env）
├── 02_train.sh           # SoM 运动基训练
├── 03_render.sh          # 渲染 + 评测（render/time/original）
├── 08_move_output.sh     # WSL：产物搬到 D 盘
├── som_motion.py         # 核心：SE(3) 基凸组合运动模型（替换 DeformModel）
├── train_som.py          # fork 官方 train.py（SOM 注释标注差异）
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
