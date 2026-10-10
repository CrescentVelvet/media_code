# deformable_human — AI 代码地图（改代码前先读这个）

> 本文件给接手的 AI 看：目标是让你在**不读完全部代码**的前提下，
> 知道改哪里、不能碰什么、怎么验证。仓级通用规范（commit / git 纪律 /
> 文档分工）在仓根 `../AGENTS.md`，本文件只记本项目特有的东西。
>
> 运行命令查 `README.md`；实验数据与结论查 `README.md` 的「实验记录」节；
> WSL 环境细节查 `README_wsl.md`。本文件不重复这些内容，只做导航。

## 1. 这个项目在解决什么

单目视频拍人，人物有动作 → 静态 3DGS 把多时刻几何平均 → **重影/拖影**。
方法：[Deformable-3D-Gaussians](https://github.com/ingra14m/Deformable-3D-Gaussians)
（canonical 高斯 + 形变 MLP，**无预训练权重，每场景从头训**）。

当前阶段：vanilla baseline 已验证（手部重影显著收敛），
正在逐项叠加改进（mask 加权 loss ✅ 有效）。Phase 2 规划 = MHR 3DMM 锚定
（复用 vggt_human 资产），**尚未开工**。

## 2. 代码在哪里（物理位置，WSL 口径）

| 内容 | 位置 |
|---|---|
| 编排脚本（本目录） | `/mnt/c/code/media_code/deformable_human/`（Windows 盘，git 仓内） |
| 官方仓（不要改） | `~/repos/Deformable-3D-Gaussians`（Linux fs，`$DG_DIR`） |
| 训练输出（过程产物） | `~/output/deformable_human_results/<scene>/`（Linux fs） |
| 最终产物（搬盘后） | `/mnt/d/output/deformable_human_results/<scene>/` |
| conda env | `deformable_human`（clone 自 vggt_human，torch 2.5.1+cu121） |

**路径纪律**：训练/编译只写 Linux fs；`/mnt/c` 只读代码、`/mnt/d` 只放最终产物。
违反会慢 5-10 倍甚至 symlink 崩坏。

**复用关系（文件做接口，禁止复制代码）**：
- `vggt_human/01a_video_to_frames.sh` ← 01 抽帧（含模糊门）
- `deformable_gaussians/03_colmap_pose.sh` ← 01 的 COLMAP 建场景
- `vggt_human/sam2_face_masks.py` + `sam3_face_masks_worker.py` ← 01d 的 SAM3 后端（跑在 `sam3` env）
- `vggt_human/run_batch.py` + `npz_to_colmap.py` ← VGGT 前馈位姿链路（已证伪，见 §6）
- `vggt_human/pose_adjuster.py` ← 01c 位姿规整

## 3. Pipeline 数据流

```
视频/图像序列
  │  01_prepare_data.sh        → <scene>/frames/（抽帧+去模糊）
  │  委托 01a + 03_colmap_pose → <scene>/colmap_scene/{images/, sparse/0/}
  │  01c（可选，位姿规整）      → 改写 sparse/0（备份到 sparse/0_raw）
  │  01d（可选，SAM3 前景mask） → colmap_scene/masks/{stem}.png
  │  01e（降级 fallback）       → colmap_phone/（手机位姿，质量 -4dB，见 §6）
  ▼
02_train.sh                    → <scene>/model*/
  │  三分支互斥选择：
  │   默认        → 官方 train.py（cd $DG_DIR 跑，相对 import 要求）
  │   USE_MASK_LOSS=1    → train_mask.py（fork，加权 L1 [+ 可选 bg 软正则]）
  │   USE_POSE_REFINE=1  → train_pose_refine.py（fork，位姿 delta；⚠️ 见 §6）
  ▼
03_render.sh                   → model*/test|train/ours_<iter>/{renders,gt} + 指标
  ▼
08_move_output.sh              → 搬到 /mnt/d（释放 vhdx）
```

## 4. 文件地图（按改动频率排序）

### 训练核心（改动热点）

| 文件 | 角色 | 修改要点 |
|---|---|---|
| `train_mask.py` | **当前主训练入口**（USE_MASK_LOSS=1）。fork 官方 train.py | 与官方的差异全部用「MASK-LOSS」注释标注。`MaskWeightCache` 按 `image_name`（无扩展名 stem）懒加载 `{mask_dir}/{stem}.png`。L1 加权按均值归一化（不改 LR 语义）；**DSSIM 故意不加权**。加新 loss/正则就改这里 |
| `02_train.sh` | 训练编排 + 所有开关的默认值 | 新开关在这里加 `${VAR:-default}` + 透传。三个训练分支的互斥检查也在这里 |
| `train_pose_refine.py` | 位姿精炼 fork（与 mask 版互斥） | **已证伪于坏初值场景**（§6），保留作存档。改它前先读 pose_delta.py 头部数学推导 |
| `pose_delta.py` | 场景侧位姿 delta 模块（被 train_pose_refine.py 引用） | 为什么不做相机侧：stock 光栅化器对 viewmatrix/projmatrix **无梯度**。数学等价推导在文件头注释，改动前必须读懂 |

### 数据准备

| 文件 | 角色 |
|---|---|
| `01_prepare_data.sh` | 抽帧 + COLMAP 编排（委托兄弟目录，支持预置帧模式免 VIDEO_PATH） |
| `01c_pose_adjust.sh` + `pose_adjust_colmap.py` | 位姿规整（主体居中/重力对齐/尺度归一化），经 colmap model_converter TXT 中转，幂等备份 |
| `01d_fg_masks.sh` + `merge_fg_masks.py` | SAM3 前景 mask（person + 持有物两遍 prompt，extra 实例须与 person 重叠 ≥0.3 防误检）。输出软 mask 直接当 loss 权重 |
| `01e_pose_from_transforms.sh` + `nerf_to_colmap.py` | 手机 transforms.json/pcd.ply → COLMAP 文本场景（fallback 管线） |

### 评测与 QC 工具（A/B 必备）

| 文件 | 角色 |
|---|---|
| `03_render.sh` | 渲染 + PSNR/SSIM/LPIPS。`ITERATION=<N>` 可评中间 checkpoint |
| `eval_fg_psnr.py` | **前景专项 PSNR**（全图指标对 20% 占比的前景不敏感，A/B 必须看这个）+ 三联对比图导出 |
| `qc_proj_fg.py` | 高斯投影到 mask 的方向约定目检（改投影代码后必跑） |

## 5. 关键不变量（改了就会坏）

1. **图像名必须纯数字**（`00000.jpg`）——官方 `dataset_readers.py` 用文件名算 fid（时序）。
   mask 命名随之是 `{stem}.png`。任何新的数据入口（如 01e）都必须遵守。
2. **test 划分 = 排序后每 8 帧取 1**（llffhold=8）。跨模型 A/B 时只要图像集相同，
   test split 就相同，指标可比。`eval_fg_psnr.py` 复用同一口径。
3. **fork 训练脚本必须在 `$DG_DIR` 里跑**（官方仓相对 import）——02 里用子 shell
   `( cd "$DG_DIR" && python ... )`，且要 `sys.path.insert(0, $DG_DIR)`。
4. **`cfg_args` 是绝对路径快照**：模型目录被 08 搬到 /mnt/d 后，里面的
   `source_path` 会失效 → 03_render 报 "Could not recognize scene type!"。
   修复 = sed 改 `cfg_args` 里的路径（有 .bak 备份惯例）。
5. **稠密化爆炸防护**：复杂室内必须
   `EXTRA_TRAIN_ARGS="-r 2 --densify_grad_threshold 0.0004 --densify_until_iter 2000"`
   （背景杂物导致高斯复利爆炸到 130 万/24GB，必须在爆炸点前冻结）。
   ⚠️ 这是实验配方但**不是 02 的默认值**——跑 A/B 时显式传，别假设默认有。
6. **stock 光栅化器对相机矩阵和内参都无梯度**——位姿只能场景侧 delta（pose_delta.py），
   内参要调只能在 01 阶段走 colmap bundle_adjuster。
7. **warm_up 语义**：前 N 步形变量=0（纯静态热身），形变 MLP 从 warm_up 后才学习。
   改 WARM_UP 会改变形变 MLP 的有效训练步数。
8. **官方仓 `~/repos/Deformable-3D-Gaussians` 不要改**。定制都走本目录的 fork
   脚本（train_mask.py / train_pose_refine.py 模式）。deformable_gaussians 项目
   共用同一官方仓，你改了会影响兄弟项目。

## 6. 已证伪的路线（别再试，数据在 README「实验记录」）

| 路线 | 结果 | 根因 |
|---|---|---|
| 迭代数 7k | -1.5~-2dB | 形变 MLP 收敛比高斯本体慢，7k 没学到位 |
| bg 变形软正则（动态判定） | 前景 -0.32dB | 手变形大→甩出 mask→误判 bg→被压，自抑制循环 |
| bg 变形软正则（canonical 静态标签） | 前景 -0.18dB | MLP 全场景共享权重，压 bg = 占 fg 的函数空间 |
| 手机 VI 位姿替 COLMAP | -4.0dB | 逐帧亚像素 VI 噪声（重投影目检完美也没用），train→test 泛化缺口 3.7dB |
| 手机位姿 + POSE_REFINE 精化 | 发散（14.78dB） | warm-up 期位姿被垃圾梯度拖走自我强化；需 BARF 式课程调度，不值 |
| VGGT-Omega 前馈位姿 | -5.3dB | 预测内参有系统差且逐帧浮动（fx median 1397.6 vs 标定 1380.8） |

**结论：COLMAP BA 的 ~12 分钟是质量入场券，不可替代。**
01e（手机位姿）保留作 COLMAP 彻底失败时的 fallback。

## 7. 已验证有效的（当前最佳配方）

```
ITERATIONS=10000  WARM_UP=1000  USE_MASK_LOSS=1  FG_WEIGHT=1.0  BG_WEIGHT=0.2 \
EXTRA_TRAIN_ARGS="-r 2 --densify_grad_threshold 0.0004 --densify_until_iter 2000"
```
hand_motion 基线数字（A/B 参照系）：
- 全图 test：29.26 dB（无 mask）/ 29.14（mask 版，-0.12 属噪声）
- **前景专项：26.53 → 26.70（+0.17）**，目视手部改善大于数字

## 8. 开放方向（backlog，按优先级）

1. **per-frame latent / 更大 deform MLP**——残余手部模糊的容量路线，改 deform_model
2. BG_WEIGHT=0.1（10:1 激进档）——最便宜的一试
3. 多人场景验证（mask 管线已支持多实例）
4. Phase 2：MHR 3DMM 锚定（canonical 模板初始化 + landmarks 正则，
   复用 vggt_human 的 MHR/657 点 landmarks；01 里 `ANCHOR_EXPORT=1` 是占位）

## 9. 标准 A/B 流程（本项目的实验纪律）

1. 改代码 → 同配方训练（§7 的配方，换输出目录 `MODEL_PATH=<scene>/model_<变体>`）
2. `03_render.sh` 渲染评测 → 记录官方 test PSNR/SSIM/LPIPS
3. `eval_fg_psnr.py` 跑前景专项对比（基线参照 §7 数字）→ 导出三联图目视手部
4. 结论写进 README「实验记录」（含否决的负结果，防止后人重试）
5. commit：`<项目名>: <一句话>`（仓根 AGENTS.md §8），**不 push**（用户手动）

**长任务注意**：训练 ~40min，WorkBuddy 的 PowerShell 后台任务有 ~2min 超时会
杀 WSL 进程树——训练必须用 `setsid nohup ... &` 脱离会话启动，再轮询日志文件。

## 10. 一句话回答「我要改 X，去哪里」

- 改 loss / 加正则 → `train_mask.py`（搜「MASK-LOSS」）
- 改训练超参默认 → `02_train.sh` 顶部
- 改数据入口 → `01_prepare_data.sh` / 新增 `01x_*.sh` + 转换器 py
- 改评测口径 → `eval_fg_psnr.py`（前景）/ `03_render.sh`（全图）
- 改位姿相关 → 先读 `pose_delta.py` 头部 + README 位姿消融，确认不踩 §6 的坑
- 改 mask 生成 → `01d_fg_masks.sh` / `merge_fg_masks.py`（SAM3 后端在 vggt_human，别复制）
