# deformable_human — 单目视频人体重建去模糊/重影（canonical + 形变场）

> AI 接手请先读 [`AGENTS.md`](AGENTS.md)（代码地图：文件导航、不变量、已证伪路线、A/B 纪律）。

针对的问题：**单目视频拍人，人物有动作 → 静态重建把多时刻几何平均，出现重影/拖影**。
路线：canonical 高斯 + 形变 MLP 显式建模运动（[Deformable-3D-Gaussians](https://github.com/ingra14m/Deformable-3D-Gaussians)，
CVPR 2024），先跑 vanilla baseline 量化去重影效果；Phase 2 再叠 MHR 3DMM 锚定
（canonical 模板初始化 + 骨架/landmarks 正则，复用 vggt_human 的 MHR 与 657 点 landmarks）。

与同仓兄弟目录的关系：
- **`deformable_gaussians/`**：同一官方仓的服务器侧编排（py3.7/torch1.13 独立 env，D-NeRF 复现 + NeRF-DS 真实序列）。
  本目录**复用**它的 `03_colmap_pose.sh`（COLMAP 建场景），不重写。
- **`vggt_human/`**：Stage A 复用它的 `01a_video_to_frames.sh`（抽帧+模糊门）；Phase 2 复用 MHR/landmarks 能力。
  文件做接口，**不复制代码**。
- 本目录 env 从 **vggt_human clone**（torch 2.5.1+cu121，nvcc 随 clone 继承），与 deformable_gaussians 的
  py3.7/torch1.13 env 互不干扰。

> 本仓训练型方法：无预训练权重，每个场景从头训一套 canonical 高斯 + 形变 MLP。

## 常用命令
> **WSL 本机请直接用 [README_wsl.md](README_wsl.md)**（主战场）。以下为服务器版。
> 铁律：每条命令显式写出输入路径、输出路径，不靠默认值。

```bash
# ── 一键（先完成首次准备）──
GPU=0 VIDEO_PATH=/data_3d/<uid>/data/xxx.mp4 SCENE_NAME=human_seq \
  RESULTS_DIR=../deformable_human_results \
  bash deformable_human/run_all.sh

# ── 分步 ──
# 1) 单目视频 → 帧（带模糊门）→ COLMAP 场景
GPU=0 VIDEO_PATH=/data_3d/<uid>/data/xxx.mp4 SCENE_NAME=human_seq \
  VIDEO_FPS=6 BLUR_THRESHOLD=100 \
  RESULTS_DIR=../deformable_human_results \
  bash deformable_human/01_prepare_data.sh
# 1c)（可选）位姿规整：主体居中 + 重力对齐 + 尺度归一化
GPU=0 SCENE_NAME=human_seq RESULTS_DIR=../deformable_human_results \
  bash deformable_human/01c_pose_adjust.sh
# 1d)（可选）SAM3 前景 mask（person+持有物，复用 vggt_human 资产，需 sam3 env）
SCENE_NAME=human_seq RESULTS_DIR=../deformable_human_results \
  bash deformable_human/01d_fg_masks.sh
# 2) vanilla Deformable-GS 训练（canonical + 形变 MLP，NeRF-DS 模式）
#    可选：USE_POSE_REFINE=1 联合精炼位姿（可学四元数+平移，内参不学）
GPU=0 SCENE_NAME=human_seq RESULTS_DIR=../deformable_human_results \
  ITERATIONS=10000 \
  bash deformable_human/02_train.sh
# 3) 渲染 + PSNR/SSIM/LPIPS（看运动区域是否还有拖影：MODE=original）
GPU=0 SCENE_NAME=human_seq RESULTS_DIR=../deformable_human_results \
  MODE=render RUN_METRICS=1 \
  bash deformable_human/03_render.sh
```

## 首次准备
```bash
# clone 本仓 + proxy.env（略，见仓级 README）

# 建 env（从 vggt_human clone）+ clone 官方仓 + 装依赖 + 编 CUDA 子模块
CLONE_FROM=vggt_human INSTALL_DEPS=1 BUILD_CUDA=1 \
  bash deformable_human/00_setup_env.sh
```

权重目录布局：无需权重（训练型）。官方仓在 `$REPO_DIR/../Deformable-3D-Gaussians`。

---

以下为详细参考。

## Config (env vars)
| var | default | note |
|---|---|---|
| `VIDEO_PATH` | （必填） | 单目视频文件，01 的输入 |
| `SCENE_NAME` | `human_seq` | 场景名，贯穿 01/02/03 默认路径 |
| `VIDEO_FPS` | `6` | 抽帧 fps（动态序列要比静态的 2 密） |
| `BLUR_THRESHOLD` | `100` | 拉普拉斯模糊门，0=关（透传 vggt_human/01a） |
| `USE_GPU` | `1` | COLMAP SIFT 用 GPU（conda-forge colmap 3.11.1 实测带 CUDA） |
| `ITERATIONS` | `10000` | 快速迭代默认；最终出片 `20000`（消融见「实验记录」） |
| `WARM_UP` | `1000` | 前 N 步形变量=0 纯静态热身（官方默认 3000，配合 10k 短训缩短） |
| `IS_6DOF` | `0` | 1=6DoF 形变变体（略准、更慢） |
| `USE_POSE_REFINE` | `0` | 1=训练中联合精炼位姿（可学四元数+平移，内参不学） |
| `POSE_REFINE_WEIGHT` | `0.01` | 位姿正则权重（拉回 COLMAP 初值） |
| `POSE_REFINE_START` | `0` | >0 时延迟到第 N 步才启用位姿精炼（BARF 式课程；坏初值场景实测仍证伪，见「实验记录」） |
| `USE_MASK_LOSS` | `0` | 1=前景 mask 加权 L1（train_mask.py；需先跑 01d；与 POSE_REFINE 互斥） |
| `MASK_DIR` | `$SOURCE_PATH/masks` | 01d 的输出目录 |
| `FG_WEIGHT` / `BG_WEIGHT` | `1.0` / `0.2` | 前景/背景像素 loss 权重（按有效像素均值归一化，不改变 loss 量级） |
| `C2F` | `0` | 1=coarse-to-fine 分辨率调度：前 30% 步 1/4 分辨率→30-70% 1/2→末段全分辨率（仅 mask 分支；评测仍按原生分辨率） |
| `C2F_SCHEDULE` | — | 显式档位覆盖默认，如 `0:4,3000:2,7000:1`（start_iter:scale 升序） |
| `POSE_ADJUST` | — | 01c：位姿规整开关（居中+重力对齐+尺度归一化） |
| `WHITE_BG` | `0` | 1=白底训练（输入抠图后开） |
| `MODE` | `render` | render/time/all/view/pose/original |
| `DG_DIR` | `../Deformable-3D-Gaussians` | 官方仓位置（WSL 由 proxy.env 覆盖为 ~/repos/） |
| `CLONE_FROM` | `vggt_human` | 00/00a 建 env 的 donor |

## 目录布局
```
<code-dir>/
├── media_code/deformable_human/      # 本目录（编排脚本）
├── Deformable-3D-Gaussians/          # 官方仓（sibling；WSL 在 ~/repos/）
└── deformable_human_results/         # 输出（sibling；WSL 在 ~/output/）
    └── <scene>/                      # 每个输入数据一个文件夹
        ├── frames/                   # 01 抽帧（模糊帧已剔除）
        ├── colmap_scene/             # 01 COLMAP 场景（images/ + sparse/0/）
        │   └── masks/                # 01d SAM3 前景 mask（person+持有物，{stem}.png，01=前景）
        ├── model/                    # 02 形变模型 + 03 渲染产物
        └── model_static/             # （可选）静态 3DGS 基线（A/B 对比用）
```

## 实验记录
**C2F 分辨率调度三轮 A/B（2026-10-10，hand_motion，同配方 10k+warm1000+mask）**：

| 配置 | 耗时 | 官方 PSNR/SSIM/LPIPS | 前景 PSNR | 点数 | 判定 |
|---|---|---|---|---|---|
| baseline（model_mask） | ~49 min | 29.14 / 0.9154 / 0.0823 | 26.70 | 921k | 参照系 |
| v1 无补偿（model_mask_c2f） | **17.3 min（2.8×）** | 28.68 / 0.9004 / 0.1146 | 26.59（-0.11） | **328k** | ❌ 欠容量 |
| v2 thr/scale 补偿 | 已中止 | — | — | ~1.7M（估） | ❌ 过冲，ETA 54min 且全分辨率段有 OOM 风险 |
| **v3 thr/√scale 补偿（model_mask_c2f2）** | **30.5 min（1.6×）** | **28.91 / 0.9075 / 0.0967** | **26.66（-0.04）** | 625k | ✅ **采纳** |

机制与剂量标定：默认档（0-30% 1/4 → 30-70% 1/2 → 末段全分辨率）下稠密化窗口
（densify_until_iter=2000）完全落在 1/4 分辨率档，viewspace 梯度在 1/16 像素上累积，
固定阈值筛不出足够的点。阈值-点数剂量曲线（hand_motion 10k 实测）：
thr×1 → 328k；thr/√scale（1/4 档 0.0002）→ 625k；thr/scale（1/4 档 0.0001）→ ~1.7M。
**1/√scale 是速度/容量的平衡点**（已设为 `--c2f_densify_comp` 默认行为，可 =0 关闭）。
残余 -0.23dB 集中在背景纹理（前景仅 -0.04，噪声带内），与点数缺口（625k vs 921k）一致；
如需再追，1/4 档阈值可微调到 0.00015（插值估计），边际收益有限。
结论：**C2F 默认档 + √scale 补偿采纳为提速方案，省 38% 训练时间（49→30.5min），
前景质量无损（-0.04dB 噪声带内），符合预期目标（20-40%）。**

**迭代数消融**（2026-10-09，同一次训练的中间 checkpoint 渲染评测，test split 17 帧）：

| 场景 | 迭代 | PSNR ↑ | Δ vs 20k | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|---|---|
| hand_motion（135 帧，含手部动作） | 7000 | 28.05 | **-1.47** | 0.9014 | 0.0989 |
| | 10000 | 28.85 | -0.67 | 0.9107 | 0.0879 |
| | 20000 | 29.52 | — | 0.9187 | 0.0778 |
| vggt_source（125 帧，微动） | 7000 | 30.42 | **-2.08** | 0.9297 | 0.1475 |
| | 10000 | 31.53 | -0.98 | 0.9402 | 0.1328 |
| | 20000 | 32.51 | — | 0.9475 | 0.1164 |

结论：**7k 不可取**——形变 MLP 比高斯本体收敛慢，7k 时形变没学到位，
vggt_source 的 30.42dB 几乎跌回静态 baseline（30.14dB），+2.37dB 形变收益被吃光。
**默认定为 10000**（省一半时间，代价 <1dB），最终出片 `ITERATIONS=20000` 手动覆盖。

**warm_up 消融**（2026-10-09，hand_motion 独立重训 10k + `--warm_up 1000`）：

| 配置 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| 旧 10k checkpoint（20k 训练中截取，warm_up=3000） | 28.85 | 0.9107 | 0.0879 |
| **新默认：10k + warm_up=1000** | **29.26** | **0.9168** | **0.0828** |
| 20k 参考（warm_up=3000） | 29.52 | 0.9187 | 0.0778 |

新默认组合比旧 10k 快照还高 +0.40dB，距 20k 满训只差 0.26dB——
缩短静态热身让形变 MLP 多学了 2000 步，收益实打实。训练耗时约 49 min（3090 单卡）。

**mask 加权 loss 消融**（2026-10-09，hand_motion，同配方 10k + warm1000，
fg=1.0/bg=0.2，mask 来自 01d 的 SAM3 video 传播「person + 恐龙玩偶」）：

| 模型 | 官方 test 指标（17 视角） | eval_fg_psnr.py 前景专项 |
|---|---|---|
| baseline（无 mask） | 29.26 / 0.9168 / 0.0828 | 前景 PSNR 26.53 |
| mask 加权 | 29.14 / 0.9154 / 0.0823 | 前景 PSNR **26.70（+0.17）** |

逐视角前景 Δ 分布：17 个 test 视角中 11 个提升，大运动帧收益显著
（00016 +1.04、00120 +0.62、00096 +0.56），个别帧小幅回退（00032 -0.81）。
目视对比（`mask_ab_compare/cmp_*.jpg`，GT|baseline|mask 三联）：
**手部重影明显收敛**——baseline 手指涂抹/重影的区域，mask 版手指根数可辨。

结论与使用建议：
- 方向验证成立：loss 加权把监督从静态背景（占像素 ~80%）重新分配给前景，
  官方全图指标持平（-0.1dB 属噪声），前景专项 +0.17dB 且目视改善大于数字。
- 权重 fg/bg=5:1 是保守档，若前景改善不足可试 10:1（BG_WEIGHT=0.1）。

**bg 变形软正则（动态判定）——否决**（2026-10-09，`--bg_deform_lambda 0.1`，
判定口径 = 变形后位置投影到当前帧 mask，model_mask_bgdef vs model_mask）：

| 指标 | mask only | mask + 动态软正则 | Δ |
|---|---|---|---|
| 官方全图 PSNR | 29.14 | 29.05 | -0.09 |
| 前景专项 PSNR | **26.70** | 26.37 | **-0.32**（17 视角 14 退 3 升） |

否决原因（机制缺陷，不是 λ 问题）：手部点变形大 → 投影甩出 GT mask 边界 →
被误判为 bg → 变形被正则压住 → 下一帧更甩出去，**自抑制循环**。
训练日志佐证：mean‖d‖ 被整体压掉一个量级（0.064 → 0.003），前景需要的大变形也被误伤。

**bg 变形软正则（canonical 静态标签修正版）——同样否决，路线收束**
（2026-10-09，标签 = canonical xyz 对 12 帧 mask 投票，warm_up 时算一次、
随 densify 点数变化重算，model_mask_bgdef_static vs model_mask）：

| 指标 | mask only | mask + 静态软正则 | Δ |
|---|---|---|---|
| 官方全图 PSNR | 29.14 | 29.09 | -0.05 |
| 前景专项 PSNR | **26.70** | 26.52 | **-0.18**（17 视角 13 退 4 升） |

静态标签消除了自抑制循环（退化从 -0.32 收窄到 -0.18），但仍为负。
根因更深一层：deform MLP 全场景**共享权重**，压 bg 点输出 = 让网络拿出一部分
函数空间去拟合「趋零函数」，fg 变形的表达力同步受损；且背景微动（布料、阴影边缘）
本就是需要的自由度——正则惩罚的是「变形本身」而非「错误的变形」。
**结论：「抑制背景变形」路线整体证伪**（λ 调小上限只是零收益）。
残余手部模糊的剩余路线：①per-frame latent / 更大 deform MLP（容量）；
②输入侧利用源数据自带 transforms.json 位姿（免 COLMAP，位姿更准）。

### 位姿来源消融（2026-10-09 晚，同配方 10k+warm1000，hand_motion）

| 位姿来源 | 初始化点数 | 官方 test PSNR | train PSNR | 结论 |
|---|---|---|---|---|
| **COLMAP BA**（01 默认） | ~10k | **29.14**（mask 版）/ 29.26（无 mask） | ~29 | ✅ 主链路 |
| 手机 VI 位姿（01e，transforms.json） | 178k（pcd.ply 稠密） | 25.14 | 28.84 | ❌ -4dB |
| 手机 VI + POSE_REFINE | 178k | 14.78 | 14.61 | ❌ 发散 |
| 手机 VI + POSE_REFINE（延迟 3000 启动） | 178k | 17.95（best@5000）→16.35@10000 | — | ❌ 延迟也救不活 |
| VGGT-Omega 前馈（vggt_human 02/03 链路） | 5.4k（voxel 下采样偏狠） | 23.85 | 25.74 | ❌ -5.3dB |

要点：
- **COLMAP BA 明显不可替代**：VI 位姿重投影目检完美、无系统偏移（±8px 网格搜索验证），
  但逐帧亚像素级噪声在全场景一致地压指标（train→test 泛化缺口 3.7dB，远大于 COLMAP）。
- 手机位姿 + 可学精化（POSE_REFINE）**发散**：drift rot mean 1.48°/trans mean 0.12m，
  warm-up 期几何未成形时位姿被垃圾梯度拖走，自我强化进坏局部最优。
  代码数学核对无 bug——是策略问题。
- **BARF 式延迟启动（2026-10-10 补测，`POSE_REFINE_START=3000`）同样证伪**：
  drift 前 3000 步严格为 0、准点启用，但启用后 rot 0.96°→1.07°、trans 爬至 0.066 不止；
  test PSNR 17.95（best@5000）→16.35@10000，仍远劣于不精炼的 25.14。
  评测用冻结原始位姿——说明位姿 delta 在启用窗口内已不可逆地烤坏高斯几何。
  根因：手机位姿噪声下「动位姿」比「修几何」更易降 loss（118 个自由位姿当泄压阀），
  延迟只推迟不解决。该开关保留默认 0，作负结果存档。
- VGGT-Omega 前馈位姿（38.5s 出 135 帧）重投影目检同样对齐，但预测内参与真实
  固定内参有系统差（fx median 1397.6 vs 手机标定 1380.8，且逐帧浮动 MAD 6px），
  联合 pose+intrinsic 误差使其垫底。
- 稠密初始化救不了位姿误差：手机版带着 178k 稠密 pcd 初始化仍 -4dB——
  瓶颈是位姿/内参精度而非初始化点数（VGGT 版 5.4k 稀疏 init 的额外劣势无法从
  本组实验剥离，如需精确归因可加大 TARGET_POINTS 重训，但即便追回 1-2dB
  也不改变排序结论）。
- 副产物保留：01e（手机位姿转换管线）可在 **COLMAP 彻底失败**（弱纹理/重复纹理场景）
  时作降级 fallback，质量预期 -4dB。
- π³x 未试：与 VGGT 同属前馈类，证据强度不足以改变结论；如需补测成本 ~1h。

## Notes
- **C2F 与稠密化的交互（已解决）**：默认档下稠密化窗口（`densify_until_iter 2000`）
  完全落在 1/4 分辨率档，固定阈值会欠容量（v1：328k 点 vs 基线 921k，-0.46dB）。
  解法是低分辨率档按 **1/√scale** 放宽稠密化阈值（`--c2f_densify_comp`，默认开），
  三轮 A/B 剂量标定见「实验记录」。**改 densify 窗口或阈值默认值时须重新标定**。
- **先 vanilla 后锚定**：02 不加任何人体先验，先看 canonical+形变把重影消到几成，
  残差（单目欠定的背面/遮挡）再决定 Phase 2 的 MHR 锚定形态。锚定导出在 01 里是
  `ANCHOR_EXPORT=1` 占位，尚未实现。
- 图像文件名必须是纯数字（Deformable-GS `dataset_readers.py` 按时序算 fid）——
  01 委托的 COLMAP 脚本已自动重命名。
- Phase 2 设计讨论见 `media_paper` 调研笔记
  `VideoGen-20260929-视频生成驱动动态人体重建调研.html` §11。
