# EXPERIMENTS.md — vggt_human 实验结论与踩坑记录

> 本文件从 AGENTS.md 第 12 节迁出（2026-09-07）：AGENTS.md 只保留全仓统一规范，
> 项目专属的实验结论 / 消融数据 / 踩坑细节归档到各项目目录下。
> 新实验结论继续追加到本文件，按日期倒序排列。

## 位姿优化 A/B 对比实验

位姿优化（PoseAdjuster + PoseRefineModule）**默认开启**。以下命令方便跑有/无对比实验，用不同 `OUTPUT_NAME` 隔离结果。

### vggt_human（本目录：VGGT-Omega → 原版 3DGS）

```bash
# ── A: 无位姿优化（关闭）──
GPU=0 \
  POSE_ADJUST=0 POSE_REFINE=0 \
  RESULTS_DIR=../../output/vggt_human_results \
  bash vggt_human/04_train_3dgs.sh
# → $RESULTS_DIR/04_model_3dgs/

# ── B: 有位姿优化（默认开启，显式写出便于对比）──
GPU=0 \
  POSE_ADJUST=1 POSE_REFINE=1 \
  RESULTS_DIR=../../output/vggt_human_results \
  bash vggt_human/04_train_3dgs.sh
# → $RESULTS_DIR/04_model_3dgs/ (同目录，B 覆盖 A — 想保留则用不同 RESULTS_DIR)

# 更清晰：用不同目录隔离
GPU=0 POSE_ADJUST=0 POSE_REFINE=0 \
  GAUSSIAN_DIR=../../output/vggt_human_results/model_3dgs_no_pose \
  RESULTS_DIR=../../output/vggt_human_results bash vggt_human/04_train_3dgs.sh
GPU=0 POSE_ADJUST=1 POSE_REFINE=1 \
  GAUSSIAN_DIR=../../output/vggt_human_results/model_3dgs_pose \
  RESULTS_DIR=../../output/vggt_human_results bash vggt_human/04_train_3dgs.sh

# 对比渲染：
#   model_3dgs_no_pose/point_cloud/iteration_30000/point_cloud.ply
#   model_3dgs_pose/point_cloud/iteration_30000/point_cloud.ply
```

### pdfgs_human（Pi3 → PDF-GS，跨目录复用本目录的 pose_adjuster.py）

pdfgs_human 新增了 `02b_pose_adjust.sh`，在 Pi3→COLMAP 之后、PDF-GS 训练之前对 COLMAP 场景做一次性 PoseAdjuster 变换（不改 train.py）。PoseRefineModule 暂不支持 PDF-GS（训练循环不同）。

```bash
# ── A: 无位姿优化（标准流程）──
GPU=0 OUTPUT_NAME=orbit bash pdfgs_human/02_pi3_colmap.sh
GPU=0 OUTPUT_NAME=orbit bash pdfgs_human/03_train_pdfgs.sh
# → $RESULTS_DIR/orbit/model_pdfgs/

# ── B: 有位姿优化（跑 02b 再训练）──
GPU=0 OUTPUT_NAME=orbit_pose bash pdfgs_human/02_pi3_colmap.sh
GPU=0 OUTPUT_NAME=orbit_pose bash pdfgs_human/02b_pose_adjust.sh
GPU=0 SOURCE_DIR=$RESULTS_DIR/orbit_pose/pi3/source_adjusted \
  OUTPUT_NAME=orbit_pose bash pdfgs_human/03_train_pdfgs.sh
# → $RESULTS_DIR/orbit_pose/model_pdfgs/

# 对比渲染：
#   $RESULTS_DIR/orbit/model_pdfgs/phase_4/point_cloud/iteration_10000/point_cloud.ply
#   $RESULTS_DIR/orbit_pose/model_pdfgs/phase_4/point_cloud/iteration_10000/point_cloud.ply
```

### 开关速查

| 场景 | 命令 |
|---|---|
| vggt_human 关闭位姿优化 | `POSE_ADJUST=0 POSE_REFINE=0 bash ...04_train_3dgs.sh` |
| vggt_human 只做训练前变换（不学内参） | `POSE_ADJUST=1 POSE_REFINE=0 bash ...04_train_3dgs.sh` |
| vggt_human 全开 + 学内参 | `POSE_ADJUST=1 POSE_REFINE=1 REFINE_INTRINSIC=1 bash ...04_train_3dgs.sh` |
| 关闭全部增强（等价原始 3DGS） | `POSE_ADJUST=0 POSE_REFINE=0 ENABLE_DYNAMIC_MASK=0 ENABLE_DYNAMIC_FILTER=0 ENABLE_MLP_DYNAMIC=0 USE_DEPTH_NORMAL=0 bash ...04_train_3dgs.sh` |
| 只开动态掩码（P0-1） | `ENABLE_DYNAMIC_FILTER=0 ENABLE_MLP_DYNAMIC=0 USE_DEPTH_NORMAL=0 bash ...04_train_3dgs.sh` |
| 只开动态点过滤（P0-1+P0-2） | `ENABLE_MLP_DYNAMIC=0 USE_DEPTH_NORMAL=0 bash ...04_train_3dgs.sh` |
| 只开 MLP 动态感知（P0-1+P0-3） | `ENABLE_DYNAMIC_FILTER=0 USE_DEPTH_NORMAL=0 bash ...04_train_3dgs.sh` |
| 只开深度-法线约束（P1-1） | `ENABLE_DYNAMIC_MASK=0 ENABLE_DYNAMIC_FILTER=0 ENABLE_MLP_DYNAMIC=0 bash ...04_train_3dgs.sh` |
| pdfgs_human 关闭 | 不跑 02b，直接 02→03 |
| pdfgs_human 开启 | 跑 02→02b→03（换 SOURCE_DIR 指向 source_adjusted） |

## 人脸 finetune 消融结论（2026-09-04，3090 实测）

对照实验（同等训练量 20k 步 @ lr_scale=0.2，唯一差异 face_weight 0 vs 1.0）证明：
**人脸监督（masked-L1 → HYPIR 增强图）对人脸区锐度无可测量贡献**（+4.04% vs +3.98%，
逐帧差在噪声内）。此前 v1~v6 的全部人脸区提升来自"30k 基线欠收敛 + 更高 lr 继续训练"。

- **推荐配方**：`06d_continue_train.sh`（ITERATION=30000, EXTRA_ITERS=20000,
  LR_SCALE=0.2, FACE_WEIGHT=0）。训练视角人脸区 Laplacian +4~7%，全图 PSNR
  27.32→27.44+，LPIPS 同步下降，无过平滑代价
- **权重已饱和**：face_weight 1.0→2.0 结果逐像素持平（瓶颈是 lr_scale 不是权重）
- **lr_scale 是瓶颈**：0.1→0.2 把增益从 +2.0% 解锁到 +2.7%/10k 步
- **增益曲线**：~+1.3%/万步递减到 ~+0.8%/万步（80k 处未完全收敛）
- **新视角注意**：常规（人物居中）新视角明显更清晰且无伪影；极端离轴暗部视角
  纤维状条纹伪影增多，此类场景用较少续训步数（50k~60k）折中
- 若要超越 GT 收敛的人脸细节，masked-L1 机制无效，需换机制（对抗损失等），未验证

### 06e：增强近景整视角注入（2026-09-04，有效，与 masked-L1 机制相反）

把 06c 的 512 增强近景作为**完整训练视角**（真实相机位姿 + COLMAP 注入）喂回续训，
与 masked-L1（增强图仅作人脸区 GT）机制不同，**实测有效**（ms 数据集，4 人 × 18 近景）：

- 归因对照（同 20k 步 @ lr_scale=0.2）：
  - 06f 纯续训无注入：近景 LPIPS 持平（0.3048→0.3049 等）、锐度 +3~6%（噪声级）
  - **06e 注入**：近景 LPIPS 全降（p00 0.305→0.260、p03 0.202→0.137），
    锐度 +28%~+324%；训练视角人脸区 Laplacian +45% 且 LPIPS vs 原图不升（无退化）
- 结论：masked-L1 无效 ≠ 增强图无用；**整视角注入（整图参与 loss）才是正确用法**
- 链路：`inject_closeup_cameras.py`（pycolmap 4.2 `add_camera_with_trivial_rig` +
  `add_image_with_trivial_frame`，pose json R/T 即 COLMAP w2c）→ 场景 06e_source_closeup
  （222 views）→ 06d 配方续训
- 坑：start_ply 的 exposure.json 只有原图条目，注入图名在 Scene.save()/testing 会
  KeyError —— trainer 已补恒等 exposure fallback（30k exposure 本就收敛为恒等阵）
- p03（此前疑 p00 重复轨迹）实为真实第 4 人（红衣），近景提升最大，无需排除
- **round2 迭代注入（06g）**：06e@50k 重渲近景→再增强→再注入（同 04b 起点干净归因），
  统一 round2 GT 评估：06g vs 06e 近景 LPIPS 再降 10~23%、锐度再涨 +22~71%、
  PSNR +1.2~1.5dB；训练视角人脸区 Laplacian 11.4→13.3，LPIPS vs 原图持平（无幻觉回声）。
  **两轮未饱和，增益递减**（round1 训练视角 +45% vs round2 +17%）；第三轮预期收益有限，
  按需取舍。脚本 `06e_closeup_inject.sh`（注入+续训一条龙）

### SSIM 与人脸 L1 目标不一致（2026-09-05，已修）

`train_face_finetune.py` 的 masked-L1 路径里，L1 在人脸区内推向 HYPIR 增强图，
SSIM 项却一直全画幅对着原图算 —— 两项在人脸区内的更新方向近乎**正交**。
合成测试（模糊渲染 / 原图 / 增强图，人脸区中央 mask）测「更新方向」与
「补细节方向」的余弦：

| 项 | 余弦 |
|---|---|
| L1 人脸项 | +0.81 |
| SSIM 旧（对原图） | +0.18 |
| SSIM 新（composite） | +0.94 |
| 总 loss 旧 → 新 | +0.57 → +0.95 |

即 λ_dssim=0.2 的 SSIM 项把总方向从 0.81 **稀释**到 0.57（不是反向，是正交稀释），
masked-L1 的效果被吃掉一大半。修法：人脸区内把 SSIM 的比对目标也换成与 L1 同一张
composite（`gt + M·w·(enh − gt)`），区外仍是原图 → 无 mask 的帧与官方完全等价。
新增 `--face_ssim_mode composite|off`（默认 composite；off 仅供 A/B 复现旧行为）。

注意：这条路径目前不是主力（主力配方 FACE_WEIGHT=0 + 整视角注入），故 06e/06g
结论不受影响。修完后 masked-L1 能否起死回生**未重测**。

### HYPIR 推理提速（2026-09-05，3090 / 512² 实测）

优化点 1 的 TorchScript 被否决后定的替代路线，三项均已落地在 `face_enhance.py`
（`SPEED_MERGE_LORA` / `SPEED_CACHE_TEXT` / `SPEED_COMPILE`，前两项默认开）：

| 配置（累积） | ms/张 | vs base | 增量 |
|---|---|---|---|
| base | 170 | 1.00x | — |
| +cache_text | 161 | 1.05x | 1.05x |
| +merge_lora | 139 | 1.22x | 1.16x |
| +compile | 130 | 1.31x | 1.09x |

- **merge_lora**（默认开）：SD2.1 UNet 共 257 层 LoRA。peft 走的是
  `inject_adapter_in_model`（非 LoraModel 包装），没有 `merge_and_unload()`，
  需逐层调 `LoraLayer.merge()`。peft 的 forward 有 `elif self.merged: 走 base_layer`
  分支，**不会重复叠加 delta**，安全。数值漂移 0.35/255（≈0.14%），可忽略
- **cache_text**（默认开）：`BaseEnhancer.enhance()` 每图都重跑 tokenizer +
  text_encoder，而本流程 prompt 恒为 "" → monkey patch `prepare_inputs` 缓存
  `self.inputs`。text_encoder 确定性且不消耗 RNG → 输出逐位不变
- **compile**：**默认 auto（按批量门控，回本线 2400 张）**。增量只有 142→130ms
  （~12ms/张），一次性编译 ~28s → 约 2300 张才回本，小批量编译是净亏
  - **必踩的坑**：diffusers 的 `Attention.forward` 里有
    `inspect.signature(self.processor.__call__)`，dynamo 对每个 attention block
    的 processor 对象 id 加 guard，UNet 十几个 block 各要一份编译；默认
    `cache_size_limit=8` 撑爆后 dynamo 放弃编译、退回 eager 并保留 dynamo 开销，
    **实测 175ms 变成 1687ms（慢 10 倍）**。必须在 compile 前把
    `torch._dynamo.config.cache_size_limit` 抬到 64（已写进 `compile_unet()`）
  - `mode=default` 不如 `reduce-overhead`（编译 39s vs 28s，稳态 132 vs 129ms）
- **A/B 对比的坑**：`vae.encode().latent_dist.sample()` 每次前向都从固定 seed 的
  随机流取新样本 → 输出对「第几次调用」敏感，相邻两次前向差 mean|Δ|≈0.011。
  对比提速前后**必须对齐调用序号**（跑两次完整脚本比同名图片），否则量到假差异；
  同序号下逐位可复现（两次独立运行 mean|Δ| = 0.000/255）
- benchmark：`.tmp_diag/bench_hypir_speed.py`（`--only_compile` 只看编译收益与回本张数）

### 06d 脚本注意

- `train_face_finetune.py` 的 `FaceData` 只扫 `face_images_dir` **直接子文件**，
  目录参数必须指到图片平铺层（如 `06c_merged_face_images/images`，不是其父目录）
- `render_novel.py` 只读 txt 格式 COLMAP 模型；BA 场景（03b）sparse/0 只有 bin，
  已加 pycolmap 自动补 txt 的 fallback

## WSL 基线验证与增强模块实测（RTX 3090，2026-09-02/03）

> 从 README_wsl.md 迁入。`02 → 03 → 04 → 05` 官方基线全流程跑通（无任何增强开关），输入 125 张 1440×1920。

### 核心结论：渲染不再是黑图

| 指标 | 旧 `train_pose.py`（已归档） | 官方 7000 iter | 官方 30000 iter |
|---|---|---|---|
| 渲染平均亮度 | 0.8（全黑 + 白点） | 127.2 | **127.6** |
| GT 平均亮度 | 127.9 | 127.9 | 127.9 |
| 亮度比值 | ~0.006 | 0.995 | **0.998** ✅ |
| PLY 大小 | 17,912 高斯（远小于正常） | 214 MB | **290 MB** |
| 场景尺度 (x-span) | 162（爆炸） | 3.35 | 3.35 |

### 各步耗时（RTX 3090）

| 步骤 | 耗时 | 关键产物 |
|---|---|---|
| 02 推理 | 35s | 2,000,000 点，`extent=[3.35, 1.62, 1.72]` |
| 03 npz→COLMAP | ~1 min | 125 图 + ~200k 初始点 |
| 04 训练 7000 iter | 7 min | `iteration_7000/point_cloud.ply` 214 MB |
| 04 训练 30000 iter | **42 min** | `iteration_30000/point_cloud.ply` 290 MB |
| 04 渲染评估 | 2 min | `train/ours_{7000,30000}/{renders,gt}/` |
| 05 新视角 | 14s | `05_source_aug/` 130 cameras（125 原图 + 5 新增） |

> Loss 0.34 → 0.040。训练速度随高斯增密递减（17.6 → 9 it/s），所以 30000 iter 不是 7000 的 4 倍耗时。
> **7000 iter 亮度比值已达 0.995，赶时间够用；出最终交付 PLY 用 30000（0.998）。**
> ⚠️ 重跑 04 会覆盖同名 `iteration_*` / `ours_*` 目录（想留旧结果先备份）。

### 全流程复现验证（2026-09-03，增强加回之后）

所有增强模块加回后，用全新 `RESULTS_DIR` 完整重跑 02→03→04(30000)→05
（全部开关默认关闭，确认默认链路 = 纯官方基线不受影响）。
脚本 `.tmp_diag/run_verify_full.sh`，总耗时 50 min，四步全 EXIT=0，
04 分支确认走官方 train.py、dynamic 预处理被正确跳过。

| 指标 | 原 30000 基线（/mnt/d） | 全流程复现（~/vggt_human_verify） |
|---|---|---|
| PSNR（125 帧同源对比） | 27.37 dB | **27.37 dB** |
| L1 | 0.0219 | 0.0220 |
| 渲染/GT 亮度比值 | 1.002 | 1.002 |
| 初始点云 | 7,163 | 7,163（02/03 输出逐位一致） |
| 05 新视角 | 5 张达覆盖阈值 | 6 张达覆盖阈值（`05_source_aug` 129 cams） |

**结论：模块加回与开关化改造对默认链路零影响，基线完全可复现。**
（04 训练内评估 PSNR：7000=25.96、30000=26.92，与原基线一致量级。）

### 增强 P1-1：depth-normal consistency（已加回 + 修复）

加回 `depth_normal_cons.py` + `train_depth_normal.py`（官方 train.py 副本 + 单一 hook）。实测 30000 iter：

| 指标 | 官方基线 | depth-normal |
|---|---|---|
| 渲染亮度 | 127.6 | 126.0 |
| GT 亮度 | 127.9 | 125.7 |
| 亮度比值 | 0.998 | 1.002 |
| PLY 大小 | 290 MB | 292.6 MB |
| 训练耗时 | 42 min | 45 min |

> DN Loss 在 densify 结束后（>15000 iter）才启用，densify 阶段跳过（否则点数爆炸导致 8.5s/iter）。
> DN Loss 稳定在 0.46，约束生效但未损害亮度指标。几何质量提升需看 PLY 细节。

**修复了原模块的 3 个 bug：**
1. **法线公式错误**（致命）：`(-dz/dx/fx, -dz/dy/fy, 1)` → `(-Zu*fx, -Zv*fy, Z)`。原版把 fx/fy 乘除弄反（差 fx² 倍），nz 用常数 1 而非深度 Z，导致法线退化成 (0,0,1)、DN Loss≈1.0 等同随机噪声、约束静默失效。
2. **缓存尺寸不匹配崩溃**：densify 后点数变化，缓存的法线索引对不上当前点云，`tensor a (7163) must match b (9679)`。
3. **densify 阶段速度崩塌**：点数爆炸导致投影+赋值开销线性增长，5450 iter 后从 16 it/s 崩到 8.5 s/iter。改为 densify 阶段跳过 DN Loss。

**用法：**
```bash
GPU=0 INPUT_DIR=... RESULTS_DIR=~/output/vggt_human_results_dn \
  USE_DEPTH_NORMAL=1 DEPTH_NORMAL_WEIGHT=0.05 \
  bash vggt_human/04_train_3dgs.sh
```

### 增强 P1-2：pose_refine（已加回，但当前后端不可用）

加回 `pose_refine.py` + `train_pose_refine.py`（可学位姿：四元数 + 平移）。修完 3 个 bug
后暴露出**根本性限制**：

1. `world_view_transform` 原用 `torch.zeros(4,4)` + in-place 赋值构造 → 断梯度，改 `torch.cat`
2. `PoseRefinedCamera` 缺 `alpha_mask` 等属性 → 加 `__getattr__` 委托 base Camera
3. `_rotmat_to_quat` 用了 `np` 但 numpy 在条件块内 import → 移到顶部

| 梯度检查 | 结果 |
|---|---|
| `world_view_transform.sum().backward()` | 位姿参数**有**梯度 ✅（说明 cat 构造确实修好了梯度链） |
| `render(...)` 后 L1 loss 再 backward | 位姿参数 `grad=None` ❌ |

**根因**：`diff_gaussian_rasterization` 的 CUDA rasterizer 把 viewmatrix 当**常量**用于投影，
只对高斯参数（means/scales/rot/opacity/SH）求梯度，**不支持可微相机位姿**。
所以位姿精炼在当前后端下拿不到梯度，无法工作。

**可行替代（已实施 03b）**：① 换 gsplat 后端（原生支持可微位姿，但改动面大）；
② 在 03 之后加一轮 COLMAP bundle adjustment（不依赖 rasterizer 梯度）——**已实施，见 03b_colmap_ba.sh**。

### 增强 P1-2b：COLMAP BA（03b，固定内参只精修位姿）

`03b_colmap_ba.sh` + `colmap_ba.py`：用 pycolmap 在 03 输出上加一轮 BA，
**固定内参**（`refine_focal_length=False`, `refine_principal_point=False`,
`refine_extra_params=False`），只优化外参（位姿）+ 3D 点。

理由：VGGT-Omega 的逐视图内参质量已足够（fx 波动仅 ±1.2%，z-score 检查无突变），
松开内参会让 BA 有自由度迁就外参误差，可能把好内参拉坏。

```bash
RESULTS_DIR=~/output/vggt_human_verify \
  bash vggt_human/03b_colmap_ba.sh
# 输出: $RESULTS_DIR/03b_source_ba/sparse/0/{cameras,images,points3D}.txt
# → 重跑 04: SOURCE_DIR=$RESULTS_DIR/03b_source_ba bash vggt_human/04_train_3dgs.sh
```

> `USE_POSE_REFINE=1` 时 `04_train_3dgs.sh` 会打印告警并退化为官方 train.py，不会静默跑坏。

### 数据集动态性诊断（决定要不要开"噪声抑制类"增强）

加 `noise_negating` / `dynamic_mask` / `dynamic_filter` 之前，先判断这个数据集到底有没有
动态内容。用基线 30000 的 `renders/` vs `gt/`（125 帧）做统计：

| 指标 | 实测 | 解读 |
|---|---|---|
| 训练 loss（基线 30000） | 0.34 → **0.040** | 静态场景的典型收敛值；若有动态物体，3DGS 拟合不了会卡在 0.1 以上 |
| 误差空间集中度（top-20% 块贡献的残差） | **46.8%** | 误差高度集中，不是随机欠拟合 |
| 高误差块跨视角重合 IoU | 0.175（随机基线 0.111） | 略高于随机。但相机绕人转、各视角图像坐标系不可比，**该指标参考性有限** |

**结论：这是静态场景**（多视角拍摄静止人物）。误差集中在难重建区域（图像边缘、头发、
反光），而不是跨视角不一致的动态物体。

- `noise_negating` / `dynamic_mask` / `dynamic_filter` 三件套**大概率有害**：
  MLP 会把"始终高误差的静态难区"当成动态区域永久屏蔽，反而丢细节。
- 后续优化应转向**提升静态重建质量**（位姿精度、点云密度、几何约束），
  而不是动态区域抑制。

### 增强 P0-3：noise negating（已加回 + 修复，但对**静态人物场景有害**，默认关闭）

加回 `noise_negating.py` + `train_noise_negate.py`：DINOv2 ViT-S/14 提特征 +
轻量 MLP（384→16→1）在线学习每帧动态掩码，loss 只在静态像素上计算。

实测 7000 iter 同源对比（125 帧全量 `renders/` vs `gt/`）：

| 指标 | 官方基线 7000 | noise negating 7000 | 差异 |
|---|---|---|---|
| PSNR | 25.97 dB | 24.35 dB | **−1.62 dB** ❌ |
| L1 | 0.0269 | 0.0314 | +16.7% ❌ |
| 亮度比值 | 1.001 | 0.996 | −0.005 |
| 训练耗时 | 7 min | 10 min（另加 3 min DINOv2 加载） | 慢 ~85% |

**为什么不适用**：MLP 把"始终高误差的静态难区"（图像边缘、头发、反光）当成动态区域屏蔽，
这些区域失去监督后高斯既不被优化、也因 densify 梯度不足而缺发育 → PSNR 掉 1.6 dB。
训练中 `Static%` 稳定在 90~99%（只屏蔽 7% 左右），说明**MLP 自己学到了"场景基本静态"**，
但残余的这点屏蔽就足以造成明显损失。

> 与上面「数据集动态性诊断」完全吻合：**静态人物场景不要开动态抑制类增强**。
> `USE_NOISE_NEGATE` 默认 0，代码与文档保留，将来有真动态数据（街景、含行人）时再用。

**修复了原模块的 7 个 bug**（单元测试全过，见 `.tmp_diag/test_nn.py`）：

1. **MLP 在全分辨率跑**（致命）：原实现把 384 通道特征插值到 `(384,H,W)` 再跑 MLP，
   1440×1920 下会产生 ~4GB 中间激活。改为在 DINO 特征图 `(F,F)` 上推理再上采样 mask
   —— 顺带让监督信号（cosine 不相似度）与 MLP 输出分辨率对齐。
2. **mask 未 detach**（致命）：3DGS 重建 loss 会顺着 mask 反传到 MLP，MLP 为最小化重建
   loss 会学会"屏蔽所有高误差区域"，形成对抗性塌缩。已 detach，单元测试验证 MLP 参数无梯度泄漏。
3. **masked L1 除以全像素数**：loss 被系统性缩小，与官方 `lambda_dssim` 配比、densify
   梯度阈值语义脱节。改为除以静态像素数（全 1 mask 时与官方 L1 逐位相同）。
4. **无动态比例兜底**：MLP 随机初始化输出饱和在 0.5，固定阈值 0.25 会让几乎全图判为
   动态、loss 归零崩塌。改为 `thr = max(固定阈值, 分位数(1−NN_MAX_DYNAMIC_RATIO))`，
   即使输出饱和也能保证 ≥50% 像素参与 loss（实测饱和时仍保持 66%）。
5. **残差边界项方向写反**：原 `relu(mask−upper)+relu(lower−mask)` 在"确定静态"区
   反而把 mask 推向 1（dynamic）。反证：旧实现给"正确标注"打 **1.0000** 分、
   给"标注反了"打 **0.0000** 分。已按 mask 语义（1=dynamic）转换边界。
6. **masked SSIM 先乘 mask**：屏蔽区两图同为 0 → SSIM≈1，虚低。反证：旧做法
   **0.8756** vs 左半真实 SSIM **0.7513**。改为 ssim map × mask 加权平均。
7. **接入点在 `no_grad` 块内**：官方 train.py 从 `with torch.no_grad():` 一直包到循环
   末尾，MLP 的 `loss.backward()` 直接报
   `element 0 of tensors does not require grad`。已用 `torch.enable_grad()` 包住。

**用法（不推荐用于静态人物场景）：**
```bash
GPU=0 INPUT_DIR=... RESULTS_DIR=~/output/vggt_human_results_nn \
  USE_NOISE_NEGATE=1 NN_MAX_DYNAMIC_RATIO=0.2 NN_WARMUP_EPOCHS=15 \
  bash vggt_human/04_train_3dgs.sh
```

### dynamic_mask + dynamic_filter 实测（2026-09-03）

机制：**训练前的点云预处理**（不是训练中的 loss）。`dynamic_mask.py` 用
GroundingDINO（文本→框）+ SAM2.1（框→掩码）给每帧训练图生成动态掩码；
`dynamic_filter.py` 把初始点云投到所有视角做多视角投票，落入掩码比例
> `DYNAMIC_THRESHOLD`(0.3) 的点删除——从源头减少动态物体污染的高斯。

**环境**：`sam2` 包（gh-proxy 镜像装，`--no-build-isolation` 复用环境 torch 2.5.1；
`configs/` 目录需从源码树补拷进 site-packages），GroundingDINO 走 hf-mirror 自动下载。
SAM2 config 文件名是缩写（`sam2.1_hiera_l.yaml`），代码已按 checkpoint 名自动映射。

**集成**：`04_train_3dgs.sh` 训练前新增第 0 步——
`ENABLE_DYNAMIC_MASK=1` 生成掩码（缓存于 `$RESULTS_DIR/dynamic_mask`，
`FORCE_DYNAMIC_MASK=1` 强制重生成）→ `ENABLE_DYNAMIC_FILTER=1` 过滤
`03_source/sparse/0/points3D.ply`（原始点云备份为 `points3D.ply.orig`，可随时还原）。
⚠️ prompt 默认 `TV screen monitor`，**刻意不含 person**：静态人物数据集里
person 是主体，过滤它等于删主体。真动态场景用
`DYNAMIC_PROMPTS="person TV screen"` 显式传。

**实测（7000 iter，prompt=person 刻意验证机制链路）**：

| 指标 | 官方基线 7000 | dynamic_mask+filter 7000 |
|---|---|---|
| 掩码生成 | — | 125 帧 ~23% dynamic，SAM2.1-large + GroundingDINO-tiny，~4 min |
| 初始点云 | 7,163 | 3,962（删 44.7%） |
| PSNR | 25.97 dB | 25.59 dB（−0.38 dB） |
| L1 | 0.0269 | 0.0281（+4.5%） |
| 训练速度 | ~17 it/s | ~13 it/s（点云减半后 densify 更快回血） |

结论：**机制端到端有效且损害温和**。即使删掉近半初始点云，训练中的 densify
也会重新长出被删区域（GT 监督仍含 person），只掉 0.38 dB。静态场景默认关闭
（`ENABLE_DYNAMIC_MASK/FILTER` 默认 0）；真动态场景（行人、屏幕闪烁）下
预期是正收益，因为那时删掉的点是被污染的。

**修复了原模块的 5 个 bug**（单元测试 `.tmp_diag/test_df.py` 全过）：

1. **GroundingDINO 后处理 API 变更**：transformers ≥4.46 把
   `post_process_grounded_object_detection` 的 `box_threshold=` 改名 `threshold=`，
   参数名不对抛 TypeError，且不在旧代码 `except AttributeError`范围内 → 直接崩。
   改为多组参数名依次尝试。
2. **SAM fallback 输入框格式错**：`SamProcessor` 期望 `[[[x1,y1,x2,y2],...]]`，
   原代码写成角点对 `[[[x1,y1],[x2,y2]]]`。
3. **有效视角分母只数有掩码的相机**（单测抓出）：原实现 `if name not in masks:
   continue` 跳过无掩码相机、分母不累加 → "125 帧只有 1 帧有掩码"时 ratio 恒为
   1/1，过度删除。改为分母数全部可见视角。
4. **过滤输出缺 nx/ny/nz 字段**：官方 `fetchPly` 读这三个字段，缺了直接
   `ValueError: no field of name nx`（外层静默吞成 `point_cloud=None` →
   `create_from_pcd` 崩 `'NoneType' object has no attribute 'points'`）。
   输出补齐字段（置 0）。
5. **`proxy.env` 无条件覆盖 `RESULTS_DIR`**：`_env.sh` 第 12 行先 source
   proxy.env，把外部传入的 RESULTS_DIR 冲掉 → 曾导致掩码/过滤写进原数据集目录
   （靠 `.orig` 备份恢复）。改为 `:-` 条件赋值（00a_setup_env.sh 同步修）。

**用法（真动态场景）：**
```bash
GPU=0 INPUT_DIR=... RESULTS_DIR=~/output/vggt_human_results_dmf \
  ENABLE_DYNAMIC_MASK=1 ENABLE_DYNAMIC_FILTER=1 \
  DYNAMIC_PROMPTS="person TV screen" \
  bash vggt_human/04_train_3dgs.sh
```
