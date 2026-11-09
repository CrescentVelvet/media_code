# da3_human — 实验记录

> 按日期倒序追加。实验计划/决策依据见 `EXPERIMENT_PLAN.md`，机制解释见 `NOTES.md`（阶段E补）。

---

## 2026-11-09 C-06 终止判定：densify 才是分水岭（P1 甜点在本场景复现）

> 起因：峰哥质疑「PSNR 15.2、1 小时，DA3 高斯头本来是秒级」。
> 复核后确认 **06 是双输**（比零训练的官方直出还低 2.7 dB，却慢 4.7 倍），
> 根因不是范式（前馈 vs 优化），而是 06 丢掉了 3DGS 的 densify。

### 一、决定性对照：最终点数几乎相同，PSNR 差 8.58 dB

**实验**：`04_run_da3_gs_baseline.sh` + `MAX_INIT_POINTS=20000 ITERATIONS=5000`
（= `compare_4methods` P1 ⑤ 的「DA3 稀疏点云 + 短步数 + 开 densify」在本场景的严格复现）

| 配置 | init 点 | **最终点** | 步数 | 训练耗时 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|---|---|---|---|
| 官方 DA3-GS 直出 | — | 1891 万 | 0 | 203 s 前向 | 17.93 | 0.7104 | — |
| 06 v3 anchor + 2DGS | 683,879 | **683,879**（不动） | 30K | 3,779 s | 15.209 | 0.6489 | — |
| **P1 DA3稀疏 + 3DGS** | **18,057** | **616,420**（densify 长大） | **5K** | **461 s** | **23.79** | **0.8200** | 0.2570 |
| 04 DA3点云 + 3DGS | 459,433 | ~500 万 | 30K | ~3,600 s | 25.505 | 0.8613 | 0.1870 |
| 03 COLMAP + 3DGS | 7,163 | — | 30K | 2,751 s | 27.573 | 0.8854 | 0.1642 |

**这张表的核心是第 2、3 行**：最终点数 68.4 万 vs 61.6 万（几乎相同），
PSNR 15.21 vs 23.79（差 **+8.58 dB**），训练耗时 63 min vs **7.7 min**（快 8.2 倍）。
点数是同一量级，唯一的结构差别是「点从哪来」：

- **densify**：点在梯度大（细节/轮廓）处自动 split/clone，在 opacity<0.005 处 prune；
  scale 与位置自由学习，贴合真实表面 → 61.6 万点全是有用的。
- **voxel 硬分**：7.2 mm 均匀网格，与几何无关；scale 初值锁在 voxel 尺寸；
  位置被 ±1px/±2% 的 offset 界焊死 → 68.4 万点里大量是均匀铺开、无处可动的"占位符"。

⇒ **densify 决定的是「点的质量分布」，不是「点的数量」**。这是本次最可复用的结论。

### 二、06 的四个可定位缺陷（全在 `06_train_da3_2dgs.py`）

| # | 缺陷 | 位置 | 后果 |
|---|---|---|---|
| 1 | **完全没有 densify** | `M = len(xyz)` (L106)，五个参数全是 `(M,·)` | 30K 步一个点都没长，容量锁死 68.4 万 |
| 2 | 有界 offset 把错误初值焊死 | `bound_cam` L119-121（±1px/±2%），`LR_GEO=1e-4` L164 | B-04b 已证 DA3 深度噪声是主因，此处明令禁止修正 |
| 3 | 每 iter 从磁盘读图 | `load_image(img_paths[vi])` L217 | 1440×1920 JPG 逐帧 `cv2.imread` 解码，无缓存 |
| 4 | 每 iter 两次全量光栅化 | L225（全分辨率）+ L236（1/4） | 单步成本翻倍 |

**但 06 的单步并不慢**：04 是 120 ms/it，06 是 126 ms/it，基本持平。
所以「慢」不是实现低效，而是 **30K 步全在做无用功**——63 分钟换来比零训练低 2.7 dB 的结果。

### 三、2DGS 不是元凶（避免错误归因）

`compare_4methods` I-03 实测：同一 PLY、同一组相机、**只换光栅器**，全图仅 −0.29 dB。
所以这 10.3 dB 的差距（04 3DGS 25.5 vs 06 2DGS 15.2）**不能记在 2DGS 账上**，
只能记在「关 densify + 锁几何」上。2DGS 仍保留一个独立用途：
`median_depth` 掠射一致性 0.41% vs 3DGS 0.80%，做**几何**时单独用，不做外观主链。

### 四、判定

| 项 | 判定 |
|---|---|
| C-06 双分支 2DGS 作为外观主链 | ❌ **终止**（负结果封存，不再投入） |
| 「DA3 几何 → 稀疏化 → 3DGS + densify」 | ✅ **确认为主链** |
| 05 anchor 点云（683,879） | ⚠️ 降级为「几何分析/初始化候选」，不再是训练终点 |
| 盘面法向诊断结论 | 保留（identity 代价 4.55 dB 成立；「朝源相机」修法不成立） |

### 五、复现

```bash
# 结果：PSNR 23.79 / SSIM 0.8200 / LPIPS 0.2570，端到端 13m26s（训练 7m41s）
ITERATIONS=5000 MAX_INIT_POINTS=20000 \
SCENE_DIR=$RESULTS_DIR/da3_colmap_scene_p1 \
GAUSSIAN_DIR=$RESULTS_DIR/p1_da3sparse_5k \
bash da3_human/04_run_da3_gs_baseline.sh
```

**下一步候选**（待定，按性价比）：
1. 扫 `MAX_INIT_POINTS` ∈ {20k, 50k, 200k} × `ITERATIONS` {5k, 10k}，找本场景的性价比拐点
   （04 的 46 万点 + 30K = 25.505，说明还有 1.7 dB 空间）；
2. 把 `07_eval_da3_gs.py` 的 sweep 能力接过来，做**留出视角** NVS 口径（当前 125 视角全 in-sample）。

---

## 2026-11-09 官方 DA3-GS 基线复现 + C-06 盘面法向诊断

> 依据峰哥「先复现官方效果，再以它为唯一参照」的规矩。**本场景的官方参照不是 22.11**：
> compare_4methods 的 22.11 出自 `佳佳抱枕_p1m`（32 视角），studio_human 是 125 视角开弧采集，
> 两者不可混用（ERRATA E6 同理）。

### 一、官方 DA3-GS 在本场景的实测基线 = 17.93 dB

| 项 | 值 |
|---|---|
| 推理 | `INFER_GS=1 EXPORT_FORMAT=mini_npz-gs_ply`，125 视角 504p，前向 203s（另有 87s 加载） |
| 内存高斯 | **23,814,000**（GSDPT 头一次前向直出） |
| 导出 PLY | 1.2GB / **18,911,929** 点（官方默认剪枝 `prune_by_depth_percent=0.9` + `prune_border_gs=True`） |
| PLY 字段 | 17 个，**含 nx/ny/nz 法向**（DA3 自带法向输出） |
| SH | 内存 `harmonics (1,N,3,9)` → **degree 2**；默认导出 `save_sh_dc_only=True` 只剩 DC |
| opacity | logit 中位 −4.39 → **α 中位 0.0122**（远比 compare_4methods 的 0.036 低） |
| **scale_factor** | **1.8666**（DA3 `is_metric=1`；gaussians 同样未乘，E-01 复现） |

**尺度口径验证（ERRATA E2 增益扫掠）**：手扫 sf ∈ {1.0,1.2,1.5,1.8,2.0,2.2,2.5}，
PSNR 为 11.72 / 11.79 / 13.25 / **16.88** / 15.27 / 14.00 / 13.20 → **峰位 1.8**，
与官方内部 `scale_factor=1.8666` 一致（粗档落在 1.8~2.0 之间）。不折算会白丢 ~5 dB。

**正式评估（全 SH + 米制，1891 万点，125 视角全量）**：

| 口径 | PSNR | SSIM | per-view p10 / p90 | min / max |
|---|---|---|---|---|
| 504p（官方口径） | **18.11** | 0.6696 | 17.0 / 19.6 | 11.6 / 20.4 |
| 1440x1920（交付口径） | **17.93** | 0.7104 | 16.9 / 19.4 | 11.5 / 20.1 |

**与既有基线对照（统一 1440×1920 / 125 视角 / in-sample）**：

| 配置 | 点数 | PSNR ↑ | SSIM ↑ | 备注 |
|---|---|---|---|---|
| 03 COLMAP + 3DGS 30K | 7,163 init | **27.573** | **0.8854** | 有 COLMAP BA |
| 04 DA3 点云 + 3DGS 30K | 459k init | 25.505 | 0.8613 | 有 30K 训练 |
| **官方 DA3-GS 直出** | **1891 万** | **17.93** | 0.7104 | **零训练** |
| 06 v3 anchor + 2DGS 30K | 684k | 15.209 | 0.6489 | 有 30K 训练 |

⇒ **两个必须记住的事实**：
1. **官方 DA3-GS 在本场景只有 17.93 dB，比 03 低 9.6 dB**。它"效果好"是 32 视角数据集的结论，
   在 125 视角开弧场景（闭合差 1.42 m、视角跨度大）不成立——这与「DA3 质量随相机到参考视角
   距离单调衰减」一致，125 视角把远端推得更远。
2. **06 v3 距官方直出仅 2.7 dB**（15.21 vs 17.93），而不是之前误以为的 6 dB。
   追赶目标应是 17.93（先不亏），再谈 27.57。

### 二、C-06 盘面法向诊断（`quats` 初始化是否锁死世界 +Z）

**背景**：06 把 `quats` 初始化为 identity，gsplat 2DGS 的盘法向 = 旋转矩阵第三列（ERRATA E10），
identity → 全部法向朝世界 +Z。DA3 视角绕人一圈，若法向锁在 +Z，多数视角是掠射。

**实测（v3 checkpoint，683,879 高斯，4 视角零训练对照渲染）**：

| 指标 / 配置 | 值 |
|---|---|
| identity quat 残留占比 | **0.0000**（训练后全动过） |
| cos(盘法线, 源相机方向) | mean −0.252 / **med −0.382** / p10 −0.898 / p90 +0.656 |
| \|cos\|>0.9（正对相机）占比 | **13.6%** |
| \|cos\|<0.3（掠射）占比 | **26.2%** |
| cos(盘法线, 世界+Z) | med **+0.868**，\|cos\|>0.9 占 **46.2%** |

| quats 配置（其余参数不变） | PSNR(4 视角) | per-view | alpha mean |
|---|---|---|---|
| as-is（训练后） | **16.21** | [20.5, 15.7, 12.9, 15.8] | 0.922 |
| identity（全朝世界+Z，= 初始状态） | **11.66** | [10.2, 14.8, 12.3, 9.4] | 0.672 |
| 盘面朝源相机 | 14.63 | [16.8, 14.9, 12.1, 14.7] | 0.946 |
| 仅掠射档(26.2%)改朝相机 | 15.41 | [18.0, 15.5, 12.7, 15.4] | 0.941 |

**结论（推测部分成立，修法不成立）**：
- ✅ **identity 初始化的代价是 4.55 dB**（11.66 → 训练后 16.21），这是真实损失，初始化确实有害；
- ❌ **但「盘面朝源相机」不是正解**（14.63 < 16.21，反而低 1.6 dB）。一个高斯被多视角共享，
  盘法向应对齐**多视角一致的表面法向**，而不是任何单个相机方向；
- ⚠️ v3 的 alpha mean 0.922 / \|α>0.5\| 0.959，**覆盖已满**，不是 v1 那种「半透明叠化」问题；
  真正的短板在几何与点数密度（684k vs 官方 1891 万）。

**下一步候选（按性价比排序）**：
1. **anchor 密度对齐官方**（68.4 万 → 数百万，体素 7.2mm → ~2mm）：官方 1891 万点，我们差 28 倍
   （表面积缩放 → 线性间距差 ~5.3 倍）。这是唯一"零算法风险"的追赶动作。
2. 用 **DA3 自带的 nx/ny/nz 法向**初始化 quats（PLY 里有，官方白送），替代 identity。
3. 有界 offset 的界值（当前 ±1px / ±2% 深度）在点数提上去后需重扫。

---

## 2026-11-05 阶段 C-06 v2/v3：梯度隔离消融——detach 是主凶，v3 去 detach 重跑

> ⛔ **路线已于 2026-11-09 终止**（见本篇开头「C-06 终止判定」）：去 detach 只解决梯度隔离，
> 但 densify 缺失才是量级问题。以下内容保留作负结果追溯。

### C-06 v2：PSNR 15.102 —— opacity 修复后仍被 detach 拖累

v1 修复（纹理分支 `opac.detach()` → 可导）后重跑 30K：**PSNR 15.102**（比 v1 仅 +0.23dB）。
训练速度反而恢复 3.5×（3,779s vs v1 12,372s），排除缓存膨胀因素，锁定**双分支梯度隔离设计本身**。
v2 checkpoint opacity 已恢复学习（mean 0.119→0.19、max 1.0），但 tex L1 轨迹与 v1 几乎一致——
系统卡在同一个 basin。

### 隔离消融（同 anchor 起点、8K iters、单视角循环，排除多视角随机性）

| 配置 | 关键设计 | PSNR | 说明 |
|---|---|---|---|
| **A** | 单分支无 detach + 正则（dist/norm） | **21.05** | 上限参考（30K 预计 +1~2dB） |
| B | detach 几何 + 10×lr + 正则 | 17.51 | detach 仍拖后腿 3.5dB |
| C | detach 几何 + 默认 lr + 正则 | 19.21 | lr 放大可救一部分 |
| D | **anchor + 有界 offset 参数化 + 正则，无 detach** | **20.65** | 与 A 相当，核心参数化成立 |
| **v2 原配置** | detach 几何 + detach 颜色 + 有界 offset | 15.10（30K） | detach 双重隔离是主因 |

**结论**：`detach`（梯度隔离）是主凶，与之前发现的 opacity 冻结同根源——纹理分支
detach 几何参数后，1/4 分辨率几何分支成了几何参数的唯一梯度来源，信号太弱/太噪。
D 配置证明：**anchor + 有界 offset 参数化 + 正则（无 detach）可接近 free-means 上限**。

### C-06 v3：去 detach + 双路共享梯度（运行中）

改动：纹理分支去掉 `means/quats/scales.detach()`，几何参数在纹理（全分辨率）和
几何（1/4 分辨率+正则）双路都有梯度。预期 30K PSNR ≥ 22（8K 消融 A/D 水平 + 收敛增益）。

- 产物：`da3_2dgs_v3/ours/`（v2 目录已归档 `da3_2dgs_v2_detach/`）
- 消融脚本：`/tmp/abl_06.py`（未入库，一次性诊断；口径与 06 评测完全一致）

---

## 2026-11-05 阶段 C-06：双分支 2DGS 首跑（负结果）与根因修复

> ⛔ **路线已于 2026-11-09 终止**。v1 的 opacity 冻结是真实 bug、v2/v3 的 detach 归因也成立，
> 但整条路线缺少 densify，最终点数锁死在 68.4 万 → 无法与官方 1891 万 / 03 基线 300~500 万竞争。

### C-06 v1：PSNR 14.877 —— 不透明度冻结 bug

| 项 | 值 |
|---|---|
| 配置 | 684k anchors（C-05）+ 有界 offset（±1px/±2%深度）双分支 2DGS，SH3，30K iters |
| 结果 | **PSNR 14.877 / SSIM 0.6347 / LPIPS 0.3962**（vs 03 基线 27.573） |
| 耗时 | 训练 12,372s（末段 ~412ms/it，前期 128ms/it，疑与缓存分配器膨胀+逐 iter 磁盘读图有关） |

**症状**：结构完全对齐（墙/桌/柜位置全对），但全图「叠化糊」——人脸帽檐水平拖影、
细纹理（藤编/发丝）全丢，像每像素叠了几十层半透明 splat。

**根因（诊断脚本实锤）**：`06_train_da3_2dgs.py` 纹理分支把 `opac.detach()` 传进
光栅化（原意只 detach 几何），几何分支也 detach —— **不透明度在两个分支都无梯度，
30K iters 后全体冻结在初始 σ(−2)=0.119**（checkpoint 实测 mean/median/p90/max 全 = 0.1192）。
低透明度 × 684k splats 重叠 → 逐像素多层 alpha 叠化 → 全局模糊。
对照实验：强制 opac=1 反而更差（11.90dB，背景层全挡死），证实是「叠化」而非「几何错」。

**修复**：纹理分支恢复 `opac` 可导（opacity 归纹理分支学），几何分支保持 detach；
训练结束新增 opacity 终态打印（std≈0 即告警）。v2 重跑进行中，结果随后补录。

---

## 2026-11-05 阶段 B-04b：内参假设隔离（负结果）

### 实验设计

04 落后 03 达 −2.07dB，三条假设排序后做隔离实验：DA3 点云/位姿不动，**仅把每相机内参按
图像名替换为 COLMAP 标定值**（cam0: DA3 1448.4/1387.2 → COLMAP 1389.6/1399.8），
区分「内参系统性偏差（假设1）」vs「深度噪声+densify 膨胀（假设2）」。

- 场景：`da3_colmap_scene_colmapintr/`（459,433 点，构建单独验证通过）
- 脚本：`build_da3_colmap_scene.py` 新增 `INTR_REF_DIR` 分支 + `04b_run_da3_gs_colmapintr.sh`

### 结果 ✅ —— **假设 1 排除，主因锁定深度噪声（假设 2）**

| 配置 | init | 内参 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|---|---|
| 03 COLMAP 全套 | 7,163 | COLMAP | **27.573** | **0.8854** | **0.1642** |
| 04 DA3 全套 | 459,433 | DA3（fx≠fy 差4%） | 25.505 | 0.8613 | 0.1870 |
| **04b DA3几何+COLMAP内参** | 459,433 | COLMAP | **25.374**（−0.13dB） | 0.8564 | 0.1946 |

- 按预先声明的判定标准（回升 <0.5dB → 假设 2 成立）：换内参**无回升反而略降**，
  DA3 预测内参偏差**不是**性能差距主因，−2.07dB 主要来自**深度噪声经 densify 放大的
  过平滑/摊平**（与 som_human「更多高斯更易摊平」经验一致：04/04b 最终 ply 均 >03）。
- 推论：阶段 C 不必强依赖 COLMAP 内参（零成本论证不成立但也不是瓶颈）；**核心矛盾在
  点云质量**——anchor 一致性筛选 + 有界 offset（D1） 的权重上调。
- 副产物：`pycolmap.Reconstruction()` 要求 cameras/images/points3D 三文件齐（只拷
  cameras/images 会读失败）；参考模型是 `.txt` 格式（预检已兼容）。

### 阶段 B 小结（03/04/04b 三配置收束）

| 配置 | 点云来源 | 位姿 | 内参 | PSNR |
|---|---|---|---|---|
| 03 | COLMAP sparse | COLMAP | COLMAP | **27.573** |
| 04 | DA3 反投影 | DA3 | DA3 | 25.505 |
| 04b | DA3 反投影 | DA3 | COLMAP | 25.374 |

**结论：在 per-scene 优化框架内，DA3 几何直出替换 COLMAP 当前不成立（−2.1~−2.2dB），
差距由深度噪声主导而非内参。** 这正是阶段 C（anchor 一致性筛选 + 双分支 2DGS +
有界 offset）要解决的问题——若 C 阶段能追平/反超 03，即证明「筛选+参数化」路线有效；
若不能，DA3 侧需升级到带 BA 精化的几何（退出本计划范围，回 BACKLOG 讨论）。

## 2026-11-05 阶段 B-03/04 基线训练

### 阶段 B-03：原版 3DGS 基线（COLMAP init）✅

| 项 | 结果 |
|---|---|
| 初始化 | COLMAP sparse **7,163 点**（vggt_human source，即现有 scene_gs 路径） |
| 训练 | 30,000 iters，全分辨率 1440×1920，**45m51s**（RTX 3090） |
| 训练内评估 | [ITER 30000] PSNR **27.10**（官方 train.py 日志口径） |
| 渲染后评估 | train 视角 125/125：PSNR **27.573** / SSIM **0.8854** / LPIPS **0.1642** |
| 产物 | `~/output/da3_human_results/baseline_3dgs_colmap/`（point_cloud/iteration_30000、train/ours_30000/{renders,gt}） |

定性观察（floater/重影对照现象）待截图补入——需与 04（DA3 init）同视角对比后统一记录。

**metrics.py 全 nan 根因（"Metric evaluation progress: 0it"）**：
1. 本仓 `metrics.py`（vggt_human 定制版）只扫 `test/` 分割，而 render.py 渲染的是 train 视角
   → 建软链 `test/ours_<iter> -> ../train/ours_<iter>` 解决；
2. ⚠️ 坑中坑：**render.py 会预先给 test split 建空的 `renders/`、`gt/` 目录**，直接
   `ln -sfn` 会把链接塞进该目录内部（`test/ours_30000/ours_30000 -> ...`），metrics 仍读空目录。
   必须先 `rmdir` 空目录再软链。`_train_3dgs.sh` 已内置该逻辑（含 rmdir），04 起自动生效。

### 阶段 C-05：anchor 点云构建 ✅

| 项 | 结果 |
|---|---|
| 输入 | 02 的 DA3 导出（深度 125×504×378 @504p，conf/extrinsics/intrinsics） |
| 筛选链 | conf≥P40（阈值 1.49）→ 8 最近邻视角深度一致性（τ_rel=3% + τ_abs=5cm，≥3 视角一致）→ 跨视角体素去重（voxel=0.004×中值深度=**7.2mm**） |
| 数量 | 反投影 **14,334,402** → 一致性后 **13,841,023**（保留 96.6%）→ 去重后 **683,879** anchors |
| 产物 | `~/output/da3_human_results/anchor_pointcloud/studio_human/`（anchors.npz / anchors.ply / anchor_summary.json） |

**观察**：
- 与 02 colmap 导出点数（14.29M）吻合，反投影链一致 ✓。
- 一致性保留率 96.6% 偏高：DA3 深度与位姿同一网络产出、天然自洽，不一致集中在遮挡边界
  （view 124 边缘视角 60986→7033 即典型）。τ 目前是宽松档，D 阶段消融扫 τ 时再收紧验证。
- anchor 密度 68 万 surfel 规模适中（原版 3DGS 03 基线 30K iters 后约 300~500 万高斯），
  双分支 2DGS 训练量可控。

### 阶段 B-04：DA3-GS 基线（DA3 点云初始化）✅ —— **负结果：朴素 DA3 替换全面落后 COLMAP**

| 项 | 结果 |
|---|---|
| 初始化 | DA3 深度反投影体素降采样 **459,433 点**（conf≥P40）+ DA3 自估位姿/内参 |
| 训练 | 30,000 iters，全分辨率，训练段 ~60min（densification 膨胀：iter 7000 已 ~5M 高斯 / ply 339MB） |
| 训练内评估 | [ITER 30000] PSNR **25.26** |
| 渲染后评估 | train 125/125：PSNR **25.505** / SSIM **0.8613** / LPIPS **0.1870** |
| 最终 ply | 383MB（03 对照 298MB） |

**03 vs 04 对照总表**（口径一致：train 视角 in-sample，30K iters 全分辨率）：

| 基线 | init 点数 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | ply |
|---|---|---|---|---|---|
| 03 COLMAP init | 7,163 | **27.573** | **0.8854** | **0.1642** | 298MB |
| 04 DA3 init | 459,433 | 25.505 (−2.07dB) | 0.8613 | 0.1870 | 383MB |

**负结果分析（三条假设，按嫌疑排序，均待验证）**：
1. **DA3 预测内参偏差**：04 内参 fx=1444.61 **≠** fy=1389.17（差 4%），而 COLMAP 标定 fy=1399.84
   （正方形像素相机）。内参系统性偏差 → 全视角 misprojection → 纹理对不齐（糊化）。
   → **最可疑**，且可零成本验证：04 场景换 COLMAP 内参重跑。
2. **深度噪声 + densification 膨胀**：459k 含噪 init → 7000 iter 即 5M+ 高斯，
   噪声点被 densify 成半透明羽化 blob（ply 反大 28%，PSNR 反降，符合"越多高斯越摊平"的历史经验）。
3. **位姿一致性误差**：02 验收 Sim(3) 对齐 mean 4.9mm 是**对齐后**残差；04 用 DA3 原始坐标系，
   残差直接进入训练（但量级小，嫌疑低于 1/2）。

**定性观察**（`baseline_compare/compare_*.png`，view 0/30/60/90 三联）：
- in-sample 视角两基线都**无明显 floater/重影**——同事观察的跨视角不一致问题在
  in-sample 上不显著，NVS（留出视角）才能充分暴露。⚠️ 当前 125 视角全用于训练，
  **无留出测试视角**，NVS 口径待补（D 阶段或重跑带留出）。
- 04 的差距表现为**整体锐度下降**（面部/头发/货架细节糊化），与 LPIPS +0.023 一致；
  非局部坏点，更像全局对齐失配——支持假设 1。

**对核心方案的意义**：朴素「DA3 几何直接替换 COLMAP」在本场景**不成立**（−2.07dB）。
阶段 C 的有界 offset + 双分支 + pose residual（D5）动机更充分——不能只信任 DA3 几何，
要给它信任域内的修正能力。

---

## 2026-11-05 阶段 A/B 启动

### 执行前调研发现（修正计划前提）

1. **⚠️ DA3 权重许可矩阵（README Model Zoo 实测）——修正计划 §6 前提**：

   | 能力组合 | 可选模型 | 权重许可 |
   |---|---|---|
   | 相对深度 + 位姿 | DA3-BASE(0.12B) / DA3-SMALL(0.08B) | Apache 2.0 |
   | 度量深度（单目，无位姿） | DA3METRIC-LARGE(0.35B) | Apache 2.0 |
   | **度量深度 + 位姿一体** | **DA3NESTED-GIANT-LARGE-1.1(1.40B) / DA3-GIANT-1.1(1.15B)** | **CC BY-NC 4.0** |

   - 计划 §7 说「depth-anything/Depth-Anything-3 (Apache 2.0)」——仅**代码**许可成立；
     计划 §6「商用 pose-free 管线（DA3 Apache 2.0 解许可）」在**权重层面不成立**：
     度量+位姿组合无 Apache 选项。商用化路径只剩：BASE/SMALL（相对深度+位姿，尺度靠 COLMAP/BA）或等官方出 Apache 度量模型。
   - **本期选择**：内部验证用 `DA3NESTED-GIANT-LARGE-1.1`（质量最高：README 称 DA3-LARGE 已可比 VGGT，NESTED 为 any-view giant + metric 嵌套），CC BY-NC 仅内部研究可接受。

2. **数据确认**（计划 §2 标注「待确认」已落实）：
   - vggt_human 棚拍 = **单场景，125 视角，1440×1920 JPG**，COLMAP sparse/0 在位（`~/output/vggt_human_results/source/`）；`/mnt/d/output/vggt_human_results` 为同场景 08 搬运产物。
   - 满足计划数据要求（≥4 视角、有内参、静态）；主体人物基本静态（vggt_human/04 注释），动态重影风险低。
   - 局限：只有 1 个场景，D 阶段敏感性实验无跨场景重复性检验（记为已知限制）。

3. **环境/网络实测（2026-11-05）**：
   - `huggingface.co` 经 v2rayN(10808) API 200 但 7s；`hf-mirror.com` 复活（2.4s）→ **下载走镜像**（10-28 记录镜像已挂，网络确会漂移，每次复测）。
   - `download.pytorch.org` 目录 403 属正常；torch 走 tuna（torch 2.14.1 拉 cu13 nvidia 依赖 ~2.5GB，~2.3MB/s）。
   - DA3 官方仓已在 `~/repos/Depth-Anything-3`（HEAD 4173623），跳过 clone。
   - 模型导出格式支持 `-` 组合（如 `mini_npz-colmap-depth_vis`）；`colmap` 导出把 cameras/images/points3D.bin 写在 export_dir 根（需整理成 `sparse/0` 给原版 3DGS）。

### 阶段 A：环境搭建 ✅

| 项 | 结果 |
|---|---|
| conda env | `da3_human`，python 3.10.22（`00a_setup_env.sh` 一键建） |
| torch | 2.14.1+cu130（tuna；拉 cu13 nvidia 依赖 ~2.5GB） |
| xformers | 0.0.35 ⚠️ 二进制是 against torch 2.10+cu128 编的，C++/CUDA 扩展加载失败 |
| DA3 本体 | editable 0.0.0（`~/repos/Depth-Anything-3` HEAD 4173623） |
| 其他关键 | numpy 1.26.4（requirements 钉 numpy<2）、pycolmap 4.2.1、open3d 0.20.0 |
| 耗时 | 装包 ~44 min（tuna ~1.5MB/s） |

**xformers 不配对的影响评估**：DA3 attention 走 `F.scaled_dot_product_attention`
（PyTorch 原生，model/utils/attention.py:57），xformers 只用于 SwiGLU 融合 MLP
（Triton 算子，运行时编译，不依赖预编译 C++ 扩展）。**预计不影响推理**，由 02 实测最终验证；
若 SwiGLU 报错，回退方案 = torch 降到 2.10.0 对齐 xformers。

装包坑：pip 对 opencv-python 回退解析多轮（moviepy 依赖冲突），最终 opencv 4.11.0.86 + moviepy 1.0.3；
numpy 被 requirements 的 `numpy<2` 从 2.2.6 降级到 1.26.4（torchvision 装的是 2.x，无冲突）。

### 阶段 B-01：权重下载 ✅

| 项 | 结果 |
|---|---|
| 模型 | `depth-anything/DA3NESTED-GIANT-LARGE-1.1` → `/mnt/d/wheel/da3_human_ms` |
| 产物 | model.safetensors **6.3GB**（fp32 1.4B）+ config.json |
| 通道 | hf-mirror.com（直连 7s vs 镜像 2.4s；镜像吞吐 ~1MB/s，全程 63 min） |
| 复跑 | `bash da3_human/01_download_models.sh`（幂等，本地已存在则秒过） |

### 阶段 B-02：DA3 推理（125 视角）✅

| 验收项 | 结果 | 判定 |
|---|---|---|
| 深度可视化合理 | median **1.797m**，范围 0.48~6.10m，NaN=0（度量输出，棚拍尺度合理） | ✅ |
| 位姿无漂移 | vs COLMAP Sim(3) 对齐后 mean **4.9mm** / p95 8.2mm / max 10.1mm，125/125 视角匹配，轨迹逐段重合（viz/trajectory_3d.png） | ✅ |
| 耗时 | 模型加载 18s + 前向 34~54s + 导出 92s ≈ **2.5 min**（504p，RTX 3090） | — |

产物（`~/output/da3_human_results/da3_inference/studio_human/`）：
`exports/{mini_npz,npz}/results.npz`（深度/置信度/extrinsics/intrinsics）、
colmap 导出（cameras/images/points3D/frames/rigs .bin，**14,288,402 点**、conf≥P40）、
`depth_vis/` 逐视角深度可视化、`da3_summary.json`、`viz/` 轨迹对比图。

**关键数字解读**：
- Sim(3) scale=0.5077：DA3 度量重建比 COLMAP 坐标系大约 1.97×（COLMAP 尺度本就任意，无问题）。
- 首末相机距 1.42m ≠ 漂移：轨迹是带起始小回环的开弧（viz 图证实），非闭环采集。

**踩坑记录**：
1. DA3 `requirements.txt` 漏声明 `addict`（da3.py:19）→ 00 脚本已补 `pip install addict`。
2. COLMAP `images.txt` 解析：points2D 空行（无 2D 点的注册图像）不能过滤，否则 header 配对
   错位约一半匹配不上（首跑 63/125 → 修后 125/125）。
3. xformers 0.0.35/torch 2.14.1 不配对：推理正常（attention 走 F.sdpa；日志出现
   "using MLP layer as FFN" 说明 SwiGLU 融合层自动回退 MLP），**无实际影响**，不回退 torch。
