# InfiniSplat v2 — 前馈多视角 3DGS 重建（WSL 复现）

从多张 RGB 图（可选位姿）前馈重建 3D Gaussian 场景。官方实现把几何基座
DA3（Depth-Anything-3）**冻结**、位置全程钉死在深度反投影上，只学 Gaussian 属性
（rotation/scale/opacity/SH0），再走体素合并 + opacity<0.01 剪枝。

- 论文：<https://arxiv.org/abs/2610.20341>（浙大 ZJU3DV）
- 官方仓：<https://github.com/zju3dv/InfiniSplat>（v2 在 `InfiniSplat_v2/`）
- 权重：`Edisoneh/InfiniSplat-v2`（HF，公开免登录，6.6 GB，含 DA3 backbone）

## ⚠️ 复现边界（先读这段，别拿本仓数字去和论文表 1 比）

官方**只开源推理包**，训练代码、refiner、官方评测划分都没有。本仓复现的是推理链路，
不是论文训练流程。

| 项 | 状态 | 说明 |
| --- | --- | --- |
| 前馈推理（表 1 "Ours" 行） | ✅ 可复现 | 代码+权重齐全，`02_run_inference.sh` |
| 三种 pose 模式 | ✅ 可复现 | provided / colmap / da3-pose（+ pose-free） |
| 高斯的 opacity 剪枝 / 体素合并 | ✅ 可复现 | 契约硬校验：voxel=median_scale×1.4、τ_α=0.01 |
| 三步 learned refinement（"Ours*" 行） | ❌ 不可得 | 发布包无 refiner 代码/权重，`import` 里都没有 |
| 训练（Stage I/II） | ❌ 不可复现 | 无训练代码；数据含私有 Raw3D；16×H20 约 61h |
| 论文表 1 的精确数值 | ⚠️ 不可直接对标 | 官方划分未公开；本仓用自建 holdout（等间隔抽目标视角）给同量级参考 |
| 渲染/评测 | ⚠️ 需自接 | 发布包刻意不带 gsplat，`03_eval_render.sh` 自己接 |

结论：**能复现"一个前馈模型吐出可渲染 3DGS 场景"这条能力链，不能复现论文的
25.8%/32.9% 提升数字**——那个数字来自训练侧三项手术（冻结几何 + GT 位姿对齐 +
大角度监督），训练没开源，无法复算。

## 实测结果（本机 3090，长边 518）

| 场景 | 输入 | 口径 | PSNR↑ | SSIM↑ |
| --- | --- | --- | --- | --- |
| hypersim_provided | 50 视角（GT 位姿） | 自洽 re-render | **27.05** | **0.918** |
| hypersim_da3_pose | 32 视角（DA3 预测位姿） | 自洽 re-render | 23.29 | 0.829 |
| dtu_provided | 16 视角（GT 位姿） | 自洽 re-render | 20.30 | 0.726 |
| dtu_provided | 16 视角 | **NVS holdout（4 目标）** | 13.10 | 0.585 |

推理耗时：16 视角 22 s、32 视角 28 s、50 视角 31 s（含 6.6 GB 权重 SHA256 校验），
产出 2.7 M / 5.3 M / 6.1 M 高斯，峰值显存约 11 GB。

⚠️ 上表是**原始导出**口径（未精调）。`02` 现在默认会串一步精调（自洽 +7.3 dB / 留出 +2.11 dB，
见「后处理精调」一节），所以直接跑 02/03 拿到的自洽数会明显高于上表；要比上表就 `FINETUNE=0`。

### 三视角自有数据集实测（`D:\dataset\物体三图数据`，8 个物体）

| 项 | 实测 |
| --- | --- |
| 端到端（冷启动，含加载） | 20.9–25.2 s，均值 **21.8 s** |
| 常驻复用单次（warm） | **0.81 s** |
| 显存峰值（`nvidia-smi`） | 9.72–9.87 GB（基线 1.55 GB，净增 **8.2 GB**） |
| `torch` allocated 峰值 | 7.09 GB |
| 高斯数 / PLY | 22–31 万 / 15–21 MB |
| 自洽 PSNR（3 图全输入） | 15.9–24.5，均值 21.5 |
| **真实 NVS（2 入 → 1 出）** | **11.3–16.1，均值 13.1** |

⚠️ 三视角有两个硬坑，动手前必读：
1. **DA3 位姿会静默塌缩**。榨汁机（透明）/ 电饭煲（浅色圆柱）被估成三张同一视角
   （两两夹角 0.2–0.9°），推理照样 exit 0 输出 PLY，但几何是废的。跑完先执行
   `python diag_pose.py` 看夹角/基线，或用文件名帧号交叉验证。`pose-free` 不能救，
   COLMAP 在这批数据上也建不起来（SIFT 太少，无初始像对）。
2. **自洽 ≠ NVS，差 8.4 dB**。留出视角里物体位置对但未见区域是空洞（alpha 0.47–0.81）。

细节与逐物体数据见 `EXPERIMENTS.md` 的 E-07~E-11。

### 人物采集包实测（`D:\dataset\手部微动样例`，10 场景 / 每场景 82 帧）

华为 Remy 采集包（`transforms.json` + `image/*.heic` + `pcd.ply`），带 GT 位姿，
走 `07_run_capture.sh` 一条命令。三个场景实测：

| 配置 | 视角 | 长边 | raw 高斯数 | 峰值显存 | 端到端 | 常驻复用 | 自洽 PSNR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 佳佳抱枕 | 16 等间隔 | 518 | 3.25 M | 13.4 GB | 25.0 s | — | 19.63 |
| 佳佳抱枕 | 21 等间隔 | 518 | 4.26 M | 14.6 GB | 26.1 s | — | 20.20 |
| 佳佳抱枕 | 32 等间隔 | 518 | 6.50 M | 18.3 GB | 35.5 s | **8.35 s** | 20.72 |
| 佳佳抱枕 | 48 等间隔 | 518 | 9.75 M | 22.8 GB | 38.4 s | — | 20.46 |
| 佳佳抱枕 | 50 等间隔 | 518 | 10.15 M | **23.8 GB** | 37.8 s | — | — |
| 佳佳抱枕 | 16 等间隔 | 777 | 7.38 M | 18.5 GB | 29.7 s | 8.05 s | 20.31 |
| 佳佳抱枕 | 21 等间隔 | 777 | 9.68 M | 22.7 GB | 34.1 s | — | 20.58 ⚠️ |
| 佳佳抱枕 | 32 等间隔 | 777 | 14.75 M | 24.1 GB | **73 s** | — | 撞顶降级 |
| 佳佳抱枕 | 32 等间隔 | 1036 | 25.5 M | 24.2 GB | **318 s** | — | 撞顶降级 |
| 佳佳抱枕 | **16 连续帧** | 518 | 3.25 M | 12.3 GB | 24.7 s | — | 21.22 ⚠️ |
| 三人聚餐 | 16 等间隔 | 518 | 3.25 M | 12.4 GB | 24.2 s | — | 19.38 |
| 女生弹琴 | 16 等间隔 | 518 | 3.25 M | 12.4 GB | 23.9 s | — | 20.15 |

- `raw 高斯数` = 有效像素反投数（`result.json:stats.raw_count`），**不是 PLY 里的数量**：
  体素合并 + opacity 剪枝后 `final_count` 约为 raw 的 **0.85–0.95**（0.91@32v/518）。
  显存公式里的 raw 就是这个列。
- ⚠️ 连续 16 帧（同一姿态段）比等间隔 16 帧高 **1.59 dB**，是视角间距变小、重叠变多带来的，
  **不能和等间隔那几行并列比较**；详见 E-18。
- ⚠️ **PSNR 是像素单位，不同长边不能并列比**。21v@777 那行 20.58 是 588×784 原生口径；
  用 BOX 重采样到 392×518 后是 **19.92**。要和同为 518 的行比较请用 19.92。

**显存 ≈ 8.5 + 1.51 × raw高斯(百万) GB**，24 GB 卡 raw 上限约 **1000 万**：

| 长边 | 处理尺寸 | 最多视角 |
| --- | --- | --- |
| 518 | 392×518 | 50 |
| 777 | 588×784 | 21 |
| 1036 | 770×1036 | 12 |

超过上限不会 OOM，而是**静默变慢 10 倍**（峰值卡在 24.1–24.2 GB）。判据就是这个。
⚠️ 手部只保留 GT 锐度的 5%（518）/ 13%（777）；手部微动使自洽 PSNR 掉 1.6 dB，
但这个尺度下看不见幽灵，只表现为更糊。
⚠️ **归因更正（Step 3）**：这里原先把「手部糊」归因为像素预算不足，**该归因已被推翻** ——
同一份重建只做后处理精调（位置冻结、位姿不动）就能把 σ_eff 从 1.5 px 压到 0.5 px、
锐度保留 0.66→0.91，分辨率和高斯数都没变。真正的原因是**默认导出参数造成的合成模糊**
（平均不透明度 0.27、一个像素被多次叠加平均）与弱纹理区的深度精度。见下节。

### 同一笔预算：加视角还是加分辨率（E-22 / E-23）

三组二因素对照（21v@777 与 48v@518 的 raw 几乎相同，用 21v@518 做基线才能把两个因子拆开）。
`11_budget_compare.py` 把渲染与 GT 都重采样到 392×518 后再比，否则长边不同 PSNR 不可比：

| 因子 | 变化 | PSNR@518 | 锐度保留 | 双峰率 | 显存 | 耗时 |
| --- | --- | --- | --- | --- | --- | --- |
| 基线 | 21v@518 | 20.20 | 0.686 | 27.6% | 14.6 GB | 26.1 s |
| **加分辨率** | → 21v@777 | **19.92** | **0.767** | 26.8% | 22.7 GB | 34.1 s |
| **加视角** | → 48v@518 | **20.46** | 0.629 | **22.4%** | 22.8 GB | 38.4 s |

- 加分辨率：**PSNR −0.28 dB、锐度 +12%、重影不变** —— 多出来的高频与 GT 不对齐，不是真细节。
- 加视角：**PSNR +0.26 dB、重影 −5.2pp、锐度 −8%** —— 重叠区被合并得更多（丢弃率 4.7%→9.3%），更干净也更平滑。
- 分辨率涨 1.51×，**绝对锐度只涨 1.2–1.3×**（σ_eff 1.50 → 1.75 px，等比放大本应 2.27）：
  高斯半径跟着 pixel footprint 走，提分辨率只是把高斯排得更密，单个高斯并不更锐。

**采集建议**：拍摄视角回放 → 32–48 视角 @518。

⚠️ **手/脸近景不要靠 ROI 裁剪**（Step 4 实测否定，E-25~E-28）：手部像素涨 3.3×
（52.7→175.6 px）后 σ@160 反而从 3.12 升到 3.88、手部 PSNR 降 0.77 dB —— 深度精度才是瓶颈，
加像素只是"把同样的糊涂等比放大"。整图提分辨率也不划算（2.27× 预算只换 1.2× 锐度，
1036 撞顶 318 s）。

另一个更该记住的坑：**任何 ROI/跟踪窗口都必须固定视场**。框跟着目标移动会让
`geometry.py:188` 的 per-view Umeyama scale 被拉开、破坏相机间距，0.70 框实测掉 **1.84 dB**
且渲染出现横向涂抹；改成固定中心立刻恢复。

⚠️ **自洽 ≠ NVS**：自洽时每个高斯中心就是从该视角反投影来的，只证明"属性学对了 + 渲染器
接对了"；能对标论文的只有 holdout 那一行。且官方划分未公开、分辨率与目标视角选取都不同，
**本表与论文表 1 不可直接比较**。推导与逐视角数据见 `EXPERIMENTS.md`。

### 后处理精调：从已导出的高斯里还能榨出多少（E-29~E-34）

`14_finetune_gs.py`：**位置冻结**（不 densify、不 prune）、**位姿不动**，只优化
SH / scale / opacity / quat，3000 步 **161 s、峰值显存 6.3 GB**。判据用**严格留出**
（另建一个只用 25 视角的重建，7 个视角的像素完全不参与重建、也不参与精调）：

| | 自洽(25 视角) | 留出(7 视角) | σ_eff | 锐度保留 | 双峰率 |
| --- | --- | --- | --- | --- | --- |
| 原始导出 | 20.67 | 19.05 | 1.50 px | 0.662 | 24.8% |
| 精调后 | 28.78 | **21.15（+2.11）** | **0.50 px** | **0.914** | **15.9%** |

- 留出 **7/7 视角全为正**（+0.58～+4.13 dB）；换随机种子复跑只差 **0.004 dB**。
- ⚠️ **自洽涨的 8.1 dB 只有 26% 迁移到新视角**，读自洽数字一律按这个折扣。
- 增益主要来自几何（scale/opacity/quat 占 88%，只调颜色 +1.85 dB）：优化把合成从
  「多次低不透明度叠加平均」变成「最近高斯主导」，所以半径变大 1.26× 反而更锐。
- 单独手调这两个旋钮都是负收益（scale 0.7× → −2.5 dB，opacity 1.5× → −0.23 dB）：
  参数是**逐个失准**，全局乘数修不了，所以只能靠优化。
- 它擦掉的是「糊」，不是「缺信息」——手部细节仍不如 GT。

**已接进默认流程**：`02_run_inference.sh` 出完 PLY 自动调 `02b_finetune.sh`，结果落
`<结果目录>/ft/`（**不覆盖原始导出**，两版并存才能对照）。`03_eval_render.sh` 见 ft 自动优先评测它。
代价是每次重建多 160 s；`FINETUNE=0` 跳过，`FINETUNE_ITERS=N` 换步数，
`FINETUNE_EXTRA_ARGS="--opt sh"` 做消融，`EVAL_FT=0` 强制评原始导出。
图像↔相机配对走 `result.json` 的 `inputs[i].sha256` 硬映射（命中率 <80% 才退回文件名排序）——
精调接进默认流程后，配错图不会再被肉眼兜住，只会把高斯往错方向拉而 PSNR 曲线照样上涨。

### 轨道外外推：会不会塌成 sprite / 露空洞（E-35 / E-36）

`16_extrap_eval.py` 由 32 台观测相机拟合球面轨道，再在**观测范围之外**造位姿渲染
（外推视角没有 GT，只能与有 GT 的观测视角比同一套 GT-free 指标）。被摄场景的轨道：
半径 0.648、俯仰 +9.7°~+31.6°、方位只覆盖 **201.5°**（158.5° 的背面没拍过）。

| 位置 | 覆盖（orig→ft） | 空洞率 | 半透明率 |
| --- | --- | --- | --- |
| 观测视角（PSNR 20.93→28.43） | 0.989→0.985 | 0→0% | 0.55→2.04% |
| 俯仰越界 ±15° / ±30° | 0.65–0.97 → 0.66–0.96 | ≤0.5% | 2.9–4.1 → 2.9–8.6% |
| 半径 0.7× / 1.3× / 1.6× | 0.86–0.98 → 0.86–0.98 | ≤0.2% | 1.9–6.1 → 3.2–6.6% |
| 方位越出弧端 20° / 40° / 60° | 0.67–0.96 → 0.73–0.96 | ≤0.4% | 5.7–8.1 → 4.3–13.8% |
| 158° 未观测侧中点 | 0.85→0.85 | ≤0.2% | 14.9→**7.8%** |

- **没有 sprite 化，也没有露空洞**：空洞率全场 ≤0.7%（精调后 ≤0.2%），拉到 1.6× 距离仍是完整场景。
- **精调没有系统性放大视角依赖**：各向 «精调−原始» 的覆盖率差都在 ±0.06 内。唯一反常是
  **弧外 40° 的其中一个边缘**：覆盖 0.686→0.773 涨了，半透明率 8.8%→19.0%，视觉上是成片
  条纹拖影（那里有一块始终没被看清的遮挡面）。⇒ **弧外 30° 内可放心用，40° 以上要当心**。
- ⚠️ **锐度指标在外推视角会骗人**：158° 未观测侧精调把锐度从 15.0 抬到 27.2，
  但视觉上是细密噪点/条纹 —— 无观测区域任何纹理都是无中生有。GT-free 的 sharp 只能**同场景**比。
- 外推的正确做法靠两个自检（E-36）：位姿构造器拿观测相机对拍（第一版 roll 差 180°、
  第二版上下颠倒，两次都照样出图不报错）；球心用**视线交汇最小二乘**而非相机中心均值
  （后者偏 35°、半径在 0.27–0.83 乱跳）。

## 常用命令

> 以下都在 WSL 里、`cd` 到本目录（`/mnt/c/code/media_code/infinisplat_v2`）执行。

```bash
# 一键（建环境 → 下权重 → 跑 dtu_provided → holdout 评测）
bash run_all.sh

# 分步
# 0) 建 conda env + torch + infinisplat + DA3 + gsplat
CONDA_ENV=infinisplat_v2 PY_VER=3.11 TORCH_VER=2.7.1 bash 00_setup_env.sh

# 1) 下权重（6.6 GB，走 aifasthub 镜像，8 路分块）+ 官方示例（90 MB）
HF_BASE_URL=https://aifasthub.com DOWNLOAD_JOBS=8 bash 01_download_models.sh
SKIP_EXAMPLES=1 bash 01_download_models.sh     # 只下权重

# 2) 推理：官方三示例（默认会串一步 161 s 精调，结果在 <输出>/ft/）
EXAMPLE=dtu_provided        LONG_SIDE=518 MAX_VIEWS=32 bash 02_run_inference.sh
EXAMPLE=hypersim_provided   LONG_SIDE=518 MAX_VIEWS=50 bash 02_run_inference.sh
EXAMPLE=hypersim_da3_pose   LONG_SIDE=518 MAX_VIEWS=32 bash 02_run_inference.sh
FINETUNE=0 EXAMPLE=dtu_provided bash 02_run_inference.sh        # 只要原始导出
EXAMPLE=dtu_provided FINETUNE_ITERS=1000 bash 02_run_inference.sh  # 省时间的半程精调

# 2') 自有图像
IMAGES_DIR=/mnt/d/data/myscene/images CAMERAS=/mnt/d/data/myscene/cameras.npz \
  OUT_NAME=myscene bash 02_run_inference.sh
IMAGES_DIR=/mnt/d/data/myscene/images POSE_MODE=da3-pose OUT_NAME=myscene \
  bash 02_run_inference.sh                       # 无相机，DA3 先测位姿

# 3) 评测（自接 gsplat 渲染）
RESULT=$HOME/output/infinisplat_v2_results/dtu_provided \
  IMAGES_DIR=$HOME/repos/InfiniSplat/InfiniSplat_v2/examples/data/dtu_provided/images \
  MODE=rerender bash 03_eval_render.sh           # A: 输入视角自洽
EXAMPLE=dtu_provided MODE=holdout N_INPUT=8 N_TARGET=4 bash 03_eval_render.sh   # B: NVS

# 4) 自有数据集批量（每子目录一个物体）+ 记录显存/耗时
DATASET_ROOT=/mnt/d/dataset/物体三图数据 bash 04_run_dataset.sh
DATASET_ROOT=... LONG_SIDE=1022 DO_EVAL=0 bash 04_run_dataset.sh

# 5) 少视角真实 NVS（2 输入 → 1 留出，位姿同源）
DATASET_ROOT=... CASES="手表 鼠标" N_INPUT=2 N_TARGET=1 bash 06_nvs_holdout.sh

# 6) 位姿自检（判断 DA3 是否塌缩）
python diag_pose.py --results ~/output/infinisplat_v2_results --prefix threeview

# 7) 拆耗时段 / 测常驻复用成本（--cameras 走 provided，跳过 DA3 位姿）
python bench_stages.py --model ~/model/infinisplat_v2/da3 \
  --images-dir /mnt/d/dataset/物体三图数据/手表 --repeat 6 --out /tmp/stages.json
python bench_stages.py --model ~/model/infinisplat_v2/da3 \
  --images-dir ~/output/infinisplat_v2_results/_captures/jiajia/images \
  --cameras  ~/output/infinisplat_v2_results/_captures/jiajia/cameras.npz \
  --long-side 518 --max-views 32 --repeat 4 --out /tmp/stages_cap.json

# 8) COLMAP 供位姿（DA3 塌缩时的备选；物体数据集上因纹理太少未能重建）
IMAGES_DIR=/mnt/d/data/imgs COLMAP_LONG_SIDE=1024 bash 05_pose_colmap.sh

# 9) 手机采集包（transforms.json + HEIC + pcd.ply）→ 推理 + 显存耗时
RAW_DIR="/mnt/d/dataset/手部微动样例/01e954de1b8a4cde9315c976f4dcb964-佳佳抱枕" \
  OUT_NAME=jiajia N_VIEWS=32 LONG_SIDE=518 bash 07_run_capture.sh
RAW_DIR=... OUT_NAME=jiajia777 N_VIEWS=16 LONG_SIDE=777 bash 07_run_capture.sh  # 手部更清，但更吃显存
RAW_DIR=... OUT_NAME=jiajia N_VIEWS=32 SKIP_INFER=1 bash 07_run_capture.sh      # 只转换不推理
RAW_DIR=... OUT_NAME=jiajia N_VIEWS=32 RENDER=1 bash 07_run_capture.sh          # 转换+推理+自洽渲染

# 10) 糊 / 重影量化（纯 CPU，读已有渲染；--selftest 先验指标灵敏度）
python 09_diag_ghost.py \
  --result ~/output/infinisplat_v2_results/jiajia \
  --images-dir ~/output/infinisplat_v2_results/_captures/jiajia/images \
  --out ~/output/infinisplat_v2_results/_diag --tag jiajia --selftest

# 11) 扫体素合并参数（只跑一次前向，本地反复合并+渲染）
python 10_sweep_merge.py --model ~/model/infinisplat_v2/da3 \
  --images-dir ~/output/infinisplat_v2_results/_captures/jiajia/images \
  --cameras    ~/output/infinisplat_v2_results/_captures/jiajia/cameras.npz \
  --long-side 518 --max-views 32 --voxels 1.4,1.0,0.7,0.5 \
  --out ~/output/infinisplat_v2_results/_sweep --tag jiajia

# 12) 同预算对照：加视角 vs 加分辨率（两组渲染先统一到 B 组尺寸再比）
python 11_budget_compare.py \
  --a-result ~/output/infinisplat_v2_results/jiajia_s2_v21 --a-tag "21v@777" \
  --a-images ~/output/infinisplat_v2_results/_captures/jiajia_s2_v21/images \
  --b-result ~/output/infinisplat_v2_results/jiajia_s2_v21_518 --b-tag "21v@518" \
  --b-images ~/output/infinisplat_v2_results/_captures/jiajia_s2_v21/images \
  --out ~/output/infinisplat_v2_results/_diag

# 13) 手部 ROI 裁切重建（先跑 SAM3 手部 mask，再裁切 + 推理 + 自洽渲染）
#     ⚠️ FIXED_CENTER=1 是必须的：框跟随目标会破坏相机间距（E-26）
SAM3_PROMPT=hand MASK_MODE=video \
  ~/miniconda3/envs/sam3/bin/python ../vggt_human/sam3_face_masks_worker.py \
  --images_dir ~/output/infinisplat_v2_results/_captures/jiajia/images \
  --output_dir ~/output/infinisplat_v2_results/_masks/hand_32v

RAW_DIR="/mnt/d/dataset/手部微动样例/01e954de1b8a4cde9315c976f4dcb964-佳佳抱枕" \
  MASK_MANIFEST=~/output/infinisplat_v2_results/_masks/hand_32v/masks_manifest.json \
  OUT_NAME=jiajia_roifix030 ROI_SOURCE=mask ROI_FRAC=0.30 FIXED_CENTER=1 \
  N_VIEWS=32 TARGET_LONG=518 RENDER=1 bash 12_run_roi.sh

# 14) 手部区域统一口径对比（σ@160 / σ_native / PSNR@160，可跨分辨率比）
python 13_roi_eval.py \
  --mask-manifest ~/output/infinisplat_v2_results/_masks/hand_32v/masks_manifest.json \
  --group "full32=~/output/.../_captures/jiajia=~/output/.../jiajia" \
  --group "fix030=~/output/.../_captures/jiajia_roifix030=~/output/.../jiajia_roifix030" \
  --hand-width 160 --out ~/output/infinisplat_v2_results/_diag/roi_eval.json

# 15) 后处理精调（位置冻结、位姿不动，只调 SH/scale/opacity/quat；161 s / 6.3 GB）
#     02 默认已经调它了；这里是单跑 / 换参数时的入口
RESULT=~/output/infinisplat_v2_results/jiajia \
  IMAGES_DIR=~/output/infinisplat_v2_results/_captures/jiajia/images \
  bash 02b_finetune.sh                                  # 默认 3000 步 → <RESULT>/ft
RESULT=... IMAGES_DIR=... FINETUNE_ITERS=1000 bash 02b_finetune.sh
RESULT=... IMAGES_DIR=... FINETUNE_EXTRA_ARGS="--opt sh" bash 02b_finetune.sh   # 只调颜色

# 16) 严格留出（25 训 / 7 留）—— 只有这个数能报
#     两个包共享 AR 世界坐标系，所以 --ref-cameras 可直接用训练包的 GT 相机
RAW=/mnt/d/dataset/手部微动样例/<包名>
CAP=~/output/infinisplat_v2_results/_captures
python 07_capture_to_npz.py --raw-dir "$RAW" --out-dir "$CAP/jiajia_tr25" \
  --frames 0,3,5,10,13,16,21,24,26,31,34,37,42,44,47,52,55,57,63,65,68,73,76,78,81
python 07_capture_to_npz.py --raw-dir "$RAW" --out-dir "$CAP/jiajia_ho7" \
  --frames 8,18,29,39,50,60,71
# 只用 25 视角重建 → 精调 → 用留出相机渲染比 PSNR
python render_check.py --result <精调后的目录> --images-dir "$CAP/jiajia_ho7/images" \
  --cameras "$CAP/jiajia_ho7/cameras.npz" --ref-cameras "$CAP/jiajia_tr25/cameras.npz" \
  --out <精调后的目录>/eval_hold --tag hold_ft --max-views 7

# 17) 轨道外外推体检（sprite / 空洞；原始 vs 精调一次出对照）
python 16_extrap_eval.py \
  --result ~/output/infinisplat_v2_results/jiajia \
  --result-b ~/output/infinisplat_v2_results/jiajia/ft \
  --images-dir ~/output/infinisplat_v2_results/_captures/jiajia/images \
  --out ~/output/infinisplat_v2_results/_extrap --tag jiajia
python 16_extrap_eval.py --result ... --stats-only    # 只看轨道几何（方位跨度/俯仰范围/自检）
# 想扫别的越界量：--el-deltas 15,30  --r-factors 1.3,1.6,0.7  --az-past 20,40,60
```

## 首次准备

```bash
# 官方仓 + DA3（00 会自动 clone；此处为手动版本）
mkdir -p ~/repos && cd ~/repos
git clone --depth 1 https://github.com/zju3dv/InfiniSplat.git
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
git -C Depth-Anything-3 checkout 41736238f5bced4debf3f2a12375d2466874866d

# 环境
cd /mnt/c/code/media_code/infinisplat_v2
bash 00_setup_env.sh
bash 01_download_models.sh
```

权重目录布局：

```
~/model/infinisplat_v2/
└── da3/
    ├── config.json          # 含 weights_sha256 与融合契约（voxel_multiplier=1.4 等）
    └── model.safetensors    # 6.6 GB，SHA256 6d23f9ce…a66e8
```

## Config (env vars)

| var | default | note |
| --- | --- | --- |
| `CONDA_ENV` | `infinisplat_v2` | 独立 env，别混用已有的 |
| `PY_VER` | `3.11` | 包要求 `>=3.10,<3.13` |
| `TORCH_VER` | `2.7.1` | 官方在 2.7.1 + cu128 + H20 上实测 |
| `HF_BASE_URL` | `https://aifasthub.com` | 官方 HF 与 hf-mirror 在本机都不通，见 Notes |
| `DOWNLOAD_JOBS` | `8` | 分块并发；单连接实测 ~1.3 MB/s |
| `INFINISPLAT_REPO` | `$HOME/repos/InfiniSplat` | 官方仓 |
| `DA3_REPO` | `$HOME/repos/Depth-Anything-3` | 钉死 commit 41736238 |
| `WEIGHTS_DIR` | `$HOME/model/infinisplat_v2/da3` | `config.json` + `model.safetensors` |
| `RESULTS_DIR` | `$HOME/output/infinisplat_v2_results` | Linux fs，跑完用 08 搬 /mnt/d |
| `LONG_SIDE` | `518` | 112–2044，内部对齐 14 的倍数（DA3 patch=14） |
| `MAX_VIEWS` | `32` | CLI 上限 50；50 视角示例要显式给 `--max-views 50` |
| `EPS2D` | `0.01` | 论文值 1e-2；gsplat 默认 0.3，渲染时显式覆盖 |
| `MODE` | `holdout` | `03` 的评测口径：holdout（NVS）/ rerender（自洽） |
| `N_INPUT` / `N_TARGET` | `8` / `4` | holdout 划分；4 对齐论文"每例 4 个目标视角" |
| `FINETUNE` | `1` | `02` 末尾是否串 `02b` 精调；0 = 只要原始导出 |
| `FINETUNE_ITERS` | `3000` | 精调步数（161 s）；0 = 干跑只 eval |
| `FINETUNE_EXTRA_ARGS` | 空 | 透传给 `14_finetune_gs.py`，如 `--opt sh` |
| `FT_SUBDIR` | `ft` | 精调结果落在 `<结果目录>/<FT_SUBDIR>`，不覆盖原始导出 |
| `EVAL_FT` | `1` | `03` 见 `<结果>/ft` 时优先评它；0 = 强制评原始导出 |

## 目录布局

```
/media_code/infinisplat_v2/          # ← 本目录（编排脚本）
├── _env.sh                          # 共享环境（proxy / conda / GPU / 路径）
├── 00_setup_env.sh                  # 建 env + torch + infinisplat + DA3 + gsplat
├── 01_download_models.sh            # 权重 + 官方示例（镜像站分块下载）
├── 02_run_inference.sh              # 前馈推理 → scene.ply（默认串一步 02b 精调）
├── 02b_finetune.sh                  # 推理后精调包装（落 <结果>/ft/，可单独跑/换参数）
├── 03_eval_render.sh                # 自接 gsplat 渲染 + 两种口径评测（自动优先评 ft/）
├── 04_run_dataset.sh                # 批量跑自有数据集 + 逐物体记显存/耗时
├── 05_pose_colmap.sh                # COLMAP SfM 供位姿 → cameras.npz
├── 06_nvs_holdout.sh                # 少视角真实 NVS（N 输入 → 留出 M 视角）
├── 07_run_capture.sh                # 手机采集包 → 转换 + 推理 + 显存耗时（一条命令）
├── 08_move_output.sh                # 结果从 Linux fs 搬到 Windows 盘
├── run_all.sh                       # 一键
├── hf_download.py                   # 分块并发下载器（断点续传 + SHA256）
├── make_holdout.py                  # 输入/目标视角划分
├── render_check.py                  # PLY → 渲染 → PSNR/SSIM + 拼图
├── 07_capture_to_npz.py             # transforms.json + HEIC → PNG + cameras.npz（含去畸变）
├── 09_diag_ghost.py                 # 糊/重影量化（σ_eff / 分块锐度 / 边缘双峰）
├── 10_sweep_merge.py                # 扫 voxel_multiplier / opacity（一次前向 + 反复合并）
├── 11_budget_compare.py             # 同预算「加视角 vs 加分辨率」（统一重采样后比）
├── 12_run_roi.sh                    # 手/脸 ROI 裁切 → 推理 → 显存耗时（一条命令）
├── 12_make_roi_capture.py           # 原图裁 ROI + 内参三处修正（**必须固定视场**，见 E-26）
├── 13_roi_eval.py                   # 手部区域统一口径对比（σ@160 / σ_native / PSNR@160）
├── 14_finetune_gs.py                # 位置冻结的 3DGS 后处理精调（自带写盘→回读对拍）
├── 15_step3_compare.py              # 精调前后 + 消融的对照图（含 |diff| 列）
├── 16_extrap_eval.py                # 轨道外外推体检（sprite/空洞，GT-free 四项指标 + 拼图）
├── ghost_metrics.py                 # 上几者的指标库（口径只改这一处）
├── bench_run.py                     # 包一条命令，线程采样 GPU 显存峰值 + 计时
├── bench_stages.py                  # 拆耗时段 + warm 重复（测常驻复用成本）
├── rebuild_bench.py                 # 从已有结果目录重建汇总表（不重跑 GPU）
├── diag_pose.py                     # 相机几何自检（两两夹角/基线 + 逐视角指标）
├── EXPERIMENTS.md                   # 实验记录：实测数值、踩坑、复现边界
└── README_wsl.md                    # 本文件

~/repos/InfiniSplat/InfiniSplat_v2/  # 官方仓（无训练代码）
~/repos/Depth-Anything-3/            # DA3 几何基座（钉死 commit）
~/model/infinisplat_v2/da3/          # 权重
~/output/infinisplat_v2_results/     # 输出（scene.ply / cameras.npz / result.json / eval/）
```

## Notes

**网络（本机 2026-10-30 实测）**
- `huggingface.co` 主站 API 能通、**大文件下载被 reset**；`hf-mirror.com` 解析正常
  （160.16.86.14）但 TCP 完全不通 —— 两者都不可用。
- 可用的是 `aifasthub.com`（HF 兼容 API + Range），与 HF 同 SHA（`10eec5ca…`）。
- 镜像站 WAF 会拦 `Python-urllib` 默认 UA（403），下载器必须带浏览器 UA。
- 镜像站 **HEAD 返回 403**，取文件大小要改用 `Range: bytes=0-0` 读 `Content-Range`。
- `github.com` 直连可用，DA3 的 `git+https` 安装没问题。

**推理契约（源码验证，别凭直觉改）**
- Gaussian 头是 **11 通道**：rot(4) + scale(3) + opacity(1) + SH0 残差(3)，
  **没有位置通道** —— 中心 100% 来自深度反投影（`model.py` 里 `depth_to_world_coords_points`）。
- `scale = exp().clamp_max(0.3)`（RDNorm 归一化空间的硬上限，0.3 不是像素单位）；
  颜色是以输入 RGB 为基底预测残差 `(rgb-0.5)/SH_C0 + sh_residual`。
- 合并契约写死在 `config.json` 并在 `pipeline.py:from_pretrained` 硬校验：
  `normalization=mean_predicted_point_radius`、`merge=uniform_mean`、
  `voxel_multiplier=1.4`、`opacity_threshold=0.01`、`eps2d=0.01`。改任何一个都会
  直接抛 `Unsupported InfiniSplat inference configuration`。
- 体素合并的边长是 **median_scale×1.4 的自适应值**（代码），论文写固定 0.004 ——
  两者关系仓库里无训练代码可证。

**输出坐标系（最坑的一处）**
- `scene.ply` 与 `cameras.npz` 都在 **RDNorm 空间**：第一相机为原点，全体反投影点
  的平均径向距离归一化为 1。所以 PLY 的 xyz 不是米，`scale_i` 存 `log(s)`、
  `opacity` 存 logit（标准 3DGS PLY 约定）。
- 在这个空间里自洽渲染没问题；但**外部 GT 相机（COLMAP/标定）活在原始尺度**，
  直接拿来渲染会出现 `alpha=0`、画面全白、PSNR 为负。必须先用输入视角做锚拟合
  Sim(3) 把目标相机搬过去（`render_check.py --ref-cameras`，实测残差 1e-7 / 0.024°）。
- 要和外部资产（如 COLMAP 稠密点云）对齐时同理，必须先统一尺度。

**渲染（发布包不带）**
- `INSTALL.md` 明说不依赖 gsplat；要出图必须自接。`03_eval_render.sh` 用 gsplat
  `rasterization`，`eps2d=0.01`（论文值）。
- **gsplat 1.5.2 的输出已含 `+0.5` 直流偏置**（实测 f_dc=0、alpha≈0.98 → 像素≈0.49），
  不要再加一次；`colors` 必须给 `[C,N,1,3]`。
- GT 图必须用与 `prepare_images` 相同的 bicubic 中心裁剪缩放，内参同步乘同一个矩阵，
  否则 GT 与渲染差一个裁剪偏移，指标被系统性拉低。
- 只给输入视角的相机 → 想渲染新视角必须自己提供目标相机（provided 模式），
  这正是接入自家 COLMAP 相机的入口。

**与 vggt_human 的对接点**
- 可作为稀疏视角 `scene_gs` 初始化的 A/B 方案：输入图像 + COLMAP 相机 →
  `scene.ply`，与现行 COLMAP 点云初始化对比。
- 局限：训练数据全是室内合成/扫描（Hypersim/Raw3D/ScanNet++），无人体专项；
  几何冻结意味着 **MHR/3DMM 这类"要改几何"的环节借不上力**，只适合背景/场景。
- 相机只收 PINHOLE / SIMPLE_PINHOLE 去畸变模型（`cameras.py` 校验），
  广角/鱼眼要先自行去畸变。
