# SplatAnything / DA3 路线验证报告（WorkBuddy 回填）

> 2026-11-12 第三轮回填（WB-1 收口 + WB-2 首轮）。**只填实测到的**；未做的保持 TODO，
> 计划阈值不复制成实测值。
> 唯一任务入口：[EXPERIMENT_PLAN.md §0](../da3_human/EXPERIMENT_PLAN.md)。
> 判据与原理：[路线评审附录](../da3_human/SPLATANYTHING_ROUTE_REVIEW.md)。
> 实验事实另见 [EXPERIMENTS.md](EXPERIMENTS.md)（E-01~E-10）。

## 1. 当前状态

- 更新日期/执行者：2026-11-12 / WorkBuddy
- 本仓 commit：见 `git log`「splatanything: WB-1 收口…」（本轮新增 `06_*`、`07b_*`、
  `_gsplat_env.sh`，`03_render_eval.py` 增锐度/分区指标）；官方 commit：`b518996`；
  工作区补丁：无（官方仓仅稀疏检出，未改代码）
- run_id / 产物绝对路径：
  - 官方示例 rgb：`~/output/splatanything_results/rgb/object`
  - 官方示例 pose：`~/output/splatanything_results/pose/object`
  - 采集包 pose：`~/output/splatanything_results/pose/jiajia_hold`
  - 转换场景：`~/output/splatanything_results/_scenes/jiajia_hold`（+ `_3v` / `_8v` / `_13v` 子集）
  - WB-1 往返验证：`~/output/splatanything_results/_roundtrip/{jiajia_hold_3v,jiajia_hold}`
  - 视角数面板：`~/output/splatanything_results/panel/{jiajia_hold_8v,jiajia_hold_13v}`
  - held-out NVS：各 run 的 `nvs/nvs_summary.json`；汇总
    `~/output/splatanything_results/nvs_compare{.json,.csv}`（人像）与
    `nvs_compare_object{.json,.csv}`（物体）
  - 体素消融：`~/output/splatanything_results/{voxel001,nomerge}/`
  - ⚠️ 未按计划落到 `~/output/splatanything_route_validation/<run_id>/`，下一轮迁移或重跑（产物可复用）
- 当前执行任务 / 下一步：WB-1 **DONE**、WB-2 **首轮 DONE**（两场景）；下一步见 §8
- UE/真实静态留出数据状态：未核实，未使用。本轮已有**真 held-out**（模型没见过的视角，
  GT 不参与建云），但仍是自采/官方示例，不是 UE 渲染对照

| 任务 | 状态 | 证据路径 | 判定或阻塞 | 下一步 |
|---|---|---|---|---|
| WB-0 输入/环境冻结 | RUNNING | 见 §1、§6；`splatanything/{00,01}_*.sh` | 权重 SHA256 已校验、加载目标已追到软链终点、版本已知；缺 `inventory.json` / `manifest.json` / `commands.sh` 与 6 包冻结 | 按本报告记录生成三件套；6 包选择需用户确认身份分组 |
| WB-1 真权重与导出渲染 | **DONE** | E-04~E-09；`03_eval_render.sh`、`06_run_roundtrip.sh` | PLY 往返 PASS（MAE 3.4e-08，比判据小 4 个数量级）、`io_audit` 已出、3v/8v/13v 人像包已跑 | — |
| WB-2 配对质量与资源 | **首轮 DONE** | E-10；`08_nvs_eval.sh`、`08b_nvs_compare.py` | held-out 入口已实现并对齐自证（残差 3.8e-08）；两场景结果方向相反，**"多视角更好"不成立** | 场景 B 的 3v>8v 根因；补人脸/手部区域指标 |
| WB-3 DA3 中心来源 | TODO | — | 需改官方代码（C0~C4 对照），本轮未动 | 待 WB-2 结论稳定后 |
| WB-4 训练改造成本 | TODO | — | 官方无训练代码，需自建可导前向 | 排期后 |

## 2. 三个问题的裁决

| 问题 | 本轮结论 | 支持证据 | 反对/缺失证据 | 适用范围 |
|---|---|---|---|---|
| SplatAnything 达标了吗 | **未判定（INCONCLUSIVE），但已能给出 held-out 水位** | held-out NVS：人像 13v source → 18.39 dB / 前景 18.91 / SSIM 0.50；物体 13v → 19.08 / 前景 18.90 / SSIM 0.49。PLY 导出与内存高斯逐位等价（E-07），所以数字不是导出链的伪影 | 没有计划里写的**达标阈值**，无法判"达标"；两场景、各一条序列；**多视角不单调变好**（物体场景 3v 最好），说明"最优视角数"本身还没定；无 LPIPS/几何/人脸手部分区指标 | 可引用"held-out 大概什么水平"，不可引用"哪个视角数最优" |
| DA3 需要重训哪一层 | **未判定** | — | C0~C4 一个都没跑 | — |
| 训练改造到 T0/T1/T2 哪一级、成本多少 | **未判定** | 确认官方无可导前向/无渲染器（`rasterization.py` 注释 "no rasterization or training"） | 未做任何训练侧探测 | — |

**本轮最重要的方法学结论（比上面的数字更重要）**：自洽 PSNR 与 held-out PSNR
在同一批数据上**方向相反** —— 自洽 3v 22.37 → 24v 19.71（降 2.7 dB），held-out
3v 15.00 → 13v 18.39（升 3.4 dB）。此前所有基于自洽分数的排序（包括本报告上一版的
"19.01 / 19.71"）**都不能用来比方法或比配置**，只能证明"能跑通"。

## 3. 质量与资源门槛逐项结果

附 `gate_results.json` / CSV：**待补**。本轮只有自洽读数，**不能当门槛结论**（理由见下）。

| 场景 | 视角 | 分辨率 | 模式 | 自洽 PSNR | 前景 PSNR | SSIM | alpha 均值 | 锐度保留 | 高斯数 | 显存峰值 | 推理耗时 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `examples/object` | 23 | 756² | rgb | 未评 | — | — | — | — | 2,329,941 | 23.53 GB | 68.3 s |
| `examples/object` | 23 | 756² | pose | 19.01 | — | 0.520 | 0.775 | — | 2,462,487 | 23.59 GB | 67.3 s |
| `jiajia_hold` | 24 | 504² | pose | 19.71 | 19.76 | 0.563 | 0.977 | 81% | 1,942,508 | 21.79 GB | 7.3 s |
| `jiajia_hold_13v` | 13 | 504² | pose | 19.95 | 20.03 | 0.581 | 0.970 | 81% | 1,297,608 | 19.17 GB | 3.5 s |
| `jiajia_hold_8v` | 8 | 504² | pose | 19.91 | 20.00 | 0.607 | 0.965 | 84% | 1,008,241 | 18.62 GB | 2.2 s |
| `jiajia_hold_3v` | 3 | 504² | pose | **22.37** | 22.56 | **0.725** | 0.951 | **85%** | 473,564 | 13.41 GB | 1.0 s |
| `jiajia_hold`（voxel 0.001） | 24 | 504² | pose | 19.16 | 19.20 | 0.529 | 0.989 | **92%** | 3,379,217 | 21.96 GB | 7.0 s |

- 口径：全部 **self**（用导出的相机渲染模型自己输入的裁剪图），**不含 held-out**。
- ⚠️ **这张表不能横着比**：E-08 用同一批 GT/相机、三个共同视角证明 **PSNR 与 SSIM 随输入
  视角数单调下降**（`022_…` 视角 23.30 → 19.55 → 19.07 → 19.0），机制是体素合并消耗样本
  （保留率 64.1%→40.8%）而非质量变差。**视角数不同 = 不可比**。
- ⚠️ **PSNR 与锐度在体素尺寸上互相矛盾**（E-09）：voxel 0.001 锐度 92% 但 PSNR 19.16；
  0.002 锐度 81% 但 PSNR 19.71。选体素不能只看 PSNR。
- **PLY 导出等价性（E-07，PASS）**：内存过滤后 vs PLY 重载，means / quats / f_dc MAE = **0**，
  log scale 1.97e-09，opacity 8.89e-09，相机漂移 0；渲染 RGB MAE **3.4e-08**、alpha **1.2e-08**
  （逐位相同）。→ 03 的指标评的确实是模型吐出的那批高斯。
- 3/8/13v、人物/脸/背景拆分、LPIPS、几何/轮廓、空洞/泄漏、CI 聚合 **均未做**。
- 独立身份/场景数 = 2（1 官方物体 + 1 人像包），不足以做聚合。

### 3.1 held-out NVS（真留出，WB-2 首轮）

口径：source 视角建云，留出视角**不参与前向**，只用于渲染评分；GT 走官方
`preprocess_view`。两场景各自的分母不同，所以另给"**交集**"列（所有配置都留出过的
同一批视角）——只有交集列可以横着比。

| 场景 | source | 留出 | PSNR | 前景 PSNR | SSIM | alpha | 锐度保留 | 交集 n | 交集 PSNR | 交集前景 | 交集 SSIM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `jiajia_hold` | 3 | 21 | 15.23 | 18.17 | 0.406 | 0.789 | 89% | 11 | 15.00 | 18.35 | 0.41 |
| `jiajia_hold` | 8 | 16 | 17.81 | 18.47 | 0.464 | 0.942 | 87% | 11 | 17.76 | 18.39 | 0.47 |
| `jiajia_hold` | 13 | 11 | **18.39** | **18.91** | **0.502** | 0.955 | 84% | 11 | **18.39** | **18.91** | **0.50** |
| `object` | 3 | 20 | **19.92** | **21.57** | **0.599** | 0.676 | 93% | 10 | **19.37** | **20.70** | **0.58** |
| `object` | 8 | 15 | 19.03 | 18.95 | 0.473 | 0.728 | 93% | 10 | 18.71 | 18.54 | 0.46 |
| `object` | 13 | 10 | 19.08 | 18.90 | 0.492 | 0.750 | 95% | 10 | 19.08 | 18.90 | 0.49 |

读法（很重要）：

- **人像场景：加视角的增益几乎全在覆盖度**。alpha 0.789 → 0.955（+0.17），而同一批
  视线上的前景 PSNR 只涨 0.56 dB（18.35 → 18.91）。所谓"3v→13v +3.4 dB"主要是
  从"大片背景没重建"变成"背景也建出来了"。
- **物体场景：3v 反而最好**，且已排除体素尺寸（0.002/0.001/0.0002 三档，18.64~19.03，
  都低于 3v 的 19.37）与覆盖度（alpha 只差 0.07）。**根因未定位**，不能引作结论。
- 对齐自证：`M` 三视角离散 2.43e-07、`det(M)=1.000000`；尺度两种独立估计都是 0.332757
  （离散 1.5e-08）；source 平移反算残差 3.84e-08。→ 渲染的位置不是错位来的。

**资源**（整卡 `nvidia-smi` 采样，不是单进程 allocated/reserved；本轮未采 torch 侧）：

| 配置 | token 数 | 推理 | ms/token | 判据 |
|---|---|---|---|---|
| 23v @756 | 67.1k | 68.30 s | 1.02 | 撞顶（>5× 面积比） |
| 24v @504 | 31.1k | 7.28 s | 0.23 | 正常 |

视角数面板补了显存随 N 的曲线（504², pose）：3v 13.41 / 8v 18.62 / 13v 19.17 / 24v 21.79 GB
（`nvidia-smi` 整卡 used）。→ **24 GB 卡上 756² × 23 视角即上限（23.5 GB）**，不是"还能加几个视角"的余量。
✅ WB-1 面板实测：**504² 下 3~24 视角全部可跑，最低一档 13.4 GB**，做控制变量不需要分批。

**已知坏例**：无（渲染图目视未见幽灵/缺件；`eval/contact_sheet.png` 已存）。
但初版渲染 PSNR 只有 6.75，真因是背景渲成白底而 GT 黑底 —— 已修为按 GT 四边估底色，
**任何拿"渲染质量差"下的结论都要先复核这一点**。3v 目视：人、椅子、背景结构都在，
差异主要落在平坦纹理区（地毯/墙面）的传感器噪声量级残差，不是结构性错误。

## 4. DA3 归因结果

| 对照 | 直接几何误差 | held-out 质量 | 覆盖/脸退化 | 判断 |
|---|---|---|---|---|
| C0 原生 | 未测 | 未测 | 未测 | — |
| C1 xy offset=0 | 未测 | 未测 | 未测 | — |
| C2 depth offset=0 | 未测 | 未测 | 未测 | — |
| C3 两种 offset=0 | 未测 | 未测 | 未测 | — |
| C4 仅替换 means | 未测 | 未测 | 未测 | — |
| GT-center oracle | 未测 | 未测 | 未测 | — |

本轮只确认了一件事：**SplatAnything 与 DA3 不是同一套几何**（WorldMirror-2 自带
camera head + depth + conf + mask，DA3 只有 depth/pose），所以 C0~C4 的"depth offset /
GS offset"命名要按 v3 的实际代码重新落地，不能直接照搬 v2 的补丁位置。

## 5. 训练改造成本

| 项目 | 已完成/实测 | 预计剩余及依据 | 证据路径 |
|---|---|---|---|
| 等价可导前向、渲染器、merge/过滤 | 不可导（前向 `@torch.no_grad()`；merge 是 scatter_add，可导但输出被 detach） | 需自建 | `src/models/models/gaussian_worldmirror.py:25`、`rasterization.py:167` |
| 可训练参数/各分支梯度/教师冻结 | 未做 | — | — |
| 1 step 权重变化与 100 step loss | 未做 | — | — |
| checkpoint + optimizer 保存恢复 | 未做 | 权重是 `state_dict` 格式，可作初始化 | `checkpoints.py:14` |
| 峰值 allocated/reserved、整卡占用 | 只有整卡 used | 需 `torch.cuda.max_memory_allocated` | — |
| 冷启动、前向/反向/merge/step 时间 | 只有前向 | 反向未测 | E-04~E-06 |
| 缓存、checkpoint、优化器状态大小 | 权重 11.33 / 5.05 / 0.18 GB | — | §1 |
| 工程改动/排错工时与依赖缺口 | 见 §6 依赖缺口 | — | — |
| 2k/5k 耗时估算（未运行） | 不适用 | — | — |

训练等级：未定。使用真权重（仅验证加载）。几何/encoder 冻结：官方前向本来就冻结。
未改变 merge/过滤/输入视角语义。

## 6. 可复查交付

**环境（WB-0 实测）**

| 项 | 值 |
|---|---|
| GPU / 驱动 | RTX 3090 24576 MiB / 595.95，空闲 1481 MiB |
| conda env | `splatanything`，python 3.11.18 |
| torch / CUDA | 2.7.1+cu126 / 12.6 |
| gsplat | **1.5.2**（1.5.3 与 torch 2.7 的 `_jit_compile` 签名不兼容，实测报错） |
| nvcc / gcc | conda `cuda-nvcc 12.6`；本机无系统 gcc，借用 `envs/infinisplat_v2/bin` |
| glm | `~/repos/glm_src`（gsplat 1.5.2 不内置，需 CPATH） |

**权重（SHA256 全部来自 HF API `lfs.oid`，下载后脚本校验通过）**

| 文件 | 字节 | SHA256（前 16 位） | 实际加载路径 |
|---|---|---|---|
| SplatAnything.ckpt | 11,334,600,799 | `acbb2544cc213a78` | `~/repos/SplatAnything/checkpoints/` → `/mnt/d/wheel/splatanything_ms/` |
| worldmirror2.safetensors | 5,053,556,568 | `0e37dcf97576b1d8` | 同上 |
| skyseg.onnx | 175,997,079 | `ab9c34c64c3d8221` | 同上 |

软链终点已用 `readlink -f` 核对；`load_model_weights` 打印的 tensor 数（1322 / 1390）
与模型 `state_dict` 完全匹配、`strict=True` 加载成功 —— 不存在"打印 A 加载 B"。

**完整命令**（可直接重放）

```bash
bash splatanything/00_setup_env.sh
bash splatanything/01_download_models.sh
EXAMPLE=object MODE=rgb  bash splatanything/02_run_inference.sh
EXAMPLE=object MODE=pose bash splatanything/02_run_inference.sh
CAPTURE="/mnt/d/dataset/手部微动样例/01e954de...-佳佳抱枕" SCENE=jiajia_hold N_VIEWS=24 \
  bash splatanything/07_run_capture.sh
INPUT=~/output/splatanything_results/_scenes/jiajia_hold MODE=pose RESOLUTION="504 504" \
  bash splatanything/02_run_inference.sh
RESULT=~/output/splatanything_results/pose/object bash splatanything/03_eval_render.sh
python splatanything/05_check_scale.py --result <02 输出目录>

# WB-1 往返验证（跑推理 + 属性/渲染双重比对 + io_audit）
INPUT=~/output/splatanything_results/_scenes/jiajia_hold_3v MODE=pose RESOLUTION="504 504" \
  bash splatanything/06_run_roundtrip.sh

# 视角数控制面板（最远点采样，天然嵌套）
python splatanything/07b_subset_scene.py --scene <源场景> --n 8 --out <目标>
INPUT=<目标> MODE=pose RESOLUTION="504 504" OUT_ROOT=~/output/splatanything_results/panel \
  bash splatanything/02_run_inference.sh
RESULT=~/output/splatanything_results/panel/<scene> bash splatanything/03_eval_render.sh
```

**原始日志/指标**：`<RESULT>/eval/{eval_summary.json, contact_sheet.png, render/*.png}`；
`<RESULT>/{config.yaml, run.json, cameras.json}`；往返验证 `<RESULT>/ply_roundtrip.json`
（含 `io_audit` / 属性 MAE / 体素合并统计）与 `roundtrip_diff.png`。
日志未入库（按纪律不入库大文件），关键行已抄进 `EXPERIMENTS.md`。

**未解决问题 / 依赖缺口**

1. 官方包缺 `validation.py`，`tests/test_inference.py` 无法运行（`04_smoke_cpu.py` 顶上）。
2. `examples/object/prior_camera.json` 的 `num_cameras=49` 与实际 24 条不符（不影响运行）。
3. gsplat 在本机需要四件套（nvcc/gcc/glm/arch），已抽成 `_gsplat_env.sh` 由 03/06 共用。
4. 03 只有自洽口径，held-out 未实现 —— 这是 WB-2 的前置，见 §7。

## 7. WB-2 的落地实现（已完成，`08_*.{sh,py}` + `08b_*.py`）

对齐是唯一的技术难点：导出的云在"source 首相机规范系 + GSN 缩放"里，留出位姿在原始采集系。
做法是**从数据拟合**而不是假设实现细节 —— 解 `R_exp = M·R_o`、`t_exp = s·(M·t_o) + u`，
再用 source 反算残差自证（实测 3.84e-08），并用"成对相机距离比"独立估 s 做交叉校验
（0.332757 / 0.332757，离散 1.5e-08）。任一校验不过就报错退出，不渲染。

踩过的坑（留给下次）：`u` 的系数必须是 `np.tile(np.eye(3), (N,1))`（[3N,3]）；
写成 `np.kron(np.eye(N), np.eye(3))` 会得到 [3N,3N] 块对角，每个方程都放开 → 最小二乘
退化成逐点最优，表现为"旋转对、平移错"（残差 13.9%）。

用法：

```bash
RESULT=<02/06 输出目录> SOURCE_SCENE=<source 场景> ALL_SCENE=<含全部视角的场景> \
  bash splatanything/08_nvs_eval.sh
python splatanything/08b_nvs_compare.py --runs a/nvs_summary.json b/... --csv cmp.csv
```

## 8. 下一步唯一建议及理由

**先查清"物体场景 3v 优于 8v/13v"的根因，同时把区域指标接到人脸/手部。**

理由：WB-2 首轮已经拿到真 held-out，但两个人像相关的问题还答不了：(1) 物体场景出现
反直觉的 "3v 最好"，而体素尺寸与覆盖度两条解释都已被排除，说明还有一个没被识别的因素
在控制质量 —— 在没搞清它之前，"选几个视角"这个对下游最关键的决策没有依据；
(2) 用户真正关心的是**人脸和手**，而当前前景指标是整块 alpha 掩膜，人像只占其中一小部分，
人脸区域的信噪比被背景稀释。ExpTracker 的 657 点与 SAM3 的 mask 管线在仓里已可用，
接上就能给出"人脸 PSNR / 手部 PSNR"，这比再刷整图 PSNR 有用得多。

代价：都不需要改官方代码，只需在 03/08 的度量里加一块掩膜；物体场景的根因排查用现成
产物 + 一次重跑即可。
