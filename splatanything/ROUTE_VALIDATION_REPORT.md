# SplatAnything / DA3 路线验证报告（WorkBuddy 回填）

> 2026-11-13 第四轮回填（R0 口径修正 + R1 角距分桶）。**只填实测到的**；未做的保持 TODO，
> 计划阈值不复制成实测值。
> 唯一任务入口：[EXPERIMENT_PLAN.md §0](../da3_human/EXPERIMENT_PLAN.md)。
> 判据与原理：[路线评审附录](../da3_human/SPLATANYTHING_ROUTE_REVIEW.md)。
> 实验事实另见 [EXPERIMENTS.md](EXPERIMENTS.md)（E-01~E-13）。

## 0. 最新裁决（2026-11-13，E-13 优先于此前所有数值）

**R0 找到了一个真 bug**：三图方法只预测 3 个相机，跨方法对齐只能用 3 个中心解 Sim(3)
（9 方程 / 7 未知数，零冗余），实测残差 2.0~2.7°。本场景是 1435² 长焦（fx≈1381），
2.7° ≈ 65 px，**PSNR 硬上限只有 13.65 dB** —— 此前"六家全挤在 8.9~12.9 dB"量的不是模型，
是对齐精度。已用「输入视角光度精化」（`sim3_refine.py`，`REFINE=1`）+ 统一 512² 修好。

修好后的四方法 holdout：**QuerySplat 16.03 > InfiniSplat v2 12.56 > SplatAnything v3 11.25
> DA3-GS 10.07**（按角距分桶后顺序一致）。VGGT-Ω+3DGS / DA3+3DGS 的自洽分（10.1/10.8）
与其自身训练渲染（51.3/56.7 dB）矛盾，**评测侧未通过，不排名**。

| 裁决 | 当前证据 | 尚缺什么 |
|---|---|---|
| 链路是否成立 | PASS：真权重、3/8/13v 推理、PLY 往返 RGB MAE 3.4e-08 | 不代表画质/几何 PASS |
| 三图质量是否正式达标 | INCONCLUSIVE：六方法齐（E-14 修完底色/主点），但**PSNR 组间差 4.0 dB 小于口径噪声**，锐度比差 20 倍且与 PSNR 排序相反 | 多场景复现、独立（非 SAM3）人脸/手 mask、几何指标、CI |
| DA3 是否需要重训几何 | INCONCLUSIVE：C0～C4 未做 | 不是新主线的必做前置；不因缺此结果继续投入 DA3 重训 |
| 训练改造成本 | **T2 通过**：等价可导前向 max\|Δ\| 4.77e-07、100 步 loss −57.3%（22.24→25.93 dB）、存恢复逐位一致、0.511 s/step、峰值 12.6 GB | 8/13v、bf16、梯度累积、分通道 lr；**不代表全模型训练可行**（E-15） |

## 0.1 Codex 审查与当时裁决（2026-11-12，历史，数值已被 E-13 取代）

用户确认目标是**不同机位的三张人像图**，并目视认为 SplatAnything 明显优于 DA3。
研发主线据此转为 SplatAnything，DA3 保留对照、暂停几何重训；正式达标仍待配对证据。
第二轮唯一执行入口为 [主计划 §0.0 R0～R4](../da3_human/EXPERIMENT_PLAN.md)。

| 裁决 | 当前证据 | 尚缺什么 |
|---|---|---|
| 链路是否成立 | PASS：真权重、3/8/13v 推理、PLY 往返 RGB MAE 3.4e-08 | 不代表画质/几何 PASS |
| 三图质量是否正式达标 | INCONCLUSIVE：1 人像 + 1 物体，只有 SA；用户主观观感支持优先投入 | R0 口径修正、DA3 配对、独立人物/脸/手 mask、几何及 CI |
| DA3 是否需要重训几何 | INCONCLUSIVE：C0～C4 未做 | 不是新主线的必做前置；不因缺此结果继续投入 DA3 重训 |
| 训练改造成本 | T0 仅推理 | 等价可导前向、单步、100 step、恢复、峰值显存/耗时均未测 |

审查直接读取了 `nvs_compare.json`、`nvs_compare_object.json`（WSL 根路径
`/home/velvet/output/splatanything_results/`），并核对 03/08/08b 源码。
人像共同 11 target 的 3v/8v/13v 全图 PSNR = 15.00/17.76/18.39 dB；这是旧底色协议读数，
不是 SA 相对 DA3 的提升。查看 `D:/splatanything_figures/figures/fig01b_human_detail.png`
（3v self）和 `fig04a_nvs_human.png`（13v held-out）可见脸/衣物细节仍有模糊，
但不能将后者当成 3v held-out 图，也未在本次审查目视 DA3 配对图。

需纠正的证据边界：

- `前景 PSNR` 实为各预测的 alpha>0.5 区域，跨运行不是同一像素集合，更非人物分割。
  不能由此判定 3→13v 增益“几乎全是背景覆盖”；R0 改为独立固定 target masks。
- 08 从第一个 target GT 边缘估底色，target 集不同可能改变底色。正式比较改为固定/仅 source
  决定的共享底色，重新渲染并标协议版本；历史数字保留，不能视为完全受控对照。
- self 可诊断输入拟合和链路，不能替代 held-out 泛化；不是“只能证明能跑通”。更多输入在
  固定 target 上可以比较，self/held-out 排序相反本身不证明体素合并是唯一病因。
- 单次体素扫参没解释物体反常，不能说已排除全部融合/覆盖原因；它降为次要待办，不阻塞三图人像。
- alpha 低不是自由空间泄漏/幽灵率；已展示图不能支持“无坏例”或差异全是传感器噪声。
- `nvidia-smi` 是整卡 used；13.41 GB / 1.0 s 是该三图运行的记录，不代表训练可用显存、
  端到端时延或硬件容量上限。必须按 WB-4 实测训练成本。
- E-11 的 prior_camera 来自参考参数+假定 look-at，非已验证实测真值；GSNorm 后基线与
  未同尺度 depth 中位数不能直接相除。其归因见实验台账审查，且与本次不同机位三图任务分开。

以下保留 WorkBuddy 实测及历史上下文；有冲突以本节和新主计划为准。

### 0.1 新增鼻部问题回填（R2-N，尚未执行）

用户反馈：物体效果很好，人物鼻子变宽、疑似重影。当前是用户观察，不是已证实双层表面。
详细矩阵、输出与判据在主计划 R2-N；外观域训练之前先归因，成本冒烟可独立准备。

| 项目 | 状态 | 必须回填 |
|---|---|---|
| 原坏例定位、是否三图、静态/运动 | TODO | run 绝对路径、source IDs、模式/相机/裁剪、用户所见视角 |
| N0/N1 输入与相机 | TODO | K/crop 审计、对应误差、RGB/可信 pose 对照；无标定明确局限 |
| N2/N3 中心/合并/footprint | TODO | 同次前向分层缓存、同表面偏移、合并前后与深度剖面 |
| 鼻部 held-out 度量 | TODO | 宽度/标注误差、脸部质量、控制案例、失败视角 |
| 最小改造建议 | INCONCLUSIVE | 相机修正 / 几何适配 / 融合尺度 / 域适配；支持与反证 |

训练数据人物少是待验证假设：数据集名单不提供人脸覆盖量，物体好/人物差也可能由姿态、
运动、脸部像素数或几何融合引起。不得直接回填“数据不足已证实”。

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
| WB-0 输入/环境冻结 | RUNNING | 见 §1、§6；`splatanything/{00,01}_*.sh` | 权重/加载/版本已核对；缺 inventory/manifest/完整命令与独立分组 | 按已有元数据保守分组；未知身份标待核，不擅称独立 |
| WB-1 真权重与导出渲染 | PASS | E-04~E-09；`03_eval_render.sh`、`06_run_roundtrip.sh` | PLY 往返 PASS、3v/8v/13v 人像已跑 | 不代表质量 PASS |
| WB-2 配对质量与资源 | PARTIAL（R0 已修，R1 首轮过，六方法齐） | E-10、E-13、**E-14**；`audit_31/32/33`、`sim3_refine.py`、`region_metrics.py` | R0 口径坑已修（3 点对齐残差主导）；R1 分桶完成；**六方法全部可渲**（E-14：底色 + 主点两个约定）；区域指标已接（person/face/hand） | 结论仍不可当排名：PSNR 组间差 4.0 dB < 口径噪声（对齐残差、网格重采样、覆盖 80~95%），改看区域指标 |
| WB-3 DA3 中心来源 | TODO | — | 需改官方代码（C0~C4 对照），本轮未动 | 待 WB-2 结论稳定后 |
| WB-4 训练改造成本 | **DONE（T2）** | **E-15**；`09_train_probe.py`、`09_run_train_probe.sh`、`train_probe.json` | 等价可导前向（max\|Δ\| 4.77e-07）、单步分支梯度、100 步 −57.3%（22.24→25.93 dB）、存恢复逐位一致、0.511 s/step、峰值 12.6 GB | 8/13v 未测；**目标图=输入视角，属过拟合冒烟**；`lr=1e-3` 一步就炸，域训练需分通道 lr |

## 2. 三个问题的裁决

| 问题 | 本轮结论 | 支持证据 | 反对/缺失证据 | 适用范围 |
|---|---|---|---|---|
| SplatAnything 达标了吗 | INCONCLUSIVE，研发优先 | SA held-out 首轮与用户观感；PLY 往返通过 | 阈值已在主计划/附录 §6；尚缺 DA3 配对、独立人物/脸/手指标、几何/LPIPS/CI 与 R0 修正 | 仅给旧协议水位，不宣称正式胜出 |
| DA3 需要重训哪一层 | **未判定** | — | C0~C4 一个都没跑 | — |
| 训练改造到 T0/T1/T2 哪一级、成本多少 | **T2（属性头微调可行）** | E-15：等价前向 4.77e-07、100 步 0.005975→0.002553、恢复后逐位一致、0.511 s/step / 12.6 GB / ckpt 377 MB | 8/13v、bf16、梯度累积未测；全模型/骨干训练未验证 | 3v·504²·RGB·冻结 encoder 与几何 |

**本轮最重要的方法学结论（比上面的数字更重要）**：自洽 PSNR 与 held-out PSNR
在同一批数据上**方向相反** —— 自洽 3v 22.37 → 24v 19.71（降 2.7 dB），held-out
3v 15.00 → 13v 18.39（升 3.4 dB）。此前所有基于自洽分数的排序（包括本报告上一版的
"19.01 / 19.71"）**都不能用来比方法或比配置**，只能证明"能跑通"。

## 3. 质量与资源门槛逐项结果

附 `gate_results.json` / CSV：**待补**。下表为 self，§3.1 已有 held-out 首轮，均尚不足正式门槛判定。

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
- 3/8/13v 已做；独立人物/脸/手区域、LPIPS、几何/轮廓、空洞/泄漏、CI 聚合仍待补。
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

C0～C4 专门用于 **DA3** 的 depth/GS offset 来源诊断，不要求适配到 SplatAnything v3。
两方法几何机制不同，不能把 SA 的推理成功当作 DA3 已定位病因。该实验当前降为备用。

## 5. 训练改造成本

**实测已完成（WB-4 → E-15，`~/output/splatanything_results/train_probe.json`）**：
3v / 504² / RGB 模式，冻结 encoder 与全部几何（depth、相机、means），
只训 `GSFeatHead` + `decoder`（128→12 的 1×1 卷积）。

| 项目 | 实测结果 | 证据路径 |
|---|---|---|
| 等价可导前向 | 与官方 `no_grad` 路径逐字段 max\|Δ\| **4.77e-07**，渲染 RGB MAE **2.53e-07**，合并点数同为 488,539 | `09_train_probe.py --stages fwd` |
| 可导光栅化 | 官方**没有**，用 gsplat 1.5.2 补（SH=0、eps2d=0.01、黑底） | 同上 |
| merge/过滤 | 官方原样实现**不可反传**（`scatter_add_` 与 `/=norm` 两处 in-place 触发 version 冲突），换成 `index_add` 等价版后通过 | E-15 §3.1 |
| 可训练参数 | **32,934,540**（63 张量 / 125.6 MB），63/63 全部更新 | `train_probe.json` |
| 各分支梯度 | quats/scales/sh/weights 全 finite；**opacity 精确为 0**（合并规则用 `Σw²/Σw` 替换它，不是链路故障） | E-15 §3.2 |
| 教师/encoder 冻结 | encoder 有梯度张量 **0**、校验和变化 **0**；教师变化 **0** | `step1.frozen_*` |
| 1 step → 100 step | loss 0.005975 → **0.002553（−57.3%）**，自洽 22.24 → **25.93 dB**，无 NaN | `steps.curve` |
| checkpoint + optimizer 恢复 | ckpt **377.0 MB**（可训 125.6 + 优化器 251.3）；恢复后一步 loss 与第 100 步**逐位相同** | `resume` |
| 峰值显存 | **12.62 GB** alloc / **14.93 GB** reserved（3090 24 GB，未 OOM） | `steps.mem` |
| 冷启动 | **24.3 s**（建模型 + 载权重；另 gsplat JIT 编译失败 1 次后命中缓存） | `build_s` |
| 稳态 step 时间 | **0.511 s** = encoder 0.302 + 解码/过滤/合并 0.076（合并仅 0.0028）+ 光栅前后向与优化器 0.133 | `steps.timings` |
| 2k/5k 耗时（**未运行**） | 线性外推 17.0 / 42.6 min；若缓存冻结特征 7.0 / 17.4 min | E-15 §7 |
| 工程改动 | 新增 2 文件、改官方仓 **0 行**；实现+排错工时**未记录** | E-15 §7 |

**训练等级：T2（100 步 + 恢复通过）。** 边界：目标图就是三个输入视角，属**过拟合冒烟**；
**不代表全模型训练可行，也不代表 NVS/人像质量会提升**；8v/13v 未测，不可按视角数外推。

**同轮发现的关键风险**：`lr=1e-3` 在**第 1 步**就把 scale 顶到 `clamp_max(0.3)`
（初始值的 169 倍），第 2 步点数被尺度过滤清零、loss 退化成常数。真做域训练必须
分通道 lr 或梯度裁剪，**不能沿用 3DGS 的常规 lr**。详见 E-15 §4。


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
4. 03 只有 self；08 已实现 pose 模式 held-out 首轮，R0 尚需修正底色/mask 和验收 RGB-only 对齐。

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

**R0 已修（3 点对齐残差）、R1 首轮完成（角距分桶）、R3/WB-4 已达 T2。剩两件事：**

1. **修两个 `cameras.json` 方法的 PLY-相机一致性**（VGGT-Ω+3DGS、DA3+3DGS）：它们自己训练渲染
   51~57 dB、经我方光栅器只有 10 dB，根因未定位。**这不修，六方法排名就永远是 4 路。**
   建议从 `colmap_scene` 的缓存相机与 `cameras.json` 对账开始（怀疑 JSON 不是训练用的那一批）。
2. **R1 补人像域独立 mask 指标**（人物/脸/手），现在全图 PSNR 被背景与地面主导。
   这正是用户"鼻子变宽/疑似重影"观察需要的口径（R2-N）。

**训练侧（E-15 已收口，报告 Codex 后再动）**：T2 只说明属性头微调跑得动。真要排域训练，
前置一条硬约束——**必须先把 loss 从"全图 RGB MSE"换成区域/Mask 口径**（否则 100 步涨的
3.7 dB 全在背景与地面），并给 scale/opacity 单独的小 lr（`lr=1e-3` 一步就炸，E-15 §4）。
**不建议**据此直接开 2k/5k：现在的 100 步是过拟合冒烟，没有 held-out 视角，涨分不能当收益。

输入张数已由用户固定为三张不同机位图，不再等待物体场景视角数反常归因。
区域指标要用独立 mask，不能只给 alpha mask 改名；底色修正需重渲染。
通过冻结几何审计后优先 A1 域适配与 A2/A3 外观/ROI 设计，只有直接几何错误证据才进入
几何小适配器。当前未实测训练，不预估“几天可改完”或保证 3090 可训练。
