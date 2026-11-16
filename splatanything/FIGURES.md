# 出图台账（PPT / 汇报用）

> 图不纳入仓库。成品在 **`D:\splatanything_figures\`**：
> `figures/` **6 张**成品（fig10/11/14/15/16/17）· `raw/` 12 张原始 contact sheet ·
> `pairs/` 14 组 439 张按视角配对的单图 · `metrics.json` 读数汇总 ·
> `_build/` 出图脚本 · 同名 zip 为打包版。
> 数据源全部来自 `~/output/splatanything_results/`，读数对应 [EXPERIMENTS.md](EXPERIMENTS.md) E-04~E-15。
>
> **2026-11-13 成品清理**：fig00~fig09、fig12/13 的成品已删（峰哥本人清理，明说不用重跑）。
> 它们的口径、红线与已知问题**仍保留在本文与 EXPERIMENTS.md 里**，需要时按文末命令重新生成即可。

## 三图前馈纹理化 2DGS 方案图（2026-11-16）

用户要求的整体算法与实验方案图，**均为机制/计划图，不是新方案实测结果**。
7 张 1920×1080 RGB PNG，16:9，可整页铺入 PPT；逐张目视检查文字、连线和页脚。

- 成品：`D:/splatanything_texture_figures/figures/`
- 总览：`D:/splatanything_texture_figures/contact_sheet.png`
- 打包：`D:/splatanything_texture_figures/splatanything_texture_plan_20261116.zip`
- 内容脚本：`D:/splatanything_texture_figures/_build/make_plan_figures.py`（一次性，不入库）
- 可复用组件：`figkit/diagram.py`（文字边界检查、节点、折线箭头、局部纹理盘面示意），
  规格见 `figkit/SPEC.md` §5.9；修改内容不会要求修改组件库。
- 清单与布局审计：输出根目录 `README.md` / `manifest.json` / `layout_audit.json`。

| 图 | 用途 | 来源与必要口径 |
|---|---|---|
| `01_overall_route.png` | 整体路线、已有基础与新候选 | NOTES §0 + 主计划 §0.0；E-15普通属性头100步通过，不代表新架构通过 |
| `02_algorithm_architecture.png` | 冻结双编码器、低密度几何与高分辨率纹理、可导渲染 | NOTES §0.1～0.4；仅借鉴QuerySplat解耦，不称其原版输出纹理2DGS |
| `03_local_texture_mechanism.png` | 普通盘面 / RGB / RGBA纹理对照 | NOTES §0.3；颜色和alpha为示意值，首版四块指2×2采样，非四个子高斯 |
| `04_training_and_evaluation.png` | source/target隔离、训练反馈及评测口径 | R0/R1；仅训练身份反传，测试只评分，target不进重建特征 |
| `05_ablation_plan.png` | Q0～Q5单变量消融 | NOTES §0.5；不含实测胜负，Q1/Q2颜色训练预算需对齐 |
| `06_nose_diagnosis.png` | 鼻部四类假设、检查和对应改造 | R2-N；用户观察不等于双层表面或人物数据不足已证实 |
| `07_milestones_and_gates.png` | 新算子、100步成本探测、域训练与独立验收 | E-15 + NOTES §0.5；新纹理架构成本未知，后续训练按实测排期 |

再生成（默认输出仓外，可用 `--figkit` / `--out` 覆盖）：

```powershell
& C:/Users/wangyufeng/.workbuddy/binaries/python/envs/default/Scripts/python.exe `
  D:/splatanything_texture_figures/_build/make_plan_figures.py
```

本组不复用近光心三摄的实测图，主任务仍是**不同机位三张人像图**。
图库与ZIP不入Git；只提交组件、规格、台账与memory。

## 三机位领导汇报页（t01–t06，2026-11-16；t06 同日补 three_cam 组）

> 成品在 **`D:\portrait_3cam_figures\figures\`**（6 张 1920×1080）·
> 一次性脚本 `D:\portrait_3cam_figures\_build\make_ppt_pages.py`（不入库，§10）·
> 素材副本在 `_build/assets/`。1059 组对应 **E-17**；three_cam 组对应其
> run.json + eval_summary.json（2026-11-16，同口径同脚本，未单列 E 号）。
> 版式：t01/t03/t05 按 §5.8 一眼版（领导 3 秒阅读），t02 按 §5.6 单方法页，t04 按 §5.7 口径对照。

| 文件 | 内容 | 口径 |
|---|---|---|
| `t01_inputs` | 1059 组三张输入图（wide/main/tele）+ 拍摄与标定约定 | 参数为工程记录值 |
| `t02_principle` | SplatAnything 5 步流水线 + 三卡（GSN/双模型/前馈）+ 数字表 | **机制示意，非实验结果** |
| `t03_results` | 1059 组两版轨道抽帧 0°/90° + 边界归因 + 交付物清单 | 自洽 + 目视 |
| `t04_rgb_vs_pose` | 1059 组 22.32 vs 19.43 dB 口径对照 + 五行读数表 | **自洽分差不判优劣（E-17 §3）** |
| `t05_conclusion` | 六格结论（已更新为两组数据）+ 两条汇报纪律 + 一句话 | — |
| `t06_three_cam` | three_cam 组（室内影棚）复测：输入三图 + **两组×两模式读数对照表** + 轨道抽帧 + 读法卡 | 同左，分差仍不排名 |

**讲这组时的红线**：

1. t04 的分差**不排名**——rgb 位姿与重建同源、自洽分结构性偏高；pose 是参考 fx + 假设
   look-at。被追问就翻 E-17 §3 与台账顶部 E-11 口径修正。
2. t03 的「纸片」**不是模型失败**：3 视角 + 1.2~1.4 cm 基线只在正面有三角约束；
   要背面得加视角或换环绕机位。这是本组（不同机位三图）的主线边界，
   与 fig10/fig11（近光心三摄，已不作为主线证据）不是同一回事。
3. 本组数据是**全身 + 街景**（脸占比小），汇报时别说成"人脸重建"。



## 图清单与口径（fig00–fig17 历史批次）

> **2026-11-12 审查修正**：以下图片尚未重新生成。fig10/11 的“真值 pose”、米制基线、
> 基线/深度比及“训练 FOV 外”归因未获支持；不能直接用于正式结论页，详见
> [实验台账顶部复核](EXPERIMENTS.md)。fig03/04/06/08 的历史数字含旧底色协议，FG 是各自
> 预测 alpha mask，并非独立人物区域。fig06 未定位病因，不应声称完全排除覆盖/融合解释。
> fig04 是 13v source 的 held-out，不能用于展示三图方法的 NVS 效果。
> 当前研究对象为不同机位三张人像图，近光心三摄图不作为主线成败证据。

| 文件 | 内容 | 口径 |
|---|---|---|
| `fig00a/b_views_*` | 24 / 23 个输入视角总览（含编号） | 输入侧 |
| `fig01_human` | 人像 3 视角 GT vs 重渲染 | **自洽** |
| `fig01b_human_detail` | 人脸与手部 3× 放大对照 | 自洽 |
| `fig02_object` | 物体 23v 756² 同版式 | 自洽 |
| `fig03_self_vs_heldout` | 两口径趋势相反（核心结论） | 自洽 + held-out |
| `fig04a/b_nvs_*` | 新视角合成，source 13v | **held-out** |
| `fig05_voxel_tradeoff` | 体素 0.002→0.001：PSNR↓ 锐度↑ | 自洽 |
| `fig06_object_paradox` | 物体场景 3v 反超（根因未定） | held-out 交集 |
| `fig07_ply_roundtrip` | 导出无损：MAE 3.4e-08 | — |
| `fig08_summary_table` | 全量读数汇总 | 混合，已分列 |
| `fig09a_cloud_human` | 高斯点云三视图 + 相机环 | — |
| `fig10_three_cam` | 近光心三摄 rgb 现象（图内因果/尺度结论待修） | 自洽，暂不引用归因 |
| `fig11_three_cam_pose` | 三摄 rgb vs 参考参数构造 pose（非实测真值） | 自洽，图中文字待修 |
| `fig12_da3gs_vs_v2` | 方法对比：DA3-GS vs InfiniSplat v2（6 步流水线 + 5 行差异表） | 机制示意，非实验结果 |
| `fig13_v2_vs_v3` | 方法对比：InfiniSplat v2 vs v3（同上版式） | 机制示意，非实验结果 |
| `fig14_r1_angle_buckets` | R1 角距分桶：四方法 × 三档角度 + 5 行读数表 | **held-out**，512² + 精化对齐 |
| `fig15_align_cost` | 3 点对齐的 dB 上限（折线）+ 精化前后输入视角读数（6 方法） | 口径诊断，左图是理论上限 |
| `fig16_region_metrics` | 人脸条 + 区域读数表：PSNR 排名与锐度排名相反 | **held-out**，512² 黑底 + SAM3 mask |
| `fig17_train_cost` | WB-4 训练成本：T0/T1/T2 里程碑 + 100 步 loss 曲线 + lr 敏感性柱图 | **实测**（`train_probe.json`），3v 504²，目标图=输入视角（过拟合冒烟） |

> **磁盘状态**：上表中 `fig00a/b` ~ `fig09`、`fig12`/`fig13` 的 png 已于 2026-11-13 删除，
> 保留的只有 `fig10`/`fig11`/`fig14`/`fig15`/`fig16`/`fig17`。读数与口径未删，照旧可引用。

## 区域指标与底色修正（fig16，2026-11-13）

对应 `EXPERIMENTS.md` **E-14**。这张是"PSNR 不能当排名"的正面证据页。

| 图 | 数据源 | 一次性脚本 |
|---|---|---|
| fig16 | `~/output/batch_4methods/eval_3view_jiajia_bgmask.json`（`audit_31` 的 `BG=auto MASKS=1` 产物）+ `render_crops/*/v021.png` | `D:/splatanything_figures/_build/fig16_region_metrics.py` |

**讲这张时的红线**：

1. 六个方法**全部偏好黑底**（DA3-GS +4.1、VGGT-Ω +7.5、DA3+3DGS +7.1 dB）。所以 fig14/15 与
   E-12/E-13 的白底读数**不可与本节相减**，只能各自内部比。
2. **PSNR 第 1 的 QuerySplat 是锐度第 6**。别把 fig16 的全区 PSNR 列当排名用；
   这张图的结论恰恰是"排名反过来"。
3. 两类失败模式要分开说：QuerySplat / DA3-GS 是**糊**（锐度 0.03/0.09）；
   VGGT-Ω / DA3+3DGS 是**锐但重影**（锐度 0.82/0.61、鬼影率 0.09/0.05）。
4. 页脚那句"3DGS 记死像素网格"不能省：DA3+3DGS 在**原生 720×960 是 43.4 dB**，
   重采样到 512² 只剩 17.9 dB。这个偏差只对 3DGS 系成立，是口径不是画质。
5. mask 来自 SAM3，不是人工标注；face/hand 已按 person 主体门控，但窄区域（手部 ~0.18% 画面）
   读数波动比人脸大，别单独拿手部数字下结论。


## R1 分桶与对齐诊断（fig14/15，2026-11-13）

这两张是本仓**实测结果**，数据来自 `EXPERIMENTS.md` E-13。

| 图 | 数据源 | 一次性脚本 |
|---|---|---|
| fig14 | `~/output/batch_4methods/eval_3view_jiajia_r512.json` → `audit_32_angle_buckets.py` 的 `angle_buckets.json` | `D:/splatanything_figures/_build/fig14_r1_buckets.py` |
| fig15 | 左：`D:/tmp/align_sensitivity.py`（GT 自比测上限）；右：同一 eval json 的 `refine.psnr_pre/_post` | `.../fig15_align_cost.py` |

**讲这两张时的红线**：

1. fig14 的 holdout 是 **512² + 精化对齐 + 固定 GT 内参**口径，与 E-10 的 18.39（全图、SA-only）
   **不可相减**；四方法之间的顺序在三个角度桶里一致，才敢当排名用。
2. fig14 里 >45° 的桶**是空的**（最近输入视角最远只差 44.7°），别在 PPT 上说"大角度外推段"。
3. fig15 左图是 **GT 与自身旋转后自比**的理论上限，不是任何方法的结果；它的用途是说明
   "2.7° 就把所有人按在 13.6 dB 以下"，不要读成"某方法只能到 13.6"。
4. fig15 右边**后三个方法几乎不动**，这是关键信息：它们的问题不在对齐，在别处（E-13 发现 D）。

## 方法对比示意图（fig12/13，2026-11-13）

版式已固化为 **`figkit/SPEC.md`**（1920×1080 满版）：左右两栏各 6 步流水线同 y 对齐 +
下方 5 行差异表，第 6 步用"灰化 / 徽章 / 高亮环"表达差异而非错位。
出图脚本 `D:/splatanything_figures/_build/fig12_13_compare.py`（`import figlib`，**一次性脚本不入库**）；
`_legacy_1200/` 存 1200 版旧图与旧脚本。

这两张**不是本仓实测结果图**，是把代码/文档里已核实的机制差异画成对照页，供 PPT 讲
"这两家到底差在哪"。文字口径全部可回查：

| 图 | 机制来源 | 数字来源 |
|---|---|---|
| fig12 | DA3：`compare_4methods/A19_method_primer_figures.py` 的机制卡（DINOv2 / depth-ray / GS-DPT）；v2：`infinisplat_v2/README_wsl.md` 融合契约 | FINDINGS §2 五方法表（耗时 / 高斯数 / 体积） |
| fig13 | v2：同上；v3：`splatanything/NOTES.md` §1/§2/§3（双模型、12 维属性、GSN、固定体素） | FINDINGS §2、EXPERIMENTS E-04~E-09 |

三条容易讲错的地方：

1. **三家都不学位置**——位置一律由深度反投影得到，网络只出属性。所以"DA3-GS 不学位置、
   v2 学位置"是错的，差别不在这一步。
2. **v2 的精度优势里含回炉**（固定中心 3000 步，自洽 +7.30 dB）；v3 官方把这一步去掉了，
   直接拿 v2 的分数去比 v3 是不对等的。
3. **v3 的 GSN 换的是"跨场景不用调参"**，不是画质。它让体素可以固定 0.002（v2 是
   median_scale×1.4 逐场景自适应）；两者 voxel 尺寸不同，PSNR 不能横比。


## 训练成本冒烟（fig17，2026-11-13）

对应 `EXPERIMENTS.md` **E-15**（WB-4）。这张回答的是"这套官方包能不能训"，不是"训得好不好"。

| 图 | 数据源 | 一次性脚本 |
|---|---|---|
| fig17 | `D:/tmp/train_probe.json`（`09_train_probe.py --stages all` 产物） | `D:/splatanything_figures/_build/fig17_train_cost.py` |

**讲这张时的红线**：

1. **目标图就是三条输入视角本身**——这是过拟合冒烟，**不是 NVS 泛化**。25.93 dB 不能跟
   fig14/fig16 的 held-out 读数放同一栏比。
2. **只训了属性头**：`GSFeatHead` + decoder 共 32.9M 参数，encoder 与全部几何（mean/quat/scale
   的位置来源）冻结。所以这张图证明的是"梯度能走通、能收敛"，**不证明几何能变好**。
3. **lr=1e-3 一步就炸**（loss 1666），图里那根红柱是真实事故记录，不是夸张。原因是
   `scales = exp(conv).clamp_max(0.3)`：1×1 卷积 256 通道累加，Adam 首步把 raw log-scale
   推 ~+5，scale 从 1.77e-3 直接顶到上限 0.3（169 倍），下一步全被 `filter_large_scale_infer`
   清空 → 黑图 → loss 恒定。**真训练必须分通道 lr 或直接回归 log-scale。**
4. **opacity 分支梯度恒为 0 是正常的**，别当 bug 查。合并时 `values = weights if name ==
   'opacities' else ...`，解码出的 opacity 通道根本不参与计算，12 个输出通道里只有它精确为 0。
5. **耗时不能按视角数外推**：0.511 s/step 是 3v/504² 的读数；encoder 占 59% 且**冻结可缓存**，
   缓存后 2k 步从 17 min 降到 7 min——这个优化还没做。
6. 8v / 13v **未测**，PPT 上别写成"训练成本随视角数线性增长"。


## 三张图的正确说法（易讲错的地方）

1. **fig01/01b/02 是自洽口径**，只证明"链路跑通 + 拟合好"，不能跨视角数比较 ——
   E-08 实测同场景自洽 PSNR 随视角数**下降**（22.37 → 19.71）。
2. **fig04 是 held-out**，但展示用的是 3 个代表性视角，全量 11 视角均值 18.39 dB 写在页脚。
3. **fig06 是未定论**：3v 19.37 > 13v 19.08 > 8v 18.71，体素/覆盖度/锐度三条解释均已排除，
   根因未定位，只能说"反常"，不能说"少视角更好"。
4. **fig10/fig11 讲的是"输入不合格"而不是"方法不行"**：同光心三摄视差只有物距的 1.2%，
   三角测量没有基线。fig11 里 pose 模式（喂真值相机）已经把焦距比和相机间距比都复原对了，
   但 tele 深度仍是另两张的 0.57×——结论是"这种输入无法验证 3D"，不能写成"模型失败"。

## 重新出图

```bash
# fig17：训练成本冒烟（数据来自 WSL 侧 09_train_probe.py --stages all）
cd D:/splatanything_figures/_build && python fig17_train_cost.py

# fig14/15：走 figkit（1920×1080，规格见 figkit/SPEC.md）；一次性脚本在 D 盘，不入库
cd D:/splatanything_figures/_build && python fig14_r1_buckets.py && python fig15_align_cost.py

# fig12/13：走 figkit（1920×1080，规格见 figkit/SPEC.md）；一次性脚本在 D 盘，不入库
cd D:/splatanything_figures/_build && python fig12_13_compare.py
FIG_ROOT=/path/to/dir python fig12_13_compare.py    # 换输出根目录

# fig10/11：2026-11-13 已按 1920×1080 重排（原 1100/1210，插入 16:9 会溢出）；fig00~fig09 仍是 1920×1200 历史口径（未套 figkit）
python fig00_views.py && python fig01_02.py && python fig03_trend.py
python fig04_nvs.py && python fig05_06.py && python fig07_08.py && python fig09_cloud.py
python fig10_three_cam.py && python fig11_three_cam_pose.py
```

依赖 pillow + numpy；中文字体走 `C:/Windows/Fonts/msyh*.ttc`。点云数据由
`D:/tmp/sa_roundtrip/export_cloud.py`（WSL 侧）从 `gaussians.ply` 导出为 npz。
