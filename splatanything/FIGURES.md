# 出图台账（PPT / 汇报用）

## 2026-11-24 三图人脸双影逐层追踪

成品 `D:/ppt/20261124-三图人脸双影逐层追踪.pptx`，8 页。P1–3 黑衣 49 同机位来源分解和移除一源，P4 佳佳 49 从鼻尖落点到 P3，P5 灰衣 32 同源及跨源体素对照，P6 两个静态人的实图，P7–8 覆盖范围及下一步。图片为原始留出相机渲染和 GT 局部放大。点图中的黑孔来自采样；P2 红、绿、蓝按输入顺序着色。自动地标的初检图只供排错，读数使用重新检测并经目检的画布。

数据根 `D:/output/results/splatanything_results/ghost_trace_20261124`，逐层 PNG 在各身份/目标帧号目录，PowerPoint 导出预览在 `ppt_preview`。源码和可移机的相对路径图片副本在 `D:/ppt/_src/20261124-三图人脸双影逐层追踪/`。以桌面运行时的 `@oai/artifact-tool/dist/artifact_tool.mjs` 设置 `ARTIFACT_TOOL_MODULE`，运行 `node build.mjs D:/ppt/20261124-三图人脸双影逐层追踪.pptx`。数据重跑命令见 [实验证据](EXPERIMENTS.md)。

## 2026-11-23 三图机位与鼻形实测

交付 `D:/ppt/20261123-三图人像机位验证.pptx`（8 页）。P2/P3 为四组相机与同条件换位姿，P4/P5 是 Daven 与灰衣的两行四列实图（目标、参考机位、SA 估计、Pi3X 估计）；P7/P8 为合成鼻区 p90 及 Daven 24 鼻尖/脸颊局部读数。P7 的校准列需要 source 真值，P8 的前后距离与视角有关。

数据根 `D:/output/results/splatanything_results/pose_nose_20261123`；原图、JSON、八张 PowerPoint 只读渲染预览在此目录及其子目录。构建脚本用 `ARTIFACT_TOOL_MODULE` 指向桌面运行时的 `@oai/artifact-tool/dist/artifact_tool.mjs`。重建命令：`node D:/ppt/figures/da3_human/_build/build_three_pose_check.mjs D:/output/results/splatanything_results/pose_nose_20261123 D:/ppt/20261123-三图人像机位验证.pptx`。

> 图不纳入仓库。成品在 **`D:\ppt\figures\splatanything\`**：
> `figures/` **6 张**成品（fig10/11/14/15/16/17）· `raw/` 12 张原始 contact sheet ·
> `pairs/` 14 组 439 张按视角配对的单图 · `metrics.json` 读数汇总 ·
> `_build/` 出图脚本 · 同名 zip 为打包版。
> 数据源全部来自 `~/output/splatanything_results/`，读数对应 [EXPERIMENTS.md](EXPERIMENTS.md) E-04~E-15。
>
> **2026-11-13 成品清理**：fig00~fig09、fig12/13 的成品已删（峰哥本人清理，明说不用重跑）。
> 它们的口径、红线与已知问题**仍保留在本文与 EXPERIMENTS.md 里**，需要时按文末命令重新生成即可。

## N1 两个真实身份外观页（2026-11-23）

- 正式数据 `n1_real_20261122/real_eval_v18/n1_real_eval.json`；v18 与 v17 的 target 像素逐项一致，
  增加 known-pose 同源 K/渲染回归；v11/v16 失效数字不得引用。
- 新稿 `D:/ppt/20261123-SA脸部局部优化.pptx` 共21页，源工程同步为 `_src/20261123-SA脸部局部优化/`。
  第18页为两个身份的五帧预注册输入；19页为K/crop测量失效；20页为主轨RGB与数值表；21页为紧脸/辅助轨和路线。
- 素材来自正式版本的 target RGB/紧脸 PNG，以原始三联/双联等比展示，文字、表格、结论条保持可编辑。
  不把Zip满覆盖写成几何更好，不隐藏Zip脸部重影；known-pose另轨、raw147脸部N/A，N2冻结。
- 四页均 lint 无诊断并做磁盘截图复核；旧 `20261122` 成品与 `_src/20261119` 历史工程保留。

## N1 单包头脸对照页（2026-11-22）

- 正式数据：`D:/output/results/splatanything_results/n1_head_compare_20261122/rosemary_v27/n1_head_compare.json`。
  命令 `22_run_n1.sh`；中间 v18～v26 不用于汇报，错误黑图/695mm/36.5%焦距均作废。
- PPT `D:/ppt/20261122-SA脸部局部优化.pptx` 新增14～17页：N1协议/主口径数值/头脸与低通/敏感性和下一步。
  沿用第2页输入数据和第13页N0证据；不把旧真人微动包与 Rosemary 静态合成身份混称。
- 图像只用 v27 原生 `*_head_face_lowpass.png` 的真实实验裁图，不整页图片化；所有结论文字、表格和结论条
  仍为可编辑 PPT 元素。页脚注明 ED 为 alpha 混合轴向 Z、无独立 FBX、Zip RGB-only vs SA known-pose。
- 主口径与 Wahba+source 深度尺度敏感性分开呈现；不能按 target 选最好对齐。鼻部均值接近不等于全脸/头部
  结构更好；N2 仍冻结，下一步两个不同身份真实静态包和同输入条件受控轨。

## N0 测量修复页（2026-11-22）

- 正式数据：`D:/output/results/splatanything_results/geometry_audit_v2_20261122/five_view_rigid_v28/`，
  `geometry_audit_v2.json` 含五视角、输入 hash、门禁和局部 ROI；`audit_command.json` 保存参数。
- PPT 第13页为“N0：尺子修好了，局部几何仍要查”，旧第12页保留并加失效提示。素材只使用
  `*_fused_error.png`、`*_premerge_lifting_error.png` 与人工 ROI 审核图；不把误差热力当 RGB 渲染。
- 读数边界：target 不参与尺度拟合；ED 不是单一表面；原 FBX 缺失；旧 218.72mm 鼻翼值因 ROI
  跨嘴角/面颊而失效，source38 保留鼻尖尾部继续查。不得恢复“几何无错/只训属性”。

## F3 网络侧审计页（2026-11-19，晚）

> **2026-11-21失效标记（未重制PPT）**：第12页深度数字用错前一帧GT，且“cameras.json
> 错误”来自深度尺度混比；暂不得引用“几何已正确/只训属性”。第11页“静态无碎裂”的文案
> 也与保存的P3裁图不符。第9页FLAME G2/G3只是当前近似审计，不能用于否定FLAME。
> 最新证据和下一步以EXPERIMENTS顶部复审与主计划N0～N2为准。

PPT 第 12 页 `12.slide`。素材（`D:\ppt\_src\20261119-SA脸部局部优化\assets\`）：
- `f3_teacher_err.png` / `f3_fused_err.png`（视角 020 误差热力，±20mm 满幅，person mask 内）
- `f3_layers.png`（视角 020 脸 ROI 高斯点按偏差着色）· `f3_layers_t34.png`（target 034，备用）
数据源 `pose_rosemary/rosemary33v/face_opt/depth_audit/`（20_f3_depth_audit.py，
渲染相机=S@pkg c2w；GT=EXR depth.V 1024²→756²）。读数见 EXPERIMENTS.md 2026-11-19 F3 条。

## 盲测页：真实静态人物段（jiajia_hold）（2026-11-19，晚）

- **PPT 已扩为 11 页**：`slides/11.slide`（盲测）+ assets（blind_t017/blind_t018 ROI 对照）。
- 数据源：[EXPERIMENTS.md](EXPERIMENTS.md) 2026-11-19「真实静态人物段盲测」节；
  产物 `pose_blind/jiajia_hold_blind3v/`（source=[0,4,22]、target=[17,18]）。
- **口径提醒**：target 008（后脑勺）/009（detection 误检货架玩偶）是协议级无效视角，
  换 target 属于预筛修正不是按指标挑选；同人不同段，不是全新身份盲测。

## F1 保守自由度对照页（2026-11-19，晚）

- **PPT 已扩为 10 页**：`slides/10.slide`（F1 保守对照）+ assets
  （target_000/001_000034_roi_GT_p0_f1a_f1b、f1b_face_crops）。
- 数据源：[EXPERIMENTS.md](EXPERIMENTS.md) 2026-11-19「F1 保守自由度对照」节
  （f1a/f1b metrics.json + target_eval/target_metrics.json）。
- **口径提醒**：F1-A 改善与 P3 同量级但冻结了中心/颜色/透明度；F1-B 98.7% 触顶是
  「上限过紧」的诊断读数，不是「位移没用」；两臂均未过 5% 门，F1 本轮已按计划停止。
- 新脚本（入库）：`19_f1_conservative_opt.py` + `19_run_f1.sh`。

## F0/F2 补充页：约束审计 + 相机 bug 排雷 + FLAME 审计（2026-11-19，晚）

- **PPT 已扩为 9 页**：`D:/ppt/20261119-SA脸部局部优化.pptx`，源工程新增
  `slides/07.slide`（F0 约束审计）/ `08.slide`（human_fbx_gt 相机 bug 排雷）/
  `09.slide`（F2 静态包对照 + FLAME 审计未过）+ 5 张 assets。
- 数据源：[EXPERIMENTS.md](EXPERIMENTS.md) 2026-11-19「F0 约束审计 + F2 静态包对照」节
  （audit/audit_metrics.json、flame_fit.json、rosemary 对照表）。
- **口径提醒**：FLAME 三门（G1 8.78px / G2 18.6mm 且鼻深顺序错 / G3 IoU 0.30）**全未过**，
  按主计划停在审计处、不把错误模板绑高斯；rosemary 仅 −1.6~−2.3% 未过 5% 推进门。
- 新脚本（入库）：`16_face_constraint_audit.py`（F0 主脚本）、`17_fbxgt_to_scene.py`
  （fbx_gt 包→场景+相机修正）、`18_flame_fit.py`（FLAME 三视角拟合+审计门）。

## P0~P3 图文 PPT 与 target 验收页（2026-11-19，晚）

> **复审标记（本轮未重制PPT/PNG）**：旧source页若写“中心几何最大瓶颈”“外观不是瓶颈”
> 或“5.15mm”，已被本次复审纠正，不能继续引用。source侧脸也有碎片，P2已开始恶化；
> 指标为2.3倍扩展脸框而非皮肤专属，位移为模型单位。汇报须连同[最新复审](EXPERIMENTS.md)
> 和主计划F0～F3解释，不把绿条当几何修复证据；两target不能支持统计显著性断言。

- **图文 PPT 成品**：`D:/ppt/20261119-SA脸部局部优化.pptx`（6 页，可编辑文本+真实实验图），
  源工程 `D:/ppt/_src/20261119-SA脸部局部优化/`（DESIGN.md + slides/01~06.slide + assets/）。
  页面：封面 / 输入数据（3 source + 2 held-out target 实拍）/ 方法链路 / P0~P3 拟合表 /
  target 验收表 / 结论与下一步。重建：`slidep upsert-dsl <pptx> --dsl-file <NN.slide>
  --project-dir <工程目录>`（assets/ 相对工程目录解析，必须显式传 --project-dir）。
- **target 验收单图**：`D:/ppt/figures/splatanything_faceopt/figures/fig_faceopt_target_20261119.png`
  （1920×1080），脚本 `D:/ppt/figures/splatanything_faceopt/_build/make_faceopt_target_page.py`；
  素材 `D:/output/results/splatanything_results/face_opt_assets/target_*_roi_*.png`。
- 数据源：[EXPERIMENTS.md](EXPERIMENTS.md) 2026-11-19「独立 target 验收」节。
- **口径提醒**：target 验收页是红结论条（推进门未过），与 fig_faceopt_p0p3 的绿结论条
  （仅拟合轨道）不矛盾——前者才是验收口径，讲的时候按 target 页为准。

## P0~P3 脸部局部优化诊断页（2026-11-19）

- 成品：`D:/ppt/figures/splatanything_faceopt/figures/fig_faceopt_p0p3_20261119.png`（1920×1080）。
- 内容：GT|P0|P1|P2|P3 五列脸部裁图（view0 正视）+ 三臂关系表（PSNR Δ / 锐度比 / 解读）
  + 保护项核验（固定块 hash / ROI 外保真 / person PSNR）+ 结论条。
- 数据源：[EXPERIMENTS.md](EXPERIMENTS.md) 2026-11-19「P0~P3 脸部局部优化诊断」节；
  裁图素材 `D:/output/results/splatanything_results/face_opt_assets/`（face mask 逐视角定框）。
- 出图脚本：`D:/ppt/figures/splatanything_faceopt/_build/make_faceopt_page.py`（一次性，不入库）；
  本机 python 直接跑（`~/.workbuddy/binaries/python/envs/default/Scripts/python.exe`，依赖 figkit）。
- **口径提醒**：三图全进优化、无 held-out target → 拟合/定性轨道，不是新视角验收；
  图上已标注，讲的时候不要说成"几何修复已验证"。


## 人物数据 LoRA 优先汇报（2026-11-17）

- 新稿：`D:/ppt/20261117-实验进展汇报.pptx`，
  14页，可编辑文本；原稿 `D:/ppt/20261116-实验进展汇报.pptx` 保留。
- 来源：[主计划 L0～L4](../da3_human/EXPERIMENT_PLAN.md)与本目录EXPERIMENTS中的证据审查。
  **方案尚未执行**；E-21外部深度替换失败不能证伪人物数据适配，E-20局域不一致不证明数据根因。
- 修改页01/03/05/06/07/11/12/13，新增14；第11页训练读数仅为11月16日历史快照。
  02/04/08/09/10为历史背景，当前决策以07与12～14页为准；不把旧进展当作LoRA实验结果。
- 一次性内容生成器：`D:/ppt/_src/20261117-实验进展汇报/_build/update_lora_report_20261117.py`。
  生成器准备 `_src` 目录内 `slides/*.slide` / `DESIGN.md` / `UPDATE.md`；pptx 平铺在 `D:/ppt/` 根。
- 再生成：先运行生成器，再用本机 `slidep` 打开副本，按页执行
  `upsert-dsl <新稿> --dsl-file <新目录>/slides/NN.slide --page-index <NN-1>`；
  第14页在13页副本上用 `--page-index -1` 追加。已有14页副本则用索引13替换，勿重复追加。
- 验证：9页逐页 `lint` 无诊断；`screenshot --from-disk` 导出保存后页面，人工查看总览及关键页；
  python-pptx复核14页。截图位于新目录 `preview/`，总览 `preview/contact_sheet.png`。
- 恢复旧汇报直接打开原稿，不删除新稿或实验产物；本节只记录汇报版本，不代表训练已完成。

## 三图前馈纹理化 2DGS 方案图（2026-11-16）

> 2026-11-17：本组保留为后续候选；当前先执行人物数据LoRA对照，见上方汇报与主计划。

用户要求的整体算法与实验方案图，**均为机制/计划图，不是新方案实测结果**。
7 张 1920×1080 RGB PNG，16:9，可整页铺入 PPT；逐张目视检查文字、连线和页脚。

- 成品：`D:/ppt/figures/splatanything_texture/figures/`
- 总览：`D:/ppt/figures/splatanything_texture/contact_sheet.png`
- 打包：`D:/ppt/figures/splatanything_texture/splatanything_texture_plan_20261116.zip`
- 内容脚本：`D:/ppt/figures/splatanything_texture/_build/make_plan_figures.py`（一次性，不入库）
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
  D:/ppt/figures/splatanything_texture/_build/make_plan_figures.py
```

本组不复用近光心三摄的实测图，主任务仍是**不同机位三张人像图**。
图库与ZIP不入Git；只提交组件、规格、台账与memory。

## 三机位领导汇报页（t01–t06，2026-11-16；t06 同日补 three_cam 组）

> 成品在 **`D:\ppt\figures\portrait_3cam\figures\`**（6 张 1920×1080）·
> 一次性脚本 `D:\ppt\figures\portrait_3cam\_build\make_ppt_pages.py`（不入库，§10）·
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
| fig16 | `~/output/batch_4methods/eval_3view_jiajia_bgmask.json`（`audit_31` 的 `BG=auto MASKS=1` 产物）+ `render_crops/*/v021.png` | `D:/ppt/figures/splatanything/_build/fig16_region_metrics.py` |

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
| fig14 | `~/output/batch_4methods/eval_3view_jiajia_r512.json` → `audit_32_angle_buckets.py` 的 `angle_buckets.json` | `D:/ppt/figures/splatanything/_build/fig14_r1_buckets.py` |
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
出图脚本 `D:/ppt/figures/splatanything/_build/fig12_13_compare.py`（`import figlib`，**一次性脚本不入库**）；
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
| fig17 | `D:/tmp/train_probe.json`（`09_train_probe.py --stages all` 产物） | `D:/ppt/figures/splatanything/_build/fig17_train_cost.py` |

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
cd D:/ppt/figures/splatanything/_build && python fig17_train_cost.py

# fig14/15：走 figkit（1920×1080，规格见 figkit/SPEC.md）；一次性脚本在 D 盘，不入库
cd D:/ppt/figures/splatanything/_build && python fig14_r1_buckets.py && python fig15_align_cost.py

# fig12/13：走 figkit（1920×1080，规格见 figkit/SPEC.md）；一次性脚本在 D 盘，不入库
cd D:/ppt/figures/splatanything/_build && python fig12_13_compare.py
FIG_ROOT=/path/to/dir python fig12_13_compare.py    # 换输出根目录

# fig10/11：2026-11-13 已按 1920×1080 重排（原 1100/1210，插入 16:9 会溢出）；fig00~fig09 仍是 1920×1200 历史口径（未套 figkit）
python fig00_views.py && python fig01_02.py && python fig03_trend.py
python fig04_nvs.py && python fig05_06.py && python fig07_08.py && python fig09_cloud.py
python fig10_three_cam.py && python fig11_three_cam_pose.py
```

依赖 pillow + numpy；中文字体走 `C:/Windows/Fonts/msyh*.ttc`。点云数据由
`D:/tmp/sa_roundtrip/export_cloud.py`（WSL 侧）从 `gaussians.ply` 导出为 npz。
