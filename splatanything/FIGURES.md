# 出图台账（PPT / 汇报用）

> 图不纳入仓库（体积 19 MB）。成品在 **`D:\splatanything_figures\`**：
> `figures/` 17 张成品 · `raw/` 原始 contact sheet · `pairs/` 436 张按视角配对的单图 ·
> `metrics.json` 读数汇总 · `_build/` 出图脚本 · 同名 zip 为打包版。
> 数据源全部来自 `~/output/splatanything_results/`，读数对应 [EXPERIMENTS.md](EXPERIMENTS.md) E-04~E-10。

## 图清单与口径

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
# fig12/13：走 figkit（1920×1080，规格见 figkit/SPEC.md）；一次性脚本在 D 盘，不入库
cd D:/splatanything_figures/_build && python fig12_13_compare.py
FIG_ROOT=/path/to/dir python fig12_13_compare.py    # 换输出根目录

# fig00~fig11：2026-11-12 的一次性脚本，仍是 1920×1200 历史口径（未套 figkit）
python fig00_views.py && python fig01_02.py && python fig03_trend.py
python fig04_nvs.py && python fig05_06.py && python fig07_08.py && python fig09_cloud.py
python fig10_three_cam.py && python fig11_three_cam_pose.py
```

依赖 pillow + numpy；中文字体走 `C:/Windows/Fonts/msyh*.ttc`。点云数据由
`D:/tmp/sa_roundtrip/export_cloud.py`（WSL 侧）从 `gaussians.ply` 导出为 npz。
