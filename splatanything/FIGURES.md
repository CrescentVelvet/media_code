# 出图台账（PPT / 汇报用）

> 图不纳入仓库（体积 19 MB）。成品在 **`D:\splatanything_figures\`**：
> `figures/` 13 张成品 · `raw/` 原始 contact sheet · `pairs/` 436 张按视角配对的单图 ·
> `metrics.json` 读数汇总 · `_build/` 出图脚本 · 同名 zip 为打包版。
> 数据源全部来自 `~/output/splatanything_results/`，读数对应 [EXPERIMENTS.md](EXPERIMENTS.md) E-04~E-10。

## 图清单与口径

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
| `fig10_three_cam` | 同光心三摄：rgb 模式的失败链（焦距差→虚构基线） | 自洽 |
| `fig11_three_cam_pose` | 三摄两模式对照（rgb vs 给定真值 pose） | 自洽 |

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
cd D:/splatanything_figures/_build
python fig00_views.py && python fig01_02.py && python fig03_trend.py
python fig04_nvs.py && python fig05_06.py && python fig07_08.py && python fig09_cloud.py
python fig10_three_cam.py && python fig11_three_cam_pose.py
```

依赖 pillow + numpy；中文字体走 `C:/Windows/Fonts/msyh*.ttc`。点云数据由
`D:/tmp/sa_roundtrip/export_cloud.py`（WSL 侧）从 `gaussians.ply` 导出为 npz。
