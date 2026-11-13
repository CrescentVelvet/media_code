# 出图台账（PPT 用）

> 图不纳入仓库（体积 + 一次性）。成品在 **`D:/compare_4methods_figures/figures/`**，
> 出图脚本在同目录 `_build/`（一次性脚本，不入库）。
> 版式规格见 [`figkit/SPEC.md`](../figkit/SPEC.md) §5.6（单方法页）/ §2（画布栅格）。

## 方法一页纸 p01–p04（2026-11-13）

1920×1080 满版，插入 PPT 铺满整页。每页固定五段：它怎么走（横向 5 步）→ 三个关键设计 →
关键数字（论文口径 / 本仓实测）→ 一句话结论 → 页脚口径。

| 文件 | 方法 | 这张图的论点 | 机制来源 | 数字来源 |
|---|---|---|---|---|
| `p01_da3gs.png` | DA3-GS | 冻结 backbone + 一个 GS-DPT 头，一次前向出高斯，全程没有一步在训练高斯 | `media_paper/3D-20251113-Depth-Anything-3.html` §5 / §11 | 论文表 2/3/5/8；本仓 FINDINGS §2 |
| `p02_querysplat.png` | QuerySplat | 几何 / 外观双查询解耦 + 冻结 VGM，因此能不要位姿；SOTA 只在 DL3DV 成立 | `media_paper/3D-20261105-QuerySplat.html` | 论文 Table 1 / Table 14；本仓 FINDINGS §2 |
| `p03_infinisplat_v2.png` | InfiniSplat v2 | 位置钉死只学属性；精度优势里含回炉，比数字先问"哪一档" | `media_paper/3D-20261014-InfiniSplat-v2.html`；[`infinisplat_v2/README_wsl.md`](../infinisplat_v2/README_wsl.md) | 论文表 1 / 表 9；本仓 FINDINGS §2 |
| `p04_splatanything.png` | SplatAnything（v3） | 换的不是画质，是"跨场景不用调参"（体素可固定 0.002） | `media_paper/3D-20261112-SplatAnything.html`；[`splatanything/NOTES.md`](../splatanything/NOTES.md) | 论文 Table 1 / Table 11；本仓 E-04 / E-06 / E-08 |

## 三条容易讲错（已写进图内卡片或页脚）

1. **DA3 的三个提升口径互不一致**：摘要 35.7% / 23.6%，正文 §7.1 是 25.1% / 21.5%，
   abs 页还有 44.3%。图上写摘要口径并注明，引用时直接引表 2 / 表 3。
2. **v2 的分数含回炉**：前馈 33.9 s 的图与加 3000 步后差 +7.30 dB（自洽口径）。
   拿它的前馈数字去跟别家比，两边都会失真。
3. **v3 换的是"跨场景不用调参"**，不是画质：ETH3D LPIPS 它不在前二、PSNR 也非最优；
   与 v2 的场景 / 分辨率 / 体素都不同，只能比机制不能比分数。

## 重新出图

```bash
cd D:/compare_4methods_figures/_build
python make_method_pages.py            # FIGKIT / FIG_ROOT 可用环境变量覆盖
```

依赖 pillow + numpy；模板与色板来自 `figkit/`（`import figlib` + `import onepager`），
中文字体走 `C:/Windows/Fonts/msyh*.ttc`。
