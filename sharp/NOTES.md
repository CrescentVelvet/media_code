# SHARP — 机制与三图人像接入评审

## 2026-11-17：值得作为细节对照，不能预设为更准确的几何

当前唯一实验入口：[三图总计划 SHARP 支线](../da3_human/EXPERIMENT_PLAN.md)。
本文记录机制/接口，实测记入 [EXPERIMENTS.md](EXPERIMENTS.md)。人物LoRA主线优先级保持。

> **2026-11-18复核**：修正双重外参后有跨视角收益，但未完成SA相同target配对。
> 先做SH1b协议修复；SH2/SH3暂缓待证，不称已否证。三份原始JSON去重16对的S−I中位
> 为+3.62dB、S−P为+2.11dB；当前结论以EXPERIMENTS顶部v4为准。

### 论文支持什么

[官方论文](https://arxiv.org/abs/2512.10685)与[项目页](https://apple-aiml-research.github.io/ml-sharp/)
说明SHARP从单张照片前馈生成高斯，重点是附近新视角的高保真渲染。其输出高分辨率与细节是
引入对照的理由，不是人脸几何比SplatAnything准确的证据。相同3D场渲染的连贯性、独立输入
预测之间的配准一致性、真实几何准确度是三个不同问题。

论文§3的1536²输入、2×768²高斯与两层表示，为细节提供容量；深度与高斯属性共同服务渲染，
因此高斯中心/渲染深度不天然等同于真实表面。基于该机制推断：三图分别生成再拼接可能出现
尺度/鼻宽不一致、重叠opacity、遮挡冲突；这些是待测风险，不能写成SHARP已失败。

### 两层深度到底是什么（代码级，2026-11-17 实测修订）

完整链路（`initializer.py` → `composer.py`）：

1. `monodepth_predictor.head[4]` 的 `out_channels=2`（`replicate_head(num_repeat=2)` 靠
   `weight.repeat(2,1,1,1)` 复制单层头权重），输出**两张深度图** `[B,2,H,W]`。
   `sorting_monodepth=False` 时**不强制近/远排序** —— 这是它为什么不能当「前/后表面」用的根源。
2. `MultiLayerInitializer(num_layers=2, stride=2, first_layer_depth_option="surface_min",
   rest_layer_depth_option="surface_min")`：两层各自 `1/depth` 后 `max_pool2d(2,2)` 池化到
   **2× 下采样**网格（`_create_surface_layer`），得 `[B,1,N,H,W]`。
3. `_create_base_xy`：`mean_x_ndc` 造成 `[B,1,N,H,W]`，`xx=arange(0.5*stride, W, stride)` 归一化
   到 NDC —— **两层共用同一套 xy 网格，只有 z 不同**。
4. `prepare_feature_input`：`cat([image(3ch), disparity_factor/depth])` → 按 `depth.shape[1]`
   是 1 还是 2 决定通道数（`GaussianDecoderParams.dim_in=5` 对应两层）。
5. `DirectPredictionHead` 每层各出 14 维 delta（`3*num_layers=6`、`11*num_layers=22`）。
6. `composer.py:138` 展平：`[B,C,N,H,W] --permute(0,2,3,4,1)--> [B,N,H,W,C] --flatten(1,3)-->`
   线性索引 `l*H*W + y*W + x`，即 **layer-major**（已用 hook 逐元素对拍证明，见
   [EXPERIMENTS.md](EXPERIMENTS.md)）。

`output_resolution = internal_resolution()//2 = 768` → `2 × 768² = 1,179,648` 个高斯，
与实测 PLY 点数一致。当前样例不宜直接删层（半分辨率黑底掉7.57～7.70dB）；
这不否定后续有监督压缩，也不意味着所有场景压缩成本相同。

**当前样例不支持固定的「前表面/遮挡背面」分工**：两层z中位差0.187mm、`|dz|<1mm`
占41.6%、45%的像素第二层更近；渲染分辨率横向错位中位0.746px。亚像素采样协同是
合理机制假设，但没有证明普遍同表面或“alpha等价低通”。跨视角错位尾部也不能忽略。
这些统计不是融合去重阈值；默认保留完整两层，详见EXPERIMENTS的SH0.5及11月18日审查。


### 输入亮度口径：`load_rgb` 返回 0..255，`predict` 内部再 /255（2026-11-17 踩坑）

官方 `cli/predict.py:168` 是 `torch.from_numpy(image.copy()).float().permute(2,0,1) / 255.0`，
而入参 `image` 来自 `io.load_rgb()`，**是 uint8 0..255**。自己封装时若把读取函数改成返回
0..1 的 float（很自然，因为要拿去算 PSNR），就会变成**两次归一化**：

- 结果不报错、不出 NaN、shape 与取值范围都正常，但网络吃到的图是 1/255 亮度（近乎全黑）；
- 几何全错：`dxy` 2.33mm → 19.3mm、`dz` +0.19mm → −4.14mm、z 中位 2.43m → 1.56m；
- 极端危险之处在于**推理链路完全不报警**，只有拿跨次运行的同一指标对拍才看得出来。

对策：把「喂网络」与「算 GT」拆成两个函数（`load_rgb` 返回 uint8 / `load_rgb_gt` 返回 0..1），
并在`predict`入口检查dtype/范围/归一化次数；当前max>1.5断言会误拒合法极暗图，
后续应改为显式uint8接口契约，不能把亮度阈值当通用正确性保证。

### 跨视角与深度口径（2026-11-18）

- 单图输出在source相机系：用相对T_BA渲染，或先完整变换到B系再以I渲染；不可重复施加。
  必须对这两条路径做数值对拍，A→A自渲通过不足以发现A→B重复变换。
- `GSplatRenderer`的RGB+D除alpha给期望深度，不是最近表面；双侧alpha交集不等于真实共可见。
- “相同mask”还须相同浮点阈值/精度/有效域；planeH的target选d是oracle，最近邻clamp偏差
  不能无实测定量。预测mask偏向SHARP，不能说oracle偏向planeH就使整体比较无偏。
- 各视图z中位受场景组成影响；一维尺度搜索不能界定完整Sim(3)上限。刚性/相似变换保留
  局部形状，并非把SHARP全部几何换成其它模型。


### `apply_transform` 必须传 3×4，不能加 batch 维（2026-11-17 踩坑）

`utils/gaussians.py:116` 是 `mean_vectors @ transform_linear.T + transform_offset`。
`.T` 对 **>2 维**张量是"整体倒序"而非矩阵转置（PyTorch 已对此 deprecate）。传 `[1,3,4]` 会得到
`[3,N,3]`，`len(mean_vectors)` 变 3，而 `GSplatRenderer` 正是用 `len(gaussians.mean_vectors)`
定 batch 大小（`gsplat.py` 里 `for ib in range(batch_size)`），于是 `quaternions[ib]`
`IndexError: index 1 is out of bounds for dimension 0 with size 1`。
**按 docstring 传 `3×4` 即可**，返回 batch 维保持不变。


### 本地接口核查（官方仓 aed6527，2026-11-17）

- `src/sharp/cli/predict.py`：目录逐图推理，非多视角模型；内部缩放1536²，输出相机坐标高斯，
  `predict_image`带no_grad。用完整3DGS初始化独立优化器可行；要训练SHARP网络则需另改训练链。
- `utils/io.py::load_rgb`：读EXIF焦距，缺失默认30mm；CLI无显式K参数。正式三图对照要接好
  真实或估计K，不将兜底度量尺度当GT，也不以EXIF存在保证几何精度。
- `utils/gaussians.py::apply_transform`：均值与协方差同时变换；只移动中心会使footprint/方向错误。
  `save_ply`写log-scale、opacity logit与颜色DC，导出颜色linearRGB→sRGB；`load_ply`恢复。
  用第三方3DGS训练器时，必须检查颜色合成、四元数和坐标约定，做导入/导出同视角恒等对拍。
- 自带 `render` 使用附近轨迹。视频看着清楚是定性证据，不能替代共同target、独立mask、
  相机已验证的SA/SHARP配对NVS与鼻部几何测试。
- 已有 `sharp/` 仅推理编排。多视角注册、融合、统一目标相机评测和初始化优化器尚无本轮验收；
  已装环境不等于上述改造已完成。原生推理小试成本低，融合/优化接口成本高于直接CLI调用。

### 三种接法的价值判断

| 接法 | 价值与优先级 | 三图前馈目标 |
|---|---|---|
| 单图细节/几何对照 | 先做；检测人像细节优势是否真实存在 | 可作基线，不是三图融合结果 |
| SA相机/结构 + SHARP局部高斯细节 | 条件候选；需配准、遮挡、重复覆盖处理 | 无梯度时可保持前馈，但增加三次SHARP及融合成本 |
| SHARP完整高斯初始化 + 三张实拍优化 | 有限离线诊断；必须配SA同预算优化 | 属于逐场景优化，不能记为前馈收益 |

不建议直接把SHARP点替换进SA已训练属性头/merge链。E-21已暴露外部几何与属性条件失配
的可能性；本轮先评完整SHARP高斯，避免把两个变量同时换掉。

若优化后有价值，离线教师/蒸馏只是后续研究方向，不是现成解决方案；人物LoRA仍先用可信
人物资产与真实留出监督。模型本地 `LICENSE_MODEL` 限研究用途，不能默认其权重/衍生模型
可直接进入产品；研究结论与产品迁移可行性分开记录。
