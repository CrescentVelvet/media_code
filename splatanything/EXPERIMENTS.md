# EXPERIMENTS — SplatAnything（InfiniSplat v3）复现实验记录

> 按日期倒序追加。机器：WSL2 / i9-12900K / RTX 3090 24 GB / 驱动 595.95。
> 数值全部为本仓实测。**与论文数字不可直接对比**（官方评测划分未公开，发布包无训练代码）。
> 截图对比：`eval/contact_sheet.png`（GT / render / |diff|×4 三列）。

---

## 2026-11-12 · 复现启动：官方包盘点 + 链路打通（E-01~E-03）

### E-01 官方发布包的实际内容（与 README 的差异）

clone `zju3dv/SplatAnything` HEAD = `b518996`，共 483 文件 / **417.7 MB**，
其中 `examples/` 占绝大部分（pano 24 张各 ~5 MB、rig 150 张）。用
`--filter=blob:none --sparse` 只取代码后为 ~1 MB。

盘点结论（会直接影响复现方式）：

| 项 | 官方说法 | 实测 |
| --- | --- | --- |
| 训练代码 | 未提供 | ✅ 确认无（`rasterization.py` 头部注释 "no rasterization or training"） |
| 光栅化器 | 无 gsplat 依赖 | ✅ 确认无，渲染必须自接 |
| `validation.py` | docs 称 "remains available" | ❌ **文件不存在**，`tests/test_inference.py:17` 必然 ImportError |
| `examples/object/prior_camera.json` | — | ⚠️ `num_cameras=49`，但 extrinsics/intrinsics 各 24 条、images 23 张（loader 不读该字段，不影响运行） |
| 权重总量 | 3 个文件 | 16.6 GB：`SplatAnything.ckpt` 11.33 GB + `worldmirror2.safetensors` 5.05 GB + `skyseg.onnx` 0.18 GB |

**下载速度的关键发现**（这条最费时间）：镜像站对 `resolve` 请求回 302，真实数据在
`us.aws.cdn.hf.co`（aifasthub）/ `cas-bridge.xethub.hf.co`（hf-mirror）。只把
`aifasthub.com` 写进 `no_proxy` 不够 —— CDN 域名仍被 v2rayN 接管，实测 **57 KB/s**；
把 `.hf.co,.xethub.hf.co` 一并排除、并给 curl 加 `--noproxy '*'` 后 **1.7~2.4 MB/s**（30×）。

单次下载观测（8 并发、96 MB 分块、SHA256 全程校验）：

| 文件 | 大小 | 结果 |
| --- | --- | --- |
| skyseg.onnx | 176 MB | ✅ SHA256 通过（`ab9c34c6…`） |
| worldmirror2.safetensors | 5.05 GB | ✅ 5,053,556,568 字节，与 API 一致 |
| SplatAnything.ckpt | 11.33 GB | 下载中（详见下方"权重"小节） |

### E-02 CPU 全链路冒烟：不开权重也能定位接口错误（`04_smoke_cpu.py`）

动机：16.6 GB 权重下载要 1 小时以上，但 config 白名单、相机 JSON 校验、GSN、
体素合并、PLY 导出这些逻辑**不需要真权重**就能验。用随机初始化的两个模型跑
2 张 56×56 图、pose 模式，`load_model_weights` 打桩跳过。

结果（CPU，1.02 s）：

```
[Input] scene: 2 images, resolution=[56, 56], enable_cond=True, use_gt_camera=True
[Inference] 1.02s
  [gaussians] saved 2245 Gaussians
  [points] saved 6272 points
```

- `points.ply` = 2 × 56 × 56 = **6272**，与像素数逐一对应 → 深度反投影与像素网格对齐；
- `gaussians.ply` = **2245** → 体素合并压缩到 35.8%（voxel 0.002 vs 实例尺度 0.0013 量级）；
- 产物 `gaussians.ply / points.ply / cameras.json / config.yaml / run.json` 齐全。

判据：本脚本通过 = 参数组合、路径解析、导出格式都没问题，真权重到了只需换
`--ckpt_path`。它不能证明画质，也不加载权重。

### E-03 华为 Remy 采集包 → v3 场景目录（`07_run_capture.sh`）

采集包 `01e954de…-佳佳抱枕`（82 帧 HEIC + transforms.json），等间隔抽 24 帧：

```
24/82 帧  K[0] fx=1380.8 cx=717.7 cy=952.9
去畸变: [1440, 1920] → [1439, 1919] (roi=[0, 0, 1439, 1919])
相机半径 0.107~2.492 m (mean 1.435)
视角两夹角 min 1.1° / median 70.0° / max 179.0°
Valid: 24 images have OpenCV c2w poses and pixel intrinsics.   ← 官方校验器
```

- 去畸变几乎不裁（1440×1920 → 1439×1919），说明这批数据的桶形畸变在 alpha=0 下
  边角损失 < 1 px；v2 时代同一批数据也是这个量级。
- **相机半径跨度 0.107~2.492 m** 是采集包坐标系的特点（原点在 AR 会话起点，不在场景
  中心），不是退化；v2 的结论是"主体居中"只影响下游把角度当绕世界原点用的脚本，
  v3 有 `cam_align`（锚到首视角）+ GSN，不受影响【待用真权重验证】。
- 夹角 median 70°、min 1.1°（首末帧近重复，等间隔抽帧含首末导致），无退化。

转换脚本对 v2 的三条约定（OpenGL→OpenCV 翻转 `M @ diag(1,-1,-1,1)`、内参按原图像素、
`getOptimalNewCameraMatrix(alpha=0)` 去畸变）逐条沿用，见
`07_capture_to_scene.py` 顶部注释。

---

## 2026-11-12 · 真权重三连跑：官方示例两模式 + 自有采集包（E-04~E-06）

权重 SHA256 全部通过后，跑了三组。都是单卡 3090 24 GB、`PROFILE=1` 采显存峰值。

| # | 场景 | 视角 | 分辨率 | 模式 | 推理 | 总耗时 | 峰值显存 | points.ply | gaussians.ply | 合并率 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E-04 | `examples/object` | 23 | 756² | rgb | 68.30 s | 113 s | **23.53 GB** | 13,119,037 | 2,329,941 | 17.8% |
| E-05 | `examples/object` | 23 | 756² | pose | 67.30 s | 117 s | **23.59 GB** | 13,119,037 | 2,462,487 | 18.8% |
| E-06 | `jiajia_hold`（Remy 采集包） | 24 | 504² | pose | 7.28 s | 44 s | **21.79 GB** | 6,084,191 | 1,942,508 | 31.9% |

"合并率" = gaussians / points。总耗时含 16.6 GB 权重的 mmap + `load_state_dict`
（两次模型共 1322 + 1390 个 tensor，逐个 shape 校验）。

### E-04/E-05 显存是硬边界：756² × 23 视角已经贴顶

23 视角 @756 峰值 **23.53 / 23.59 GB**（rgb / pose），24 GB 卡只剩 ~0.4 GB 余量。
这不是"还能再塞几个视角"的余量，跑第二个进程必炸。推算：

- 756² → 每视角 54×54 = 2916 token；23 视角 = 67k token；
- 504² → 1296 token/视角；24 视角 = 31k token，但峰值仍有 21.79 GB ——
  **固定开销（两个 fp32 模型的权重 + DPT 解码缓冲）占了绝大部分**，token 数只是二阶项。

⚠️ 直接的后果是**速度断崖**，和 v2 记录的"撞顶静默变慢 10 倍"是同一类现象：

| 配置 | token 数 | 推理耗时 | 单位 token 耗时 |
| --- | --- | --- | --- |
| 23v @756 | 67.1k | 68.30 s | 1.02 ms/token |
| 24v @504 | 31.1k | 7.28 s | 0.23 ms/token |

面积只差 2.25×，耗时差 **9.4×**。判断依据（下次复现时先看这条）：**耗时/面积比超过
5× 就是撞顶了**，不是模型本身慢。想跑 24+ 视角请直接用 504²，或分批跑再合并。

### E-06 GSN 确实把两个不同尺度的场景压到同一量级

这是 v3 相对 v2 最实质的改动，所以单独验：读导出的 `gaussians.ply` 统计
`scale` 中位数，与代码常量 `spacing_target = 1.3228e-3` 比（脚本 `05_check_scale.py`）。

| 场景 | 相机到原点 p50 | 高斯 scale 中位数（最大轴） | scale / spacing_target | 1-NN 间距 p50 |
| --- | --- | --- | --- | --- |
| `examples/object`（rgb） | 1.746 | 0.001064 | **0.804** | 0.00335 |
| `examples/object`（pose） | 1.746 | 0.001091 | **0.825** | 0.00325 |
| `jiajia_hold`（人像采集） | 0.452 | 0.001805 | **1.365** | 0.00326 |

三条读数各自说明一件事：

1. **两个场景的原始尺度差 3.9×（1.746 vs 0.452），但 scale/spacing 都落在 0.8~1.4**
   —— GSN 把不同场景归一到同一数值区间，网络学的确实只是"相对 spacing 的比例"。
2. `voxel_size=0.002` 的体素对角线是 0.00346，三组的 1-NN p50 都是 **0.0032~0.0034**
   —— 合并网格确实以 0.002 起作用（同格只留一个，1-NN 落在对角线附近而非 0.002，
   因为点云是表面薄壳、相邻占用格多为对角相邻）。
3. 人像组的 scale 偏大（1.365 vs 0.8）：人像视角间距 70°、重叠少，同一表面被采样
   次数少 → 体素内分担的点更少 → 合并后 scale 更大。这是**数据特性**，不是模型差异。

### 产物与输出格式（实测）

`02` 一次产出（`$RESULTS_DIR/<MODE>/<scene>/`）：

| 文件 | 大小（E-04） | 内容 |
| --- | --- | --- |
| `gaussians.ply` | 158.4 MB | 标准 3DGS PLY，SH DC / log scale / logit opacity / wxyz |
| `points.ply` | 196.8 MB | 合并前逐像素中心 + RGB（过滤后） |
| `cameras.json` | 小 | 23 条 OpenCV c2w（**GSN 缩放后**）+ 处理分辨率内参 |
| `depth/*.npy,*.png` | — | 逐视角深度（**GSN 之前**的预测尺度，见 NOTES 2.3） |
| `rgb/*.png` | — | 实际喂给模型的裁剪图（756×756） |
| `config.yaml`, `run.json` | 小 | 生效配置与 CLI/输入记录 |

`points.ply` 的 13,119,037 = 23 × 756 × 756 − 26,291：差的 0.2% 被
`max_scale_pct=99.8` 的尺度过滤丢掉（第一个过滤在合并之前）。

---

## 待补

- 真权重下的官方示例（`examples/object` 23 张 @756）冒烟：耗时 / 显存 / 高斯数 / 自洽 PSNR
- 显存与输入规模的关系（756² 下 24 视角能不能压进 24 GB）
- 采集包 24 视角 @756 的实测（pose 模式）
- GSN 的 spacing 分布是否符合"归一到 0.0013227513"的预期
