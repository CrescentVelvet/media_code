# StructSplat @ WSL2 / 单卡 3090 —— 推理复现

> 官方仓：https://github.com/DazzlingSun/StructSplat （ECCV 2026，arXiv 2606.28321）
> 论文：前馈、免位姿（pose-free）泛化式 3DGS 新视角合成，结构化解耦 geometry / semantics / texture。
> 本项目范围：**只做推理 + 渲染**，不做训练（原因见 §1）。

---

## 1. 为什么只做推理

官方训练配置是 **deepspeed ZeRO-3 + CPU offload、60k steps、batch 4、H100 集群**。
单卡 3090 24GB 显存 + 本机 PCIe 带宽，训练不可行——不是"慢一点"，是规模完全不匹配
（ZeRO-3 参数切分需要多卡才体现收益，单卡 offload 只会变成每步都在换页）。

推论：**评测数据也不用官方 DL3DV**。`train.py` / `evaluation.py` 都硬绑 DL3DV 的
annotation 格式（`{path, src[], tar[]}`），且需要 `data/` 下整套预处理产物。
本仓改走 pose-free 前向，用自己的机拍场景直接出结果——这也是论文真正声称的能力。

---

## 2. 目录与文件职责

| 文件 | 作用 |
|---|---|
| `_env.sh` | 共享环境：代理、CA、conda 激活、**CUDA/host compiler 修复**、路径变量 |
| `00a_setup_env.sh` | 建 conda env（clone `querysplat`）→ 装依赖 → 依赖自检 |
| `00b_fix_toolchain.sh` | 一次性修复 gsplat JIT 编译（gcc 别名 + CUDA 头路径），带渲染冒烟 |
| `00c_par_download.py` | 并行分块下载器（HF 单流 90KB/s → 16 并发 1MB/s 以上），断点续传 |
| `00d_test_gt_camera.py` | GT 相机转置变换的**纯几何单测**（不需权重/GPU） |
| `00e_probe_vram.py` | 逐帧数扫描显存，找 3090 上的安全上限 |
| `00f_test_image_transpose.py` | 图片预处理 + 转置语义 + GT 相机的**回归判据**（不需权重/GPU） |
| `00g_test_selfcheck.py` | `evaluate_src_views` 自检函数本身的判据（能抓错位/空画面，不过度敏感） |
| `00h_test_vggt_weight_format.py` | VGGT 必须 `model.safetensors` 的离线判据（源码分支 + 正反用例） |
| `00i_test_undistort.py` | OPENCV 去畸变链路判据（识别/不误触发/自洽性/误差量级/实图，15 项） |
| `00j_test_sim3_align.py` | Sim(3) 对齐单测（不需权重） |
| `00k_offline_align.py` | 离线对齐（用缓存相机复算，不重跑前馈） |
| `00t_axes_check.py` | 对齐残差解构：`abs`（含 gauge）/ `rel`（两两相对，gauge 无关）/ 逐轴夹角 |
| `00u_handedness.py` | 三种转置约定 × det 对比，证明问题不在约定而在「强制 proper」 |
| `00v_reflect_align.py` | **2×2 mode × reflect 扫描（决定性实验）**，见 §4.2c |
| `00w_target_eval.py` | 官方口径 target 评测（src-only Sim(3)，防泄漏），见 §4.7 |
| `00x_cam_geometry.py` | 输入包相机几何体检：基线 / 方位仰角跨度 / 共线度，判断 Sim(3) 是否可观测 |
| `00y_target_sweep.py` | 多组 target 扫描 + **覆盖率诊断**（全图 PSNR vs 覆盖区 PSNR） |
| `00z_oracle_align.py` | **oracle 上界**：对刚体修正量 `C` 随机搜索，判定低分是不是对齐误差 |
| `00aa_alpha_diag.py` | 分层 α 诊断：区分「外观错误 / 预乘稀释 / 半透明晕圈」，见 §4.7.1 |
| `01_download_models.sh` | 拉三份权重：VGGT-1B / DINOv3-ConvNeXt-L / StructSplat 主 ckpt |
| `02_run_inference.sh` | **主入口**：任意图片目录 → 前馈高斯 → 渲染 + 导出 |
| `infer.py` | 通用推理实现（官方仓缺这一环，见 §3） |
| `requirements_runtime.txt` | 运行时依赖（与官方 requirements.txt 的差异逐条注释） |

一次性探针脚本放 `_tmp_view/`（不入库）。

---

## 3. 为什么必须自写 infer.py

官方仓**没有通用推理入口**。`evaluation.py` 走 pytorch-lightning + `DataModule`，
数据侧要求 DL3DV 的 `{path, src[], tar[]}` 三元组；`model.splat()` 内部调
`get_tar_camera()`，需要「src 图 + tar 图」两组做 leakage-free 的 Sim(3) 对齐。

只有一个图片目录时，`src_number == S`、tar 为空，那条分支没有意义。
所以 `infer.py` 直接走：

```
wrapper(src)                    # 一次前向，出高斯 + 预测相机
  ├─ preds["coordinate"] (N,3)  # 世界坐标
  ├─ preds["rotation"]   (N,4)  # wxyz，已转 c2w
  ├─ preds["scale"]      (N,3)  # 已乘 scale_modifier
  ├─ preds["opacity"]    (N,)
  ├─ preds["color"]      (N,3)
  └─ preds["camera"]     (1,S,9) → pose_encoding_to_extri/intri
gsplat.rasterization(...)        # 自己渲染
```

这样是**真 pose-free**：相机与高斯同一次前向得到，不存在 src/tar 泄漏路径。

---

## 4. 关键口径（改动前必读）

### 4.1 网络输入尺寸
`[294, 518]`：`H ≤ W` 且都能被 `patch_size=14` 整除。
**竖图会被转置**成横图（`rearrange("c h w -> c w h")`），不是裁剪也不是旋转。

### 4.1b 竖图转置的坑（踩过两次，都已修）

**坑 A：转置写成空操作。** 官方 `rearrange(img, "c h w -> c w h")` 的语义是**交换最后两维**，
等价 `t.transpose(1, 2)`。`infer.py` 早期写成

```python
t = t.permute(1, 2, 0).contiguous()   # (C,H,W) -> (H,W,C)
t = t.permute(2, 0, 1).contiguous()   # (H,W,C) -> (C,H,W)  ← 绕回原样
```

两次 permute 恰好互逆，**等于完全没转**。后果：1440×1920 竖图被硬拉伸到 518×294，
水平方向压缩 **2.3 倍**，人脸/人物全部横向压扁；实测与官方实现的最大像素差 **0.973**。
更隐蔽的是它与坑 B 互相矛盾——相机按"已转置"算，图像却没转，两者对不上。

**坑 B：相机只换了内参没换外参。** 见 §4.2。修好后 `00d_test_gt_camera.py` 与真实
GT 重投影均为 **0.000 px**。

判据：`_tmp_view/transpose_bug.sh` 里两条断言——图像与官方等价实现逐像素相同、
GT 相机重投影误差为 0。**动 `load_image_tensor` 或 `load_gt_cameras` 后必须重跑。**

### 4.1c VGGT 权重格式的坑（model.pt ≠ model.safetensors）

现场报错长这样，看着像路径写错，其实**路径没错**：

```
FileNotFoundError: No such file or directory: ckpts/vggt/model.safetensors
```

真因在 `huggingface_hub`：`VGGT(nn.Module, PyTorchModelHubMixin)` 的
`_from_pretrained` 有两条分支，**本地目录分支硬编码只读 safetensors**，没有 `.pt` 兜底：

```python
if os.path.isdir(model_id):                                    # 本地目录
    model_file = os.path.join(model_id, "model.safetensors")   # ← 写死，无 fallback
    return cls._load_as_safetensor(...)
else:                                                          # hub 下载
    try:    hf_hub_download(..., "model.safetensors")
    except EntryNotFoundError:
            hf_hub_download(..., "pytorch_model.bin")          # 只有这里才 fallback
```

而 HF 上 `facebook/VGGT-1B` **只发布 `model.pt`**（pickle），没有 safetensors。
所以「下载对了、软链也对了、`ls` 也能看到 `model.pt`」，`VGGT.from_pretrained("ckpts/vggt")`
依然报文件不存在。

修法：把 `model.pt` 就地转成 `model.safetensors`（`01_download_models.sh` 已内置
`he_vggt_safetensors`，幂等自动转）。实测 5,026,874,952 B → 5,026,367,224 B，
load 13.5s + save 34.8s。

转换后验证（`chk_vggt_load.sh`）：

| 项 | 结果 |
|---|---|
| `VGGT.from_pretrained` | ✅ 12.6s |
| aggregator 参数 | 909,112,320（std 0.27093，非随机） |
| camera_head 参数 | 216,174,610（std 0.02229，含 `pose_branch` → 9 维） |
| wrapper 总参数 | 1,535,006,438 |

判据：`00h_test_vggt_weight_format.py`（纯静态 + 极小模型行为验证，不需 GPU/大权重）。
**动 VGGT 加载或 01 脚本后必须重跑。**

### 4.2 竖图转置对 GT 相机的影响（踩过坑，已修）
转置 `t[c,i,j] = img[c,j,i]` 意味着 `u' = v, v' = u`。要在转置后的坐标系里正确投影：

- 内参：`fx' = fy·(new_w/H0)`、`fy' = fx·(new_h/W0)`，`cx'/cy'` 同理互换
- 外参：新相机坐标 `(x',y',z') = (y,x,z) = M·Xc`，故 **`R' = R·M`（右乘）**，`t` 不变

⚠️ **必须右乘**。M 作用在**相机系**，不是世界系。误写成 c2w 左乘 M 会把相机整体转错方向，
实测投影误差 8805 px（比不修的 265 px 还差 33 倍）。判据见 `00d_test_gt_camera.py`：
右乘误差 0.000 px。

### 4.2b OPENCV 畸变必须处理（2026-11-19 新增，`--camera gt` 默认开启）
华为 Remy 包的 `transforms.json` 是 `camera_model: OPENCV`，带
`k1=+0.0909 k2=-0.2374 p1=-0.00021 p2=+0.00088 k3=+0.1914`。
早期 `load_gt_cameras` 只读 `fx/fy/cx/cy`，等于把畸变当 0。实测误差：

| 尺度 | 中位 | P95 | 最大 |
|---|---|---|---|
| 原图 1440×1920 | 7.06 px | 10.68 px | 18.29 px |
| 缩到 518 宽 | 2.54 px | 3.84 px | 6.58 px |

这个量级会让「渲染输入视角」的 PSNR 自检把**畸变残差**误读成**相机估计不准**，必须先消除。

做法：`probe_distortion()` 读畸变 → `cv2.undistort` 重采样图像（`alpha=0`，本例
`roi=(0,0,1439,1919)` 几乎无损，边框仅 16 个黑像素）→ **内参换成 `newK`**
（`cv2.getOptimalNewCameraMatrix`，本例 `fx ×1.0157`、`cx +1.58px`）。
去畸变是重采样，有效焦距与主点都会变，**图像与内参必须同时换**，只换图会用旧 K 整体错位。

两个易错点：
1. `dist` 是 OpenCV 顺序 `(k1,k2,p1,p2,k3)`，而 JSON 是命名字段 `k1/k2/k3/p1/p2`，
   **直接拼会错位**（k3 会当 p1 用）。
2. 必须用 uint8 走 cv2（像素单位重采样），先归一到 `[0,1]` 再 `undistort` 要来回转两次，
   且后续 `interpolate` 的边缘外推行为会变。

判据：`python 00i_test_undistort.py`（15 项，含「全零/无字段必须不触发」的反向用例）。
自洽性判据是 `newK·Xc_norm == undistortPoints(K0·Xc_norm, P=newK)`，实测残差 4e-05 px。
关掉用 `--no-undistort`（对照实验用）。

`pose-free` 分支拿不到变换文件，无从得知畸变 → **只在 `--camera gt` 且给了
`--transforms` 时生效**。这也意味着 pose-free 分支对这组数据天然带 ~2.5px 中位残差，
是它的固有下限，不是实现 bug。

### 4.2c GT→模型系对齐：手性破缺与反射（2026-11-20，最反直觉的一条）

`--camera gt` 分支要把 GT 相机搬进**模型坐标系**（高斯的坐标系由前馈预测的外参定义，
见 `utils_3d.py:34` 的 `depth_to_world_coords_points`，被 `gaussian_wrapper.py:586` 调用）。
搬法是一个相似变换 `Sim(3) = (s, R, t)`，用 `align_gt_to_model()` 拟合。

**症状**：拟合完朝向残差恒为 `90.000°`，PSNR 只有 6~7 dB，怎么调都下不来。

**根因**：`pred_w2c` 的 `det = +1`（正常旋转），`gt_w2c` 的 `det = −1`（反射）。
两者**手性不一致**。而 Kabsch 若强制 `det(R) = +1`（`R = V·diag(1,1,det(VUᵀ))·Uᵀ`），
反射无法被任何 proper rotation 抵消——最优解永远在可行域的边界上，残差恰好卡在 90°
（`trace(M) = 1` 的旋转正好是 90°），而且残差是**常数**、与数据无关，这正是它可疑的地方。

手性为什么会破？**竖图转置本身就是一次 2D 镜像**（`u' = v, v' = u`）。
它在相机系上诱导的基变换

```
M = [[0,1,0],
     [1,0,0],
     [0,0,1]]        det(M) = −1
```

`det = −1` 是**数学必然，不是 bug**。模型吃的是镜像图，它重建出的场景相对真实场景
就是镜像的；模型系与 GT 系之间的真实变换是「相似变换 + 一次反射」，不是纯 `Sim(3)`。

**修法**：不要强制 `det = +1`，让 SVD 自己选。

```python
R_raw = Vt.T @ U.T
if allow_reflect:          # 手性不一致时（reflect="auto" 会自动判定）
    R = R_raw              # 反射关系 → det(R) = −1
else:
    R = R_raw @ diag(1, 1, sign(det(R_raw)))
```

关键一步在**应用**时：`R_model = R_gt · Rᵀ`，`(−1)×(−1) = +1` →
**对齐后 GT 相机变回 proper rotation**，正是 gsplat 需要的。反射被"用掉"在相似变换里，
不残留到相机上。

实测（4 视角 GT 分支）：

| mode | reflect | fit_resid | 朝向残差 | 中心 RMS | PSNR |
|---|---|---|---|---|---|
| center | 强制 +1 | 1.0056 | 90.00° | 0.858 | 10.67 |
| axes | 强制 +1 | 1.0476 | 90.00° | 0.858 | 10.65 |
| center | **自选** | 0.0130 | 1.51° | 0.013 | 16.06 |
| **axes** | **自选** | **0.0179** | **1.31°** | **0.014** | **16.47** |

`fit_resid` 降 **58 倍**，端到端 **10.65 → 16.47 dB（+5.8 dB）**。
`reflect` 是决定因素；`axes` 比 `center` 再高 0.4 dB。

**为什么 `axes` 优于 `center`**：`center` 只拟合相机中心，在 4 台近共线相机上
「绕连线的旋转」欠定（中心不变但朝向可任意）；`axes` 把相机三根基向量一起拟合，
朝向被钉住。用 `--align-mode {axes,center}` 切换，默认 `axes`。

**为什么不该换约定来"消除"手性**：`00u_handedness.py` 试了三种（现状 transpose /
rot90 误乘 A / rot90 正确乘 Aᵀ），都闭合不干净（90.0° / 38.1° / 88.1°）。
问题不在「用哪个约定」，而在「强制 proper」这个约束本身。

判据脚本：`00t_axes_check.py`（残差解构：abs / rel / 逐轴）、`00u_handedness.py`（三约定）、
`00v_reflect_align.py`（2×2 mode × reflect，**决定性**）。
CLI：`--align-mode axes --align-reflect auto`（默认；`off` 可复现历史 90° 现象）。

⚠️ **target 视角必须复用「仅在 src 上估计」的同一个 `(s,R,t)`**。若把 target 也放进拟合，
就是 GT 泄漏（`align_gt_to_model` 的返回值带了 `_R` / `_t` 供此用途，`00w`/`00y`/`00z` 都遵守）。

### 4.2d 官方设计输入是 2 张图（不是 24 张）

容易误读：`config/dl3dv.yaml` 有 `max_images` 之类，但评测 annotation 是 `src=2, tar=2`，
训练 `min_src_number=2`。StructSplat 是**少量视角**前馈模型，喂 24 张不是它的设计工况。

实测（本机 3090）：

| src 数 | 前馈耗时 | 高斯数 | 自洽 PSNR |
|---|---|---|---|
| 2 | 2.7 s | 2,842,784 | **22.46 dB** |
| 4 | 93.7 s | 5,685,568 | 20.47 dB |

2 图既快又准，与官方设计一致。**多喂图反而变差**（跨视角几何冲突被平均进来）。
显存上 4 图已接近红线（WSL2 统一内存会让分配器峰值超物理显存，只有 4 帧真正落在 24GiB 内，
`--max-images` 在 3090 上的实测安全上限就是 4）。

### 4.3 渲染分辨率
`--render-size` 指**宽**（长边）。高度按 `new_h/new_w` 比例推，K 同步按 `render_size/new_w` 缩放。
早期实现把画布设成正方形，输入 294×518 时右侧约 43% 画面被裁掉——已修。

### 4.4 不用的东西
- `model.splat()`：依赖 `get_tar_camera()`，单目录无意义（§3）
- `deepspeed`：只为了 `texture_encoder.py` 的顶层 import 能过；`DS_BUILD_OPS=0` 走纯 Python
- torch 重装：沿用 `querysplat` 的 torch 2.8.0+cu128 + gsplat 1.5.3，其 CUDA 扩展
  已编译（缓存键 `py311_cu128`），重装会丢缓存

### 4.5 导出格式的两个易错点（已核）

**scale 是线性的，不是 log。** `activations` 第四项 `softplus_beta-log2` 只是把 softplus 的
`beta` 设成 `log(2)`（`model/utils.py` 的 `ACTIVATION_DICT`），输出仍是**正值线性尺度**；
`scale_modifier: 0.005` 已经在 `output2gaussian` 内部乘上去了（见 `gaussian_wrapper.py` 里
`scale = rearrange(...) * scale_modifier`）。所以导 PLY 时取一次 `log` 是对的，
**不要再额外乘 `scale_modifier`，也不要对 `preds["scale"]` 再取一次 log。**

**color 是原始 RGB。** 第三项激活 `mul_0p25` 即 `color = raw * 0.25`，**无界、无 sigmoid**；
官方 `_render` 把 `colors` 原样喂 `rasterization` 且**不传 `sh_degree`**。
`infer.py` 一致。若给 `rasterization` 传 `sh_degree=0`，会触发
`assert (sh_degree+1)**2 <= colors.shape[-2]` 并要求 `(N,1,3)` 布局——别传。

### 4.6 自检哨兵：跑通 ≠ 跑对

`splat()` 那套会打印 PSNR，但我们没走它。所以 `infer.py` 在渲染后加了一道自检：
**用预测相机重渲染输入视角，与输入图算 PSNR/SSIM**。

这能一次性抓住五类翻车：全黑/全白、相机整体错位、竖图转置方向错、内外参缩放不同源、
K 行列写反。判读：

| PSNR 均值 | 含义 |
|---|---|
| ≥ 18 dB | ✅ 链路自洽（相机/反投影/渲染三者一致） |
| 10~18 dB | ⚠️ 偏低，可能输入视角太离散，或场景本身难 |
| < 10 dB | ⚠️ 画面对不上，优先查相机而不是查权重 |

⚠️ 这不是论文评测口径（论文报 held-out target 视角），只是哨兵，**不要拿去和论文数字比**。
结果同时写进 `meta.json` 的 `self_check` 字段。

自检函数本身由 `00g_test_selfcheck.py` 把关（无需 GPU/权重）：完全相同→inf、
平移/反色/全黑/转 90°→均 <15dB、0.02 噪声→PSNR 34dB 且 SSIM 0.95。

### 4.7 官方口径的 target 评测（以及为什么数字不能直接对外比）

自检只能证明**链路自洽**：输入视角本来就是高斯反投影的来源，重渲自己天然高分。
官方评测（annotation `src=2 / tar=2`）是**喂 2 张重建，另取数据集里别的帧渲染**。
只有这个数字能对外报。脚本：`00w_target_eval.py`（单组）、`00y_target_sweep.py`（多组 + 覆盖诊断）。

**硬纪律两条**（本项目都踩过）：
1. **`Sim(3)` 只用 src 的 ≥2 台相机估计，绝不碰 target**。单台 src 会退化
   （无基线 → 尺度不可观测、旋转只剩朝向），实测 gt 分支塌到 8.92 dB。
   代码里 `if len(src_paths) < 2: raise SystemExit`。
2. **target 图不能来自高斯缓存**，否则高斯已含该视角信息 = 泄漏。
   target 路径单独给（`--tar-images`），代码里 `if _s & _t: raise SystemExit`。

**实测（佳佳抱枕，82 帧，2 图 src = 首尾两张）**：

先体检 src 是否真的可观测（`00x_cam_geometry.py`）：基线 2.19 = 相机半径的 **140%**、
方位角差 39.5°、俯仰差 82.4°、共线度 0.000 → **不退化**，`Sim(3)` 可解。

| target 组 | 帧号 | 覆盖率 | PSNR 全图 | PSNR 覆盖区 | SSIM |
|---|---|---|---|---|---|
| 紧邻 src0 | 1,2 | 100% | **17.38** | 17.38 | 0.393 |
| 紧邻 src81 | 79,80 | 95.0% | 16.12 | 16.40 | 0.339 |
| 中段 | 10,30 | 64.1% | 12.52 | 15.35 | 0.276 |
| 大角度 | 21,42 | 63.3% | **11.07** | 14.22 | 0.269 |
| 大角度 2 | 41,61 | 75.9% | 11.93 | 14.94 | 0.302 |

**必须先看覆盖率，再看 PSNR**。2 视角重建出的高斯只覆盖两锥相交处，大角度新视角
大片空白；空白区的 PSNR 就是「黑图 vs 真值」，会把均值拖到 11 dB，与几何准不准无关。
`00y` 把 `coverage = mean(alpha > 0.5)` 与「只在被覆盖像素上算的 PSNR」分开报。

**归因已经查到底（`00z_oracle_align.py`，oracle 上界搜索）**：
在 src-only `Sim(3)` 结果上左乘世界系刚体修正 `C`，对 `C` 做粗→细随机搜索（600 次，5 档），
最大化 target PSNR —— 这正好覆盖 `Sim(3)` 的平移/旋转误差。结果：

```
正式口径 (C=I)     11.07 dB
oracle 上界        11.26 dB     差 +0.19 dB   → 对齐已到底
最优修正量          旋转 3.58° / 平移 3.6% 半径
```

`+0.19 dB` 说明**对齐/位姿不是瓶颈**，`src-only Sim(3)` 已接近该数据下的最优。

**瓶颈是覆盖 + 前馈几何外推**。同批 target 用 4 图 src 对照（`far` 组 21,42）：

| src | 覆盖率 | PSNR 全图 | PSNR 覆盖区 |
|---|---|---|---|
| 2 图 | 63.3% | 11.07 | **14.22** |
| 4 图 | **94.0%** | 13.33 | 13.80 |

覆盖率 `+30.7 pt`（63→94），PSNR 只 `+2.26 dB`，且**覆盖区内反而略降**（14.22→13.80）。
即：覆盖不足解释了约 2.3 dB，**剩下的瓶颈是前馈几何在大角度新视角上的精度**，
不是视角数量。多喂图能补覆盖，补不了精度。

⚠️ **口径边界（对外必须说清）**：`StructSplat` 前馈**只预测输入视角的相机**，不预测未见视角。
所以 target 相机只能来自 `transforms.json`（GT），再用 src-only `Sim(3)` 搬进模型系。
→ 这不是「纯 pose-free 新视角」，而是「**模型几何 + GT target 相机**」，
衡量的是**重建几何精度**，不是位姿估计能力。位姿估计能力由 `infer.py` 的自洽
PSNR（输入视角重渲，2 图 22.46 dB）体现。两者不可混谈。

**去畸变在 target 侧几乎无影响**（11.07 vs 关掉 11.06），但会略微压低 src 自洽
（21.64 vs 关掉 23.64）——`cv2.undistort` 的重采样本身有损。几何结论不受影响，
但它解释了「为什么关掉畸变自洽分数更高」这个反直觉现象，别误判成「畸变处理是错的」。

#### 4.7.1 「偏暗/过平滑」不是外观错误（`00aa_alpha_diag.py`）

`00y` 存下的 PNG 里，pred 的梯度只有 GT 的 **30%**、全图线性校正能提 **+2.69 dB**，
很容易误读成「渲染整体偏暗」。逐层拆开后**结论相反**，这里记下来防止重复误判。

**(1) 空白区不是黑的，是低 alpha 晕圈**。`far_021` 亮度直方图：`[0.02,0.05)` 占 **25.1%**，
纯黑 `<1e-6` 只有 **0.00%**。所以「coverage 60%、非黑像素 100%」可以同时成立——
`coverage` 的口径是 `alpha>0.5`，和「非黑像素比」不是一回事。

**(2) alpha 在饱和区没有稀释**：核心 `α>0.95` 区实测 α 均值 **0.992**（far）/ 0.994（adj、src）；
覆盖区 `α>0.5` 均值被拉低到 0.924 / 0.936。半透明晕圈占比 **21.0% / 20.3%（far）**
vs 2.3% / 3.8%（adj）。

**(3) 色调本身是对的——但掩码必须选对**。这条是本节最值钱的经验：

| 掩码 | a（斜率） | b（偏移） | 误读风险 |
|---|---|---|---|
| `α>0.5`（含 21% 黑晕） | 0.93 | +0.133 | ❌ 会误判成「曝光偏低」 |
| **`α>0.95`（核心）** | **1.00** | +0.086 | ✅ 色调正常，只有轻微压暗 |

用 `α>0.5` 掩码做回归时，半透明黑晕会把斜率系统性拉低（一度算出 a=0.6~0.75）。
**换核心掩码后 a 精确等于 1**——所谓「偏暗」是测量假象，不是渲染缺陷。

**(4) 反预乘的收益只出现在有晕圈的视角**，与晕圈占比强相关：

| 组 | 半透明占比 | 反预乘 ΔPSNR | 核心区 ΔPSNR |
|---|---|---|---|
| far:21,42 | 20~21% | **+2.18 / +2.40** | +0.10 / +0.14 |
| adj:1,2 | 2.3~3.8% | −0.03 / −0.05 | −0.03 / −0.02 |

核心区几乎不变（15.08→15.18）⇒ **饱和区内部颜色已经正确**，收益全来自边缘修补。

**(5) 还有一层与本视角无关的分辨率上限**。紧邻 src 的 `adj` 组（100% 覆盖、几乎无晕圈）
梯度仍只有 GT 的 **63.4~63.6%**，PSNR 17.3~17.5 dB；src 自洽也只有 21.6 dB。
⇒ 平滑里有一层来自「518 宽输入 + ConvNeXt 语义特征」的上限，
**调对齐、调位姿都拿不到**，要往上推只能改输入分辨率或换权重。

**修正后的完整归因**（替换前面「剩下是几何外推」的单因结论）：

```
11.1 dB  =  覆盖不足(约 −2.3dB, 4图对照已证)
          × 半透明晕圈(约 −2.2dB, 反预乘可量化)
          × 几何外推(oracle 仅 +0.19dB → 对齐不是瓶颈)
          × 分辨率上限(src 自洽 21.6dB 即天花板)
```

**没有一条是「Sim(3) 没调好」**。要提升 target，优先级是：换更均匀的 src 对/加图补覆盖
> 改输入分辨率 > 稠密化后期处理；调对齐参数已经没收益。


---

## 5. 本机环境坑（都已固化进脚本）

### 5.1 gsplat 编译失败的真根因：没有系统 gcc
现象：30 个 `.cu` 全失败，被 `warning #20012-D: __device__ annotation is ignored` 淹没，
报 `gcc: No such file or directory` / `nvcc fatal: Failed to preprocess host compiler properties`。

**本机 WSL 根本没有 `/usr/bin/gcc`**（`gcc`/`g++`/`cc`/`c++` 全无，也无 `gcc-N`）。
conda-forge 的 `gcc_linux-64 12.4.0` 存在，但可执行文件叫 `x86_64-conda-linux-gnu-gcc`，
**没建 `gcc` 别名**，nvcc 找不到。

修法（`_env.sh` 第 4 节，幂等）：
1. `$CONDA_PREFIX/bin` 下补 `gcc/g++/cc/c++` 四个软链 → nvcc 自动找到
2. 显式 `CC`/`CXX`/`CUDACXX` + `NVCC_PREPEND_FLAGS=-ccbin`
3. `CUDA_HOME=$CONDA_PREFIX`、`CPATH` 指向 `$CONDA_PREFIX/targets/x86_64-linux/include`
   （conda 的 CUDA 头不在 `$CONDA_PREFIX/include`）
4. `TORCH_CUDA_ARCH_LIST=8.6`（3090 = sm_86；不设会用全架构列表，编译时间暴涨）

修完：`gsplat: CUDA extension has been set up successfully in 197.56 seconds`。

**验证扩展真的加载了**（别查 `gsplat.cuda._C`，那个属性不存在，会误判成走了 Python 回退）：
```python
from gsplat.cuda._backend import _C
# → <module 'gsplat_cuda' from '~/.cache/torch_extensions/py311_cu128/gsplat_cuda/gsplat_cuda.so'>
```
`.so` 落在 torch 的 JIT 缓存里（`py311_cu128`），**不在 site-packages**。
所以别用 `glob site-packages/**/*gsplat*.so` 找，找不到很正常。

### 5.2 vggt 要 pip install
`third_party/vggt` 只是源码子模块，不装则 `VGGT.from_pretrained` 报
`ModuleNotFoundError: No module named 'vggt'`。必须：
```bash
cd $STRUCTSPLAT_REPO/third_party/vggt && pip install -e . --no-deps -q
```
**务必加 `--no-deps`**，否则会把 numpy / Pillow 降级，破坏已装好的环境。

### 5.3 HF 下载通道（实测数据）
| 通道 | 速度 | 结论 |
|---|---|---|
| HF 直连 | `curl(35) Connection reset` | ❌ 完全不通 |
| HF 走代理（单流） | ~90 KB/s | 5GB 要 18 小时 |
| HF 走代理（16 并发分块） | **1.0~1.1 MB/s** | ✅ 可用，见 `00c_par_download.py` |
| ModelScope 直连 | **2.0~2.2 MB/s** | ✅ 首选（但无 StructSplat ckpt） |
| hf-mirror.com 直连 | 1 KB/s | ❌ |

ModelScope 有 `facebook/VGGT-1B` 全集（config.json + model.pt 5026874952 B），
**没有** `DazzlingSun/structsplat`（API 返回 `record not found`）。
所以：VGGT 走 ModelScope，DINOv3 用本机已有文件，StructSplat ckpt 走 HF 多并发。

### 5.4 权重池约定
先搜 `/mnt/d/wheel`，新下载落 `D:\wheel\structsplat_ms/`，消费侧一律**软链不复制**。
```
/mnt/d/wheel/structsplat_ms/
  ├── VGGT-1B/                    → ckpts/vggt
  ├── dinov3_convnext_large/      → ckpts/dinov3_convnext_large
  └── structsplat/pytorch_model.bin → ckpts/structsplat/
```
`config/dl3dv.yaml` 里全是**相对路径** `ckpts/xxx`，所以软链必须落在仓内 `ckpts/` 下。

---

## 6. 使用

```bash
# 0. 环境（首次）
bash structsplat/00a_setup_env.sh
bash structsplat/00b_fix_toolchain.sh      # gsplat 编译，约 200s

# 1. 权重
bash structsplat/01_download_models.sh all

# 2. 推理（pose-free，默认黑底，导出 ply + splat）
bash structsplat/02_run_inference.sh \
  "/mnt/d/dataset/手部微动样例/01e954de1b8a4cde9315c976f4dcb964-佳佳抱枕/image" \
  hand01 512 pose-free

# 3. GT 相机对照（把相机估计误差从渲染误差里分离）
bash structsplat/02_run_inference.sh \
  "/mnt/d/dataset/手部微动样例/01e954de1b8a4cde9315c976f4dcb964-佳佳抱枕/image" \
  hand01_gt 512 gt \
  --transforms "/mnt/d/dataset/手部微动样例/01e954de1b8a4cde9315c976f4dcb964-佳佳抱枕/transforms.json"

# 4. 显存探测（先定 max-images 再跑全量）
python structsplat/00e_probe_vram.py --images <目录> --frames 4,8,16,24,32

# 5. 官方口径：2 图重建 → held-out target（§4.7）
#    5a 先 dump 前馈缓存（--max-images 2 = 官方设计输入）
python structsplat/infer.py --images "<包>/image" --out "$RESULTS_DIR/hand01_src21" \
  --config "$STRUCTSPLAT_REPO/config/dl3dv.yaml" --render-size 512 --max-images 2 \
  --dump-forward "$RESULTS_DIR/_cache/hand01_src21.pt"

#    5b 相机几何体检（确认 src 基线可观测，否则 Sim(3) 退化）
python structsplat/00x_cam_geometry.py --transforms "<包>/transforms.json"

#    5c target 评测（--tar-images 必须与 src 不重叠，代码里会拦）
python structsplat/00w_target_eval.py \
  --cache "$RESULTS_DIR/_cache/hand01_src21.pt" \
  --transforms "<包>/transforms.json" \
  --tar-images "<包>/image/5093313540985078.heic,<包>/image/5093317807498922.heic"

#    5d 覆盖诊断（全图 PSNR vs 覆盖区 PSNR，务必分开看）
python structsplat/00y_target_sweep.py \
  --cache "$RESULTS_DIR/_cache/hand01_src21.pt" \
  --transforms "<包>/transforms.json" --images "<包>/image" \
  --groups "adj:1,2" "mid:10,30" "far:21,42"

#    5e oracle 上界（判定低分是不是对齐误差；差 <0.8dB 即对齐已到底）
python structsplat/00z_oracle_align.py \
  --cache "$RESULTS_DIR/_cache/hand01_src21.pt" \
  --transforms "<包>/transforms.json" --images "<包>/image" \
  --tar-idx 21,42 --iters 600

#    5f 分层 α 诊断（区分外观错/预乘稀释/晕圈；§4.7.1 的全部数字）
python structsplat/00aa_alpha_diag.py \
  --cache "$RESULTS_DIR/_cache/hand01_src21.pt" \
  --transforms "<包>/transforms.json" \
  --groups "far:21,42" "adj:1,2"
# 判读：核心掩码 α>0.95 下 a≈1 → 色调没问题；反预乘 ΔPSNR 与晕圈占比强相关
```

产物（`$RESULTS_DIR/<name>/`）：`renders/`、`depths/`、`compare.png`、
`gaussians.ply`、`gaussians.splat`、`meta.json`。

---

## 7. 数据约定

输入是华为「Remy」采集包：
```
<包>/
  ├── image/*.heic        # 82 张，1440×1920，单变焦
  ├── transforms.json     # OPENCV，c2w（transform_matrix），det(R)=+1
  └── pcd.ply             # 79355 点，Remy 导出
```
`.heic` 需要 `pillow-heif` + `pillow_heif.register_heif_opener()`（`infer.py` 已处理）。

`transforms.json` 的 `transform_matrix` 是 **c2w**（已验证 det=1、正交性 0 误差），
转 w2c 要取逆。`fl_x`/`fl_y` 在全 82 帧上完全一致（单一变焦，无 per-frame 内参）。
