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
