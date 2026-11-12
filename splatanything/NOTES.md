# SplatAnything（InfiniSplat v3）原理与排错

本文件只写**读代码能确认的机制**和踩过的坑；运行命令见 [README_wsl.md](README_wsl.md)，
实测数字见 EXPERIMENTS.md。凡是没跑过的推断都标了【推测】。

官方仓：<https://github.com/zju3dv/SplatAnything>（commit `b518996`，2026-11-12 clone）
权重：<https://huggingface.co/ritianyu/SplatAnything>（apache-2.0，非 gated）

---

## 1. 推理链路：两个模型串行，学生不吃教师梯度

```
images (+ 可选 prior_camera.json)
        │
        ├─► GeometryWorldMirror   (worldmirror2.safetensors)   "teacher"
        │      ViT-L/14 (embed 1024 × 24 层)
        │      ├ CamHead   → camera_params[9] → c2w + K      [inference.py:146-150]
        │      └ DPTHead   → depth, depth_conf, mask_logits   output_dim=3
        │
        └─► GaussianWorldMirror   (SplatAnything.ckpt)          "student"
               同一个 encoder（独立权重）
               ├ GSFeatHead → 128ch 特征 × N 视角
               └ DPTDecoder → 每像素 12 维 = quat4 + scale3 + opacity1 + SH_residual3 + weight1
                              means 直接取 depth 反投影（不学位置）
```

关键事实（都有代码依据）：

| 项 | 值 | 出处 |
| --- | --- | --- |
| 每像素高斯原始输出 | 12 维 `[quat4, scale3, opacity1, sh_res3, weight1]` | `rasterization.py:60` |
| 位置来源 | `depth_to_world_coords_points(depth, pose, K)`，**不参与学习** | `rasterization.py:62-64` |
| SH | 只有 DC：`sh = (rgb-0.5)/C0 + residual`，C0=0.28209 | `rasterization.py:68` |
| scale 激活 | `exp().clamp_max(0.3)` | 同上 `:67` |
| opacity / weight | 各自 `sigmoid()` | 同上 |
| 解码精度 | 全程 FP32（显式关掉 autocast） | `gaussian_worldmirror.py:31` |
| 分块解码 | `decode_chunk_size=8`（按视角切，省显存） | `rasterization.py:55` |
| 发布包能力 | **只有推理**：`rasterization.py` 注释写明 "no rasterization or training" | 文件头 |

`cond_flags`：pose 模式 `[1,0,1]`（启用 pose + rays 条件），rgb 模式 `[0,0,0]`。
`enable_cond` 与 `model.use_gt_camera` 必须同时为真或同时为假，否则 `load_config` 直接抛错。

---

## 2. Gaussian Spacing Normalization（GSN）—— v3 的核心

这是 v3 相对 v2 最实质的改动，也是"一个模型跨任意场景"能成立的原因。

### 2.1 做什么

```
spacing = depth / sqrt(fx * fy)      # 每像素在世界里的前向 footprint
factor  = spacing_target / median(spacing)      # spacing_target = 0.0013227513（硬编码）
depth  *= factor                     # 所有视角同一个 factor
c2w[:3,3] *= factor                  # 只缩平移，不缩旋转
```

代码：`rasterization.py:88-127`（`normalize_gaussian_spacing`）。

`spacing_target` 不在 YAML 里（`inference.py:68-83` 的白名单会拒绝多余字段），
是代码常量，注释说它是"per-pixel spacing and geometric mean focal"的归一目标。

### 2.2 为什么需要

不同场景深度差几个数量级（物体 0.3 m、街景 50 m、航拍 300 m），同一模型输出的
高斯 scale 也就差几个数量级。若体素合并用固定 voxel size，小场景会被合并成一个点、
大场景合并等于没做。GSN 把所有场景压到**同一个 spacing 尺度**，于是：

- 网络只需学"相对于 spacing 的 scale"，不用学绝对尺度；
- 体素合并可以用**固定** `gs_voxel_size=0.002`（≈ 1.5 × spacing_target）。

这正是 v2 做不到的事：v2 的 voxel 是 `median_scale × 1.4` 逐场景自适应
（见 `infinisplat_v2/README_wsl.md` 的口径说明），一旦换场景就得重算，且没有统一定义。

### 2.3 pose 模式下的深度尺度对齐

pose 模式给了 GT 相机，但模型自己的深度是"预测尺度"。`da3_pose_align_factor`
（`rasterization.py:8-41`）沿用了 DA3 的 Umeyama fallback：只取**相机中心两两距离的
中位数比值**

```
factor = median(‖C_ref_i − C_ref_j‖ / ‖C_pred_i − C_pred_j‖)
```

方向无关（只比距离），所以对旋转误差不敏感。三个退化保护：

1. 视角数 < 2 → factor = 1；
2. 预测基线 ≈ 0 → 跳过该对；
3. GT 基线 ≈ 0（全景/立方体，所有相机同中心）→ 跳过该对，否则比值被拉到 ~0、
   点云会整体塌到原点（注释明确写了这个坑）。

⚠️ **导出坐标不是米制**。`docs/prepare_data.md` 原文："exported coordinates are not
guaranteed to be metric"。GSN 之后一切都是归一化尺度，量尺寸要自己反算。

---

## 3. 体素合并与过滤

顺序（`rasterization.py:143-161`，全部在合并**之前**判）：

1. `filter_large_scale_infer`：`max_scale_abs`（绝对上限，默认 null）与
   `max_scale_pct`（默认 99.8，丢最大 0.2% 的长条高斯）**同时生效**；
2. `filter_flying_points_infer`（默认 false）：深度不连续处的像素；
3. sky（`--sky_seg` 才启用）：sky 像素既被丢弃，也**不参与** spacing 统计
   （`valid_mask` 只影响统计，不影响缩放本身）。

合并规则（`merge_voxels`，`:167-196`）：

```
voxel 索引 = floor(mean / voxel_size)
means/scales/sh/quats = 按 weight 加权平均（quat 再归一化）
opacity = Σ(w·w) / Σw          ← 注意不是加权平均，是"权重平方和"
```

最后导出层再按 `infer_opacity_thresh=0.05` 丢一批。

**推论**：`points.ply` 是合并前的逐像素中心（带 RGB），`gaussians.ply` 是合并后的。
两者数量差多少，直接反映该场景的重叠率。v2 的经验是丢弃率 5–9%。

### 3.1 合并实际在做什么（E-08/E-09 实测，和直觉相反）

同一场景同一分辨率，只改输入视角数：

| 视角 | 合并保留率 | 每体素均/最多样本 | 体素内 RMS 离散度 | 锐度保留 | PSNR |
| --- | --- | --- | --- | --- | --- |
| 3 | 64.1% | 1.56 / 14 | 0.380 × voxel | 85% | 22.37 |
| 24 | 40.8% | 2.45 / 55 | 0.416 × voxel | 81% | 19.71 |

两条容易搞反的结论：

1. **合并消耗样本的比例随视角数上涨（36%→59%），但体素内离散度几乎不变**（0.38→0.42
   voxel）。所以不是"多视角打架打得很凶"，而是"更多样本被换成了邻域平均"。
   高斯 footprint 四档完全一致（scale 中位数 0.0013/0.0018/0.0009 逐位相同），
   于是表面变稀（1-NN 间距 0.0023→0.0033，+42%）。
2. **合并是低通降噪器，不是单纯的糊**。把 `gs_voxel_size` 从 0.002 降到 0.001：
   锐度保留 81% → **92%**，但 PSNR 19.71 → **19.16**、SSIM 0.563 → 0.529。
   机制：体素变小 → 更少样本被合并 → 相邻视角对同一表面点的两份"略有偏差的副本"
   同时留下 → 渲染出重叠双层/斑纹。合并把它们平均成一个干净表面。
   → **PSNR 和锐度在体素尺寸上互相矛盾，选参不能只看 PSNR。**

所以自洽 PSNR **不能跨视角数比较**（E-08 里同一个视角 22→19.0 单调下降）。
要排名必须做 held-out。

---

## 4. 预处理：以主点为中心裁正方形

`preprocess_view`（`scene_io.py:28-50`）三步，DUSt3R 血统：

1. 以 `K` 的 `(cx,cy)` 为中心，裁出**最大对称方形** `(cx-mx, cy-my, cx+mx, cy+my)`；
2. `rescale_image_depthmap` 缩到 `--resolution`（默认 756×756）；
3. 用 `camera_matrix_of_crop` 再精调一次 bbox（吸收主点取整误差）。

后果：**非正方形输入会丢两边**。1920×1080 只保留中间 1080×1080（左右各丢 420 px）。
`rgb/` 目录存的就是裁完的图，PSNR 对比必须与之对齐，不能拿原图算。

`--resolution` 必须是 14 的正整数倍（patch size）。文档说支持 1K 以上。

---

## 5. 与 InfiniSplat v2 的差异（迁移经验时注意）

| 维度 | v2 | v3（SplatAnything） |
| --- | --- | --- |
| 几何基座 | DA3（Depth-Anything-3，钉死 commit） | **WorldMirror-2**（自带 camera head + depth + conf + mask） |
| 位姿模式 | provided / colmap / da3-pose / pose-free | 只有 `rgb`（自估）/ `pose`（读 JSON） |
| 尺度归一 | 无统一口径，voxel = median_scale×1.4 | **GSN**：spacing 归一到 0.0013227513，voxel 固定 0.002 |
| 输入相机格式 | `cameras.npz`（c2w OpenCV + K） | `prior_camera.json`（按 `camera_id`=图片 stem 匹配） |
| 输出 | `scene.ply` + `cameras.npz` + `result.json` | `gaussians.ply` + `points.ply` + `cameras.json` + `depth/` + `rgb/` |
| 精调 | 有（v2 仓自写 `02b`，非官方） | **无**，需要自己接 |
| 训练代码 | 无 | 无 |
| 渲染器 | 不带，需自接 gsplat | 不带，需自接 |

共同点：都是"位置 = 深度反投影、只学高斯属性"，都不含训练代码，都要自己接渲染做评测。

---

## 6. 官方发布包的已知缺陷（实测）

1. **`validation.py` 不存在**。`docs/inference.md` 写 "the lightweight `validation.py`
   entry point remains available"，但 `git ls-tree HEAD` 里没有这个文件，
   `tests/test_inference.py:17` 的 `from validation import load_config, main, parse_args`
   必然 ImportError。→ 整组 `unittest discover` 跑不了，只能单跑
   `tests.test_camera_conversion`。
2. **`examples/object/prior_camera.json` 的 `num_cameras=49`，但 extrinsics/intrinsics
   只有 24 条，images 里 23 张**。load 时按 entries 匹配、`num_cameras` 不被使用
   （`camera_io.py`），所以**不影响运行**，但说明打包脚本里那个字段是错的。
3. **仓库 418 MB**，其中 examples/ 占绝大部分（pano 24 张各 ~5 MB、rig 150 张）。
   建议 `--filter=blob:none --sparse` 克隆，只在需要时
   `git sparse-checkout add examples/<category>`。

---

## 7. 排错

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 下载 50–60 KB/s | HF 镜像 302 到 `us.aws.cdn.hf.co` / `cas-bridge.xethub.hf.co`，这两个域名没进 `no_proxy`，被塞进 v2rayN | `_env.sh` 的 `no_proxy` 加 `.hf.co,.xethub.hf.co`；`hf_download.py` 的 curl 加 `--noproxy '*'`。修完 1.7 MB/s |
| `Missing conda env` / `python: command not found` | `01` 依赖 `00` 建好的 env | 先跑 `00_setup_env.sh` |
| `pose 模式要求 prior_camera.json` | JSON 必须和 `images/` 同级 | 用 `scripts/convert_cameras.py` 生成 |
| `No camera entry for image 'xxx'` | `camera_id` 必须等于图片**文件名去扩展名** | 转换时统一命名，别手改 |
| `All Gaussians are below opacity_thresh` | 权重没对上，或 scale 过滤把点全丢了 | 先确认权重 SHA256；再试 `--opacity_thresh 0` |
| 输出目录同名覆盖 | `inference.py` 直接写 `OUT/SCENE/` | `02` 已处理：改输出根加时间戳 |
| gsplat `AttributeError: 'NoneType' object has no attribute 'CameraModelType'` | CUDA 后端没编出来（缺 nvcc） | `conda install -n $CONDA_ENV -c nvidia cuda-nvcc=12.6` |
| gsplat `_jit_compile() missing 2 required positional arguments` | gsplat 1.5.3 与 torch 2.7.1 的 JIT 接口不兼容 | 降到 `gsplat==1.5.2`（v2 实测可用的版本） |
| gsplat 编译 `#include <glm/gtc/type_ptr.hpp>` 找不到 | 1.5.2 不内置 glm（1.5.3 才自带 third_party/glm） | 需要 glm 头文件；本机尚未打通，见 EXPERIMENTS 遗留项 |
| nvcc `gcc: No such file or directory` | 本机 WSL 没装 gcc/g++ | 借用已有 conda env 的 gcc（`envs/infinisplat_v2/bin`），或 `conda install gxx_linux-64` |
