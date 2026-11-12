# SplatAnything（InfiniSplat v3）— 前馈多视角 3DGS，跨任意场景（WSL 复现）

从多张 RGB 图（可有/无位姿）一次前馈重建 3D Gaussian 场景。相对 v2 的核心改动是
**Gaussian Spacing Normalization（GSN）**：把每个场景的中位"每像素前向间距"
（`depth / sqrt(fx·fy)`）归一到固定目标值 0.0013227513，深度与相机平移同步缩放，
于是体素合并可以用固定 `voxel_size=0.002`，不再逐场景自适应。

- 论文：SplatAnything: Gaussian Spacing Normalization for Feed-Forward 3DGS across Arbitrary Scenes（ZJU3DV + 腾讯混元）
- 官方仓：<https://github.com/zju3dv/SplatAnything>（本仓实测 commit `b518996`）
- 权重：<https://huggingface.co/ritianyu/SplatAnything>（apache-2.0，非 gated）
- 原理与坑：见 [NOTES.md](NOTES.md)；实测数字见 EXPERIMENTS.md
- 前作参照：[../infinisplat_v2/](../infinisplat_v2/)（DA3 基座 + 自适应 voxel）

## ⚠️ 复现边界（先读这段）

官方**只开源推理包**（`rasterization.py` 注释写明 "no rasterization or training"）：
没有训练代码、没有 refiner、没有光栅化器、没有官方评测划分。

| 项 | 状态 | 说明 |
| --- | --- | --- |
| 前馈推理（rgb / pose 两模式） | ✅ 可复现 | 代码 + 权重齐全，`02_run_inference.sh` |
| GSN + 固定体素合并 + scale/opacity 过滤 | ✅ 可复现 | 参数全在 `configs/inference.yaml` + 命令行 |
| 导出标准 3DGS PLY | ✅ 可复现 | SH DC / log scale / logit opacity / wxyz |
| 渲染与 PSNR/SSIM | ⚠️ 需自接 | 发布包不带 gsplat，`03_eval_render.sh` 自接 |
| 训练（stage-1/2） | ❌ 不可得 | 无训练代码 |
| 论文表里的精确数值 | ⚠️ 不可直接对标 | 官方划分未公开；本仓只做自洽 + 自建 holdout |

## 实测结果（本机 3090 24 GB）

| 场景 | 视角 | 分辨率 | 模式 | 推理 | 总耗时 | 峰值显存 | 高斯 | 自洽 PSNR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `examples/object` | 23 | 756² | rgb | 68.3 s | 113 s | **23.53 GB** | 2.33 M | — |
| `examples/object` | 23 | 756² | pose | 67.3 s | 117 s | **23.59 GB** | 2.46 M | 19.01 |
| `jiajia_hold`（采集包） | 24 | 504² | pose | 7.3 s | 44 s | 21.79 GB | 1.94 M | 19.71 |

⚠️ **756² × 23 视角就是 24 GB 卡的上限**（23.5 GB）。超过不会 OOM 而是严重变慢：
23v@756 是 1.02 ms/token，24v@504 是 0.23 ms/token —— 面积只差 2.25×，耗时差 9.4×。
**判据：耗时/面积比超过 5× 就是撞顶了。** 要跑 24+ 视角请用 504²。

GSN 生效的硬证据（`05_check_scale.py`）：物体示例与人像采集包的原始相机半径差 3.9×
（1.746 vs 0.452），但高斯 `scale / spacing_target` 都落在 **0.80 ~ 1.37**，
1-NN 间距 p50 都是 **0.0032~0.0034**（≈ √3 × voxel 0.002 的对角线）。

### 视角数控制面板（`jiajia_hold`，pose，504²，最远点采样、三档嵌套）

| 视角 | 高斯 | 合并保留率 | 1-NN p50 | 自洽 PSNR | SSIM | 锐度保留 | 显存 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 473 K | 64.1% | 0.00229 | **22.37** | 0.725 | 85% | 13.41 GB |
| 8 | 1.01 M | 49.7% | 0.00279 | 19.91 | 0.607 | 84% | 18.62 GB |
| 13 | 1.30 M | 39.3% | 0.00287 | 19.95 | 0.581 | 81% | 19.17 GB |
| 24 | 1.94 M | 40.8% | 0.00326 | 19.71 | 0.563 | 81% | 21.79 GB |

⚠️ **自洽 PSNR/SSIM 随视角数单调下降，不能跨视角数比较**（同一视角 `022_…`：23.30 → 19.55
→ 19.07 → 19.0）。尺度、场景包围盒、高斯 footprint 四档一致（排除 GSN 混淆），前景 PSNR
同样下降（排除背景稀释），锐度只掉 4 个百分点（排除单纯变糊）。实际机制是体素合并消耗样本
（保留率 64%→41%），把更多样本换成邻域平均。详见 `EXPERIMENTS.md` E-08/E-09。

### PLY 导出等价性（`06_run_roundtrip.sh`，PASS）

内存过滤后 vs PLY 重载：means / quats / f_dc **MAE = 0**（逐位相同），log scale 1.97e-09，
opacity 8.89e-09，相机漂移 0，渲染 RGB MAE **3.4e-08** / alpha 1.2e-08。
→ 03 的指标评的确实就是模型吐出的那批高斯。

### held-out NVS（`08_nvs_eval.sh`，真留出）

| 场景 | source | 留出 | PSNR | 前景 PSNR | SSIM | alpha | 交集 PSNR | 交集前景 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `jiajia_hold` | 3 | 21 | 15.23 | 18.17 | 0.406 | 0.789 | 15.00 | 18.35 |
| `jiajia_hold` | 8 | 16 | 17.81 | 18.47 | 0.464 | 0.942 | 17.76 | 18.39 |
| `jiajia_hold` | 13 | 11 | **18.39** | **18.91** | **0.502** | 0.955 | **18.39** | **18.91** |
| `object` | 3 | 20 | **19.92** | **21.57** | **0.599** | 0.676 | **19.37** | **20.70** |
| `object` | 8 | 15 | 19.03 | 18.95 | 0.473 | 0.728 | 18.71 | 18.54 |
| `object` | 13 | 10 | 19.08 | 18.90 | 0.492 | 0.750 | 19.08 | 18.90 |

⚠️ 三条读之前必须知道的：

1. **自洽口径与 held-out 方向相反**（人像场景自洽 3v 22.37 → 24v 19.71 在降，held-out
   3v 15.00 → 13v 18.39 在升）。自洽分数**不能用来排位**，只能证明链路通。
2. 人像场景的增益几乎全在**覆盖度**（alpha 0.789→0.955），同视角前景 PSNR 只涨 0.56。
3. 物体场景 **3v 反而最好**，体素尺寸与覆盖度两条解释都已排除，根因未定位 —— 不要引用。

对齐自证：`M` 三视角离散 2.4e-07、`det=1.000000`、两种独立尺度估计都是 0.332757
（离散 1.5e-08）、source 反算残差 3.8e-08。任一不过 `08` 直接报错退出，不渲错位图。

## 常用命令

> 首次跑前先做「首次准备」。每条命令都显式写出权重、输入、输出路径。

```bash
# ── A) 官方示例（先检出 examples，仓库 418 MB 只拉需要的那一类）──
git -C ~/repos/SplatAnything sparse-checkout add examples/object

GPU=0 EXAMPLE=object MODE=rgb \
  bash splatanything/02_run_inference.sh
#   → ~/output/splatanything_results/rgb/object/{gaussians.ply,points.ply,cameras.json,depth/,rgb/}

GPU=0 EXAMPLE=object MODE=pose RESOLUTION="504 504" \
  bash splatanything/02_run_inference.sh

# ── B) 自有场景（目录里必须有 images/，pose 模式再加 prior_camera.json）──
GPU=0 INPUT=/home/velvet/output/splatanything_results/_scenes/jiajia_hold MODE=pose \
  bash splatanything/02_run_inference.sh

# ── C) 华为 Remy 采集包 → 场景目录 → 推理（一条链）──
GPU=0 CAPTURE="/mnt/d/dataset/手部微动样例/01e954de1b8a4cde9315c976f4dcb964-佳佳抱枕" \
  SCENE=jiajia_hold N_VIEWS=24 \
  bash splatanything/07_run_capture.sh

GPU=0 INPUT="$HOME/output/splatanything_results/_scenes/jiajia_hold" MODE=pose \
  bash splatanything/02_run_inference.sh

# ── D) 渲染评测（自洽口径；缺 gsplat 会自动装）──
GPU=0 RESULT="$HOME/output/splatanything_results/pose/object" \
  bash splatanything/03_eval_render.sh

# ── E) 不加载权重验链路（CPU，秒级；改完代码先跑这个）──
python splatanything/04_smoke_cpu.py

# ── F) 导出往返验证（跑一次推理，比"内存高斯"与"PLY 重载"）+ io_audit ──
GPU=0 INPUT="$HOME/output/splatanything_results/_scenes/jiajia_hold_3v" MODE=pose \
  RESOLUTION="504 504" bash splatanything/06_run_roundtrip.sh
#   → _roundtrip/<scene>/{ply_roundtrip.json, roundtrip_diff.png}

# ── G) 抽子集做视角数面板（最远点采样，天然嵌套 3v⊂8v⊂13v）──
python splatanything/07b_subset_scene.py \
  --scene "$HOME/output/splatanything_results/_scenes/jiajia_hold" --n 8 \
  --out "$HOME/output/splatanything_results/_scenes/jiajia_hold_8v"
#   meta.json 里的 held_out 就是 WB-2 的 target 清单

# ── H) held-out NVS（source 建云，留出视角只评分）──
GPU=0 RESULT="$HOME/output/splatanything_results/panel/jiajia_hold_8v" \
  SOURCE_SCENE="$HOME/output/splatanything_results/_scenes/jiajia_hold_8v" \
  ALL_SCENE="$HOME/output/splatanything_results/_scenes/jiajia_hold" \
  bash splatanything/08_nvs_eval.sh
#   → RESULT/nvs/{nvs_summary.json, contact_sheet.png, render/*.png}

# 多配置汇总（交集视角上严格可比）
python splatanything/08b_nvs_compare.py \
  --runs a/nvs_summary.json b/nvs_summary.json --csv cmp.csv
```

**03 新增指标**（同一份 `eval_summary.json`）：`psnr_fg`（alpha>0.5 前景）、`psnr_bg`
（alpha<0.01，抓"幽灵在 GT 掩膜之外"）、`sharp_ratio`（前景掩膜上的梯度均值比，<1 = 糊）。
为什么必须分区：整体 PSNR 会把"前景糊"和"背景铺得广不广"混在一起；锐度保留率则把
"糊"和"颜色偏"分开。

## 首次准备

```bash
# 1) clone 本仓 + proxy.env
cd ~ && git clone <本仓地址> && cd media_code && cp proxy.env.example proxy.env

# 2) 建 env（python 3.11 + torch 2.7.1）+ clone 官方仓（稀疏）+ 装依赖 + 自检
bash splatanything/00_setup_env.sh

# 3) 下权重（16.6 GB，走 aifasthub 镜像，分块并发 + SHA256）
bash splatanything/01_download_models.sh
```

权重目录布局（`$SPLATANYTHING_MODEL_DIR`，默认 `/mnt/d/wheel/splatanything_ms`）：

```
/mnt/d/wheel/splatanything_ms/
├── SplatAnything.ckpt          11.3 GB  完整高斯学生（stage-1）
├── worldmirror2.safetensors     5.05 GB  WorldMirror-2 几何（teacher）
└── skyseg.onnx                  176 MB  可选，--sky_seg 才用
```

`01` 会自动在官方仓 `checkpoints/` 建同名软链（`configs/inference.yaml` 的路径锚在 repo root）。

## Config（env vars）

| var | default | note |
| --- | --- | --- |
| `MODE` | `rgb` | `rgb` = 模型自估位姿；`pose` = 读 `INPUT/prior_camera.json` |
| `RESOLUTION` | `756 756` | 模型输入宽高，必须 14 的正整数倍；文档称支持 1K+ |
| `INPUT` / `EXAMPLE` | — | 场景目录（含 `images/`）；`EXAMPLE` 指官方 `examples/<名>` |
| `OUT_ROOT` | `$RESULTS_DIR/$MODE` | 输出根；inference 再套一层 `<scene>/` |
| `SKY_SEG` | `0` | `1` = 天空分割（需 `skyseg.onnx`），天空像素同时不参与 spacing 统计 |
| `OPACITY_THRESH` | YAML `0.05` | 导出层的最小不透明度 |
| `EXTRA_ARGS` | — | 透传，如 `--no_cam_align`（不锚到首个视角） |
| `PROFILE` | `1` | 采样 GPU 峰值显存 |
| `CAPTURE` / `N_VIEWS` | — / `24` | 07：采集包路径 / 等间隔抽帧数（2..50） |
| `EPS2D` | `0.01` | 03：屏幕空间膨胀 |
| `CONDA_ENV` | `splatanything` | |

`configs/inference.yaml` 只接受固定字段（多一个就报错），`spacing_target` 和
`decode_chunk_size` 是代码常量、**不能**从 YAML 改。

## 目录布局

```
media_code/splatanything/
├── _env.sh                  # 代理 / conda / 路径（含 CDN 直连的 no_proxy）
├── _gsplat_env.sh           # gsplat 安装 + 编译四件套（03/06 共用）
├── 00_setup_env.sh          # 稀疏 clone + conda env + torch + 依赖 + 自检
├── 01_download_models.sh    # 3 个权重 + SHA256 + checkpoints 软链
├── 02_run_inference.sh      # 推理（含显存采样与产物统计）
├── 03_eval_render.sh        # 自接 gsplat 渲染 + PSNR/SSIM/锐度/分区（自洽口径）
├── 03_render_eval.py
├── 04_smoke_cpu.py          # CPU 全链路自检（不加载真权重，56×56 图，秒级）
├── 04b_test_render.py       # 造最小 result 目录验渲染链路
├── 05_check_scale.py        # 验 GSN 尺度契约（scale/spacing_target、1-NN 间距）
├── 06_run_roundtrip.sh      # PLY 往返验证 + io_audit + 体素合并统计
├── 06_ply_roundtrip.py
├── 07_run_capture.sh        # Remy 采集包 → 场景目录
├── 07_capture_to_scene.py
├── 07b_subset_scene.py      # 场景抽子集（最远点采样，出 held_out 清单）
├── 08_nvs_eval.sh           # held-out NVS 评测（source 建云，留出只评分）
├── 08_nvs_eval.py
├── 08b_nvs_compare.py       # 多配置汇总（在交集视角上重算，才可比）
├── hf_download.py           # 镜像分块并发下载（含 --noproxy）
├── NOTES.md                 # 原理（GSN / 体素合并 / 预处理）与排错
├── EXPERIMENTS.md           # 实测台账（E-01~E-10）
└── ROUTE_VALIDATION_REPORT.md  # 按 WB 任务单回填的验收报告

官方仓（Linux fs，稀疏检出）: ~/repos/SplatAnything
权重（D 盘）              : /mnt/d/wheel/splatanything_ms
输出（Linux fs）          : ~/output/splatanything_results
```

## 输入格式

```
my_scene/
├── images/               # jpg/png/webp/bmp/tif；文件名 stem 必须唯一（用作 camera_id）
└── prior_camera.json     # pose 模式必需；rgb 模式忽略（即使存在）
```

`prior_camera.json` 要 **OpenCV c2w**（X 右 / Y 下 / Z 前）+ **原图像素**内参：

```json
{ "extrinsics": [{"camera_id": "000", "matrix": [[...4x4 c2w...]]}],
  "intrinsics": [{"camera_id": "000", "matrix": [[fx,0,cx],[0,fy,cy],[0,0,1]]}] }
```

- `camera_id` = 图片**文件名去扩展名**，必须每张图都有条目
- 内参要求零 skew、主点落在原图内（否则 loader 直接抛错）
- 已有 COLMAP 的用官方 `scripts/convert_cameras.py colmap` 转；自有矩阵用 `arrays` 模式
- 相机中心会被重新锚到第一个视角（`--no_cam_align` 可关），且**导出坐标不是米制**（GSN 归一化过）

## Notes

- 预处理以**主点为中心裁正方形**再缩到 `RESOLUTION`：1920×1080 只保留中间 1080×1080。
  `rgb/` 存的是裁后的图，评测必须用它当 GT。
- 官方包两个已知缺陷：`validation.py` 缺失（导致 `tests/test_inference.py` 无法运行）、
  `examples/object/prior_camera.json` 的 `num_cameras` 与实际条目数不一致（不影响运行）。
  详见 NOTES.md 第 6 节。
- 下载慢先查 `no_proxy`：镜像站 302 到 `us.aws.cdn.hf.co` / `cas-bridge.xethub.hf.co`，
  这两个域名被 v2rayN 代理接管后实测只有 57 KB/s（直连 1.7 MB/s）。
