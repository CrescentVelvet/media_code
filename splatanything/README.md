# SplatAnything（InfiniSplat v3）— 黄区普通服务器运行手册

在**黄区普通 Ubuntu + NVIDIA 容器服务器**上运行 [SplatAnything](https://github.com/zju3dv/SplatAnything)：多张 RGB 图一次前馈输出 3D Gaussian PLY、点云、相机与逐视角深度。支持模型自估位姿（`rgb`）和已知相机（`pose`）两种模式。本目录只含编排脚本，官方代码自动 clone，不复制进本仓。

> **两套服务器不要混淆：** 本文 00–11 节及 35 节的通用命令只面向**黄区普通服务器**，**不是蓝区 H100 集群的操作手册**。蓝区的专用路径、环境和 Slurm 运行方式见 [README_H100.md](README_H100.md)，集群公共规则见 [canada_h100/README.md](../canada_h100/README.md)。
>
> 📄 本机 WSL 路径与人脸研究命令见 [README_wsl.md](README_wsl.md)；GSN、相机坐标及排错见 [NOTES.md](NOTES.md)；本机实测与人脸实验见 [EXPERIMENTS.md](EXPERIMENTS.md)。黄区普通服务器的基础前馈流程尚未完整实跑；35 号 LGTM 小试只在**蓝区 H100** 独立环境实测，两者验证状态分开记录。

## 复现边界

官方发布的是**推理包**：没有官方训练代码、refiner、光栅化器或论文评测划分。本仓的 03/06/08 自接 `gsplat` 做渲染验收；09 只是属性头训练成本探针，不能视为官方训练复现。PLY 和导出相机经过 Gaussian Spacing Normalization（GSN），坐标不是米制。

| 功能 | 状态 |
|---|---|
| 前馈推理、GSN、固定体素合并、标准 3DGS PLY | ✅ 官方代码与公开权重 |
| 自洽渲染、PLY 往返、留出视角评测 | ✅ 本仓自接 `gsplat` |
| 训练成本探针 | ⚠️ 研究脚本，仅训练属性头 |
| LGTM 纹理小试（35） | ⚠️ 单场景、冻结几何；投影链通过，画质未通过 |
| 官方训练与论文表格精确复现 | ❌ 发布包不支持 |

## 常用命令

> 假设已进入容器，当前目录为 `media_code`。首次跑前先完成「首次准备」。每条命令显式给出代码、权重、输入和输出路径，不依赖本机默认值。以下 `../../model`、`../SplatAnything`、`../splatanything_results` 均相对本仓目录解析。

```bash
# ── 1) 官方物体示例：rgb 自估位姿 ──
# 输入：../SplatAnything/examples/object/images/（23 张图）
# 输出：../splatanything_results/rgb/object/
git -C ../SplatAnything sparse-checkout add examples/object
GPU=0 CONDA_ENV=splatanything \
  SPLATANYTHING_DIR=../SplatAnything \
  SPLATANYTHING_MODEL_DIR=../../model/splatanything \
  SPLATANYTHING_RESULTS_DIR=../splatanything_results \
  EXAMPLE=object INPUT= MODE=rgb RESOLUTION="756 756" \
  OUT_ROOT=../splatanything_results/rgb \
  bash splatanything/02_run_inference.sh

# ── 2) 同一示例：pose 已知相机 ──
GPU=0 CONDA_ENV=splatanything \
  SPLATANYTHING_DIR=../SplatAnything \
  SPLATANYTHING_MODEL_DIR=../../model/splatanything \
  SPLATANYTHING_RESULTS_DIR=../splatanything_results \
  EXAMPLE=object INPUT= MODE=pose RESOLUTION="504 504" \
  OUT_ROOT=../splatanything_results/pose \
  bash splatanything/02_run_inference.sh
# 输出：../splatanything_results/pose/object/
#   gaussians.ply  # 过滤、体素合并后的标准 3DGS
#   points.ply     # 过滤、合并前的场景点云
#   cameras.json   # 与 PLY 同坐标系的 OpenCV c2w + 内参
#   depth/ rgb/    # 逐视角深度与模型实际输入图
#   config.yaml run.json

# ── 3) 自有场景：目录含 images/，pose 另需 prior_camera.json ──
GPU=0 CONDA_ENV=splatanything \
  SPLATANYTHING_DIR=../SplatAnything \
  SPLATANYTHING_MODEL_DIR=../../model/splatanything \
  SPLATANYTHING_RESULTS_DIR=../splatanything_results \
  INPUT=../data/my_scene EXAMPLE= MODE=pose RESOLUTION="756 756" \
  OUT_ROOT=../splatanything_results/pose \
  bash splatanything/02_run_inference.sh
# 输出：../splatanything_results/pose/my_scene/

# ── 4) Remy 采集包转场景（transforms.json + image/*.heic + pcd.ply）──
CONDA_ENV=splatanything \
  SPLATANYTHING_DIR=../SplatAnything \
  SPLATANYTHING_MODEL_DIR=../../model/splatanything \
  SPLATANYTHING_RESULTS_DIR=../splatanything_results \
  CAPTURE=../data/remy_capture SCENE=my_capture N_VIEWS=24 FRAMES= \
  OUT_DIR=../splatanything_results/_scenes/my_capture \
  bash splatanything/07_run_capture.sh
# 输出：../splatanything_results/_scenes/my_capture/{images/,prior_camera.json,meta.json}
# 如需指定原始帧：FRAMES=0,40,81，优先级高于 N_VIEWS；不要把多个值误写成未定义的路径。

# ── 5) 自洽渲染：PLY 与 cameras.json 回放，对拍模型实际输入 rgb/ ──
GPU=0 CONDA_ENV=splatanything \
  SPLATANYTHING_DIR=../SplatAnything \
  SPLATANYTHING_MODEL_DIR=../../model/splatanything \
  SPLATANYTHING_RESULTS_DIR=../splatanything_results \
  RESULT=../splatanything_results/pose/object \
  OUT=../splatanything_results/pose/object/eval \
  GLM_DIR=../glm GCC_DIR=/usr/bin EPS2D=0.01 MAX_VIEWS=0 \
  bash splatanything/03_eval_render.sh
# 输出：eval_summary.json、逐视角 render/ 与 contact_sheet.png。
# 自洽分数只验证导出和回放链，不代表新视角质量。

# ── 6) PLY 往返：重新前馈，对比内存高斯与 PLY 重载 ──
GPU=0 CONDA_ENV=splatanything \
  SPLATANYTHING_DIR=../SplatAnything \
  SPLATANYTHING_MODEL_DIR=../../model/splatanything \
  SPLATANYTHING_RESULTS_DIR=../splatanything_results \
  INPUT=../SplatAnything/examples/object MODE=pose RESOLUTION="504 504" \
  OUT=../splatanything_results/_roundtrip \
  GLM_DIR=../glm GCC_DIR=/usr/bin EPS2D=0.01 CHUNK=4 \
  bash splatanything/06_run_roundtrip.sh
# 输出：_roundtrip/object/{ply_roundtrip.json,roundtrip_diff.png}；会再次加载权重并推理。

# ── 7) 真留出 NVS：先抽 source 子集，再推理、只在其余视角评分 ──
CONDA_ENV=splatanything SPLATANYTHING_DIR=../SplatAnything \
  "$HOME/miniconda3/envs/splatanything/bin/python" splatanything/07b_subset_scene.py \
  --scene ../SplatAnything/examples/object --n 3 \
  --out ../splatanything_results/_scenes/object_3v --hardlink 0
GPU=0 CONDA_ENV=splatanything \
  SPLATANYTHING_DIR=../SplatAnything \
  SPLATANYTHING_MODEL_DIR=../../model/splatanything \
  SPLATANYTHING_RESULTS_DIR=../splatanything_results \
  INPUT=../splatanything_results/_scenes/object_3v EXAMPLE= MODE=pose \
  RESOLUTION="504 504" OUT_ROOT=../splatanything_results/pose \
  bash splatanything/02_run_inference.sh
GPU=0 CONDA_ENV=splatanything \
  SPLATANYTHING_DIR=../SplatAnything \
  SPLATANYTHING_MODEL_DIR=../../model/splatanything \
  SPLATANYTHING_RESULTS_DIR=../splatanything_results \
  RESULT=../splatanything_results/pose/object_3v \
  SOURCE_SCENE=../splatanything_results/_scenes/object_3v \
  ALL_SCENE=../SplatAnything/examples/object \
  RESOLUTION="504 504" OUT=../splatanything_results/pose/object_3v/nvs \
  GLM_DIR=../glm GCC_DIR=/usr/bin EPS2D=0.01 MAX_HELD_OUT=0 CHUNK=4 \
  bash splatanything/08_nvs_eval.sh
# 输出：nvs_summary.json、render/、contact_sheet.png。
# 子集目录必须不存在；已有 PLY 时不要重复前馈，直接从 08 开始。

# ── 8) 360° 轨道视频（前馈稀疏视角之外可能为空，属预期）──
GPU=0 CONDA_ENV=splatanything \
  SPLATANYTHING_DIR=../SplatAnything \
  SPLATANYTHING_MODEL_DIR=../../model/splatanything \
  SPLATANYTHING_RESULTS_DIR=../splatanything_results \
  RESULT=../splatanything_results/pose/object \
  OUT=../splatanything_results/pose/object/orbit.mp4 \
  GLM_DIR=../glm GCC_DIR=/usr/bin FRAMES=120 FPS=30 EPS2D=0.01 \
  bash splatanything/11_run_orbit.sh
```

**不要混用评测口径：**
- 03 用导出相机和 `rgb/` 对拍，是自洽检查；输入越多时体素合并保留率会改变，不能用它给视角数排位。
- 08 的 `ALL_SCENE` 必须含全部相机与图片，`SOURCE_SCENE` 是实际推理子集；留出 = 全集 − source。target 不参与建云或对齐。
- `RESOLUTION` 必须与该 PLY 的推理配置一致。原生回放直接用导出相机和 K，不要再额外拟合 Sim(3)。

## 首次准备

```bash
# 先进入你的代码目录，例如：cd /data_3d/<uid>/code
git clone https://github.com/CrescentVelvet/media_code.git
cd media_code
cp proxy.env.example proxy.env   # 按需填代理；凭据不要提交

# 创建专用 Python 3.11 + torch 2.7.1 环境，稀疏 clone 官方仓并安装依赖。
CONDA_ENV=splatanything \
  SPLATANYTHING_DIR=../SplatAnything \
  SPLATANYTHING_MODEL_DIR=../../model/splatanything \
  SPLATANYTHING_RESULTS_DIR=../splatanything_results \
  PY_VER=3.11 TORCH_VER=2.7.1 \
  PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple \
  bash splatanything/00_setup_env.sh

# 下载三份公开权重并校验 SHA256，在官方仓 checkpoints/ 建同名软链。
CONDA_ENV=splatanything \
  SPLATANYTHING_DIR=../SplatAnything \
  SPLATANYTHING_MODEL_DIR=../../model/splatanything \
  SPLATANYTHING_RESULTS_DIR=../splatanything_results \
  HF_BASE_URL=https://aifasthub.com DOWNLOAD_JOBS=8 \
  bash splatanything/01_download_models.sh

# CPU 冒烟：随机权重检查接口、GSN 和 PLY 导出，不验证画质。
# 此脚本内部固定从 ~/repos/SplatAnything 导入；先建立兼容软链，不改服务器官方仓。
mkdir -p "$HOME/repos"
ln -sfn "$(cd ../SplatAnything && pwd)" "$HOME/repos/SplatAnything"
"$HOME/miniconda3/envs/splatanything/bin/python" splatanything/04_smoke_cpu.py

# 可选渲染依赖：nvcc、gcc/g++、glm。GCC_DIR 指向含 gcc/g++ 的目录。
conda install -n splatanything -c nvidia cuda-nvcc=12.6 -y
git clone --depth 1 https://github.com/g-truc/glm.git ../glm
# 无系统编译器时，改为安装到专用环境，并在渲染命令中设置：
# conda install -n splatanything -c conda-forge gxx_linux-64=12 python=3.11 -y
# GCC_DIR="$HOME/miniconda3/envs/splatanything/bin"
```

⚠️ 00/01/02 会读取脚本默认值，因此服务器每次都要传上述三个 `SPLATANYTHING_*` 路径，或在 `proxy.env` 中统一覆盖。`_env.sh` 会激活 `CONDA_ENV`；00 负责创建它，无需预先手动建环境。

⚠️ 当前默认镜像与 `hf_download.py` 的 `curl --noproxy '*'` 会绕过代理；服务器若不能直接访问镜像，先测试连通性并更换 `HF_BASE_URL`，必要时调整下载方式。SSL/企业代理 CA 的处理见仓级规范与 `proxy.env.example`。

权重目录：

```
../../model/splatanything/
├── SplatAnything.ckpt          # 完整高斯学生，约 11.3 GB
├── worldmirror2.safetensors    # WorldMirror-2 几何教师，约 5.05 GB
└── skyseg.onnx                 # 可选天空分割，约 176 MB
```

## 输入格式

```
my_scene/
├── images/               # jpg/png/webp/bmp/tif；文件名 stem 用作 camera_id，必须唯一
└── prior_camera.json     # pose 必需；rgb 模式忽略
```

`prior_camera.json` 使用 **OpenCV camera-to-world**：X 右、Y 下、Z 前；K 必须对应**原始输入图像素**，不是推理分辨率：

```json
{
  "extrinsics": [{"camera_id": "000", "matrix": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}],
  "intrinsics": [{"camera_id": "000", "matrix": [[1000,0,500],[0,1000,500],[0,0,1]]}]
}
```

- `camera_id` 必须等于图片文件名去扩展名，每个视角都要有外参与内参；零 skew，主点在原图内。
- 预处理以主点为中心裁正方形，再缩放到 `RESOLUTION`；非正方形输入会丢两侧或上下内容，先看 `rgb/`。
- 官方默认将相机锚到首个视角，PLY 与导出相机同处 GSN 后的模型坐标系。不要直接把深度或平移当米制。
- Remy 转换器执行 OpenGL→OpenCV 翻转及去畸变；已有其他格式时先核对外参定义，不能套用未知相机矩阵。

## Config（env vars）

### 路径与环境（00–11 共用）

| var | 脚本默认值 | 服务器建议 / 说明 |
|---|---|---|
| `CONDA_ENV` | `splatanything` | 专用环境，不与其他算法混装 |
| `GPU` | 未设置 | 物理卡号；设置后写入 `CUDA_VISIBLE_DEVICES` |
| `SPLATANYTHING_DIR` | `~/repos/SplatAnything` | 服务器显式用 `../SplatAnything` |
| `SPLATANYTHING_MODEL_DIR` | `/mnt/d/wheel/splatanything_ms` | 服务器显式用 `../../model/splatanything` |
| `SPLATANYTHING_RESULTS_DIR` | `~/output/splatanything_results` | 服务器显式用 `../splatanything_results` |
| `PY_VER` / `TORCH_VER` | `3.11` / `2.7.1` | 00 创建环境及安装 PyTorch |
| `PIP_INDEX` | 清华 PyPI 镜像 | 可按服务器网络改源 |
| `CUDA_INDEX` | 空 | 非空时 00 改用指定 PyTorch CUDA 源 |
| `FORCE_RECREATE` | `0` | `1` 删除并重建同名 conda env，谨慎使用 |
| `HF_BASE_URL` | `https://aifasthub.com` | 01 下载源，按服务器网络选择 |
| `DOWNLOAD_JOBS` | `8` | 下载分块并发数 |
| `SKIP_SKYSEG` | `0` | `1` 不下载可选天空模型 |
| `SKIP_SHA` | `0` | `1` 跳过 SHA256，不建议 |

### 前馈推理（02）

| var | default | note |
|---|---|---|
| `INPUT` / `EXAMPLE` | 二选一 | 自有场景目录 / 官方示例名；另一项显式置空 |
| `MODE` | `rgb` | `rgb` 自估位姿；`pose` 读取 `prior_camera.json`，仅在标定可信时使用 |
| `RESOLUTION` | `756 756` | 宽、高均须是 14 的正整数倍 |
| `OUT_ROOT` | `$SPLATANYTHING_RESULTS_DIR/$MODE` | 最终输出再套一层 `<scene>/` |
| `SKY_SEG` | `0` | `1` 启用天空分割，需 `skyseg.onnx` |
| `OPACITY_THRESH` | YAML `0.05` | 导出时的最小不透明度 |
| `NO_DEPTH_RGB` | `0` | `1` 不导出逐视角 depth/rgb；03 评测需要 rgb，勿开启 |
| `EXTRA_ARGS` | 空 | 透传官方参数，例如 `--no_cam_align` |
| `PROFILE` | `1` | 每 0.5 秒采样整卡显存；不是模型独占显存 |

同名 `<OUT_ROOT>/<scene>/` 已存在时，02 会给 **OUT_ROOT** 加时间戳，不覆盖旧结果；后续 03/08 要读取终端打印的实际路径，不能假定仍在原目录。

### 采集转换与评测

| var | default | note |
|---|---|---|
| `CAPTURE` / `SCENE` | 必填 / 采集包目录名 | 07 的输入与输出场景名 |
| `N_VIEWS` / `FRAMES` | `24` / 空 | 等间隔抽 2–50 帧 / 显式原始帧号，后者优先 |
| `OUT_DIR` | `$SPLATANYTHING_RESULTS_DIR/_scenes/$SCENE` | 07 的输出场景目录 |
| `MAX_SIDE` / `NO_UNDISTORT` | `0` / `0` | 原图长边上限；`1` 跳过去畸变仅用于调试 |
| `RESULT` | 03/08/11 必填 | 所读的 PLY 结果目录；06 实际必填的是 `INPUT` |
| `OUT` | 各脚本默认子目录 | 评测/往返/视频输出，避免覆盖已有证据 |
| `SOURCE_SCENE` / `ALL_SCENE` | 08 必填 | 推理源子集 / 含留出视角的全集 |
| `EPS2D` | `0.01` | 渲染屏幕空间膨胀 |
| `MAX_VIEWS` / `MAX_HELD_OUT` | `0` / `0` | 03/08 最多评几个视角，0 为全部 |
| `CHUNK` | `4` | 06/08 每批渲染视角数 |
| `GLM_DIR` | `~/repos/glm_src` | 须含 `glm/` 子目录；服务器示例为 `../glm` |
| `GCC_DIR` | 本机 v2 env | 含 `gcc/g++` 的目录；有系统编译器用 `/usr/bin` |
| `TORCH_CUDA_ARCH_LIST` | 按 GPU 推断 | 避免 JIT 编译所有架构 |

`configs/inference.yaml` 只接受固定字段；`spacing_target` 与 `decode_chunk_size` 是代码常量，不能靠在 YAML 添加字段修改。

## 可选：训练成本探针（09）

```bash
# 只做可导前向等价性检查，不进行官方训练。
GPU=0 CONDA_ENV=splatanything \
  SPLATANYTHING_DIR=../SplatAnything \
  SPLATANYTHING_MODEL_DIR=../../model/splatanything \
  SPLATANYTHING_RESULTS_DIR=../splatanything_results \
  SCENE=../splatanything_results/_scenes/object_3v/images \
  RES=504 POSE=0 GLM_DIR=../glm GCC_DIR=/usr/bin \
  bash splatanything/09_run_train_probe.sh --stages fwd \
    --out ../splatanything_results/train_probe_fwd.json
```

09 冻结 encoder、深度、相机与 means，只开放两个高斯属性头；`steps` / `all` 是本地损失、优化器与 checkpoint 的研究性探针。不要把输出称作官方 stage-1/2 训练结果。

## 可选：LGTM 式局部纹理小试（35，研究脚本）

`35_lgtm_texture.py` 借用 LGTM 的源图投影局部纹理、textured 2DGS 与可学习 RGB 残差。两臂都在 **SA 体素合并前**保留逐源图元、冻结几何和透明度，仅压平最小厚度轴、保留两条切向尺度。`lgtm` 使用 K=4 纹理，`plain` 使用每图元一个 RGB 残差；二者参数量不同，不是等参数消融。**这不是 LGTM 官方跨场景训练、DepthSplat 权重实验或 4K 复现，也未通过视觉验收，勿用于人物主线。**

### 黄区普通服务器：安装与运行

下面只给黄区容器的相对路径示例；`LGTM_ENV` 是专用环境前缀，可按实际落点替换，**不是蓝区 H100 的历史路径**。先完成黄区 SA 环境准备，另建 LGTM 环境，避免覆盖原 SA 的 `gsplat`。patched gsplat 应在有 GPU、nvcc 和编译器的节点安装，官方 LGTM 仓保持只读。

```bash
# 在 media_code 目录执行。
git clone https://github.com/apple-aiml-research/ml-lgtm ../ml-lgtm
LGTM_ENV="$HOME/miniconda3/envs/splatanything_lgtm"
conda create -p "$LGTM_ENV" python=3.11 -y
"$LGTM_ENV/bin/pip" install torch==2.7.1 torchvision
"$LGTM_ENV/bin/pip" install -r ../SplatAnything/requirements.txt
"$LGTM_ENV/bin/pip" install --no-build-isolation --no-deps -e ../ml-lgtm/extern/gsplat
"$LGTM_ENV/bin/pip" install jaxtyping einops scipy pillow

# 黄区运行命令：不依赖蓝区 Slurm、H100 账号或历史作业。
SPLATANYTHING_DIR=../SplatAnything \
SPLATANYTHING_MODEL_DIR=../../model/splatanything \
LGTM_DIR=../ml-lgtm \
"$LGTM_ENV/bin/python" splatanything/35_lgtm_texture.py \
  --scene ../SplatAnything/examples/object --pose-mode --res 252 \
  --n-source 3 --texture-size 4 --steps 40 \
  --out ../splatanything_results/lgtm40_object
```

输出 `report.json`、`smoke.json`、双臂 checkpoint、逐视角 GT/before/after/diff 和 alpha。先确认 `smoke.projection_center_identity.passed=true` 及 `initial_consistency` 两差为零，再读训练曲线；**数值门通过不代表画质通过**。首次在新的官方 LGTM commit 上复跑前须核对纹理接口；本次使用 commit `587c121`，Python 3.11 + torch 2.7.1+cu126。不要加载 `gsplat_rayfix` 实验副本。

### 蓝区 H100

蓝区的专用路径、已建环境、Slurm 复跑方式及本次实验产物见 [README_H100.md](README_H100.md)，不要在黄区命令中替换几个路径就当蓝区运行。

正式报告、读数和实图结论见 [EXPERIMENTS.md](EXPERIMENTS.md)，投影与半像素约定见 [NOTES.md](NOTES.md)。

## 目录布局

```
<code-dir>/
├── media_code/splatanything/    # 本目录：编排脚本与文档
├── SplatAnything/               # 官方仓，00 稀疏 clone
├── glm/                         # 渲染所需 glm 头文件
├── splatanything_results/       # PLY、相机、深度、评测与视频
└── ../model/splatanything/      # 共享权重，即仓内看到的 ../../model/splatanything
```

## Notes

- 本机 RTX 3090 上，官方物体 23 视角、756² 的整卡显存采样约 23.5 GB；这是 WSL 单例参考，**不是黄区或蓝区服务器各 GPU 的保证**。遇到显存压力先降到 504²，并减小评测 `CHUNK`。
- 03/06/08/09/11 首次运行会安装 `gsplat==1.5.2` 并 JIT 编译；需要匹配 PyTorch 的 nvcc、gcc/g++ 与 glm。无 GPU 的登录节点不要执行这些脚本。
- 官方 `tests/test_inference.py` 依赖发布包未包含的 `validation.py`；00 只运行自包含相机测试。官方 `examples/object/prior_camera.json` 的 `num_cameras` 与实际图片数不一致，加载按真实图片条目处理。
- 人脸几何、三图/五图、双影和深度尺度属于研究实验，输入与验收命令见 WSL 手册和实验记录；本黄区文档不绑定本机 D 盘数据，也不把测量通过写成几何分支已裁决。
- 官方代码与权重遵循各自许可证；本目录只负责编排。`proxy.env` 含本机配置/凭据，已在 gitignore 中，不要提交。
