# Deformable-3D-Gaussians runner

在 Ubuntu + NVIDIA 服务器上跑 [Deformable-3D-Gaussians](https://github.com/ingra14m/Deformable-3D-Gaussians)（CVPR 2024，单目动态场景重建 / Deformable 3DGS）的**训练 / 渲染推理 / 定量评测**。本目录只含编排脚本——官方代码自动 clone（含两个 CUDA 子模块）、D-NeRF 数据集从 GitHub release 下载。

> 这是**训练型**方法：没有发布预训练权重，每个场景都要从头训出一套高斯点云（即「模型」）。所以「模型下载脚本」下载的是复现数据集（调整版 D-NeRF），「推理脚本」是渲染训好的高斯 + 算 PSNR/SSIM/LPIPS。

## 常用命令

> 假设已进入容器并 `conda activate deformable_gaussians`；路径取各脚本默认值（可改）；`GPU=0` 按需换卡。首次跑前先做下方「首次准备」。

```bash
# ── 一键流水线(run_all) ── clone官方仓(+子模块) → 装依赖+编CUDA → 下数据 → 训hook(7000步) → 渲染+评测
GPU=0 bash deformable_gaussians/run_all.sh
# 复现论文指标(训满 40000 步；时间更长但贴近论文 PSNR)
GPU=0 ITERATIONS=40000 bash deformable_gaussians/run_all.sh
# 换场景(lego 用了作者调整的 val-as-test；其余 7 个是标准 D-NeRF)
GPU=0 SCENE=lego bash deformable_gaussians/run_all.sh
# 6DoF 变换变体(指标略高、训推更慢)
GPU=0 IS_6DOF=1 bash deformable_gaussians/run_all.sh

# ── 只训练(train.py) ── D-NeRF(Blender 合成，需 --is_blender)
GPU=0 SCENE=hook bash deformable_gaussians/run_all.sh   # 走 run_all 的训练段
# 直接调官方 train.py(不经过 run_all；-s 数据路径 -m 输出路径)
cd ../Deformable-3D-Gaussians
python train.py -s ../../model/deformable-3d-gaussians/data/D-NeRF/hook -m output/hook --eval --is_blender
# NeRF-DS / HyperNeRF(真实世界，不传 --is_blender，20000 步)
python train.py -s /path/to/nerf-ds/scene -m output/scene --eval --iterations 20000

# ── 只渲染+评测(02，对已训好的场景) ──
GPU=0 MODEL_PATH=../Deformable-3D-Gaussians/output/hook bash deformable_gaussians/02_run_inference.sh
# 换渲染模式：render=全部测试图 | time=D-NeRF时间插值(出video.mp4) | all=时间+视角 | view=视角游走 | original=真实数据时间+视角
GPU=0 MODEL_PATH=../Deformable-3D-Gaussians/output/hook MODE=time bash deformable_gaussians/02_run_inference.sh
# 渲染指定 checkpoint(--iteration，默认 -1=最新)
GPU=0 MODEL_PATH=../Deformable-3D-Gaussians/output/hook ITERATION=10000 bash deformable_gaussians/02_run_inference.sh
# 只渲染不算指标(跳过 metrics.py)
GPU=0 MODEL_PATH=../Deformable-3D-Gaussians/output/hook RUN_METRICS=0 bash deformable_gaussians/02_run_inference.sh

# ── 只下数据集(01) ──
bash deformable_gaussians/01_download_models.sh   # -> $MODEL_DIR/data/D-NeRF/<scene>/
```

- 结果：训练 → `../Deformable-3D-Gaussians/output/<scene>/point_cloud/iteration_<N>/point_cloud.ply`（高斯点云）+ `cfg_args`（render/metrics 读它恢复 `--is_blender/--is_6dof/--source_path`）；渲染 → `output/<scene>/test/ours_<N>/{renders,gt,depth}/`；评测 → `output/<scene>/test/results.json`（PSNR/SSIM/LPIPS）。
- D-NeRF 共 8 个场景：`bouncing hell hook jump lego mutant standup trex`。`lego` 是作者调整过的（用 val 当 test + 把 val 第一帧加进 train，因原版 lego 的 train/test 不一致——铲子翻转角度不同）。

## 首次准备
```bash
cd <your-code-dir>            # e.g. /data_3d/<uid>/code
git -c http.sslVerify=false clone https://github.com/CrescentVelvet/media_code.git
cd media_code && cp proxy.env.example proxy.env   # 填 http_proxy / https_proxy（公司代理用；自用网络可跳过）
conda create -n deformable_gaussians python=3.7 -y && conda activate deformable_gaussians
# torch 必须是 CUDA 版，且 CUDA 大版本要和后面编译 rasterizer 用的 CUDA toolkit 一致（cu116 → 11.6）
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
INSTALL_DEPS=1 BUILD_CUDA=1 bash deformable_gaussians/00_setup_env.sh   # 装 requirements.txt + 编两个 CUDA 子模块
bash deformable_gaussians/01_download_models.sh                        # 下 D-NeRF 数据集(~258MB)
```
⚠️ Deformable-GS 的 `torch==1.13.1+cu116`（python 3.7）版本 pin 与本仓其他算法冲突，务必用专用 env（`CONDA_ENV=deformable_gaussians`），别装进共享的 `doll`。

或一键（clone + 装依赖 + 编 CUDA + 下数据 + 训 hook + 渲染评测）：
```bash
GPU=0 bash deformable_gaussians/run_all.sh
```

---

以下为详细参考（流程原理 / 各脚本参数 / 排错 / 目录布局）。

## Inference (02 — 更多用法)
`02_run_inference.sh` 调官方 `render.py` + `metrics.py`（不改官方文件）。`render.py` 用 `get_combined_args` 读 `<model_path>/cfg_args`（train.py 训练时写入）恢复 `--is_blender / --is_6dof / --source_path`，所以只需传 `-m`。`metrics.py` 只对 **test** split 算指标。手放了 checkpoint 没 `cfg_args`？用 `EXTRA_RENDER_ARGS="--is_blender --source_path /data/D-NeRF/hook"` 补。

```bash
# 渲染 train+test 两套(默认只渲 test，SKIP_TRAIN=1 省时；想看 train 也渲就 SKIP_TRAIN=0)
GPU=0 MODEL_PATH=../Deformable-3D-Gaussians/output/hook SKIP_TRAIN=0 bash deformable_gaussians/02_run_inference.sh
# 渲染 NeRF-DS 真实数据的时间+视角(出 video.mp4)：
GPU=0 MODEL_PATH=../Deformable-3D-Gaussians/output/as MODE=original bash deformable_gaussians/02_run_inference.sh
```

## Pipeline（方法原理）
对应官方代码 `scene/deform_model.py` + `gaussian_renderer/__init__.py` + `train.py`。一个动态场景从训练到渲染：

1. **初始化点云** — `Scene` 识别场景类型（D-NeRF 看 `transforms_train.json` → Blender 分支；NeRF-DS 看 `sparse/` → COLMAP；HyperNeRF 看 `dataset.json` → nerfies 等）。用 `simple_knn`（CUDA 子模块）从首帧深度/点云建初始 3D 高斯。D-NeRF 是 Blender 合成数据，带相机内外参 + `--is_blender`。
2. **变形网络（DeformModel）** — 核心创新：一套 MLP 把「规范空间(static)的高斯」映射到「时间 t 的变形位置/旋转/缩放」。每步给每个高斯一个时间输入 `fid`（归一化帧号），MLP 输出 `d_xyz, d_rotation, d_scaling`。前 `warm_up=3000` 步变形量为 0（先让 static 稳定），之后才开变形；真实数据还加一个小 `ast_noise`（高斯噪声）增广时间输入提升泛化。6DoF 变体用 6 自由度变换替代 3 变量平移（指标略高、更慢）。
3. **可微光栅化** — `diff_gaussian_rasterization`（depth+alpha 变体，`filter-norm` 分支，CUDA 子模块）把变形后的高斯 splat 成图像 + depth。可微，所以能反向传播。
4. **损失** — `(1-λ)·L1 + λ·(1-SSIM)`（`λ_dssim=0.2`），无 GAN。和 vanilla 3D-GS 一致，区别只在前置了一个变形网络。
5. **致密化+剪枝** — 训练前 15000 步按梯度阈值（`densify_grad_threshold=0.0007`）周期性（每 100 步）分裂/克隆大梯度高斯、剪掉透明的；每 3000 步重置不透明度。这就是为什么 D-NeRF hook 能长到 ~50k 高斯。
6. **渲染+评测** — `render.py`：对 test 相机，每个高斯取其时间 `fid` → 变形网络出变形 → 光栅化。`mode=render` 渲全部 test 图；`time/all/view` 做时间/视角插值出 `video.mp4`（D-NeRF 专属；真实数据用 `original`）。`metrics.py` 对 test renders vs GT 算 PSNR/SSIM/LPIPS(VGG)。

> 一句话：**初始高斯 → 每帧经变形 MLP 平移/旋转/缩放 → 光栅化 → L1+SSIM 反传 + 致密化 → 训完渲染 test 帧算指标**。变形网络让一套高斯表达整段动态视频，是论文标题里 "Deformable" 的含义。

## Train（训练参数）
`train.py` 主参数（`OptimizationParams` 默认值，均可 env 覆盖 run_all）：
| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--source_path` / `-s` | _(必填)_ | 数据集场景路径（D-NeRF: `.../D-NeRF/hook`） |
| `--model_path` / `-m` | _(必填)_ | 输出目录（run_all 默认 `output/<scene>`） |
| `--eval` | off | 划分 train/test（评测必需；run_all 默认开） |
| `--is_blender` | off | D-NeRF 合成数据开；NeRF-DS/HyperNeRF 关（run_all: `IS_BLENDER=1`） |
| `--is_6dof` | off | 6DoF 变换变体（run_all: `IS_6DOF=1`） |
| `--iterations` | 40000 | D-NeRF 默认 40000；NeRF-DS/HyperNeRF 用 20000（run_all 默认 7000 做快速 demo） |
| `--test_iterations` | 5000,6000,7000,10000-40000(每1000) | 训练中评测的步 |
| `--save_iterations` | 7000,10000,20000,30000,40000 | 存 point_cloud 的步；末步必存 |

- 训练产物：`<model_path>/point_cloud/iteration_<N>/point_cloud.ply`、`<model_path>/cfg_args`、变形权重 `<model_path>/deform/`（`deform.save_weights`）、`<model_path>/input.ply` + `cameras.json`。
- TensorBoard 看曲线：`tensorboard --logdir ../Deformable-3D-Gaussians/output --port 6006`。

## Datasets（数据集）
`01_download_models.sh` 下的是 **Deformable-GS 调整版 D-NeRF**（GitHub release `v0.1-pre-released` / `D-NeRF-Deformable-GS.zip`，~258MB，公开免 token）：
- 解压到 `$MODEL_DIR/data/D-NeRF/<scene>/`，每场景含 `transforms_train.json` + `transforms_test.json` + `images/`（Blender 格式，400×400）。
- 8 场景：`bouncing hell hook jump lego mutant standup trex`。`lego` 用 val 当 test（作者调整，见 release 说明）。
- 来源是 GitHub **release asset**（不是 HuggingFace），下载会 301 到 `objects.githubusercontent.com`；公司代理下若 SSL 报错，脚本自动 `--insecure` 重试，或 `DL_DISABLE_SSL=1` 强制。

**NeRF-DS / HyperNeRF（真实世界）不自动下**：无脚本化、许可清晰的公开镜像。手动放好再训（`01` 只下 D-NeRF，用 `SKIP_DATA=1` 跳过它；`DNERF_DIR` 指你的数据根，`SCENE` 指场景子目录）：
```bash
# NeRF-DS: https://jokeryan.github.io/projects/nerf-ds/  -> 放到 <model>/deformable-3d-gaussians/data/NeRF-DS/<as|basin|...>/
# HyperNeRF: https://hypernerf.github.io/               -> 放到 <model>/deformable-3d-gaussians/data/HyperNeRF/{interp,misc,vrig}/<scene>/
# 路径换成你机器上实际的；NeRF-DS 是 COLMAP 格式，不传 --is_blender，20000 步
GPU=0 SCENE=as DNERF_DIR=/abs/path/to/NeRF-DS IS_BLENDER=0 ITERATIONS=20000 SKIP_DATA=1 \
  bash deformable_gaussians/run_all.sh
# 或直接调官方 train.py（不经 run_all）：
cd ../Deformable-3D-Gaussians
python train.py -s /abs/path/to/nerf-ds/scene -m output/scene --eval --iterations 20000
```

## 可能遇到的问题

公司代理做 HTTPS 中间人解密，下面按流水线阶段列出常见报错与修法（命令在服务器上、`deformable_gaussians` 环境已激活时执行）。

**1. clone 官方仓 / 子模块报 SSL / 认证**
公开仓免认证，加 `-c http.sslVerify=false`（`run_all.sh` 克隆失败会自动带它重试）。注意 `simple-knn` 子模块在 `gitlab.inria.fr`，递归克隆失败时手动补：
```bash
cd ../Deformable-3D-Gaussians
git -c http.sslVerify=false submodule update --init --recursive
```
git 连不上（非 SSL）就设全局代理（密码特殊字符必须 URL 编码：`*`→`%2A`、`+`→`%2B`、`@`→`%40`）：
```bash
git config --global https.proxy http://USER:PASS@proxyhk.huawei.com:8080
git config --global  http.proxy http://USER:PASS@proxyhk.huawei.com:8080
```

**2. `BUILD_CUDA` 编 `diff-gaussian-rasterization` 报 `nvcc not found` / `CUDA_HOME` 错**
rasterizer 的 `setup.py` 用 `torch.utils.cpp_extension`，需要 `nvcc`（CUDA toolkit）。且 toolkit 大版本要和 torch 的 CUDA 版本对齐：
- `torch==1.13.1+cu116` → 装 CUDA 11.6 toolkit，`export CUDA_HOME=/usr/local/cuda-11.6`（或装好后的默认 `/usr/local/cuda`）；
- 验证：`$CUDA_HOME/bin/nvcc --version`；
- 版本不匹配（如 torch cu116 但系统只有 CUDA 12.x toolkit）会编不过或运行时段错。重装匹配的 torch 或装对应 toolkit。

**3. `BUILD_CUDA` 编译报 `error: no member named '...' in 'at::...'` / ABI 不匹配**
torch 版本和 rasterizer 代码不兼容。Deformable-GS 的 `filter-norm` 分支按 torch 1.13 写。用了更新的 torch(2.x) 会遇到 ATen API 变动。修法：用官方 pin `torch==1.13.1+cu116`（python 3.7）；或 fork 已适配新 torch 的 rasterizer（自行替换 `submodules/depth-diff-gaussian-rasterization`）。

**4. `pip install -r requirements.txt` 报 SSL / 超时**
```bash
pip config set global.trusted-host "pypi.org pypi.python.org files.pythonhosted.org download.pytorch.org"
INSTALL_DEPS=1 bash deformable_gaussians/00_setup_env.sh
```
`00` 已把 `submodules/` 行过滤掉（它们走 `BUILD_CUDA` 单独编），避免缺 toolkit 时整条 install 崩。

**5. `01` 下数据报 SSL / 连不上 GitHub release**
release asset 下载会 301 到 `objects.githubusercontent.com`，公司代理 MITM 证书可能不被信任。`_env.sh` 已把 CA bundle 给 curl；仍失败：
```bash
DL_DISABLE_SSL=1 bash deformable_gaussians/01_download_models.sh    # 强制 --insecure
```
或手动从 `https://github.com/ingra14m/Deformable-3D-Gaussians/releases/download/v0.1-pre-released/D-NeRF-Deformable-GS.zip` 下载，放到 `$MODEL_DIR/data/.dnerf_stage/D-NeRF-Deformable-GS.zip` 再重跑 `01`（会自动续传解压）。

**6. 推理(02)报 `Could not recognize scene type!` / 找不到 cfg_args**
`render.py` 读 `<model_path>/cfg_args` 恢复参数；没 train 过或手放的 checkpoint 缺这个文件。修法：要么先 train（run_all / train.py 会写 cfg_args）；要么手补参数：
```bash
EXTRA_RENDER_ARGS="--is_blender --source_path /abs/path/to/D-NeRF/hook" \
  MODEL_PATH=../Deformable-3D-Gaussians/output/hook GPU=0 bash deformable_gaussians/02_run_inference.sh
```

**7. 推理(02)报 `No module named 'diff_gaussian_rasterization'`**
CUDA 子模块没编。`BUILD_CUDA=1 bash deformable_gaussians/00_setup_env.sh`（见排错 #2/#3）。

**8. 训练 / 推理 OOM（显存不足）**
D-NeRF 400×400 单场景显存占用不高（hook ~50k 高斯，几 GB），但致密化后期会涨。降 `--iterations`（提前停）、或换小场景（`hell` 仅 ~16k 高斯）。`render.py` 默认渲全 test；只算指标可 `SKIP_TRAIN=1`（默认即是）。仍紧张 `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

**9. `metrics.py` 报 LPIPS 下载失败 / `lpips` 初始化卡住**
`lpips.LPIPS(net='vgg')` 首次从作者 URL 下 VGG 权重（走 `torch.hub`）。代理失败：先 `bash hypir/setup_ca_bundle.sh`（或任何等价建包），或手动把 `vgg.pth` 放进 `~/.cache/torch/hub/checkpoints/`。`02` 在 metrics 失败时不影响已渲好的图（renders 仍在）。

**10. 跑 `.sh` 报 `syntax error near unexpected token ('`（CRLF 行尾）**
脚本被 CRLF 污染（Windows→服务器非 git 方式同步带过去）。修法：
```bash
sed -i 's/\r$//' deformable_gaussians/*.sh
git checkout -- deformable_gaussians/*.sh   # git 同步的：.gitattributes 还原 LF
```
预防：用 `git pull` 同步（本仓 `.gitattributes` 强制 LF）。

> 通用：`proxy.env`（代理凭证）在仓内 gitignored，`~/.ca-bundle.crt` 在家目录，都不入库；切勿把凭证写进脚本。

## Config (env vars, all optional)
| var | default | note |
|---|---|---|
| `CONDA_ENV` | `deformable_gaussians` | conda env（专用，torch pin 与其他算法冲突） |
| `GPU` | _(unset)_ | physical GPU id to pin, e.g. `GPU=0`; remaps `CUDA_VISIBLE_DEVICES` |
| `DG_DIR` | `../Deformable-3D-Gaussians` | official code path |
| `MODEL_DIR` | `../../model/deformable-3d-gaussians` | weights/data path |
| `DG_REPO` | official GitHub URL | clone source (run_all) |
| `INSTALL_DEPS` | `0` (run_all: `1`) | `1` = install requirements.txt in `00` |
| `BUILD_CUDA` | `0` (run_all: `1`) | `1` = build simple-knn + depth-diff-gaussian-rasterization |
| `CUDA_HOME` | `/usr/local/cuda` | CUDA toolkit root (must match torch's cu major, e.g. 11.6 for cu116) |
| `SKIP_TORCH` | `0` | `1` = filter torch pins (no-op here; requirements has no torch line) |

### Dataset (01)
| var | default | note |
|---|---|---|
| `DATA_DIR` | `$MODEL_DIR/data` | dataset root |
| `DNERF_DIR` | `$DATA_DIR/D-NeRF` | where D-NeRF scenes are unpacked |
| `RELEASE_TAG` | `v0.1-pre-released` | GitHub release tag |
| `ASSET_NAME` | `D-NeRF-Deformable-GS.zip` | release asset filename |
| `ZIP_URL` | _(derived)_ | full download URL (override for a mirror) |
| `DL_DISABLE_SSL` | `0` | `1` = curl `--insecure` |
| `PURGE` | `0` | `1` = remove the staging zip after unpack |

### Inference (02)
| var | default | note |
|---|---|---|
| `MODEL_PATH` | `$DG_DIR/output/$SCENE` | trained scene output dir (what `-m` points at) |
| `SCENE` | _(unset)_ | scene name; sets `MODEL_PATH` when `MODEL_PATH` unset |
| `ITERATION` | `-1` | -1 = latest saved point_cloud |
| `MODE` | `render` | `render` \| `time` \| `all` \| `view` \| `pose` \| `original` |
| `SKIP_TRAIN` | `1` | `1` = skip rendering the train split (metrics only need test) |
| `SKIP_TEST` | `0` | `1` = skip test split (then metrics are skipped) |
| `RUN_METRICS` | `1` | `0` = render only, skip `metrics.py` |
| `EXTRA_RENDER_ARGS` | _(unset)_ | forwarded to render.py (e.g. `--is_blender --source_path ...`) |

### Train (run_all)
| var | default | note |
|---|---|---|
| `SCENE` | `hook` | D-NeRF scene to train (`hook`/`lego`/`trex`/...); falls back to first available |
| `ITERATIONS` | `7000` | train steps; `40000` reproduces paper (D-NeRF default) |
| `IS_BLENDER` | `1` | `1` = pass `--is_blender` (D-NeRF); `0` for NeRF-DS/HyperNeRF |
| `IS_6DOF` | `0` | `1` = pass `--is_6dof` (6DoF transform variant) |
| `DG_OUTPUT_ROOT` | `$DG_DIR/output` | trained-model output root |
| `SOURCE_PATH` | `$DNERF_DIR/$SCENE` | `-s` value passed to train.py (override for non-D-NeRF data) |
| `DNERF_DIR` | `$MODEL_DIR/data/D-NeRF` | data root; set to your dataset dir for NeRF-DS/HyperNeRF |
| `SKIP_DATA` | `0` | `1` = skip `01` D-NeRF download (you placed data under `DNERF_DIR`) |

## Outputs
- **01 dataset**: `$MODEL_DIR/data/D-NeRF/<scene>/`（`transforms_train.json` + `transforms_test.json` + `images/`）。
- **train**: `$DG_DIR/output/<scene>/point_cloud/iteration_<N>/point_cloud.ply`（高斯点云）+ `cfg_args` + `deform/`（变形 MLP 权重）+ `input.ply` + `cameras.json` + TensorBoard events。
- **02 render+metrics**: `$MODEL_PATH/test/ours_<iter>/{renders,gt,depth}/*.png`（test 渲染 + GT + 深度）+ `$MODEL_PATH/test/results.json`（PSNR/SSIM/LPIPS）。`MODE=time/all/view` 还出 `interpolate_*/renders/video.mp4`。

## 目录布局
```
<code-dir>/
├── media_code/                       # 本仓
│   ├── proxy.env                     # 代理 + 覆盖项, gitignored
│   └── deformable_gaussians/         # 编排脚本(本目录)
├── Deformable-3D-Gaussians/          # 官方代码(自动 clone 到 ../Deformable-3D-Gaussians, 含 submodules/)
│   └── submodules/
│       ├── simple-knn/                       # gitlab.inria.fr/bkerbl/simple-knn (CUDA)
│       └── depth-diff-gaussian-rasterization # ingra14m/diff-gaussian-rasterization-extentions @ filter-norm (CUDA)
└── ../../model/deformable-3d-gaussians/       # 权重/数据(在 <code-dir> 上一级, 各算法共享)
    └── data/D-NeRF/<scene>/          # 01 下载的调整版 D-NeRF
```
默认：官方代码 `../Deformable-3D-Gaussians`、数据 `../../model/deformable-3d-gaussians/data`、训练输出 `$DG_DIR/output/<scene>`（相对本目录）；用 `DG_DIR` / `MODEL_DIR` / `DG_OUTPUT_ROOT` 覆盖。复用现有 conda env（默认 `deformable_gaussians`），但 Deformable-GS 的 `torch==1.13.1+cu116` pin 与其他算法冲突——建议专用 env。

## Notes
- Official code & dataset follow their own license (Deformable-3D-Gaussians = [non-commercial research/eval](https://github.com/ingra14m/Deformable-3D-Gaussians/blob/main/LICENSE.md)). This folder only orchestrates; no official code is copied.
- `.gitattributes` (repo root) forces LF so Windows-pushed scripts run cleanly on Ubuntu.
- `proxy.env` (proxy creds / path / env overrides) is gitignored — never committed. Don't put credentials in scripts.
- SSL behind a TLS-intercepting corporate proxy: pip uses `PIP_CERT`/`--trusted-host`; `git` uses `GIT_SSL_CAINFO`; `curl` uses `CURL_CA_BUNDLE` (`_env.sh` prefers `~/.ca-bundle.crt`, built by `hypir/setup_ca_bundle.sh` or equivalent).
- No pretrained weights exist — every scene is trained from scratch; `01` downloads the D-NeRF *dataset* (the reproduction input), `02` renders + scores the *trained* gaussians.
