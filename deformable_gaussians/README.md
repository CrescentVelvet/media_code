# Deformable-3D-Gaussians runner

在 Ubuntu + NVIDIA 服务器上跑 [Deformable-3D-Gaussians](https://github.com/ingra14m/Deformable-3D-Gaussians)（CVPR 2024，单目动态场景重建 / Deformable 3DGS）的**训练 / 渲染推理 / 定量评测**。本目录只含编排脚本——官方代码自动 clone（含两个 CUDA 子模块）、D-NeRF 数据集从 GitHub release 下载。

> 这是**训练型**方法：没有发布预训练权重，每个场景都要从头训出一套高斯点云（即「模型」）。所以「模型下载脚本」下载的是复现数据集（调整版 D-NeRF），「推理脚本」是渲染训好的高斯 + 算 PSNR/SSIM/LPIPS。

本仓支持**两条路径**（同一个 env / 官方代码 / CUDA 扩展，互不干扰）：
- **路径 A：D-NeRF 复现**（论文 benchmark）— `00 装 env → 01 下 D-NeRF → 02 渲染+评测`，`run_all.sh` 一键。`hook/lego/trex/...` 8 个合成场景，`--is_blender`，40000 步贴近论文 PSNR。
- **路径 B：真实拍摄人体序列**（NeRF-DS 设定，解决人物微动）— `03 COLMAP 位姿 → 04 训练 → 05 渲染+评测`。摄入一组多视角图像（人物保持静止时有微动——呼吸/衣摆/姿势漂移），COLMAP 估位姿，Deformable-GS 的 deformation MLP 把微动从 canonical 高斯分离 → 解决静态方法（2DGS/GOF）重建出现的"鬼影/拖影"。**⚠️ 不要把 Wan2.2 生成的旋转视频喂给 03**——详细原因见下方「为什么用真实拍摄序列，而非 Wan2.2 生成的旋转视频」小节。

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

### 路径 B：真实拍摄人体序列（03 COLMAP → 04 训练 → 05 渲染）
> 适合摄入真实多视角图像（人物有微动）。摄入 Wan2.2 旋转视频**不合适**——原因见下方「为什么用真实拍摄序列，而非 Wan2.2 生成的旋转视频」小节。

```bash
# ── 3) COLMAP 位姿估计（拍摄图像序列 → NeRF-DS COLMAP 格式） ──
#    输入: INPUT_DIR/image/*.jpg (跟 wan22_rotate INPUT_DIR 同模式)
#    ⚠️ 拍摄时按时间顺序命名图像 (如 frame_0001.jpg, frame_0002.jpg, ...)
#       03 会按字典序排序重命名为 00000.jpg, 00001.jpg, ...
#       (Deformable-GS dataset_readers 要求文件名是纯数字 → fid = int(name)/(N-1))
#    需要 colmap 二进制 (apt install colmap)
GPU=0 INPUT_DIR=../Reconstruction/dataset/B003_Human_Data_w_pose/test_task_id_3a8b3cc746304f49b9e3275e36aa9374 \
  SCENE_NAME=alice \
  bash deformable_gaussians/03_colmap_pose.sh
# 输出: $MODEL_DIR/data/real/alice/{images/, sparse/0/{cameras,images,points3D}.bin}
#       (mapper 重建失败的视角会被丢弃; KEEP_DISTORTED=1 保留临时目录排查)

# ── 4) 训练 deformation MLP + canonical 高斯（NeRF-DS 模式, 默认 20000 步） ──
GPU=0 SCENE_NAME=alice \
  bash deformable_gaussians/04_train_real.sh
# 可选: WHITE_BG=1 (输入是白底分割图时开; NeRF-DS 原图关)
# 可选: ITERATIONS=40000 (NeRF-DS 标配是 20000, 想多训可加)
# 可选: IS_6DOF=1 (6DoF 变体, 指标略高更慢)
# 输出: $DG_DIR/output/real_alice/{point_cloud/iteration_<N>/point_cloud.ply, deform/, cfg_args, cameras.json}

# ── 5) 渲染 + 评测（封装 02, 默认 MODE=original 出 video.mp4） ──
GPU=0 SCENE_NAME=alice \
  bash deformable_gaussians/05_render_real.sh
# 可选: MODE=render   只渲 test 视角 PNG (仅算指标, 不出 video)
# 可选: ITERATION=10000 渲指定 checkpoint
# 可选: RUN_METRICS=0  跳过 metrics.py
# 输出: $DG_DIR/output/real_alice/test/ours_<N>/{renders,gt,depth}/*.png
#       $DG_DIR/output/real_alice/test/interpolate_original_<N>/renders/video.mp4 (看微动效果)
#       $DG_DIR/output/real_alice/test/results.json (PSNR/SSIM/LPIPS)

# ── 路径 B 一键（不带 run_all.sh; 假设 env 已建好, 跳过 D-NeRF 下载） ──
GPU=0 INPUT_DIR=.../human_subject SCENE_NAME=alice \
  bash deformable_gaussians/03_colmap_pose.sh && \
GPU=0 SCENE_NAME=alice bash deformable_gaussians/04_train_real.sh && \
GPU=0 SCENE_NAME=alice bash deformable_gaussians/05_render_real.sh
```

- 结果：训练 → `../Deformable-3D-Gaussians/output/<scene>/point_cloud/iteration_<N>/point_cloud.ply`（高斯点云）+ `cfg_args`（render/metrics 读它恢复 `--is_blender/--is_6dof/--source_path`）；渲染 → `output/<scene>/test/ours_<N>/{renders,gt,depth}/`；评测 → `output/<scene>/test/results.json`（PSNR/SSIM/LPIPS）。
- D-NeRF 共 8 个场景：`bouncing hell hook jump lego mutant standup trex`。`lego` 是作者调整过的（用 val 当 test + 把 val 第一帧加进 train，因原版 lego 的 train/test 不一致——铲子翻转角度不同）。
- 路径 B 训练产物在 `output/real_<scene>/`（与 D-NeRF 复现的 `output/<scene>/` 分开，互不覆盖）。

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

## 路径 B：真实拍摄人体序列（03 COLMAP → 04 训练 → 05 渲染）
对应官方代码 `scene/__init__.py:45` 的 COLMAP 分支 + `scene/dataset_readers.py` 的 `readColmapSceneInfo`。

**适用场景**：手持相机环绕人物拍摄的多视角图像序列，人物保持静止时仍有真实微动（呼吸/衣摆/姿势漂移）。等价于 NeRF-DS 设定。deformation MLP 把微动从 canonical 高斯分离 → 解决静态方法（2DGS/GOF）的"鬼影/拖影"。

### 为什么用真实拍摄序列，而非 Wan2.2 生成的旋转视频

Deformable-GS 的核心是一个 **deformation MLP**，输入是归一化帧号 `fid`，输出每个高斯在该时刻的 `d_xyz / d_rotation / d_scaling`。这个 MLP 只有在「帧间物体真的在动」时才有意义——它学的是**真实物理形变随时间的演化**。所以选择输入数据时，关键看「帧间的"动"是什么来源」：

| 维度 | 真实拍摄序列 ✅ | Wan2.2 旋转视频 ❌ |
|---|---|---|
| **场景类型** | multi-view dynamic（轻微动态）| multi-view static（理想）/ noisy（实际）|
| **帧间"动"的来源** | 真实物理形变（呼吸、衣摆、姿势漂移）| 视频生成模型的时序不一致噪声（衣服抖、面部漂、姿态漂）|
| **deformation MLP 学到** | 真实形变 → 泛化好 | 生成噪声 → overfit 训练集，新视角全是伪影 |
| **canonical 高斯** | 干净（微动被分离出去）| 被伪动污染（deformation 把噪声"吸收"进静态场）|
| **重建新视角** | 能正确去除微动 → 干净渲染 | 把训练集噪声外推到新视角 → 拖影/扭曲 |
| **场景本质** | 人物不动，相机转 + 微动 | 人物不动，相机转（理想静态）|
| **静态方法够不够** | 不够（2DGS/GOF 会出现"鬼影/拖影"）| 够（2DGS/GOF 已是最佳）|

**核心矛盾**：Wan2.2 生成的旋转视频，理想情况下是「人物静止 + 相机环绕」的 **multi-view static scene**——这正是 2DGS/GOF 的标准设定（[wan22_rotate/05_3dgs_recon.sh](../wan22_rotate/05_3dgs_recon.sh) / [05a_3dgs_recon.sh](../wan22_rotate/05a_3dgs_recon.sh) 已经能完美处理）。把它喂给 Deformable-GS 等于把方法的强项（建模真实形变）用错了地方：

1. **理想静态情况下，deformation MLP 是冗余的**——人物 0 微动时 `d_xyz/d_rot/d_scale` 应该恒为 0，MLP 等价于一个空操作。这时 Deformable-GS 退化为 vanilla 3DGS，多训了一个 MLP 反而引入了过拟合风险。
2. **实际 Wan2.2 视频有"伪动"**——视频生成模型固有的时序不一致（同一像素在不同帧生成时不稳定）会造成衣服抖动、面部表情漂移、姿态微漂。这些不是真实物理形变，是生成噪声。
3. **deformation MLP 会去拟这种伪动噪声**——MLP 不会区分"真实形变"和"生成噪声"，它会把所有帧间差异都拟下来。训练 PSNR 可能很高（拟了训练集），但 canonical 高斯被噪声污染，**新视角渲染会出现拖影/扭曲**（把训练集的伪动外推到未见视角）。
4. **真实拍摄序列的微动是物理形变**——呼吸、衣摆、姿势漂移这些都有真实物理结构（衣物形变、肌肉形变、骨骼位姿），deformation MLP 学到的是这些物理形变的低维流形，能正确泛化到新视角（同一人物新角度的微动模式跟训练集一致）。

**结论**：Wan2.2 旋转视频走 [wan22_rotate/05_3dgs_recon.sh](../wan22_rotate/05_3dgs_recon.sh) 或 [05a](../wan22_rotate/05a_3dgs_recon.sh)（静态方法，2DGS/GOF）；真实拍摄人体序列走本目录的 03/04/05（动态方法，Deformable-GS）。两条 pipeline 服务不同场景，不要混用输入。

**不适用场景**：
- **Wan2.2 生成的旋转视频**——见上方对比表的详细分析。用 2DGS/GOF（[wan22_rotate/05_3dgs_recon.sh](../wan22_rotate/05_3dgs_recon.sh) / [05a](../wan22_rotate/05a_3dgs_recon.sh)）即可，Deformable-GS 是退化甚至有害的。
- **Pi3 位姿估计**——Pi3 是前馈位姿估计器，假设场景静态（同一物体在所有帧中位置不变）；动态场景上 Pi3 会把人物微动当噪声，位姿估计误差大。03 用 COLMAP SfM 通过 bundle adjustment 联合优化点云+相机位姿，对动态场景更鲁棒。
- **完全静止的拍摄**（人物 0 微动）——deformation MLP 学不到任何形变，等价于多此一举；用 2DGS/GOF 即可（更快、更稳）。

### Step 03 — COLMAP 位姿估计（`03_colmap_pose.sh`）
调系统 `colmap` 二进制跑 SfM（feature_extractor → exhaustive_matcher → mapper → image_undistorter），把 `INPUT_DIR/image/*.jpg` 转成 Deformable-GS 期望的 NeRF-DS COLMAP 格式：
```
$OUTPUT_SCENE/                              # 默认 $MODEL_DIR/data/real/<SCENE_NAME>/
  input/                                    # 临时: 重命名后的纯数字序号图像
    00000.jpg, 00001.jpg, ...
  distorted/                                 # 临时: feature_extractor + mapper 的中间产物
    database.db                              #   SIFT 特征 + 匹配
    sparse/0/                                #   mapper 输出的有畸变 SfM
  images/                                    # ✅ 最终输出: 去畸变图像
  sparse/0/
    cameras.bin                              # ✅ 相机内参 (PINHOLE / SIMPLE_PINHOLE)
    images.bin                               # ✅ 相机外参 (每帧 qvec + tvec)
    points3D.bin                             # ✅ 稀疏 3D 点 (初始化高斯用)
```

**⚠️ 关键约束：图像文件名必须是纯数字**。Deformable-GS `dataset_readers.py:136` 算 `fid = int(image_name)/(num_frames-1)`，文件名非纯数字会抛 `ValueError`。03 会按字典序排序输入图像，复制+重命名为 `00000.jpg, 00001.jpg, ...`。所以**拍摄时请按时间顺序命名图像**（如 `frame_0001.jpg, frame_0002.jpg, ...`），字典序就是时序，deformation MLP 才能正确学到随时间的形变。

**为什么不能用 Pi3 替代 03**：Pi3 是前馈位姿估计器，假设场景静态（同物体在所有帧中位置不变）；动态场景上 Pi3 会把人物微动当噪声，位姿估计误差大。COLMAP SfM 通过 bundle adjustment 联合优化点云+相机位姿，对动态场景更鲁棒（虽然 NeRF-DS 标准做法是先把人物当成静态来 SfM，再用 deformation field 处理残差微动）。

**与 wan22_rotate/05_3dgs_recon.sh 的关系**：05 用 Pi3 + COLMAP 导出格式（Pi3 估位姿再写成 cameras.txt/images.txt/points3D.txt）。03 不用 Pi3，直接跑真 COLMAP SfM——因为 Deformable-GS 训练数据要更准的位姿（动态场景），不能前馈估。两者输出格式一样（`sparse/0/*.bin` 或 `.txt`），可以互换数据但精度不同。

### Step 04 — 真实人体训练（`04_train_real.sh`）
`train.py -s $SOURCE_PATH -m $MODEL_PATH --eval --iterations 20000`（不带 `--is_blender`）。默认 NeRF-DS 模式 20000 步（vs D-NeRF 40000）。可选 `WHITE_BG=1`（输入是白底分割图时开，Deformable-GS `train.py` 支持 `--white_background` flag；NeRF-DS 原图关）。可选 `IS_6DOF=1`（6DoF 变体，指标略高更慢）。

训练产物在 `$DG_DIR/output/real_<scene>/`（与 D-NeRF 复现的 `output/<scene>/` 分开，互不覆盖）：
```
output/real_<scene>/
  point_cloud/iteration_<N>/point_cloud.ply     # 高斯点云 (变形 MLP 应用前)
  deform/                                         # 变形 MLP 权重 (deform.save_weights 输出)
  cfg_args                                        # render/metrics 读它恢复 is_blender/source_path
  input.ply                                       # 初始点云 (sparse/0/points3D 复制过来)
  cameras.json                                    # 所有相机参数 (train + test)
  events.out.tfevents.*                           # TensorBoard
```

### Step 05 — 渲染 + 评测（`05_render_real.sh`，薄封装 02）
默认 `MODE=original`（NeRF-DS 真实数据专用：时间+视角插值出 `video.mp4`，看人物微动的动态重建效果）。`MODE=render` 只渲所有 test 视角 PNG（不出 video，适合仅算指标）。

5 个渲染模式的区别（`render.py` `--mode`）：
| mode | 输出 | 用途 |
|---|---|---|
| `render` | `{renders,gt,depth}/*.png` | 渲所有 test 视角（算指标必需） |
| `time` | `interpolate_time/renders/video.mp4` | 同视角不同时刻（D-NeRF 时间插值） |
| `view` | `interpolate_view/renders/video.mp4` | wander path 视角游走（D-NeRF 视角合成） |
| `all` | `interpolate_all/renders/video.mp4` | 时间+视角都变（D-NeRF 专属） |
| `pose` | `interpolate_pose/renders/video.mp4` | 起止位姿线性插值 |
| `original` | `interpolate_hyper_view/renders/video.mp4` | 真实数据时间+视角插值（NeRF-DS/HyperNeRF） |

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

**11. `03_colmap_pose.sh` 报 `colmap 二进制不在 PATH 也非可执行文件`**
系统没装 colmap。Ubuntu: `apt install colmap`（CPU 版，慢但能跑）；要 GPU 加速 SIFT 提特征+匹配，从源码编（`-DENABLE_CUDA=ON`，需 CUDA toolkit）。或装好指定路径：`COLMAP_EXECUTABLE=/path/to/colmap bash deformable_gaussians/03_colmap_pose.sh`。

**12. `03` 报 `mapper 跑了但 sparse/0/ 没生成（重建失败）`**
COLMAP incremental SfM 重建失败，通常是匹配点不足。常见原因：
- 拍摄视角重叠加不够（相邻帧 < 70% 重叠）→ 增加拍摄密度
- 图像模糊 / 低纹理（白墙人物）→ 拍摄背景复杂些（衣物图案/纹理）
- 图像分辨率太低 → 不低于 1080p
排查：`KEEP_DISTORTED=1 bash 03...` 保留中间产物，用 colmap GUI 加载 `$OUTPUT_SCENE/distorted/database.db` 看匹配矩阵。

**13. `04` 训练报 `ValueError: invalid literal for int() with base 10: 'IMG_001'`**
图像文件名不是纯数字——03 没正确重命名。检查 `$OUTPUT_SCENE/input/` 下文件名是否为 `00000.jpg, 00001.jpg, ...`。若不是，`FORCE_RECOLMAP=1 bash 03...` 重跑 COLMAP（会清旧 input/ 重命名）。

**14. `04` 训练 OOM / `mapper` 报点云过大**
多视角拍摄图分辨率高（4K+）→ COLMAP 点云几百万点 → 高斯初始化 OOM。`03` 加 `NUM_IMAGES_MAX=80`（取前 80 张）；或先在 `INPUT_DIR/image/` 把图像缩到 1080p。`04` 加 `EXTRA_TRAIN_ARGS="--resolution 2"`（图像降 2 倍分辨率训练）。

**15. `04` 训练完发现 deformation 没学到（renders 看起来跟初始高斯一样）**
通常是 `warm_up=3000` 步前 deformation=0，且训练步数不够（`< 3000` 跑不出变形）。`ITERATIONS` 至少 7000。也可能是输入数据本质静态（人物 0 微动），这时 deformation field 学不到东西——换 2DGS/GOF 更合适（见 README 顶部"不适用场景"）。

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

### COLMAP pose (03) — 路径 B 真实数据
| var | default | note |
|---|---|---|
| `INPUT_DIR` | `$REPO_DIR/../wan22_rotate_results` | 拍摄图像目录（含 `image/` 子文件夹） |
| `SCENE_NAME` | `real_scene` | 场景名（影响默认 `OUTPUT_SCENE`） |
| `OUTPUT_SCENE` | `$MODEL_DIR/data/real/$SCENE_NAME` | 输出场景目录（NeRF-DS COLMAP 格式） |
| `COLMAP_EXECUTABLE` | `colmap` | colmap 二进制（PATH 找不到时设绝对路径） |
| `CAMERA_MODEL` | `OPENCV` | `OPENCV` / `SIMPLE_PINHOLE` / `PINHOLE` |
| `SINGLE_CAMERA` | `1` | `1` = 单相机假设（多视角同设备拍，NeRF-DS 默认） |
| `USE_GPU` | `1` | `1` = SIFT 提特征+匹配用 GPU（CPU 慢很多） |
| `SKIP_MATCHING` | `0` | `1` = 跳过 feature/matcher（复用已有 `distorted/database.db`） |
| `KEEP_DISTORTED` | `0` | `1` = 保留 `distorted/` 临时目录（colmap GUI 排查用） |
| `NUM_IMAGES_MAX` | `0` | `>0` = 只取前 N 张（防 COLMAP mapper OOM；0=全用） |
| `FORCE_RECOLMAP` | `0` | `1` = 强制重跑 COLMAP（清旧 `input/` + `distorted/`） |

### Train real (04) — 路径 B 真实数据
| var | default | note |
|---|---|---|
| `SCENE_NAME` | `real_scene` | 场景名（接 03；影响默认 `SOURCE_PATH` + `MODEL_PATH`） |
| `SOURCE_PATH` | `$MODEL_DIR/data/real/$SCENE_NAME` | COLMAP 场景目录（03 输出） |
| `MODEL_PATH` | `$DG_DIR/output/real_$SCENE_NAME` | 训练输出目录（与 D-NeRF `output/<scene>/` 分开） |
| `ITERATIONS` | `20000` | NeRF-DS 标配；D-NeRF 用 40000 |
| `IS_BLENDER` | `0` | `0` = 真实数据（默认）；`1` = D-NeRF 合成（加 `--is_blender`） |
| `IS_6DOF` | `0` | `1` = 6DoF 变体（指标略高更慢） |
| `WHITE_BG` | `0` | `1` = 白底训练（输入白底分割图时开；NeRF-DS 原图关） |
| `EVAL` | `1` | `1` = 划分 train/test（评测必需；`0`=全 train 不留 test） |
| `TEST_ITERATIONS` | _(train.py default)_ | 评测步（如 `"5000 10000 20000"`） |
| `SAVE_ITERATIONS` | _(train.py default)_ | 存 checkpoint 步（如 `"7000 20000"`） |
| `SKIP_VERIFY` | `0` | `1` = 跳过 CUDA 扩展 import 校验（已知装好时省秒） |
| `EXTRA_TRAIN_ARGS` | _(unset)_ | 透传 train.py（如 `--sh_degree 2 --port 0`） |

### Render real (05) — 路径 B 渲染（封装 02）
| var | default | note |
|---|---|---|
| `SCENE_NAME` | `real_scene` | 场景名（接 04；影响默认 `MODEL_PATH`） |
| `MODEL_PATH` | `$DG_DIR/output/real_$SCENE_NAME` | 训练输出目录（接 04） |
| `MODE` | `original` | `original`=NeRF-DS 时间+视角 video；`render`=仅 test PNG；其他见 02 |
| `ITERATION` | `-1` | `-1` = 最新 checkpoint |
| `SKIP_TRAIN` | `1` | `1` = 跳过渲染 train split |
| `SKIP_TEST` | `0` | `0` = 渲 test split（metrics 需要） |
| `RUN_METRICS` | `1` | `1` = 跑 metrics.py 算 PSNR/SSIM/LPIPS |

## Outputs
- **01 dataset**: `$MODEL_DIR/data/D-NeRF/<scene>/`（`transforms_train.json` + `transforms_test.json` + `images/`）。
- **03 COLMAP 场景**: `$MODEL_DIR/data/real/<SCENE_NAME>/{images/, sparse/0/{cameras,images,points3D}.bin}`。
- **train**: `$DG_DIR/output/<scene>/point_cloud/iteration_<N>/point_cloud.ply`（高斯点云）+ `cfg_args` + `deform/`（变形 MLP 权重）+ `input.ply` + `cameras.json` + TensorBoard events。路径 B 在 `output/real_<SCENE_NAME>/`（与 D-NeRF 复现分开）。
- **02 render+metrics**: `$MODEL_PATH/test/ours_<iter>/{renders,gt,depth}/*.png`（test 渲染 + GT + 深度）+ `$MODEL_PATH/test/results.json`（PSNR/SSIM/LPIPS）。`MODE=time/all/view/original/pose` 还出 `interpolate_*/renders/video.mp4`。

## 目录布局
```
<code-dir>/
├── media_code/                       # 本仓
│   ├── proxy.env                     # 代理 + 覆盖项, gitignored
│   └── deformable_gaussians/         # 编排脚本(本目录)
├── Deformable-3D-Gaussians/          # 官方代码(自动 clone 到 ../Deformable-3D-Gaussians, 含 submodules/)
│   ├── submodules/
│   │   ├── simple-knn/                       # gitlab.inria.fr/bkerbl/simple-knn (CUDA)
│   │   └── depth-diff-gaussian-rasterization # ingra14m/diff-gaussian-rasterization-extentions @ filter-norm (CUDA)
│   └── output/                               # 训练输出
│       ├── <scene>/                          #   路径 A: D-NeRF 复现 (hook/lego/...)
│       │   └── point_cloud/iteration_<N>/point_cloud.ply
│       └── real_<SCENE_NAME>/                #   路径 B: 真实人体 (03→04→05)
│           ├── point_cloud/iteration_<N>/point_cloud.ply
│           ├── deform/                       #   变形 MLP 权重
│           └── cfg_args                       #   render/metrics 读它恢复参数
└── ../../model/deformable-3d-gaussians/       # 权重/数据(在 <code-dir> 上一级, 各算法共享)
    └── data/
        ├── D-NeRF/<scene>/          # 01 下载的调整版 D-NeRF (路径 A)
        └── real/<SCENE_NAME>/      # 03 COLMAP 输出的 NeRF-DS 场景 (路径 B)
            ├── images/             #   去畸变图像
            ├── input/              #   临时: 纯数字序号原图 (重命名复制)
            └── sparse/0/           #   COLMAP SfM 重建结果
                ├── cameras.bin
                ├── images.bin
                └── points3D.bin
```
默认：官方代码 `../Deformable-3D-Gaussians`、数据 `../../model/deformable-3d-gaussians/data`、训练输出 `$DG_DIR/output/<scene>`（路径 A）或 `$DG_DIR/output/real_<scene>`（路径 B，相对本目录）；用 `DG_DIR` / `MODEL_DIR` / `DG_OUTPUT_ROOT` / `OUTPUT_SCENE` 覆盖。复用现有 conda env（默认 `deformable_gaussians`），但 Deformable-GS 的 `torch==1.13.1+cu116` pin 与其他算法冲突——建议专用 env。

## Notes
- Official code & dataset follow their own license (Deformable-3D-Gaussians = [non-commercial research/eval](https://github.com/ingra14m/Deformable-3D-Gaussians/blob/main/LICENSE.md)). This folder only orchestrates; no official code is copied.
- `.gitattributes` (repo root) forces LF so Windows-pushed scripts run cleanly on Ubuntu.
- `proxy.env` (proxy creds / path / env overrides) is gitignored — never committed. Don't put credentials in scripts.
- SSL behind a TLS-intercepting corporate proxy: pip uses `PIP_CERT`/`--trusted-host`; `git` uses `GIT_SSL_CAINFO`; `curl` uses `CURL_CA_BUNDLE` (`_env.sh` prefers `~/.ca-bundle.crt`, built by `hypir/setup_ca_bundle.sh` or equivalent).
- No pretrained weights exist — every scene is trained from scratch; `01` downloads the D-NeRF *dataset* (the reproduction input), `02` renders + scores the *trained* gaussians.
