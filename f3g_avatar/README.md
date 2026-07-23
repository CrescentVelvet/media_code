# F3G-Avatar runner

在 Ubuntu + NVIDIA 服务器上跑 [F3G-Avatar](https://github.com/wjmenu/F3G-avatar)（CVPRW 2026，Face Focussed Full-body Gaussian Avatar）的**推理 / 数据准备 / 训练**。本目录只含编排脚本——官方代码自动 clone、权重从 HuggingFace 下载、SMPL-X 手动放置（gated）。

## 常用命令

> 假设已进入容器并 `conda activate f3g_avatar`；路径取各脚本默认值（可改）；`GPU=0` 按需换卡。首次跑前先做下方「首次准备」。

```bash
# ── 一键（clone 官方仓 + 装依赖 + 建CUDA扩展 + 下权重 + SMPL-X 检查）──
GPU=0 INSTALL_DEPS=1 BUILD_CUDA=1 bash f3g_avatar/run_all.sh

# ── 推理(02)：渲染 avatar ──
# 1) 用发布权重 + 自备数据（自由视角环绕）
GPU=0 DATA_DIR=/data_3d/w00xxxxxx/code/F3G-avatar/data/avatarrex_zzr \
  VIEW_SETTING=free bash f3g_avatar/02_run_inference.sh
# 2) 用发布权重 + 外部 AMASS 动作序列（动画）
GPU=0 DATA_DIR=/data_3d/w00xxxxxx/code/F3G-avatar/data/avatarrex_zzr \
  POSE_DATA=/data/AMASS/CMU/10/10_05_poses.npz POSE_FRAME_RANGE=0,500 \
  VIEW_SETTING=free bash f3g_avatar/02_run_inference.sh
# 3) 用自己训的权重（checkpoint 在 results/<subject>/avatar/batch_<N>/net.pt）
GPU=0 PREV_CKPT=../F3G-avatar/results/avatarrex_zzr/avatar/batch_700000 \
  DATA_DIR=/data_3d/w00xxxxxx/code/F3G-avatar/data/avatarrex_zzr \
  VIEW_SETTING=free bash f3g_avatar/02_run_inference.sh
# 4) 训练视角（相机）渲染 + 存 PLY / 纹理图
GPU=0 DATA_DIR=.../avatarrex_zzr VIEW_SETTING=camera RENDER_VIEW_IDX=13 \
  SAVE_PLY=1 SAVE_TEX_MAP=1 bash f3g_avatar/02_run_inference.sh

# ── 训练(官方 main_avatar.py，本仓不改官方文件) ──
# 5) 从零训（发布权重未释出时）：编辑 configs/avatarrex_zzr/avatar.yaml 的
#    train.data.data_dir -> 你的数据，然后：
cd ../F3G-avatar && python main_avatar.py -c configs/avatarrex_zzr/avatar.yaml -m train
```

- 结果：训练 → `../F3G-avatar/results/<subject>/avatar/batch_<step>/net.pt`（+ `optm.pt`）；推理 → `test.output_dir`，默认 `../F3G-avatar/test_results/<subject>/<exp>/.../batch_<iter>/rgb_map/%08d.jpg`。
- 推理链路：`02_run_inference.sh` 调 `gen_test_config.py` 由官方 `configs/avatarrex_zzr/avatar.yaml` 派生一份 test YAML（不改官方文件），再 `python main_avatar.py -c <派生> -m test`。
- ⚠️ 发布权重在 HF 上标注 "Coming soon"——未释出前 `01` 会提示从零训练（见「首次准备」）。
- ⚠️ F3G 的 `torch==2.1.0+cu121 / pytorch3d / torch-scatter / triton` 版本 pin 与本仓其他算法冲突，务必用专用 env（`CONDA_ENV=f3g_avatar`），别装进共享的 `doll`。

## 首次准备
```bash
cd <your-code-dir>            # e.g. /data_3d/<uid>/code
git -c http.sslVerify=false clone https://github.com/CrescentVelvet/media_code.git
cd media_code && cp proxy.env.example proxy.env   # 填 http_proxy / https_proxy（公司代理用）
conda create -n f3g_avatar python=3.10 -y && conda activate f3g_avatar
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
INSTALL_DEPS=1 BUILD_CUDA=1 bash f3g_avatar/00_setup_env.sh   # 装官方 requirements + 建两个 CUDA 扩展
bash f3g_avatar/01_download_models.sh                          # 下发布权重（若已释出）+ 检查 SMPL-X
```
SMPL-X 是 gated 模型（需注册登录 https://smpl-x.is.tue.mpg.de/download.php）。`01` 检测到 `../F3G-avatar/smpl_files/smplx/SMPLX_NEUTRAL.npz` 缺失时会打印手动下载步骤——下完放进该目录再跑一次 `01` 即可。

---

以下为详细参考（流程原理 / 各脚本参数 / 排错 / 目录布局）。

## Inference (02 — 更多用法)
`02_run_inference.sh` 由 `gen_test_config.py` 填好一份派生 YAML 后调 `python main_avatar.py -c <派生> -m test`。官方 `test()` 加载一次 checkpoint，按 `view_setting` 循环每帧渲染并写盘，逐帧 `tqdm` 进度。其它覆盖示例：
```bash
# 自由视角环绕 vs 训练相机视角（同一数据各跑一次，OUTPUT_DIR 分开）
GPU=0 DATA_DIR=.../zzr VIEW_SETTING=free OUTPUT_DIR=../F3G-avatar/results/free  bash f3g_avatar/02_run_inference.sh
GPU=0 DATA_DIR=.../zzr VIEW_SETTING=camera RENDER_VIEW_IDX=13 OUTPUT_DIR=../F3G-avatar/results/cam bash f3g_avatar/02_run_inference.sh
# 关 PCA（vanilla，按原 pose 渲染）/ 调 PCA 强度做动作变化
GPU=0 DATA_DIR=.../zzr N_PCA=0 bash f3g_avatar/02_run_inference.sh
GPU=0 DATA_DIR=.../zzr N_PCA=20 SIGMA_PCA=3.0 bash f3g_avatar/02_run_inference.sh
# 缩放渲染分辨率（img_scale<1 更快）
GPU=0 DATA_DIR=.../zzr IMG_SCALE=0.5 bash f3g_avatar/02_run_inference.sh
```
派生 YAML 落在 `../F3G-avatar/configs/avatarrex_zzr/_test_derived.yaml`（gitignored），想手改可直接编辑后再跑 `02`。

## Pipeline（推理流程详解）
对应官方 `main_avatar.py::AvatarTrainer.test` + `network/avatar.py::AvatarNet.render` + `dataset/dataset_mv_rgb.py`。一个已训好的 avatar 从 checkpoint 到渲染出图经过：

1. **加载 checkpoint** — `load_ckpt(test.prev_ckpt)` 读 `<prev_ckpt>/net.pt`，取出 `avatar_net` state_dict 载入 `AvatarNet`（双分支：body + face-focused）。同时取 `epoch_idx/iter_idx` 用于输出目录命名（`batch_%06d`）。
2. **建训练数据集（仅取 SMPL shape / PCA）** — `test()` 总是先 `MvRgbDataset(**train.data, training=False)`：读 `data_dir/smpl_params.npz`（betas 提供 SMPL-X shape）、若 `n_pca>=1` 还在 `smpl_pos_map/` 上算 PCA（对 pose 做变化）。所以 `train.data.data_dir` 必须是有 `smpl_params.npz + smpl_pos_map/` 的已备数据。⚠️ 这一步需要 SMPL-X 模型（`smplx.SMPLX(model_path=.../smpl_files/smplx, gender='neutral')`）——没放会崩。
3. **建测试数据集** — 有 `test.pose_data` 时用 `PoseDataset`（外部 AMASS/`.npz` 动作序列 + 上一步的 betas），否则用 `MvRgbDataset(test.data, training=False)`（渲染训练帧）。`PoseDataset` 用 smpl_shape 在线生成 pose。
4. **逐帧渲染** — 按 `view_setting` 算相机外参（`free`=环绕转台 / `camera`=训练相机 / `front|back|moving|cano`），`getitem` 组装 SMPL 顶点、关节、`cano2live` LBS 矩阵、pose_map（若无则 `avatar_net.get_pose_map` 生成）。`AvatarNet.render`：把 MHR 模板点 + 双分支（body 的 pose-dependent 偏移 + face 分支）解码成 3D 高斯，经 LBS 蒙皮 pose 后用可微高斯泼溅光栅化（`diff_gaussian_rasterization`）出 RGB + mask。
5. **写盘** — `rgb_map/%08d.jpg`（必出）、`mask_map/%08d.png`（有 mask_map 时）、`save_ply`→`posed_gaussians/%08d.ply`、`save_tex_map`→`cano_tex_map/%08d.jpg`、`render_skeleton`→`live_skeleton/`。

### 为什么推理也要「已备数据」
F3G 的 avatar 是**个体模型**——checkpoint 编码的是某一个被拍对象的高斯。渲染该对象的任意 pose/视角时，仍需要：(a) 该对象的 SMPL-X shape（betas，来自 `smpl_params.npz`）；(b) pose 信号（来自 `smpl_pos_map/*.exr`，或外部 `pose_data`）。两者都在「数据准备」阶段产生。所以 `DATA_DIR` = 你当初训练用的那个多视角采集夹（至少要有 `smpl_params.npz`；自由视角动画时 pose 由 `pose_data` 提供，但 `smpl_pos_map` 仍用于 PCA）。

> 一句话：**checkpoint 存高斯 → 取对象 SMPL shape → 给 pose（pos_map 或 AMASS）→ 双分支解码高斯 → LBS 蒙皮 → 高斯泼溅出图**。

## Data Preparation（数据准备，复述官方 README）
训练/推理需要的不是裸图，而是：**穿衣 MHR 模板**（带 pose map 的全身）+ **头部裁剪**（face 分支）。设 `export DATA=/path/to/your_subject`，在官方仓根目录跑：
```bash
# 1) 人脸裁剪（4D-DRESS/Graphonomy + 可选 SAM），产出独立 face 数据集
python tools/python/crop_dataset_rex_faces.py --src_root "$DATA" --dst_root /path/to/crops
# 2) 打包多视角输入
python create_template.py --data_dir "$DATA" --out_dir mesh_data
cp mesh_data/calibration_full.json mesh_data/transforms.json
# 3) 建 MHR 模板（NeuS2 重建 + 4D-Dress 语义分割 + PhysAvatar 组合/LBS 权重）
python tools/python/pipeline_multiview_to_template.py \
  --images_dir mesh_data/images --transforms_json mesh_data/transforms.json \
  --smpl_params_npz mesh_data/smpl_params.npz --out_dir mesh \
  --run_neus2 --neus2_name neus2_exp --neus2_steps 100000 --outfit Outer --weights_mode inpaint
# 4) 光栅化 pose map 进训练集（读 mesh/template/，写 $DATA/smpl_pos_map/）
python -m gen_data.gen_pos_maps -c configs/avatarrex_zzr/avatar.yaml
```
- 这些脚本依赖 `othercode/{NeuS2,4d-dress,PhysAvatar,StyleAvatar}`。`CLONE_EXTRA=1 bash f3g_avatar/01_download_models.sh` 会把这四个仓 clone 进 `../F3G-avatar/othercode/`（NeuS2 仍需 cmake 编译；4D-Dress 需 Graphonomy/SAM 权重，见官方 [4D-Dress model install](https://github.com/eth-ait/4d-dress#model-installation)）。
- 已有 mesh 的话，第 3 步可加 `--mesh_obj /path/to/mesh.obj` 跳过 NeuS2。

## Training（训练，官方 main_avatar.py）
`main_avatar.py -c <yaml> -m train` 会先 `pretrain()`（前 500 步学位置/opacity/scale/rotation 对齐 cano 高斯），再 `train()`（L1 + LPIPS + offset + 可选 GAN/face 损失）。编辑 `configs/avatarrex_zzr/avatar.yaml` 的 `train.data.{data_dir,crop_data_dir,subject_name}` 指向你的数据即可（或复制一份改）。checkpoint 落在 `train.net_ckpt_dir/batch_<step>/{net.pt,optm.pt}`，推理时 `PREV_CKPT` 指向该目录：
```bash
cd ../F3G-avatar
python main_avatar.py -c configs/avatarrex_zzr/avatar.yaml -m train
# 续训：把 test.prev_ckpt 改成 .../batch_<N>，或用 -m test 时的 PREV_CKPT
```
> 发布权重释出后，可把它当 `train.pretrained_dir` 暖启动（`01` 下的 `net.pt` 含 `avatar_net` state_dict，`load_ckpt(load_optm=False)` 能读）。

## 可能遇到的问题

公司代理做 HTTPS 中间人解密，下面按流水线阶段列出常见报错与修法（命令在服务器上、conda 环境已激活时执行）。

**1. clone/pull 本仓或官方仓报错**
- `SSL certificate problem`：公开仓免认证，加 `-c http.sslVerify=false`（`run_all.sh` 已对官方仓兜底）。
- `Failed to connect to github.com port 443`（连不上，非 SSL）：git 没走代理。设全局代理（密码特殊字符须 URL 编码）：
  ```bash
  git config --global https.proxy http://USER:PASS@proxyhk.huawei.com:8080
  git config --global  http.proxy http://USER:PASS@proxyhk.huawei.com:8080
  ```
- `No route to host` 连代理都连不上：docker 网桥网段和代理 IP 冲突。查 `getent hosts <proxy>` + `ip route | grep <net>`，加主机路由 `sudo ip add <proxy-ip> via <gw> dev <nic>`；治本在 `/etc/docker/daemon.json` 配 `default-address-pools` 给 docker 分不冲突子网再 `systemctl restart docker`。

**2. `pip install -r requirements.txt` 装不上 pytorch3d / torch-scatter**
这两个不能从裸 PyPI pin 装。`00_setup_env.sh` 在 `INSTALL_DEPS=1` 时会先装其余依赖，再 best-effort：
```bash
# pytorch3d（优先 conda 渠道，否则源码编译，需 nvcc + 匹配的 torch）
conda install -n f3g_avatar -y -c pytorch3d pytorch3d=0.7.8
# 或: pip install --no-build-isolation "pytorch3d==0.7.8"
# torch-scatter（用 pyG 的 cu121 wheel）
pip install torch-scatter==2.1.2 --no-index --find-links https://data.pyg.org/whl/torch-2.1.0+cu121.html
```
仍失败多半是 nvcc/torch 版本不匹配——确认 `python -c "import torch;print(torch.__version__)"` 是 `2.1.0+cu121`、`nvcc -V` 是 12.1。H100（sm90）需 cu118+ 的 torch，见 #4。

**3. `python setup.py install`（CUDA 扩展）报 `No CUDA toolkit found` / `nvcc` 缺失**
`diff_gaussian_rasterization` 和 `styleunet` 要 nvcc 编译。装 CUDA 12.1 toolkit（与 torch cu121 对齐）：
```bash
conda install -n f3g_avatar -y -c "nvidia/label/cuda-12.1.1" cuda-toolkit
export CUDA_HOME=$CONDA_PREFIX  # 让 setup.py 找到 nvcc
BUILD_CUDA=1 bash f3g_avatar/00_setup_env.sh
```

**4. H100（sm90）跑不动 / 编译报 `no kernel image is available`**
`torch==2.1.0+cu121` 不含 sm90 预编译 kernel。换 cu118 的 torch（含 sm90）：
```bash
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
# torch-scatter 也换 cu118 wheel：--find-links https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

**5. `01` 下发布权重报 404 / `Entry not found`**
HF model card 标 "Coming soon"——权重未释出。`01` 会打印从零训练指引并继续（`exit 0`）。此时先按「Data Preparation + Training」自训，再 `PREV_CKPT=.../batch_<N> bash 02`。镜像若 gated（HF 对未认证返 401 伪装成 404），在仓库页接受许可 + 建 read token：`HF_TOKEN=<token> bash f3g_avatar/01_download_models.sh`。

**6. `01` 报 `SSLCertVerificationError`**
代理根 CA 不在系统包。先建 CA 包（`_env.sh` 会自动用 `~/.ca-bundle.crt`）：
```bash
bash hypir/setup_ca_bundle.sh    # 抓代理证书链 -> ~/.ca-bundle.crt
bash f3g_avatar/01_download_models.sh
```
仍报 SSL（CDN 端点用不同 MITM 证书）→ `01` 自动回退 SSL 免验下载器；或 `HF_DISABLE_SSL=1 bash f3g_avatar/01_download_models.sh`。

**7. 推理报 `SMPLX_NEUTRAL model not found` / `smplx` 加载崩**
SMPL-X 没放。从 https://smpl-x.is.tue.mpg.de/download.php 下 SMPL-X 包，把 `SMPLX_NEUTRAL.npz`（及 MALE/FEMALE）放进 `../F3G-avatar/smpl_files/smplx/`。代码用 `gender='neutral'`。

**8. 推理报 `FileNotFoundError: smpl_params.npz` / 找不到 `smpl_pos_map/*.exr`**
`DATA_DIR` 不是已备数据。它必须是跑过 `gen_data.gen_pos_maps` 的采集夹（含 `smpl_params.npz` + `smpl_pos_map/`）。见「Data Preparation」。自由视角动画用 `POSE_DATA` 时仍需 `smpl_params.npz`（取 betas）。

**9. 推理报 `import diff_gaussian_rasterization / styleunet failed`**
CUDA 扩展没建。`BUILD_CUDA=1 bash f3g_avatar/00_setup_env.sh`（首次跑 `run_all` 默认会建一次）。

**10. `02` 生成的派生 YAML 路径不对 / 想用别的 base config**
默认派生自 `../F3G-avatar/configs/avatarrex_zzr/avatar.yaml`。换实验就 `BASE_CONFIG=../F3G-avatar/configs/<exp>/avatar.yaml` 再跑 `02`（或在 `configs/` 下复制一份改）。

**11. 跑 `.sh` 报 `syntax error near unexpected token ('`（CRLF 行尾）**
脚本被 CRLF 污染（Windows→服务器用 scp/zip 等非 git 方式同步）。修：
```bash
sed -i 's/\r$//' f3g_avatar/*.sh
# 或 git checkout -- f3g_avatar/<file>.sh   # .gitattributes 强制 LF
```

> 通用：`proxy.env`（代理凭证）在仓内 gitignored，`~/.ca-bundle.crt` 在家目录，都不入库；切勿把凭证写进脚本。

## Config (env vars, all optional)
| var | default | note |
|---|---|---|
| `CONDA_ENV` | `f3g_avatar` | conda env to activate (dedicated — pins conflict with other algos) |
| `GPU` | _(unset)_ | physical GPU id to pin, e.g. `GPU=0`; `cuda:0` in-process == physical GPU N |
| `F3G_DIR` | `../F3G-avatar` | official code path |
| `MODEL_DIR` | `../../model/F3G-avatar` | weights path (avatarrex_zzr/net.pt) |
| `F3G_REPO` | official GitHub URL | clone source |
| `INSTALL_DEPS` | `0` (run_all: `0`) | `1` = `pip install -r requirements.txt` |
| `BUILD_CUDA` | `1` (run_all) / `0` (00) | `1` = build diff-gaussian-rasterization + StyleUNet |
| `SKIP_TORCH` | `0` | `1` = filter torch/torchvision/triton pins out of requirements (keep existing torch) |
| `HF_REPO` | `wjmenu/F3G-avatar` | HF checkpoint repo |
| `CKPT_FILE` | `checkpoints/avatarrex_zzr/epoch_latest.pt` | file path inside HF_REPO |
| `HF_DISABLE_SSL` | `0` | `1` = download checkpoint with SSL verification disabled |
| `HF_TOKEN` | _(unset)_ | only if you point HF_REPO at a gated repo |
| `SMPLX_DIR` | `$F3G_DIR/smpl_files/smplx` | SMPL-X model dir (gated, manual) |
| `CLONE_EXTRA` | `0` | `1` = clone NeuS2/4D-Dress/PhysAvatar/StyleAvatar into othercode/ |
| `PREV_CKPT` | `$MODEL_DIR/avatarrex_zzr` | checkpoint dir (must contain `net.pt`); used as `test.prev_ckpt` |
| `DATA_DIR` | _(required)_ | prepared multiview capture (smpl_params.npz + smpl_pos_map/) |
| `SUBJECT_NAME` | basename(DATA_DIR) | subject name for output paths |
| `DATA_FRAME_RANGE` | base default | `start,end[,step]` for train.data + test.data |
| `POSE_DATA` | _(unset)_ | external `.npz` pose sequence -> free-view animation (PoseDataset) |
| `POSE_FRAME_RANGE` | base default | `start,end[,step]` for pose_data |
| `VIEW_SETTING` | `free` | `free` / `camera` / `front` / `back` / `moving` / `cano` |
| `RENDER_VIEW_IDX` | `13` | camera id for `view_setting=camera` |
| `IMG_SCALE` | `1.0` | render resolution scale |
| `SAVE_PLY` | `0` | `1` = save posed_gaussians/%08d.ply |
| `SAVE_TEX_MAP` | `0` | `1` = save cano_tex_map/%08d.jpg |
| `N_PCA` | `20` | `<1` disables PCA (vanilla); `>=1` enables pose variation |
| `SIGMA_PCA` | `2.0` | PCA pose variation strength |
| `GLOBAL_ORIENT` | `1` | align free-view camera to subject global orient |
| `OUTPUT_DIR` | code default | `test.output_dir` (else `./test_results/<subject>/<exp>/...`) |
| `BASE_CONFIG` | `$F3G_DIR/configs/avatarrex_zzr/avatar.yaml` | base YAML to derive from |

## 目录布局
```
../F3G-avatar/                         # 官方仓（run_all 自动 clone）
  configs/avatarrex_zzr/avatar.yaml     # base config（02 派生自它）
  configs/avatarrex_zzr/_test_derived.yaml  # 02 生成的派生 test config（gitignored）
  smpl_files/smplx/SMPLX_NEUTRAL.npz    # SMPL-X（gated，手动放）
  othercode/{NeuS2,4d-dress,PhysAvatar,StyleAvatar}  # CLONE_EXTRA=1 时 clone
  results/<subject>/avatar/batch_<N>/net.pt   # 训练 checkpoint
  test_results/<subject>/<exp>/.../rgb_map/%08d.jpg  # 推理输出
../../model/F3G-avatar/avatarrex_zzr/net.pt   # 发布权重（01 下载，格式同 net.pt）
```

## Acknowledgement
复现自 [F3G-Avatar](https://github.com/wjmenu/F3G-avatar)（CVPRW 2026），其构建于 [Animatable Gaussians](https://github.com/lizhe00/AnimatableGaussians)、[NeuS2](https://github.com/19reborn/NeuS2)、[4D-Dress](https://github.com/eth-ait/4d-dress) + [PhysAvatar](https://github.com/y-zheng18/PhysAvatar) + [StyleAvatar](https://github.com/LizhenWangT/StyleAvatar)、[3D Gaussian Splatting](https://github.com/ashawkey/diff-gaussian-rasterization)。

## Citation
```bibtex
@misc{menu2026f3gavatarfacefocused,
  title={F3G-Avatar : Face Focused Full-body Gaussian Avatar},
  author={Willem Menu and Erkut Akdag and Pedro Quesado and Yasaman Kashefbahrami and Egor Bondarev},
  year={2026},
  eprint={2604.09835},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2604.09835},
}
```
