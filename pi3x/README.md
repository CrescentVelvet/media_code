# Pi3X — 无序多图 → 相机位姿 + 稠密点云（前馈，无优化）

[Pi3X](https://github.com/yyfz/Pi3) 是 π³（ICLR 2026，permutation-equivariant 前馈几何网络）的工程升级版（2025-12-28 发布）：卷积头消除网格伪影、连续置信度、**支持注入相机位姿/内参/深度作为条件**、近似 metric 尺度重建。本目录只做编排——官方代码在 `$PI3_DIR`（WSL 默认 `~/repos/Pi3`，00a 自动 clone），权重在 `$MODEL_DIR`。

一次前向从 N 张无序图像直接回归：每帧 c2w 位姿（OpenCV 约定）+ 稠密点云 + 连续置信度；内参可由 local_points 的射线方向恢复（`recover_intrinsic_from_rays_d`）。

## 常用命令

> 假设在 WSL；脚本自动激活 `pi3x` env；`GPU=0` 按需换卡。首次跑前先做下方「首次准备」。
> **铁律：每条命令都必须显式写出模型地址、输入路径、输出路径，不能全靠脚本里的默认值。**

```bash
# ── 一键（佳佳抱枕采集包：准备 → 纯图像推理 → 条件注入推理 → GT 评测）──
GPU=0 SCENE=jiajia_baozhen \
  RAW_DIR="/mnt/d/dataset/手部微动样例/01e954de1b8a4cde9315c976f4dcb964-佳佳抱枕" \
  RESULTS_DIR=~/output/pi3x_results \
  bash pi3x/run_all.sh

# ── 分步 ──
# 0) 环境（WSL：建 env + clone + 装依赖 + 验证）
INSTALL_DEPS=1 bash pi3x/00a_setup_env.sh
# 1) 下权重（~4GB，HF 境外走 proxy.env 代理；落 /mnt/d/wheel/pi3x_ms + 软链）
PI3X_MODEL_DIR=~/model/pi3x bash pi3x/01_download_models.sh
# 2) 采集包 → 去畸变 PNG + conditions.npz（GT 内参/位姿）
SCENE=jiajia_baozhen \
  RAW_DIR="/mnt/d/dataset/手部微动样例/01e954de1b8a4cde9315c976f4dcb964-佳佳抱枕" \
  RESULTS_DIR=~/output/pi3x_results \
  bash pi3x/02_prepare_images.sh
# 3) 纯图像推理
GPU=0 SCENE=jiajia_baozhen RESULTS_DIR=~/output/pi3x_results \
  bash pi3x/03_run_inference.sh
# 3b) 注入 GT 内参+位姿条件推理（Pi3X 招牌能力）
GPU=0 SCENE=jiajia_baozhen USE_CONDITIONS=1 RESULTS_DIR=~/output/pi3x_results \
  bash pi3x/03_run_inference.sh
# 4) GT 评测（Sim(3) 对齐后 ATE / 旋转 / 内参误差）
SCENE=jiajia_baozhen TAG=nocond RESULTS_DIR=~/output/pi3x_results bash pi3x/04_eval_gt.sh
SCENE=jiajia_baozhen TAG=cond   RESULTS_DIR=~/output/pi3x_results bash pi3x/04_eval_gt.sh
# 8) 搬到 Windows 盘
RESULTS_DIR=~/output/pi3x_results bash pi3x/08_move_output.sh
```

- 结果：`$RESULTS_DIR/$SCENE/images/`（去畸变 PNG）、`conditions.npz`（GT）、
  `{nocond,cond}/predictions.npz + dense_cloud.ply + poses.json + intrinsics.json + eval_gt.json`。

## 首次准备

```bash
cd <your-code-dir>            # e.g. /data_3d/<uid>/code 或 WSL ~/code
git clone <media_code 仓> && cd media_code
cp proxy.env.example proxy.env   # 填 http_proxy / https_proxy（HF 下载要走代理）

# 1. 建 pi3x env（CPython 3.10 + torch 2.5.1+cu124，官方 requirements 钉版）
INSTALL_DEPS=1 bash pi3x/00a_setup_env.sh

# 2. 下 Pi3X 权重（公开，免 token；~4GB）
bash pi3x/01_download_models.sh
# 或手动：https://huggingface.co/yyfz233/Pi3X/tree/main → model.safetensors
# 放到 $PI3X_WHEEL_DIR（默认 /mnt/d/wheel/pi3x_ms）
```

## Config（env vars）

| var | default | note |
| --- | --- | --- |
| `SCENE` | `jiajia_baozhen` | 场景名（纯 ASCII，目录名） |
| `RAW_DIR` | _(必填)_ | Remy 采集包（transforms.json + image/*.heic + pcd.ply） |
| `CONDA_ENV` | `pi3x` | conda env |
| `PI3_DIR` | `~/repos/Pi3` | 官方代码 |
| `PI3X_MODEL_DIR` | `~/model/pi3x` | 权重根（消费侧，软链到 wheel 池） |
| `PI3X_WHEEL_DIR` | `/mnt/d/wheel/pi3x_ms` | 权重复用池 |
| `RESULTS_DIR` | `~/output/pi3x_results` | 输出根 |
| `USE_CONDITIONS` | `0` | 03：`1` = 注入 GT 内参+位姿 |
| `TAG` | `nocond`/`cond` | 03/04 输出子目录名 |
| `PIXEL_LIMIT` | `255000` | 输入像素上限（官方默认；加大多细节但吃显存） |
| `CONF_THRES` | `0.1` | sigmoid(conf) 点云过滤阈值 |
| `GPU` | _(unset)_ | physical GPU id |

## 可能遇到的问题

**1. 03 报 OOM**
Pi3X 显存随帧数增长（3090 24G 实测 82 帧 @255k 像素可跑）。降压：减少输入帧数，或 `PIXEL_LIMIT=150000`。

**2. 01 下载不动**
HF 是境外站，确认 proxy.env 的代理段已填且 VPN 段 no_proxy 只排除国内源。也可迅雷/浏览器下好手动放 `$PI3X_WHEEL_DIR/model.safetensors`（脚本幂等，已有就跳过）。

**3. 02 报「解码尺寸与 json 不符」**
HEIC 带 EXIF 旋转时 pillow-heif 会自动转正，若采集包 json 里的 w/h 是转前尺寸就会 mismatch——本采集包（HarmonyOS 手机）实测无此问题；其它来源数据先单帧验证再全量。

**4. 04 的尺度看着不对**
Pi3X 纯图像模式输出是**近似 metric**（官方自述 approximate），不保证准；`eval_gt.json` 里的 `sim3_scale` 是 pred→m 的实际换算，离 1 不远说明 metric 尺度基本靠谱。

## Notes
- 官方代码 BSD-3-Clause；权重 CC BY-NC 4.0 **严禁商用**。本目录只编排，未拷贝官方代码。
- 与 `pi3_3dgs/` 的关系：那个是**原版 Pi3** → COLMAP → 2DGS 训练；本目录是 **Pi3X** 纯前馈评测（含条件注入与 GT 对账），不接 splatting 训练。
- 位姿约定：Pi3X 输出/条件都是 c2w OpenCV（z 前 y 下 x 右）；采集包 transform_matrix 是 c2w OpenGL，转换 `M @ diag(1,-1,-1,1)`（som_human 实测判定）。
