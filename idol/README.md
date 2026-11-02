# Idol — 单张图转 3D Gaussian Splatting 物体模型

Idol 由 [speridlabs](https://speridlabs.com) 开源（Apache-2.0，2026-10-29 首发），把**任意单张物体图**（渲染图 / 生成图 / 实拍照片）
转成标准 3DGS `.ply`：BiRefNet 抠图 → DINOv3 ViT-H+/16 + FLUX.2 VAE 双路图像条件 → 370M DiT 流匹配采样
→ 八叉树解码器放锚点 → 高斯解码器每锚点出 32 个高斯。官方口径：7.2 GB 显存可跑，输出 32,768 ~ 1,048,576 个高斯
（默认 262,144），H100 上 6.8 s / 图。

本目录是 WSL 本机复现（单卡 RTX 3090）。**模型与安装包统一落在 `D:\wheel`，重复复现不重下。**
官方代码位于 `~/repos/idol-sdk`（WSL Linux fs），以 editable 方式安装，源码不改动。

## 常用命令

> **铁律：命令里显式写出模型路径、输入路径、输出路径，不要依赖脚本默认值。**

```bash
# 一键（建环境 -> 下权重 -> 单图推理）
IMAGE=~/repos/idol-sdk/demo/examples/fur-egg-chair.webp \
  MODE=synthetic NUM_GAUSSIANS=262144 \
  RESULTS_DIR=~/output/idol_results \
  bash idol/run_all.sh

# 分步
# 1) 建 env + 装依赖 + clone 官方仓
CONDA_ENV=idol PY_VER=3.11 TORCH_VER=2.8.0 \
  bash idol/00_setup_env.sh

# 2) 下权重（约 5 GB，落到 D:\wheel\hf_home_idol）
bash idol/01_download_models.sh

# 3) 推理：单张
IMAGE=~/repos/idol-sdk/demo/examples/fur-egg-chair.webp \
  MODE=synthetic NUM_GAUSSIANS=262144 STEPS=50 SEED=42 \
  OUT=~/output/idol_results \
  bash idol/02_run_inference.sh

# 3b) 推理：整个目录批量
IMAGE=~/repos/idol-sdk/demo/examples \
  MODE=synthetic NUM_GAUSSIANS=1048576 \
  OUT=~/output/idol_results_batch \
  bash idol/02_run_inference.sh

# 4) 渲染验证：把 PLY 投影成 6 视角 PNG（不需要 CUDA 栅格器）
PLY=~/output/idol_results/fur-egg-chair.ply \
  OUT=~/output/idol_results/fur-egg-chair_preview.png \
  bash idol/04_preview.sh

# 5) 本地 Gradio demo（等价 HF Space）
PORT=7860 bash idol/03_run_demo.sh

# 6) 结果从 Linux fs 搬到 Windows 盘
RESULTS_DIR=~/output/idol_results DEST=/mnt/d/output/idol_results \
  bash idol/08_move_output.sh
```

## 首次准备

```bash
# 1) proxy.env 里必须有 HF_TOKEN（两个图像编码器是 gated 仓库）
#    本机 token 已确认对以下三个仓库均有权限（账号 wyf08）：
#      speridlabs/idol
#      facebook/dinov3-vith16plus-pretrain-lvd1689m
#      black-forest-labs/FLUX.2-dev
grep -n "HF_TOKEN" proxy.env        # 确认第 8 行 export HF_TOKEN=hf_xxx 未注释

# 2) 建环境 + 下权重
bash idol/00_setup_env.sh
bash idol/01_download_models.sh
```

## Config (env vars)

| var | default | note |
|---|---|---|
| `CONDA_ENV` | `idol` | 独立 env，勿装进 vggt_human（transformers 版本冲突） |
| `PY_VER` | `3.11` | **必须 3.11**：D:\wheel 里 torch/nvidia/flash-attn wheel 全是 cp311 |
| `TORCH_VER` / `TV_VER` | `2.8.0` / `0.23.0` | 与本地 nvidia-cu12.8 全套 wheel 对齐 |
| `WHEEL_DIR` | `/mnt/d/wheel` | 模型 + 安装包长期落地盘 |
| `HF_HOME` | `/mnt/d/wheel/hf_home_idol` | HF 缓存，复现不重下的关键 |
| `IDOL_MODELS` | `/mnt/d/wheel/idol_ms` | 01 导出的稳定软链目录，供 `--weights` 用 |
| `IDOL_REPO` | `~/repos/idol-sdk` | 官方代码 |
| `RESULTS_DIR` / `OUT` | `~/output/idol_results` | 推理产物 |
| `MODE` | `synthetic` | `synthetic` 裁切 1.2（渲染/生成图）；`real` 裁切 1.7（照片） |
| `NUM_GAUSSIANS` | `262144` | 32,768 ~ 1,048,576，须为 32 的倍数 |
| `STEPS` | `50` | 采样步数 |
| `CFG_STRENGTH` | `3.0` | CFG 强度 |
| `RESCALE_T` | `3.0` | 时间步 shift，越高越把步数花在高噪声段 |
| `SEED` | `42` | 同 seed 同物体 |
| `DINOV3_DTYPE` | `bf16` | `fp32` 更准，多占 1.7 GB |
| `OPTIMIZE` | `1` | torch.compile + cuDNN attn + TF32；首次生成要编译 |
| `INSTALL_FLASH_ATTN` | `1` | 失败自动退回 sdpa（结果一致，仅变慢） |
| `PLY` / `OUT` | — | `04_preview.sh` 的输入 PLY 与输出 PNG（`PLY` 也可传目录，批量渲染） |
| `PREVIEW_SIZE` | `480` | `04` 单视角画布边长；**勿改名为 `SIZE`**（撞 conda binutils 变量） |

## 目录布局

```
/home/velvet/
├── repos/idol-sdk/            # 官方代码（editable 安装）
└── output/idol_results/       # 推理产物（跑完用 08 搬到 /mnt/d/output/）

/mnt/d/wheel/                  # = D:\wheel，长期复用
├── hf_home_idol/hub/          # HF 缓存：models--speridlabs--idol 等
├── idol_ms/                   # 稳定软链（config + 4 个 safetensors）
└── pip_cache/                 # pip 下载缓存
```

每个输入图产出三件：`<name>.ply`（标准 3DGS）、`<name>_input.png`（模型实际看到的图：抠图 + 裁切后）、
`<name>_camera.json`（预测的输入相机 yaw/pitch/radius/fov）。

## 实测数据（RTX 3090 24 GB / WSL2，2026-11-02）

| 项 | 结果 |
|---|---|
| 依赖安装 | 首次约 6 min（torch 2.8.0+cu128 与 nvidia-* 全部复用 `D:\wheel` 本地 wheel，只从 tuna 拉纯 python 小包） |
| 权重下载 | 6.4 GB / 48 min（走宿主 10808 代理，实测 1.5~2.3 MB/s） |
| 262,144 高斯 / 50 步 | 单张 **80 s**（首次含 torch.compile）；批量同进程后续每张 **≈40 s** 采样 |
| 1,048,576 高斯 / 50 步 | **69~70 s** |
| 采样速率 | 1.28~1.33 it/s（50 步约 40 s） |
| GPU 峰值显存 | **9,456 MiB**（1M 高斯配置）；262k 配置同为 9,452 MiB —— 瓶颈是权重常驻，不是高斯数 |
| PLY 体积 | 262,144 → 18 MB；1,048,576 → 69 MB |
| 权重占用 | `D:\wheel\hf_home_idol` 6.4 GB（idol 2.9 + dinov3 2.8 + FLUX vae 0.7） |

**复现验证**（`04_preview.sh` 六视角投影 + 肉眼核对）：蛋椅（薄壳椅背 + 星形腿）、机甲机器人
（面罩格栅 + 肩部武器）、甜甜圈（环面孔洞 + 糖针颜色）、玩具火车（轮组 + 车厢）、Q 版熊猫
（黑眼圈 + 绿竹）五类均结构自洽，无塌陷/重影。

⚠️ **与官方口径的差异**：官方 RTX 4090 报 19.1 s / 图，本机 3090 单张约 40 s（纯采样）。
未逐项对齐（未测 `ATTN_BACKEND=flash_attn` vs `sdpa` 的耗时差，虽然 flash-attn 2.8.3 已装）。
这个差距符合 3090/4090 的算力比，不视为复现问题。

## Notes

- **HF 直连在本机不通**（WSL 实测 `curl` 返回 000，`huggingface_hub` 反复 Connection reset）。
  `_env.sh` 会动态取网关 IP + 探测 10808 端口自动挂代理；网关 IP 每次开机可能变，不要写死。
- **`HF_HUB_DISABLE_XET=1` 是必须的**：装了 `hf-xet` 后大文件走 Xet 协议，在代理下只有 0.54 MB/s；
  关掉后走普通 CDN 达 1.7~2.3 MB/s（快 3 倍）。
- **`HF_HUB_ENABLE_HF_TRANSFER` 必须为 0**：hf_transfer 在「WSL + 宿主 HTTP 代理」下会卡死
  （2 分钟只落 129K，incomplete 文件长期 0 字节）。
- **FLUX.2-dev 全仓 30 GB+，只取 `vae/` 子目录**（~0.7 GB）。`01_download_models.sh` 用
  `allow_patterns=["vae/*","model_index.json"]` 限定，别改成全量 `snapshot_download`。
- **torch 装法**：`--find-links /mnt/d/wheel` 让本地 wheel 优先命中；**不能加 `--no-index`**，
  否则 `filelock` / `numpy` 等纯 Python 依赖解析直接失败（`D:\wheel` 里没有这些小包）。
- **依赖必须钉版本**：放开 `transformers>=4.56` / `tensordict>=0.8` 会被解析到 transformers 5.x +
  tensordict 新版，后者硬要求 `torch>=2.11`，把 torch 2.8 顶成 2.15（多下 540 MB 且与本地
  CUDA wheel 全不对齐）。脚本第 3 步装完会复查 torch 版本。
- **`tensordict` 必须 `--no-deps` 装**，代价是它的纯 python 依赖（`cloudpickle`）要手工补，
  否则挂 `No module named 'cloudpickle'`。
- **`torch.compile` 需要 C 编译器**：本机 WSL 无系统 gcc（apt 要 sudo 密码），用
  `conda install -n idol -c conda-forge gcc_linux-64 gxx_linux-64 -y`，它只提供
  `x86_64-conda-linux-gnu-gcc`，而 triton 硬找 `gcc` —— 靠 `_env.sh` 里的 `CC`/`CXX` 指定。
- **别用 `SIZE` 当环境变量名**：conda-forge 的 binutils activate 脚本会把 `SIZE` 注入成
  `x86_64-conda-linux-gnu-size` 的路径，取到的是垃圾值（`04_preview.sh` 因此改名 `PREVIEW_SIZE`）。
- **别在脚本运行中编辑该脚本**：bash 按字节偏移逐行读取正在执行的脚本，中途 Edit 会让它读到错位
  内容并报 `unexpected EOF while looking for matching '"'`（下载那 ~48 min 里就踩了一次，
  脚本本身语法是正确的）。
- `--weights` 传 `IDOL_MODELS`（软链目录即可，`from_pretrained` 走 `os.path.isdir` 分支，不再联网）。
- 首次推理含 `torch.compile`，慢；同一进程后续图才恢复到全速。
- 显存：官方口径峰值 7.2 GB；本机实测 9.46 GB（含 CUDA context）。3090 24 GB 余量充足。

## 参考

- 官方仓：https://github.com/speridlabs/idol-sdk
- HF Space：https://huggingface.co/spaces/speridlabs/idol
- 权重：https://huggingface.co/speridlabs/idol
- 基于 TripoSplat；用 DINOv3、FLUX.2 VAE、BiRefNet
