# H100 集群使用指南（ca-uae / 迪拜）

> 账号 `00950754` · 登录 `ssh h100` · 实测记录 2026-11-13
> 节点 8×H100 PCIe 80GB · 192 核 · 1.35T 内存

## 快速开始

```bash
ssh h100                                    # 已配好别名
source /data/00950754/envs/activate_splatanything.sh
cd /data/00950754/repos
sbatch 01_verify_env.sbatch                 # 环境体检（约 1 分钟）
```

## 一、环境已建好

| 项 | 值 |
|---|---|
| 路径 | `/data/00950754/envs/splatanything`（7.7 G） |
| 激活 | `source /data/00950754/envs/activate_splatanything.sh` |
| Python | 3.11.18 |
| PyTorch | 2.7.1+cu126 |
| numpy | 1.26.4（**必须 <2**，SplatAnything 要求） |
| gsplat | 1.5.2（JIT 编译已验证：81.5 s 首次，之后走缓存 0.5 s） |
| CUDA 工具链 | cuda-nvcc 12.6 + conda gcc 12.4（系统只有 gcc 13、无 nvcc） |

**为什么环境建在 `/data/00950754/envs/` 而不是 `/data/conda/envs/`**：共享 conda 整个目录只读，`conda create -n` 会失败。必须用 `-p <前缀>` 建到自己的目录。

**为什么不用 `set -u`**：conda 的 `activate.d/activate-gcc_linux-64.sh` 引用未定义变量 `SYS_SYSROOT`，`set -u` 下激活即崩。sbatch 脚本里只写 `set -eo pipefail`。

## 二、跑训练的完整模板

```bash
#!/bin/bash
#SBATCH -J my_train
#SBATCH -p h100a011              # 分区（会自动路由到你的隐藏子分区）
#SBATCH --gres=gpu:8             # 8 卡
#SBATCH -c 32                    # CPU 核（dataloader 用）
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH -o /data/00950754/logs/%x_%j.out

set -eo pipefail                 # ⚠️ 不要加 -u
source /data/00950754/envs/activate_splatanything.sh

cd /data/00950754/repos/<your_repo>
python train.py --out /data/00950754/runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
```

提交 / 查看：

```bash
sbatch train.sh
squeue -a -u $(whoami)           # ⚠️ 必须加 -a，见下面「坑 1」
tail -f /data/00950754/logs/<jobname>_<jobid>.out
scancel <JOBID>
```

## 三、三个必须知道的坑

### 坑 1：裸 `squeue` 看不到自己的任务

集群给每个用户 × 每个节点建了一个**隐藏分区**（`Hidden=YES`），名字形如 `h100a011_u_00950754`。`sbatch` 提交后任务被路由到那里，而 `squeue` 默认跳过隐藏分区。

```bash
squeue -u 00950754    # ❌ 空
squeue                # ❌ 空
squeue -a             # ✅ 能看到
squeue -a -u $(whoami)
```

**一律用 `squeue -a`。**

### 坑 2：`sacct` / `sstat` 查不到历史

`slurmdbd` 记账服务未运行，`sacctmgr list assoc` 是 0 条记录。所以：
- 任务结束后**查不到**耗时、退出码、峰值显存
- **对策：日志里自己打全**（时间戳、显存、loss、最终指标）

### 坑 3：登录节点看不到 GPU

登录节点 `nvidia-smi` 报 `Failed to initialize NVML` 是正常的。要看 GPU 必须进计算节点：

```bash
srun -p h100a011 --gres=gpu:1 --pty bash      # 交互式开一台
srun --jobid=<JOBID> --overlap -n1 nvidia-smi # 看正在跑的任务的 GPU
```

另外：`scontrol show job` 显示 RUNNING 但 torch 还在 import 时，GPU 占用是 0，别误判。

## 四、GPU 拓扑（影响多卡策略）

8×H100 **PCIe**，不是 NVSwitch 全互联：

- 只有 GPU 对（0-1、2-3、4-6、5-7）走 `NV12` NVLink
- 跨对全部走 PCIe
- NCCL all-reduce 实测 **22.8 GB/s** @256MB（全 NVSwitch 的 H100 能到 300+）

**结论**：8 卡训练用 **FSDP / ZeRO**，不要用纯 DDP（DDP 每步都要 all-reduce 梯度，PCIe 是瓶颈）。

## 五、存储

详见 `STORAGE.md`。一句话：

- **正在训的数据集 → `/data/00950754/datasets/`**（SSD 2.0 GB/s）
- **原始/归档 → `/data/sata/00950754/`**（HDD 179 MB/s，107 T）
- **家目录只有 51 G，不放数据**

## 六、网络

**服务器在迪拜。** 国内镜像反而慢（清华 12 s 超时），用官方源：

```bash
pip install -i https://pypi.org/simple <pkg>
pip install --index-url https://download.pytorch.org/whl/cu126 torch
```

HF 直连可用（0.069 s），**不需要镜像**。GitHub 直连 0.034 s。

## 七、文件清单

| 文件 | 用途 |
|---|---|
| `00_setup_env.sh` | 建 conda 环境（可改 `ENV_NAME` / `PY_VER` / `TORCH_VER` 复用） |
| `activate_env.sh` | 环境激活脚本（含 gcc 别名、glm、arch 设置） |
| `01_verify_env.py` | 环境体检：GPU / gsplat JIT / 渲染 / 反传 |
| `01_verify_env.sbatch` | 体检的提交脚本 |
| `STORAGE.md` | 存储策略详解 |
| `README.md` | 本文件 |
