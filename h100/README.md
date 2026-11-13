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

日常最常用的两条：

```bash
cd /data/00950754/repos
./10_hold.sh start        # 占卡（8 卡 24h）
./10_hold.sh stat         # 看队列 + 看卡，一条命令
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

## 四、保活（占卡）与监控

两个脚本，配套用：`10_hold.sh` 提交保活作业，`11_gpu_monitor.py` 实时看卡。

### 4.1 保活：`10_hold.sh`

```bash
cd /data/00950754/repos
./10_hold.sh start              # 8 卡 / 24h / high 强度
./10_hold.sh start 4 12 mid 20  # 4 卡 / 12h / mid / 每卡垫 20G 显存
./10_hold.sh stat               # 队列 + 显卡，一条命令看全
./10_hold.sh ls                 # 我的 hold 作业
./10_hold.sh tail               # 跟日志
./10_hold.sh stop               # 停掉全部 hold 作业
```

`start` 的参数位：`[gpus] [hours] [intensity] [memgib]`，默认 `8 24 high 0`。

**开销**：8 卡 high 实测 `util 99~100%`、`347W/卡`、`20.8 GiB 显存`（垫 20G 时）。8 卡满功耗约 2.8 kW，别长时间开 high 白烧电——除非就是要占住。

**强度只调占空比，不调 kernel 大小**：H100 上 matmul 连续在跑就顶到 340~350W，把方阵从 4096 缩到 768 对功耗几乎没影响（实测 mid 用小 kernel 仍是 347W）。占空比实测值（日志里 `实测占空比=` 一栏，CUDA event 累加出来的真值）：

| 档位 | 标称 | 实测 | 备注 |
|---|---|---|---|
| `high` | 100% | 100% | 8 卡 347W/卡，满负载 |
| `mid` | 75% | **76%** | 354W，与 high 差别很小 |
| `low` | 35% | **37%** | 时长一半 350W、一半 93W，**均功耗约 210W** |

⚠️ `low`/`mid` 在 `util` 那一列仍会在 0% 和 100% 之间跳（那是 `nvidia-smi` 采样窗的锅，不是真没在算）——**看 `实测占空比` 那一列才是准的**。

**链式续期**：作业到期前 120 s 自己 `sbatch` 下一个同类作业，实现"永不掉卡"。`./10_hold.sh start` **默认开**（想单次跑完：`HOLD_CHAIN=0 ./10_hold.sh start`）；直接 `sbatch 10_hold_gpu.sbatch` 则默认关。崩溃退出时**不会**续期。

环境变量（全部 `HOLD_` 前缀）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `HOLD_INTENSITY` | `high` | `low` / `mid` / `high` |
| `HOLD_SIZE` | `4096` | matmul 方阵边长 |
| `HOLD_DTYPE` | `bf16` | `bf16` / `fp16` / `fp32` |
| `HOLD_MEM_GIB` | `0` | 每卡额外垫多少 GiB 显存（占显存用） |
| `HOLD_INTERVAL` | `30` | 状态打印间隔（秒） |
| `HOLD_CHAIN` | `0` | `1` = 开链式续期 |

输出示例（日志里 `实测占空比` 是 CUDA event 累加的真实忙时占比，`util` 是 `nvidia-smi` 的采样窗读数，后者会跳）：

```
[gpu0] CUDA Graph 就绪：64 次 matmul/图，单图 0.018s（单次 0.29 ms）
[gpu0] 每 burst replay 54 次（约 0.99s 忙），睡 0.33s  → 标称占空比 75%
[gpu0] t=42s util=100%(近4次均 90%) 实测占空比=75% mem=20.8/79.6GiB temp=58C power=347W iters=100224
```

⚠️ **必须 `HOLD_` 前缀**：裸 `SIZE` 会被 conda binutils 激活钩子覆盖成 `x86_64-conda-linux-gnu-size`，python 直接报 `invalid int value`。

**产物**：日志 `/data/00950754/logs/hold_<JOBID>.out`；采样 `/data/00950754/runs/hold/job_<JOBID>/gpuN.jsonl`（带 `duty` 字段便于事后核算）。

### 4.2 监控：`11_gpu_monitor.py`

```bash
python3 11_gpu_monitor.py                    # 实时刷新，5s 一次，Ctrl-C 退出
python3 11_gpu_monitor.py --interval 2       # 2s
python3 11_gpu_monitor.py --once             # 只打一次
python3 11_gpu_monitor.py --jobid 201362394  # 指定作业（默认自动找最新的）
python3 11_gpu_monitor.py --probe            # 没作业时临时排队探针取数（约 3.5s 一次）
```

输出：

```
  GPU 型号                  算力   算力占用       显存   显存占用       显存GiB     温度  功耗
  ------------------------------------------------------------------------------------
  0   NVIDIA H100 PCIe      100%   ████████████   26%    ███░░░░░░░░░   20.8/80     64C   347/350W
  1   NVIDIA H100 PCIe      100%   ████████████   26%    ███░░░░░░░░░   20.8/80     61C   347/350W
```

**取数路径**（自动选）：
1. 已在计算节点（有 `SLURM_JOB_ID`）→ 直接 `nvidia-smi`
2. 在登录节点 → `srun --jobid=<ID> --overlap -n1 nvidia-smi`（借用已有作业的资源切片，不重新排队）
3. 加 `--probe` → 临时 `srun -p h100a011 --gres=gpu:1 --overlap` 起个 3.5s 的探针

### 4.3 保活脚本踩过的坑（都已修，留档防复发）

**坑 A：8 卡任务全挤在卡 0**。`multiprocessing` spawn 的子进程继承父进程 `CUDA_VISIBLE_DEVICES="0,1,..."`，每个子进程里的 `cuda:0` 都是物理卡 0。必须在 `import torch` **之前**把环境变量收窄成单卡号，并加 `device_count()==1` 自检。

**坑 B：`util` 读数在 0% 和 100% 之间乱跳**。`nvidia-smi` 的 `utilization.gpu` 是"**最近一次查询到当前**这一窗口内，有 kernel 在跑的时间占比"——采样点落在睡眠段就恒读 0%。做法是采样点选在 burst 刚提交完、`synchronize` 之前。注意 `nvidia-smi -i GPU-<uuid>` 会报 `No devices were found`（torch 给的 uuid 不带 `GPU-` 前缀），所以用数字索引。

**坑 B-2（曾让 8 卡日志全线造假）：`nvidia-smi` 根本不认 `CUDA_VISIBLE_DEVICES`**。原以为 worker 里 CVD 收窄成单卡后不带 `-i` 就只列一张卡 —— 实测 **CVD=1/2/3 时 `--query-gpu` 仍输出全部 8 行，且 `out[0]` 恒为物理卡 0**。于是 8 个 worker 全部读到卡 0 的数（这就是 8 卡日志 power 几乎一样的原因）。必须显式 `-i <物理卡号>`，索引与 `torch.cuda.get_device_properties(0).uuid` 对齐验证过无漂移。

**坑 D：同步调 `nvidia-smi` 会把 GPU 饿死**。单次 `nvidia-smi` 开销实测 **0.60~1.23 s**（NVML 初始化 + PCI 枚举），比一次 CUDA Graph replay（19~48 ms）长 20 倍。在主循环里同步采样，等于每轮都有 1 秒多不提交 kernel，日志就恒读 `0% / 10xW`，而外部 `nvidia-smi` 同期看到的是 `100% / 349W`。修法：采样挪到**守护线程**，主循环只读共享变量；线程里 `subprocess` 等待会释放 GIL，不挡提交。修后单卡实测 `util=100% 实测占空比=94~99% power=346W`，与外部观测一致。

**坑 C（最隐蔽）：Python 喂不饱 GPU，占空比形同虚设**。`mid` 标称 75%，实测只有 **17%**、功耗 128W。根因不是睡眠算法，而是 **Python 每秒只能提交约 490 次 matmul**（每次 GPU 侧约 0.3ms），GPU 每秒只拿到约 0.17s 的活就断了。把睡眠时长怎么折算都没用。

解法是 **CUDA Graph**：把 64 次 matmul 录成一张图，之后每次 `replay()` 只需一次 CPU 调用，GPU 端拿到背靠背的 kernel 序列。实测单图 18ms，`mid` 从 17% → **76%**。录制失败会自动退回逐次提交模式（能跑，但占空比偏低）。

相关数字：`4096` 方阵 bf16，单次 matmul 约 0.29ms（GPU 侧），H100 PCIe 算力约 27 TFLOPS（实测，非峰值）。

## 五、GPU 拓扑（影响多卡策略）

8×H100 **PCIe**，不是 NVSwitch 全互联：

- 只有 GPU 对（0-1、2-3、4-6、5-7）走 `NV12` NVLink
- 跨对全部走 PCIe
- NCCL all-reduce 实测 **22.8 GB/s** @256MB（全 NVSwitch 的 H100 能到 300+）

**结论**：8 卡训练用 **FSDP / ZeRO**，不要用纯 DDP（DDP 每步都要 all-reduce 梯度，PCIe 是瓶颈）。

## 六、存储

详见 `STORAGE.md`。一句话：

- **正在训的数据集 → `/data/00950754/datasets/`**（SSD 2.0 GB/s）
- **原始/归档 → `/data/sata/00950754/`**（HDD 179 MB/s，107 T）
- **家目录只有 51 G，不放数据**

## 七、网络

**服务器在迪拜。** 国内镜像反而慢（清华 12 s 超时），用官方源：

```bash
pip install -i https://pypi.org/simple <pkg>
pip install --index-url https://download.pytorch.org/whl/cu126 torch
```

HF 直连可用（0.069 s），**不需要镜像**。GitHub 直连 0.034 s。

## 八、文件清单

| 文件 | 用途 |
|---|---|
| `00_setup_env.sh` | 建 conda 环境（可改 `ENV_NAME` / `PY_VER` / `TORCH_VER` 复用） |
| `activate_env.sh` | 环境激活脚本（含 gcc 别名、glm、arch 设置） |
| `01_verify_env.py` | 环境体检：GPU / gsplat JIT / 渲染 / 反传 |
| `01_verify_env.sbatch` | 体检的提交脚本 |
| `10_hold.sh` | 保活 CLI：`start` / `ls` / `stat` / `stop` / `tail` |
| `10_hold_gpu.py` | 保活负载本体（多进程 matmul + 占空比 + 链式续期） |
| `10_hold_gpu.sbatch` | 保活的提交脚本（由 `10_hold.sh` 调用，也可直接 `sbatch`） |
| `11_gpu_monitor.py` | 显卡监控（算力占用 / 显存占用 / 温度 / 功耗） |
| `STORAGE.md` | 存储策略详解 |
| `README.md` | 本文件 |
