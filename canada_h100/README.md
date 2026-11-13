# H100 集群使用指南（ca-uae / 迪拜）

> 账号 `00950754` · 登录 `ssh h100` · 实测记录 2026-11-13
> 节点 8×H100 PCIe 80GB · 192 核 · 1.35T 内存

## ⭐ 最常用命令（先看这里）

```bash
ssh h100
cd /data/00950754/code/media_code/canada_h100
```

| 我要做什么 | 命令 |
|---|---|
| **占卡（启动保活）** | `./10_hold.sh start` — 8 卡 / 24h / high / 显存 90% / 自动续期 |
| **看卡（实时监控）** | `./10_hold.sh mon` — 显存 MEM + 算力 UTIL + 进程表，1s 刷新，Ctrl-C 退出 |
| 看队列 + 显卡快照 | `./10_hold.sh stat` |
| 跟保活日志 | `./10_hold.sh tail` |
| **停止保活** | `./10_hold.sh stop` → **再跑 `./10_hold.sh ls` 复查** |
| 关掉某一个作业 | `scancel <JOBID>`；按名批量 `scancel --name=hold_gpu` |

保活的三个常用变体：

```bash
./10_hold.sh start 4 12 high      # 4 卡 / 12h（其余默认）
HOLD_MEM_RATIO=0 ./10_hold.sh start   # 只占算力，不垫显存
HOLD_CHAIN=0 ./10_hold.sh start       # 不自动续期，跑完就退
```

监控的三个常用变体：

```bash
python3 11_gpu_monitor.py -i 0.5      # 0.5s 刷新（最小 0.2s）
python3 11_gpu_monitor.py --once      # 只打一次，适合脚本里取数
python3 11_gpu_monitor.py --no-procs  # 不显示进程表
```

> ⚠️ **停止必须两步**：`stop` 只杀当前作业，链式续期可能已经提交了下一个。跑完 `stop` 一定再 `ls` 复查，还有 RUNNING/PENDING 就 `./10_hold.sh stop <JOBID>`。否则典型症状是「我明明停了，十分钟后卡又满了」。详见 §4.4。
>
> ⚠️ **登录节点没有 GPU**，`nvidia-smi` 报 `Failed to initialize NVML` 是正常的。看卡一律用 `./10_hold.sh mon`（它会自己 `srun` 进计算节点采样）。

参数细节：保活见 §4.1，监控见 §4.2，启停操作手册见 §4.4。

## 目录约定（2026-11-13 定）

**本目录 `canada_h100/` 只放「这台机器本身」的运维脚本**：建环境、体检、保活、监控、存储策略。

**具体算法的复现脚本放各自算法目录**（`splatanything/`、`da3_human/`、`vggt_human/` …），
在里面按 AGENTS.md 的编号规范写 `00_setup_env.sh` / `01_download_models.sh` / `02_run_inference.sh`。
判断标准：**换台机器还能不能跑**——「h100a011 分区、隐藏队列、登录节点无 GPU」这类知识只在本目录，
「SplatAnything 要 100 step 冒烟」这类知识在算法目录。

服务器上的代码位置：`/data/00950754/code/media_code/canada_h100/`（本仓 clone，跟着 git 走）。
下面所有 `cd` 都指这里；脚本内部一律用 `$(dirname $0)` 解析自身目录，**不硬编码绝对路径**。

## 快速开始

```bash
ssh h100                                    # 已配好别名
source /data/00950754/envs/activate_splatanything.sh
cd /data/00950754/code/media_code/canada_h100
sbatch 01_verify_env.sbatch                 # 环境体检（约 1 分钟）
```

日常保活与监控的命令见页首「最常用命令」速查表。

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

cd /data/00950754/code/media_code/<algo>     # 算法目录，不是本目录
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
cd /data/00950754/code/media_code/canada_h100
./10_hold.sh start              # 8 卡 / 24h / high / 显存 90% / 自动续期（推荐，就是这条）
./10_hold.sh start 4 12 high    # 4 卡 / 12h / high
./10_hold.sh stat               # 队列 + 显卡，一条命令看全
./10_hold.sh ls                 # 我的 hold 作业
./10_hold.sh tail               # 跟日志
./10_hold.sh mon                # 起实时监控（= 11_gpu_monitor.py）
./10_hold.sh stop               # 停掉全部 hold 作业
```

`start` 的参数位：`[gpus] [hours] [intensity] [memratio]`，默认 `8 24 high 0.90`。

**默认就是 high，别传 low/mid。** `low`/`mid` 是早期为「占着卡但别烧电」设计的省电档，靠「跑一阵睡一阵」把均值功耗压到 210W 左右。它有两个致命缺点：

1. **`util` 读数会在 0% 和 100% 之间跳**，管理员那边看曲线像在反复掉线，观感比 full 满载差得多；
2. 本集群单人独占，省电没有任何收益，只是给自己制造"这卡到底在不在干活"的解释成本。

传了非 high 档脚本会打警告并 `sleep 3`，给你 3 秒 Ctrl-C 反悔。

**显存拉到 90%**：`--mem-ratio 0.90` 按卡容量自动算差额，分块 `torch.empty` 垫到目标。留 10% 余量是因为 CUDA context / cuBLAS workspace 还要占，顶到 100% 必 OOM。实测 H100 80G 卡垫到 **71.7/79.2 GiB（90%）**，`nvml` 记账确认生效，不是虚的。

| 写法 | 效果 |
|---|---|
| `./10_hold.sh start` | 默认 0.90，自动算 |
| `HOLD_MEM_RATIO=0.95 ./10_hold.sh start` | 更激进，风险自负 |
| `HOLD_MEM_RATIO=0 ./10_hold.sh start` | 不垫片，只占算力（约 21 GiB） |
| `./10_hold.sh start 8 24 high 40` | 每卡固定垫 40 GiB（绝对值覆盖比例） |

**开销**：8 卡 high + 90% 显存实测 `util 99~100%`、`347W/卡`、`71.7/79.2 GiB`。整机约 2.8 kW。

**强度只调占空比，不调 kernel 大小**：H100 上 matmul 连续在跑就顶到 340~350W，把方阵从 4096 缩到 768 对功耗几乎没影响。

**链式续期**：作业到期前 120 s 自己 `sbatch` 下一个同类作业，实现"永不掉卡"。`./10_hold.sh start` **默认开**（想单次跑完：`HOLD_CHAIN=0 ./10_hold.sh start`）；直接 `sbatch 10_hold_gpu.sbatch` 则默认关。崩溃退出时**不会**续期。

环境变量（全部 `HOLD_` 前缀）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `HOLD_INTENSITY` | `high` | 保持 `high`，理由见上 |
| `HOLD_MEM_RATIO` | `0.90` | 显存垫片目标占用比（留 10% 防 OOM） |
| `HOLD_MEM_GIB` | `0` | 每卡垫固定 GiB，`>0` 时覆盖 `HOLD_MEM_RATIO` |
| `HOLD_SIZE` | `4096` | matmul 方阵边长 |
| `HOLD_DTYPE` | `bf16` | `bf16` / `fp16` / `fp32` |
| `HOLD_INTERVAL` | `30` | 状态打印间隔（秒） |
| `HOLD_CHAIN` | `0` | `1` = 开链式续期 |

输出示例：

```
[gpu0] CUDA Graph 就绪：64 次 matmul/图，单图 0.018s（单次 0.29 ms）
[gpu0] 垫片 69.8 GiB → 显存占用 90%（71.7/79.2 GiB）
[gpu0] t=42s util=100%(均 99%, n=4) mem=71.7/79.2GiB(90%) temp=58C power=347W(均 346W) iters=100224
```

⚠️ **必须 `HOLD_` 前缀**：裸 `SIZE` 会被 conda binutils 激活钩子覆盖成 `x86_64-conda-linux-gnu-size`，python 直接报 `invalid int value`。

**产物**：日志 `/data/00950754/logs/hold_<JOBID>.out`；采样 `/data/00950754/runs/hold/job_<JOBID>/gpuN.jsonl`（字段 `duty_nominal` / `duty_from_nvml` / `util_avg` / `power_avg`）。

⚠️ **不要再信「实测占空比」**：旧版靠 `torch.cuda.Event` 的 `elapsed_time` 算忙时占比，但那是**GPU 时间轴上的跨度**，中途 GPU 空转也被算进忙时。旧日志报 `实测占空比 91%` 时，`nvml` 同时刻读到 `0% / 101W`。现在日志只报 `nvml` 的 `util`（带窗口均值），jsonl 里旧量降级为 `ev_span_s` 仅供参考。

### 4.2 监控：`11_gpu_monitor.py`

```bash
./10_hold.sh mon                             # 等价于下面第一条
python3 11_gpu_monitor.py                    # 实时刷新，1s 一次，Ctrl-C 退出
python3 11_gpu_monitor.py -i 2               # 2s 一次（最小 0.2s）
python3 11_gpu_monitor.py --once             # 只打一次
python3 11_gpu_monitor.py --jobid 201362394  # 指定作业（默认自动找最新的 RUNNING）
python3 11_gpu_monitor.py --no-procs         # 不显示进程表（快一点）
python3 11_gpu_monitor.py --no-bar           # 纯文本（不画进度条，好复制）
python3 11_gpu_monitor.py --log mon.jsonl    # 顺带落盘
```

输出（三块：合计行 + 每卡表 + 进程表）：

```
[2026-11-13 08:38:29] 登录节点 h100a011 · job=201362583(RUNNING) · 刷新 1.0s
数据来源: pynvml @ 计算节点 h100a011

  合计  8 卡 · 显存 574.5G/637.2G (90%) · 平均 UTIL 100% · 功耗 2780W
  ──────────────────────────────────────────────────────────────────────────
  GPU  型号/状态      UTIL 算力占用            MEM% 显存占用      显存GiB   温度 功耗
  ──────────────────────────────────────────────────────────────────────────
  0    H100 80G        100% ██████████████       90% █████████████░  71.8/79.6  66C  349/350W
  1    H100 80G        100% ██████████████       90% █████████████░  71.8/79.6  64C  346/350W
  ...

  PROCS  8 个进程
  ──────────────────────────────────────────────────────────────────────────
  GPU  PID       USER        进程名                  显存
  0    3957060   00950754    python                  71.3G
```

> `MEM%` 是**真实占用率**（已用/总量）。NVML 另有一个 `memory util`（显存控制器忙闲）字段，和占用率是两回事、并排看极易混淆，所以**表格里不显示**了，但 `--log` 的 jsonl 里仍保留 `mem_util`，需要时自己取。

**为什么不用 nvitop**：nvitop 的本体（`cli.py` / `tui/`）是 **GPL-3.0**，抄进仓库会传染整个项目；它的 API 层（`nvitop/api/`）虽是 Apache-2.0，但它依赖 `nvmlInit()` 直连 NVML，而**本集群登录节点没有 NVML**（`nvmlInit()` 报 `NVMLError_Unknown`）。两个理由一起决定了：**学它的原理，不抄它的代码**。

**架构：远端常驻采样器 + 本地渲染。**

```
登录节点                                             计算节点（作业内）
  python3 11_gpu_monitor.py
        │
        └─ srun --jobid=<ID> --overlap -n1 bash -c "source <env>; python -c 'exec(stdin)'"
                │  源码经 stdin 喂过去，只握手一次
                ▼
           常驻循环：pynvml 采样 → 每帧一行 JSON → stdout
                │
        ◄───────┘
   本地读管道 → 渲染表格 → 清屏重绘
```

**关键性能数字**：`pynvml`（`nvidia-ml-py`）进程内直连 NVML，实测 50 次（util+mem+temp+power）共 **4.7 ms ≈ 0.094 ms/次**；`nvidia-smi` 单次 **0.62~1.61 s**，相差约**一万倍**。这是能把刷新率压到 1s（甚至 0.2s）的唯一原因——旧版 5s 一次是因为每次都要付 `nvidia-smi` 的启动税。

⚠️ **计算节点的非交互 shell 没有 `python`**（`which python` → exit 127）。远端命令串必须先 `source` 环境脚本，默认 `/data/00950754/envs/activate_splatanything.sh`，可用 `--env-script` 改。改了环境路径没同步会看到「远端采样器已退出」，脚本会把远端 stderr 最后 3 行拼进提示。

**取数路径**（自动选）：
1. 已在计算节点（有 `SLURM_JOB_ID`）→ 直接本地 `pynvml`
2. 在登录节点 → `srun --jobid=<ID> --overlap -n1` 起远端采样器（借用已有作业的资源切片，不重新排队）
3. 加 `--native` → 退回 `nvidia-smi -lms` 流式模式（无 pynvml 时的兜底，8 卡相邻行间隔约 2 ms，全程 CPU 占用极低）

### 4.3 保活脚本踩过的坑（都已修，留档防复发）

**坑 A：8 卡任务全挤在卡 0**。`multiprocessing` spawn 的子进程继承父进程 `CUDA_VISIBLE_DEVICES="0,1,..."`，每个子进程里的 `cuda:0` 都是物理卡 0。必须在 `import torch` **之前**把环境变量收窄成单卡号，并加 `device_count()==1` 自检。

**坑 B：`util` 读数在 0% 和 100% 之间乱跳**。`nvidia-smi` 的 `utilization.gpu` 是"**最近一次查询到当前**这一窗口内，有 kernel 在跑的时间占比"——采样点落在睡眠段就恒读 0%。做法是采样点选在 burst 刚提交完、`synchronize` 之前。注意 `nvidia-smi -i GPU-<uuid>` 会报 `No devices were found`（torch 给的 uuid 不带 `GPU-` 前缀），所以用数字索引。

**坑 B-2（曾让 8 卡日志全线造假）：`nvidia-smi` 根本不认 `CUDA_VISIBLE_DEVICES`**。原以为 worker 里 CVD 收窄成单卡后不带 `-i` 就只列一张卡 —— 实测 **CVD=1/2/3 时 `--query-gpu` 仍输出全部 8 行，且 `out[0]` 恒为物理卡 0**。于是 8 个 worker 全部读到卡 0 的数（这就是 8 卡日志 power 几乎一样的原因）。必须显式 `-i <物理卡号>`，索引与 `torch.cuda.get_device_properties(0).uuid` 对齐验证过无漂移。

**坑 D：同步调 `nvidia-smi` 会把 GPU 饿死**。单次 `nvidia-smi` 开销实测 **0.60~1.23 s**（NVML 初始化 + PCI 枚举），比一次 CUDA Graph replay（19~48 ms）长 20 倍。在主循环里同步采样，等于每轮都有 1 秒多不提交 kernel，日志就恒读 `0% / 10xW`，而外部 `nvidia-smi` 同期看到的是 `100% / 349W`。修法：采样挪到**守护线程**，主循环只读共享变量；线程里 `subprocess` 等待会释放 GIL，不挡提交。

**坑 D-2（坑 D 的正解）：换 `pynvml` 后这个坑自动消失**。`pynvml` 是进程内直连 NVML，单次 **0.094 ms**，比 `nvidia-smi` 快约一万倍，同步调用也不会饿死 GPU。守护线程保留（代码结构不动），但最短采样周期从 1.0s 放宽到 **0.5s**。

**坑 C（最隐蔽）：Python 喂不饱 GPU，占空比形同虚设**。`mid` 标称 75%，实测只有 **17%**、功耗 128W。根因不是睡眠算法，而是 **Python 每秒只能提交约 490 次 matmul**（每次 GPU 侧约 0.3ms），GPU 每秒只拿到约 0.17s 的活就断了。把睡眠时长怎么折算都没用。

解法是 **CUDA Graph**：把 64 次 matmul 录成一张图，之后每次 `replay()` 只需一次 CPU 调用，GPU 端拿到背靠背的 kernel 序列。实测单图 18ms，`mid` 从 17% → **76%**。录制失败会自动退回逐次提交模式（能跑，但占空比偏低）。

相关数字：`4096` 方阵 bf16，单次 matmul 约 0.29ms（GPU 侧），H100 PCIe 算力约 27 TFLOPS（实测，非峰值）。

### 4.4 操作手册：自己启、自己停

**启动**（一条命令，默认 8 卡 / 24h / high / 显存 90% / 自动续期）：

```bash
cd /data/00950754/code/media_code/canada_h100
./10_hold.sh start
```

看到 `Submitted batch job <JOBID>` 就是提交成功。等 10~30 秒后 `./10_hold.sh mon` 确认 8 张卡都在 `100% / ~347W / 71.7 GiB`。想先不真跑，加 `--dry-run` 只打印配置。

**关闭**，按这个顺序做，别只 `scancel` 一个：

```bash
# ① 停掉保活链
./10_hold.sh stop

# ② 立刻复查：链式续期可能在 stop 的瞬间已经提交了下一个作业
./10_hold.sh ls

# ③ 如果 ② 还有 RUNNING/PENDING 的 hold 作业，再停一次（带 JOBID）
./10_hold.sh stop <JOBID>
```

⚠️ **为什么必须复查**：`--chain` 是「到期前 120 s 提交下一个」，提交完旧作业还在跑。`stop` 杀掉当前作业后，**新提交的那个已经是独立作业**，不受影响，必须单独停。漏掉这一步的典型症状是「我明明停了，十分钟后卡又满了」。

**手动保底**（`10_hold.sh` 本身出问题时）：

```bash
squeue -u $USER                          # 看有没有 hold 作业
scancel <JOBID>                          # 停单个
scancel --name=hold_gpu                  # 按作业名批量停（sbatch 里的 -J 名）
scontrol show job <JOBID> | grep -i time # 确认到期时间
```

**确认真的停了**：`squeue -u $USER` 输出为**空**，且 `./10_hold.sh stat` 的显卡那一段回落到 `0% / 个位数 W / 0 GiB`。两个条件都满足才算停干净（GPU 显存释放有几十秒延迟，别急）。

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
| `10_hold.sh` | 保活 CLI：`start` / `ls` / `stat` / `stop` / `tail` / `mon` |
| `10_hold_gpu.py` | 保活负载本体（多进程 matmul + 显存垫片 + 链式续期） |
| `10_hold_gpu.sbatch` | 保活的提交脚本（由 `10_hold.sh` 调用，也可直接 `sbatch`） |
| `11_gpu_monitor.py` | 显卡监控（算力 / 显存 / 温度 / 功耗 / 进程表，远端 pynvml 采样） |
| `STORAGE.md` | 存储策略详解 |
| `README.md` | 本文件 |
