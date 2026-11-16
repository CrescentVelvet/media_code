# H100 集群使用指南（ca-uae / 迪拜）

> 账号 `00950754` · 登录 `ssh h100` · 实测记录 2026-11-13
> 节点 8×H100 PCIe 80GB · 192 核 · 1.35T 内存

## ⭐ 最常用命令（先看这里）

```bash
ssh h100
cd /data/00950754/code/media_code/canada_h100
chmod +x *.sh *.py    # 首次在新 clone 里必做，否则 ./12_sa_train.sh 报 Permission denied
```

| 我要做什么 | 命令 |
|---|---|
| **起训练（8 卡 DDP）** | `./12_sa_train.sh start` — 8 卡 / 168h / 全参微调 SplatAnything / 25 万步 |
| **看卡（常驻监控）** | `python3 11_gpu_monitor.py` — 显存 MEM + 算力 UTIL + 进程表，1s 刷新，**只有 Ctrl-C 能退** |
| 看队列 + 显卡快照 | `./12_sa_train.sh stat` |
| 跟训练日志 | `./12_sa_train.sh tail` |
| **逐卡指标汇总** | `./12_sa_train.sh metrics` — 各 rank 的 loss / 步速 / 显存 / util / 功耗 |
| **停止训练** | `./12_sa_train.sh stop` → **再跑 `./12_sa_train.sh ls` 复查** |
| 关掉某一个作业 | `scancel <JOBID>`；按名批量 `scancel --name=sa_train` |

**最省心的用法：监控开一个终端常驻，另一个终端随便折腾。**

```bash
# 终端 1：开着就不管了
python3 11_gpu_monitor.py

# 终端 2：start / stop / sbatch 随便来，终端 1 都不会断
./12_sa_train.sh start
./12_sa_train.sh stop
```

监控会自动跟随「当前占用 GPU 最多的作业」：训练在跑就跟训练，作业全没了就显示「等待中」，来新作业自动接上。**它只在你按 Ctrl-C 时退出。**

训练的常用变体：

```bash
./12_sa_train.sh start                  # 8 卡 / 168h / 全参 / 25 万步
./12_sa_train.sh start 4 12h gshead     # 4 卡 / 12h / 只训解码头（省显存，约 33M 参数）
./12_sa_train.sh scene <场景目录>        # 固定单场景（默认 examples/ 下 6 个按 rank 轮转）
SA_ITERS=30 SA_ROTATE=15 ./12_sa_train.sh start 2 1h   # 2 卡 30 步冒烟
./12_sa_train.sh metrics                # 最近一次运行的逐卡汇总（loss/步速/显存/util/功耗）
```

监控的常用变体：

```bash
python3 11_gpu_monitor.py -i 0.5      # 0.5s 刷新（最小 0.2s）
python3 11_gpu_monitor.py --once      # 只打一次，适合脚本里取数
python3 11_gpu_monitor.py --jobid <ID>  # 锁定某个作业，它没了就干等、不换台
```

> ⚠️ **停止两步走**：`stop` 只 `scancel` 当前作业，跑完一定再 `ls` 复查一次，
> 若还有 RUNNING 就带 JOBID 再停一遍。详见 §4.4。
>
> ⚠️ **登录节点没有 GPU**，`nvidia-smi` 报 `Failed to initialize NVML` 是正常的。看卡一律用
> `python3 11_gpu_monitor.py`（它会自己 `srun` 进计算节点采样）。
>
> ⚠️ `stop` **只 `scancel` 作业，不杀任何本地进程**。监控是常驻的，与作业生命周期解耦。

参数细节：训练见 §4，监控见 §4.2。

## 目录约定（2026-11-13 定）

**本目录 `canada_h100/` 只放「这台机器本身」的运维脚本**：建环境、体检、训练、监控、存储策略。

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

日常训练与监控的命令见页首「最常用命令」速查表。

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

## 四、训练与监控

两个脚本，配套用：`12_sa_train.sh` 提交训练，`11_gpu_monitor.py` 实时看卡。

### 4.1 训练：`12_sa_train.sh`

作业形态：**单节点 8 卡，torchrun 起 8 个进程，NCCL 数据并行**。每个 rank 读一个真实场景
（`examples/` 下 `aerial` / `indoor` / `object` / `pano` / `rig` / `street`，8 个 rank 按序轮转），
每步 `all-reduce` 平均梯度后各自 `optimizer.step()`，快照只在 rank0 写。

```bash
cd /data/00950754/code/media_code/canada_h100
./12_sa_train.sh start                  # 8 卡 / 168h / 全参 / 25 万步（推荐）
./12_sa_train.sh start 4 12h gshead     # 4 卡 / 12h / 只训解码头
./12_sa_train.sh scene /path/to/images  # 固定单场景
./12_sa_train.sh stat                   # 队列 + 显卡，一条命令看全
./12_sa_train.sh ls                     # 我的训练作业
./12_sa_train.sh tail                   # 跟日志
./12_sa_train.sh metrics                # 逐 rank 汇总：loss/步速/显存/util/功耗
./12_sa_train.sh stop                   # 停掉训练作业（之后一定再 ls 复查）
```

`start` 的参数位：`[gpus] [hours] [scope]`，默认 `8 168h all`。

**规模与预算**：全参 9.42 亿参数（encoder + gs_head + 冻结的几何先验），
每 rank 稳态显存 **18.6 GiB 已用 / 65.9 GiB 预留**，实测 `util 100%`、`约 270~300 W/卡`、
`1.18 s/step`。8 卡 25 万步约 **82 小时**，落在 168h 时限内自然结束，不续期。

**任务致死只有两种原因**：跑完（退出码 0）或外部 `scancel`。丢卡、OOM 都会直接退出并
在 `.out` 里留 traceback，不会静默重试。

环境变量（全部 `SA_` 前缀）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `SA_SCENE` | 空 | 单场景 `images/` 目录；给了就 8 卡共用 |
| `SA_SCENES` | 空 | 逗号分隔多场景；空 = `examples/` 下 6 个按 rank 轮转 |
| `SA_RES` | `504` | 分辨率，**必须是 14 的倍数**（patch size） |
| `SA_NVIEWS` | `8` | 每步视角数（场景图多时等间隔均匀抽样到这个数） |
| `SA_ROTATE` | `500` | 每 N 步轮换一次视角窗口 |
| `SA_ITERS` | `250000` | 总步数，跑完自然结束 |
| `SA_LR` | `1e-5` | 学习率 |
| `SA_SCOPE` | `all` | `all` 全参 / `gshead` 只解码头 |
| `SA_CKPT_EVERY` | `20000` | 快照间隔（写 `<rundir>/ckpt.pt`，覆盖式） |
| `SA_RESUME` | `0` | `1` = 从 `<rundir>/ckpt.pt` 续训 |
| `SA_SEED` | `1234` | 随机种子 |

### 4.2 监控：`11_gpu_monitor.py`

**监控是常驻独立程序，与任何作业无关。** 开一个终端跑着，然后随便 stop / start / sbatch，它都不会退出，只会自己换台跟。

```bash
python3 11_gpu_monitor.py                    # 常驻监控，1s 刷新，只有 Ctrl-C 能退
python3 11_gpu_monitor.py -i 2               # 2s 一次（最小 0.2s）
python3 11_gpu_monitor.py --jobid 201362394  # 锁定某个作业（它没了就干等，不换台）
python3 11_gpu_monitor.py --once             # 只打一次（无作业时返回码 2）
python3 11_gpu_monitor.py --rescan 5         # 无作业时每 5s 扫一次队列（默认 3s）
python3 11_gpu_monitor.py --no-procs         # 不显示进程表（快一点）
python3 11_gpu_monitor.py --no-bar           # 纯文本（不画进度条，好复制）
python3 11_gpu_monitor.py --log mon.jsonl    # 顺带落盘
```

**跟随规则**（不加 `--jobid` 时）：扫 `squeue` 找一个 **RUNNING 且分到 GPU** 的作业，按 `(GPU 张数, 作业号)` 倒序取第一个。所以：

| 你的操作 | 监控的反应 |
|---|---|
| 作业结束 / `scancel` | 采样器报 `STEP CANCELLED` → 转「等待中」，**进程不退出** |
| 提交新作业 | 最多 `--rescan` 秒后自动接上新作业 |
| `sbatch` 别的作业 | 同样自动接上；同时在跑就跟 GPU 多的那个 |
| 采样器崩 / 环境失效 | 退避重试（2s→30s 封顶），**进程不退出** |
| `Ctrl-C` | 唯一能结束监控的操作 |

> ⚠️ `--jobid` 是**锁定**语义：那个 ID 不在 RUNNING 就一直等，绝不偷偷换到别的作业。要「跟着当下占 GPU 的作业」就别加这个参数。

输出（三块：合计行 + 每卡表 + 进程表）：

```
[2026-11-14 12:07:22] h100a011 · job=201366403 (sa_train,8卡) @ h100a011 · RUNNING · 刷新 1.0s
数据来源: pynvml @ 计算节点 h100a011

  合计  8 卡 · 显存 534.8G/637.2G (84%) · 平均 UTIL 100% · 功耗 2210W
  ──────────────────────────────────────────────────────────────────────────
  GPU  型号/状态      UTIL  算力占用             MEM%  显存占用            显存GiB      温度  功耗
  ──────────────────────────────────────────────────────────────────────────
  0    H100 80G        100% ██████████████        84%  ████████████░        68.5/79.6    68C   288/350W
  1    H100 80G        100% ██████████████        84%  ████████████░        68.5/79.6    69C   286/350W
  ...

  PROCS  8 个进程
  ──────────────────────────────────────────────────────────────────────────
  GPU  PID       USER        进程名                            显存
  0    3964906   00950754    python                            68.5G
```

无作业时的样子：

```
[2026-11-14 12:09:59] h100a011 · 无占用 GPU 的作业 · 每 2s 重扫队列

  作业 201366403 已结束/被取消，等待新作业…

  监控常驻：执行 start / stop / sbatch 都不影响本面板，
  检测到占 GPU 的作业会自动接上。Ctrl-C 退出监控。
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

### 4.3 提交与停止的操作手册

**启动**（一条命令，默认 8 卡 / 168h / 全参 / 25 万步）：

```bash
cd /data/00950754/code/media_code/canada_h100
./12_sa_train.sh start
```

看到 `Submitted batch job <JOBID>` 就是提交成功。等 30~60 秒（8 个进程要各自加载 1322 个
张量），然后：

```bash
./12_sa_train.sh ls          # 确认 RUNNING
./12_sa_train.sh tail        # 看是否出现 "[rank0/gpu0] 开始训练"
./12_sa_train.sh metrics     # 各 rank 的 loss / 步速 / 显存 / util
```

**停止**：

```bash
# ① 停掉作业
./12_sa_train.sh stop

# ② 立刻复查——scancel 后作业会先进 COMPLETING，还会在 squeue 里挂一会儿
./12_sa_train.sh ls

# ③ 若还有 RUNNING/PENDING，带 JOBID 再停一次
./12_sa_train.sh stop <JOBID>
```

**手动兜底**（`12_sa_train.sh` 本身出问题时）：

```bash
squeue -a -u $USER                       # ⚠️ 必须加 -a，见 §3 坑 1
scancel <JOBID>                          # 停单个
scancel --name=sa_train                  # 按作业名批量停（sbatch 里的 -J 名）
scontrol show job <JOBID> | grep -i time # 确认到期时间
```

**确认真的停了**：`squeue -a -u $USER` 输出为**空**，且 `./12_sa_train.sh stat` 的显卡那段
回落到 `0% / 个位数 W / 0 GiB`。两个条件都满足才算停干净（GPU 显存释放有几十秒延迟，别急）。

### 4.4 DDP 实现要点（改脚本前必读）

**1. 只用 6 个真实场景给 8 个 rank 分**，`scene_path = scenes[rank % len(scenes)]`。
所以 rank0 与 rank6、rank1 与 rank7 读同一个场景——日志里这两对 loss 会完全相同，
这是**梯度同步生效的证据**，不是 bug。

**2. 梯度按 128 MB 分桶 all-reduce**，不能一次性 `all_reduce` 全参梯度（9.4 亿参数 ≈ 7.5 GiB
的梯度缓冲），否则显存峰值会顶穿。实现用 `torch._utils._flatten_dense_tensors` /
`_unflatten_dense_tensors` 做桶内拍平与还原，每桶 `all_reduce(SUM)` 后 `div_(world_size)`
取平均。

**3. NCCL 句柄必须用物理卡号，不能用 local_rank**。`CUDA_VISIBLE_DEVICES` 只影响 torch，
**NVML / `nvidia-smi` 完全不认它**（实测 CVD=1 时 `--query-gpu` 仍输出全部 8 行，且
`out[0]` 恒为物理卡 0）。所以日志里的 util/功耗要走 `phys_gpu(local_rank)` 映射回物理号再取句柄。
这条坑在旧版一卡一进程时曾让 8 卡日志全线读到卡 0 的数。

**4. 快照只在 rank0 写，并且前后各一道 `dist.barrier()`**。`ckpt.pt` 是覆盖式的（全参
1 份约 11.3 GB），`--resume` 从它恢复；所有 rank 的 optimizer 状态由 DDP 保证一致，
只存 rank0 即可。

**5. 每步顺序不能改**：

```
zero_grad → forward → gsplat 渲染 → MSE 光度 loss → backward
        → sync_grads（NCCL all-reduce 平均）→ clip_grad_norm → opt.step
```

`sync_grads` 必须在 `opt.step()` 之前、`clip_grad_norm` 之前，否则各 rank 会用不同的
梯度走下去，几百步后模型就发散。

**6. 启动方式固定为 `torchrun --standalone --nnodes=1 --nproc_per_node=$NGPU`**，
`NGPU` 取自 `SLURM_GPUS_ON_NODE`。不要改回 `python 12_sa_train.py`——单进程模式下
`init_process_group` 会退化成 world_size=1，8 张卡只有一张真正在算。

## 4.5 训练内容：`12_sa_train.py`

### 跑的是什么

```
8 视角图像 → encoder(RoPE ViT) → gs_head 出高斯参数 → 体素合并
          → gsplat 可导光栅化 → 渲染 RGB → MSE 光度 loss → backward
          → NCCL all-reduce 平均梯度 → AdamW
```

学生网络 `GaussianWorldMirror`（全参 9.42 亿）+ 冻结的几何先验 `GeometryWorldMirror`。
数据是 `SplatAnything/examples/` 下 6 个真实场景的图片，**不需要额外准备数据集**。

固定 **25 万步**（约 1.18 s/step → 约 82 小时）后自然结束。

### 实测数据（2026-11-14，8×H100 PCIe，DDP）

| 项 | 8 卡 DDP 全参 | 2 卡 DDP 全参 | 1 卡全参（旧一卡一进程） |
|---|---|---|---|
| 可训参数 | 942,046,860 | 同 | 同 |
| 显存（已用/预留） | 18.6 / 65.9 GiB | 18.6 / 65.9 GiB | 63.6 峰值 |
| NVML 单卡占用 | 68.5 / 81.5 GiB | — | 63.6 GiB |
| util | **100%**（8 卡全满） | 100% | 100% |
| 功耗 | 235~301 W/卡 | 321~323 W/卡 | — |
| 步速 | 1184~1187 ms/step | 1000 ms/step | 963 ms |
| loss | 0.0080 → 0.00364（30 步短测） | → 0.0024 | → 0.00637 |

`gshead` 范围（只训解码头，≈33M 参数）更快更省：426 ms/step。

### 视角窗口是怎么定的

`build_views` 会读**全部**图片（object 23 张、rig 150 张），但全参训练时 encoder 激活随
视角数线性涨，23 视角就能把 80G 卡撑爆。所以 `--n-views` 对真实场景也生效：按等间隔
**均匀抽帧**，既保住 360° 覆盖又把激活压回预算。日志会打 `视角抽样 23 → 8`。

视角窗口每 `--rotate-every`（默认 500）步在场景内换一组，窗口起始位按 `n_all // n_views`
步进，保证 360° 都被覆盖到。

### 日志长什么样

```
[rank0/gpu0] 场景 aerial：100 视角，504x504
[rank0/gpu0] 可训 942,046,860 参数（all）｜场景 aerial｜100 视角
[rank0/gpu0] 开始训练：iters=250000 lr=1e-05 world=8
[rank0/gpu0] 视角窗口 #1（8 视角）
[rank0/gpu0] step     20/250000｜loss 0.003378（EMA 0.005037，best 0.003378）｜1184 ms/step｜0.84 it/s｜剩 249980 步 / 约 82.2 h｜显存 18.6/65.9 GiB｜util 100%｜  235W｜α 0.973
```

每行都带 `util` 和功耗是**故意的**：外部 `nvidia-smi` 看到的是一回事，训练日志里逐步打印
util/功耗是另一回事 —— 后者是「这个 rank 此刻真在算」的直接证据，且按物理卡号取值，
不会被 `CUDA_VISIBLE_DEVICES` 骗。

### 为什么现在用 DDP

一卡一进程（8 份互不相关的副本）原来是为了「每张卡都有负载」。但那本质上是 8 个独立实验，
不是一次训练。改成 DDP 后：

- 8 个 rank 共享同一份模型状态，等效 batch 是单卡的 8 倍，梯度是真平均过的；
- 每步多出一次梯度 all-reduce（本机 PCIe 互联，实测 22.8 GB/s，见 §5），8 卡实测
  1184 ms/step vs 2 卡 1000 ms，**额外开销约 18%**，换来的是真正的一次训练而不是八次；
- 容错变差（一个 rank 挂全组退出），但换来的是可复现、可续训、指标有意义的单条训练曲线。


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
| `11_gpu_monitor.py` | 显卡监控（算力 / 显存 / 温度 / 功耗 / 进程表，远端 pynvml 采样） |
| `12_sa_train.py` | **SplatAnything 全参训练**（8 卡 NCCL 数据并行，25 万步） |
| `12_sa_train.sbatch` | 训练的提交脚本（由 `12_sa_train.sh` 调用，torchrun 起 N 进程） |
| `12_sa_train.sh` | 训练的登录节点入口（start / scene / ls / stat / metrics / stop / tail） |
| `STORAGE.md` | 存储策略详解 |
| `README.md` | 本文件 |
