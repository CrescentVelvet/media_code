# H100 集群存储策略（ca-uae / 迪拜）

> 实测日期 2026-11-13。所有数字都是 `dd` 实测 + `df` 实查，不是文档抄的。

## 一、先回答"数据集放哪儿"

**分三层放，不要一股脑塞进 `/data`。**

| 层 | 路径 | 容量 | 实测速度 | 放什么 |
|---|---|---|---|---|
| **热**（训练中随机读） | `/data/00950754/` | 912 G 可用 | SSD **2.0 GB/s 写 / 1.2 GB/s 读** | 正在训练的 dataset、checkpoint、代码、conda env |
| **温**（暂存/近期可能重跑） | `/data/sata/00950754/` | **107 T 可用** | HDD **179 MB/s 写 / 214 MB/s 读** | 原始采集包、已预处理但暂时不训的数据集、大 tar/zip |
| **冷**（归档） | `/data/sata/00950754/archive/` | 同上 | 同上 | 实验完的 checkpoint、原始素材长期保存 |

**核心判断**：两者差 **10 倍**（2.0 GB/s vs 179 MB/s）。3DGS 训练的 dataloader 每个 iter 要随机读几十张图，放 HDD 上会被 I/O 拖死——GPU 利用率掉到 30% 以下很常见。所以：

- **正在训的数据集 → 必须放 `/data`（SSD）**
- **训练完就搬走 → `/data/sata`**

## 二、为什么不能用 `/data` 顶层或共享目录

| 目录 | 状态 | 说明 |
|---|---|---|
| `/data/00950754/` | ✅ 可写 | 我的专属目录（ACL `user:00950754:rwx`） |
| `/data/sata/00950754/` | ✅ 可写 | 我的专属目录，107 T |
| `/data/conda/` | ❌ 只读 | 共享 conda 安装，连 `pkgs/` 都不可写 → 环境必须建到自己目录 |
| `/data/datasets/` | 👁 只读 | 168 G 共享数据集，但内容全是猫狗（catbreed / DogBreeds），跟人像 3DGS 无关 |
| `/data/checkpoints/` | 👁 只读 | 别人的 SCAIL-2 |
| `/mnt/shared/` | ❌ 无权限 | NFS 110 T，`ls` 都进不去 |
| `/data/sata/team-shared/` | ❌ 无权限 | `drwx--x--x root root` |
| `~`（家目录） | ⚠️ 只有 51 G | 和系统盘一体，**千万别放数据**，塞满会连 ssh 都登不上 |

**没有配额系统**（`mount` 显示 `noquota`），但 `/data` 已经 75% 满（2.7T/3.5T），所以：
- 在 `/data` 上要自觉，别囤不用的 checkpoint
- `/data/sata` 有 107 T，可以放心用

## 三、推荐的目录结构

```
/data/00950754/                      # SSD，2.0 GB/s
├── envs/splatanything/              # 7.7 G  conda 环境（用 -p 建的）
├── conda_pkgs/                      # conda 包缓存
├── .pip_cache/                      # pip 缓存
├── .torch_extensions/               # gsplat JIT 编译缓存（删了会重新编 81s）
├── repos/                           # 代码仓
│   ├── 00_setup_env.sh
│   ├── 01_verify_env.py
│   ├── 01_verify_env.sbatch
│   └── glm_src/                     # gsplat 编译要的 glm 头文件
├── datasets/                        # ★ 正在训练的数据集（热数据）
│   └── <scene_name>/
├── runs/                            # ★ 训练输出：checkpoint + tensorboard
│   └── <exp_name>/
└── logs/                            # sbatch 日志

/data/sata/00950754/                 # HDD，179 MB/s
├── datasets/                        # 原始采集包、暂不训练的数据集
│   └── raw/<capture_id>/
└── archive/                         # 归档
    └── <exp_name>_<date>/
```

## 四、工作流：数据怎么在两层之间流动

```bash
# 1) 原始数据先落 SATA（省 SSD，也不怕大）
rsync -av /path/to/capture/ /data/sata/00950754/datasets/raw/<id>/

# 2) 要开始训了，把「这一个 scene」搬到 SSD
rsync -av /data/sata/00950754/datasets/raw/<id>/ /data/00950754/datasets/<id>/

# 3) 训练输出写 SSD（checkpoint 频繁写，HDD 会卡）
#    sbatch 脚本里 --out /data/00950754/runs/<exp>

# 4) 训完：checkpoint 归档到 SATA，SSD 上的数据集删掉腾地方
mv /data/00950754/runs/<exp> /data/sata/00950754/archive/<exp>_$(date +%Y%m%d)
rm -rf /data/00950754/datasets/<id>       # ⚠️ 确认 SATA 上有副本再删
```

### 数据量大到 SSD 放不下怎么办

如果单个 scene 预处理后超过 ~50 G，两个办法：

1. **符号链接 + 分片**：把 `images/` 切分，热的那部分（训练采样频率高的视角）放 SSD，冷放 SATA，用 symlink 拼起来。dataloader 随机读时命中率会降，但比全放 HDD 好。
2. **预处理成 lmdb/webdataset**：把散图打成单文件顺序读，HDD 的 179 MB/s 顺序读能跑满，随机读的瓶颈就没了。这是大数据集的标准解法。

## 五、网络：别用国内镜像

**服务器在迪拜**（`ipinfo` 实测 `69.48.159.15 → Dubai, AE`，AS42156 GNX B.V.）。实测延迟：

| 源 | 响应 | 结论 |
|---|---|---|
| `pypi.org` | **2.2 s** | ✅ 用这个 |
| `download.pytorch.org` | **0.93 s** | ✅ 用这个（torch wheel 238 MB/s） |
| `mirrors.aliyun.com/pypi` | 1.7 s | 可用但没必要 |
| `pypi.tuna.tsinghua.edu.cn` | **12.0 s（超时）** | ❌ 别用，绕地球半圈 |
| `github.com` | 0.034 s | ✅ 直连 |
| `huggingface.co` | 0.069 s | ✅ **直连可用**（不用镜像！） |

所以：
- **不要**抄 `splatanything/_env.sh` 里那套代理 + `aifasthub` 镜像配置，那是给国内 WSL 用的
- HF 直连就够快，下权重直接用 `huggingface-cli download`
- 装包一律 `-i https://pypi.org/simple`，torch 用 `--index-url https://download.pytorch.org/whl/cu126`

## 六、磁盘写满的后果（务必避免）

`/`（含家目录）只有 98 G，已用 46%。如果塞满：
- ssh 登录可能失败（无法写 `.ssh/authorized_keys`、无法创建 session 文件）
- `sbatch` 提交会失败（slurm 要写 spool）
- conda 环境操作全挂

**已经做的防护**：conda pkgs → `/data/00950754/conda_pkgs`，pip cache → `/data/00950754/.pip_cache`，torch extensions → `/data/00950754/.torch_extensions`。都在 `activate_splatanything.sh` 里写死了。

**检查命令**：

```bash
df -h / /data /data/sata
du -sh /data/00950754/* | sort -rh | head
du -sh /data/sata/00950754/* | sort -rh | head
```
