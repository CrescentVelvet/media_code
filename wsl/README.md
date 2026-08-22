# WSL2 + Ubuntu 环境搭建（Win11 + RTX 3090）

在 Windows 11 上用 WSL2 跑 Ubuntu 22.04，GPU 直通跑算法复现。C 盘固态 800GB 空闲，分 100GB 给 WSL vhdx；机械盘当仓库归档原始数据集与权重。

> 🎮 RTX 3090（24GB / Ampere / 算力 8.6），主流框架全支持。

## 0. 前置检查

```powershell
# Windows 端（管理员 PowerShell）
wsl --status
wsl -l -v
nvidia-smi            # 确认 3090 + 驱动正常
```

> ⚠️ 如果 `wsl --install` 报错，多半是 BIOS 没开虚拟化（Intel VT-x / AMD-V），进 BIOS 打开。

---

## 1. 安装 WSL2 + Ubuntu 22.04

**管理员 PowerShell**：

```powershell
wsl --install -d Ubuntu-22.04
```

这一条会自动启用「虚拟机平台」+「适用于 Linux 的 Windows 子系统」并下载 Ubuntu。**重启电脑**后 Ubuntu 窗口自动弹出，设置用户名和密码（记下这个用户名，迁移后还要用）。

```powershell
# 重启后回到 PowerShell
wsl -l -v              # VERSION 列必须是 2
wsl --update           # 确保 WSL 内核最新
```

---

## 2. 验证 GPU 直通

WSL 内核走 Windows 那套驱动，**WSL 里不装显卡驱动**，只装 CUDA toolkit。

```bash
# 进入 Ubuntu
nvidia-smi
# ✅ 应看到 RTX 3090 + 驱动版本 + CUDA Version（这是驱动支持的上限，不是已装 toolkit）
```

---

## 3. 把 vhdx 迁到 C:\WSL（控制 100GB）

默认 vhdx 装在 `%UserProfile%\AppData\Local\Packages\CanonicalGroupLimited.Ubuntu22.04LTS_*`，不好管理。导出 → 注销 → 导入到 `C:\WSL\Ubuntu2204`：

```powershell
# 📁 建目标目录
mkdir C:\WSL\Ubuntu2204

# 🛑 关闭所有 WSL 实例
wsl --shutdown

# 📦 导出当前系统到 tar（临时放哪个机械盘都行，比如 D 盘）
wsl --export Ubuntu-22.04 D:\ubuntu-backup.tar

# ❌ 注销原来的（会删掉原 vhdx）
wsl --unregister Ubuntu-22.04

# 📦 导入到 C 盘新位置
wsl --import Ubuntu2204 C:\WSL\Ubuntu2204 D:\ubuntu-backup.tar --version 2

# 🧹 清理临时 tar
del D:\ubuntu-backup.tar
```

---

## 4. 恢复默认登录用户

`--import` 后默认用 root 登录，改回第一步设的普通用户。

```powershell
wsl -d Ubuntu2204
```

```bash
# 编辑 /etc/wsl.conf（把 你的用户名 换成实际用户名）
sudo nano /etc/wsl.conf
```

写入：

```ini
[user]
default=你的用户名

[automount]
enabled=true
options=metadata,umask=22
```

保存（`Ctrl+O` 回车，`Ctrl+X` 退出），回 PowerShell：

```powershell
wsl --shutdown
wsl -d Ubuntu2204        # 这次直接用普通用户登录
whoami                   # ✅ 不是 root
```

---

## 5. 限制 vhdx 上限 100GB + 资源配额

编辑（或新建）`C:\Users\<你>\.wslconfig`：

```powershell
notepad $env:USERPROFILE\.wslconfig
```

写入（按你的机器调，内存/核数给足点跑得快）：

```ini
[wsl2]
memory=24GB            # 给总内存的 1/2 ~ 2/3
processors=8
swap=16GB
vhdxSize=100GB         # 把 vhdx 上限钉死在 100GB
```

```powershell
wsl --shutdown
wsl -d Ubuntu2204
```

---

## 6. 装 CUDA Toolkit（WSL 版，不含驱动）

按要跑的框架版本选。PyTorch 官方目前主推 CUDA 11.8 / 12.1 / 12.4。下面以 12.1 为例：

```bash
# 📦 加 NVIDIA 仓库 + 装 cuda-toolkit（注意是 wsl-ubuntu 仓库，不装驱动）
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-1

# 🔧 写入环境变量
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# ✅ 验证
nvcc -V                 # CUDA compiler 版本
```

---

## 7. 装 Miniconda + PyTorch

```bash
# 📦 Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh        # 按提示走，最后让它写入 .bashrc
source ~/.bashrc
```

之后复现模型的标准流程：

```bash
# 📦 建独立 env
conda create -n mymodel python=3.10 -y
conda activate mymodel

# 📦 装 PyTorch（CUDA 12.1 wheel）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# ✅ 链路自检
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 期望输出: True NVIDIA GeForce RTX 3090
```

---

## 8. vhdx 维护（定期压缩）

vhdx 是**只增不减**的动态盘，删了数据空间也不会自动回收。过一两个月压一次：

```powershell
wsl --shutdown
# 用 diskpart 压缩
diskpart
```

在 diskpart 里：

```
select vdisk file="C:\WSL\Ubuntu2204\ext4.vhdx"
attach vdisk readonly
compact vdisk
detach vdisk
exit
```

---

## ⚠️ 踩坑提醒

1. **数据集放 WSL 内部**（`~/data/`），**不要**放 `/mnt/d/`（机械盘挂载点），否则 DataLoader 慢到怀疑人生。机械盘只当仓库，需要时 `cp` 进 WSL。
2. **原始大权重**归档到机械盘，复现前再拷到 `~/models/`。
3. **`.wslconfig` 的 `vhdxSize` 是上限不是预分配**，实际占用按需增长，但不会超过 100GB。
4. **新版本 PyTorch 的 bf16 / tf32 精度默认行为**注意对齐论文设置。
5. **70B+ 大模型** 24GB 显存吃紧，可能要上量化（GPTQ/AWQ）或 CPU offload。
6. **WSL2 网络**默认 NAT，要走 Windows 代理时可能要额外配端口转发，或用 `mirrored` 模式（`.wslconfig` 里 `networkingMode=mirrored`，Win11 22H2+ 支持）。

---

## 📁 目录布局

```
C:\WSL\Ubuntu2204\
└── ext4.vhdx          # 100GB 上限的虚拟磁盘

\\wsl$\Ubuntu2204\home\<你>\    # WSL 内部文件系统（高速）
├── data\              # 数据集（训练热数据放这里）
├── models\            # 模型权重
└── code\              # 算法代码

D:\ 或 E:\（机械盘）           # 仓库
├── datasets_archive\  # 原始数据集归档
└── models_archive\    # 大权重归档
```
