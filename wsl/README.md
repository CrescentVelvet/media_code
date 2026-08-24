# WSL2 + Ubuntu 环境搭建（Win11 + RTX 3090）

在 Windows 11 上用 WSL2 跑 Ubuntu 24.04，GPU 直通跑算法复现。C 盘固态 800GB 空闲，分 100GB 给 WSL vhdx；机械盘当仓库归档原始数据集与权重。

> 🎮 RTX 3090（24GB / Ampere / 算力 8.6），主流框架全支持。

## 0. 前置检查

```powershell
# Windows 端（管理员 PowerShell）
wsl --status
wsl -l -v
nvidia-smi            # 确认 3090 + 驱动正常
```

> ⚠️ 如果 `wsl --install` 报错 `0x80370102`，多半是 BIOS 没开虚拟化（Intel VT-x / AMD-V），进 BIOS 打开。
> ⚠️ 如果 `wsl --install -d <distro>` 报 `WININET_E_TIMEOUT` 拉 `raw.githubusercontent.com` 失败，跳到第 1 步用 Microsoft Store 装（国内/公司代理墙）。

---

## 1. 安装 WSL2 + Ubuntu 24.04

分两步：先启用 WSL2 本体（不拉发行版元数据，绕过 `raw.githubusercontent.com`），再用 Microsoft Store 装 Ubuntu。

### 1a. 启用 WSL2 功能（管理员 PowerShell）

```powershell
wsl --install --no-distribution
```

这一条会启用「虚拟机平台」+「适用于 Linux 的 Windows 子系统」并下载 WSL2 内核，但**不下载任何发行版**（避免拉 GitHub 元数据超时）。**重启电脑**。

```powershell
# 重启后回到 PowerShell 验证
wsl --status           # 默认版本: 2
wsl --update           # 确保 WSL 内核最新
wsl --version          # 看到 WSL 版本 + 内核版本即可
```

### 1b. 从 Microsoft Store 装 Ubuntu 24.04 LTS

1. **开始菜单 → Microsoft Store**
2. 搜索 `Ubuntu 24.04 LTS` → 点 **Get / Install**（约 500MB，走商店 CDN，国内能下）
3. 装完后开始菜单出现 **Ubuntu 24.04 LTS**，点击打开
4. 弹出窗口设 UNIX 用户名 + 密码（**记下这个用户名，迁移后还要用**）

> ⚠️ Microsoft Store 装的发行版**名字是 `Ubuntu`**（不是 `Ubuntu-24.04`），用 `wsl -l -v` 确认。

```powershell
$env:WSL_UTF8=1; wsl -l -v
# 期望: Ubuntu  Running  2
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

Microsoft Store 装的默认 vhdx 在 `%UserProfile%\AppData\Local\Packages\CanonicalGroupLimited.Ubuntu24.04LTS_*`，不好管理。导出 → 注销 → 导入到 `C:\WSL\Ubuntu2404`：

```powershell
# 📁 建目标目录
mkdir C:\WSL\Ubuntu2404

# 🛑 关闭所有 WSL 实例
wsl --shutdown

# 📦 导出当前系统到 tar（临时放哪个机械盘都行，比如 D 盘）
wsl --export Ubuntu D:\ubuntu-backup.tar

# ❌ 注销原来的（会删掉原 vhdx）
wsl --unregister Ubuntu

# 📦 导入到 C 盘新位置（新发行版名定为 Ubuntu2404，避免和商店版重名）
wsl --import Ubuntu2404 C:\WSL\Ubuntu2404 D:\ubuntu-backup.tar --version 2

# 🧹 清理临时 tar
del D:\ubuntu-backup.tar
```

> ⚠️ 迁移后用 `wsl -d Ubuntu2404` 进入；商店版原 `Ubuntu` 名已注销，开始菜单的图标会失效，可右键取消固定。

---

## 4. 恢复默认登录用户

`--import` 后默认用 root 登录，改回第 1b 步设的普通用户。

```powershell
wsl -d Ubuntu2404
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
wsl -d Ubuntu2404        # 这次直接用普通用户登录
whoami                   # ✅ 不是 root
```

---

## 5. 限制 vhdx 上限 100GB + 资源配额

`.wslconfig` 必须放在 Windows 用户目录（`C:\Users\<你>\.wslconfig`），不在 WSL 内。两种编辑方式任选：

**Windows PowerShell**：

```powershell
notepad $env:USERPROFILE\.wslconfig
```

**WSL bash**（用 nano，`<Windows用户名>` 换成实际，如 `wangyufeng`）：

```bash
nano /mnt/c/Users/<Windows用户名>/.wslconfig
```

> nano 操作：`Ctrl+O` 回车保存，`Ctrl+X` 退出。文件不存在会自动新建。

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
wsl -d Ubuntu2404
```

---

## 6. 装 CUDA Toolkit（WSL 版，不含驱动）

按要跑的框架版本选。PyTorch 官方目前主推 CUDA 11.8 / 12.1 / 12.4。下面以 12.1 为例：

```bash
# 📦 加 NVIDIA 仓库 + 装 cuda-toolkit（注意是 wsl-ubuntu 仓库，不装驱动）
cd ~                    # 避开从 /mnt/c/Windows/system32 启动导致的 Permission denied
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

> ⚠️ Ubuntu 24.04 自带 Python 3.12；如要跑老代码用 conda 装 3.10/3.11 env 即可，不要动系统 Python。

---

## 7. 装 Miniconda + PyTorch

```bash
# 📦 Miniconda
cd ~                    # 避开从 /mnt/c/Windows/system32 启动导致的 Permission denied
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
select vdisk file="C:\WSL\Ubuntu2404\ext4.vhdx"
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
7. **`wsl --install -d <distro>` 在国内/公司代理下会超时**：它走 WinINet 拉 `raw.githubusercontent.com/microsoft/WSL/.../DistributionInfo.json`，PowerShell 的 `$env:HTTPS_PROXY` 对 `wsl.exe` 不生效。绕过：`wsl --install --no-distribution` 启用本体 + Microsoft Store 装发行版。
8. **Microsoft Store 装的发行版名是 `Ubuntu`**（不是 `Ubuntu-24.04`），迁移 `--import` 时另起新名（如 `Ubuntu2404`）避免和商店版重名。
9. **从 Windows PowerShell 启动 wsl 会继承当前目录**：在 `C:\Windows\system32>` 跑 `wsl -d Ubuntu2404` 会进入 `/mnt/c/Windows/system32/`，普通用户没写权限，`wget`/`pip install` 报 `Permission denied`。先进 `cd ~` 回 home 再操作，或启动时用 `wsl -d Ubuntu2404 --cd ~`。

---

## 📁 目录布局

```
C:\WSL\Ubuntu2404\
└── ext4.vhdx          # 100GB 上限的虚拟磁盘

\\wsl$\Ubuntu2404\home\<你>\    # WSL 内部文件系统（高速）
├── data\              # 数据集（训练热数据放这里）
├── models\            # 模型权重
└── code\              # 算法代码

D:\ 或 E:\（机械盘）           # 仓库
├── datasets_archive\  # 原始数据集归档
└── models_archive\    # 大权重归档
```
