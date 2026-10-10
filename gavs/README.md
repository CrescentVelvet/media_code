# GaVS — 3D Grounded 视频稳像（SIGGRAPH 2025）

复现 [GaVS: 3D-Grounded Video Stabilization via Temporally-Consistent Local Reconstruction and Rendering](https://arxiv.org/abs/2506.23957)（[官方仓](https://github.com/sinoyou/GaVS)）：GLOMAP 位姿 + 逐帧前馈局部 3DGS 重建（Flash3D/UniDepth 骨干）+ 平滑位姿渲染，测试时 3 epoch 微调（多视角动态补偿光度监督 + 空洞跨帧正则），全画幅免裁剪输出。论文调研笔记见 media_paper 仓 `3D-20250630-GaVS.html`。

**本机复现走 [`README_wsl.md`](README_wsl.md)**（WSL Ubuntu 24.04 + RTX 3090，含迅雷下载链接）。

## 常用命令（WSL）

```bash
# 首次：建环境（conda env gavs + torch 2.2.2 cu121 + 编译光栅化扩展）
bash gavs/00a_setup_env.sh

# 权重/数据就位后（迅雷下载到 D:\wheel\gavs_ms\，链接见 README_wsl.md）
bash gavs/01_download_models.sh

# 微调 + 评估（3 epoch 微调后自动评估，σ_s=4）
SCENE=dynamic_dance bash gavs/02_run_finetune.sh

# 仅评估：换稳定强度重渲染（不用重新微调）
SCENE=dynamic_dance STABILITY=8 bash gavs/03_run_eval.sh
```

## Config (env vars)

| var | default | note |
|---|---|---|
| `CONDA_ENV` | `gavs` | conda 环境名 |
| `GAVS_DIR` | `~/repos/GaVS` | 官方仓（Linux fs，要编译扩展） |
| `GAVS_DATA_ROOT` | `~/model/gavs/gavs-data` | 数据 + 预训练权重根 |
| `GAVS_RESULTS_DIR` | `~/output/gavs_results` | 微调/评估产物 |
| `GAVS_STAGING_DIR` | `/mnt/d/wheel/gavs_ms` | 迅雷下载暂存区 |
| `SCENE` | `dynamic_dance` | 场景名（15 选 1） |
| `STABILITY` | `4` | 03 的稳定强度 σ_s（4/8） |
| `GPU` | （不设置） | 选卡，如 `GPU=0` |

## 目录布局

```
~/repos/GaVS/                    # 官方仓（Linux fs）
~/model/gavs/gavs-data/
├── re10k_v2_checkpoints/        # Flash3D 预训练权重（迅雷）
└── dataset/<scene>/             # 15 场景预处理数据（迅雷，按需）
~/output/gavs_results/<scene>/   # 微调 checkpoint + 渲染视频 + 指标
/mnt/d/wheel/gavs_ms/            # 迅雷下载暂存区（Windows D 盘）
```
