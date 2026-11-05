# querysplat/README_wsl.md — QuerySplat 前馈 3DGS 重建（WSL 复现）

从多视角图像（免位姿，pose-free）前馈重建 3D Gaussian 场景。QuerySplat 的核心是
**几何/外观双查询解耦**：可学习几何查询 slot 在冻结 VGM（VGGT-Ω）先验下决定"放哪"，
独立外观查询决定"长什么样"，DL3DV 上比最强 baseline +2.30dB。

- 论文：<https://arxiv.org/abs/2608.01186>
- 官方仓：<https://github.com/inspatio/QuerySplat>（仅推理代码，Apache 2.0；
  vendored VGGT-Ω/DINOv3 是 FAIR 非商用研究许可，商用需替换 backbone）
- 权重：<https://huggingface.co/inspatio/querysplat>（公开）+
  **<https://huggingface.co/facebook/VGGT-Omega>（⚠️ gated，需申请授权）**

## 常用命令

```bash
cd /mnt/c/code/media_code

# 0) 建 env（clone gen3r 的 torch 2.8.0+cu128 + 编译 gsplat/fused-ssim，约 10-20 min）
bash querysplat/00a_setup_env.sh

# 1) 下权重（QuerySplat 公开 + VGGT-Ω gated；后者需先在 HF 点 Request access 并 hf login）
bash querysplat/01_download_models.sh

# 2) Remy 采集包 → 前馈重建 → ply
RAW_DIR="/mnt/d/dataset/手部微动样例/01e954de1b8a4cde9315c976f4dcb964-佳佳抱枕" \
  N_VIEWS=32 bash querysplat/02_run_inference.sh

# 可选：开 TTO（冻结查询、只优化特征内存，20 步提质，慢）
RAW_DIR=... TTO=1 bash querysplat/02_run_inference.sh
```

输出：`~/output/querysplat_results/<场景名>/`（高斯 PLY、预测相机 JSON、
输入视角渲染、VGM 深度点云 `--save_vggt_depth_pointcloud` 等）。

## 首次准备

- 官方仓已 clone 到 `~/repos/QuerySplat`（00a 缺失时会自动补）。
- 权重约定死在 `~/repos/QuerySplat/checkpoints/`（yaml 内 `vggt_checkpoint` 是相对路径，
  **不要挪到别处**，要挪就得同步改 yaml）。
- VGGT-Ω 权重 gated：去 <https://huggingface.co/facebook/VGGT-Omega> 申请
  （自动审核），然后 `hf login`。vggt_omega 目录下载到 `~/model/VGGT-Omega` 的副本
  也可以软链过来：`ln -s ~/model/VGGT-Omega/vggt_omega_1b_512.pt ~/repos/QuerySplat/checkpoints/`。

## Config (env vars)

| var | default | note |
|---|---|---|
| `CONDA_ENV` | `querysplat` | clone gen3r 而来 |
| `QUERYSPLAT_REPO` | `~/repos/QuerySplat` | 官方仓 |
| `RESULTS_DIR` | `~/output/querysplat_results` | Linux fs，跑完自行搬到 /mnt/d |
| `N_VIEWS` | `32` | 采集包等间隔抽帧数 |
| `OPACITY_THRES` | `0.05` | 保存高斯的不透明度阈值（yaml 默认 0.005） |
| `TTO` | `0` | 1 = 开 test-time optimization |

## 风险与边界

- **官方测试配置 Python 3.12 + torch 2.11.0 + cu128**；本机复用 gen3r 的
  python 3.11 + torch 2.8.0+cu128，import/前向若出现版本不兼容，按官方配置重建 env。
- 官方只开源推理；TTO 侧 loss 权重可在代码核实，训练侧（390K 步）不可复现。
- 推理是 pose-free：输出高斯在 VGM 自建坐标系，跨方法对比前必须做 Sim(3) 对齐
  注册到 GT 坐标系（Remy 包有 GT 位姿与 GT 点云 `pcd.ply`）。
- 输入视角数影响查询预算：`num_input_views: 4` 是训练配置；更多视角直接喂，
  显存按官方 benchmark（H200 12 视角 1.89s）估算，24GB 卡注意别超。
