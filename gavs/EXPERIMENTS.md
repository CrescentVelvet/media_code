# EXPERIMENTS.md — GaVS 复现实验记录

> 按日期倒序追加。跑通流程见 README_wsl.md；排坑细节见 .workbuddy/memory/2026-10-11.md。

## 2026-10-12 dynamic_dance 首次复现（WSL2 / RTX 3090 24GB）

### 结论

**跑通。** 微调 3 epochs + 自动评估全链路无人工干预完成，产物 7 个 mp4 + 指标 json。
训练 45 分钟，显存 23.4 GB 贴顶（24 GB 卡可跑，余量 ~1.2 GB）。

### 实测指标（官方评估管线输出，σ_s=4 默认）

| 指标 | 实测值 | 说明 |
|---|---|---|
| Distortion value | 0.9828 | 越高越好，上界 1 |
| Stability Score (avg) | 0.9367 | 越高越好 |
| — trans | 0.8834 | 平移分量 |
| — rot | 0.9901 | 旋转分量 |
| crop_ratio / min | 1.0 / 0.9747 | 评估裁剪比 |

官方参考值：论文只报跨场景聚合（media_paper 笔记记录的结论是 Stability 与 GC-S/GC-D
上最优或并列最优）；逐场景参考在 HF 数据仓 `result_and_comparison/`（未下载）。
本组数值是否对齐官方【待核实】——若要核对，下 `result_and_comparison/dynamic_dance*`
里的官方数值文件对比。

### 运行参数

- 数据：`~/model/gavs/gavs-data/dataset/dynamic_dance`（455 帧，含动态人体）
- 预训练：`re10k_v2_checkpoints/model_re10k_v2.pth`（Flash3D，仅微调 GS 解码器）
- dataloader：batch=5；epoch 间采样 stride 1/3/5/7 → 151+50+30 批，共 231 steps
- 耗时：epoch0 30min（含 CUDA 扩展首次 JIT/缓存预热），epoch1 9.3min，epoch2 5.6min
- 评估渲染：455 帧 ~3.8 min（2 it/s），ffmpeg 合成 mp4

### 产物位置

- WSL：`~/output/gavs_results/dynamic_dance/`（checkpoints/model_0000231.pth、visualization/、depth_scale.json、train/evaluate.log）
- Windows（视频副本）：`D:\output\gavs_results\dynamic_dance\*.mp4`

| mp4 | 内容 |
|---|---|
| stable_color_gauss.mp4 | **主结果**：稳像后彩色（3DGS 渲染） |
| stable_alpha_gauss / stable_depth_gauss.mp4 | 稳像后 alpha / 深度 |
| unstable_color.mp4 | 稳像前彩色（原始抖动轨迹，对比用） |
| unstable_color_gauss / unstable_depth_gauss / unstable_unidepth.mp4 | 稳像前 渲染/深度/UniDepth 中间量 |

### 本次踩坑（全部已修复并回写脚本，详见 memory 2026-10-11）

1. torch 2.2.2 pybind11 cast.h vs nvcc 12.1 → patch（PR #4893 写法）
2. requirements 带飞 numpy 2.x → 钉回 1.26.4
3. 官方 requirements.txt 漏 rich → 补装
4. HF 离线缓存缺 refs/main → 01 自动补建
5. **torchvision 联网下 ResNet50 挂死训练初始化**（download.pytorch.org 通而不稳、urllib 无超时）
   → 峰哥迅雷手动下载，01 种入 `~/.cache/torch/hub/checkpoints/`（注意 hub/ 一层）
6. hydra.run.dir 使产物落 exp/<scene>/ 而非仓根 → 02 搬运路径修正

### 稳定强度消融（σ_s=4 vs σ_s=8，同一 checkpoint 仅重渲染，03 脚本验证通过）

| σ_s | Stability ↑ | Distortion ↑ | crop_ratio_min ↑ |
|---|---|---|---|
| 4（默认） | 0.9367 | 0.9828 | 0.9747 |
| 8 | **0.9401** | 0.9693 | 0.9578 |

趋势符合论文核心 claim「平滑与重建解耦 trade-off」：σ_s 加大 → 更稳、畸变/裁剪代价上升。
03 eval-only 链路（不重新训练，换 σ_s 重渲染）验证通过，单次 ~7 分钟。

### 后续

- [ ] 其余 14 场景批量（下载清单见 README_wsl.md，总计 ~17.5 GB）
- [ ] 与官方 result_and_comparison 逐场景数值核对【待核实】
