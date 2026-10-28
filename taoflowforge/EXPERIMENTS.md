# EXPERIMENTS.md — TaoFlowForge 复现实验记录

> 按日期倒序追加。结论先行，数字必须能回原始日志核对（日志在 WSL `~/output/taoflowforge_*.log`）。

## 待回填（下载完成后逐项补）

### E-01 首跑基准：剃须刀单图（3090）
| 指标 | 值 |
|---|---|
| 输入 | `/mnt/d/dataset/物体三图数据/剃须刀/01000000.jpg` |
| 总耗时 | 【待填】s |
| 峰值显存 | 【待填】MiB |
| 输出顶点/面数 | 【待填】 |
| mesh.glb 体积 | 【待填】 |
| 视觉质量备注 | 【待填：底部纸张杂物是否被建进网格】 |

### E-02 批量 8 物体（VIEW_SEL=first）
| 物体 | 耗时 | 顶点 | 质量备注 |
|---|---|---|---|
| 剃须刀 | | | |
| 手表 | | | |
| 榨汁机 | | | |
| 电动牙刷 | | | |
| 电饭煲 | | | |
| 耳机 | | | |
| 路由器 | | | |
| 鼠标 | | | |

---

## 2026-10-28 环境搭建阶段（先于推理）

### 已核实的环境事实
- 上游 requirements.txt 有 bug：`Pillow==12.0.0` ↔ gradio 5.50.0（要求 pillow<12.0）
  冲突，pip 新解析器直接 ResolutionImpossible。00a 运行时 sed 降级 Pillow==11.3.0。
- open3d 0.19（pip wheel）依赖系统 libgomp.so.1；libgcc-ng 15.x 不再捆绑，
  `conda install -c conda-forge libgomp` 解决（WSL 无 sudo，apt 不可用）。
- `--offload` 上游 choices 为 auto/on/off（默认 auto），README 早期写的 "stage" 有误，
  02 脚本已修正——**以 cli.py 为准**。
- ModelScope SDK 下载有总时限：26GB 权重 1 小时报 `[E1001] 下载总时限已耗尽`，
  续传时限不放大 → 大权重走 Windows 侧迅雷/curl 直下 resolve 直链更稳。
- stage0.pt 若不内嵌 dino_encoder.* 权重 → 推理必须 `--dino-checkpoint`
  （备用镜像：ModelScope `facebook/dinov3-vitl16-pretrain-lvd1689m`，非 gated）。
  下载完成后用 `04_check_weights.sh` 一键探测。
