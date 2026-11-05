# da3_human — 实验记录

> 按日期倒序追加。实验计划/决策依据见 `EXPERIMENT_PLAN.md`，机制解释见 `NOTES.md`（阶段E补）。

---

## 2026-11-05 阶段 B-03/04 基线训练

### 阶段 B-03：原版 3DGS 基线（COLMAP init）✅

| 项 | 结果 |
|---|---|
| 初始化 | COLMAP sparse **7,163 点**（vggt_human source，即现有 scene_gs 路径） |
| 训练 | 30,000 iters，全分辨率 1440×1920，**45m51s**（RTX 3090） |
| 训练内评估 | [ITER 30000] PSNR **27.10**（官方 train.py 日志口径） |
| 渲染后评估 | train 视角 125/125：PSNR **27.573** / SSIM **0.8854** / LPIPS **0.1642** |
| 产物 | `~/output/da3_human_results/baseline_3dgs_colmap/`（point_cloud/iteration_30000、train/ours_30000/{renders,gt}） |

定性观察（floater/重影对照现象）待截图补入——需与 04（DA3 init）同视角对比后统一记录。

**metrics.py 全 nan 根因（"Metric evaluation progress: 0it"）**：
1. 本仓 `metrics.py`（vggt_human 定制版）只扫 `test/` 分割，而 render.py 渲染的是 train 视角
   → 建软链 `test/ours_<iter> -> ../train/ours_<iter>` 解决；
2. ⚠️ 坑中坑：**render.py 会预先给 test split 建空的 `renders/`、`gt/` 目录**，直接
   `ln -sfn` 会把链接塞进该目录内部（`test/ours_30000/ours_30000 -> ...`），metrics 仍读空目录。
   必须先 `rmdir` 空目录再软链。`_train_3dgs.sh` 已内置该逻辑（含 rmdir），04 起自动生效。

### 阶段 B-04：DA3-GS 基线（DA3 点云初始化）🔄

场景已就绪（`da3_colmap_scene/`：images 软链 + sparse/0，**459,433 点**，由 DA3 深度反投影
14,288,402 点体素降采样；原版 3DGS `read_model` 验证通过，125 cams/imgs PINHOLE）。
训练 2026-11-05 启动，结果稍后补录。

---

## 2026-11-05 阶段 A/B 启动

### 执行前调研发现（修正计划前提）

1. **⚠️ DA3 权重许可矩阵（README Model Zoo 实测）——修正计划 §6 前提**：

   | 能力组合 | 可选模型 | 权重许可 |
   |---|---|---|
   | 相对深度 + 位姿 | DA3-BASE(0.12B) / DA3-SMALL(0.08B) | Apache 2.0 |
   | 度量深度（单目，无位姿） | DA3METRIC-LARGE(0.35B) | Apache 2.0 |
   | **度量深度 + 位姿一体** | **DA3NESTED-GIANT-LARGE-1.1(1.40B) / DA3-GIANT-1.1(1.15B)** | **CC BY-NC 4.0** |

   - 计划 §7 说「depth-anything/Depth-Anything-3 (Apache 2.0)」——仅**代码**许可成立；
     计划 §6「商用 pose-free 管线（DA3 Apache 2.0 解许可）」在**权重层面不成立**：
     度量+位姿组合无 Apache 选项。商用化路径只剩：BASE/SMALL（相对深度+位姿，尺度靠 COLMAP/BA）或等官方出 Apache 度量模型。
   - **本期选择**：内部验证用 `DA3NESTED-GIANT-LARGE-1.1`（质量最高：README 称 DA3-LARGE 已可比 VGGT，NESTED 为 any-view giant + metric 嵌套），CC BY-NC 仅内部研究可接受。

2. **数据确认**（计划 §2 标注「待确认」已落实）：
   - vggt_human 棚拍 = **单场景，125 视角，1440×1920 JPG**，COLMAP sparse/0 在位（`~/output/vggt_human_results/source/`）；`/mnt/d/output/vggt_human_results` 为同场景 08 搬运产物。
   - 满足计划数据要求（≥4 视角、有内参、静态）；主体人物基本静态（vggt_human/04 注释），动态重影风险低。
   - 局限：只有 1 个场景，D 阶段敏感性实验无跨场景重复性检验（记为已知限制）。

3. **环境/网络实测（2026-11-05）**：
   - `huggingface.co` 经 v2rayN(10808) API 200 但 7s；`hf-mirror.com` 复活（2.4s）→ **下载走镜像**（10-28 记录镜像已挂，网络确会漂移，每次复测）。
   - `download.pytorch.org` 目录 403 属正常；torch 走 tuna（torch 2.14.1 拉 cu13 nvidia 依赖 ~2.5GB，~2.3MB/s）。
   - DA3 官方仓已在 `~/repos/Depth-Anything-3`（HEAD 4173623），跳过 clone。
   - 模型导出格式支持 `-` 组合（如 `mini_npz-colmap-depth_vis`）；`colmap` 导出把 cameras/images/points3D.bin 写在 export_dir 根（需整理成 `sparse/0` 给原版 3DGS）。

### 阶段 A：环境搭建 ✅

| 项 | 结果 |
|---|---|
| conda env | `da3_human`，python 3.10.22（`00a_setup_env.sh` 一键建） |
| torch | 2.14.1+cu130（tuna；拉 cu13 nvidia 依赖 ~2.5GB） |
| xformers | 0.0.35 ⚠️ 二进制是 against torch 2.10+cu128 编的，C++/CUDA 扩展加载失败 |
| DA3 本体 | editable 0.0.0（`~/repos/Depth-Anything-3` HEAD 4173623） |
| 其他关键 | numpy 1.26.4（requirements 钉 numpy<2）、pycolmap 4.2.1、open3d 0.20.0 |
| 耗时 | 装包 ~44 min（tuna ~1.5MB/s） |

**xformers 不配对的影响评估**：DA3 attention 走 `F.scaled_dot_product_attention`
（PyTorch 原生，model/utils/attention.py:57），xformers 只用于 SwiGLU 融合 MLP
（Triton 算子，运行时编译，不依赖预编译 C++ 扩展）。**预计不影响推理**，由 02 实测最终验证；
若 SwiGLU 报错，回退方案 = torch 降到 2.10.0 对齐 xformers。

装包坑：pip 对 opencv-python 回退解析多轮（moviepy 依赖冲突），最终 opencv 4.11.0.86 + moviepy 1.0.3；
numpy 被 requirements 的 `numpy<2` 从 2.2.6 降级到 1.26.4（torchvision 装的是 2.x，无冲突）。

### 阶段 B-01：权重下载 ✅

| 项 | 结果 |
|---|---|
| 模型 | `depth-anything/DA3NESTED-GIANT-LARGE-1.1` → `/mnt/d/wheel/da3_human_ms` |
| 产物 | model.safetensors **6.3GB**（fp32 1.4B）+ config.json |
| 通道 | hf-mirror.com（直连 7s vs 镜像 2.4s；镜像吞吐 ~1MB/s，全程 63 min） |
| 复跑 | `bash da3_human/01_download_models.sh`（幂等，本地已存在则秒过） |

### 阶段 B-02：DA3 推理（125 视角）✅

| 验收项 | 结果 | 判定 |
|---|---|---|
| 深度可视化合理 | median **1.797m**，范围 0.48~6.10m，NaN=0（度量输出，棚拍尺度合理） | ✅ |
| 位姿无漂移 | vs COLMAP Sim(3) 对齐后 mean **4.9mm** / p95 8.2mm / max 10.1mm，125/125 视角匹配，轨迹逐段重合（viz/trajectory_3d.png） | ✅ |
| 耗时 | 模型加载 18s + 前向 34~54s + 导出 92s ≈ **2.5 min**（504p，RTX 3090） | — |

产物（`~/output/da3_human_results/da3_inference/studio_human/`）：
`exports/{mini_npz,npz}/results.npz`（深度/置信度/extrinsics/intrinsics）、
colmap 导出（cameras/images/points3D/frames/rigs .bin，**14,288,402 点**、conf≥P40）、
`depth_vis/` 逐视角深度可视化、`da3_summary.json`、`viz/` 轨迹对比图。

**关键数字解读**：
- Sim(3) scale=0.5077：DA3 度量重建比 COLMAP 坐标系大约 1.97×（COLMAP 尺度本就任意，无问题）。
- 首末相机距 1.42m ≠ 漂移：轨迹是带起始小回环的开弧（viz 图证实），非闭环采集。

**踩坑记录**：
1. DA3 `requirements.txt` 漏声明 `addict`（da3.py:19）→ 00 脚本已补 `pip install addict`。
2. COLMAP `images.txt` 解析：points2D 空行（无 2D 点的注册图像）不能过滤，否则 header 配对
   错位约一半匹配不上（首跑 63/125 → 修后 125/125）。
3. xformers 0.0.35/torch 2.14.1 不配对：推理正常（attention 走 F.sdpa；日志出现
   "using MLP layer as FFN" 说明 SwiGLU 融合层自动回退 MLP），**无实际影响**，不回退 torch。
