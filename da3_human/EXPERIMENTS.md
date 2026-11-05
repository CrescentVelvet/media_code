# da3_human — 实验记录

> 按日期倒序追加。实验计划/决策依据见 `EXPERIMENT_PLAN.md`，机制解释见 `NOTES.md`（阶段E补）。

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

### 阶段 B-01：权重下载

| 项 | 结果 |
|---|---|
| 模型 | `depth-anything/DA3NESTED-GIANT-LARGE-1.1` → `/mnt/d/wheel/da3_human_ms` |
| 状态 | 待补（经 hf-mirror 下载中） |
