# archive/ — 暂存的自研增强脚本

这些脚本是本项目早期为"论文级人物重建"从零手搓的增强模块。**当前 gaussian-splatting
fork（官方原版）上它们与训练语义脱节，会把 3DGS 训练跑坏**（详见下方每条 bug），
因此从主链路（step 02→03→04→05 纯官方重建）中摘出归档，**不参与任何默认脚本调用**。

> 设计意图：先跑通干净、可复现的官方基线（04/07 现在恒走 `train.py`），
> 之后**逐个**把这些模块以"正确接入官方 `Scene`/`Camera`/`render` 框架"的方式加回，
> 每个加回后单独跑一遍 04 对照，确认不与官方语义冲突再继续下一个。

## 归档清单

| 文件 | 原职责 | 已知问题 / 为什么归档 |
|---|---|---|
| `train_pose.py` | 自定义训练 wrapper（姿态变换+精炼+动态+深度法线全耦合） | 绕开官方 `Scene`，手写 COLMAP 加载 + 训练循环；缺 `oneupSHdegree`、exposure/sparse_adam 语义；`render(...,render_depth=True)` 参数不存在被静默 except；训练 30000 步后 opacity 爆到 12580、高斯不生长（仅 1.8万 而非 36万）、渲染全黑。**主 bug 源头** |
| `pose_refine.py` | 训练中可学位姿（四元数+平移） | import 依赖 pose_adjuster；未接入官方 camera/optimizer 语义 |
| `dynamic_mask.py` | 动态掩码（SAM2+GroundingDINO） | 需外部大模型权重；基于错误 API 写的，官方 fork 上不对接 |
| `dynamic_filter.py` | 动态点云过滤 | 依赖 pose_adjuster、dynamic_mask 产物 |
| `noise_negating.py` | DINOv2+MLP 动态感知 / 噪声抑制 | 需 DINOv2；NN loss 接入点与官方光栅化不匹配 |
| `depth_normal_cons.py` | 深度-法线一致性约束 | 期望 `render_pkg["render_depth"]/["render_normal"]`，官方 render 只有 `"depth"` 且无法线输出 → 该约束在官方 fork 上永远静默不生效 |

> `pose_adjuster.py` **不在归档内**：它被另一个算法目录 `pdfgs_human/02b_pose_adjust.sh`
> 跨目录 import（`sys.path` 指向 `../vggt_human` 后 `from pose_adjuster import`），用于
> PDF-GS 训练前的一次性位姿变换，是共享依赖，故保留在 `vggt_human/` 根目录供 pdfgs_human 使用。
> vggt_human 主链路（02→05 官方重建）不调用它。

## 主链路现状（未归档、参与默认流程）

- `02_run_inference.py`(run_batch) → `03_npz_to_colmap.py` → **04/07 官方 train.py**
  → `05 render_novel.py + denoise_images.py` → `06 face_enhance.py`（可选）
- `render_novel.py` 中对 pose_adjuster 的条件 transform 已移除（官方 train 不再产生
  `pose_adjuster.json`，novel 视图直接在源场景坐标系插入）。

## 加回步骤（每个增强的通用流程）

1. 把目标模块从 `archive/` git mv 回 `vggt_human/`（或先在 archive 内改好）。
2. 以**官方 `Scene`/`loadCameras`/`render` 签名为准**重写其接入点（对照
   `~/repos/gaussian-splatting/train.py` 的语义：oneupSHdegree、exposure、sparse_adam、`render_pkg["depth"]`）。
3. 只在 `04_train_3dgs.sh` 里**加回该增强对应的开关分支**，其余增强仍关闭。
4. 跑一遍完整 04，用官方同源对照：若高斯能正常生长、loss 收敛、渲染非黑，则通过。

> 注意：archive 内个别模块 import `pose_adjuster`（如 pose_refine.py 顶部
> `from pose_adjuster import quat_to_rotmat`）。`pose_adjuster` 现在留在主目录（vggt_human/），
> 所以把这些模块从 archive 移回主目录后其 import 直接可用，无需额外处理。
