# som_human (WSL Ubuntu 24.04) — 本地复现指南

本目录训练侧**直接复用 deformable_human env 与官方仓**（方案 §9.2），因此 WSL
实操与 `deformable_human/README_wsl.md` 高度同源，这里只记差异与坑。

## 与 deformable_human 的关系
| 项 | 复用 | 新增 |
|---|---|---|
| conda env | ✅ `deformable_human`（torch 2.5.1+cu121 + fork 光栅化器） | 无（隔离需要时 `CONDA_ENV=som_human` 覆盖再 clone） |
| 官方仓 | ✅ `~/repos/Deformable-3D-Gaussians`（同一 DG_DIR） | 无 |
| 数据准备 | ✅ `deformable_human/01_prepare_data.sh` 的 colmap_scene | 无 |
| 运动模型 | ❌ 替换 | `som_motion.py`（SoM 式 SE(3) 基凸组合） |
| 训练/渲染 | ❌ fork | `train_som.py` / `render_som.py` |
| 输出目录 | — | `~/output/som_human_results/<scene>/`（与 deformable_human_results 分开，便于同场景对比） |

## 全流程命令
见 [README.md](README.md)「常用命令」。WSL 下唯一注意：`$HOME/output/...` 在
Linux fs，训练完跑 `08_move_output.sh` 搬到 `/mnt/d/output/som_human_results/`。

## 可能遇到的问题（WSL 专属）
- **稠密化爆炸**：与 deformable_human 完全同因（渲染/稠密化基建没变）。
  hand_motion 类复杂背景**必须**带：
  `EXTRA_TRAIN_ARGS='-r 2 --densify_grad_threshold 0.0004 --densify_until_iter 2000'`
  （deformable_human 实测爆炸点 ~2600 步，2000 冻结后 17GB / 3.1 it/s）。
- **num_frames 断言失败**（`som_motion.load_weights`）：训练后场景相机数变了
  （重新跑了 01 / 换了 EVAL 划分）。渲染必须与训练同一份 colmap_scene。
- **num_bases 断言失败**：渲染时 `NUM_BASES` 与训练不一致。03_render.sh 默认
  NUM_BASES=10，改了训练端就同步改渲染端。
- **cfg_args 路径过期**（迁移目录后 "Could not recognize scene type!"）：
  `sed -i 's|旧路径|新路径|' model/cfg_args`（同 deformable_human）。
- **Python 版本红线**：目标 env 3.10.x，本仓脚本提交前过 `99h_check_py_syntax.py`
  口径的人工检查（vggt_human 教训；本目录代码已按 3.10 语法写）。
- **基不收敛/全程接近 identity**：先查 `SMOOTH_WEIGHT` 是否过大、
  `BASIS_LR` 是否被手抖改小；再怀疑「无 Procrustes 初始化」这一已知差异
  （README.md「与 SoM 原版的差异」表第 2 行——SoM 消融里跳初始化 EPE +35%，
  那是 Phase 1 不达标时的首个补丁方向）。
