# da3/README_wsl.md — DA3（metric 深度+位姿）+ 传统 3DGS（WSL 复现）

DA3（Depth-Anything-3，ByteDance-Seed）前馈预测逐视角 metric 深度 + 相机位姿 + 内参，
深度反投影建点云（COLMAP 场景）后接官方 gaussian-splatting 逐场景训练出 ply。
这是"几何基座 + 传统 3DGS"两阶段路线的代表。

- 官方仓：`~/repos/Depth-Anything-3`（AGENTS.md 约定，infinisplat_v2/da3_human 共用）
- 权重：`depth-anything/DA3NESTED-GIANT-LARGE-1.1`（HF；⚠️ CC BY-NC 4.0 非商用；
  HF 缓存里 da3_human 已下载过一份）

## 常用命令

```bash
cd /mnt/c/code/media_code

# 0) 依赖检查（不建重复 env：DA3 推理=da3_human env，3DGS=vggt_human env）
bash da3/00a_setup_env.sh

# 1) 下权重（缓存已有会秒过；下到 ~/model/da3）
bash da3/01_download_models.sh

# 2) Remy 采集包 → DA3 深度+位姿 → COLMAP 场景
RAW_DIR="/mnt/d/dataset/手部微动样例/01e954de1b8a4cde9315c976f4dcb964-佳佳抱枕" \
  N_VIEWS=32 bash da3/02_run_inference.sh

# 3) 传统 3DGS 训练 → ply
RAW_DIR="/mnt/d/dataset/手部微动样例/01e954de1b8a4cde9315c976f4dcb964-佳佳抱枕" \
  ITERATIONS=30000 bash da3/03_train_3dgs.sh
```

输出：`~/output/da3_results/<场景名>/{mini_npz/, colmap/, depth_vis/, 04_model_3dgs/}`

## 复用关系（重要）

本仓是**编排层**，不重写已验证的链路：

| 步骤 | 实际执行 | 环境 |
|---|---|---|
| 采集包转换 | `infinisplat_v2/07_capture_to_npz.py` | infinisplat_v2（HEIC 解码） |
| DA3 推理 | `da3_human/02_run_da3_inference.sh` | da3_human |
| 3DGS 训练 | `vggt_human/04_train_3dgs.sh`（官方 gaussian-splatting） | vggt_human |

da3_human 的实验计划（EXPERIMENT_PLAN.md）里还有 DA3+2DGS 冻结方案（阶段 C），
本仓只取"DA3 + 传统 3DGS"基线路线做四方法横评。

## Config (env vars)

| var | default | note |
|---|---|---|
| `DA3_MODEL_REPO` | `depth-anything/DA3NESTED-GIANT-LARGE-1.1` | CC BY-NC 4.0 |
| `DA3_MODEL_DIR` | `~/model/da3` | 有 model.safetensors 时优先于 repo |
| `N_VIEWS` | `32` | 采集包等间隔抽帧 |
| `PROCESS_RES` | `504` | DA3 处理分辨率 |
| `ITERATIONS` | `30000` | 3DGS 训练步数 |
| `RES` | `2` | 3DGS 分辨率因子；要全分辨率传 `RES=`（空） |

## 风险与边界

- 横评口径注意：本路线是"前馈几何 + 逐场景优化"，与 QuerySplat/InfiniSplat v2
  的纯前馈不在同一成本量级，汇报时要分组比。
- DA3 位姿弱纹理/重复纹理场景可能塌缩（infinisplat_v2 三视角教训），跑完先看
  `da3_human` 的位姿可视化（02b_viz_trajectory.py）。
- DA3 深度是 metric 尺度，与 Remy GT 的尺度关系需在评测对齐阶段统一。
