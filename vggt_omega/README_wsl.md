# vggt_omega/README_wsl.md — VGGT-Ω（位姿+深度）+ 传统 3DGS（WSL 复现）

VGGT-Ω（CVPR 2026 Oral，Meta/VGG，1B 参数）前馈预测相机位姿 + 深度，
深度反投影建点云、转 COLMAP 后接官方 gaussian-splatting 逐场景训练出 ply。
与 v1 VGGT 的区别：register attention、训练省 ~70% 显存、支持动态场景、
点由深度反投影生成（无 point-map head）。

- 官方仓：<https://github.com/facebookresearch/vggt-omega>（代码已 clone 到 `~/repos/vggt-omega`）
- 权重：<https://huggingface.co/facebook/VGGT-Omega>（⚠️ **gated，需申请**，CC-BY-NC 4.0）

## 权重复用池（D:\wheel）

**下载模型前先搜 `D:\wheel`，新下载的模型也落盘到 `D:\wheel` 下对应子目录，消费侧一律软链**（2026-11-05 峰哥约定）。

本权重复用池位置：`D:\wheel\vggt_human_ms\VGGT-Omega\vggt_omega_1b_512.pt`（SHA256 已对照官方校验）。`01_download_models.sh` 会按 wheel → 消费侧 → 下载到 wheel 的顺序自动处理；下载前会做 SHA256 校验。

## 常用命令

```bash
cd /mnt/c/code/media_code

# 0) env（转发链路用 vggt_human env，本 env 供独立调官方 API）
bash vggt_omega/00a_setup_env.sh

# 1) 下权重（gated：先 HF 申请 + hf login）
bash vggt_omega/01_download_models.sh

# 2) Remy 采集包 → VGGT-Ω 推理 → COLMAP 场景
RAW_DIR="/mnt/d/dataset/手部微动样例/01e954de1b8a4cde9315c976f4dcb964-佳佳抱枕" \
  N_VIEWS=32 bash vggt_omega/02_run_inference.sh

# 3) 传统 3DGS 训练 → ply
RAW_DIR="/mnt/d/dataset/手部微动样例/01e954de1b8a4cde9315c976f4dcb964-佳佳抱枕" \
  ITERATIONS=30000 bash vggt_omega/03_train_3dgs.sh
```

输出：`~/output/vggt_omega_results/<场景名>/{02_vggt/, 03_source/(images+sparse/0), 04_model_3dgs/}`

## 复用关系（重要）

本仓是**编排层**，不重写已验证的链路：

| 步骤 | 实际执行 | 环境 |
|---|---|---|
| 采集包转换 | `infinisplat_v2/07_capture_to_npz.py` | infinisplat_v2（HEIC 解码） |
| VGGT-Ω 推理 | `vggt_human/02_run_inference.sh`（run_batch.py） | vggt_human |
| npz → COLMAP | `vggt_human/03_npz_to_colmap.sh` | vggt_human |
| 3DGS 训练 | `vggt_human/04_train_3dgs.sh`（官方 gaussian-splatting） | vggt_human |

vggt_human 本身就是围绕 VGGT-Ω 建的 pipeline（predictions.npz 的字段名即 VGGT-Ω 输出）。

## Config (env vars)

| var | default | note |
|---|---|---|
| `N_VIEWS` | `32` | 采集包等间隔抽帧 |
| `RESOLUTION` | `512` | VGGT-Ω 输入分辨率（1b_512） |
| `CONF_THRES` | `20` | 深度置信度百分位（过滤点云） |
| `ITERATIONS` | `30000` | 3DGS 训练步数 |
| `RES` | `2` | 3DGS 分辨率因子；要全分辨率传 `RES=`（空） |

## 风险与边界

- **权重 gated 是硬阻塞**：没申请前 01 必然失败；QuerySplat 也需要同一份权重。
- VGGT-Ω 点云由深度反投影生成，位姿/深度误差会复合成 3D 噪声——COLMAP 环节有
  置信度过滤 + 体素降采样（vggt_human/03 的 TARGET_POINTS）。
- 输出在 vggt_human 尺度系（npz 的 world_points_from_depth），与 GT 坐标系的差异
  需在评测对齐阶段统一处理。
