# Depth Pro 2 (WSL Ubuntu 24.04) — 本地复现指南

本文件是 [`README.md`](README.md) 的 WSL 实测版。H+ 三帧联合推理已在 RTX 3090 / `sharp` 环境完成；实验结论与限制见 [EXPERIMENTS.md](EXPERIMENTS.md)。

## 前提与路径

- WSL2 Ubuntu 24.04、NVIDIA 驱动、Miniconda；本机复用 `sharp`：Python 3.13.15、PyTorch 2.8.0+cu128。
- 权重：Apple H+ 与 FOV fallback，非商业科研用途。
- 官方代码：`~/repos/ml-depth-pro-2`，实验使用 revision `aa389cbb0b74341664ed663a0966c859905cd6c0`。
- Windows 输入通过 `/mnt/d/` 只读访问，推理输出先写 Linux 文件系统。

| 数据 | WSL 路径 |
|---|---|
| 人像 | `/mnt/d/dataset/手部微动样例/01e954de1b8a4cde9315c976f4dcb964-佳佳抱枕` |
| 路由器三图 | `/mnt/d/dataset/物体三图数据/路由器` |
| 路由器长序列位姿 | `/mnt/d/dataset/物体反光数据/路由器` |
| 归档 H+ 与 FOV 权重 | `/mnt/d/wheel/depth_pro_2_ms/` |
| 用户原始 H+ 下载（已保留） | `/mnt/d/迅雷下载/depthpro2-h-eivtm5w7m5.pt` |

## 首次准备

```bash
cd /mnt/c/code/media_code

# ── 0) 官方代码、依赖、D:\wheel 两份权重的软链接 ──
# 输出：~/repos/ml-depth-pro-2、~/model/depth_pro_2、~/output/depth_pro_2_results
CONDA_ENV=sharp \
DEPTH_PRO_2_DIR=/home/velvet/repos/ml-depth-pro-2 \
MODEL_DIR=/home/velvet/model/depth_pro_2 \
RESULTS_DIR=/home/velvet/output/depth_pro_2_results \
bash depth_pro_2/00a_setup_env.sh
```

若 H+ 已手动下载到其他位置，设 `CKPT_PATH=/实际路径/depthpro2-h-eivtm5w7m5.pt`。默认归档为 `D:\wheel\depth_pro_2_ms\`，H+ SHA-256 为 `ae39cfb7...a818340a2a`，FOV 为 `37fd480f...fa659a0a817c4`；也可设 `WEIGHTS_DIR` 覆盖。环境安装会读取 `proxy.env`；境外 GitHub/CDN 与境内 pip 镜像的代理规则不要混用。

## 全流程命令

```bash
cd /mnt/c/code/media_code

# ── 1) 人像：抽首/中/尾帧并联合推理 ──
# 输入：82 帧 Remy 包；输出：3 张 1152x1536 RGB、深度与相机
SCENE=human \
RAW_DIR=/mnt/d/dataset/手部微动样例/01e954de1b8a4cde9315c976f4dcb964-佳佳抱枕 \
OUT_DIR=/home/velvet/output/depth_pro_2_results/human \
bash depth_pro_2/01_prepare_data.sh

DEPTH_PRO_2_DIR=/home/velvet/repos/ml-depth-pro-2 \
CKPT_PATH=/home/velvet/model/depth_pro_2/depthpro2-h-eivtm5w7m5.pt \
FOV_CKPT_PATH=/home/velvet/model/depth_pro_2/depthpro2-fov-ew55zg4kq7.pt \
INPUT_DIR=/home/velvet/output/depth_pro_2_results/human \
OUT_DIR=/home/velvet/output/depth_pro_2_results/human \
GPU=0 bash depth_pro_2/02_run_inference.sh

# ── 2) 人像：点云、splat、PCD depth、两路视频与一致性 ──
SCENE_DIR=/home/velvet/output/depth_pro_2_results/human \
bash depth_pro_2/03_render_geometry.sh

# ── 3) 路由器：读取三图及同名长序列 COLMAP 位姿 ──
# 会校验三图与位姿序列原图 SHA-256 完全相同。
SCENE=router \
RAW_DIR=/mnt/d/dataset/物体三图数据/路由器 \
POSE_DIR=/mnt/d/dataset/物体反光数据/路由器 \
OUT_DIR=/home/velvet/output/depth_pro_2_results/router \
bash depth_pro_2/01_prepare_data.sh

DEPTH_PRO_2_DIR=/home/velvet/repos/ml-depth-pro-2 \
CKPT_PATH=/home/velvet/model/depth_pro_2/depthpro2-h-eivtm5w7m5.pt \
FOV_CKPT_PATH=/home/velvet/model/depth_pro_2/depthpro2-fov-ew55zg4kq7.pt \
INPUT_DIR=/home/velvet/output/depth_pro_2_results/router \
OUT_DIR=/home/velvet/output/depth_pro_2_results/router \
GPU=0 bash depth_pro_2/02_run_inference.sh

# ── 4) 路由器：先审核 SfM，再导出一个整组尺度的 PCD ──
# 不做 ICP、不逐帧拟合。必须先运行本步，之后才能用正确 SfM 单位渲染。
SCENE_DIR=/home/velvet/output/depth_pro_2_results/router \
POSE_DIR=/mnt/d/dataset/物体反光数据/路由器 \
bash depth_pro_2/03a_audit_colmap.sh
SCENE_DIR=/home/velvet/output/depth_pro_2_results/router \
bash depth_pro_2/03_render_geometry.sh

# ── 4b) 两组各抽同一 10 万点子集，投至输入/内插/外插相机 ──
# 输入：已有 depth/*.npz、manifest.json、geometry.json；不重跑模型
for SCENE in human router; do
  env SCENE_DIR=/home/velvet/output/depth_pro_2_results/$SCENE \
    VIEW_POINT_COUNT=100000 VIEW_SEED=20261123 VIEW_EXTRAPOLATION=-0.5 \
    python depth_pro_2/render_view_splat_depth.py
done
# 输出：各场景 view_splat_depth_100k/{input,interpolated,extrapolated}_*
#       sampled_100k.ply、相机 JSON 与 view_render_audit.json
# Git Bash 调 wsl 时：用 env VAR=/mnt/... 传路径，避免路径被宿主转换。

# ── 4c) 全点九视角：每组三输入、三内插、三外插，不抽样 ──
for SCENE in human router; do
  env SCENE_DIR=/home/velvet/output/depth_pro_2_results/$SCENE \
    VIEW_POINT_RADIUS=0.8 \
    ~/miniconda3/envs/sharp/bin/python depth_pro_2/render_full_view_splat_depth.py
done
# 输出 view_splat_depth_full_9views/：18 PNG + 9 相机 JSON + 审计/来源哈希。
# ── 4d) 独立单帧对照：三次 T=1 推理 + 各帧原相机全点投影 ──
# 不覆盖三帧联合结果；原始输入、K、fx 和 H+ 权重相同。
env DEPTH_PRO_2_DIR=/home/velvet/repos/ml-depth-pro-2 \
  CKPT_PATH=/home/velvet/model/depth_pro_2/depthpro2-h-eivtm5w7m5.pt \
  FOV_CKPT_PATH=/home/velvet/model/depth_pro_2/depthpro2-fov-ew55zg4kq7.pt \
  RESULTS_DIR=/home/velvet/output/depth_pro_2_results \
  GPU=0 OFFICIAL_REVISION=aa389cbb0b74341664ed663a0966c859905cd6c0 \
  bash depth_pro_2/02b_run_single_frame_inference.sh
for SCENE in human router; do
  env SCENE_DIR=/home/velvet/output/depth_pro_2_results/$SCENE \
    ~/miniconda3/envs/sharp/bin/python depth_pro_2/render_single_frame_splat_depth.py
done
# 输出 single_frame/{depth,normalized_inverse_depth,own_camera_splat_depth,pcd}。
# 无跨帧注意力/融合/ICP/逐帧尺度；路由器复用联合实验整组尺度。

# ── 4e) 已有单帧深度时：仅导出六个逐帧 PLY，不重跑模型 ──
for SCENE in human router; do
  SCENE_DIR=/home/velvet/output/depth_pro_2_results/$SCENE \
    ~/miniconda3/envs/sharp/bin/python depth_pro_2/export_single_frame_pcd.py || exit 1
done
# 输出各场景 single_frame/pcd/{00,01,02}.ply 与 pcd_export_audit.json。
# 全部有效点、原 c2w、原 RGB；回读校验坐标/颜色/来源，不额外合并三帧。

# ── 4f) 位姿接口与三帧 c2w 复核：复用联合深度，不重跑模型 ──
# 官方 infer 只有 RGB 与 f_px；外参只能在输出深度后用于世界系反投影。
# 此包三 HEIC 与旧实验相同，已应用的 c2w 也逐项相同，不声称新增位姿收益。
SCENE_DIR=/mnt/d/output/results/depth_pro_2_results/human \
RAW_DIR=/mnt/d/dataset/人像手部微动三图数据/01e954de1b8a4cde9315c976f4dcb964-佳佳抱枕 \
OUT_DIR=/mnt/d/output/results/depth_pro_2_results/human/pose_conditioned \
DEPTH_PRO_2_DIR=/home/velvet/repos/ml-depth-pro-2 \
bash depth_pro_2/03b_audit_pose.sh
# 输出：pose_audit.json、忽略/使用外参的同深度指标、重投影对照图和 1/8 抽样 PLY。
# 旧全点 PLY 保持不变；无 ICP、逐帧尺度或模型结构修改。
# ── 4g) 82 连续机位：全部 RGB 深度、局部邻域 PCD 视频与全点 PLY ──
# 输入：82 HEIC + transforms.json；外参只用于深度输出后的世界系反投影。
# 正式输入准备 / 推理 / 渲染分别执行；已有推理时只执行最后的渲染命令。
SCENE=human_sequence_82 \
RAW_DIR=/mnt/d/dataset/手部微动样例/01e954de1b8a4cde9315c976f4dcb964-佳佳抱枕 \
OUT_DIR=/mnt/d/output/results/depth_pro_2_results/human_sequence_82/input \
MAX_SIDE=1536 PLAYBACK_FPS=12 \
bash depth_pro_2/01c_prepare_sequence.sh

DEPTH_PRO_2_DIR=/home/velvet/repos/ml-depth-pro-2 \
CKPT_PATH=/home/velvet/model/depth_pro_2/depthpro2-h-eivtm5w7m5.pt \
FOV_CKPT_PATH=/home/velvet/model/depth_pro_2/depthpro2-fov-ew55zg4kq7.pt \
INPUT_DIR=/mnt/d/output/results/depth_pro_2_results/human_sequence_82/input \
OUT_DIR=/mnt/d/output/results/depth_pro_2_results/human_sequence_82 \
WINDOW=16 OVERLAP=4 OFFICIAL_REVISION=aa389cbb0b74341664ed663a0966c859905cd6c0 \
GPU=0 bash depth_pro_2/02c_run_sequence_inference.sh

SCENE_DIR=/mnt/d/output/results/depth_pro_2_results/human_sequence_82 \
PCD_PIXEL_STRIDE=2 NEIGHBORHOOD_RADIUS=8 PCD_POINT_RADIUS=0.65 \
VIDEO_FPS=12 FULL_PLY=1 METRIC_STRIDE=8 \
bash depth_pro_2/06c_render_sequence_results.sh
# 输出：rgb_depth_comparison.mp4、pcd_splat_depth_comparison.mp4；
# pcd/000000.ply～000081.ply、fused_all_82_no_registration.ply。
# 全点 PLY 不抽样；视频仅每 2x2 像素取 1 点；无 ICP / 逐帧尺度 / 体素合并。
```

输出：`human/`、`router/` 各有 `pcd/00.ply`～`02.ply`、`fused_no_registration.ply`、`splat/`、`depth_rgb/`（近红远蓝）、`pcd_depth/`、`pcd_reprojection/`、`consistency.json`、`rgb_depth_pcd_comparison.mp4`、`crossview_pcd_comparison.mp4`。路由器另有 `colmap_audit.json`、`sfm_scale.json`。

已有缓存的一键复跑：

```bash
SCENE_DIR=/home/velvet/output/depth_pro_2_results \
HUMAN_RAW_DIR=/mnt/d/dataset/手部微动样例/01e954de1b8a4cde9315c976f4dcb964-佳佳抱枕 \
ROUTER_RAW_DIR=/mnt/d/dataset/物体三图数据/路由器 \
ROUTER_POSE_DIR=/mnt/d/dataset/物体反光数据/路由器 \
DEPTH_PRO_2_DIR=/home/velvet/repos/ml-depth-pro-2 \
CKPT_PATH=/home/velvet/model/depth_pro_2/depthpro2-h-eivtm5w7m5.pt \
FOV_CKPT_PATH=/home/velvet/model/depth_pro_2/depthpro2-fov-ew55zg4kq7.pt \
GPU=0 bash depth_pro_2/04_run_all.sh
# ── 5) 归档：复制到 Windows，保留 WSL 缓存并核对每个文件哈希 ──
SCENE_DIR=/home/velvet/output/depth_pro_2_results \
FINAL_DIR=/mnt/d/output/results/depth_pro_2_results \
bash depth_pro_2/08_copy_results.sh
```

## 注意

- Git Bash 可能把 `VAR=/home/...` 转成 Windows 路径。直接 `wsl bash` 执行，或用 `env SCENE_DIR=/home/... bash ...`，不要让 Windows 提前展开 `$HOME`。
- 三帧是联合前向，不是三次独立推理。视频每帧重复三次慢放，没有插值。
- 路由器原始 PCD 在未统一尺度时严重错位；`router_unscaled/` 只作追溯。最终版 `metric_to_sfm_scale=1.8113489289`。
- 人像原始采集 PCD 不是独立几何真值；没有 ICP、逐帧调尺度或遮挡感知融合。
- 结果最终复制到 `/mnt/d/output/results/depth_pro_2_results/`，不删除 WSL 缓存。
