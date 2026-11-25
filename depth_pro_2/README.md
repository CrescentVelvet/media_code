# Depth Pro 2 H+ — 三帧深度、位姿点云与跨帧一致性

使用 Apple 官方 [ml-depth-pro-2](https://github.com/apple-aiml-research/ml-depth-pro-2) H+（872M）进行三帧联合推理，按已知相机内参恢复轴向深度，输出逐帧 PCD、有限半径点 splat、PCD depth 和 RGB/深度合成视频。官方代码作为仓外 sibling，不复制进本仓。

**权重许可仅允许非商业科研用途。** 实验实测及限制见 [EXPERIMENTS.md](EXPERIMENTS.md)；WSL 从零运行见 [README_wsl.md](README_wsl.md)。

## 常用命令

以下相对路径均从 `media_code/` 解析；显式提供模型、输入、位姿和输出，不依赖脚本默认值。

```bash
# 首次准备：克隆官方代码、安装依赖、链接 H+ 权重并下载 FOV fallback。
DEPTH_PRO_2_DIR=../ml-depth-pro-2 \
MODEL_DIR=../../model/depth_pro_2 \
RESULTS_DIR=../depth_pro_2_results \
bash depth_pro_2/00a_setup_env.sh

# 两组三帧端到端：准备 -> H+ -> COLMAP 审核（路由器）-> PCD/视频/一致性。
DEPTH_PRO_2_DIR=../ml-depth-pro-2 \
MODEL_DIR=../../model/depth_pro_2 \
CKPT_PATH=../../model/depth_pro_2/depthpro2-h-eivtm5w7m5.pt \
FOV_CKPT_PATH=../../model/depth_pro_2/depthpro2-fov-ew55zg4kq7.pt \
RESULTS_DIR=../depth_pro_2_results \
HUMAN_RAW_DIR=../data/human_capture \
ROUTER_RAW_DIR=../data/router_three_images \
ROUTER_POSE_DIR=../data/router_colmap_sequence \
SCENE_DIR=../depth_pro_2_results \
GPU=0 bash depth_pro_2/04_run_all.sh
```

单场景已有缓存时，仅重跑几何与一致性，不重复 H+ 推理。`02b_run_single_frame_inference.sh` 是独立单帧对照：每组三帧分别调用 `model.infer(T=1)`，不共享跨帧特征；`render_single_frame_splat_depth.py` 用原始相机生成全部有效点的自身视角 PCD splat / PCD depth。路由器仍继承联合实验的整组 SfM 尺度，禁止逐帧尺度配准。具体命令见 [README_wsl.md](README_wsl.md)。

**官方 H+ 不能输入相机外参。** `model.infer(x, f_px=None)` 只接收 RGB 与焦距；跨帧注意力使用图像 patch 的二维 RoPE，没有外参、相机射线或相对位姿条件。`transforms.json` 的 c2w 用于输出深度后的反投影与世界坐标对齐，不能约束网络直接预测多视图一致的深度。佳佳原始联合与独立单帧 PLY **已经应用这套位姿**，不是局部相机坐标直接拼接；`03b_audit_pose.sh` 复核输入/位姿身份、复现旧 PLY，并以相同深度对照忽略外参的反事实情况，不重跑模型、不做 ICP。

```bash
SCENE_DIR=../depth_pro_2_results/human bash depth_pro_2/03_render_geometry.sh
SCENE_DIR=../depth_pro_2_results/router \
POSE_DIR=../data/router_colmap_sequence \
bash depth_pro_2/03a_audit_colmap.sh
SCENE_DIR=../depth_pro_2_results/router bash depth_pro_2/03_render_geometry.sh
```

## 输入与相机

- `human`：Remy `transforms.json` 与 HEIC 序列，抽首/中/尾帧；原始 `transform_matrix = c2w OpenGL`，转为 `c2w_cv = transform_matrix @ diag(1,-1,-1,1)`。
- `router`：三张 JPG，必须与 `POSE_DIR/image/` 同名图像逐字节一致；使用长序列 COLMAP 的对应相机，不把三图独立 SfM 失败当作有效位姿。
- COLMAP 只支持 `PINHOLE`，`w2c=(R,t)`，`c2w=[R.T, -R.T@t]`。RGB 去畸变、裁边、缩放时同步变换 K。
- 官方 `inverse_depth = fx/(W*Z)`；脚本显式传入 manifest fx，并保存原始 `normalized_inverse_depth`。同一联合 clip 要求相同 fx。
- 人像使用米制相机；路由器为 SfM 任意单位。路由器通过三帧**合并的稀疏点深度比例中位数**求一个整组尺度，不逐帧拟合、不 ICP。

## 输出

```text
<SCENE_DIR>/
  human/
    rgb/ depth/ normalized_inverse_depth/ depth_rgb/
    pcd/00.ply ... 02.ply
    fused_no_registration.ply
    splat/ pcd_depth/ pcd_reprojection/
    single_frame/
      depth/ normalized_inverse_depth/ own_camera_splat_depth/
      pcd/00.ply ... 02.ply  # 独立 T=1 推理，全部有效点，不抽样/不融合
      pcd/pcd_export_audit.json
    pose_conditioned/         # 外参接口审计与同深度后处理对照，不是位姿条件推理
      pose_audit.json supplied_poses_world_metrics.json
      identity_extrinsics_counterfactual_metrics.json
      side_by_side/ fused_pose_comparison.jpg
      fused_supplied_poses_world.ply  # 每 8 像素取 1 点的预览，不替代全点原 PLY
    rgb_depth_pcd_comparison.mp4
    crossview_pcd_comparison.mp4
    view_splat_depth_100k/ # 输入/内插/外插三视角，固定 10 万点 splat/depth
    view_splat_depth_full_9views/ # 三输入+三内插+三外插，全部有效点不抽样
    consistency.json consistency/
  router/
    ...同上...
    colmap_audit.json sfm_scale.json
  router_unscaled/       # 未统一 SfM 尺度的追溯诊断，不是最终推荐点云
  router_sfm/            # 三图独立 SfM 失败日志，未提供替代相机
```

视频把每个输入帧重复三次慢放，不是生成或插值的新帧。`splat` 为远到近点圆盘渲染，**不是训练得到的 3D Gaussian Splatting**；`pcd_depth` 来自点云投影，数值自检使用未膨胀单点。每帧百万级点投回自身相机、圆盘半径触及 2.5px 上限时几乎完全覆盖，缩图后与 RGB/原始深度相似是预期现象；跨相机投影才更容易看到空洞和错位。深度 RGB 与 PCD depth 统一采用官方近红远蓝的 TURBO 配色，同场景共用 2%～98% 显示范围；颜色不改变 NPZ/NPY 数值。

`render_view_splat_depth.py` 另从三帧有效融合点集中以固定种子 `20261123` 无放回抽 **100,000** 点，三视角共享同一子集：真实 Frame 01 输入相机、`t=0.5` 内插、`t=-0.5` 外插。虚拟相机沿 00→01 做相对旋转 SLERP（rotvec 等价实现）与中心线性插值，外插为反方向半个基线；K 使用共同输入内参。输出 RGB 点圆盘、投影轴向 Z、原始 NPY、相机 JSON、`sampled_100k.ply` 与 `view_render_audit.json`。不做 ICP、逐帧调尺度或新模型推理；空洞混合抽样、遮挡、几何/位姿与运动误差，覆盖率不是精度。运行与 PPT 再生成见 [FIGURES.md](FIGURES.md)。

`render_full_view_splat_depth.py` 为全点九视角版本：每组使用三帧融合 PCD 的全部有效点，不抽样；佳佳 5,289,530 点，路由器 3,981,312 点。输入为真实 Frame 00/01/02；内插为 00→01、01→02、00→02 的 `t=0.5`；外插为 00→01 `t=-0.5`、01→02 `t=1.5`、00→02 `t=-0.25`。虚拟相机使用第二端点 K；没有新几何或采集图像。输出 `view_splat_depth_full_9views/` 的 18 张 PNG、相机 JSON、点数/源 PLY/源深度哈希与审计，不重复写大 PLY。

## 连续机位：82 帧 RGB／深度与世界系 PCD

`01c_prepare_sequence.sh`、`02c_run_sequence_inference.sh`、`06c_render_sequence_results.sh` 处理全部有序 Remy 相机，官方 buffered 滑窗只接收 RGB 与已知 fx，c2w 仅用于深度输出后的世界系反投影。输出 `human_sequence_82/` 的两段 82 帧视频、82 份逐帧全点 PLY 和不配准／不体素合并的融合 PLY；视频每帧用前后各 8 帧作局部邻域，预览每 2×2 像素取 1 点。PCD splat 是 RGB 点圆盘，不是训练的 3DGS；完整 PLY 与预览抽样必须区分。

实测密机位没有自动消除人物重影：同一 buffered 深度、固定目标相机下，全 82 帧直接叠加比三帧保留更多时刻的姿态。详见 [EXPERIMENTS.md](EXPERIMENTS.md)；WSL 显式路径命令见 [README_wsl.md](README_wsl.md)，PPT 重建和验收见 [FIGURES.md](FIGURES.md)。

## 视频输入：官方 buffered 滑窗

`05_run_video_inference.sh` 用官方 `infer_video(window=16, overlap=4, mode="buffered")` 逐帧处理整段视频，默认使用第一窗口估计的全视频共享焦距；`VIDEO_ONLY=1 bash depth_pro_2/06_render_video_results.sh` 只输出原 RGB／全场景深度并排 MP4，不生成 PCD 或额外表格。省略 `VIDEO_ONLY` 时，旧模式另输出代表帧局部相机 PLY、PCD splat/depth 与来源审计。视频没有标定 K / 相机位姿时，**不得把逐帧局部 PCD 拼成已注册世界坐标点云**。官方天空逆深度可为 0、深度为 `inf`，只从有限深度导点；展示时以共同色阶保留无有限点区域，不伪造深度。

```bash
VIDEO_PATH=/mnt/c/Users/wangyufeng/Downloads/7ef19637-35f5-4473-8b59-9241e1a083a7.mp4 \
OUT_DIR=/mnt/d/output/results/depth_pro_2_results/video_7ef19637 \
DEPTH_PRO_2_DIR=/home/velvet/repos/ml-depth-pro-2 \
CKPT_PATH=/home/velvet/model/depth_pro_2/depthpro2-h-eivtm5w7m5.pt \
FOV_CKPT_PATH=/home/velvet/model/depth_pro_2/depthpro2-fov-ew55zg4kq7.pt \
WINDOW=16 OVERLAP=4 FRAME_STRIDE=1 MAX_SIDE=1280 GPU=0 \
bash depth_pro_2/05_run_video_inference.sh
SCENE_DIR=/mnt/d/output/results/depth_pro_2_results/video_7ef19637 \
PCD_FRAME_STRIDE=30 RGB_FROM_SOURCE=1 \
bash depth_pro_2/06_render_video_results.sh
```

## Config

| 变量 | 含义 |
|---|---|
| `SCENE_DIR` | 输出根（04）或单场景目录（03/03a） |
| `HUMAN_RAW_DIR` | Remy 人像采集目录 |
| `ROUTER_RAW_DIR` | 三 JPG 目录 |
| `ROUTER_POSE_DIR` | 含 `image/`、`sparse/0/` 的 COLMAP 序列 |
| `CKPT_PATH` | 本地 H+ 权重 |
| `FOV_CKPT_PATH` | FOV 权重，保留官方 fallback；有 manifest fx 时不用于最终深度 |
| `DEPTH_PRO_2_DIR` | 官方仓 sibling |
| `CONDA_ENV` / `GPU` | conda 环境 / 选卡 |
