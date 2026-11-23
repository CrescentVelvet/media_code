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

单场景已有缓存时，仅重跑几何与一致性，不重复 H+ 推理：

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
    rgb_depth_pcd_comparison.mp4
    crossview_pcd_comparison.mp4
    consistency.json consistency/
  router/
    ...同上...
    colmap_audit.json sfm_scale.json
  router_unscaled/       # 未统一 SfM 尺度的追溯诊断，不是最终推荐点云
  router_sfm/            # 三图独立 SfM 失败日志，未提供替代相机
```

视频把每个输入帧重复三次慢放，不是生成或插值的新帧。`splat` 为远到近点圆盘渲染，**不是训练得到的 3D Gaussian Splatting**；`pcd_depth` 来自点云投影，数值自检使用未膨胀单点。深度 RGB 与 PCD depth 统一采用官方近红远蓝的 TURBO 配色，同场景共用 2%～98% 显示范围；颜色不改变 NPZ/NPY 数值。

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
