# Scene Splatter — 原理与排错手册

README 只放运行信息；原理、坑点、决策依据记在这里。

## 1. 流水线四段（scenesplatter.py 实际执行顺序）

1. **Flash3D 初始化**：单图前馈出整段轨迹的 3DGS（`GaussianPredictor`，RE10K 上训练的
   v2 ckpt）。`num_render_frames = len(poses)`，一次前向渲染全部轨迹帧。
2. **轨迹抽稀**：`image_list/camera_list = [::camera_sample_interval]`（默认 2）。
3. **ViewCrafter 增强**：轨迹按 `per_video_length=16`（重叠 `num_overlap_frames=2`）切段，
   每段用点云条件 I2V（DynamiCrafter 系，`viewcrafterpipe.ViewCrafterPipeline`）生成增强视频。
   动量机制：latent 级动量（原图特征回注降噪）+ 像素级动量（按 3DGS scale map 融合
   「一致版」与「自由版」两路生成）。
4. **迭代 3DGS**：每段视频监督一轮 `optimize_gaussian`（5000 步，L1+SSIM，densify/prune
   与原版 3DGS 相同），最后一轮用全轨迹渲染出 `render_video_{i}.mp4` 和 confidence_maps。

迭代轮数 = `len(camera_list) // (per_video_length - overlap)`。默认参数下 N=100 帧轨迹
抽稀成 50 → 3 轮（50//14）。

## 2. 为什么官方没有 PLY

`main()` 最后 `return gaussians` 就结束——`gaussians` 是内存里的 `GaussianModel`，
不落盘。官方论文指标都是对渲染视频算的，没打算让人拿去做资产。横评要 PLY 必须补
`gaussians.save_ply(...)`（`patch_save_ply.py`，幂等）。

## 3. 配置只能改 config.yaml 的原因

`main(cfg)` 内部 `parser = get_parser(); opts = parser.parse_args()`（ViewCrafter 的
argparse）拿的是**整个** `sys.argv[1:]`。hydra 的 CLI 覆盖（如 `config.image_path=...`）
会原样传进 ViewCrafter 的 parser → unrecognized arguments 报错。所以官方 README 才说
"refer to configs/config.yaml to change"。`02_run_inference.sh` 用正则直改 yaml 五行。

## 4. 相机轨迹格式（README 与代码的矛盾，待首跑用示例核实）

- README："`['intrinsics', 'poses']`...formed as 4 × 4 matrix"。
- 代码：`process_projs(proj)` 取 `proj[0..3] = fx, fy, cx, cy`（RE10K 风格归一化），
  即 intrinsics 每项是 flat-4；4×4 矩阵喂进去 `proj[1]=0` 会把 fy 置零。
- `data_to_c2w(w2c)`：`poses` 是 **w2c**，取逆得 c2w；OpenCV 约定（`get_sparse_depth`
  里 depth 取 `T_w2c @ X` 的 z，z 前向为正）。
- **结论（【待首跑核实】）**：示例 pickle 大概率是 intrinsics=per-frame flat-4 np.float32、
  poses=per-frame 4×4 w2c。`02_gen_traj.py --inspect-example` 打印示例真实结构；
  若示例确实是别的布局，改生成器的 `intr` 布局即可（轨迹几何本身不受影响）。

## 5. 权重与 ckpt 选择

| 权重 | 来源 | 说明 |
|---|---|---|
| `flash3d/model_re10k_v2.pth` | HF `einsafutdinov/flash3d` | 官方 README 指定 v2（RE10K） |
| `viewcrafter/checkpoints/model.ckpt` | HF `Drexubery/ViewCrafter_25` | 25 帧单图 NVS ckpt；Scene Splatter 配 `per_video_length=16`，lvdm 对帧数是否敏感【待实测】，若 shape 报错换 `ViewCrafter_16` |
| `viewcrafter/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth` | naverlabs 官方直链 | ViewCrafter 点云条件用 |

三者均无官方 SHA256 发布，01 只标注来源、不做校验和核对。

## 6. 横评注意事项（compare_4methods 口径）

- **输入 OOD**：Flash3D/ViewCrafter 都训于 RE10K 场景，人物输入大概率劣化；
  评测首选用场景图，人物样例单独标注。
- **轨迹统一**：Scene Splatter 的轨迹完全由我们生成（02_gen_traj），天然可控；
  把 YAW_SPAN/RADIUS/FOV_Y 调成与 FlashWorld/Gen3R 同口径再比。
- **生成式无全 GT**：单图输入的脑补区域无真值，指标只对观测弧有效
  （复用 compare_4methods `audit_23` 的 ARC=1 口径思路）。
- **耗时结构不同**：FlashWorld 秒级前馈 / Gen3R 分钟级前馈 / Scene Splatter 迭代优化
  （轮数 × 5000 步）——横评表要分开报「前馈耗时」和「优化耗时」。

## 7. vendored 代码与许可

Scene-Splatter 把 flash3d（MIT）、3DGS（Inria LICENSE，研究用）、ViewCrafter 裁剪版
直接 vendored 进仓（非 submodule），整体 MIT。编译光栅器时注意 CUDA 版本与 torch
cu118 对齐（conda 的 nvcc 11.8）。
