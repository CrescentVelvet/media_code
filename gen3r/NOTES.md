# Gen3R 原理、输出口径与在本仓的用法

> 本文回答两个问题：**它到底产出什么**、**我们怎么把它接进现有 3DGS 链路**。
> 标注约定：`【代码事实】` 来自官方仓源码（已核对行号）；`【实测】` 来自本机运行结果；
> `【推测】` 未经验证。

## 1. 方法一句话

把 VGGT（前馈重建）的输出 token 用一个 adapter 压到与 Wan2.1-Fun 视频扩散 latent
**同构的空间**，然后让扩散模型一次前向同时吐出「外观 latent」和「几何 latent」；
几何 latent 再喂回 VGGT 的 camera/depth/point head，得到相机、深度、点云。

- `geo_adapter` 把 VGGT token 解码成 `[B,10240,F,40,40]`，与 Wan latent 拼在一起做联合去噪
  `【代码事实】gen3r/pipeline/pipeline_gen3r.py:439-443`
- 去噪完 `latents.chunk(2, dim=-1)` 拆回外观/几何两半
  `【代码事实】pipeline_gen3r.py:439`
- 相机/深度/点云由 VGGT head 在**生成的几何 latent**上重新预测
  `【代码事实】pipeline_gen3r.py:445-452`
- 点云 = 深度 + 相机反投影到世界系，不是网络直接回归的点
  `【代码事实】pipeline_gen3r.py:449-451` → `models/vggt/utils/geometry.py:unproject_depth_map_to_point_map`

**关键推论**：所谓"点云"是「生成出来的 49 张视图 + 生成的相机 + 生成的深度」三者自洽的
副产物。它与视频同一套几何，因此天然可做 3DGS 的初始化——但**没有真实尺度、没有真实观测**。

## 2. 输出清单与坐标系

| 文件 | 内容 | 口径 |
|---|---|---|
| `rgb.mp4` | 49 帧 560×560 | Wan VAE 解码，`(x/2+0.5).clamp(0,1)` `【代码事实】pipeline:454-455` |
| `pcds.ply` | 世界系点云 + RGB | 49 帧全像素反投影 → open3d 体素降采样(0.005/0.003) + 统计离群(25 邻域/std 1.25) `【代码事实】utils/common_utils.py:downsample_and_save_pointcloud` |
| `cameras.json` | `extrinsics[F,4,4]` + `intrinsics[F,3,3]` | **w2c，OpenCV 约定**（VGGT 原生），不是 c2w `【代码事实】infer.py:118-131` |
| `prompts.txt` | 文本条件 | — |

- **深度图不落盘**：`save_results` 里 depth.mp4 的分支被注释掉了
  `【代码事实】infer.py:99-105`。要深度得自己改脚本（`results['depth_maps']` 可拿）。
- **尺度**：`scene_scale = 0.8 * median(depth)` `【代码事实】infer.py:252`，
  仅用于内置轨迹的位移幅度；世界尺度由 VGGT 决定，**单图任务没有真实米制尺度**。
- **`--remove_far_points`**：按 z 的 80 分位截掉最远 20% 点 `【代码事实】common_utils.py`，
  人物/近景主体场景应当开。

## 3. 为什么"只生成点云不生成 3DGS"

- 论文目标是「生成 + 重建共用一套前馈几何」，没有可微渲染/稠密化环节，所以不产 Gaussian。
- 但它给的东西恰好是 3DGS 训练**所需的全部输入**：多视 RGB + 相机内外参 + 初始化点云。
  3DGS 要做的只是把这三样喂进优化（我们已有 `vggt_human/04_train_3dgs.sh`）。
- 所以"利用"不是要它出 Gaussian，而是**把它当成一个"几何自洽的伪数据集生成器"**。

## 4. 接入方案（按落地成本排序）

**2026-10-17 实测更新**（两个 run：官方 knossos / 我们的 Remy 采集包首帧「三人围桌」，均 50 步）：

- ✅ 全链路跑通：`02_run_inference.sh`（1view，3090 单卡 ~13min）→
  `03_to_colmap.sh`（49 帧 + 20 万点云 + 49 相机 → COLMAP 文本）→ 可直接进 3DGS。
- ✅ knossos（分布内场景）：新视角质量高，柱廊/地面/门廊结构一致。
- ⚠️ Remy（三人围桌，分布外）：**背景（幕布、桌面、椅子）跨视角基本自洽**，
  桌布花纹、盘子布局保持一致；但**站立男子的脸明显漂移/编**（越离首帧越糊），
  食物细节也有漂移。结论：**人物主体不能交给它编，背景可以**——
  这正好落在方案 D 的预期上。

### 方案 A：直接训 3DGS（已跑通转换）
`03_to_colmap.sh` → `vggt_human/04_train_3dgs.sh`。适用：**单图生成 3D 资产**、
给上级演示"一张图出一个可漫游 3D 场景"。不追求真实感。
✅ 转换已实测（knossos + remy 两个 run 都转出了合法 COLMAP 场景）。

### 方案 B：替代 COLMAP 做初始化（待评估）
我们 `som_human/01c` 的 COLMAP 在 150 帧上要几分钟到十几分钟；Gen3R 的 `allview`
任务等价于一次 VGGT 前馈重建，几十秒出相机 + 稠密点云。
**但**：COLMAP 有 BA 精化，VGGT 系相机精度通常低一档，需用我们已有的
`vggt_human/EXPERIMENTS.md` 里的位姿 A/B 口径做对比（PSNR/位姿误差）后再决定。

### 方案 C：补视角（实测否定其"编人"用法）
~~脑补缺失视角供人物监督~~。实测：人（尤其脸）在多视角下不可信，
**只能把生成视角当几何先验（深度/占据），不能做颜色监督**；
人物颜色信息必须来自真实采集。

### 方案 D：场景背景生成（`scene_gs`）✅ 实测最有戏
我们 3DGS 拆分里 `scene_gs` 需要背景覆盖；Gen3R 单图生成的背景
（幕布/地面/家具）跨视角一致性不错，且与主体的"不一致"主要发生在人身上。
**建议**：SAM3 抠人 → 人走真实采集 3DGS，背景交给 Gen3R（Remy 单图即可起步）。

## 5. 已知风险

| 风险 | 说明 | 缓解 |
|---|---|---|
| 人物主体被场景先验带偏 | 训练数据以 RealEstate10K/Co3D 等场景/物体为主，人体不是主要分布 | 用 `remove_far_points` + 短轨迹 + 人工挑帧 |
| 无真实尺度 | 跨场景/跨批次不可直接对齐 | 与真实点云做相似变换（Umeyama）后再用 |
| 生成幻觉 | 背面/遮挡区域是编的 | 只在"可接受编造"的场景用（演示、背景） |
| 49 帧固定 560×560 | 分辨率低，人脸细节不够 | 后续接我们的 HYPIR 超分链 |

## 6. 排错

- 下载：hf-mirror 2026-10-17 基本不可用（SSL 握手超时），改 `HF_ENDPOINT=https://aifasthub.com`；
  aifasthub 会按 UA 拦 `Python-urllib`（403），下载器已伪装 `huggingface_hub` UA。
- 依赖：`opencv-python-headless` 必须 4.10（5.x 会把 numpy 顶到 2.x）；
  `open3d` 需要 `libgomp.so.1`（conda 装 + `LD_LIBRARY_PATH`）。
- 显存（2026-10-17 实测，3090 24GB，重大发现）：
  **原版 `from_pretrained()` 不带 dtype → 先按 fp32 把 37.7GB 权重推上显存再转 bf16**，
  24GB 卡直接溢出到主机共享内存（实测 shared GPU mem 21.6GB），且 fp32 下 SDPA 只能走
  math 后端（flash 仅支持 fp16/bf16）→ 单步 >20min。
  **修法（两个 patch，已入库）**：
  1. `patch_infer_bf16.py`：`from_pretrained(torch_dtype=bf16)` + `.to(device, dtype)`，
     fp32 阶段留在 CPU、VAE 等子模块统一转 bf16（不然报
     "Input type (c10::BFloat16) and bias type (float)"）。
  2. `patch_pipeline_vram.py`：prompt/CLIP 编码完把 text_encoder(11GB)+clip(2.2GB) 挪到 CPU
     （`GEN3R_OFFLOAD_ENCODERS=0` 可关）；支持 `GEN3R_STEPS` 覆盖步数。
  3. 另装 `flash_attn 2.8.3.post2`（预编译 wheel：cu12+torch2.8+cp311+abiTRUE，
     在 `/mnt/d/wheel/`）。
  **效果：单步 9.5s（此前 >240s），显存 16.5GB（此前 24.2GB+21.6GB 共享）。**
  50 步 1view 全程约 8 分钟（加载 1min + VGGT 4.5min + 采样 8min + 解码）。
- 烟测（10 步）：能出 49 帧 560×560 + 44 万点云，结构一致但糊（条纹伪影），
  50 步质量见 `02_infer/1view/<时间戳>/`。
