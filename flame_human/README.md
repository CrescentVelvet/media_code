# flame_human — FLAME + 表情驱动的多人 3DGS 人脸链路

用 **FLAME 2020**（5023 顶点 / 300 shape / 100 expr）替代自研 3DMM，配 **DECA** 给 shape/expr 初值、
**MediaPipe 468 点**做 2D 观测，走「三阶段 3DMM 对齐 → 重心绑定高斯 → 表情驱动训练 →
三分支（head/body/scene）修整 → 渲染后增强」的链路。SfM/COLMAP 与 person mask 复用上游
`vggt_human` 的产出，本目录不重做。

与 `vggt_human` 的核心差异：那边是「468 点 + 中性脸 + 每帧刚体、无表情」；这边表情系数
贯穿高斯初始化、训练、后处理三个阶段（`local_exp` 是可训练参数，渲染 loss 直接 refine）。

**设计文档（决策 + 论证 + 遗留项）见
[../vggt_human/DESIGN_face_pipeline.md](../vggt_human/DESIGN_face_pipeline.md)。**

## 阶段 → 脚本

**推荐链路**（`run_all.sh` 即此顺序）：

| 阶段 | 脚本 | 说明 |
|---|---|---|
| — 权重检查 | `01_download_models.sh` | FLAME / DECA / HYPIR 就位检查 |
| — 468 点嵌入 | `01b_build_lm468_embedding.sh` | MediaPipe 468 → FLAME 重心嵌入 |
| — person mask | `01c_gen_person_masks.sh` | SAM3 person mask（复用 vggt_human worker）|
| 一 人脸检测 | `02_detect_faces.sh` | face bbox + 5 点（5 点仅副产物，下游不用）|
| 二 人脸匹配 | `03_match_faces.sh` | 三层 ID 关联：point-in-mask → IoM → 匈牙利 + 时序 |
| 三 人脸重建 | `04_recon_faces.sh` | 各向同性 crop → 512²；DECA 出初值；MediaPipe 出 468 点 |
| 四 3DMM 对齐 | `05_align_3dmm.sh` | 4.1 global / 4.2 local / 4.3 coeff（4.3 长程精修，见 NOTES）|
| 五 Avatar 初始化 | `06_init_avatar_gs.sh` | 表面采样保留 `(face_id, bary)`，硬绑定 + 自由高斯 |
| 七 Body 初始化 | `07_init_body_gs.sh` | point-to-triangle 切分 + 脖子下边界补缝 |
| 七a 三分支切分 | `07a_split_body_scene.sh` | 多视角投票把上游 ply 切成 body / scene |
| 六/八 头训练 | `08_train.sh` | `local_exp` 联合优化；epoch 48 触发增强 |
| 八d scene finetune | `08d_finetune_scene.sh` | 治大 yaw 雾噪（mask 补集监督 + 泄漏惩罚）|
| 八e 三分支剪枝 | `08e_prune_all.sh` | head 尺度门控 + scene 反向投票（治人物黑雾）|
| 九 渲染+后处理 | `09_enhance_post.sh` | 三分支合成渲染 → HYPIR 人脸增强（带清晰度闸门）|
| 十 WSL 搬运 | `10_move_output.sh` | Linux fs → `/mnt/d/` |

**实验性 / 负结果分支**（不在 `run_all.sh` 内，结论见 [EXPERIMENTS.md](EXPERIMENTS.md)）：

| 脚本 | 结论 |
|---|---|
| `01d_enhance_faces.sh` | HYPIR 增强**监督图** → head 锐度无增益（逐帧 2D 细节被 3DGS 平均掉）|
| `08b_finetune_body.sh` | body finetune → composite 变差（正解是 08e 剪枝）|
| `08c_prune_body.sh` | 08b 产物的剪枝后处理（工具可复用）|

> 原理、机制、排错与踩坑见 [NOTES.md](NOTES.md)；实验数据与结论见 [EXPERIMENTS.md](EXPERIMENTS.md)。

## 首次准备

```bash
cd <code-dir>/media_code
cp proxy.env.example proxy.env      # 确认 http_proxy / https_proxy 已取消注释

# WSL 本机
bash flame_human/00a_setup_env.sh
bash flame_human/00b_fix_env.sh     # 修 00a 遗留（numpy 降级 / csn 混装 / DECA 归位）
# 服务器
bash flame_human/00_setup_env.sh

bash flame_human/01_download_models.sh   # FLAME2020.pkl / DECA 权重 / lm468 嵌入
```

> WSL 路径策略：仓库/权重放 Linux fs（`~/repos`），权重根默认走 proxy.env 的
> `FLAME_HUMAN_MODEL_DIR=/mnt/d/wheel/flame_human_ms`，输出走
> `FLAME_HUMAN_RESULTS_DIR=$HOME/output/flame_human_results`（Linux fs，跑完 09 搬回 D 盘）。
> 通用名 `MODEL_DIR`/`RESULTS_DIR` 已被 vggt_human 占用，flame_human 一律不读。

权重目录布局：

```
$MODEL_DIR/                       # WSL 默认 /mnt/d/wheel/flame_human_ms
├── FLAME2020/generic_model.pkl   # 官网注册下载，见 download_urls.md
├── flame_lm468_embedding.npz     # MediaPipe 468 → FLAME 顶点重心嵌入（必须）
└── deca_model.tar                # DECA 预训练权重
```

## 常用命令

> 铁律：每条命令显式写出输入路径与输出路径，不要全靠脚本默认值。

```bash
# 一键（推荐链路 02 → 09）
GPU=0 \
  SOURCE_DIR=../vggt_human_results/03_source \
  PERSON_MASKS_DIR=../vggt_human_results/03_sam3_person_masks \
  RESULTS_DIR=../flame_human_results \
  bash flame_human/run_all.sh

# 分步（关键步骤；02/03/04/05/06/07/07a 同理）
GPU=0 SOURCE_DIR=../vggt_human_results/03_source \
  RESULTS_DIR=../flame_human_results \
  bash flame_human/02_detect_faces.sh

# 08 头训练（增强图监督版本用 IMAGES_DIR 指向 01d 产物）
GPU=0 RESULTS_DIR=../flame_human_results PID=0 \
  bash flame_human/08_train.sh

# 08d scene 定向 finetune（治大 yaw 雾噪）
GPU=0 RESULTS_DIR=../flame_human_results \
  SCENE_PLY=../flame_human_results/07_body_gs_src/scene_gs.ply \
  OUT_DIR=../flame_human_results/08d_finetune_scene \
  bash flame_human/08d_finetune_scene.sh

# 08e 三分支剪枝（head 尺度门控 + scene 反向投票）
GPU=0 RESULTS_DIR=../flame_human_results \
  bash flame_human/08e_prune_all.sh

# 09 渲染 + HYPIR 后处理增强（SKIP_RENDER=1 可复用已有渲染）
GPU=0 RESULTS_DIR=../flame_human_results \
  SKIP_RENDER=0 MIN_SHARPNESS=7.0 \
  bash flame_human/09_enhance_post.sh

# 10 搬回 Windows 盘（WSL）
bash flame_human/10_move_output.sh
```

## Config (env vars)

| var | default | note |
|---|---|---|
| `CONDA_ENV` | `flame_human` | conda 环境名 |
| `GPU` | 未设 | 设了才 export `CUDA_VISIBLE_DEVICES` |
| `UPSTREAM_DIR` | `../vggt_human_results` | 上游 SfM 与 mask 的根 |
| `SOURCE_DIR` | `$UPSTREAM_DIR/03_source` | COLMAP 场景（images + sparse）|
| `PERSON_MASKS_DIR` | `$UPSTREAM_DIR/03_sam3_person_masks` | SegTrack/SAM3 person mask；没有上游产物时先跑 01c（默认输出 `$RESULTS_DIR/01c_sam3_person_masks`）|
| `RESULTS_DIR` | `FLAME_HUMAN_RESULTS_DIR` 或 `../flame_human_results` | 本链路输出（WSL 见 proxy.env）|
| `MODEL_DIR` | `FLAME_HUMAN_MODEL_DIR` 或 `../../model/flame_human` | 权重根（WSL 见 proxy.env）|
| `FLAME_MODEL` | `$MODEL_DIR/FLAME2020/generic_model.pkl` | FLAME 2020 |
| `FLAME_LM468_EMBEDDING` | `$MODEL_DIR/flame_lm468_embedding.npz` | 468 点重心嵌入，缺了跑不了阶段四 |
| `DECA_CKPT` | `$MODEL_DIR/deca_model.tar` | DECA 权重 |
| `DECA_DIR` / `GS_DIR` | `../DECA` / `../gaussian-splatting` | 官方仓 |
| `PID` | `0` | 人物编号（产物名 `avatar_p0.ply` 是**一位**，mask 是 `p00` 两位）|
| `COEFF_ITERS` / `COEFF_GAMMA` | `3000` / `0.9995` | 05 阶段 4.3 精修步数与 LR 衰减（旧值 300/0.95 严重欠训练）|
| `N_BOUND` / `N_FREE` | `50000` / `20000` | 06 绑定 / 自由高斯采样数 |
| `SCALE_FACTOR` | `1.0` | 06 初始 scale 缩放（实验过 0.5，无增益）|
| `SH_DEGREE` | `3` | 08 SH 阶数（**官方 CUDA 内核只支持 ≤3**，设 4 是空操作）|
| `HEAD_REGION` / `HEAD_OUT_W` | `0` / `0.0` | 08 head 监督区收窄开关（实验为负结果，默认关）|
| `HEAD_SCALE_MAX` / `SCENE_DROP_RATIO` | `0.02` / `0.85` | 08e head 尺度门控 / scene 反向投票阈值 |
| `MIN_SHARPNESS` | `7.0`（09）/ `0`（01d） | 人脸裁剪的拉普拉斯方差闸门，低于则跳过增强 |
| `FACE_PADDING` / `UPSCALE` | `0.35`（09）/ `0.25`（01d）；`2` | 人脸框外扩比例 / HYPIR 上采样倍数 |
| `ENHANCE_ENV` | `vggt_human` | HYPIR 依赖所在 conda 环境（flame_human 环境没有）|
| `SKIP_RENDER` | `0` | 09 复用已有渲染（调增强参数时用）|

## 目录布局

```
<code-dir>/
├── media_code/flame_human/        # 本目录（编排脚本，不复制官方代码）
├── DECA/                          # 官方（00/00a clone）
├── gaussian-splatting/            # 官方（rasterization / simple_knn 子模块）
└── flame_human_results/           # 输出
    ├── 01c_sam3_person_masks/     # person mask（无上游产物时）
    ├── 01d_enhanced_faces/        # 实验：HYPIR 增强监督图（无增益）
    ├── 02_faces/                  # 阶段一：bbox + 5 点
    ├── 03_match/                  # 阶段二：face↔person ID 映射
    ├── 04_recon/                  # 阶段三：crop / DECA 系数 / 468 点
    ├── 05_align/                  # 阶段四：SRT + id + exp
    ├── 06_avatar_gs/              # 阶段五：AvatarGaussian 初始 PLY + 绑定
    ├── 07_body_gs/                # 阶段七：BodyGaussian 初始 PLY
    ├── 07_body_gs_src/            # 阶段七a：body_gs_p0.ply + scene_gs.ply
    ├── 08_train/                  # 阶段六/八：head checkpoint 与日志
    ├── 08d_finetune_scene/        # 阶段八d：scene finetune 产物
    ├── 08e_pruned/                # 阶段八e：三分支剪枝产物（后续默认用这版）
    └── 09_render/                 # 阶段九：raw/（原始渲染）+ images/（增强后）
```
