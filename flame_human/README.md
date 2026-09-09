# flame_human — FLAME + 表情驱动的多人 3DGS 人脸链路

用 **FLAME 2020**（5023 顶点 / 300 shape / 100 expr）替代自研 3DMM，配 **DECA** 给 shape/expr 初值、
**MediaPipe 468 点**做 2D 观测，走「三阶段 3DMM 对齐 → 重心绑定高斯 → 表情驱动训练 → 后处理增强」
的八阶段链路。SfM/COLMAP 与 person mask 复用上游 `vggt_human` 的产出，本目录不重做。

与 `vggt_human` 的核心差异：那边是「468 点 + 中性脸 + 每帧刚体、无表情」；这边表情系数
贯穿高斯初始化、训练、后处理三个阶段（`local_exp` 是可训练参数，渲染 loss 直接 refine）。

**设计文档（决策 + 论证 + 遗留项）见
[../vggt_human/DESIGN_face_pipeline.md](../vggt_human/DESIGN_face_pipeline.md)。**

## 阶段 → 脚本

| 阶段 | 脚本 | 说明 |
|---|---|---|
| — 权重检查 | `01_download_models.sh` | FLAME / DECA / HYPIR 就位检查 |
| — 468 点嵌入 | `01b_build_lm468_embedding.sh` | MediaPipe 468 → FLAME 重心嵌入 |
| — person mask | `01c_gen_person_masks.sh` | SAM3 person mask（复用 vggt_human worker）|
| 一 人脸检测 | `02_detect_faces.sh` | face bbox + 5 点（5 点仅副产物，下游不用）|
| 二 人脸匹配 | `03_match_faces.sh` | 三层 ID 关联：point-in-mask → IoM → 匈牙利 + 时序 |
| 三 人脸重建 | `04_recon_faces.sh` | 各向同性 crop → 512²；DECA 出初值；MediaPipe 出 468 点 |
| 四 3DMM 对齐 | `05_align_3dmm.sh` | 4.1 global / 4.2 local / 4.3 coeff 三阶段 |
| 五 Avatar 初始化 | `06_init_avatar_gs.sh` | 表面采样保留 `(face_id, bary)`，硬绑定 + 自由高斯 |
| 七 Body 初始化 | `07_init_body_gs.sh` | point-to-triangle 切分 + 脖子下边界补缝 |
| 六/八 训练 | `08_train.sh` | `local_exp` 联合优化；epoch 48 触发增强 |
| WSL 搬运 | `09_move_output.sh` | Linux fs → `/mnt/d/` |

## 首次准备

```bash
cd <code-dir>/media_code
cp proxy.env.example proxy.env      # 确认 http_proxy / https_proxy 已取消注释

# WSL 本机
bash flame_human/00a_setup_env.sh
# 服务器
bash flame_human/00_setup_env.sh

bash flame_human/01_download_models.sh   # FLAME2020.pkl / DECA 权重 / lm468 嵌入
```

权重目录布局：

```
$MODEL_DIR/                       # 默认 ../../model/flame_human
├── FLAME2020/generic_model.pkl   # 官网注册下载，见 download_urls.md
├── flame_lm468_embedding.npz     # MediaPipe 468 → FLAME 顶点重心嵌入（必须）
└── deca_model.tar                # DECA 预训练权重
```

## 常用命令

> 铁律：每条命令显式写出输入路径与输出路径，不要全靠脚本默认值。

```bash
# 一键（阶段一 → 阶段七）
GPU=0 \
  SOURCE_DIR=../vggt_human_results/03_source \
  PERSON_MASKS_DIR=../vggt_human_results/03_sam3_person_masks \
  RESULTS_DIR=../flame_human_results \
  bash flame_human/run_all.sh

# 分步
GPU=0 SOURCE_DIR=../vggt_human_results/03_source \
  RESULTS_DIR=../flame_human_results \
  bash flame_human/02_detect_faces.sh
# ... 03 / 04 / 05 / 06 / 07 / 08 依次
```

## Config (env vars)

| var | default | note |
|---|---|---|
| `CONDA_ENV` | `flame_human` | conda 环境名 |
| `GPU` | 未设 | 设了才 export `CUDA_VISIBLE_DEVICES` |
| `UPSTREAM_DIR` | `../vggt_human_results` | 上游 SfM 与 mask 的根 |
| `SOURCE_DIR` | `$UPSTREAM_DIR/03_source` | COLMAP 场景（images + sparse）|
| `PERSON_MASKS_DIR` | `$UPSTREAM_DIR/03_sam3_person_masks` | SegTrack/SAM3 person mask；没有上游产物时先跑 01c（默认输出 `$RESULTS_DIR/01c_sam3_person_masks`）|
| `RESULTS_DIR` | `../flame_human_results` | 本链路输出 |
| `MODEL_DIR` | `../../model/flame_human` | 权重根 |
| `FLAME_MODEL` | `$MODEL_DIR/FLAME2020/generic_model.pkl` | FLAME 2020 |
| `FLAME_LM468_EMBEDDING` | `$MODEL_DIR/flame_lm468_embedding.npz` | 468 点重心嵌入，缺了跑不了阶段四 |
| `DECA_CKPT` | `$MODEL_DIR/deca_model.tar` | DECA 权重 |
| `DECA_DIR` / `GS_DIR` | `../DECA` / `../gaussian-splatting` | 官方仓 |

## 目录布局

```
<code-dir>/
├── media_code/flame_human/        # 本目录（编排脚本，不复制官方代码）
├── DECA/                          # 官方（00/00a clone）
├── gaussian-splatting/            # 官方（rasterization / simple_knn 子模块）
└── flame_human_results/           # 输出
    ├── 02_faces/                  # 阶段一：bbox + 5 点
    ├── 03_match/                  # 阶段二：face↔person ID 映射
    ├── 04_recon/                  # 阶段三：crop / DECA 系数 / 468 点
    ├── 05_align/                  # 阶段四：SRT + id + exp
    ├── 06_avatar_gs/              # 阶段五：AvatarGaussian 初始 PLY + 绑定
    ├── 07_body_gs/                # 阶段七：BodyGaussian 初始 PLY
    └── 08_train/                  # 阶段六/八：checkpoint 与日志
```
