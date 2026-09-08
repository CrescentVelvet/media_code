# 权重下载清单（flame_human）

> 运行 `bash 01_download_models.sh` 会按下面清单逐个检查；缺失的会打印下载指引。
> 本文件记录**为什么需要**与**从哪来**，URL 变动时改这里。

## 1. FLAME 2020 — `generic_model.pkl`

| 项 | 值 |
|---|---|
| 落盘 | `$MODEL_DIR/FLAME2020/generic_model.pkl` |
| 来源 | https://flame.is.tue.mpg.de/download.php |
| 授权 | 需注册并接受 FLAME 许可证，**不能脚本自动下载** |
| 用途 | 参数化头模型本体：5023 顶点 / 300 shape / 100 expr / 5 joints |

注册后下载 "FLAME 2020" 里的 `generic_model.pkl`（约 30 MB）。
**不要用 `FLAME_neutral_*.pkl`**——那个不带 expression 空间，阶段四的 `local_exp` 无从谈起。

## 2. DECA — `deca_model.tar`

| 项 | 值 |
|---|---|
| 落盘 | `$MODEL_DIR/deca_model.tar` |
| 来源 | DECA 官方 README 里的 Google Drive 链接（见下方镜像） |
| 授权 | 需先有 FLAME 2020（DECA 依赖 FLAME）|
| 用途 | 阶段三给 shape/expr/pose **初值**，仅此而已 |

官方仓 `https://github.com/yfeng95/DECA` 的 README 给了下载链接。
社区镜像（若官方链接失效可试）：HuggingFace 上有若干 `deca_model.tar` 转存，
但版本不一，优先用官方。

> ⚠️ DECA **只取初值，不取 landmark**（DECA 只输出 68 点，解不了 100 维 exp，
> 见设计文档 §4.0 的可解性表）。468 点观测由 MediaPipe 提供。

## 3. MediaPipe 468 → FLAME 顶点重心嵌入

**不需要下载**，由 `01b_build_lm468_embedding.sh` 现场生成：

```
MediaPipe canonical_face_model.obj（468 个 3D 点，随 mediapipe 包自带）
        │  Procrustes 对齐到 FLAME mean face
        ▼
每个点找 FLAME 表面最近点 → (face_id, bary) → flame_lm468_embedding.npz
```

自己生成比找外部嵌入文件可靠：不依赖第三方转存的 NPZ，换 FLAME 版本时重跑即可。
（做法与 `vggt_human/prepare_3dmm_template.py` 建 `lm468_idx`/`lm468_bary` 一致。）

## 4. HYPIR（阶段五/八人脸增强）

复用 `vggt_human` 已下载的那一份，路径由 `HYPIR_MODEL_DIR` 指向
`../../model/HYPIR`。缺的话先跑 `vggt_human/00a_setup_env.sh`。
