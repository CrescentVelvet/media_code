# EXPERIMENTS.md — vggt_human 实验结论与踩坑记录

> 本文件从 AGENTS.md 第 12 节迁出（2026-09-07）：AGENTS.md 只保留全仓统一规范，
> 项目专属的实验结论 / 消融数据 / 踩坑细节归档到各项目目录下。
> 新实验结论继续追加到本文件，按日期倒序排列。

## 人脸 finetune 消融结论（2026-09-04，3090 实测）

对照实验（同等训练量 20k 步 @ lr_scale=0.2，唯一差异 face_weight 0 vs 1.0）证明：
**人脸监督（masked-L1 → HYPIR 增强图）对人脸区锐度无可测量贡献**（+4.04% vs +3.98%，
逐帧差在噪声内）。此前 v1~v6 的全部人脸区提升来自"30k 基线欠收敛 + 更高 lr 继续训练"。

- **推荐配方**：`06d_continue_train.sh`（ITERATION=30000, EXTRA_ITERS=20000,
  LR_SCALE=0.2, FACE_WEIGHT=0）。训练视角人脸区 Laplacian +4~7%，全图 PSNR
  27.32→27.44+，LPIPS 同步下降，无过平滑代价
- **权重已饱和**：face_weight 1.0→2.0 结果逐像素持平（瓶颈是 lr_scale 不是权重）
- **lr_scale 是瓶颈**：0.1→0.2 把增益从 +2.0% 解锁到 +2.7%/10k 步
- **增益曲线**：~+1.3%/万步递减到 ~+0.8%/万步（80k 处未完全收敛）
- **新视角注意**：常规（人物居中）新视角明显更清晰且无伪影；极端离轴暗部视角
  纤维状条纹伪影增多，此类场景用较少续训步数（50k~60k）折中
- 若要超越 GT 收敛的人脸细节，masked-L1 机制无效，需换机制（对抗损失等），未验证

### 06e：增强近景整视角注入（2026-09-04，有效，与 masked-L1 机制相反）

把 06c 的 512 增强近景作为**完整训练视角**（真实相机位姿 + COLMAP 注入）喂回续训，
与 masked-L1（增强图仅作人脸区 GT）机制不同，**实测有效**（ms 数据集，4 人 × 18 近景）：

- 归因对照（同 20k 步 @ lr_scale=0.2）：
  - 06f 纯续训无注入：近景 LPIPS 持平（0.3048→0.3049 等）、锐度 +3~6%（噪声级）
  - **06e 注入**：近景 LPIPS 全降（p00 0.305→0.260、p03 0.202→0.137），
    锐度 +28%~+324%；训练视角人脸区 Laplacian +45% 且 LPIPS vs 原图不升（无退化）
- 结论：masked-L1 无效 ≠ 增强图无用；**整视角注入（整图参与 loss）才是正确用法**
- 链路：`inject_closeup_cameras.py`（pycolmap 4.2 `add_camera_with_trivial_rig` +
  `add_image_with_trivial_frame`，pose json R/T 即 COLMAP w2c）→ 场景 06e_source_closeup
  （222 views）→ 06d 配方续训
- 坑：start_ply 的 exposure.json 只有原图条目，注入图名在 Scene.save()/testing 会
  KeyError —— trainer 已补恒等 exposure fallback（30k exposure 本就收敛为恒等阵）
- p03（此前疑 p00 重复轨迹）实为真实第 4 人（红衣），近景提升最大，无需排除
- **round2 迭代注入（06g）**：06e@50k 重渲近景→再增强→再注入（同 04b 起点干净归因），
  统一 round2 GT 评估：06g vs 06e 近景 LPIPS 再降 10~23%、锐度再涨 +22~71%、
  PSNR +1.2~1.5dB；训练视角人脸区 Laplacian 11.4→13.3，LPIPS vs 原图持平（无幻觉回声）。
  **两轮未饱和，增益递减**（round1 训练视角 +45% vs round2 +17%）；第三轮预期收益有限，
  按需取舍。脚本 `06e_closeup_inject.sh`（注入+续训一条龙）

### SSIM 与人脸 L1 目标不一致（2026-09-05，已修）

`train_face_finetune.py` 的 masked-L1 路径里，L1 在人脸区内推向 HYPIR 增强图，
SSIM 项却一直全画幅对着原图算 —— 两项在人脸区内的更新方向近乎**正交**。
合成测试（模糊渲染 / 原图 / 增强图，人脸区中央 mask）测「更新方向」与
「补细节方向」的余弦：

| 项 | 余弦 |
|---|---|
| L1 人脸项 | +0.81 |
| SSIM 旧（对原图） | +0.18 |
| SSIM 新（composite） | +0.94 |
| 总 loss 旧 → 新 | +0.57 → +0.95 |

即 λ_dssim=0.2 的 SSIM 项把总方向从 0.81 **稀释**到 0.57（不是反向，是正交稀释），
masked-L1 的效果被吃掉一大半。修法：人脸区内把 SSIM 的比对目标也换成与 L1 同一张
composite（`gt + M·w·(enh − gt)`），区外仍是原图 → 无 mask 的帧与官方完全等价。
新增 `--face_ssim_mode composite|off`（默认 composite；off 仅供 A/B 复现旧行为）。

注意：这条路径目前不是主力（主力配方 FACE_WEIGHT=0 + 整视角注入），故 06e/06g
结论不受影响。修完后 masked-L1 能否起死回生**未重测**。

### HYPIR 推理提速（2026-09-05，3090 / 512² 实测）

优化点 1 的 TorchScript 被否决后定的替代路线，三项均已落地在 `face_enhance.py`
（`SPEED_MERGE_LORA` / `SPEED_CACHE_TEXT` / `SPEED_COMPILE`，前两项默认开）：

| 配置（累积） | ms/张 | vs base | 增量 |
|---|---|---|---|
| base | 170 | 1.00x | — |
| +cache_text | 161 | 1.05x | 1.05x |
| +merge_lora | 139 | 1.22x | 1.16x |
| +compile | 130 | 1.31x | 1.09x |

- **merge_lora**（默认开）：SD2.1 UNet 共 257 层 LoRA。peft 走的是
  `inject_adapter_in_model`（非 LoraModel 包装），没有 `merge_and_unload()`，
  需逐层调 `LoraLayer.merge()`。peft 的 forward 有 `elif self.merged: 走 base_layer`
  分支，**不会重复叠加 delta**，安全。数值漂移 0.35/255（≈0.14%），可忽略
- **cache_text**（默认开）：`BaseEnhancer.enhance()` 每图都重跑 tokenizer +
  text_encoder，而本流程 prompt 恒为 "" → monkey patch `prepare_inputs` 缓存
  `self.inputs`。text_encoder 确定性且不消耗 RNG → 输出逐位不变
- **compile**：**默认 auto（按批量门控，回本线 2400 张）**。增量只有 142→130ms
  （~12ms/张），一次性编译 ~28s → 约 2300 张才回本，小批量编译是净亏
  - **必踩的坑**：diffusers 的 `Attention.forward` 里有
    `inspect.signature(self.processor.__call__)`，dynamo 对每个 attention block
    的 processor 对象 id 加 guard，UNet 十几个 block 各要一份编译；默认
    `cache_size_limit=8` 撑爆后 dynamo 放弃编译、退回 eager 并保留 dynamo 开销，
    **实测 175ms 变成 1687ms（慢 10 倍）**。必须在 compile 前把
    `torch._dynamo.config.cache_size_limit` 抬到 64（已写进 `compile_unet()`）
  - `mode=default` 不如 `reduce-overhead`（编译 39s vs 28s，稳态 132 vs 129ms）
- **A/B 对比的坑**：`vae.encode().latent_dist.sample()` 每次前向都从固定 seed 的
  随机流取新样本 → 输出对「第几次调用」敏感，相邻两次前向差 mean|Δ|≈0.011。
  对比提速前后**必须对齐调用序号**（跑两次完整脚本比同名图片），否则量到假差异；
  同序号下逐位可复现（两次独立运行 mean|Δ| = 0.000/255）
- benchmark：`.tmp_diag/bench_hypir_speed.py`（`--only_compile` 只看编译收益与回本张数）

### 06d 脚本注意

- `train_face_finetune.py` 的 `FaceData` 只扫 `face_images_dir` **直接子文件**，
  目录参数必须指到图片平铺层（如 `06c_merged_face_images/images`，不是其父目录）
- `render_novel.py` 只读 txt 格式 COLMAP 模型；BA 场景（03b）sparse/0 只有 bin，
  已加 pycolmap 自动补 txt 的 fallback
