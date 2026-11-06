# P1 · 前馈 init + 短精调：能不能用 2~4 分钟换掉 20 分钟

> 🔴 **前馈 init 两组（②③④）已作废（2026-11-06）**：初值 PLY 漏乘 DA3 的度量
> `scale_factor≈2.0`，点云只有真实尺寸的 0.55，而相机是米制 → 5k 步挪不动 0.9 m 的错位，
> 「严重重影」的主因是尺度错位，**不是**下文归因的"多壳拓扑"。见 `ERRATA.md` E-01。
> ⑤⑥（COLMAP/深度反投影 init）不受影响。修复后 DA3 直出 = 18.97 dB，
> 与 ③ 的 20.14 dB 只差 1.2 dB → **"前馈初值值多少钱"这个问题要重测**。

**前置**：P0（见 `P0_EXPERIMENTS.md`）三项"加容量"的干预全部判 B，并定位到真正的瓶颈是
**跨视角几何不一致**——同一份 DA3 输出在参考视角能到 20.8 dB，离开参考视角单调衰减到 8~10 dB。
P1 就是针对这个瓶颈的手段：用光度一致性把跨视角误差压下去。

## 0. 目标与判据（跑之前写定）

**问题**：把 DA3 的前馈高斯当作 3DGS 的初值，只跑 3k~5k 步，能否达到
「COLMAP 稀疏点云 init + 30k 步」的质量，同时把总耗时压到 2~4 分钟？

**判据**（预先定义，避免事后调标准）：

| 结论 | 条件 |
|---|---|
| **A（成立）** | 前馈 init + ≤5k 步的 PSNR 与 30k 基线差 ≤1 dB，且总耗时 ≤4 min |
| **B（部分成立）** | 与 30k 差 1~3 dB，或耗时 4~8 min |
| **C（不成立）** | 与 30k 差 >3 dB，或比同迭代次数的 COLMAP init 还差 |

**必须有的对照**：`COLMAP init + 同样 5k 步`。否则"前馈 init 有效"无法与
"少跑 25k 步本来就够"区分——这是最容易犯的归因错误。

## 1. 实施时躲不开的三个约定（错一个就静默出坏结果）

写这一节是因为三条全都踩过，且**没有任何报错**。

### 1.1 3DGS 的 `--start_checkpoint` 只吃 `.pth`，不吃 PLY

原版 `train.py` 有两条带初值的路，但只有一条能用：

- `Scene.__init__` 里 `if self.loaded_iter:` → `gaussians.load_ply(...)`，
  但 `train.py` 调 `Scene(dataset, gaussians)` 时**没有传 `load_iteration`**，这条分支永远不走。
- `--start_checkpoint` → `torch.load()` 拿到 `(model_args, first_iter)` → `restore()`，
  其中 `model_args` 是 `capture()` 的 12 元组（含 optimizer state_dict）。

所以 `98b_ply_to_3dgs_ckpt.py` 直接造 `.pth`，**不改第三方仓一行代码**。

### 1.2 参数空间必须原样搬运，且**必须包成 `nn.Parameter`**

| 量 | DA3 导出 | 3DGS 内部 | 处理 |
|---|---|---|---|
| scale | `scales.log()` | `_scaling` 也是 log | 原样 |
| opacity | `inverse_sigmoid` | `_opacity` 也是 logit | 原样 |
| SH | `sh_degree=2`（9 系数） | 写死 `sh_degree=3`（15 rest） | 前 8 阶填入、后 7 阶补 0 |

**最隐蔽的一条**：`restore()` 是直接赋值 `self._xyz = model_args[1]`，不做任何包装。
官方 checkpoint 存的是 `nn.Parameter`；如果用 `torch.from_numpy()` 存普通 tensor
（`requires_grad=False`），autograd 根本不追踪 → 梯度永远 None → Adam 一步也不更新。

症状极具欺骗性：训练正常打印 `Training complete`、耗时正常、loss 在 0.30 附近"波动"
（其实是不同随机相机的差异），但**存出的 PLY 与初值逐字节相同**——
实测四个 iteration（0/200/1000）的 PLY md5 全是 `9aebd9de...`。
定位手法：`md5sum` 比对 PLY + 读 ckpt 看 `requires_grad`。

### 1.3 `optimizer.load_state_dict` 会**整体重建** param_groups

`restore()` 里 `training_setup()` 刚设好的 per-group lr，会被 `load_state_dict(opt_dict)`
用 ckpt 里的值覆盖。两个后果：

- ckpt 里 lr 填 0 → 除 xyz（每步被 `update_learning_rate` 重设）之外的
  `f_dc / f_rest / opacity / scaling / rotation` **全部不更新**；
- ckpt 里的 group 缺 `"name"` 键 → 3DGS 的 `if param_group["name"] == "xyz"` 直接 `KeyError`。

所以 `build_opt_dict()` 必须按 3DGS 默认超参填真实 lr，并补回 `name`。

### 1.4 内参与 PLY 必须同一次推理、同一个分辨率口径

DA3 的 `export_to_colmap` 会**原地**把 `prediction.intrinsics` 按原图分辨率放大
（fx 372 → 1417，即 1439/378 = 3.81 倍）。COLMAP 场景配原图是对的，但渲染用的是
处理分辨率（378×504）。若在导出**之后**才存 npz，拿去渲染就会 FOV 收窄、画面只剩局部，
**看起来像几何错位**（我第一次就是这样白跑了一轮 200 步）。

→ `98_da3_ff_init.py` 的顺序固定为：**前向 → 存 npz → 再导出 gs_ply/colmap**。

## 2. 结果

全部 32 视角、与输入帧比（自一致性口径，非 NVS holdout）。
PSNR 统一在 378×504 下算（各方法渲染分辨率不同，用 INTER_AREA 归一到同一尺寸）。

| # | 组 | init | 点数 | iters | densify | PSNR | SSIM | 锐度 | v0 | 远端 | 训练耗时 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ① | DA3 直出（初值） | 前馈高斯 | 484 万 | 0 | — | 11.38 | 0.268 | 149 | 20.57 | — | 0 s |
| ② | 前馈 init | 前馈高斯 | 415 万 | 2 000 | 关 | 19.67 | 0.700 | 464 | 20.26 | 19.91 | 209 s |
| ③ | **前馈 init（A1）** | 前馈高斯 | 415 万 | 5 000 | 关 | 20.14 | 0.728 | 626 | 20.67 | 20.08 | 466 s |
| ④ | **前馈 init 剪枝（A2）** | 前馈高斯 | 57.4 万 | 5 000 | 开 | 21.72 | 0.763 | 573 | 22.45 | 21.86 | 169 s |
| ⑤ | **COLMAP 稀疏（B1）** | DA3 点云 | 1.86 万 | 5 000 | 开 | **22.23** | 0.738 | 465 | 23.86 | 23.10 | **123 s** |
| ⑥ | COLMAP 稀疏（30k 基线） | DA3 点云 | 1.86 万 | 30 000 | 开 | **24.33** | 0.808 | 881 | 23.18 | 24.83 | 1253 s |

（① 的 32 视角数字取自 P0 的 `p01_ref0_metrics.json` 的 r504 组，同口径；
⑤⑥ 的 init 点云同样是 DA3 深度反投影——365 万点经体素降采样到 1.86 万，
所以"COLMAP init"并不是传统意义上的 COLMAP 稀疏重建，而是 **DA3 几何的稀疏形式**。）

图：`ppt_assets/fig9_p1_tradeoff.png`（耗时↔质量）、`ppt_assets/fig10_p1_renders.png`（渲染对照）。
原始数字：`ppt_assets/p1_metrics.json`（逐视角 PSNR）、`p1_init_summary.json`、`p1_ckpt_summary.json`。

## 3. 判定与归因

**结论：B（部分成立）。** 最快可用配置是「DA3 稀疏点云 + 5k 步」——
训练 123 s、端到端约 2.6 min，22.23 dB；对 30k 基线（24.33 dB / 21.5 min）**少 2.10 dB，省 8 倍时间**。

但真正的发现是另一件事：**加速来自"少跑步数"，不是来自"前馈初值"。**

三条 init 路线的 PSNR 是**单调反比于初始点数**：

```
1.86 万点 → 22.23 dB (123 s)   ← 让 3DGS 自己 densify
57.4 万点 → 21.72 dB (169 s)   ← 剪到只剩高不透明度点，仍可 densify
415 万点  → 20.14 dB (466 s)   ← 全量灌入，反而不许 densify
```

**为什么稠密 init 反而更差**（三个原因，前两个有实测支撑）：

1. **opacity 空间不匹配**。DA3 输出的 opacity 中位数只有 **0.036**（分位：25% = 0.0127、
   50% = 0.0356、90% = 0.107），也就是绝大多数点处于"半透明"状态。
   3DGS 的 `densify_and_prune` 用的是 `opacity < 0.005` 才剪，于是这些低不透明度点
   既不会被剪、也不会被合并，只能**大量叠加成一层"糊"**——这正是 ③ 的画面比 ④ 更软的
   直接原因（锐度 626 vs 573 但 SSIM 更低，说明高频是"伪锐"）。
2. **scale 量级偏小**。DA3 的 scale 中位数 0.8 mm，而场景 extent 2.09 m。
   这些过小的椭球无法覆盖真实表面，必须靠 clone/split 长回去——但一旦关掉 densify
   （因为 415 万点开着会直接吃满 24 GB），它们就没有机会长。
3. **单步成本与信息量不成比例**。415 万点时 11 it/s，1.86 万点（densify 后约 25 万）时
   40 it/s。同样是 5k 步，前者付 4 倍时间，却因为每步只采样 1 个视角而拿到同样的梯度样本数。

所以 **DA3 的前馈几何是有用的**（①→⑤ 涨了 10.85 dB，且 5k 步就能到 30k 基线的 91%），
但它的正确接口是「**稀疏点云 + 让 3DGS 自己决定密度**」，而不是「把前馈高斯整套搬过来」。

## 4. 对后续计划的影响

- **P2（双头解耦）、P3（2DGS 原语）继续后置**：P1 已经说明，在 504 输入 + 32 视角
  这个条件下，瓶颈不在"外观头"或"原语"，而在**点的密度分配**。
- **马上可用的配置**：`DA3 点云（1~2 万点）+ 5k 步 + densify`，2.6 min 拿到 22.2 dB。
  如果要更高保真，再退化到 30k（21.5 min / 24.3 dB），中间没有更好的折中点——
  这条曲线在 5k 之后非常平（P1 的 2k→5k 只涨 0.47 dB，5k→30k 涨 4.19 dB）。
- **一个值得单独立项的疑问**：④ 的锐度（573）已经超过 ⑤（465），SSIM 也更高（0.763 vs 0.738），
  只是 PSNR 低 0.51 dB。说明剪枝后的前馈 init 在**细节**上不差，差的是"大面积色块的准确度"
  （PSNR 对大误差敏感）。如果目标是视觉保真而非 PSNR，④ 的性价比可能更高。这条没测够，先记为待验证。

---

## 附：复现命令

```bash
# 0) 一次推理，同源出 gs_ply + colmap + cameras.npz（钉死参考视角 = 帧 0）
DA3_MODEL_DIR=/mnt/d/wheel/da3_human_ms \
IMAGES_DIR=~/output/da3_results/_captures/佳佳抱枕/images \
OUT_DIR=~/output/da3_gs_results/佳佳抱枕_p1 \
PROCESS_RES=504 REF_STRATEGY=first \
  ~/miniconda3/envs/da3_human/bin/python compare_4methods/98_da3_ff_init.py

# 1) 组装 COLMAP 场景（3DGS 要 images/ + sparse/0/）
DA3_INFER_DIR=~/output/da3_gs_results/佳佳抱枕_p1 \
SCENE_DIR=~/output/da3_gs_results/佳佳抱枕_p1/colmap_scene \
IMAGES_DIR=~/output/da3_results/_captures/佳佳抱枕/images MAX_INIT_POINTS=20000 \
  ~/miniconda3/envs/da3_human/bin/python da3_human/build_da3_colmap_scene.py

# 2) PLY → 3DGS checkpoint
PLY=.../gs_ply/0000.ply NPZ=.../cameras.npz OUT_DIR=.../ffinit_ckpt \
  ~/miniconda3/envs/querysplat/bin/python compare_4methods/98b_ply_to_3dgs_ckpt.py

# 3) 短精调（前馈 init）
SCENE_DIR=.../colmap_scene OUT_DIR=.../final_A1_ff5k INIT_CKPT=.../chkpnt_ffinit.pth \
ITERATIONS=5000 DENSIFY=0 RENDER=0 \
  bash compare_4methods/03b_train_short.sh
```
