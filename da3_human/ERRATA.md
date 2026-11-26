# da3_human ERRATA — DA3 复现错误存档与防再犯清单

> 2026-11-09 峰哥严令：**跑 DA3 的推理/训练/评估前必读本文**；本清单只适用于 da3_human
> （DA3 官方高斯头复现与微调），其它算法项目不适用。实验数值详表见 EXPERIMENTS.md 同日节。

## ⛔ 事故清单（全部有实测代价）

| # | 错误 | 实测代价 | 教训 |
|---|---|---|---|
| 1 | 把 125 视角全喂 DA3，还归因"模型慢" | 前向 133.9s vs 32 视角 **6.6s**（视角 3.9×，耗时 20×） | DA3 是稀疏视角模型，输入 32 就够；attention 开销随视角数超线性增长 |
| 2 | 拿官方**默认导出**当 baseline | 白丢 **2.51dB**（裁边 + 剪最远 10% 深度 + 只存 DC 三件套） | 正式口径 = 不裁剪 + 全 SH |
| 3 | `07_eval.load_gt` 假设 `sorted(images)[vi]` = 第 vi 视角 | 抽帧后 GT 整体错位，PSNR 假摔到 10 | 抽帧必须存 `view_indices`，GT 按索引取 |
| 4 | NVS 评估重复乘 `sf32/sf125` | **−2.29dB** | PLY 与 extrinsics **各自已经是米制**，禁止二次缩放；变换前先验证单位 |
| 5 | 导出 28.3s 归因"953MB 大文件写盘" | 实际官方 `list(map(tuple,...))` Python 开销；numpy `tofile` 后 **1.8s** | 耗时先分段 profile，不许归因"文件大/磁盘慢" |
| 6 | 672p 掉 2.87dB 归因"分辨率提升没用" | 实际**跨分辨率 FOV 不同**（504p fx/W=0.9780 vs 672p 0.9566）混用了相机，假跌；自洽口径 **+0.83dB** | 不同 `process_res` 的相机与高斯严禁混用 |
| 7 | 快速导出验证拿**两次独立推理**的 PLY 互比 | 浮点抖动（sf 2.1204 vs 2.1193）造成 3cm 假差异，排查方向被带偏 | 单变量铁律：同一次推理、同一份 gaussians 派生两条路径互比（41 列 max\|Δ\|=0） |
| 8 | 拿 cameras.json 的 `rotation` 直接当 w2c 用 | 该 rotation 是 **c2w**（渲染判决：c2w 约定 25~28 dB vs w2c 5~8 dB）；D1 诊断 FT_30k 本脚本重渲值 5~28 dB 即此坑 | 用 cameras.json 前先渲染判决约定；该数据包 transforms.json 约定也与 som_human 标准包不同，勿套 `w2c=inv(M@flip)` |
| 9 | Umeyama/Kabsch 行映射 `Q=U@Vt` 与列向量混用 | 对齐残差假大 0.8~1.6 m，还引出"预测世界与 GT 互为镜像"的假结论（实际 det=+1） | 行向量映射 `dst≈s·src@Q+t`；套列向量世界变换用 `Q^T`，w2c' = R·Q/s |
| 10 | DA3 `cameras.npz`：extrinsics 是 **w2c 直接用**（勿求逆，求逆=静默出貌似合理的错图）；intrinsics/depth 已是 process_res 口径（504p 时 378×504，**勿再缩放**）；平移 ÷ scale_factor 回到高斯尺度 | 渲染人物位置全错但画面"看起来还行"，PSNR 掉 5~10 dB 不报警 | 三查：w2c/c2w 方向、K 的分辨率口径、sf 是否只除一次（对齐 29m 与 29_d1_diag 的写法） |
| 11 | 输入相机后把 API 返回外参当原始预测相机，再拿去对齐高斯；或不转高斯只换相机 | 海思 case16 试验 PLY 41 字段、有限值全通过，源视角却整体放大错位 | API 在 `_align_to_input_extrinsics_intrinsics` 覆盖外参；高斯头仍按预测相机反投影。先保留对齐前外参，再用同一 RANSAC Sim(3) 变换 means/scales/rotations/SH，源视角回放才可验收 |
| 12 | 把官方 `mat_to_quat` / `quat_to_mat` 的 `xyzw` 输出误按 `wxyz` 解读 | 海思输入相机对齐试验中高斯朝向错误 | 官方 `gs_adapter` 的历史注释仅指 raw 输入；两个转换函数输出为 **xyzw**，最终 Gaussians/PLY 沿用其分量顺序。PLY 的 `f_rest` 是 channel-major，但官方 `rotate_sh` 需要 `(N,3,K)`；gsplat 需要 `(N,K,3)`，两者转换必须显式 transpose |
| 13 | 官方 `gs_adapter.py:89-101` 在 `gt_extrinsics` 分支只对齐尺度，只乘 c2w 平移与深度，漏掉 Umeyama 旋转和平移 | 若直接采用该分支结果，高斯仍在预测世界而输出相机已换到输入世界，产生整体错位；纯缩放 clamp 不能修复 | 相机版必须从对齐前预测外参显式求完整 Sim(3)，同步 means/scales/rotations/SH。`R_c2w'=R R_c2w`、`C'=s R C+t`，**不是给旋转矩阵乘 s**；数值检查是 `isfinite(s) & (s>0)`，不能写 `isfinite(s>0)`。海思36号后处理已覆盖；官方 adapter 的异常回退仍是另一待修项 |

### 2026-11-26 新 glTF 数据的焦距、原生相机与多头问题

- 同一 JSON 的静态 `intrinsics.K` 对应 95° 视场，动画 `camera_path.yfov` 却对应约 54°；海思新增人像原始 RGB 在 RGB-only 下的自估焦距约 `fx=678, fy=706 @700`，接近后者。动画 yfov 条件的预测焦距约665–671/699–703，比顶层 K 条件的306–313/333–341合理；但**焦距一致不等于几何一致**。最终两例带位姿输入的原生相机回放仍多头/重影，撤回此前“动画焦距使人物接近 RGB-only 清晰度”的过度乐观描述。
- 原始 704×704 图片不等于网络处理尺寸。旧默认504导致每例2,540,160点；`process_res=704` 实际经 PATCH_SIZE=14 圆整至700，完整点数4,900,000。同事704 PLY为4,776,138/4,776,089点、SH DC only；不能将更多点数等同于更高分辨率或更好画质。
- `align_poses_umeyama(ext_ref, ext_est)` 返回 **est → ref** 的 R/t/s，点变换为 `X_ref=s X_est R^T+t`。已对 adapter 原生高斯直接使用 `align_poses_umeyama(input, adapter_pred)` 的单一 Sim(3)，**不再另乘 Nested sf 或除 adapter a**；sf/a 只在捕获的原始预测相机中进入拟合一次。旧 stage1/SfM 后处理保持独立，不能未经其各自回归而覆盖。
- 原生高斯不能直接配 API 最终输入相机。最终 `model_man` 的原生预测世界和输入世界均多头，RGB-only 原生世界仅软化而无多个人脸结构；所以本例多头主要已经在带位姿条件的前向预测中出现，不能归因于单一世界变换、PLY 导出或回放 K。它仍未区分 glTF 位姿/帧对应关系错误与 DA3 相机条件响应错误。
- 相机文件无 depth/image_shape 时必须显式提供 `CAMERA_BASE_SIZE=WxH`。700px K 不可未经缩放直接用于504px渲染；全部回放须保存 K 基准与相机世界说明，拒绝 NaN/Inf 进入渲染器。


- 官方代码版本 `41736238f5bced4debf3f2a12375d2466874866d`。`gs_adapter.py:89-101` 调用 `batch_align_poses_umeyama` 后丢弃 R/t，只缩放 c2w 平移和 `gs_depths`；`pose_align.py:39-45` 已知正确形式是旋转左乘 R、平移 `sRC+t`。因此“输入相机条件分支的 adapter 对齐不完整”成立，但“用 `einsum` 缩放整个旋转矩阵”不成立：`s` 属于点/中心/深度/高斯尺度，相机旋转仍必须保持 SO(3)。
- 海思36号入口在 `inference_preserving_raw_cameras()` 中捕获真正预测相机，再调用 `transform_gaussians()` 覆盖完整 R/t/s、旋转四元数与 SH，最后仅施加显示转正。它不是直接采用 adapter 的相机对齐结果；RGB-only 34号的 `gt_extrinsics=None`，也不走此分支。故本轮不重跑17例、不修改已有 PLY；现有170源视角回放仍是有效工程验收。
- 17例保存的 R/t/s 已复核：全部有限、尺度正、det(R)=1（浮点误差内），`pose_scale=0.23403496–5.24071031`。官方 `[1/3,3]` clamp 若直接搬到本路径会改变实际尺度，尤其 case14/17，不能照搬。
- `torch.isfinite(pose_scales > 0)` 先得到 bool，再判断有限，恒为 True；正确形式为 `torch.isfinite(pose_scales) & (pose_scales > 0)`。36号还检查 R/t 有限、R 为 SO(3)，非法结果拒绝导出，不能把失败静默写成错位 PLY。
- 官方 adapter 的 `except` 只令 `pose_scales=1` 且没有 R/t 对齐，不是“完整 Sim(3) 无变换”的充分修复。若之后直接依赖该分支，应返回 B×3×3 单位 R、B×3 零 t、B 个1的 s，校验三者后统一施加 `C'=sRC+t`，并先小规模 A/B。该修改与当前海思独立后处理路径无兼容性必要，官方仓本轮保持未修改。

## 防再犯清单（报数/下结论前必查）

- 【⛔ 取消裁剪（2026-11-11 峰哥决策）】**所有 DA3 高斯导出一律用不裁剪 + 全 SH**（08 max 档）；
  三视角下裁剪会把背景空洞率从 0.7% 放大到 34.8%（D1 实测，误差清单 #2 的正式裁决）；

- 【默认参数即坑】官方默认（导出档位、`ref_view_strategy`、`process_res`、裁剪阈值）**先读源码**再当 baseline；
- 【报数前三查】① 渲染/GT 视角索引对齐 ② 尺度变换是否双重施加 ③ 相机与高斯是否同一口径（须同一推理产物）；
- 【慢先剖因】看到耗时先分段 profile，不许直接归因"模型大/文件大/磁盘慢"；
- 【假摔三源】掉分先查：视角索引错位 / 尺度双重施加 / 跨分辨率混相机——全排除后才允许归因质量；
- 【单变量铁律】对比验证必须同一次前向产物派生，排除浮点非确定性；
- 【口径对等】拿 A 方法报的数对比 B 方法前，确认输入视角数/训练量/分辨率/是否留出同口径，否则数字不可比。

## DA3 训练事实（2026-11-09 查证 arxiv 2511.10647）

- 主模型：视角数**均匀采样 [2,18]**、基分辨率 504、200K 步、128×H100；batch 按 token 数动态。
- **GS-DPT 高斯头 = 冻结整个 backbone、只训 DPT 头**；损失 = 渲染 photometric（MSE+LPIPS）+
  观测视角 scale-shift-invariant 深度损失。
- ⇒ 我们微调的正确顺序：**先复刻"冻结 backbone 只训头"**（几 M 参数直接全训），
  LoRA backbone 放第二阶段；训练视角采样覆盖 [8,32]，因为推理用 32 已超出官方训练上限 18。

## 历史同类事故（本仓库更早）

- C-06 双分支 2DGS：关 densify + 焊死 offset，三轮负结果浪费多轮 GPU（2026-11-05~08，
  详见 EXPERIMENTS.md「C-06 终止判定」）；
- 区域指标固定人脸框：view 8 落在书架上，人脸锐度虚高 +180（IMPROVEMENTS.md 已录）。
