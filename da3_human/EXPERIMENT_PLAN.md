# da3_human — DA3+2DGS per-scene 重建实验计划

> 本文档是实验计划与决策依据(调研结论 + 复现路线 + 评测协议),**不是执行脚本**。
> 执行脚本按 `../AGENTS.md` 规范编号落本目录;实验数据/结论按日期倒序追加到 `EXPERIMENTS.md`。
> 调研依据:`../media_paper/3D-20261105-QuerySplat.html` §15-④(三方证据闭环),2026-11-05 定稿。

---

## 0. 背景与结论先行

**问题**:同事观察到 DA3 直接重建**精细但有 floater/重影**(归因:跨视角位姿不一致 → 同一表面被重建到微错位位置、自由空间长出伪点);QuerySplat 无 floater/重影但**偏糊**(外观容量不足:SH0 + 低维查询)。

**同事已验证的方案**(内部实证,2026-11):**冻结 DA3 几何骨干 + 低分辨率 2DGS 几何分支 + 高分辨率纹理分支**(纹理清晰度↑)+ **带范围约束的像素中心 offset 与深度 offset**(重影/floater↓)。

**三方证据闭环**(详见 media_paper 笔记):

| 组件 | QuerySplat 证据 | InfiniSplat v2 证据 | 机制 |
|---|---|---|---|
| 冻结几何骨干 | 冻结 VGM,几何查询容量受控 | w/o G 消融:微调几何反而退化(ETH3D LPIPS -16.6%、DTU -23.2%) | 阻断"移动点补偿属性误差"退化通路 |
| 高分辨率纹理分支 | 去外观消融 -4.25 dB | 属性分支微调最后 10 块 | 属性解耦;进一步做到**分辨率维度解耦** |
| 有界 offset | 瞬态 CD 正则(软,仅前 20K 步,全程施加失稳) | w/ O 消融:**全开 offset 有害**(DTU LPIPS 0.2070→0.2826) | 硬信任域,把位姿错位吸收为局部亚像素位移 |

**本实验要回答的问题**:per-scene DA3+2DGS 冻结方案能否替代/增强 vggt_human 的 scene_gs 初始化(原 COLMAP 稀疏点云环节)?

---

## 1. 目标(OKR)

| 级别 | 目标 | 验收 |
|---|---|---|
| O1 | 跑通 per-scene DA3+2DGS 基线 | 多视角棚拍数据上,无 floater/重影、纹理清晰 |
| O2 | 验证相对 3DGS(原版)/ DA3-GS(同事参考)的收益 | NVS:PSNR/LPIPS 不降;几何:Chamfer/F-score 提升 |
| O3 | 量化 offset 界的敏感性 | 消融:offset 界 × {0, 小, 中, 大} 对 floater/重影的影响 |
| O4(可选) | 给出 pose residual 联合优化的增益估计 | 与 O3 联合:开 pose residual 后 offset 界可收紧多少 |

**非目标**:本实验不做前馈网络蒸馏(训练成本工业级,先 per-scene 验证收益再谈);不做动态物体处理(静态场景假设)。

---

## 2. 数据来源

| 数据 | 用途 | 来源 | 状态 |
|---|---|---|---|
| **vggt_human 棚拍多视角数据** | 主评测(真实工作场景) | 现有 pipeline 的多视角图像 + COLMAP 位姿 | 待确认可用视角数/分辨率 |
| DL3DV (子集) | 公开基准对比 | 现成下载 | 备选 |
| DTU | 几何指标评测(有 GT mesh) | 现成下载 | 备选 |

**数据要求**:静态场景、≥4 视角、有相机内参(可来自 COLMAP 或 DA3 自估)、无大面积动态物体。

---

## 3. 实验阶段与脚本编号

按 `../AGENTS.md` 第 1 节规范,脚本数字编号 + 跨平台变体用字母。以下脚本尚未创建,创建时按此编号。

### 阶段 A:环境搭建(0.5 天)

| 脚本 | 内容 |
|---|---|
| `_env.sh` | 共享环境:conda env `da3_human`、代理、CA bundle、路径(DA3 模型/数据根) |
| `00_setup_env.sh` | 服务器:装 torch + 依赖,clone `depth-anything/Depth-Anything-3`(若尚无) |
| `00a_setup_env.sh` | WSL 变体(从零建 env) |

**关键依赖**:`depth-anything/Depth-Anything-3`(Apache 2.0,模型权重 HF)、`diff-surfel-rasterization`(2DGS 光栅化)、`gsplat`(若需要混合 primitive 对比)。

### 阶段 B:基线复现(2 天)

| 脚本 | 内容 | 验收 |
|---|---|---|
| `01_download_models.sh` | 下载 DA3 权重(HF) | DA3 可推理 |
| `02_run_da3_inference.sh` | 对棚拍数据跑 DA3 → 每视角深度图 + 位姿 | 深度可视化合理、位姿可视化无漂移 |
| `03_run_3dgs_baseline.sh` | 原版 3DGS(随机初始化点云)基线 | 有 floater/重影(复现同事观察) |
| `04_run_da3_gs_baseline.sh` | DA3 点云初始化 3DGS(对照组) | 同上 |

**产出**:`EXPERIMENTS.md` 追加基线指标(NVS + 几何)。

### 阶段 C:核心方案 per-scene DA3+2DGS(5 天)

**核心实现思路**(同事方案):

```
输入: n 视角 RGB + 内参 + DA3 位姿
  ↓
[冻结] DA3 推理 → 每视角 metric 深度图 + 位姿
  ↓
[锚定] 反投影建 anchor 点云(降采样到 1/4 分辨率)
        + 多视角一致性筛选(reprojection 误差 > τ 的点剔除)
  ↓
[优化] 2DGS surfel 参数化:
        - 中心 = anchor + 有界 offset(像素中心 tanh + 深度 tanh)
        - offset 界: ±0.5~1 像素 footprint,深度 ±1~2%
        - 几何分支:低分辨率(1/4),只学 2D scale + 法线
        - 纹理分支:高分辨率(全分辨率),只学 SH/RGB + 不透明度
        ↓
[损失] 光度(L1 + SSIM) on 训练视角 + 2DGS 原生 depth-distortion + normal-consistency
        + (可选)opacity 地板正则(来自 QuerySplat 的 α=sigmoid(raw−2.0) 初始化)
  ↓
[后处理] opacity 0.01 剪枝(InfiniSplat v2 零成本嫁接件:命中 51.8% 坏点、仅伤 7.5% 总点)
```

| 脚本 | 内容 |
|---|---|
| `05_build_anchor_pointcloud.py` | DA3 深度反投影 + 降采样 + 多视角一致性筛选 |
| `06_train_da3_2dgs.py` | 冻结 DA3 骨干,双分支 2DGS 优化(核心训练脚本) |
| `07_evaluate.py` | NVS(PSNR/LPIPS)+ 几何(Chamfer/F-score,有 GT 时) |

**关键超参初值**(待调):

| 参数 | 初值 | 依据 |
|---|---|---|
| 几何分支分辨率 | 1/4 of input | 同事方案(低通可变形层) |
| 纹理分支分辨率 | full | 同事方案 |
| offset 界(像素) | ±0.5~1 footprint | InfiniSplat v2 §16-①(建议沿视线 1 自由度;此处给 2D+深度) |
| offset 界(深度) | ±1~2% of depth | 同上,按 DA3 深度不确定度标定 |
| opacity 剪枝阈值 | 0.01 | InfiniSplat v2 实测 |
| 优化器 | Adam,几何 lr 1e-4,纹理 lr 5e-3(参考 InfiniSplat v2) | InfiniSplat v2 Appendix |
| 总迭代 | 10K~30K(依数据规模) | 【待实测】 |

### 阶段 D:消融与敏感性(3 天)

| 实验 | 变量 | 目的 |
|---|---|---|
| D1:offset 界 | {0(锁死), 0.5, 1.0, 2.0, ∞(全开)} | 验证 InfiniSplat v2 w/ O 消融(全开有害)+ 找最优界 |
| D2:解耦方式 | 共享 vs 分离纹理/几何分支 | 验证 QuerySplat -4.25 dB 消融在本框架成立 |
| D3:2DGS vs 3DGS | 同框架换 primitive | 验证 2DGS surfel 抗 floater 收益 |
| D4:剪枝 | 开/关 opacity 0.01 剪枝 | 验证零成本后处理收益 |
| D5(可选):pose residual | 不开/开 SE(3) 小量残差联合优化 | 估计位姿修正的增益上限 |

**消融优先级**:D1 > D4 > D2 > D3 > D5。若时间紧,D5 可推迟到下一期。

### 阶段 E:结论与文档(1 天)

| 事项 | 内容 |
|---|---|
| `EXPERIMENTS.md` | 全部实验数据按日期倒序追加,含失败案例 |
| `NOTES.md` | 机制解释:为什么有界 offset 有效(信任域吸收位姿错位);与 QuerySplat/InfiniSplat v2 的异同 |
| README.md | 更新一行摘要 + 复现步骤 |

---

## 4. 评测协议(必须分开报)

| 类别 | 指标 | 数据来源 |
|---|---|---|
| NVS(新视角合成) | PSNR ↑ / LPIPS ↓ / SSIM ↑ | held-out 视角(建议留 20% 视角不参与训练) |
| 几何 | Chamfer-L1 ↓ / F-score ↑ | 有 GT mesh 时(DTU);否则用 DA3 深度做 pseudo-GT(标注【待核实】) |
| floater/重影 | 人工定性评分(0~3 分) + 自由空间点数占比 | 统一视角截图对比 |

**警示**(InfiniSplat v2 教训):InfiniSplat v2 主指标是 LPIPS/DISTS(感知),其 PSNR 前馈仍输 ReSplat 0.63 dB——高分辨率纹理分支 + LPIPS 损失的感知收益可能掩盖几何误差。**必须用 PSNR + Chamfer 双指标兜底**。

---

## 5. 风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| DA3 深度局部不一致(多视角) | anchor 带毒 → 优化发散 | anchor 建前多视角一致性筛选(阶段 C 已含);调整筛选阈值 τ |
| offset 界标定困难 | 太松 floater 回来,太紧重影修不动 | D1 消融直接扫,找最优界;后续升级置信度自适应界(下一期) |
| 纹理分支欠拟合(仍糊) | 误以为几何问题 | 先冻结几何只看纹理分支 loss;2DGS 只稳几何不给清晰度,糊的主因在纹理分支容量/训练量 |
| 动态物体(行人) | 重影另一来源 | 本期不做;后续用 SAM3 person mask 做 visibility gating(参考 vggt_human pipeline 经验) |
| 前馈化诱惑 | 训练成本失控 | 本期明确不做;若 per-scene 收益确认,下一期再评估蒸馏成本 |

---

## 6. 下一步(本实验之后)

| 方向 | 前提 | 预期收益 |
|---|---|---|
| 置信度自适应 offset 界 | D1 确认有界 offset 有效 | 进一步压 floater,论文级空白 |
| pose residual 联合优化 | D5 有正收益 | 治根(位姿不一致),pose-free 路线真天花板 |
| 前馈蒸馏 | per-scene 收益显著 | 可商用 pose-free 前馈管线(对标 QuerySplat,但 DA3 Apache 2.0 解许可) |
| 动态物体处理 | 业务需求出现 | SAM3 mask + visibility gating |

---

## 7. 参考资源

- **QuerySplat 笔记(三方证据闭环)**:`../media_paper/3D-20261105-QuerySplat.html` §15-④、§16
- **InfiniSplat v2 笔记(嫁接件来源)**:`../media_paper/3D-20261014-InfiniSplat-v2.html`
- **DA3 笔记(替换候选)**:`../media_paper/3D-20251113-Depth-Anything-3.html`
- **2DGS 笔记(primitive 参考)**:`../media_paper/3D-20240326-2DGS.html`
- **代码库**:`depth-anything/Depth-Anything-3`(Apache 2.0)、`hbb1/diff-surfel-rasterization`(2DGS)、`nerfstudio-project/gsplat`(3DGS/2DGS 光栅化)
