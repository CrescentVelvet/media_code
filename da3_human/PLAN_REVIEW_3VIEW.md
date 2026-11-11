# 三视角方案评审记录（已归档）

> 本文档保留 2026-11-11 的方案评审结论，不再作为独立实验计划。当前唯一执行计划是 [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md)，实测结果见 [`EXPERIMENTS.md`](EXPERIMENTS.md)。

## 评审结论

1. 目标锁定为 **DA3 前馈模型直出**；per-scene 优化、InfiniSplat 短训和测试时位姿优化不属于目标链路。
2. 必须先保存输入 RGB/深度、mask、alpha、Gaussian centers、opacity、scale/rotation、渲染对照和相机审计，再讨论“灰、糊、白洞”。
3. D1 诊断已经完成：白洞由导出裁剪造成；人物出框和模糊来自前馈几何跨视角不一致；人物发灰来自远视角外观/颜色衰减。
4. D2 已冻结为不裁剪 + 全 SH。D3 负责跨视角一致性前馈头重训，D4 只有在 D3 后仍存在独立人脸瓶颈时才启动。

## 已否决或后置的条目

| 条目 | 处理 | 原因 |
|---|---|---|
| DA3 init + 2k/5k 步精调 | 后置，仅作诊断上下界 | 与前馈直出目标冲突 |
| 未重训的全局 4×输入/特征 | 否决 | 504→1008 已测为负收益，且改变了参考系 |
| 单独扫 Gaussian scale/opacity | 并入 D1/D3 | 人物前景 alpha 已接近满覆盖，scale 不是独立根因 |
| 2DGS 光栅器/法向 oracle | 终止 | 横评已证实不能解释主差距 |
| InfiniSplat 输出作为训练标签 | 否决 | 只能作为外部对照，不能替代 GT |

## 固定评测约束

- 真实三视角只报告输入视角自一致性；held-out NVS 结论使用 UE 合成数据。
- 人脸指标逐视角定框；mask 外泄漏和 alpha 必须与 mask 内 PSNR 同报。
- 跨运行固定 `REF_STRATEGY`、`PROCESS_RES`、输入帧、导出参数和渲染器。
- 所有“提升 X dB”都要写明相对哪个口径、哪些视角和哪一套导出参数。

## 历史来源

详细的原始 GPT 方案、冲突核查和变更过程已由 Git 历史及 `EXPERIMENTS.md` 的 2026-11-11 D1 节保存。长篇 DA3-GT 接口/损失草案见 [`28_da3_gt_feedforward_plan.md`](28_da3_gt_feedforward_plan.md)。
