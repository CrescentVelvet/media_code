#!/usr/bin/env python3
"""09d — 离线分析 latent_decode1.pt：transformer 输出到底偏了多少。

背景（2026-09-02）：09b 跑完 50/50 步，出片仍是马赛克。钩子打印的 z 是
pipeline 已反归一化后的 raw latent（09a 已证明 vae.decode 期望 raw 输入：
raw 版 roundtrip 完美、norm 版低对比度）。所以 09b 里"std≈1"的判读基准
写错了——正确基准是：raw latent 的整体 std ≈ RMS(latents_std) ≈ 1.93。

实测 z 整体 std=3.47（偏大约 1.8 倍）。本脚本逐通道分解这个偏差：
  · 均匀放大   → 全局问题（guidance / scheduler / 量化整体偏移）
  · 少数通道爆 → 局部问题（int8 量化打坏个别层/通道）
  · 空间块状   → 与 VAE tile 网格对齐的话另有说法

纯 CPU、秒级完成。latent 文件由 09b 的 logged_decode 钩子存盘。

Env: LATENT_PT 默认 D:/output/minimaxh3_rotate_results/diag_baseline/latent_decode1.pt
     （WSL 路径 /mnt/d/output/minimaxh3_rotate_results/diag_baseline/latent_decode1.pt）
"""
import os
import sys

import torch

LATENT_PT = os.environ.get(
    "LATENT_PT",
    "D:/output/minimaxh3_rotate_results/diag_baseline/latent_decode1.pt")


def main():
    if not os.path.exists(LATENT_PT):
        sys.exit(f"❌ 找不到 {LATENT_PT}（可用 LATENT_PT=... 指定）")

    blob = torch.load(LATENT_PT, map_location="cpu", weights_only=True)
    z = blob["z"].float()                       # [1, C, T, H, W] raw latent
    lm = torch.tensor(blob["latents_mean"]).view(1, -1, 1, 1, 1)
    ls = torch.tensor(blob["latents_std"]).view(1, -1, 1, 1, 1)
    C = z.shape[1]
    print(f"{'=' * 68}")
    print("🔬 09d — latent_decode1.pt 逐通道分析")
    print(f"  z shape = {tuple(z.shape)}   steps={blob['meta'].get('steps')}"
          f"   seed={blob['meta'].get('seed')}")
    print(f"{'=' * 68}")

    # ── 整体基准 ───────────────────────────────────────────────────────
    rms_std = ls.view(-1).pow(2).mean().sqrt().item()
    print(f"\n  基准：RMS(latents_std) = {rms_std:.3f}"
          f"  （归一化后每通道 std=1 时 raw latent 的期望整体 std）")
    print(f"  实测：z 整体 std      = {z.std().item():.3f}"
          f"   → 等效归一化 std ≈ {z.std().item() / rms_std:.2f}x")

    # ── 逐通道：归一化后的 mean/std（理想：mean=0, std=1）──────────────
    zn = (z - lm) / ls                          # [1,C,T,H,W]
    cm = zn.mean(dim=(0, 2, 3, 4))
    cs = zn.std(dim=(0, 2, 3, 4))
    cmax = z.view(C, -1).abs().amax(dim=1) / ls.view(-1)  # 逐通道 |max|/std

    print(f"\n  📊 逐通道（z 反归一化后再归一化；理想 mean=0 std=1）")
    print(f"  {'ch':<4}{'mean':>8}{'std':>8}{'|max|/ls':>10}   判定")
    worst = []
    for c in range(C):
        flag = ""
        if cs[c] > 1.5 or abs(cm[c]) > 1.0 or cmax[c] > 8:
            flag = "  ← 异常"
            worst.append(c)
        print(f"  {c:<4}{cm[c]:>+8.2f}{cs[c]:>8.2f}{cmax[c]:>10.1f}{flag}")

    n_bad = len(worst)
    print(f"\n  异常通道数: {n_bad}/{C}"
          + (f"  → {worst}" if worst else ""))

    # ── 空间/时间分解：偏差集中在哪 ────────────────────────────────────
    print(f"\n  📊 归一化幅值的空间/时间分布（|zn| 的平均，理想应均匀）")
    a = zn.abs().squeeze(0)                     # [C,T,H,W]
    print(f"    按时间帧  : 前 1/4={a[:, :a.shape[1]//4].mean():.2f}"
          f"  中段={a[:, a.shape[1]//4:-a.shape[1]//4].mean():.2f}"
          f"  末 1/4={a[:, -a.shape[1]//4:].mean():.2f}")
    print(f"    按垂直位置: 顶={a[:, :, :a.shape[2]//3].mean():.2f}"
          f"  中={a[:, :, a.shape[2]//3:-a.shape[2]//3].mean():.2f}"
          f"  底={a[:, :, -a.shape[2]//3:].mean():.2f}")
    print(f"    按水平位置: 左={a[:, :, :, :a.shape[3]//3].mean():.2f}"
          f"  中={a[:, :, :, a.shape[3]//3:-a.shape[3]//3].mean():.2f}"
          f"  右={a[:, :, :, -a.shape[3]//3:].mean():.2f}")

    # ── 分位数：是均匀偏大还是长尾 outlier ─────────────────────────────
    q = torch.quantile(zn.flatten(),
                       torch.tensor([0.001, 0.01, 0.5, 0.99, 0.999]))
    print(f"\n  📊 zn 分位数: 0.1%={q[0]:+.2f}  1%={q[1]:+.2f}  "
          f"50%={q[2]:+.2f}  99%={q[3]:+.2f}  99.9%={q[4]:+.2f}")
    print("     （标准正态的 1%=-2.33 / 99%=+2.33；若 99% 明显更大 → 长尾）")

    # ── 判读 ──────────────────────────────────────────────────────────
    med_std = cs.median().item()
    print(f"\n{'=' * 68}")
    print("👉 判读（2026-09-02 修正，guidance/scheduler 已查源码排除）：")
    if n_bad == 0 and 0.8 < med_std < 1.2:
        print("  · 逐通道全部正常（std 中位数≈1，无异常通道）")
        print("      → transformer 输出在分布内，马赛克另有来源：")
        print("        重点查 VAE 解码路径差异（09b leaf offload + fp32；09a 全常驻）")
        print("        → 进 09c 用常驻 VAE 重解码这份 latent")
    else:
        spread = (cs.max() / cs.min().clamp(min=1e-6)).item()
        print(f"  · 偏差形态：std 中位数={med_std:.2f}  最大/最小={spread:.1f}x")
        if spread > 2.5:
            print("      → 偏差集中在少数通道 = 量化打坏了特定权重 → int8 嫌疑最大")
            print("        下一步：transformer 改 bf16 + leaf offload 对照跑")
        else:
            print("      → 各通道均匀偏离 = 全局尺度问题 → 量化累积或 offload 路由")
            print("        关键补充：跑 09d 的 GPU 端扩展（帧间相关、空间差分剖面）")
            print("        判定「量化损失 vs offload 路径问题 vs diffusers pipeline bug」")
            print("        → 推荐：09e 用官方 ComponentsManager.auto_cpu_offload 对照")
    print(f"{'=' * 68}")


if __name__ == "__main__":
    main()
