#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
structsplat/00g_test_selfcheck.py — 链路自检函数的离线判据（无需 GPU/权重）

evaluate_src_views 是 infer.py 里唯一的「跑通 ≠ 跑对」防线，它本身必须可信。
这里用构造张量验证三件事：
  A. 完全相同 → PSNR=inf（不能误报低分）
  B. 平移 / 反色 / 全黑 → PSNR 掉到很低（必须能抓到错位与空画面）
  C. 轻微噪声 → PSNR 在合理区间（不能过度敏感，否则真结果也被判错）
用法: python 00g_test_selfcheck.py
"""

import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("STRUCTSPLAT_REPO") or os.path.expanduser("~/repos/StructSplat")
sys.path.insert(0, REPO)

spec = importlib.util.spec_from_file_location("inf_sc", os.path.join(HERE, "infer.py"))
inf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inf)

H, W = 64, 96
torch.manual_seed(0)

# 造一张有【块状结构】的输入图，而不是平滑梯度。
# 理由：SSIM 依赖局部均值/方差，纯平滑 gradient 的局部方差接近 0，
# 对 0.02 的噪声也会掉到 0.84，是测试图病态，不是函数有病。
# 真实照片有边缘和纹理，用随机方块更贴近实际（也是这次判据修正的唯一改动）。
blk = 8
patch = torch.rand(1, 3, H // blk, W // blk)
gt = patch.repeat_interleave(blk, dim=2).repeat_interleave(blk, dim=3)
gt = (gt * 0.9 + 0.05)

# rendered 侧是 (S,H,W,3)
def as_render(t3):
    return t3.permute(0, 2, 3, 1).contiguous()


fails = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ✅ {name}  {detail}")
    else:
        print(f"  ❌ {name}  {detail}")
        fails.append(name)


print("[A] 完全相同 → 应无误差")
p, s = inf.evaluate_src_views(as_render(gt), gt, H, W)
check("PSNR 为 inf 或极大", p[0] > 60, f"psnr={p[0]:.2f}")
check("SSIM 接近 1", (not s) or s[0] > 0.999, f"ssim={s[0] if s else 'n/a'}")

print("\n[B] 明显错误 → 必须被抓到")
# B-1 水平平移 30%（模拟相机 / 转置方向错）
shift = torch.roll(gt, shifts=W // 3, dims=3)
p1, _ = inf.evaluate_src_views(as_render(shift), gt, H, W)
check("平移错位 PSNR < 15dB", p1[0] < 15.0, f"psnr={p1[0]:.2f}")

# B-2 反色
inv = 1.0 - gt
p2, _ = inf.evaluate_src_views(as_render(inv), gt, H, W)
check("反色 PSNR < 15dB", p2[0] < 15.0, f"psnr={p2[0]:.2f}")

# B-3 全黑（渲染失败的典型症状）
blk = torch.zeros_like(gt)
p3, _ = inf.evaluate_src_views(as_render(blk), gt, H, W)
check("全黑 PSNR < 12dB", p3[0] < 12.0, f"psnr={p3[0]:.2f}")

# B-4 竖图转置方向错（把 H/W 当作可交换，实际内容被转 90°）
#     这是本次真正踩过的坑，必须能被这个哨兵抓到
sq = torch.rand(1, 3, 64, 64)
p4, _ = inf.evaluate_src_views(as_render(sq.transpose(2, 3)), sq, 64, 64)
check("转置 90° PSNR < 12dB", p4[0] < 12.0, f"psnr={p4[0]:.2f}")
print("\n[C] 轻微噪声 → 不应过度敏感")
noisy = (gt + torch.randn_like(gt) * 0.02).clamp(0, 1)
p5, s5 = inf.evaluate_src_views(as_render(noisy), gt, H, W)
check("噪声 0.02 时 PSNR > 25dB", p5[0] > 25.0, f"psnr={p5[0]:.2f}")
check("噪声 0.02 时 SSIM > 0.9", (not s5) or s5[0] > 0.9,
      f"ssim={s5[0] if s5 else 'n/a'}")

print("\n" + "=" * 50)
if fails:
    print(f"❌ {len(fails)} 项未过: {fails}")
    sys.exit(1)
print("🎉 全部通过：自检函数能区分「对/错位/空画面」，且不过度敏感")
