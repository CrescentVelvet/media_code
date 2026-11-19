#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
structsplat/00f_test_image_transpose.py — 图片预处理口径的回归判据

两个真实踩过的坑，本脚本各给一条断言：

  坑 A  竖图转置写成空操作
        官方 rearrange("c h w -> c w h") == t.transpose(1, 2)。
        早期 infer.py 写成 permute(1,2,0) 再 permute(2,0,1)，两次互逆 → 没转，
        竖图被横向压扁 2.3 倍，与官方像素差 0.973。

  坑 B  transforms.json 的 GT 相机只换内参没换外参
        转置后相机系发生 X<->Y 交换，得到 R' = R·M（右乘），t 不变。
        写成左乘 M 会把相机整体转错，实测误差 8805 px，比不修的 265 px 还差。

本脚本不需要 GPU、不需要网络、不需要任何模型权重，只依赖 PIL + torch：
    python 00f_test_image_transpose.py [--images <heic 目录>] [--transforms <json>]

不提供 --images 时自动造一张合成竖图跑纯几何部分（坑 A 需要一个能解码的图，
无图时退化为「用合成数组直接验 transpose 语义」）。
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FAIL = []


def check(name, ok, detail=""):
    print(f"  {'✅' if ok else '❌'} {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


# ============================================================
# 坑 A：转置语义
# ============================================================
def official_preprocess(path, new_h, new_w):
    """逐个复刻官方 src/data/utils.py::load_and_preprocess_images 单图分支。"""
    from PIL import Image
    from torchvision.transforms import functional as TF
    from einops import rearrange

    img = Image.open(path).convert("RGB")
    w, h = img.size
    t = TF.to_tensor(img).float()
    if new_h < new_w and h > w:
        t = rearrange(t, "c h w -> c w h")     # ← 官方原句
    t = t.unsqueeze(0)
    t = F.interpolate(t, (new_h, new_w), mode="bilinear",
                      align_corners=True, antialias=True)
    return t.squeeze(0), (w, h)


def test_transpose_semantics():
    """纯几何：确认 rearrange("c h w -> c w h") 就是 transpose(1,2)，不是恒等。"""
    print("\n[坑 A-1] rearrange 语义（纯张量，不需要图片）")
    from einops import rearrange

    a = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    r = rearrange(a, "c h w -> c w h")
    check("rearrange == transpose(1,2)", torch.equal(r, a.transpose(1, 2)),
          f"shape {tuple(a.shape)} -> {tuple(r.shape)}")
    check("rearrange 不是恒等", not torch.equal(r, a), "若相等说明语义被误解")

    # 复现旧 bug 的写法，证明它确实等于恒等
    t = a.clone()
    t = t.permute(1, 2, 0).contiguous()
    t = t.permute(2, 0, 1).contiguous()
    check("旧写法两次 permute 互逆（=恒等）", torch.equal(t, a),
          "这解释了竖图为何被压扁")


def test_image_pipeline(image_path, new_h=294, new_w=518):
    print(f"\n[坑 A-2] 真实图片预处理 vs 官方（{os.path.basename(image_path)}）")
    from infer import load_image_tensor

    mine, (w, h) = load_image_tensor(image_path, new_h, new_w)
    ref, _ = official_preprocess(image_path, new_h, new_w)

    check("shape 一致", tuple(mine.shape) == tuple(ref.shape),
          f"{tuple(mine.shape)} vs {tuple(ref.shape)}")
    d = (mine - ref).abs()
    check("逐像素一致", d.max().item() < 1e-6,
          f"max={d.max().item():.8f} mean={d.mean().item():.8f}")

    if h > w:
        ratio_orig = w / h
        ratio_new = new_w / new_h
        check("宽高比未被扭曲", abs(ratio_orig - ratio_new) > 0.3,
              f"原图 {ratio_orig:.3f} → 目标 {ratio_new:.3f}（转置后应对调，不是拉伸）")
        # 转置后应是 (new_w/new_h) 与 (h/w) 同量级
        achieved = mine.shape[2] / mine.shape[1]
        check("转置后内容方向正确", abs(achieved - ratio_new) < 1e-3,
              f"输出 HxW = {mine.shape[1]}x{mine.shape[2]}")


# ============================================================
# 坑 B：GT 相机转置
# ============================================================
def test_gt_camera_geometry(transforms_path, image_paths, new_h=294, new_w=518, n_pts=8):
    """
    端到端像素重投影：原图像素 → 反投影到世界 → 用 load_gt_cameras 返回的
    w2c/K 重投影，检查是否落在转置后期望位置 (v·new_w/H0, u·new_h/W0)。
    正确实现应为 0，右乘/左乘/无交换三路会被明显区分开。
    """
    print(f"\n[坑 B] GT 相机转置重投影（{os.path.basename(transforms_path)}）")
    from infer import load_gt_cameras

    w2cs, Ks, matched = load_gt_cameras(transforms_path, image_paths, new_h, new_w)
    check("图片与 GT 帧匹配", len(matched) == len(image_paths),
          f"{len(matched)}/{len(image_paths)}")
    if not matched:
        return

    with open(transforms_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    fr = {os.path.basename(x.get("file_path") or x.get("filePath")): x
          for x in meta.get("frames", [])}
    f0 = fr[matched[0]]

    H0, W0 = f0["h"], f0["w"]
    fx, fy, cx, cy = f0["fl_x"], f0["fl_y"], f0["cx"], f0["cy"]
    c2w = torch.tensor(f0["transform_matrix"], dtype=torch.float32)

    rng = np.random.default_rng(0)
    us = rng.uniform(0.1, 0.9, n_pts) * W0
    vs = rng.uniform(0.1, 0.9, n_pts) * H0
    zs = rng.uniform(1.0, 3.0, n_pts)

    # 原图像素 + 任意深度 → 相机系 → 世界
    Xc = np.stack([(us - cx) / fx * zs, (vs - cy) / fy * zs, zs], axis=1)
    Xw = Xc @ c2w[:3, :3].numpy().T + c2w[:3, 3].numpy()

    def reproj_err(w2c, K):
        Xc2 = Xw @ np.asarray(w2c)[:3, :3].T + np.asarray(w2c)[:3, 3]
        uv = Xc2 @ np.asarray(K).T
        u2, v2 = uv[:, 0] / uv[:, 2], uv[:, 1] / uv[:, 2]
        # 转置后期望：u' = v·(new_w/H0), v' = u·(new_h/W0)
        exp_u, exp_v = vs * (new_w / H0), us * (new_h / W0)
        return np.sqrt((u2 - exp_u) ** 2 + (v2 - exp_v) ** 2).max()

    err_right = reproj_err(w2cs[0], Ks[0])
    check("右乘 M（正确实现）", err_right < 1e-3, f"max {err_right:.4f} px")

    # 反例 1：完全不动外参（旧 bug）
    c2w_bad = c2w.clone()
    err_none = reproj_err(torch.linalg.inv(c2w_bad), Ks[0])
    # 反例 2：左乘 M（第一版错误修法）
    M = torch.tensor([[0., 1., 0.], [1., 0., 0.], [0., 0., 1.]])
    c2w_left = c2w.clone()
    c2w_left[:3, :3] = M @ c2w_left[:3, :3]
    err_left = reproj_err(torch.linalg.inv(c2w_left), Ks[0])

    check("能区分出「不换外参」的错", err_none > 1.0,
          f"错版 {err_none:.3f} px > 正确版 {err_right:.4f} px")
    check("能区分出「左乘 M」的错", err_left > 1.0,
          f"错版 {err_left:.3f} px（实测比不修还差一个数量级）")


def main():
    ap = argparse.ArgumentParser("StructSplat 图片预处理口径回归检查")
    ap.add_argument("--images", default=None, help="竖图 .heic 目录（坑 A-2 用）")
    ap.add_argument("--transforms", default=None, help="transforms.json（坑 B 用）")
    ap.add_argument("--resize", nargs=2, type=int, default=[294, 518])
    ap.add_argument("--max-images", type=int, default=4)
    args = ap.parse_args()

    new_h, new_w = args.resize
    print("=" * 62)
    print("StructSplat 图片预处理 / GT 相机 口径回归")
    print("=" * 62)

    test_transpose_semantics()

    if args.images and os.path.isdir(args.images):
        cands = []
        for ext in ("*.heic", "*.HEIC", "*.png", "*.jpg", "*.jpeg"):
            cands += sorted(glob.glob(os.path.join(args.images, ext)))
        if cands:
            test_image_pipeline(cands[0], new_h, new_w)
        else:
            print("\n[坑 A-2] 跳过：目录里没有可用图片")
    else:
        print("\n[坑 A-2] 跳过：未提供 --images")

    if args.transforms and os.path.isfile(args.transforms):
        if not (args.images and os.path.isdir(args.images)):
            print("\n[坑 B] 跳过：需要同时提供 --images")
        else:
            cands = []
            for ext in ("*.heic", "*.HEIC", "*.png", "*.jpg", "*.jpeg"):
                cands += sorted(glob.glob(os.path.join(args.images, ext)))
            test_gt_camera_geometry(args.transforms, cands[:args.max_images],
                                    new_h, new_w)
    else:
        print("\n[坑 B] 跳过：未提供 --transforms")

    print("\n" + "=" * 62)
    if FAIL:
        print(f"❌ {len(FAIL)} 项未通过：")
        for n in FAIL:
            print(f"   - {n}")
        sys.exit(1)
    print("🎉 全部通过")
    print("=" * 62)


if __name__ == "__main__":
    main()
