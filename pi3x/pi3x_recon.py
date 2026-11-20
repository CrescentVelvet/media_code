#!/usr/bin/env python3
"""Pi3X 前馈推理：图像文件夹 → 位姿 + 稠密点云（+ 可选 GT 条件注入）。

输出到 SAVE_DIR：
  predictions.npz  points/camera_poses/conf(f16)/images(u8)/frame_names/K_recovered/target_wh
  dense_cloud.ply  置信度+边缘过滤后的彩色点云（官方 example_mm 同款过滤）
  poses.json       每帧 c2w 4x4（OpenCV 约定）
  intrinsics.json  逐帧恢复内参（resize 后像素单位）+ resize 信息

Env vars:
  IMAGES_DIR      图像文件夹（必填；02 产出的 images/）
  SAVE_DIR        输出目录（必填）
  PI3_DIR         Pi3 官方仓路径（_env.sh 给）
  PI3X_CKPT       权重 safetensors 路径（_env.sh 给）
  CONDITIONS_NPZ  可选，含 intrinsics/poses 的 npz（02 产出）；空 = 纯图像模式
  PIXEL_LIMIT     输入像素上限（默认 255000，官方默认）
  CONF_THRES      sigmoid(conf) 阈值（默认 0.1，官方默认）
  DEVICE          cuda / cpu
"""
import json
import os
import sys

import numpy as np
import torch

IMAGES_DIR = os.environ.get("IMAGES_DIR", "")
SAVE_DIR = os.environ.get("SAVE_DIR", "")
PI3_DIR = os.environ.get("PI3_DIR", "")
CKPT = os.environ.get("PI3X_CKPT", "")
CONDITIONS_NPZ = os.environ.get("CONDITIONS_NPZ", "")
PIXEL_LIMIT = int(os.environ.get("PIXEL_LIMIT", "255000"))
CONF_THRES = float(os.environ.get("CONF_THRES", "0.1"))
DEVICE = os.environ.get("DEVICE", "cuda")

sys.path.insert(0, PI3_DIR)


def main():
    if not IMAGES_DIR or not os.path.isdir(IMAGES_DIR):
        sys.exit("❌ IMAGES_DIR 未设或不存在: %s" % IMAGES_DIR)
    if not SAVE_DIR:
        sys.exit("❌ SAVE_DIR 未设")
    if not os.path.isfile(CKPT):
        sys.exit("❌ 权重不存在: %s（先跑 pi3x/01_download_models.sh）" % CKPT)
    os.makedirs(SAVE_DIR, exist_ok=True)

    from pi3.models.pi3x import Pi3X
    from pi3.utils.basic import load_multimodal_data, write_ply
    from pi3.utils.geometry import depth_normal_edge, recover_intrinsic_from_rays_d

    device = torch.device(DEVICE)
    print("🚀 [pi3x] 推理")
    print("  🖼️  输入: %s" % IMAGES_DIR)
    print("  🏋️  权重: %s" % CKPT)
    print("  📐 PIXEL_LIMIT=%d  CONF_THRES=%.2f" % (PIXEL_LIMIT, CONF_THRES))

    # --- 条件（可选）---
    conditions = None
    if CONDITIONS_NPZ:
        data = np.load(CONDITIONS_NPZ, allow_pickle=True)
        conditions = {"intrinsics": data["intrinsics"], "poses": data["poses"], "depths": None}
        print("  🎯 条件注入: intrinsics + poses <- %s" % CONDITIONS_NPZ)

    imgs, conditions = load_multimodal_data(
        IMAGES_DIR, conditions, interval=1, PIXEL_LIMIT=PIXEL_LIMIT, device=device)
    n, _, th, tw = imgs.shape[1], imgs.shape[2], imgs.shape[3], imgs.shape[4]
    print("  🧮 tensor: N=%d  %dx%d" % (n, th, tw))

    use_multimodal = any(v is not None for v in conditions.values())
    model = Pi3X(use_multimodal=use_multimodal).eval()
    if CKPT.endswith(".safetensors"):
        from safetensors.torch import load_file
        weight = load_file(CKPT)
    else:
        weight = torch.load(CKPT, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(weight, strict=False)
    print("  📦 load_state_dict: missing=%d unexpected=%d" % (len(missing), len(unexpected)))
    if not use_multimodal:
        model.disable_multimodal()
    model = model.to(device)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    t0 = __import__("time").time()
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=dtype):
            res = model(imgs=imgs, **conditions)
    dt = __import__("time").time() - t0
    print("  ⏱️  前向 %.1fs（%d 帧）" % (dt, n))

    # 恢复内参（射线方向法，官方 example_mm 同款）
    rays_d = torch.nn.functional.normalize(res["local_points"], dim=-1)
    K_rec = recover_intrinsic_from_rays_d(rays_d, force_center_principal_point=True)[0]  # (N,3,3)
    print("  🔍 首帧恢复内参:\n%s" % K_rec[0].cpu().numpy())

    # --- 存 predictions.npz ---
    frame_names = sorted(x for x in os.listdir(IMAGES_DIR)
                         if x.lower().endswith((".png", ".jpg", ".jpeg")))
    np.savez(os.path.join(SAVE_DIR, "predictions.npz"),
             points=res["points"][0].float().cpu().numpy(),
             camera_poses=res["camera_poses"][0].float().cpu().numpy(),
             conf=res["conf"][0].half().cpu().numpy(),
             images=(imgs[0].permute(0, 2, 3, 1) * 255).byte().cpu().numpy(),
             frame_names=np.array(frame_names),
             K_recovered=K_rec.float().cpu().numpy(),
             target_wh=np.array([tw, th]))
    print("  ✅ predictions.npz")

    # --- 稠密点云（官方过滤：conf + 深度法向边缘）---
    masks = torch.sigmoid(res["conf"][..., 0]) > CONF_THRES
    non_edge = ~depth_normal_edge(res["local_points"], rtol=0.03, mask=masks)
    masks = torch.logical_and(masks, non_edge)[0]
    ply_path = os.path.join(SAVE_DIR, "dense_cloud.ply")
    write_ply(res["points"][0][masks].cpu(), imgs[0].permute(0, 2, 3, 1)[masks], ply_path)
    keep = int(masks.sum().item())
    total = masks.numel()
    print("  ✅ dense_cloud.ply  %d 点（过滤后占比 %.1f%%）" % (keep, 100.0 * keep / total))

    # --- 人类可读位姿/内参 ---
    c2w = res["camera_poses"][0].float().cpu().numpy()
    with open(os.path.join(SAVE_DIR, "poses.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({frame_names[i]: c2w[i].tolist() for i in range(n)}, f, indent=2)
    with open(os.path.join(SAVE_DIR, "intrinsics.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({
            "target_wh": [tw, th],
            "K_recovered": {frame_names[i]: K_rec[i].float().cpu().numpy().tolist()
                            for i in range(n)},
        }, f, indent=2)
    print("  ✅ poses.json + intrinsics.json")
    print("🎉 Done. -> %s" % SAVE_DIR)


if __name__ == "__main__":
    main()
