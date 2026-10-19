#!/usr/bin/env python3
"""把 FlashWorld 导出的 gaussians.ply 渲染成图（修正版渲染链路）。

背景：FlashWorld 导出 PLY 时做了 inverse activation（官方 utils.py:374-375）：
  opacity 存 logit（要 sigmoid 还原）、scale 存 log（要 exp 还原）。
直接拿 PLY 里的原始 float 喂 gsplat.rasterization 会全黑。

坐标系：export_gaussians 只把 T_norm 乘回去，**没乘 ref_w2c 的逆**——
所以 PLY 的坐标系是「json 里 cameras[0] 的第一帧相机系」（OpenGL 约定，看 -Z）。
要渲染第 i 帧视角，相机矩阵必须变换到同一系：ref_w2c @ c2w_i。

相机约定：json 四元数是 [w,x,y,z]、OpenGL（看 -Z）；gsplat 用 OpenCV（看 +Z），
viewmat = diag(1,-1,-1,1) @ inv(c2w)。

用法（WSL，flashworld env）：
  python flashworld/03_render_ply.py <ply> <input_json> <out_dir>
Env vars:
  OPACITY_MIN   sigmoid 还原后的 opacity 下限，低于则剪枝（默认 0，不剪）。
                官方 cli.py 导出不剪枝，实测 96% 高斯近透明 → 单图生成的
                多深度重影副本会全留在 PLY 里，剪 0.1~0.3 可显著去重影。
  FRAMES        只渲染这些帧序号，逗号分隔（默认全部）
"""
import json
import math
import os
import re
import sys

import numpy as np
import torch


def load_ply(path):
    with open(path, "rb") as f:
        raw = f.read(8192)
        end = raw.index(b"end_header") + len(b"end_header") + 1
        head = raw[:end].decode("ascii", "replace")
        n = int(re.search(r"element vertex (\d+)", head).group(1))
        props = [l.split()[2] for l in head.splitlines() if l.startswith("property")]
        f.seek(end)
        d = np.fromfile(f, dtype=np.float32).reshape(n, len(props))
    nm = {k: i for i, k in enumerate(props)}
    sh_keys = sorted([k for k in props if k.startswith("f_dc_")],
                     key=lambda k: int(k.split("_")[-1]))
    sh = d[:, nm[sh_keys[0]]:nm[sh_keys[0]] + len(sh_keys)].reshape(n, -1, 3)
    return (d[:, 0:3], sh, d[:, nm["opacity"]],
            d[:, nm["scale_0"]:nm["scale_0"] + 3], d[:, nm["rot_0"]:nm["rot_0"] + 4])


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def cam_to_c2w(c):
    M = np.eye(4)
    M[:3, :3] = quat_to_R(np.array(c["quaternion"], dtype=np.float64))
    M[:3, 3] = np.array(c["position"], dtype=np.float64)
    return M


FLIP = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenGL → OpenCV


def main():
    ply_path, json_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)
    xyz, sh, op, sc, rot = load_ply(ply_path)
    with open(json_path) as f:
        meta = json.load(f)
    _, H, W = meta["resolution"]
    cams = meta["cameras"]

    # json 里的内参是「输入原图」分辨率的；app.py 会 center-crop 到 H×W 并 scale 内参。
    # PLY 是模型在 H×W 下生成的，渲染也必须用裁剪后的内参，否则视野放大数倍。
    fx, fy = cams[0]["fx"], cams[0]["fy"]
    cx, cy = cams[0]["cx"], cams[0]["cy"]
    img_path = meta.get("image_prompt", "")
    if os.path.exists(img_path):
        from PIL import Image
        iw, ih = Image.open(img_path).size
        scale = max(H / ih, W / iw)
        new_w, new_h = int(W / scale), int(H / scale)
        fx, fy = fx * scale, fy * scale
        cx = (cx - (iw - new_w) // 2) * scale
        cy = (cy - (ih - new_h) // 2) * scale
        print("原图 %dx%d -> crop+resize %dx%d, scale=%.4f" % (iw, ih, W, H, scale))
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    print("N=%d  res=[%d,%d]  fx=%.1f fy=%.1f cx=%.1f cy=%.1f" % (len(xyz), H, W, fx, fy, cx, cy))

    ref_w2c = np.linalg.inv(cam_to_c2w(cams[0]))

    device = "cuda"
    means = torch.from_numpy(xyz).float().to(device)
    quats = torch.from_numpy(rot).float().to(device)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    scales = torch.exp(torch.from_numpy(sc).float()).to(device)      # log → 线性
    opac = torch.sigmoid(torch.from_numpy(op).float()).to(device)    # logit → 概率
    colors = torch.from_numpy(sh).float().to(device)

    op_min = float(os.environ.get("OPACITY_MIN", "0"))
    if op_min > 0:
        keep = opac >= op_min
        n0 = len(opac)
        means, quats, scales, opac, colors = (x[keep] for x in
                                              (means, quats, scales, opac, colors))
        print("opacity>=%.3f 剪枝: %d -> %d (%.1f%%)" % (op_min, n0, len(opac), 100.0 * len(opac) / n0))

    frames_env = os.environ.get("FRAMES", "")
    if frames_env:
        sel = set(int(x) for x in frames_env.split(","))
        cams = [c for i, c in enumerate(cams) if i in sel]
    K_t = torch.from_numpy(K).float().to(device)

    from gsplat import rasterization
    import imageio.v2 as imageio

    for i, c in enumerate(cams):
        # 有 FRAMES 过滤时 i 是过滤后的序号，文件名要回写真实帧号
        idx = i if not frames_env else int(frames_env.split(",")[i])
        M = ref_w2c @ cam_to_c2w(c)
        viewmat = torch.from_numpy(FLIP @ np.linalg.inv(M)).float().to(device)[None]
        with torch.no_grad():
            img, alpha, _ = rasterization(
                means=means, quats=quats, scales=scales, opacities=opac,
                colors=colors, viewmats=viewmat, Ks=K_t[None],
                width=W, height=H,
                sh_degree=int(round(math.sqrt(colors.shape[1]))) - 1,
                render_mode="RGB", packed=False)
        p = os.path.join(out_dir, "render_%02d.png" % idx)
        imageio.imwrite(p, (img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
        print("  %s  mean=%.3f" % (os.path.basename(p), img[0].mean().item()))
    print("✅ 输出:", out_dir)


if __name__ == "__main__":
    main()
