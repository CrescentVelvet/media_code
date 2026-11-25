#!/usr/bin/env python3
"""从已交付的 DA3 PLY 回放源视角，验证文件、相机和渲染约定是否一致。

Env vars:
  ROOT          海思案例根目录
  CASE_FILTER   可选，单个 case 完整目录名
  OUT_NAME      案例内输出图名，默认 depthanything3/render_check.jpg
  PLY_NAME      案例内 PLY 文件名，默认 depthanything3.ply
  CAMERAS_NPZ   案例内相机文件，默认 depthanything3/cameras.npz
  VIEW_INDICES  可选，逗号分隔源视角；默认全部十视角
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from plyfile import PlyData


def load_ply(path):
    v = PlyData.read(str(path))["vertex"].data

    def cols(names):
        return np.ascontiguousarray(np.stack([v[n] for n in names], axis=-1))

    means = cols(["x", "y", "z"])
    scales = cols(["scale_0", "scale_1", "scale_2"])
    quats = cols(["rot_0", "rot_1", "rot_2", "rot_3"])
    dc = cols(["f_dc_0", "f_dc_1", "f_dc_2"])
    rest_names = sorted((n for n in v.dtype.names if n.startswith("f_rest_")),
                        key=lambda n: int(n.rsplit("_", 1)[1]))
    # PLY 的 f_rest 是 channel-major；gsplat 期望 (N, K, 3)，不能直接 reshape。
    rest = cols(rest_names).reshape(len(v), 3, -1).transpose(0, 2, 1)
    sh = np.concatenate([dc[:, None, :], rest], axis=1)
    opacity = np.ascontiguousarray(v["opacity"])
    return [torch.from_numpy(x).cuda().float() for x in (means, scales, quats, dc, sh, opacity)]


def main():
    from gsplat import rasterization
    from plyfile import PlyData

    root = Path(os.environ.get("ROOT", "/mnt/d/dataset/测试数据/海思端侧重建拍摄数据"))
    selected = os.environ.get("CASE_FILTER", "")
    out_name = os.environ.get("OUT_NAME", "depthanything3/render_check.jpg")
    ply_name = os.environ.get("PLY_NAME", "depthanything3.ply")
    cameras_npz = os.environ.get("CAMERAS_NPZ", "depthanything3/cameras.npz")
    view_indices = [int(x) for x in os.environ.get("VIEW_INDICES", "").split(",") if x]
    for relative in (out_name, ply_name, cameras_npz):
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            sys.exit(f"❌ 输出/输入必须是案例目录内相对路径: {relative}")
    cases = sorted(root.glob("case*_nopose_recentered"),
                   key=lambda p: int(p.name.split("_")[0][4:]))
    if selected:
        cases = [p for p in cases if p.name == selected]
    if not cases:
        sys.exit("❌ 没有匹配案例")
    rows = []
    for case in cases:
        cam = np.load(case / cameras_npz, allow_pickle=False)
        means, scales, quats, _, sh, opacity = load_ply(case / ply_name)
        args = dict(means=means, scales=torch.exp(scales),
                    quats=torch.nn.functional.normalize(quats, dim=-1),
                    opacities=torch.sigmoid(opacity), colors=sh)
        pairs = []
        for i, name in enumerate(cam["image_names"]):
            if view_indices and i not in view_indices:
                continue
            w2c = torch.eye(4, device="cuda")
            w2c[:3] = torch.from_numpy(cam["extrinsics"][i]).cuda().float()
            color, _, _ = rasterization(
                **args, viewmats=w2c[None],
                Ks=torch.from_numpy(cam["intrinsics"][i:i + 1]).cuda().float(),
                width=504, height=504, packed=False, sh_degree=2,
                eps2d=0.01, backgrounds=torch.ones(1, 3, device="cuda"), render_mode="RGB")
            rendered = Image.fromarray((color[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
            source = Image.open(case / "splatanything/images" / str(name)).convert("RGB").resize((504, 504))
            pair = Image.new("RGB", (1008, 534), "white")
            pair.paste(source, (0, 30))
            pair.paste(rendered, (504, 30))
            ImageDraw.Draw(pair).text((8, 8), f"{case.name} | input {i:02d} / DA3 PLY render", fill="black")
            pairs.append(pair)
        sheet = Image.new("RGB", (2016, 534 * 5), "white")
        for i, pair in enumerate(pairs):
            sheet.paste(pair, ((i % 2) * 1008, (i // 2) * 534))
        out = case / out_name
        out.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(out, quality=92)
        print(f"✅ {case.name}: {out}", flush=True)
        del means, scales, quats, sh, opacity, args
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
