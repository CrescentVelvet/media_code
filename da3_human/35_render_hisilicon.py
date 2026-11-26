#!/usr/bin/env python3
"""从已交付的 DA3 PLY 回放源视角，验证文件、相机和渲染约定是否一致。

Env vars:
  ROOT          海思案例根目录
  CASE_FILTER   可选，单个 case 完整目录名
  OUT_NAME      案例内输出图名，默认 depthanything3/render_check.jpg
  PLY_NAME      案例内 PLY 文件名，默认 depthanything3.ply
  CAMERAS_NPZ   案例内相机文件，默认 depthanything3/cameras.npz
  VIEW_INDICES  可选，逗号分隔源视角；默认全部已保存视角
  SOURCE_IMAGES_DIR 可选，案例内源 RGB 目录，默认 splatanything/images
  EXTERNAL_PLY 可选，外部 PLY 绝对路径；用于与同事结果同相机回放
  RENDER_SIZE 可选，WxH；按 CAMERA_BASE_SIZE 或 depth 尺寸等比例调整内参
  CAMERA_BASE_SIZE 可选，WxH；无 depth 时必须显式提供 K 对应的图像尺寸
  RENDER_REPORT 可选，回放参数 JSON 路径，默认与 OUT_NAME 同目录同名 .json
  GSPLAT_CACHED_SO 可选，已编译且与当前环境匹配的 gsplat_cuda.so；
                    直接加载缓存，避免无源码包触发 JIT 重编译
"""
import os
import sys
import importlib.util
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
    if rest_names:
        rest = cols(rest_names).reshape(len(v), 3, -1).transpose(0, 2, 1)
    else:
        rest = np.empty((len(v), 0, 3), dtype=np.float32)
    sh = np.concatenate([dc[:, None, :], rest], axis=1)
    opacity = np.ascontiguousarray(v["opacity"])
    if not np.isfinite(opacity).all():
        bad = int((~np.isfinite(opacity)).sum())
        raise ValueError(f"{path}: 有 {bad} 个 opacity NaN/Inf；请用含 finite_mask 的诊断导出重跑")
    return [torch.from_numpy(x).cuda().float() for x in (means, scales, quats, dc, sh, opacity)]


def camera_resolution(cam):
    """从深度或显式尺寸获得内参基准；绝不根据当前输出尺寸反推。"""
    from PIL import Image

    if "depth" in cam:
        base_h, base_w = (int(v) for v in cam["depth"].shape[-2:])
        base = (base_w, base_h)
        if os.environ.get("CAMERA_BASE_SIZE"):
            explicit = tuple(int(v) for v in os.environ["CAMERA_BASE_SIZE"].lower().replace("*", "x").split("x"))
            if explicit != base:
                raise ValueError(f"CAMERA_BASE_SIZE={explicit} 与 depth 对应的 K 基准 {base} 不一致")
        return base
    if os.environ.get("CAMERA_BASE_SIZE"):
        return tuple(int(v) for v in os.environ["CAMERA_BASE_SIZE"].lower().replace("*", "x").split("x"))
    if "image_shape" in cam:
        base_h, base_w = (int(v) for v in np.asarray(cam["image_shape"]).reshape(-1))
        return base_w, base_h
    if "intrinsics_base_size" in cam:
        return tuple(int(v) for v in np.asarray(cam["intrinsics_base_size"]).reshape(-1))
    names = [str(n) for n in cam["image_names"]]
    if names and not os.environ.get("ALLOW_UNSAFE_INTRINSICS_INFERENCE"):
        raise ValueError("NPZ 无 depth/image_shape；须显式设置 CAMERA_BASE_SIZE=WxH，禁止猜测 K 基准")
    if names and os.environ.get("ALLOW_UNSAFE_INTRINSICS_INFERENCE") == "1":
        with Image.open(names[0]) as im:
            return im.size
    raise ValueError("相机文件缺少图像名与内参基准尺寸")


def load_intrinsics(cam, width, height):
    base_w, base_h = camera_resolution(cam)
    if min(base_w, base_h) <= 0:
        raise ValueError(f"非法内参基准尺寸: {base_w}x{base_h}")
    intrinsics = cam["intrinsics"].astype(np.float32).copy()
    intrinsics[:, 0] *= width / base_w
    intrinsics[:, 1] *= height / base_h
    return intrinsics, (base_w, base_h)


def main():
    if os.environ.get("GSPLAT_CACHED_SO"):
        cached = Path(os.environ["GSPLAT_CACHED_SO"]).resolve()
        if not cached.is_file() or cached.is_symlink():
            raise RuntimeError(f"缓存扩展不是普通文件: {cached}")
        spec = importlib.util.spec_from_file_location("gsplat_cuda", cached)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules["gsplat_cuda"] = module
        sys.modules["gsplat.csrc"] = module
        import gsplat
        gsplat.csrc = module
        print(f"使用已编译渲染扩展: {cached}", flush=True)
    from gsplat import rasterization
    from plyfile import PlyData

    root = Path(os.environ.get("ROOT", "/mnt/d/dataset/测试数据/海思端侧重建拍摄数据"))
    selected = os.environ.get("CASE_FILTER", "")
    out_name = os.environ.get("OUT_NAME", "depthanything3/render_check.jpg")
    ply_name = os.environ.get("PLY_NAME", "depthanything3.ply")
    cameras_npz = os.environ.get("CAMERAS_NPZ", "depthanything3/cameras.npz")
    source_images_dir = os.environ.get("SOURCE_IMAGES_DIR", "splatanything/images")
    external_ply = os.environ.get("EXTERNAL_PLY", "")
    view_indices = [int(x) for x in os.environ.get("VIEW_INDICES", "").split(",") if x]
    if external_ply and not Path(external_ply).is_file():
        sys.exit(f"❌ 外部 PLY 不存在: {external_ply}")
    for relative in (out_name, ply_name, cameras_npz, source_images_dir):
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            sys.exit(f"❌ 输出/输入必须是案例目录内相对路径: {relative}")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from importlib.util import module_from_spec, spec_from_file_location
    _spec = spec_from_file_location(
        "hisilicon_rgb", Path(__file__).resolve().parent / "34_run_hisilicon.py")
    _io = module_from_spec(_spec)
    _spec.loader.exec_module(_io)
    cases = [case for case in _io.discover_cases(root, selected)
             if (case / cameras_npz).is_file()
             and (Path(external_ply).is_file() if external_ply else (case / ply_name).is_file())]
    if not cases:
        sys.exit("❌ 没有匹配案例")
    rows = []
    reports = []
    for case in cases:
        source_dir = source_images_dir
        if source_dir == "splatanything/images" and not (case / source_dir).is_dir():
            source_dir = "images"
        z = np.load(case / cameras_npz, allow_pickle=False)
        cam = {k: z[k] for k in z.files}
        if cam["extrinsics"].ndim == 4 and cam["extrinsics"].shape[0] == 1:
            cam["extrinsics"] = cam["extrinsics"][0]
        if cam["intrinsics"].ndim == 4 and cam["intrinsics"].shape[0] == 1:
            cam["intrinsics"] = cam["intrinsics"][0]
        means, scales, quats, _, sh, opacity = load_ply(
            Path(external_ply) if external_ply else case / ply_name)
        if sh.shape[1] not in (1, 4, 9, 16):
            raise ValueError(f"不支持的 SH 维度: {sh.shape}")
        sh_degree = int(round(sh.shape[1] ** 0.5)) - 1
        args = dict(means=means, scales=torch.exp(scales),
                    quats=torch.nn.functional.normalize(quats, dim=-1),
                    opacities=torch.sigmoid(opacity), colors=sh)
        for key, tensor in args.items():
            if not torch.isfinite(tensor).all():
                raise ValueError(f"PLY 加载后 {key} 含 NaN/Inf，拒绝把非有限值当作画质结论")
        pairs = []
        width, height = 504, 504
        if os.environ.get("RENDER_SIZE"):
            width, height = (int(v) for v in os.environ["RENDER_SIZE"].lower().replace("*", "x").split("x"))
        intrinsics, base_size = load_intrinsics(cam, width, height)
        for i, name in enumerate(cam["image_names"]):
            if view_indices and i not in view_indices:
                continue
            w2c = torch.eye(4, device="cuda")
            w2c[:3] = torch.from_numpy(cam["extrinsics"][i, :3]).cuda().float()
            color, _, _ = rasterization(
                **args, viewmats=w2c[None],
                Ks=torch.from_numpy(intrinsics[i:i + 1]).cuda().float(),
                width=width, height=height, packed=False, sh_degree=sh_degree,
                eps2d=0.01, backgrounds=torch.ones(1, 3, device="cuda"), render_mode="RGB")
            rendered = Image.fromarray((color[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
            source = Image.open(case / source_dir / str(name)).convert("RGB").resize((width, height))
            pair = Image.new("RGB", (2 * width, height + 30), "white")
            pair.paste(source, (0, 30))
            pair.paste(rendered, (width, 30))
            ImageDraw.Draw(pair).text((8, 8), f"{case.name} | input {i:02d} / DA3 PLY render", fill="black")
            pairs.append(pair)
        sheet = Image.new("RGB", (4 * width, (height + 30) * ((len(pairs) + 1) // 2)), "white")
        for i, pair in enumerate(pairs):
            sheet.paste(pair, ((i % 2) * 2 * width, (i // 2) * (height + 30)))
        out = case / out_name
        out.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(out, quality=92)
        report = {"case": case.name, "ply": str(external_ply or case / ply_name),
                  "cameras": str(case / cameras_npz), "render_size": [width, height],
                  "intrinsics_base_size": list(base_size),
                  "camera_frame_note": str(cam.get("camera_frame", "见相机生成脚本")),
                  "view_indices": [int(i) for i, name in enumerate(cam["image_names"])
                                   if not view_indices or i in view_indices]}
        reports.append(report)
        report_path = os.environ.get("RENDER_REPORT", str(out.with_suffix(".json")))
        report_path = Path(report_path)
        if not report_path.is_absolute():
            report_path = case / report_path
        report_path.parent.mkdir(parents=True, exist_ok=True)
        import json
        with open(report_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(reports if len(cases) > 1 else report, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"✅ {case.name}: {out} | K {base_size[0]}x{base_size[1]} -> {width}x{height}", flush=True)
        del means, scales, quats, sh, opacity, args
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
