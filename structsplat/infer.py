#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
structsplat/infer.py — 任意图片目录 → 前馈高斯重建 + 渲染 + 导出

官方仓只有 train.py / evaluation.py，两者都硬绑 DL3DV annotation 格式，
没有「给一个图片目录就出结果」的入口。本脚本补这一环。

设计（依据 src/model/gaussian_wrapper.py 的实际实现）：

1. 相机来源两条路
   - pose-free（默认）：把全部图片当 src 送给 model()，用 VGGT camera_head 预测的
     外参/内参直接渲染。这是论文真正声称的能力（camera-free）。
   - gt（可选）：读 transforms.json 的 c2w，转 w2c 后用。用于验证「几何对了没有」，
     把相机估计误差从渲染误差里分离出来。

2. 为什么不用 model.splat()
   splat() 内部会调 get_tar_camera()——它需要「src 图 + tar 图」两组，用 src 上估计的
   相机去对齐 tar 上估计的相机（leakage-free 的 Sim(3) 对齐）。当我们只有一个图片
   目录、全部当 src 时，src_number == S，tar 为空，那条分支没有意义。
   所以这里直接走 model(src) → 拿 coordinate/rotation/scale/opacity/color + 相机，
   自己调 gsplat.rasterization 渲染，避免空分支和多余的一次编码器前向。

3. 渲染沿用官方口径
   scale_modifier / sh_degree / rasterize_mode / white_bg 全部从 config 读，
   camera 用 pose_encoding_to_extri/intri，与 gaussian_wrapper 完全一致。

用法：
  python infer.py --images <目录或图片列表> --out <输出目录>
                  [--config config/dl3dv.yaml] [--resize 294 518]
                  [--render-size 512] [--camera pose-free|gt]
                  [--transforms <transforms.json>] [--max-images 24]
                  [--export-splat] [--export-ply]
"""

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

# ---------- 让 `import vggt` / `import src.*` 生效（仓根 + third_party/vggt）----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("STRUCTSPLAT_REPO") or os.path.join(
    os.path.dirname(SCRIPT_DIR), "repos", "StructSplat"
)
if not os.path.isdir(REPO):
    for cand in (os.path.expanduser("~/repos/StructSplat"), "/home/velvet/repos/StructSplat"):
        if os.path.isdir(cand):
            REPO = cand
            break
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "third_party", "vggt"))

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".heic", ".heif", ".tif", ".tiff"}


# ============================================================
# 图片收集与预处理（复刻 src/data/utils.py 的口径）
# ============================================================
def collect_images(images_arg, max_images=None, stride=None):
    """支持：目录 / 单个文件 / 逗号分隔列表 / glob 模式。"""
    paths = []
    for part in images_arg.split(","):
        part = part.strip()
        if not part:
            continue
        if os.path.isdir(part):
            for f in sorted(os.listdir(part)):
                if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                    paths.append(os.path.join(part, f))
        elif os.path.isfile(part):
            paths.append(part)
        else:
            paths.extend(sorted(glob.glob(part)))
    # 去重保序
    seen, uniq = set(), []
    for p in paths:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            uniq.append(ap)
    paths = uniq
    if stride and stride > 1:
        paths = paths[::stride]
    if max_images and len(paths) > max_images:
        # 均匀抽帧，避免只取到连续相邻视角
        idx = np.linspace(0, len(paths) - 1, max_images).round().astype(int)
        paths = [paths[i] for i in sorted(set(idx.tolist()))]
    return paths


def load_image_tensor(path, new_h, new_w, crop=False):
    """
    复刻 load_and_preprocess_images 单图逻辑：
      - to_tensor 归一到 [0,1]
      - 竖图且目标是横图时转置（rearrange c h w -> c w h）
      - crop=False 走 bilinear interpolate（不是 crop，别混）
    heic 走 pillow-heif 兜底。
    """
    from PIL import Image

    ext = os.path.splitext(path)[1].lower()
    if ext in (".heic", ".heif"):
        try:
            import pillow_heif  # noqa: F401
            pillow_heif.register_heif_opener()
        except Exception as e:
            raise RuntimeError(
                f"读 {ext} 需要 pillow-heif：pip install pillow-heif（{e}）"
            ) from e

    img = Image.open(path).convert("RGB")
    w, h = img.size
    arr = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)  # H W C
    t = arr.permute(2, 0, 1).contiguous()  # C H W

    if new_h < new_w and h > w:
        # 官方写的是 rearrange(img, "c h w -> c w h")，einops 这个模式就是
        # 交换最后两维，等价于 t.transpose(1, 2)：
        #     t[c, v, u]  ->  (3, W, H)
        # 早期版本写成 permute(1,2,0) 再 permute(2,0,1)，两次互逆 → 等于没转，
        # 竖图会被直接横向压扁到 294×518（实测水平压缩 2.3 倍、与官方差 0.97）。
        # 且 load_gt_cameras 是按「已转置」推导 K 与外参的，这里不转会与相机矛盾。
        t = t.transpose(1, 2).contiguous()

    t = t.unsqueeze(0)
    if crop:
        from torchvision.transforms.functional import center_crop, resize
        scale = max(new_h / t.shape[-2], new_w / t.shape[-1])
        t = resize(t, (round(t.shape[-2] * scale), round(t.shape[-1] * scale)),
                   antialias=True)
        t = center_crop(t, (new_h, new_w))
    else:
        t = F.interpolate(t, (new_h, new_w), mode="bilinear",
                          align_corners=True, antialias=True)
    return t.squeeze(0), (w, h)


# ============================================================
# GT 相机（transforms.json）→ StructSplat 需要的 w2c + K
# ============================================================
def load_gt_cameras(transforms_path, image_paths, new_h, new_w):
    """
    从 Remy/Huawei 风格 transforms.json 读 c2w 与内参。

    ⚠️ 竖图转置的完整处理（易错，错在左乘/右乘就完全失效）：
      load_image_tensor 对竖图做的是 rearrange("c h w -> c w h")，即
          t[c, i, j] = img[c, j, i]
      新图 (v'=i, u'=j) 采样原图 (v=j, u=i)，所以
          u' = v,  v' = u
      代入投影式：
          原:  u = fx*x/z + cx ,  v = fy*y/z + cy
          新:  u' = v = fy*y/z + cy   →  K'_00=fy, K'_02=cy，用相机 y
               v' = u = fx*x/z + cx   →  K'_11=fx, K'_12=cx，用相机 x
      即新相机坐标 (x', y', z') = (y, x, z) = M·Xc，M=[[0,1,0],[1,0,0],[0,0,1]]。

      ★ 关键是 M 作用在**相机坐标**上，不是世界坐标：
          X'c = M·Rᵀ(Xw − t)  ⟹  R' = R·M（右乘），t 不变
      因为 (R·M)ᵀ = Mᵀ·Rᵀ = M·Rᵀ（M 对称）。
      若误写成 c2w 左乘 M（即把世界 X/Y 轴交换），相机会整体转错方向，
      实测投影误差可达数千像素——比不修还差。左乘是错的，必须右乘。
    """
    with open(transforms_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    frames = meta.get("frames", [])
    if not frames:
        raise RuntimeError(f"{transforms_path} 里没有 frames")

    by_base = {}
    for fr in frames:
        fp = fr.get("file_path") or fr.get("filePath")
        if fp:
            by_base[os.path.basename(fp)] = fr

    # 转置坐标系下的轴交换矩阵：X<->Y
    M = torch.tensor([[0.0, 1.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0]], dtype=torch.float32)

    c2ws, Ks, matched = [], [], []
    for p in image_paths:
        fr = by_base.get(os.path.basename(p))
        if fr is None:
            continue
        W0, H0 = fr["w"], fr["h"]
        fx, fy = fr["fl_x"], fr["fl_y"]
        cx, cy = fr["cx"], fr["cy"]
        # 是否会被 load_image_tensor 转置：目标 H<=W，而原图是竖的
        transposed = (new_h < new_w) and (H0 > W0)
        if transposed:
            # 转置后未 resize 的图尺寸 = (H=W0, W=H0)
            # u' 方向源自原 v（尺度 fy/cy，在原 H0 像素上）→ 按 new_w/H0
            # v' 方向源自原 u（尺度 fx/cx，在原 W0 像素上）→ 按 new_h/W0
            s_u, s_v = new_w / H0, new_h / W0
            fx2, fy2 = fy * s_u, fx * s_v
            cx2, cy2 = cy * s_u, cx * s_v
        else:
            sx, sy = new_w / W0, new_h / H0
            fx2, fy2 = fx * sx, fy * sy
            cx2, cy2 = cx * sx, cy * sy
        K = torch.tensor([[fx2, 0.0, cx2],
                          [0.0, fy2, cy2],
                          [0.0, 0.0, 1.0]], dtype=torch.float32)
        c2w = torch.tensor(fr["transform_matrix"], dtype=torch.float32)
        if transposed:
            # 轴交换发生在相机系 → 对 c2w 的旋转右乘 M（见上面推导）
            c2w = c2w.clone()
            c2w[:3, :3] = c2w[:3, :3] @ M
        c2ws.append(c2w)
        Ks.append(K)
        matched.append(os.path.basename(p))

    if not c2ws:
        raise RuntimeError(
            "transforms.json 里没有一张图能和给定图片对上（检查 file_path 文件名）"
        )

    c2ws = torch.stack(c2ws)          # (N,4,4)
    Ks = torch.stack(Ks)              # (N,3,3)
    w2cs = torch.linalg.inv(c2ws)     # (N,4,4) OpenCV world->camera
    return w2cs, Ks, matched


# ============================================================
# 链路自检：用预测相机重渲染输入视角，与输入图算 PSNR/SSIM
# ============================================================
def evaluate_src_views(rendered, src, render_h, render_w):
    """
    为什么要这一步：
      高斯是从 (depth, 预测外参, 预测内参) 反投影出来的，渲染再用同一组相机，
      所以「渲染输入视角」在自洽链路上应当高度接近输入图。这条判据能一次性
      抓住五类翻车：画面全黑/全白、相机整体错位、竖图转置方向错、
      内外参缩放不一致、K 的行列写反。

      经验量级（DL3DV 这种宽基线）：PSNR 20~30dB 属正常；
      掉到 10dB 以下基本等于「画面对不上」，优先查相机而不是查权重。

    注意：这不是论文的评测指标（论文报的是 held-out target 视角），
    只是本脚本的自检哨兵，不要拿去和论文数字比。
    """
    n = min(rendered.shape[0], src.shape[0])
    psnrs, ssims = [], []
    for i in range(n):
        gt = F.interpolate(src[i].unsqueeze(0), (render_h, render_w),
                           mode="bilinear", align_corners=False, antialias=True)[0]
        gt = gt.clamp(0, 1)
        pr = rendered[i].permute(2, 0, 1).clamp(0, 1)
        mse = float(((gt - pr) ** 2).mean())
        psnrs.append(float("inf") if mse <= 1e-12 else 10 * math.log10(1.0 / mse))
        try:
            from skimage.metrics import structural_similarity as _ssim
            a = gt.permute(1, 2, 0).cpu().numpy()
            b = pr.permute(1, 2, 0).cpu().numpy()
            ssims.append(float(_ssim(a, b, channel_axis=2, data_range=1.0)))
        except Exception:
            pass  # 没装 scikit-image 就只报 PSNR
    return psnrs, ssims


# ============================================================
# 导出
# ============================================================
def export_ply(path, xyz, scale, quat, opacity, rgb):
    """
    标准 3DGS PLY（log-scale / logit-opacity / SH DC），能被 SuperSplat、
    gsplat、3DGS viewer 直接打开。
    """
    from einops import rearrange  # noqa: F401  保持依赖一致性
    N = xyz.shape[0]
    xyz = xyz.detach().float().cpu().numpy()
    scale = scale.detach().float().cpu().numpy()          # 线性尺度
    quat = quat.detach().float().cpu().numpy()            # wxyz
    opacity = opacity.detach().float().cpu().numpy()      # [0,1]
    rgb = rgb.detach().float().cpu().numpy()              # [0,1]

    log_scale = np.log(np.clip(scale, 1e-12, None))
    op = np.clip(opacity, 1e-6, 1 - 1e-6)
    logit_op = np.log(op / (1 - op))

    # SH DC：color = 0.5 * C0 + SH_DC，C0 = 0.28209479177387814
    C0 = 0.28209479177387814
    sh_dc = (rgb - 0.5) / C0

    fields = [
        ("x", xyz[:, 0]), ("y", xyz[:, 1]), ("z", xyz[:, 2]),
        ("nx", np.zeros(N)), ("ny", np.zeros(N)), ("nz", np.zeros(N)),
        ("f_dc_0", sh_dc[:, 0]), ("f_dc_1", sh_dc[:, 1]), ("f_dc_2", sh_dc[:, 2]),
    ]
    for i in range(3):
        fields.append((f"f_rest_{i}", np.zeros(N)))
    fields += [
        ("opacity", logit_op),
        ("scale_0", log_scale[:, 0]), ("scale_1", log_scale[:, 1]), ("scale_2", log_scale[:, 2]),
        ("rot_0", quat[:, 0]), ("rot_1", quat[:, 1]),
        ("rot_2", quat[:, 2]), ("rot_3", quat[:, 3]),
    ]

    dtype = np.dtype([(n, "f4") for n, _ in fields])
    buf = np.zeros(N, dtype=dtype)
    for n, v in fields:
        buf[n] = v

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {N}\n"
        + "".join(f"property float {n}\n" for n, _ in fields)
        + "end_header\n"
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        buf.tofile(f)
    print(f"  💾 PLY: {path}  ({os.path.getsize(path)/1e6:.1f} MB, {N} 高斯)")


def export_splat(path, xyz, scale, rotation, opacity, color):
    """
    复用官方 src/callbacks/gaussian_evaluation.py 的 save_as_splat 逻辑
    （32 字节/高斯的紧凑格式，供 antimatter15/splat 类 viewer 用）。
    rotation 需为 wxyz 且已归一化；官方实现按 *127.5+127.5 编码到 uint8。
    """
    N = xyz.shape[0]
    xyz = xyz.detach().float().cpu().numpy()
    scale = scale.detach().float().cpu().numpy()
    rotation = rotation.detach().float().cpu().numpy()
    opacity = opacity.detach().float().cpu().numpy()
    color = color.detach().float().cpu().numpy()

    rgba = np.concatenate([color, opacity[:, None]], axis=1)
    rgba = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)
    rot_uint8 = (rotation * 127.5 + 127.5).clip(0, 255).astype(np.uint8)

    dt = np.dtype([
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("sx", "f4"), ("sy", "f4"), ("sz", "f4"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"), ("a", "u1"),
        ("rot_0", "u1"), ("rot_1", "u1"), ("rot_2", "u1"), ("rot_3", "u1"),
    ])
    buf = np.zeros(N, dtype=dt)
    buf["x"], buf["y"], buf["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    buf["sx"], buf["sy"], buf["sz"] = scale[:, 0], scale[:, 1], scale[:, 2]
    buf["r"], buf["g"], buf["b"], buf["a"] = rgba[:, 0], rgba[:, 1], rgba[:, 2], rgba[:, 3]
    buf["rot_0"], buf["rot_1"] = rot_uint8[:, 0], rot_uint8[:, 1]
    buf["rot_2"], buf["rot_3"] = rot_uint8[:, 2], rot_uint8[:, 3]

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    buf.tofile(path)
    print(f"  💾 SPLAT: {path}  ({os.path.getsize(path)/1e6:.1f} MB, {N} 高斯)")


# ============================================================
# 主流程
# ============================================================
def main():
    ap = argparse.ArgumentParser("StructSplat 通用推理（pose-free 前馈重建）")
    ap.add_argument("--images", required=True,
                    help="图片目录 / 单文件 / 逗号分隔列表 / glob")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--config", default=os.path.join(REPO, "config", "dl3dv.yaml"))
    ap.add_argument("--ckpt", default=None, help="默认读 config 里的 gaussian_evaluation_stage.ckpt")
    ap.add_argument("--resize", nargs=2, type=int, default=[294, 518],
                    help="网络输入尺寸 H W，必须能被 14 整除（默认 294 518）")
    ap.add_argument("--render-size", type=int, default=512,
                    help="渲染分辨率（正方形边长，0=保持网络输入尺寸）")
    ap.add_argument("--camera", choices=["pose-free", "gt"], default="pose-free")
    ap.add_argument("--transforms", default=None, help="--camera gt 时需要的 transforms.json")
    ap.add_argument("--max-images", type=int, default=None, help="最多用几张（均匀抽样）")
    ap.add_argument("--stride", type=int, default=None, help="按步长抽帧")
    ap.add_argument("--export-splat", action="store_true")
    ap.add_argument("--export-ply", action="store_true")
    ap.add_argument("--render-src", action="store_true",
                    help="同时渲染输入视角（默认只渲染，不区分 src/tar）")
    ap.add_argument("--white-bg", action="store_true", help="白底（默认黑底，与官方一致）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    new_h, new_w = args.resize
    assert new_h % 14 == 0 and new_w % 14 == 0, \
        f"--resize {new_h} {new_w} 必须都能被 14 整除（patch_size=14）"
    assert new_h <= new_w, "当前实现要求 H <= W"

    # ---------- 1. 收集图片 ----------
    paths = collect_images(args.images, args.max_images, args.stride)
    if len(paths) < 2:
        raise SystemExit(f"❌ 至少需要 2 张图，实际 {len(paths)} 张")
    print(f"📷 输入 {len(paths)} 张图")
    print(f"   首张: {paths[0]}")

    t0 = time.time()
    imgs, orig_sizes = [], []
    for p in paths:
        t, sz = load_image_tensor(p, new_h, new_w)
        imgs.append(t)
        orig_sizes.append(sz)
    src = torch.stack(imgs).unsqueeze(0).to(device)          # (1, S, 3, H, W)
    print(f"   resize → ({new_h}, {new_w})  耗时 {time.time()-t0:.1f}s")

    # ---------- 2. 建模型 ----------
    from src.config import load_configs
    from src.model.gaussian_wrapper import build_gaussian_wrapper

    cfg = load_configs(args.config, [])
    ckpt_path = args.ckpt or cfg.gaussian_evaluation_stage.ckpt
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"❌ ckpt 不存在: {ckpt_path}\n"
                         f"   先跑 bash structsplat/01_download_models.sh ss")

    # config 里 geo_encoder.path / sem_encoder.path 都是**相对仓根**的 ckpts/xxx。
    # build_gaussian_wrapper 内部会 VGGT.from_pretrained(那个路径)，缺了会抛
    # 一堆 huggingface 的 FileNotFoundError，看不出真正原因，所以先自查。
    #
    # ⚠️ 光查目录存在还不够（2026-11-19 实测踩坑）：huggingface_hub 的
    #    PyTorchModelHubMixin._from_pretrained 在 model_id 是**本地目录**时硬编码
    #    只读 model.safetensors：
    #        if os.path.isdir(model_id):
    #            model_file = os.path.join(model_id, constants.SAFETENSORS_SINGLE_FILE)
    #    只有从 hub 下载才 fallback 到 pytorch_model.bin。而 HF 上 facebook/VGGT-1B
    #    只提供 model.pt，于是目录存在也照样报：
    #        FileNotFoundError: ... ckpts/vggt/model.safetensors
    #    这个报错看起来像「路径写错了」，实际是格式不匹配，极易误判。
    _missing = []
    _enc = (("geo_encoder.path", cfg.module.geo_encoder.path, "model.safetensors"),
            ("sem_encoder.path", getattr(cfg.module.sem_encoder, "path", None),
             "model.safetensors"))
    for _key, _p, _need in _enc:
        if not _p:
            continue
        _abs = _p if os.path.isabs(_p) else os.path.join(REPO, _p)
        if not os.path.isdir(_abs):
            _missing.append(f"{_key} = {_p}  (→ {_abs})  目录不存在")
        elif not os.path.exists(os.path.join(_abs, _need)):
            _has_pt = os.path.exists(os.path.join(_abs, "model.pt"))
            _hint = ("  目录里只有 model.pt → 需要转成 safetensors" if _has_pt
                     else "")
            _missing.append(f"{_key} = {_p}  (→ {_abs})  缺 {_need}{_hint}")
    if _missing:
        raise SystemExit(
            "❌ 编码器权重缺失（config 里是相对仓根的路径）：\n  "
            + "\n  ".join(_missing)
            + "\n\n   VGGT 必须提供 model.safetensors（不是 model.pt），原因是\n"
              "   huggingface_hub 对本地目录分支硬编码只认 safetensors。\n"
              "   两个修法任选其一：\n"
              "     1) bash structsplat/01_download_models.sh vggt   ← 已内置自动转换\n"
              "     2) 手动转：python -c \"import torch;from safetensors.torch import save_file;"
              "save_file(torch.load('<dir>/model.pt',map_location='cpu',weights_only=True),"
              "'<dir>/model.safetensors')\"\n"
              "   或建软链（若权重池已备好）：\n"
              "     ln -sfn /mnt/d/wheel/structsplat_ms/VGGT-1B "
              f"{REPO}/ckpts/vggt"
        )

    print(f"📦 加载 ckpt: {ckpt_path}")
    param_dict = torch.load(ckpt_path, map_location="cpu")
    module_dict = {"gaussian_predictor": {}}
    for k, v in param_dict.items():
        if k.startswith("gaussian_predictor."):
            module_dict["gaussian_predictor"][k[len("gaussian_predictor."):]] = v

    t0 = time.time()
    wrapper = build_gaussian_wrapper(module_dict, cfg)
    wrapper = wrapper.to(device).eval()
    for p in wrapper.parameters():
        p.requires_grad = False
    print(f"   构建完成 {time.time()-t0:.1f}s")

    # ---------- 3. 前向：预测高斯 + 相机 ----------
    t0 = time.time()
    with torch.no_grad():
        preds = wrapper(src)          # ★ 直接走 forward，不走 splat()
    if device.type == "cuda":
        torch.cuda.synchronize()
    inf_ms = (time.time() - t0) * 1000
    print(f"🧠 前向完成 {inf_ms:.0f} ms")

    means = preds["coordinate"][0].float()     # (N,3)
    quats = preds["rotation"][0].float()       # (N,4) wxyz
    scales = preds["scale"][0].float()         # (N,3)
    opac = preds["opacity"][0].float()         # (N,)
    colors = preds["color"][0].float()         # (N,3)
    N = means.shape[0]
    print(f"   {N} 个高斯 / {len(paths)} 视角 = {N//len(paths)} per image "
          f"(期望 {cfg.module.gaussian_decoder.gaussians_per_pixel} × "
          f"{(new_h//14)*(new_w//14)} patches)")

    # ---------- 4. 相机 ----------
    from src.utils.utils_3d import pose_encoding_to_extri, pose_encoding_to_intri

    if args.camera == "gt":
        if not args.transforms:
            raise SystemExit("❌ --camera gt 需要 --transforms <transforms.json>")
        w2cs, Ks, matched = load_gt_cameras(args.transforms, paths, new_h, new_w)
        print(f"📐 GT 相机: 匹配 {len(matched)}/{len(paths)} 张")
        if len(matched) != len(paths):
            print("   ⚠️  有图没匹配上 GT 相机，请检查 transforms.json 的 file_path")
        w2cs = w2cs.to(device)
        Ks = Ks.to(device)
        # 渲染分辨率下的 K
        # render_size 指「宽」还是「高」？统一口径：按宽 = render_size 缩，高度按宽高比推。
        # 输入恒为 H<=W（294×518），宽是长边，这样不会把两侧裁掉。
        if args.render_size:
            s = args.render_size / new_w
            render_w = args.render_size
            render_h = int(round(args.render_size * new_h / new_w))
            Ks_r = Ks.clone()
            Ks_r[:, 0, :] *= s
            Ks_r[:, 1, :] *= s
        else:
            Ks_r = Ks
            render_h, render_w = new_h, new_w
    else:
        cam = preds["camera"]                       # (1,S,9)
        extrinsic = pose_encoding_to_extri(cam)     # (1,S,4,4)
        intrinsic = pose_encoding_to_intri(cam, (new_h, new_w))   # (1,S,3,3)
        w2cs = extrinsic[0].to(device)
        Ks = intrinsic[0].to(device)
        # 与 gt 分支同一口径：K 与画布同步按 render_size/new_w 缩放。
        # 旧写法是 render_h = render_w = render_size（正方形），输入 294×518 时
        # 宽被压成 512 而 K 仍按 512/294 放大 → 右侧约 43% 的画面被裁掉。
        if args.render_size:
            s = args.render_size / new_w
            render_w = args.render_size
            render_h = int(round(args.render_size * new_h / new_w))
            Ks_r = Ks.clone()
            Ks_r[:, 0, :] *= s
            Ks_r[:, 1, :] *= s
        else:
            Ks_r = Ks
            render_h, render_w = new_h, new_w
    print(f"   相机来源: {args.camera}  渲染 {render_h}×{render_w}")

    # ---------- 5. 渲染 ----------
    from gsplat import rasterization

    gs = cfg.gs_setting
    S = len(paths)
    rendered = []
    depths = []
    chunk = 6
    t0 = time.time()
    bg_val = 1.0 if args.white_bg else 0.0
    with torch.no_grad():
        for i in range(0, S, chunk):
            j = min(i + chunk, S)
            out, _alpha, info = rasterization(
                means=means, quats=quats, scales=scales,
                opacities=opac, colors=colors,
                viewmats=w2cs[i:j], Ks=Ks_r[i:j],
                width=render_w, height=render_h,
                render_mode="RGB+D",
                absgrad=getattr(gs, "absgrad", False),
                rasterize_mode=getattr(gs, "rasterize_mode", "classic"),
                packed=False,
                backgrounds=torch.full((j - i, 3), bg_val,
                                       dtype=torch.float32, device=device),
            )
            # gsplat 返回 (S, H, W, C)；render_mode="RGB+D" 时通道顺序是 RGBD
            rgb_i = out[..., :3]
            d_i = out[..., 3:4]
            rendered.append(rgb_i)
            depths.append(d_i)
    rendered = torch.cat(rendered, 0)     # (S,H,W,3)
    depths = torch.cat(depths, 0)         # (S,H,W,1)
    print(f"🎨 渲染 {S} 视角 耗时 {time.time()-t0:.1f}s")

    # ---------- 5b. 链路自检 ----------
    psnrs, ssims = evaluate_src_views(rendered, src[0], render_h, render_w)
    if psnrs:
        p_mean = float(np.mean(psnrs))
        p_min, p_max = float(np.min(psnrs)), float(np.max(psnrs))
        msg = f"🔎 自检（渲染输入视角 vs 输入图）PSNR 均值 {p_mean:.2f} dB  [{p_min:.2f}, {p_max:.2f}]"
        if ssims:
            msg += f"  SSIM 均值 {float(np.mean(ssims)):.4f}"
        print(msg)
        # 分层提示：别让「跑通了」被误读成「结果对」
        if p_mean < 10.0:
            print("   ⚠️  PSNR < 10dB，画面对不上。优先查：相机内外参缩放是否同源、"
                  "竖图转置方向、render_h/render_w 与 K 的缩放比")
        elif p_mean < 18.0:
            print("   ⚠️  PSNR 偏低。可能是输入视角过于离散（pose-free 对宽基线更敏感），"
                  "或该场景本身难；结合 renders/ 里逐张图判断")
        else:
            print("   ✅ 链路自洽（相机/反投影/渲染三者一致）")

    # ---------- 6. 落盘 ----------
    from torchvision.utils import save_image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vis_dir = os.path.join(args.out, "renders")
    dep_dir = os.path.join(args.out, "depths")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(dep_dir, exist_ok=True)
    for i in range(S):
        name = f"{i:03d}_{os.path.splitext(os.path.basename(paths[i]))[0]}.png"
        save_image(rendered[i].clamp(0, 1).permute(2, 0, 1), os.path.join(vis_dir, name))
        d = depths[i, ..., 0].float().cpu().numpy()
        dv = np.percentile(d[d > 0], 98) if (d > 0).any() else 1.0
        plt.imsave(os.path.join(dep_dir, name), d, cmap="inferno", vmin=0, vmax=max(dv, 1e-6))
    print(f"🖼️  渲染图 {S} 张 → {vis_dir}")

    # 对比图（原图 resize | 渲染 | 深度）
    n_show = min(S, 8)
    show_idx = np.linspace(0, S - 1, n_show).round().astype(int)
    fig, axes = plt.subplots(3, n_show, figsize=(2.6 * n_show, 8))
    if n_show == 1:
        axes = axes[:, None]
    for col, i in enumerate(show_idx):
        gt = F.interpolate(src[0, i].unsqueeze(0), (render_h, render_w),
                           mode="bilinear", align_corners=False, antialias=True)
        axes[0, col].imshow(gt[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy())
        axes[0, col].set_title(f"input {i}", fontsize=8)
        axes[1, col].imshow(rendered[i].clamp(0, 1).cpu().numpy())
        axes[1, col].set_title("rendered", fontsize=8)
        axes[2, col].imshow(depths[i, ..., 0].cpu().numpy(), cmap="inferno")
        axes[2, col].set_title("depth", fontsize=8)
        for r in range(3):
            axes[r, col].axis("off")
    plt.tight_layout()
    cmp_path = os.path.join(args.out, "compare.png")
    plt.savefig(cmp_path, dpi=110)
    plt.close()
    print(f"📊 对比图: {cmp_path}")

    # 导出
    if args.export_splat:
        export_splat(os.path.join(args.out, "gaussians.splat"),
                     means, scales, quats, opac, colors)
    if args.export_ply:
        export_ply(os.path.join(args.out, "gaussians.ply"),
                   means, scales, quats, opac, colors)

    # 元信息
    meta = {
        "repo": REPO,
        "config": args.config,
        "ckpt": ckpt_path,
        "n_images": S,
        "n_gaussians": int(N),
        "resize": [new_h, new_w],
        "render": [render_h, render_w],
        "camera": args.camera,
        "inference_ms": round(inf_ms, 1),
        "self_check": {
            "note": "渲染输入视角 vs 输入图，仅为本脚本的自检哨兵，非论文评测口径",
            "psnr_mean": round(float(np.mean(psnrs)), 3) if psnrs else None,
            "psnr_min": round(float(np.min(psnrs)), 3) if psnrs else None,
            "psnr_max": round(float(np.max(psnrs)), 3) if psnrs else None,
            "psnr_per_view": [round(p, 3) for p in psnrs],
            "ssim_mean": round(float(np.mean(ssims)), 5) if ssims else None,
        },
        "images": [os.path.basename(p) for p in paths],
        "orig_sizes": orig_sizes,
        "note": "pose-free 前向：model(src) 直接预测相机+高斯，未用 get_tar_camera（无需 src/tar 划分）",
    }
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"✅ 完成 → {args.out}")


if __name__ == "__main__":
    main()
