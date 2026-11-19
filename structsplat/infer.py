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

# 输出重定向到文件/管道时 Python 默认块缓冲（4~8KB），长任务会看起来像卡死。
# 2026-11-19 实测：02_run_inference.sh 输出到日志时，前向 3 分钟没有任何新行。
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

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


def probe_distortion(transforms_path, tol=1e-6):
    """
    从 transforms.json 读 OPENCV 畸变，返回去畸变三件套 (K0, dist, newK)；
    无畸变（或字段缺失）时返回 None。

    为什么需要（2026-11-19 实测）：华为 Remy 包 camera_model=OPENCV，
    k1=+0.0909 / k2=-0.2374 / p1=-0.00021 / p2=+0.00088 / k3=+0.1914。
    load_gt_cameras 只读 fx/fy/cx/cy，等于把畸变当 0，实测角点偏差
    16.7~20.1 px（1440 宽原图），缩到 518 宽后仍有 6.0 px —— 足以让
    几何自检把「畸变残差」误读成「相机估计不准」。

    为什么返回 newK：去畸变是重采样，画面的有效焦距会变。cv2 的
    getOptimalNewCameraMatrix 给出「去畸变后画面里真实存在的那一块」对应的
    新内参；用旧 K 配新图会整体错位。alpha=0 保留全部无黑边的源像素
    （本数据 roi=(0,0,1439,1919)，几乎无损），比 alpha=1（裁掉边缘）更适合重建。

    注意：dist 必须是 OpenCV 顺序 (k1,k2,p1,p2,k3)，而 transforms.json 里
    是命名字段 k1/k2/k3/p1/p2，顺序不同，直接拼会错位。
    """
    with open(transforms_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    frames = meta.get("frames", [])
    if not frames:
        return None

    def _dist_of(fr):
        need = ("k1", "k2", "p1", "p2")
        if not all(k in fr for k in need):
            return None
        d = [float(fr["k1"]), float(fr["k2"]),
             float(fr["p1"]), float(fr["p2"]),
             float(fr.get("k3", 0.0))]
        return d

    fr0 = frames[0]
    d0 = _dist_of(fr0)
    if d0 is None or max(abs(x) for x in d0) <= tol:
        return None

    # 所有帧的畸变与内参是否一致：不一致则逐帧算，这里只支持共享一套
    for fr in frames[1:]:
        d = _dist_of(fr)
        if d is None or any(abs(a - b) > tol for a, b in zip(d, d0)):
            print("   ⚠️  各帧畸变系数不一致，跳过去畸变（如需支持请改为逐帧解）")
            return None

    import cv2
    W0, H0 = int(fr0["w"]), int(fr0["h"])
    K0 = np.array([[fr0["fl_x"], 0.0, fr0["cx"]],
                   [0.0, fr0["fl_y"], fr0["cy"]],
                   [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.array(d0, dtype=np.float64)
    newK, roi = cv2.getOptimalNewCameraMatrix(K0, dist, (W0, H0), 0.0, (W0, H0))
    print(f"🔧 检测到 OPENCV 畸变 k1={d0[0]:+.4f} k2={d0[1]:+.4f} "
          f"k3={d0[4]:+.4f} → 去畸变 "
          f"(fx {newK[0,0]/K0[0,0]:.4f}x, cx {newK[0,2]-K0[0,2]:+.2f}px, roi={roi})")
    return K0, dist, newK


def load_image_tensor(path, new_h, new_w, crop=False, undist=None):
    """
    复刻 load_and_preprocess_images 单图逻辑：
      - to_tensor 归一到 [0,1]
      - 竖图且目标是横图时转置（rearrange c h w -> c w h）
      - crop=False 走 bilinear interpolate（不是 crop，别混）
    heic 走 pillow-heif 兜底。
    undist=(K0, dist, newK) 时先按 OPENCV 模型去畸变再走上面流程。
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
    if undist is not None:
        # 必须用 uint8 走 cv2：它以像素为单位做重采样，先归一到 [0,1] 再 undistort
        # 数值上等价但要来回转两次，且 interpolate 的边缘外推行为会变。
        import cv2
        K0, dist, newK = undist
        u8 = np.asarray(img, dtype=np.uint8)
        u8 = cv2.undistort(u8, K0, dist, None, newK)
        arr = torch.from_numpy(u8.astype(np.float32) / 255.0)
    else:
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
def load_gt_cameras(transforms_path, image_paths, new_h, new_w, undist=None):
    """
    从 Remy/Huawei 风格 transforms.json 读 c2w 与内参。

    undist=(K0, dist, newK)：去畸变后内参要用 newK 而不是原 fl_x/fl_y/cx/cy，
    因为 cv2.undistort 是重采样，有效焦距与主点都会变（本数据 fx ×1.0157、
    cx +1.58px）。若这里仍用原 K，去畸变后的图与相机就会整体错位。

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

    # 去畸变时用 newK 的内参，否则用 JSON 里的原始内参
    if undist is not None:
        _Kuse = undist[2]          # (3,3) np.float64
    else:
        _Kuse = None

    # 转置坐标系下的轴交换矩阵：X<->Y
    M = torch.tensor([[0.0, 1.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0]], dtype=torch.float32)
    # Remy/华为包的 transform_matrix 是 **OpenGL 相机系** c2w（y 上、z 朝后）。
    # 转成 OpenCV（y 下、z 朝前）要在**相机系**右乘 diag(1,-1,-1)：
    #     c2w_cv = c2w_gl @ diag(1,-1,-1,1)
    # 2026-11-20 用 pcd.ply 实测判定（8000 点投影，见 _tmp_view/chk_conv.py）：
    #     a) 原样当 CV           正深度   0%  画面内  0%   ✗
    #     b) Rc @ diag(1,-1,-1)  正深度 100%  画面内 86%   ✅  ← 唯一正确
    #     c) diag(1,-1,-1) @ Rc  正深度  20%  画面内  0%   ✗（左乘是错的）
    #     d) Rc^T 再取负 z        正深度  13%  画面内  0%   ✗
    # 与 som_human/01_convert_capture.py 的结论一致："M_cv = M @ FLIP"。
    # ⚠️ 之前这里只做了 @ M（x/y 交换），漏了 GL→CV，导致 GT 相机整体朝向错，
    #    view1/view3 的高斯全部落到相机背后（正深度 0% / 2.9%）。
    FLIP = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=torch.float32))

    c2ws, Ks, matched = [], [], []
    for p in image_paths:
        fr = by_base.get(os.path.basename(p))
        if fr is None:
            continue
        W0, H0 = fr["w"], fr["h"]
        if _Kuse is not None:
            fx, fy = float(_Kuse[0, 0]), float(_Kuse[1, 1])
            cx, cy = float(_Kuse[0, 2]), float(_Kuse[1, 2])
        else:
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
        # ① GL(相机 y 上/z 朝后) → CV(相机 y 下/z 朝前)：**相机系**右乘 FLIP。
        #    这一步与是否转置无关，必须先做（见上面 FLIP 的实测判定表）。
        c2w = c2w.clone()
        c2w[:3, :3] = c2w[:3, :3] @ FLIP[:3, :3]
        if transposed:
            # ② 竖图转置的轴交换同样发生在相机系 → 再右乘 M。
            #    GL→CV 与转置都是「相机系坐标变换」，可以连乘：
            #    顺序是先 FLIP（纠正约定）再 M（纠正转置），因为 M 描述的是
            #    转置后像素的 (u',v') 与相机 (x,y) 的对应关系。
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
# GT 相机 → 模型坐标系（Sim(3) 对齐）
# ============================================================
def _cam_centers(w2c):
    """相机中心 c = -R_w2cᵀ · t_w2c，形状 (S,3)"""
    Rw = w2c[:, :3, :3]
    tw = w2c[:, :3, 3]
    return -torch.einsum("sji,sj->si", Rw, tw)


def _cam_axes(w2c):
    """
    相机在世界系下的三个基向量（单位化），形状 (S,3,3)。

    w2c 的行向量就是相机基在世界系里的表示：Rw[0]=右、Rw[1]=下、Rw[2]=前(光轴)。
    这些是**纯方向量**，不含平移，也不像相机中心那样在窄基线轨迹上退化。

    ⚠️ 为什么要归一化：预测外参可能带轻微非正交/非单位（VGGT 头输出没严格约束），
       不归一化会把「长度偏差」混进旋转拟合。
    """
    Rw = w2c[:, :3, :3]
    return Rw / Rw.norm(dim=2, keepdim=True).clamp_min(1e-9)


def align_gt_to_model(pred_w2c, gt_w2c, gt_K=None, pred_K=None, iters=1,
                      mode="axes", reflect="auto"):
    """
    把 GT 相机从「采集包坐标系」搬到「模型坐标系」（估计一个 Sim(3)，允许反射）。

    ── 为什么必须做（2026-11-19 实测，GT 分支 PSNR 只有 6.41dB 的根因）──
    output2gaussian 里：
        coordinate = depth_to_world_coords_points(depth, extrinsic, intrinsic)
    这里 extrinsic 是**模型自己预测的**，所以高斯的绝对位置只跟预测相机自洽；
    换成 GT 相机去渲染，等于拿两套不相干的世界坐标系硬拼 → 整体错位。
    实测：pose-free 20.53dB，直接塞 GT 相机 6.41dB。

    ── 解什么 ──
    官方 get_tar_camera 用的是同一套数学（四元数均值定旋转 + 平移最小二乘定 s,t）：
        X_model = s·R·X_gt + t
    这里求出 (s,R,t) 后作用到 GT 的 w2c 上，让 GT 视角落进模型坐标系。

    ── ★ 旋转必须用「朝向」而不是「中心」（2026-11-20 实测）──
    旧实现用相机中心做 Kabsch。但本数据 4 帧是同一段轨迹上均匀抽的，4 个中心
    近乎共线 → 绕「连线轴」的旋转自由度完全不可观测，SVD 解是任意的。
    改用朝向（每台相机 3 轴 × N 台 = 3N 样本，比中心法多 3 倍，且朝向变化大）。

    ── ★★ 必须允许反射（det=-1），这是 90.001° 残差的真因（2026-11-20）──
    逐层判据（00t / 00u / 00v）：
        rel 旋转差（gauge 无关）  = 2.88 / 0.35 / 1.37°   ← 相对几何很好
        abs 旋转差                = 90.001 / 90.033 / 90.003 / 90.015°
        pred_w2c det = +1.000×4 ，gt_raw det = **-1.000×4**   ← 手性不一致
        D4 八变换 + 平移搜索全救不回（00s，最佳仍 k=0）
    根因链：
      1. 输入是竖拍 1440×1920，模型只吃 H<=W 的横图 → 图像被 transpose。
      2. 图像 transpose 在几何上就是**关于对角线的镜像**，把右手像素系变左手；
         要让 u'=fx'·x'/z+cx' 成立必须 x'=y、y'=x ⇒ 基变换 det=-1。
      3. 所以配对上转置图的 GT 相机必然是 det=-1 的**非正常正交阵**，
         `gt_raw det = -1.000` 是**正确的**，代码里乘 M 正是这个事实的体现。
      4. 而 pred 相机 det=+1（模型只输出正常旋转）→ 两者之间差一个**整体反射**。
         反射无法表示为任何旋转：Kabsch 只能在 det=+1 的流形边界停住，
         残差恰好 90°（trace=1 的旋转恰为 90°），且两种 mode 都停在同一个 90°。
    修法：不强制 det=+1，让 SVD 自选。数据真是反射关系时 R 自然 det=-1，
    此时 Rm = R_g·Rᵀ（(-1)×(-1)=+1）→ **对齐后 GT 相机变回正常旋转**，正合所需。
    实测（00v，本数据 4 帧）：
        proper  强制+1 : fit_resid 1.0476, abs 残差 90.000°
        reflect 自选   : fit_resid 0.0179, abs 残差 **1.308°**  ← 降 58 倍

    mode:
      "axes"   —— 朝向 Kabsch 定 R + 中心定 s,t   ★ 默认
      "center" —— 中心 Kabsch（旧实现，保留对照）
    reflect:
      "auto"  —— 检测到 gt/pred 手性不一致时自动允许反射  ★ 默认
      "on"    —— 总是允许
      "off"   —— 强制 det=+1（仅用于复现历史 90° 现象）

    返回 (aligned_w2c, info)
    """
    P = _cam_centers(pred_w2c).double()      # (S,3) 模型坐标系
    G = _cam_centers(gt_w2c).double()        # (S,3) 采集包坐标系
    Pc, Gc = P.mean(0, keepdim=True), G.mean(0, keepdim=True)

    # 1) 尺度：两组中心到各自质心的平均距离之比（X_model = s·R·X_gt + t 里的 s）
    dP = (P - Pc).norm(dim=1).mean()
    dG = (G - Gc).norm(dim=1).mean()
    # 中心几乎重合（相机没基线）→ 尺度退化，退化为 1
    s = float(dP / dG) if float(dG) > 1e-9 else 1.0

    # 2) 旋转
    if mode == "center":
        # 2a) 中心 Kabsch：用中心化后的点，SVD 求最优正交阵
        fit_src, fit_dst = (G - Gc) * s, P - Pc
    elif mode == "axes":
        # 2b) 朝向 Kabsch：A = GT 系方向，B = 模型系方向，求 R 使 R·a ≈ b。
        #     方向量对尺度不敏感，所以这里不乘 s。
        fit_src = _cam_axes(gt_w2c).reshape(-1, 3).double()
        fit_dst = _cam_axes(pred_w2c).reshape(-1, 3).double()
    else:
        raise ValueError(f"未知 mode: {mode}（可选 axes / center）")
    H = fit_src.transpose(0, 1) @ fit_dst

    U, _, Vt = torch.linalg.svd(H)
    # 手性判决：GT 相机 det 与 pred 相机 det 是否一致
    det_g = float(torch.linalg.det(gt_w2c[:, :3, :3].double()).mean())
    det_p = float(torch.linalg.det(pred_w2c[:, :3, :3].double()).mean())
    mismatched = (det_g < 0) != (det_p < 0)
    allow_reflect = (reflect == "on") or (reflect == "auto" and mismatched)

    R_raw = Vt.transpose(0, 1) @ U.transpose(0, 1)
    if allow_reflect:
        # 让 SVD 自选：数据是反射关系时 R 自然 det=-1
        R = R_raw
    else:
        # 防反射：det 必须 +1，否则是镜像（渲染出来左右颠倒且深度全负）
        d = torch.sign(torch.linalg.det(R_raw))
        # ⚠️ device 必须跟 H 一致：`torch.tensor([...])` 默认建在 CPU 上，
        #    而 P/G 来自 cuda 上的 w2c（double 只是 dtype，不改 device）→ 下一步 matmul 报
        #    "Expected all tensors to be on the same device, but got mat2 is on cpu"。
        D = torch.diag(torch.tensor([1.0, 1.0, float(d)],
                                    dtype=torch.float64, device=H.device))
        R = R_raw @ D
    fit_resid = float((fit_src @ R.transpose(0, 1) - fit_dst).norm(dim=1).mean())

    # ⚠️ t 必须用**原始**质心 Gc，不能用 (G-Gc).mean(0)（后者恒等于 0）。
    #    2026-11-19 踩过：写成 Pc - R@0 后 t 恒等于 Pc，identity 用例
    #    对齐后中心整体平移了 Pc，残差 0.4 而不是 0。
    t = Pc.squeeze(0) - s * (R @ Gc.squeeze(0))

    R = R.to(pred_w2c.dtype)
    t = t.to(pred_w2c.dtype)
    s = float(s)

    # 3) 作用到 w2c
    #
    # ⚠️ 这里不能用 `gt_w2c @ inv(Sim)`（2026-11-19 踩坑）：那个写法把尺度也乘进
    #    旋转块，得到 R_g·Rᵀ/s —— **非正交矩阵**，gsplat 会渲出错乱画面。
    #
    # 正确推导（关键：把相机空间整体乘 s 不改变投影，因为 u=fx·x/z）：
    #   世界点映射   X_m = s·R·X_g + t
    #   GT 相机      X_cam = R_g·X_g + t_g
    #   代入并整理   X_cam = (1/s)·R'(X_m − t) + t_g ,  R' = R_g·Rᵀ（正交✓）
    #   两边乘 s     s·X_cam = R'(X_m − t) + s·t_g
    #   ⇒ X_cam' ≡ R'·X_m + (s·t_g − R'·t) = s·X_cam
    #   s·X_cam 与 X_cam 的 x/z、y/z 完全相同 → 投影一致，且 R' 正交。
    Rm = gt_w2c[:, :3, :3] @ R.transpose(0, 1)              # (S,3,3) 正交
    tm = s * gt_w2c[:, :3, 3] - torch.einsum("sij,j->si", Rm, t)
    aligned = gt_w2c.clone()
    aligned[:, :3, :3] = Rm
    aligned[:, :3, 3] = tm

    # 4) 残差自检：对齐后 GT 中心与 pred 中心的 RMS 偏差（应为 0，除非 S<3 有歧义）
    resid = (_cam_centers(aligned) - _cam_centers(pred_w2c)).norm(dim=1)
    # 朝向残差：对齐后两套 w2c 的相对旋转角（度）。纯净判据 —— 与高斯、K 都无关。
    # 实测（本轮）：axes 法约 3°，center 法恒 90°。
    _Ra = aligned[:, :3, :3] @ pred_w2c[:, :3, :3].transpose(-1, -2)   # (S,3,3)
    _cos = ((_Ra.diagonal(dim1=1, dim2=2).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
    rot_deg = torch.rad2deg(torch.acos(_cos))
    info = {
        "mode": mode,
        "reflect": "on" if allow_reflect else "off",
        "det_gt": det_g,
        "det_pred": det_p,
        "det_R": float(torch.linalg.det(R.double())),
        "scale_gt_to_model": float(s),
        "fit_resid": fit_resid,
        "center_rms": float(resid.mean()),
        "center_max": float(resid.max()),
        "rot_deg_mean": float(rot_deg.mean()),
        "rot_deg_max": float(rot_deg.max()),
        "rot_deg_each": [round(float(x), 3) for x in rot_deg],
        # ★ 把 (R,t) 一并交出去：target 视角必须复用**仅在 src 上估计**的同一个变换，
        #   若拿 target 参与重估 = GT 泄漏，指标会虚高（DA3 项目踩过这个坑）。
        #   s 已在 scale_gt_to_model 里，这里是剩下两件。
        "_R": R.detach().clone(),
        "_t": t.detach().clone(),
    }
    return aligned, info


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
                    help="网络输入尺寸 H W，必须能被 14 整除（默认 294 518 = DL3DV 口径）")
    ap.add_argument("--no-auto-aspect", dest="auto_aspect", action="store_false",
                    help="关掉按原图宽高比重算 H（默认开启）。"
                         "关闭后严格用 --resize 的 H，竖拍图会被拉伸 —— 只在复现"
                         "官方 DL3DV 口径的对照实验里才关")
    ap.add_argument("--render-size", type=int, default=512,
                    help="渲染分辨率（正方形边长，0=保持网络输入尺寸）")
    ap.add_argument("--camera", choices=["pose-free", "gt"], default="pose-free")
    ap.add_argument("--transforms", default=None, help="--camera gt 时需要的 transforms.json")
    ap.add_argument("--align-mode", choices=["axes", "center"], default="axes",
                    help="GT→模型系 Sim(3) 的旋转估计方式。axes=用相机朝向做 Kabsch（默认）；"
                         "center=用相机中心做 Kabsch（旧实现，窄基线轨迹上欠定）")
    ap.add_argument("--align-reflect", choices=["auto", "on", "off"], default="auto",
                    help="对齐时是否允许反射(det=-1)。图像转置本身是镜像，竖图输入下 GT 相机"
                         "det=-1 而 pred det=+1，强制 det=+1 会让残差卡在 90°。"
                         "auto=手性不一致时自动允许（默认）；off=复现历史 90° 现象")
    ap.add_argument("--no-undistort", action="store_true",
                    help="--camera gt 时不做去畸变（默认按 transforms.json 的 "
                         "k1/k2/p1/p2/k3 去畸变，仅当模型为 OPENCV 且有非零系数时生效）")
    ap.add_argument("--max-images", type=int, default=None, help="最多用几张（均匀抽样）")
    ap.add_argument("--stride", type=int, default=None, help="按步长抽帧")
    ap.add_argument("--export-splat", action="store_true")
    ap.add_argument("--export-ply", action="store_true")
    ap.add_argument("--render-src", action="store_true",
                    help="同时渲染输入视角（默认只渲染，不区分 src/tar）")
    ap.add_argument("--white-bg", action="store_true", help="白底（默认黑底，与官方一致）")
    ap.add_argument("--dump-forward", default=None,
                    help="把前向产出的高斯/相机/输入缓存成 .pt（调试对齐时省掉重跑前馈）")
    ap.add_argument("--from-forward", default=None,
                    help="从 --dump-forward 存下的 .pt 直接装载高斯/相机，跳过前馈与模型构建")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 先把用户给的路径绝对化，再 chdir（否则 chdir 后相对路径全废）
    args.out = os.path.abspath(args.out)
    if args.transforms:
        args.transforms = os.path.abspath(args.transforms)
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

    # ---------- 1b. 按真实宽高比重算 H（关键，别跳过）----------
    #
    # 为什么必须做（2026-11-20 实测，GT 分支只有 7.26dB 的第二个根因）：
    #   官方 config 写的是 new_size: [294, 518]，那是 **DL3DV 的宽高比**
    #   （294/518 = 0.568，接近 16:9 横图）。直接拿来套竖拍手机图会出事：
    #       原图 1440×1920 → 转置后 1920×1440，真实宽高比 0.75
    #       硬 resize 到 294×518（0.568）  ⇒ 水平方向被**拉伸 32%**
    #   而网络的几何头假设方形像素（预测内参 fx/fy = 0.9992），
    #   拉伸后真实内参变成 fx/fy = 1.3246 —— **各向异性畸变**。
    #   相似变换是各向同性的，吸收不了这个差异，所以 Sim(3) 对齐再怎么修，
    #   GT 分支都只能到 7.4dB，而 pose-free 自洽链路有 17.9dB。
    #
    #   修法：宽度取目标值，高度按「转置后」的真实宽高比推，再对齐到 14 的倍数。
    #   1440×1920 → 392×518（392/518 = 0.757，误差 0.9%），各向异性基本消失。
    if args.auto_aspect:
        from PIL import Image as _Image
        _p = paths[0]
        _ext = os.path.splitext(_p)[1].lower()
        if _ext in (".heic", ".heif"):
            try:
                import pillow_heif
                pillow_heif.register_heif_opener()
            except Exception:
                pass
        _w0, _h0 = _Image.open(_p).size
        # load_image_tensor 会在「竖图 + 目标横图」时转置，转置后 H/W = _w0/_h0
        _transposed = (_h0 > _w0)
        _asp = (_w0 / _h0) if _transposed else (_h0 / _w0)
        _h = int(round(args.resize[1] * _asp / 14.0) * 14)
        if _h < 14:
            _h = 14
        if _h > args.resize[1]:
            _h = args.resize[1]        # 不越界：官方要求 H <= W
        print(f"   📐 自动宽高比: 原图 {_w0}×{_h0}"
              f"{' (转置)' if _transposed else ''}  aspect(H/W)={_asp:.4f}"
              f"  → resize {_h}×{args.resize[1]}"
              f"   各向异性误差 {abs(_h/(args.resize[1]*_asp) - 1)*100:.2f}%")
        new_h, new_w = _h, args.resize[1]

    # 畸变：只有 GT 分支能拿到变换文件，pose-free 分支无从得知 → 只在 gt 且
    # 给了 --transforms 时探。--no-undistort 可强制关掉（用于对照实验）。
    undist = None
    if args.camera == "gt" and args.transforms and not args.no_undistort:
        undist = probe_distortion(args.transforms)

    t0 = time.time()
    imgs, orig_sizes = [], []
    for p in paths:
        t, sz = load_image_tensor(p, new_h, new_w, undist=undist)
        imgs.append(t)
        orig_sizes.append(sz)
    src = torch.stack(imgs).unsqueeze(0).to(device)          # (1, S, 3, H, W)
    print(f"   resize → ({new_h}, {new_w})  耗时 {time.time()-t0:.1f}s")

    # ---------- 2. 建模型 ----------
    # ⚠️ 建模型前必须切到仓根（2026-11-19 实测）：config 里 geo_encoder.path /
    #    sem_encoder.path / ckpt 都是**相对仓根**的字面量，但下游是
    #    AutoBackbone.from_pretrained / VGGT.from_pretrained / os.path.exists
    #    按 **cwd** 解析的。在 structsplat/ 下直接跑 infer.py 会分别报
    #    「is not a local folder and is not a valid model identifier」和
    #    「ckpt 不存在」，看着像权重坏了，其实只是 cwd 不对。
    #    02_run_inference.sh 里有 cd，直接调用本脚本时没有 → 这里兜住。
    #    用户路径（--out/--images/--transforms）在上面已经绝对化，chdir 安全。
    os.chdir(REPO)

    from src.config import load_configs

    cfg = load_configs(args.config, [])

    # ---------- 2. 高斯来源：前向缓存 or 现场前馈 ----------
    # ⚠️ inf_ms 只在现场前馈分支里赋值；--from-forward 时没有前馈，
    #    旧代码在写 meta 时直接 round(inf_ms, 1) → UnboundLocalError。
    #    先置 None，meta 里判空。
    inf_ms = None
    if args.from_forward:
        # 为什么有这条路（2026-11-20）：一次前向在 3090 上 3~4 分钟，
        # 而调相机对齐/评测口径时高斯本身完全不变。缓存装载后可秒级迭代。
        print(f"📥 装载前向缓存: {args.from_forward}")
        _d = torch.load(args.from_forward, map_location="cpu", weights_only=False)
        _cached_paths = _d.get("paths", [])
        if _cached_paths and [os.path.basename(p) for p in _cached_paths] != \
                [os.path.basename(p) for p in paths]:
            raise SystemExit(
                "❌ 缓存里的图与 --images 选出的图不一致：\n"
                f"   缓存: {[os.path.basename(p) for p in _cached_paths]}\n"
                f"   当前: {[os.path.basename(p) for p in paths]}\n"
                "   缓存的相机与高斯绑定同一批图，换图必须重跑前馈。"
            )
        new_h, new_w = int(_d["new_h"]), int(_d["new_w"])
        means = _d["means"].to(device)
        quats = _d["quats"].to(device)
        scales = _d["scales"].to(device)
        opac = _d["opac"].to(device)
        colors = _d["colors"].to(device)
        src = _d["src"].to(device)
        _pred_w2c = _d.get("pred_w2c")
        _pred_K = _d.get("pred_K")
        if _pred_w2c is None:
            raise SystemExit("❌ 缓存缺 pred_w2c：该缓存是旧版 dump，请重跑一次 --dump-forward")
        # 缓存是按 .cpu() 存的，装回来必须搬回 device，否则 align_gt_to_model 里
        # Gs.transpose @ (P - Pc) 会报 device 不一致（2026-11-20 实测）。
        _pred_w2c = _pred_w2c.to(device)
        _pred_K = _pred_K.to(device)
        N = means.shape[0]
        print(f"   {N} 个高斯 / {len(paths)} 视角；输入 {new_h}×{new_w}（来自缓存）")
    else:
        from src.model.gaussian_wrapper import build_gaussian_wrapper

        # config 里 ckpt 是**相对仓根**的 "ckpts/structsplat/pytorch_model.bin"，
        # 不是相对 cwd（2026-11-19 实测：在 structsplat/ 下跑会误报 ckpt 不存在）。
        ckpt_path = args.ckpt or cfg.gaussian_evaluation_stage.ckpt
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(REPO, ckpt_path)
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

        _t = time.time()
        wrapper = build_gaussian_wrapper(module_dict, cfg)
        wrapper = wrapper.to(device).eval()
        for p in wrapper.parameters():
            p.requires_grad = False
        print(f"   构建完成 {time.time()-_t:.1f}s")

        # ---------- 3. 前向：预测高斯 + 相机 ----------
        _t = time.time()
        with torch.no_grad():
            preds = wrapper(src)          # ★ 直接走 forward，不走 splat()
        if device.type == "cuda":
            torch.cuda.synchronize()
        inf_ms = (time.time() - _t) * 1000
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

        from src.utils.utils_3d import pose_encoding_to_extri, pose_encoding_to_intri
        _pred_w2c = pose_encoding_to_extri(preds["camera"])[0].to(device)
        _pred_K = pose_encoding_to_intri(preds["camera"], (new_h, new_w))[0].to(device)

        # ---------- 3b. 缓存前向结果（调试对齐用）----------
        # 为什么需要：一次前向在 3090 上要 3~4 分钟，而调对齐/评测参数常常只需要
        # 「同一批高斯 + 同一组相机」反复渲。把 tensors 落盘后，离线脚本可秒级重试，
        # 不必每次重跑 1.5B 参数的前馈。
        if args.dump_forward:
            torch.save({
                "means": means.cpu(), "quats": quats.cpu(), "scales": scales.cpu(),
                "opac": opac.cpu(), "colors": colors.cpu(),
                "src": src.cpu(),
                "new_h": new_h, "new_w": new_w,
                "paths": paths,
                "pred_w2c": _pred_w2c.cpu(), "pred_K": _pred_K.cpu(),
            }, args.dump_forward)
            print(f"💾 前向缓存 → {args.dump_forward}")

    # ---------- 4. 相机 ----------
    # 预测相机无论如何都要拿到：GT 分支要靠它把 GT 搬进模型坐标系（见 align_gt_to_model）。
    pred_ext = _pred_w2c
    pred_int = _pred_K

    if args.camera == "gt":
        if not args.transforms:
            raise SystemExit("❌ --camera gt 需要 --transforms <transforms.json>")
        w2cs_raw, Ks, matched = load_gt_cameras(args.transforms, paths, new_h, new_w,
                                                undist=undist)
        print(f"📐 GT 相机: 匹配 {len(matched)}/{len(paths)} 张")
        if len(matched) != len(paths):
            print("   ⚠️  有图没匹配上 GT 相机，请检查 transforms.json 的 file_path")
        w2cs_raw = w2cs_raw.to(device)
        Ks = Ks.to(device)

        # ★ GT 相机必须搬进模型坐标系，否则整体错位（2026-11-19 实测 6.41dB 的根因）。
        #   output2gaussian 用**预测**外参反投影出高斯，高斯只跟预测的世界系自洽；
        #   transforms.json 的 c2w 在采集包自己的世界系里，两套坐标系差一个 Sim(3)。
        w2cs, align_info = align_gt_to_model(pred_ext, w2cs_raw, Ks, pred_int,
                                             mode=args.align_mode,
                                             reflect=args.align_reflect)
        print(f"🔗 GT→模型系 Sim(3) [{align_info['mode']}/reflect={align_info['reflect']}]: "
              f"尺度 {align_info['scale_gt_to_model']:.4f}  "
              f"拟合残差 {align_info['fit_resid']:.4f}  "
              f"det(gt)={align_info['det_gt']:+.2f} det(pred)={align_info['det_pred']:+.2f} "
              f"det(R)={align_info['det_R']:+.2f}")
        print(f"   朝向残差 {align_info['rot_deg_mean']:.2f}° (max {align_info['rot_deg_max']:.2f}°) "
              f"逐视角 {align_info['rot_deg_each']}")
        print(f"   中心残差 RMS {align_info['center_rms']:.4f} / max {align_info['center_max']:.4f}")
        if align_info["rot_deg_max"] > 5.0:
            print("   ⚠️  朝向残差 >5°：Sim(3) 没真正闭合。先看 det(gt)/det(pred) 是否异号"
                  "（竖图转置会引入镜像，需 --align-reflect auto/on），再查 FLIP 约定")
        if align_info["center_max"] > 0.05:
            print("   ⚠️  相机中心对齐残差偏大：输入视角太少或近乎共视，"
                  "Sim(3) 欠定（S<3 时旋转/尺度没有冗余可摊）")
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
        w2cs = pred_ext
        Ks = pred_int
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

    # 相机一并缓存：离线对齐实验要同时拿到「预测相机」和「GT 相机原始值」。
    if args.dump_forward:
        _d = torch.load(args.dump_forward, map_location="cpu")
        _d.update({
            "w2cs": w2cs.cpu(), "Ks": Ks.cpu(),
            "render_h": render_h, "render_w": render_w,
        })
        if args.camera == "gt":
            _d["gt_w2c_raw"] = w2cs_raw.cpu()
        torch.save(_d, args.dump_forward)
        print(f"💾 相机已并入缓存 → {args.dump_forward}")

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
        "ckpt": (args.from_forward if args.from_forward else ckpt_path),
        "from_forward": args.from_forward,
        "n_images": S,
        "n_gaussians": int(N),
        "resize": [new_h, new_w],
        "render": [render_h, render_w],
        "camera": args.camera,
        "undistort": (None if undist is None else {
            "dist_cv_order": [float(x) for x in undist[1]],
            "K_src": undist[0].tolist(),
            "K_new": undist[2].tolist(),
        }),
            "inference_ms": round(inf_ms, 1) if inf_ms is not None else None,
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
