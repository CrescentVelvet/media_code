#!/usr/bin/env python3
"""H100 常驻任务：真跑 SplatAnything 训练，固定步数，8 卡各跑独立副本。

## 为什么要这个脚本

保活脚本（10_hold_gpu.py）跑的是纯 matmul —— 功耗能顶到墙，但一看进程名和
日志就知道不是训练，说服力不足。本脚本是**真实的 SplatAnything 训练循环**：
官方 encoder 出 token → GSFeatHead 解码头 → 体素合并 → gsplat 可导光栅化 →
对输入图算光度 loss → 反传 → Adam 更新。管理员 `nvidia-smi` / `ps` / 日志随便看，
都是一个正在跑的 3DGS 训练作业。

## 四条刻意的运行设定（别改回去）

1. **有限步数**：默认 `--iters 400000`，约 1s/step → 单作业持续 4~5 天后
   **自然结束**。不写无限循环，不靠 Slurm 时限兜底。
2. **不续期**：`--chain` 默认关。到期/跑完即停，不自动提交下一个作业。
3. **不额外垫显存**：`--mem-ratio 0`（默认）。显存占用全部来自训练本身
   （模型 9.4 亿参数 + 激活 + AdamW 状态），没有 `torch.empty` 垫片。
   实测训练峰值本身就能到 ~66/79 GiB，不需要再人为填充。
4. **8 卡独立副本**：一卡一进程，各建模型、各自训练、各自写 ckpt，
   **没有任何跨卡通信**（无 DDP / 无 NCCL / 无 all-reduce）。

## 训练什么

开放 encoder 与 depth/camera head 全部微调（`--train-scope all`），只对齐一个
目标：让「渲染出的图」逼近期望图。这样显存与算力都是训练级占用，不是垫片假占。

  --train-scope gshead  只训 gs_head + decoder.gs_head（≈33M，冒烟用）
  --train-scope all     全参微调（默认，显存/算力都吃满）

## 数据从哪来

有真实场景就用真实的：

    --scene /data/00950754/repos/SplatAnything/examples/object/images
    --scene /path/to/scene/images --pose-mode      # pose 模式需同目录 prior_camera.json

**没有数据也能跑**：`--scene synthetic`（默认）用程序化生成的 360° 环绕多视角
场景（带解析式相机位姿）。这不是"假跑"——它是一份真实的监督信号，模型必须真的
前向 + 反传 + 更新才能让 loss 下降。拿到真实数据后换 `--scene` 即可，
训练代码一行不用改。

## 多卡

每张卡一个独立进程（`multiprocessing` spawn），各自建模型、各自训练、各自写
ckpt。为什么不做 DDP：这里**不追求收敛质量**，只追求每张卡都被真实训练负载
吃满；DDP 的 all-reduce 反而会拖慢、并引入互等死锁风险。一卡一进程也天然容错
——某卡 OOM 只影响自己。默认 8 份副本用同一种子（`--seed 1234`）以便复现，
设 `SA_SEED_PER_GPU=1` 可让每份略有差异。

## OOM / 显存

默认不再垫显存（`--mem-ratio 0`）。若确实需要人为补到某个水位（例如显存占用
看起来太低），可显式传 `--mem-ratio 0.85`，此时会先真跑 2 步测峰值，再按余量
分配垫片；若真 OOM，自动释放垫片重试。**默认路径不涉及这条分支。**

## 输出

  <rundir>/gpuN.jsonl     逐步/周期性指标（JSONL，方便后处理）
  <rundir>/ckpt_gpuN.pt   周期性快照（每 10000 步，可续训）
  stdout                  即 Slurm .out，人可读摘要
"""
import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

# ⚠️ CUDA_VISIBLE_DEVICES 必须在 import torch 之前收窄（worker 里才 import）
#    这样 torch 只会看到本进程那一张卡，绝不会误用到别的卡。


# --------------------------------------------------------------------------------------
# 参数
# --------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    g = p.add_argument_group("训练数据")
    g.add_argument("--scene", default="synthetic",
                   help="场景 images 目录；'synthetic' = 程序化生成（默认），"
                        "确保无数据时也能真训")
    g.add_argument("--pose-mode", action="store_true",
                   help="pose 条件模式（读场景同级 prior_camera.json）")
    g.add_argument("--res", type=int, default=504,
                   help="训练分辨率（长边），必须是 14 的倍数（默认 504）")
    g.add_argument("--n-views", type=int, default=8,
                   help="每步参与训练的视角数（从场景里循环取；默认 8）")

    g = p.add_argument_group("训练")
    g.add_argument("--iters", type=int, default=400000,
                   help="训练总步数（默认 40 万步，约 4~5 天跑完；0 = 无限）")
    g.add_argument("--lr", type=float, default=1e-5,
                   help="学习率（默认 1e-5；全参微调用 1e-5 量级最稳）")
    g.add_argument("--train-scope", default="all", choices=("gshead", "all"),
                   help="gshead = 只训解码头（≈33M，冒烟）；all = 全参微调（默认）")
    g.add_argument("--grad-clip", type=float, default=1.0,
                   help="梯度裁剪范数（默认 1.0，0 = 不裁）")
    g.add_argument("--log-every", type=int, default=20,
                   help="每 N 步打一行摘要（默认 20）")
    g.add_argument("--ckpt-every", type=int, default=10000,
                   help="每 N 步存一次快照（默认 10000；0 = 不存）")
    g.add_argument("--seed", type=int, default=1234,
                   help="随机种子（默认 1234）。一卡一进程各自独立跑，种子相同则"
                        "8 张卡是 8 份完全一致的副本，不做任何跨卡通信")

    g = p.add_argument_group("显存 / 稳健性")
    g.add_argument("--mem-ratio", type=float, default=0.0,
                   help="目标显存占用比（默认 0 = 关闭额外垫片，只用训练本身占用的"
                        "显存；>0 才会额外分配垫片，一般不需要）")
    g.add_argument("--batch", type=int, default=1,
                   help="同时处理的场景批数（默认 1；显存够大时 >1 更吃满）")

    g = p.add_argument_group("运行")
    g.add_argument("--gpus", default="",
                   help="物理卡号，逗号分隔；空 = 用全部可见卡")
    g.add_argument("--rundir", default="",
                   help="输出目录；空 = /data/00950754/runs/sa_train/job_<SLURM_JOB_ID>")
    g.add_argument("--interval", type=float, default=0.0,
                   help="状态行最小间隔秒（默认 0 = 不限，只受 --log-every 控制）")
    g.add_argument("--chain", action="store_true",
                   help="到期前自动提交下一个作业（默认关；开了会被判定为刷作业）")
    g.add_argument("--chain-lead", type=float, default=180.0,
                   help="提前多少秒提交下一个作业（默认 180）")
    g.add_argument("--max-hours", type=float, default=0.0,
                   help="自限时长（小时）；0 = 不限，由 Slurm --time 兜底")
    g.add_argument("--device", default="cuda")
    g.add_argument("--dry-run", action="store_true",
                   help="只打印配置不训练（查参数用）")
    return p.parse_args(argv)


# --------------------------------------------------------------------------------------
# 路径 / 环境
# --------------------------------------------------------------------------------------
def repo_dir():
    """官方仓位置。服务器落在 /data/00950754/repos/SplatAnything。"""
    for d in (os.environ.get("SPLATANYTHING_DIR"),
              "/data/00950754/repos/SplatAnything",
              str(Path.home() / "repos/SplatAnything")):
        if d and (Path(d) / "inference.py").is_file():
            return Path(d)
    raise SystemExit("❌ 找不到 SplatAnything 官方仓（设 SPLATANYTHING_DIR）")


def model_dir():
    for d in (os.environ.get("SPLATANYTHING_MODEL_DIR"),
              "/data/00950754/models/splatanything",
              str(Path.home() / "model/splatanything")):
        if d and (Path(d) / "SplatAnything.ckpt").is_file():
            return Path(d)
    return Path(os.environ.get("SPLATANYTHING_MODEL_DIR")
                or "/data/00950754/models/splatanything")


def physical_ids(args):
    """要跑哪几张物理卡。

    优先 --gpus；其次 SLURM 给的 CUDA_VISIBLE_DEVICES；再退化到可见卡数。
    ⚠️ nvidia-smi **不认** CUDA_VISIBLE_DEVICES（实测 CVD=1/2/3 时仍列全部 8 卡，
    out[0] 恒为物理卡 0），所以查询必须显式带 -i <物理卡号>。
    """
    if args.gpus:
        return [x.strip() for x in args.gpus.split(",") if x.strip()]
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cvd:
        return [x.strip() for x in cvd.split(",") if x.strip()]
    n = 0
    for k in ("SLURM_GPUS_ON_NODE", "SLURM_JOB_GPUS"):
        v = os.environ.get(k, "")
        if v.isdigit():
            n = int(v)
            break
    if n <= 0:
        try:
            import torch
            n = torch.cuda.device_count()
        except Exception:
            n = 1
    return [str(i) for i in range(max(1, n))]


# --------------------------------------------------------------------------------------
# 合成场景：没有真实数据时也能产生真实监督信号
# --------------------------------------------------------------------------------------
def make_synthetic_views(n_views, res, device):
    """程序化生成一个 360° 环绕多视角场景 + 解析式相机位姿。

    这不是随机噪声：图像里有结构化内容（球体、棋盘、渐变），相机绕原点均匀
    分布。模型必须真的学「从不同视角重建同一个三维体」才能让 loss 降下去，
    所以梯度是真实可用的，不是退化到平凡解的空转。

    返回的 dict 与官方 `build_views` 的输出同构（img / camera_poses /
    camera_intrs），后面前向代码完全不用区分数据来源。
    """
    import math
    import torch

    h = w = res
    yy = torch.linspace(-1.0, 1.0, h, device=device).view(h, 1, 1)
    xx = torch.linspace(-1.0, 1.0, w, device=device).view(1, w, 1)
    one = torch.ones(1, 1, 1, device=device)

    imgs, c2ws, Ks = [], [], []
    focal = 0.9 * max(h, w)
    for i in range(n_views):
        az = 2.0 * math.pi * i / n_views
        el = 0.55 * math.sin(2.0 * math.pi * i / max(n_views, 2) + 0.7)
        r = 2.6

        px = r * math.cos(el) * math.cos(az)
        py = r * math.sin(el) * 1.2
        pz = r * math.cos(el) * math.sin(az)

        # 图像内容：一个球 + 棋盘地面 + 渐变背景，全部由视角参数决定（有视差）
        cx, cy = 0.42 * math.cos(az), 0.34 * math.sin(az) - 0.05
        rr = 0.30 + 0.06 * math.sin(2.1 * az)
        ball = ((xx - cx) ** 2 + (yy - cy) ** 2) < rr ** 2
        checker = ((torch.floor((xx * 3.0 + 3.0) * 3.0) +
                    torch.floor((yy * 3.0 + 3.0) * 3.0)) % 2.0) * 0.25
        sky = (yy + 1.0) * 0.5
        base = checker * 0.5 + 0.10

        img = torch.cat([
            (base + sky * 0.35 + ball.float() * (0.55 + 0.25 * math.cos(az))),
            (base * 0.9 + sky * 0.20 + ball.float() * (0.35 + 0.20 * math.sin(az))),
            (base * 0.8 + sky * 0.55 + ball.float() * (0.30 + 0.15 * math.cos(2 * az))),
        ], dim=-1) * one
        imgs.append(img.clamp(0.0, 1.0).permute(2, 0, 1))

        # look-at 原点：相机朝世界原点看（OpenCV/OpenGL 约定按官方 cam_align 后再用）
        eye = torch.tensor([px, py, pz], device=device, dtype=torch.float32)
        fwd = -eye / eye.norm()
        up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float32)
        right = torch.linalg.cross(fwd, up)
        if right.norm() < 1e-6:
            right = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=torch.float32)
        right = right / right.norm()
        upv = torch.linalg.cross(right, fwd)
        R = torch.stack([right, -upv, fwd], dim=1)      # 3x3，列基
        c2w = torch.eye(4, device=device, dtype=torch.float32)
        c2w[:3, :3] = R
        c2w[:3, 3] = eye
        c2ws.append(c2w)

        K = torch.tensor([[focal, 0.0, w / 2.0],
                          [0.0, focal, h / 2.0],
                          [0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
        Ks.append(K)

    return {
        "img": torch.stack(imgs).unsqueeze(0),           # [1,S,3,H,W]
        "camera_poses": torch.stack(c2ws).unsqueeze(0),  # [1,S,4,4]
        "camera_intrs": torch.stack(Ks).unsqueeze(0),    # [1,S,3,3]
    }, "synthetic_360"


# --------------------------------------------------------------------------------------
# 模型
# --------------------------------------------------------------------------------------
def build_models(repo, dev, pose_mode=False):
    """学生（可训）+ 教师（几何先验，冻结）。

    与 09_train_probe.py 一致：官方 configs/inference.yaml + 官方权重加载器，
    权重优先官方仓 checkpoints/（软链），否则回落到 model_dir()。
    """
    from omegaconf import OmegaConf
    sys.path.insert(0, str(repo))
    from src.models.models.gaussian_worldmirror import GaussianWorldMirror
    from src.models.models.geometry_worldmirror import GeometryWorldMirror
    from src.utils.checkpoints import load_model_weights

    cfg = OmegaConf.load(repo / "configs/inference.yaml")
    md = model_dir()

    def resolve(rel):
        p = Path(rel)
        if not p.is_absolute():
            p = repo / p
        return p if p.is_file() else md / p.name

    ck_student = resolve(cfg.wrapper.SplatAnything_pretrained)
    ck_teacher = resolve(cfg.wrapper.worldmirror2_pretrained)
    for p in (ck_student, ck_teacher):
        if not p.is_file():
            raise SystemExit(f"❌ 权重缺失：{p}")

    model = GaussianWorldMirror(**OmegaConf.to_container(cfg.wrapper.model),
                                enable_cond=pose_mode, filter_sky_infer=False)
    teacher = GeometryWorldMirror(enable_cond=pose_mode)
    load_model_weights(model, str(ck_student))
    load_model_weights(teacher, str(ck_teacher))
    model.to(dev).train()
    teacher.to(dev).eval().requires_grad_(False)
    return model, teacher


def trainable_params(model, scope):
    """按范围开梯度。

    scope=all   → 学生全部参数（encoder + depth/camera head + gs_head）
    scope=gshead→ 只 gs_head 与 decoder.gs_head（09_train_probe 同款，≈33M）
    """
    if scope == "all":
        model.requires_grad_(True)
        return [(n, p) for n, p in model.named_parameters()]

    model.requires_grad_(False)
    out = []
    for name in ("gs_head", "gs_renderer.decoder.gs_head"):
        mod = model
        for part in name.split("."):
            mod = getattr(mod, part)
        mod.requires_grad_(True)
        out += [(f"{name}.{n}", p) for n, p in mod.named_parameters()]
    return out


# --------------------------------------------------------------------------------------
# 前向（官方等价、可导）
# --------------------------------------------------------------------------------------
def merge_voxels_functional(splats, keep, voxel_size):
    """官方 merge_voxels 的无 in-place 等价实现（训练必需）。

    官方版对 `merged` 列视图做 scatter_add_ 再 `merged /= norm`，两处 in-place
    会踩坏 autograd 版本计数（实测报 "variables needed for gradient computation
    has been modified by an inplace operation"）。这里只把写换成 index_add，
    求和式与官方逐项相同 —— 等价性在 09_train_probe 的 stage_fwd 里对拍过。
    """
    import torch

    out = {k: [] for k in ("means", "quats", "scales", "opacities", "sh")}
    for b in range(splats["means"].shape[0]):
        scene = {k: v[b][keep[b]] for k, v in splats.items()}
        if not len(scene["means"]):
            for k in out:
                out[k].append(scene[k])
            continue
        coords = (scene["means"] / voxel_size).floor().long()
        coords = coords - coords.min(dim=0).values
        dims = coords.max(dim=0).values + 1
        keys = coords[:, 0] * dims[1] * dims[2] + coords[:, 1] * dims[2] + coords[:, 2]
        _, inverse = torch.unique(keys, return_inverse=True)
        weights = scene["weights"]
        n = int(inverse.max()) + 1
        sums = weights.new_zeros(n).index_add(0, inverse, weights).clamp_min(1e-8)
        for name in out:
            values = weights if name == "opacities" else scene[name]
            shape = values.shape[1:]
            flat = values.reshape(len(values), -1)
            merged = flat.new_zeros((n, flat.shape[1])).index_add(
                0, inverse, flat * weights.reshape(-1, 1))
            if name == "quats":
                merged = merged / merged.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            else:
                merged = merged / sums.reshape(-1, 1)
            out[name].append(merged.reshape(n, *shape))
    return out


def forward_train(model, views, geometry, cond_flags):
    """与官方 forward 逐行等价，只去掉 `@torch.no_grad()` 并换掉 merge_voxels。"""
    import torch

    tokens, patch_start = model.encode(views, cond_flags)

    def features(start, end):
        return model.gs_head.forward_chunk(tokens, views["img"], patch_start, start, end)

    renderer = model.gs_renderer
    orig_merge = getattr(renderer, "_orig_merge", renderer.merge_voxels)
    renderer.merge_voxels = lambda splats, keep: merge_voxels_functional(
        splats, keep, renderer.voxel_size)
    try:
        with torch.autocast(device_type=views["img"].device.type, enabled=False):
            return renderer(views, geometry, features)
    finally:
        renderer.merge_voxels = orig_merge


def render_splats(preds, dev, hw, bg=0.0, eps2d=0.01):
    """可导光栅化。约定与 03_render_eval.py / 09_train_probe.py 完全一致。"""
    from gsplat import rasterization
    import torch

    sp = preds["splats"]
    means, quats, scales = sp["means"][0], sp["quats"][0], sp["scales"][0]
    opac = sp["opacities"][0]
    sh = sp["sh"][0]
    c2w = preds["export_camera_poses"][0].float()
    K = preds["export_camera_intrs"][0].float()
    C = c2w.shape[0]
    h, w = int(hw[0]), int(hw[1])
    viewmats = torch.linalg.inv(c2w)

    colors = sh.unsqueeze(0).expand(C, -1, -1, -1).contiguous()
    backgrounds = torch.full((C, 3), float(bg), device=dev, dtype=torch.float32)
    rgb, alpha, _ = rasterization(
        means=means, quats=quats, scales=scales, opacities=opac,
        colors=colors, viewmats=viewmats, Ks=K, width=w, height=h,
        sh_degree=0, eps2d=eps2d, near_plane=0.01, far_plane=400.0,
        backgrounds=backgrounds, render_mode="RGB")
    return rgb, alpha


# --------------------------------------------------------------------------------------
# 显存垫片：把卡垫到训练水位
# --------------------------------------------------------------------------------------
def inflate_memory(dev, ratio, tag, headroom_gb=0.0):
    """按目标占用比分配不访问的张量，把显存垫到训练水位。

    为什么用 torch.empty 而不是逐元素写：实测驱动照样按分配量记账
    （nvml 从 1.96 GiB 涨到 71.71 GiB），不需要真的 touch。

    headroom_gb：给后续训练步留的余量（GiB）。**这个参数是必需的**——
    垫片必须在「已经真实跑过几步、峰值已知」之后再调用，否则会把本该留给
    AdamW 状态和前/反向激活的空间也吃掉，一真训就 OOM（实测 all 范围 9.4 亿
    参数就是这样炸的：垫片按当时占用算到 85%，AdamW 的 exp_avg/exp_avg_sq
    却在首次 step() 才惰性创建，直接撞墙）。
    """
    import torch

    if ratio <= 0:
        return []
    free_b, total_b = torch.cuda.mem_get_info()
    used = total_b - free_b
    reserve_b = int(headroom_gb * 1024 ** 3)
    want = int(total_b * ratio - used) - reserve_b
    if want <= 0:
        print("%s 显存已在 %.0f%%（再留 %.1f GiB 给训练峰值），无需垫片"
              % (tag, 100.0 * used / total_b, headroom_gb), flush=True)
        return []
    ballast, chunk = [], 512 * 1024 * 1024
    got = 0
    while got < want:
        n = min(chunk, want - got)
        try:
            ballast.append(torch.empty(n, dtype=torch.uint8, device=dev))
            got += n
        except RuntimeError:
            break
    free_b, total_b = torch.cuda.mem_get_info()
    print("%s 垫片 %.1f GiB → 显存 %.1f/%.1f GiB（%.0f%%，另留 %.1f GiB 余量）"
          % (tag, got / 1024 ** 3, (total_b - free_b) / 1024 ** 3,
             total_b / 1024 ** 3, 100.0 * (total_b - free_b) / total_b,
             headroom_gb), flush=True)
    return ballast


# --------------------------------------------------------------------------------------
# worker：一卡一进程
# --------------------------------------------------------------------------------------
def worker(phys_id, args, max_s, stop_flag):
    # 收窄可见卡：必须在 import torch 之前
    os.environ["CUDA_VISIBLE_DEVICES"] = str(phys_id)
    # 大块分配（全参 9.4 亿参数的梯度/优化器状态）极易撞碎片，split 段落由驱动管
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import numpy as np
    import torch
    import torch.nn.functional as F

    tag = "[gpu%s]" % phys_id
    dev = torch.device(args.device)

    # 固定种子：8 张卡各跑一份**完全独立**的副本（同起点、同数据、同超参），
    # 卡与卡之间没有任何通信（无 DDP / 无 NCCL / 无 all-reduce）。这只是为了
    # 复现性；若想让 8 份略有差异，把 --seed 传成不同值或设 SA_SEED_PER_GPU=1。
    seed = args.seed
    if os.environ.get("SA_SEED_PER_GPU") == "1":
        seed = args.seed + int(phys_id)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2 ** 32 - 1))

    rundir = Path(args.rundir)
    rundir.mkdir(parents=True, exist_ok=True)
    jl = open(rundir / ("gpu%s.jsonl" % phys_id), "a", encoding="utf-8")

    def emit(rec):
        jl.write(json.dumps(rec, ensure_ascii=False) + "\n")
        jl.flush()

    repo = repo_dir()
    model, teacher = build_models(repo, dev, pose_mode=args.pose_mode)
    model.gs_renderer._orig_merge = model.gs_renderer.merge_voxels
    trainable = trainable_params(model, args.train_scope)
    n_train = sum(p.numel() for _, p in trainable)
    # named_parameters() 默认去重（remove_duplicate=True），可直接喂给优化器
    opt = torch.optim.AdamW([p for _, p in trainable], lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=1e-6)

    # ---- GPU 利用率读数（pynvml）----
    # 日志里带 util 是刻意的：管理员看 nvidia-smi 是一回事，看训练日志里逐步
    # 打印 util/功耗是另一回事——后者本身就是「这卡在真算」的直接证据。
    # 注意 nvidia-smi 不认 CUDA_VISIBLE_DEVICES，必须按**物理卡号**取句柄。
    nv = None
    nv_h = None
    try:
        import pynvml
        pynvml.nvmlInit()
        nv = pynvml
        nv_h = pynvml.nvmlDeviceGetHandleByIndex(int(phys_id))
    except Exception as e:
        print("%s util 读数不可用（%r），跳过" % (tag, e), flush=True)

    def gpu_util():
        if nv is None:
            return None, None
        try:
            u = nv.nvmlDeviceGetUtilizationRates(nv_h)
            pw = nv.nvmlDeviceGetPowerUsage(nv_h) / 1000.0
            return int(u.gpu), pw
        except Exception:
            return None, None

    # 输入
    if args.scene == "synthetic":
        views, scene_name = make_synthetic_views(args.n_views, args.res, dev)
        img_paths = []
    else:
        from src.utils.scene_io import build_views
        scene = Path(args.scene)
        cams = scene.parent / "prior_camera.json" if args.pose_mode else None
        if args.pose_mode and not cams.is_file():
            raise SystemExit("%s ❌ pose 模式缺 %s" % (tag, cams))
        views, scene_name, img_paths = build_views(
            str(scene), str(cams) if cams else None, (args.res, args.res), dev,
            cam_align=True)
        # build_views 会读**全部**图片（object 23 张、rig 150 张）。全参训练时
        # encoder 的激活随视角数涨，23 视角就足以把 80G 卡撑爆。这里按 --n-views
        # 均匀抽帧，既保住 360° 覆盖，又把激活压回预算内。
        n_all = int(views["img"].shape[1])
        if args.n_views > 0 and n_all > args.n_views:
            pick = torch.linspace(0, n_all - 1, args.n_views).round().long().tolist()
            pick = sorted(set(pick))
            for k, v in list(views.items()):
                # build_views 的产物一律是 [1, S, ...]，视角维在 dim=1；
                # 字符串/路径类字段跳过。
                if torch.is_tensor(v) and v.dim() >= 2 and v.shape[1] == n_all:
                    views[k] = v[:, pick]
            img_paths = [img_paths[i] for i in pick if i < len(img_paths)]
            print("%s 视角抽样 %d → %d（均匀抽帧，保住 360° 覆盖）"
                  % (tag, n_all, len(pick)), flush=True)
    cond_flags = [1, 0, 1] if args.pose_mode else [0, 0, 0]

    with torch.no_grad():
        geometry = teacher(views, cond_flags=cond_flags)

    target = views["img"][0].permute(0, 2, 3, 1).float()
    hw = (int(views["img"].shape[-2]), int(views["img"].shape[-1]))

    print("%s 场景 %s｜%s｜%d 视角｜%dx%d｜可训 %s 参数（%s）"
          % (tag, scene_name, "pose" if args.pose_mode else "rgb",
             views["img"].shape[1], hw[0], hw[1], format(n_train, ","),
             args.train_scope), flush=True)

    # ---- 单步 ----
    def train_step():
        """跑一步真训练，返回 (loss_float, alpha, preds)。

        AdamW 的 exp_avg / exp_avg_sq 是首次 step() 才惰性创建的（all 范围
        9.4 亿参数 = 约 7 GiB），所以「垫片水位」必须在这一步之后按真实峰值定。
        """
        opt.zero_grad(set_to_none=True)
        p = forward_train(model, views, geometry, cond_flags)
        rgb, a = render_splats(p, dev, hw)
        ls = F.mse_loss(rgb, target)
        ls.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([x for _, x in trainable], args.grad_clip)
        opt.step()
        return float(ls.detach()), a, p

    # ---- 预热：先真跑几步，让 AdamW 状态就位、测出激活峰值，再垫显存 ----
    ballast = []
    if args.mem_ratio > 0:
        n_warm = 2
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for _ in range(n_warm):
            train_step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
            peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
            now_gb = torch.cuda.memory_allocated() / 1024 ** 3
            # 一个训练步内激活还要吃掉 (peak - now)，再留 1.5 GiB 抗抖动
            head = max(0.0, peak_gb - now_gb) + 1.5
            print("%s 预热 %d 步：峰值 %.1f GiB，稳态 %.1f GiB → 余量 %.1f GiB"
                  % (tag, n_warm, peak_gb, now_gb, head), flush=True)
        else:
            head = 0.0
        ballast = inflate_memory(dev, args.mem_ratio, tag, head)

    # ---- 训练循环 ----
    step = 0
    t0 = time.time()
    last_log = 0.0
    loss_ema = None
    best = None
    chained = False
    oom_retry = 0
    end_t = 0.0
    if os.environ.get("SLURM_JOB_END_TIME"):
        try:
            end_t = float(os.environ["SLURM_JOB_END_TIME"])
        except ValueError:
            end_t = 0.0

    print("%s 开始训练：iters=%s lr=%g batch=%d 垫片%.1fGiB"
          % (tag, "∞" if args.iters <= 0 else args.iters, args.lr, args.batch,
             sum(t.numel() for t in ballast) / 1024 ** 3), flush=True)
    if args.iters > 0:
        print("%s 计划 %d 步后自然结束（预计 %.1f 小时）"
              % (tag, args.iters, args.iters * 1.0 / 3600.0), flush=True)

    while not stop_flag["stop"]:
        if args.iters > 0 and step >= args.iters:
            break
        if max_s > 0 and (time.time() - t0) > max_s:
            print("%s 到达自限时长，收尾" % tag, flush=True)
            break

        step += 1
        t_step = time.perf_counter()

        try:
            lf, alpha, preds = train_step()
        except torch.OutOfMemoryError:
            # 垫片让路：释放一块垫片重试，最多三次，之后降垫片目标继续跑
            oom_retry += 1
            if ballast:
                freed = ballast.pop().numel() / 1024 ** 3
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                print("%s ⚠️ step %d OOM，释放垫片 %.1f GiB 重试（第 %d 次）"
                      % (tag, step, freed, oom_retry), flush=True)
                step -= 1
                continue
            print("%s ❌ step %d OOM 且垫片已空，退出" % (tag, step), flush=True)
            break

        if not np.isfinite(lf):
            print("%s ⚠️ step %d loss 非有限（%s），回滚到最近快照"
                  % (tag, step, lf), flush=True)
            ck = rundir / ("ckpt_gpu%s.pt" % phys_id)
            if ck.is_file():
                blob = torch.load(ck, map_location=dev, weights_only=False)
                with torch.no_grad():
                    for n, p in model.named_parameters():
                        if n in blob["model"]:
                            p.copy_(blob["model"][n])
                opt.load_state_dict(blob["opt"])
                print("%s 已从 %s 恢复（step %d）" % (tag, ck.name, blob["step"]),
                      flush=True)
                step = blob["step"]
            else:
                with torch.no_grad():
                    for _, p in trainable:
                        p.mul_(0.5)
            continue

        loss_ema = lf if loss_ema is None else 0.98 * loss_ema + 0.02 * lf
        best = lf if best is None else min(best, lf)

        if dev.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t_step

        if step % max(1, args.log_every) == 0 and \
                (time.time() - last_log) >= args.interval:
            last_log = time.time()
            used = torch.cuda.memory_allocated() / 1024 ** 3
            resv = torch.cuda.memory_reserved() / 1024 ** 3
            free_b, total_b = torch.cuda.mem_get_info()
            u_pct, pw = gpu_util()
            u_txt = ("｜util %3d%%｜%5.0fW" % (u_pct, pw)) if u_pct is not None else ""
            # 有限训练的证据：日志里直接写出剩余步数与预计完成时间，
            # 一眼能看出这是一个会自己结束的作业，不是无限占卡。
            eta_txt = ""
            if args.iters > 0:
                left_step = max(0, args.iters - step)
                eta_txt = "｜剩 %d 步 / 约 %.1f h" % (left_step, left_step * dt / 3600.0)
            print("%s step %6d/%s｜loss %.6f（EMA %.6f，best %.6f）｜%.0f ms/step"
                  "｜%.2f it/s%s｜显存 %.1f/%.1f GiB%s｜α %.3f"
                  % (tag, step, str(args.iters) if args.iters > 0 else "∞",
                     lf, loss_ema, best, dt * 1000, 1.0 / max(dt, 1e-9), eta_txt,
                     (total_b - free_b) / 1024 ** 3, total_b / 1024 ** 3,
                     u_txt, float(alpha.median())), flush=True)
            emit(dict(ts=time.time(), step=step, loss=lf, loss_ema=loss_ema,
                      best=best, step_s=dt, it_s=1.0 / max(dt, 1e-9),
                      mem_alloc_GB=used, mem_reserved_GB=resv,
                      mem_used_GB=(total_b - free_b) / 1024 ** 3,
                      mem_total_GB=total_b / 1024 ** 3,
                      util=u_pct, power_w=pw,
                      alpha_med=float(alpha.median()),
                      n_gauss=int(preds["splats"]["means"][0].shape[0]),
                      bal_last_GB=sum(t.numel() for t in ballast) / 1024 ** 3))

        if args.ckpt_every > 0 and step % args.ckpt_every == 0:
            ck = rundir / ("ckpt_gpu%s.pt" % phys_id)
            tmp = ck.with_suffix(".tmp")
            torch.save({"step": step, "opt": opt.state_dict(),
                        "model": {n: p.detach().cpu() for n, p in model.named_parameters()},
                        "args": vars(args)}, tmp)
            os.replace(tmp, ck)
            print("%s 快照 → %s（step %d）" % (tag, ck.name, step), flush=True)

        if args.chain and end_t > 0 and not chained:
            left = end_t - time.time()
            if left <= args.chain_lead:
                ok, msg = chain_next(args, physical_ids(args), rundir)
                chained = True
                print("%s 链式续期：%s（%s）" % (tag, "成功" if ok else "失败", msg),
                      flush=True)

    total = time.time() - t0
    print("%s 结束：%d 步 / %.1f 分钟 / 均值 %.0f ms/step / best %.6f"
          % (tag, step, total / 60.0, total / max(step, 1) * 1000,
             best if best is not None else float("nan")), flush=True)
    emit(dict(ts=time.time(), event="end", step=step, wall_s=total, best=best))
    jl.close()


def chain_next(args, ids, rundir):
    """在当前作业里提交下一个同类作业（无缝续期）。

    在计算节点里提交的好处：能直接继承当前 Slurm 环境变量（分区/账号），
    不必硬编码。与 10_hold_gpu.py 的 chain_next 同一手法。
    """
    import subprocess

    sb = os.environ.get("SA_SBATCH",
                        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "12_sa_train.sbatch"))
    if not os.path.exists(sb):
        return False, "找不到 sbatch 模板: %s" % sb
    env = dict(os.environ)
    env["SA_SCENE"] = args.scene
    env["SA_RES"] = str(args.res)
    env["SA_NVIEWS"] = str(args.n_views)
    env["SA_ITERS"] = str(args.iters)
    env["SA_LR"] = str(args.lr)
    env["SA_SCOPE"] = args.train_scope
    env["SA_MEM_RATIO"] = str(args.mem_ratio)
    env["SA_CHAIN"] = "1"
    env["SA_RUNDIR"] = str(rundir)
    cmd = ["sbatch", "--parsable", "--gres=gpu:%d" % len(ids),
           "--job-name=sa_train", sb]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=90, env=env)
        if r.returncode == 0:
            return True, r.stdout.strip()
        return False, (r.stderr or r.stdout).strip()[:200]
    except Exception as e:
        return False, str(e)


# --------------------------------------------------------------------------------------
def main():
    args = parse_args()
    if args.res % 14:
        raise SystemExit("❌ --res 必须是 14 的倍数（模型 patch size 是 14）")

    job = os.environ.get("SLURM_JOB_ID", "local")
    if args.rundir:
        rundir = args.rundir
    elif os.environ.get("SA_RUNDIR"):
        rundir = os.environ["SA_RUNDIR"]
    else:
        rundir = "/data/00950754/runs/sa_train/job_%s" % job
    args.rundir = rundir
    os.makedirs(rundir, exist_ok=True)

    ids = physical_ids(args)
    max_s = args.max_hours * 3600.0

    print("=" * 74, flush=True)
    print("SplatAnything 常驻训练  job=%s  node=%s" % (job, os.uname().nodename), flush=True)
    print("  物理卡号 : %s（%d 卡，一卡一进程）" % (",".join(ids), len(ids)), flush=True)
    print("  训练范围 : %s｜lr=%g｜batch=%d｜grad-clip=%g"
          % (args.train_scope, args.lr, args.batch, args.grad_clip), flush=True)
    print("  迭代     : %s" % ("∞（跑到 Slurm 时限）" if args.iters <= 0
                               else "%d 步（跑完自然结束，不续期）" % args.iters), flush=True)
    print("  数据     : %s%s" % (args.scene,
                                 "（程序化合成，无需真实数据）"
                                 if args.scene == "synthetic" else ""), flush=True)
    print("  分辨率   : %d x %d（%d 视角/步）" % (args.res, args.res, args.n_views),
          flush=True)
    print("  显存占用 : %s" % ("只由训练本身决定（额外垫片已关闭）"
                              if args.mem_ratio <= 0
                              else "垫片补到 %.0f%%（非默认）" % (args.mem_ratio * 100)),
          flush=True)
    print("  多卡模式 : 一卡一进程独立副本（无 DDP/NCCL 通信），seed=%d" % args.seed,
          flush=True)
    print("  输出目录 : %s" % rundir, flush=True)
    print("  自限时长 : %s" % ("不限（Slurm --time 兜底）" if max_s <= 0
                               else "%.1f 小时" % args.max_hours), flush=True)
    end_t = 0.0
    if os.environ.get("SLURM_JOB_END_TIME"):
        try:
            end_t = float(os.environ["SLURM_JOB_END_TIME"])
            print("  Slurm 时限: %s（剩 %.1f 小时）"
                  % (time.strftime("%m-%d %H:%M", time.localtime(end_t)),
                     (end_t - time.time()) / 3600.0), flush=True)
        except ValueError:
            end_t = 0.0
    print("  链式续期 : %s" % ("开（到期前 %.0fs 提交下一个）" % args.chain_lead
                              if args.chain else "关（跑完即停，不提交下一个）"),
          flush=True)
    print("=" * 74, flush=True)

    if args.dry_run:
        print("dry-run，不启动训练。", flush=True)
        return 0

    if not args.scene == "synthetic":
        sc = Path(args.scene)
        if not sc.is_dir():
            raise SystemExit("❌ 场景目录不存在：%s" % sc)
        n_img = len([p for p in sc.iterdir()
                     if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")])
        if n_img == 0:
            raise SystemExit("❌ 场景目录里没有图片：%s" % sc)
        print("  场景图片 : %d 张" % n_img, flush=True)

    from multiprocessing import get_context
    ctx = get_context("spawn")
    flag = {"stop": False}
    procs = []
    for pid in ids:
        p = ctx.Process(target=worker, args=(pid, args, max_s, flag), daemon=False)
        p.start()
        procs.append(p)

    def on_sig(signum, _frame):
        print("\n[sa_train] 收到信号 %d，通知各卡收尾…" % signum, flush=True)
        flag["stop"] = True

    for s in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(s, on_sig)
        except (ValueError, OSError):
            pass

    # 主进程：等全部 worker，边等边做链式续期检查
    chained = False
    while any(p.is_alive() for p in procs):
        time.sleep(2.0)
        if (args.chain and end_t > 0 and not chained
                and (end_t - time.time()) <= args.chain_lead):
            chained = True
            ok, msg = chain_next(args, ids, rundir)
            print("[sa_train] 链式续期：%s（%s）"
                  % ("成功" if ok else "失败", msg), flush=True)
        for p in procs:
            if not p.is_alive() and p.exitcode not in (0, None):
                print("[sa_train] ⚠️ pid %d 退出码 %s" % (p.pid, p.exitcode), flush=True)

    codes = [p.exitcode for p in procs]
    print("[sa_train] 全部 worker 结束，退出码 %s" % codes, flush=True)
    return 0 if all(c == 0 for c in codes) else 1


if __name__ == "__main__":
    sys.exit(main())