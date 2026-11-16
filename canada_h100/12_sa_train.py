#!/usr/bin/env python3
"""SplatAnything 单节点多卡训练（NCCL 数据并行）。

torchrun 起 N 个进程，每张卡一份模型副本；各 rank 读不同的场景，前向 + 反传
后由 NCCL all-reduce 平均梯度，rank0 统一落盘快照。

    torchrun --standalone --nproc_per_node=8 12_sa_train.py \
        --iters 250000 --res 504 --n-views 8 --train-scope all

数据：默认取官方仓 examples/ 下的真实场景（aerial / indoor / object / pano /
rig / street），rank i 用第 i % n 个场景，并在自己的场景内按 --rotate-every
轮换视角窗口。也可用 --scene 指定单场景、全卡共用。

目标：渲染出的高斯图像与输入图像做 MSE。学生 GaussianWorldMirror（全参微调时
9.4 亿参数），几何先验来自冻结的 GeometryWorldMirror。

输出：

  <rundir>/metrics_rank{N}.jsonl   逐 rank 指标（JSONL）
  <rundir>/ckpt.pt                rank0 汇总快照（覆盖写，可 --resume）
  stdout                          即 Slurm .out
"""
import argparse
import json
import os
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

SAMPLE_SCENES = ("aerial", "indoor", "object", "pano", "rig", "street")


# --------------------------------------------------------------------------------------
# 参数
# --------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    g = p.add_argument_group("数据")
    g.add_argument("--scene", default="",
                   help="单个场景 images 目录；给了就全卡共用这一个场景")
    g.add_argument("--scenes", default="",
                   help="逗号分隔的多个场景 images 目录；rank i 用第 i %% n 个")
    g.add_argument("--pose-mode", action="store_true",
                   help="pose 条件模式（读场景同级 prior_camera.json）")
    g.add_argument("--res", type=int, default=504,
                   help="训练分辨率长边，必须是 14 的倍数（默认 504）")
    g.add_argument("--n-views", type=int, default=8,
                   help="每步参与训练的视角数（默认 8）")
    g.add_argument("--rotate-every", type=int, default=500,
                   help="每 N 步换一次视角窗口（默认 500；0 = 固定窗口）")

    g = p.add_argument_group("优化")
    g.add_argument("--iters", type=int, default=250000,
                   help="训练总步数（默认 25 万步；0 = 不限，由 Slurm 时限兜底）")
    g.add_argument("--lr", type=float, default=1e-5, help="学习率（默认 1e-5）")
    g.add_argument("--train-scope", default="all", choices=("gshead", "all"),
                   help="gshead = 只训解码头（≈33M）；all = 全参微调（默认）")
    g.add_argument("--grad-clip", type=float, default=1.0,
                   help="梯度裁剪范数（默认 1.0，0 = 不裁）")
    g.add_argument("--seed", type=int, default=1234, help="随机种子（默认 1234）")

    g = p.add_argument_group("日志与快照")
    g.add_argument("--log-every", type=int, default=20, help="每 N 步打一行摘要")
    g.add_argument("--ckpt-every", type=int, default=20000,
                   help="每 N 步存一次快照（默认 20000；0 = 不存）")
    g.add_argument("--rundir", default="", help="输出目录")
    g.add_argument("--resume", action="store_true", help="从 <rundir>/ckpt.pt 续训")
    g.add_argument("--max-hours", type=float, default=0.0,
                   help="自限时长（小时）；0 = 不限")

    g.add_argument("--device", default="cuda", help="计算设备（默认 cuda）")
    g.add_argument("--dry-run", action="store_true", help="只打印配置")
    return p.parse_args(argv)


# --------------------------------------------------------------------------------------
# 路径
# --------------------------------------------------------------------------------------
def repo_dir():
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
    raise SystemExit("❌ 找不到权重目录（设 SPLATANYTHING_MODEL_DIR）")


def resolve_scenes(args, repo):
    """要用的场景目录列表。"""
    if args.scene:
        paths = [Path(args.scene)]
    elif args.scenes:
        paths = [Path(x.strip()) for x in args.scenes.split(",") if x.strip()]
    else:
        base = repo / "examples"
        paths = [base / n / "images" for n in SAMPLE_SCENES]
        paths = [p for p in paths if p.is_dir()]
    if not paths:
        raise SystemExit("❌ 没有可用场景目录")
    for p in paths:
        if not p.is_dir():
            raise SystemExit("❌ 场景目录不存在：%s" % p)
        if not any(p.iterdir()):
            raise SystemExit("❌ 场景目录是空的：%s" % p)
    return paths


# --------------------------------------------------------------------------------------
# 分布式
# --------------------------------------------------------------------------------------
def setup_dist():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank


def phys_gpu(local_rank):
    """local_rank → 物理卡号（NVML 句柄要用物理号）。"""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cvd:
        parts = [x.strip() for x in cvd.split(",") if x.strip()]
        if local_rank < len(parts) and parts[local_rank].isdigit():
            return int(parts[local_rank])
    return local_rank


def is_main():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def sync_grads(params, world_size, bucket_mb=128):
    """跨 rank 平均梯度（NCCL all-reduce，按 dtype 分桶控制峰值显存）。"""
    if world_size <= 1:
        return
    limit = bucket_mb * 1024 * 1024
    groups = defaultdict(list)
    for p in params:
        if p.grad is not None:
            groups[p.grad.device].append(p.grad)

    def flush(bucket):
        if not bucket:
            return
        flat = torch._utils._flatten_dense_tensors(bucket)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(world_size)
        for g, r in zip(bucket, torch._utils._unflatten_dense_tensors(flat, bucket)):
            g.copy_(r)

    for grads in groups.values():
        bucket, nbytes = [], 0
        for g in grads:
            bucket.append(g)
            nbytes += g.numel() * g.element_size()
            if nbytes >= limit:
                flush(bucket)
                bucket, nbytes = [], 0
        flush(bucket)


# --------------------------------------------------------------------------------------
# 模型
# --------------------------------------------------------------------------------------
def build_models(repo, dev, pose_mode=False):
    """学生（可训）+ 几何先验教师（冻结）。"""
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

    scope=all    → 学生全部参数（encoder + depth/camera head + gs_head）
    scope=gshead → 只 gs_head 与 decoder.gs_head（≈33M）
    """
    if scope == "all":
        model.requires_grad_(True)
        return [p for p in model.parameters() if p.requires_grad]

    model.requires_grad_(False)
    keep = []
    for name in ("gs_head", "gs_renderer.decoder.gs_head"):
        mod = model
        for part in name.split("."):
            mod = getattr(mod, part)
        mod.requires_grad_(True)
        keep += [p for p in mod.parameters()]
    return keep


# --------------------------------------------------------------------------------------
# 前向（官方等价、可导）
# --------------------------------------------------------------------------------------
def merge_voxels_functional(splats, keep, voxel_size):
    """官方 merge_voxels 的无 in-place 等价实现。

    官方版对 `merged` 列视图做 scatter_add_ 再 `merged /= norm`，两处 in-place
    会踩坏 autograd 版本计数。这里改成 index_add，求和式与官方逐项相同。
    """
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
    """可导光栅化（与推理脚本同一套相机约定）。"""
    from gsplat import rasterization

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
# 场景
# --------------------------------------------------------------------------------------
def load_scene(path, res, pose_mode, tag):
    """读一个场景的全部视角，常驻显存供窗口轮换复用。"""
    from src.utils.scene_io import build_views

    cams = path.parent / "prior_camera.json" if pose_mode else None
    if pose_mode and not cams.is_file():
        raise SystemExit("%s ❌ pose 模式缺 %s" % (tag, cams))
    views, name, _ = build_views(
        str(path), str(cams) if cams else None, (res, res),
        torch.device("cpu"), cam_align=True)
    n_all = int(views["img"].shape[1])
    print("%s 场景 %s：%d 视角，%dx%d"
          % (tag, name, n_all, views["img"].shape[-2], views["img"].shape[-1]),
          flush=True)
    return views, name, n_all


def window_pick(n_all, index, n_views):
    """按窗口序号轮换视角子集，步长均匀以保住 360° 覆盖。"""
    if n_views <= 0 or n_all <= n_views:
        return list(range(n_all))
    stride = max(1, n_all // n_views)
    st = (index * stride) % n_all
    pick = [(st + i * stride) % n_all for i in range(n_views)]
    pick = sorted(set(pick))
    i = 0
    while len(pick) < n_views and i < n_all:
        if i not in pick:
            pick.append(i)
        i += 1
    return sorted(pick)


def slice_views(full, n_all, index, n_views, dev):
    """从全量视角里切一个窗口，搬到显存。"""
    pick = window_pick(n_all, index, n_views)
    out = {}
    for k, v in full.items():
        if torch.is_tensor(v) and v.dim() >= 2 and v.shape[1] == n_all:
            out[k] = v[:, pick].to(dev, non_blocking=True)
        else:
            out[k] = v
    target = out["img"][0].permute(0, 2, 3, 1).float()
    return out, target


# --------------------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------------------
def main():
    args = parse_args()
    if args.res % 14:
        raise SystemExit("❌ --res 必须是 14 的倍数（模型 patch size 是 14）")

    local_rank = setup_dist()
    rank = dist.get_rank()
    world = dist.get_world_size()
    pgpu = phys_gpu(local_rank)
    use_cuda = torch.cuda.is_available()
    dev = torch.device("cuda", local_rank) if use_cuda else torch.device("cpu")
    tag = "[rank%d/gpu%d]" % (rank, pgpu)

    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)

    job = os.environ.get("SLURM_JOB_ID", "local")
    rundir = Path(args.rundir or os.environ.get("SA_RUNDIR")
                  or "/data/00950754/runs/sa_train/job_%s" % job)
    if is_main():
        rundir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    repo = repo_dir()
    scenes = resolve_scenes(args, repo)
    scene_path = scenes[rank % len(scenes)]

    if is_main():
        print("=" * 74, flush=True)
        print("SplatAnything 训练  job=%s  node=%s"
              % (job, os.uname().nodename), flush=True)
        print("  并行     : %d 进程 / NCCL 数据并行（all-reduce 平均梯度）" % world,
              flush=True)
        print("  场景     : %d 个 → %s"
              % (len(scenes), ", ".join(s.parent.name for s in scenes)), flush=True)
        print("  训练范围 : %s｜lr=%g｜grad-clip=%g"
              % (args.train_scope, args.lr, args.grad_clip), flush=True)
        print("  迭代     : %s" % ("不限（由 Slurm 时限兜底）" if args.iters <= 0
                                   else "%d 步" % args.iters), flush=True)
        print("  分辨率   : %d x %d（%d 视角/步，每 %d 步轮换窗口）"
              % (args.res, args.res, args.n_views, max(1, args.rotate_every)),
              flush=True)
        print("  输出目录 : %s" % rundir, flush=True)
        print("=" * 74, flush=True)

    if args.dry_run:
        dist.barrier()
        dist.destroy_process_group()
        return 0

    model, teacher = build_models(repo, dev, pose_mode=args.pose_mode)
    model.gs_renderer._orig_merge = model.gs_renderer.merge_voxels
    params = trainable_params(model, args.train_scope)
    n_train = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=1e-6)

    ckpt_path = rundir / "ckpt.pt"
    step = 0
    if args.resume and ckpt_path.is_file():
        blob = torch.load(ckpt_path, map_location=dev, weights_only=False)
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in blob["model"]:
                    p.copy_(blob["model"][n].to(dev))
        opt.load_state_dict(blob["opt"])
        step = int(blob.get("step", 0))
        if is_main():
            print("[resume] 从 %s 恢复，step %d" % (ckpt_path, step), flush=True)
    dist.barrier()

    # NVML 读数：按物理卡号取句柄（CUDA_VISIBLE_DEVICES 对 NVML 无效）
    nv = nv_h = None
    try:
        import pynvml
        pynvml.nvmlInit()
        nv = pynvml
        nv_h = pynvml.nvmlDeviceGetHandleByIndex(int(pgpu))
    except Exception:
        nv = None

    def gpu_util():
        if nv is None:
            return None, None
        try:
            return (int(nv.nvmlDeviceGetUtilizationRates(nv_h).gpu),
                    nv.nvmlDeviceGetPowerUsage(nv_h) / 1000.0)
        except Exception:
            return None, None

    full, scene_name, n_all = load_scene(scene_path, args.res, args.pose_mode, tag)
    cond_flags = [1, 0, 1] if args.pose_mode else [0, 0, 0]
    hw = (args.res, args.res)

    print("%s 可训 %s 参数（%s）｜场景 %s｜%d 视角"
          % (tag, format(n_train, ","), args.train_scope, scene_name, n_all),
          flush=True)

    jl = open(rundir / ("metrics_rank%d.jsonl" % rank), "a", encoding="utf-8")

    def emit(rec):
        jl.write(json.dumps(rec, ensure_ascii=False) + "\n")
        jl.flush()

    stop = {"flag": False}

    def on_sig(signum, _frame):
        print("%s 收到信号 %d，收尾" % (tag, signum), flush=True)
        stop["flag"] = True

    for s in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(s, on_sig)
        except (ValueError, OSError):
            pass

    t0 = time.time()
    loss_ema = None
    best = None
    win = -1
    win_bad = 0
    views = target = geometry = None
    max_s = args.max_hours * 3600.0

    print("%s 开始训练：iters=%s lr=%g world=%d"
          % (tag, "∞" if args.iters <= 0 else args.iters, args.lr, world), flush=True)

    while not stop["flag"]:
        if args.iters > 0 and step >= args.iters:
            break
        if max_s > 0 and (time.time() - t0) > max_s:
            print("%s 到达自限时长，收尾" % tag, flush=True)
            break

        cur = step // max(1, args.rotate_every)
        if cur != win:
            win = cur
            # 先把上一窗口的张量真正放掉再建新的：赋值前旧 views 与该步 teacher
            # 前向的中间激活会同时驻留，实测 reserved 从 65.9 一路爬到 73.3 GiB。
            views = target = geometry = None
            if use_cuda:
                torch.cuda.empty_cache()
            views, target = slice_views(full, n_all, cur, args.n_views, dev)
            with torch.no_grad():
                geometry = teacher(views, cond_flags=cond_flags)
            if use_cuda:
                torch.cuda.empty_cache()
            if cur > 0:
                print("%s 视角窗口 #%d（%d 视角）"
                      % (tag, cur, views["img"].shape[1]), flush=True)

        step += 1
        t_step = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        oom = False
        try:
            preds = forward_train(model, views, geometry, cond_flags)
            rgb, alpha = render_splats(preds, dev, hw)
            loss = F.mse_loss(rgb, target)
            loss.backward()
        except torch.OutOfMemoryError:
            oom = True

        # 跳窗必须整组一起跳：若各 rank 的 step 走岔，下一次 all_reduce 就对不上
        # 集合点，NCCL 会一直挂到作业超时。
        if world > 1:
            flag = torch.tensor([1 if oom else 0], device=dev, dtype=torch.int32)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            oom = bool(flag.item())
        if oom:
            # 视角恰好贴近几何时，gsplat 的 isect_tiles 会按 n_gauss × 覆盖 tile 数
            # 申请大块临时缓冲（实测 2.93 GiB）而顶穿。同窗口重试形状不变，直接跳过。
            opt.zero_grad(set_to_none=True)
            preds = rgb = alpha = loss = None
            if use_cuda:
                torch.cuda.empty_cache()
            win_bad += 1
            if win_bad > 3:
                raise SystemExit("%s ❌ 连续 %d 个窗口显存不足，放弃" % (tag, win_bad))
            print("%s ⚠️ step %d 显存不足，跳过窗口 #%d（本作业第 %d 次）"
                  % (tag, step, cur, win_bad), flush=True)
            step = (cur + 1) * max(1, args.rotate_every)
            continue

        sync_grads(params, world)
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        opt.step()

        lf = float(loss.detach())
        if not torch.isfinite(loss):
            print("%s ⚠️ step %d loss 非有限，跳过该步" % (tag, step), flush=True)
            continue

        loss_ema = lf if loss_ema is None else 0.98 * loss_ema + 0.02 * lf
        best = lf if best is None else min(best, lf)
        if use_cuda:
            torch.cuda.synchronize()
        dt = time.perf_counter() - t_step

        if step % max(1, args.log_every) == 0:
            used = torch.cuda.memory_allocated() / 1024 ** 3 if use_cuda else 0.0
            resv = torch.cuda.memory_reserved() / 1024 ** 3 if use_cuda else 0.0
            u_pct, pw = gpu_util()
            u_txt = ("｜util %3d%%｜%5.0fW" % (u_pct, pw)) if u_pct is not None else ""
            eta = ""
            if args.iters > 0:
                left = max(0, args.iters - step)
                eta = "｜剩 %d 步 / 约 %.1f h" % (left, left * dt / 3600.0)
            print("%s step %6d/%s｜loss %.6f（EMA %.6f，best %.6f）｜%.0f ms/step"
                  "｜%.2f it/s%s｜显存 %.1f/%.1f GiB%s｜α %.3f"
                  % (tag, step, str(args.iters) if args.iters > 0 else "∞",
                     lf, loss_ema, best, dt * 1000, 1.0 / max(dt, 1e-9), eta,
                     used, resv, u_txt, float(alpha.median())), flush=True)
            emit(dict(ts=time.time(), step=step, rank=rank, world=world,
                      loss=lf, loss_ema=loss_ema, best=best, step_s=dt,
                      it_s=1.0 / max(dt, 1e-9), scene=scene_name,
                      mem_alloc_GB=used, mem_reserved_GB=resv,
                      util=u_pct, power_w=pw, alpha_med=float(alpha.median()),
                      n_gauss=int(preds["splats"]["means"][0].shape[0])))

        if args.ckpt_every > 0 and step % args.ckpt_every == 0:
            dist.barrier()
            if is_main():
                tmp = ckpt_path.with_suffix(".tmp")
                torch.save({"step": step, "opt": opt.state_dict(),
                            "model": {n: p.detach().cpu()
                                      for n, p in model.named_parameters()},
                            "args": vars(args)}, tmp)
                os.replace(tmp, ckpt_path)
                print("%s 快照 → %s（step %d）"
                      % (tag, ckpt_path.name, step), flush=True)
            dist.barrier()

    total = time.time() - t0
    print("%s 结束：%d 步 / %.1f 分钟 / 均值 %.0f ms/step / best %.6f"
          % (tag, step, total / 60.0, total / max(step, 1) * 1000,
             best if best is not None else float("nan")), flush=True)
    emit(dict(ts=time.time(), event="end", step=step, wall_s=total, best=best))
    jl.close()

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
