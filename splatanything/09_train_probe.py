#!/usr/bin/env python3
"""WB-4 / R3：SplatAnything 训练改造成本冒烟（等价可导前向 → 1 step → ≤100 step → 存恢复）。

为什么需要适配层：官方发布包**只有推理**——`inference.py` 用 `inference_mode()`，
`GaussianWorldMirror.forward` 上挂 `@torch.no_grad()`，`rasterization.py` 更是在注释里
写明 "no rasterization or training"。本脚本**不复制官方源码**，只在官方模块上补三件事：

  1. 绕开 `no_grad` 装饰器：直接调 `model.encode` + `model.gs_renderer`，
     与官方 `forward` 逐行等价（官方 forward 也就是这两句）。
  2. 补可导光栅化：用 gsplat，坐标/底色约定同 `03_render_eval.py`。
  3. 补 loss / optimizer / checkpoint / 计时显存。

基线：官方仓 commit `b518996`（见 NOTES.md 顶部）。

冻结策略（按主计划 WB-4 第 2 条）：encoder、depth head、camera head、means 全部冻结，
只训 `gs_head`（GSFeatHead）+ `gs_renderer.decoder.gs_head`（128→12 的 1x1 卷积）。
GSN、体素合并、过滤**全部保留官方原样**，不替换成简化实现。

用法（WSL，env splatanything）：
  python 09_train_probe.py --stages fwd                 # 只做等价性对照
  python 09_train_probe.py --stages step1               # 单步反传 + 梯度分支检查
  python 09_train_probe.py --stages steps --n-steps 100 # 成本曲线
  python 09_train_probe.py --stages all --n-steps 100   # 全套 + 存恢复，写 train_probe.json
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def official_repo():
    d = os.environ.get("SPLATANYTHING_DIR") or str(Path.home() / "repos/SplatAnything")
    p = Path(d).expanduser()
    if not (p / "inference.py").is_file():
        raise SystemExit(f"❌ 找不到官方仓：{p}（设 SPLATANYTHING_DIR）")
    return p


def model_dir():
    return Path(os.environ.get("SPLATANYTHING_MODEL_DIR")
                or str(Path.home() / "model/splatanything")).expanduser()


# --------------------------------------------------------------------------------------
# 计时 / 显存
# --------------------------------------------------------------------------------------
class Chrono:
    """CUDA 侧真时间：所有读数前先 synchronize，避免把异步排队算进 wall clock。"""

    def __init__(self, dev):
        self.dev = dev
        self.t = {}

    def sync(self):
        if self.dev.type == "cuda":
            torch.cuda.synchronize()

    def __call__(self, key):
        chrono = self

        class _Ctx:
            def __enter__(self):
                chrono.sync()
                self.t0 = time.perf_counter()
                return self

            def __exit__(self, *a):
                chrono.sync()
                chrono.t[key] = chrono.t.get(key, 0.0) + (time.perf_counter() - self.t0)
                return False

        return _Ctx()


def mem_stats(dev):
    if dev.type != "cuda":
        return {}
    return dict(peak_alloc_GB=torch.cuda.max_memory_allocated() / 2 ** 30,
                peak_reserved_GB=torch.cuda.max_memory_reserved() / 2 ** 30)


def reset_peak(dev):
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------
# 模型 / 输入
# --------------------------------------------------------------------------------------
def build_models(repo, dev, pose_mode=False):
    from omegaconf import OmegaConf
    from src.models.models.gaussian_worldmirror import GaussianWorldMirror
    from src.models.models.geometry_worldmirror import GeometryWorldMirror
    from src.utils.checkpoints import load_model_weights

    cfg = OmegaConf.load(repo / "configs/inference.yaml")
    md = model_dir()
    ck_student = Path(cfg.wrapper.SplatAnything_pretrained)
    ck_teacher = Path(cfg.wrapper.worldmirror2_pretrained)
    for p in (ck_student, ck_teacher):
        if not p.is_file():
            p = md / p.name
            if not p.is_file():
                raise SystemExit(f"❌ 权重缺失：{p}")

    model = GaussianWorldMirror(**OmegaConf.to_container(cfg.wrapper.model),
                                enable_cond=pose_mode, filter_sky_infer=False)
    teacher = GeometryWorldMirror(enable_cond=pose_mode)
    load_model_weights(model, str(ck_student if ck_student.is_file() else md / ck_student.name))
    load_model_weights(teacher, str(ck_teacher if ck_teacher.is_file() else md / ck_teacher.name))
    model.to(dev).eval()
    teacher.to(dev).eval().requires_grad_(False)
    return cfg, model, teacher


def build_inputs(scene_images, res, dev, pose_mode=False):
    """官方 preprocess：以主点裁正方形再缩到 res（与 02/07 完全同一套，不另写）。"""
    from src.utils.scene_io import build_views

    scene = Path(scene_images)
    cameras = scene.parent / "prior_camera.json" if pose_mode else None
    if pose_mode and not cameras.is_file():
        raise SystemExit(f"❌ pose 模式需要 {cameras}")
    views, scene_name, img_paths = build_views(
        str(scene), str(cameras) if cameras else None, (res, res), dev, cam_align=True)
    return views, scene_name, img_paths


def set_trainable(model):
    """只留 gs_head + decoder.gs_head 可训；其余（含 encoder / 教师）全部冻结。"""
    model.requires_grad_(False)
    trainable = []
    for name in ("gs_head", "gs_renderer.decoder.gs_head"):
        mod = model
        for part in name.split("."):
            mod = getattr(mod, part)
        mod.requires_grad_(True)
        trainable += [(f"{name}.{n}", p) for n, p in mod.named_parameters()]
    return trainable


# --------------------------------------------------------------------------------------
# 前向：官方等价路径（绕开 no_grad 装饰器）+ 分段计时
# --------------------------------------------------------------------------------------
def merge_voxels_functional(splats, keep, voxel_size):
    """官方 `merge_voxels` 的数值等价、**无 in-place** 实现 —— 训练必需。

    官方版把每一列 `scatter_add_` 写进 `merged` 的列视图，再对 quats 做 `merged /= norm`。
    这两处 in-place 会踩坏 autograd 的版本计数（实测报
    "one of the variables needed for gradient computation has been modified by an
    inplace operation: [488539, 4] ... version 5; expected 4"）。
    这里只把写操作换成 out-of-place 的 `index_add`，求和式与官方逐项相同；
    等价性由 stage_fwd 与官方路径对拍验证（同一权重、同一 keep）。
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


def forward_train(model, views, geometry, cond_flags, chrono, stash, freeze_encoder=True):
    """与 GaussianWorldMirror.forward 逐行等价，只去掉 `@torch.no_grad()`。

    freeze_encoder=True 时 encoder 在 no_grad 下跑（省显存，梯度不回流 encoder）。
    """
    ctx = torch.no_grad() if freeze_encoder else torch.enable_grad()
    with chrono("enc_s"), ctx:
        tokens, patch_start = model.encode(views, cond_flags)

    def features(start, end):
        return model.gs_head.forward_chunk(tokens, views["img"], patch_start, start, end)

    # 把渲染器里 merge_voxels 的入参存下来：反传后要逐分支查梯度，也要量 merge 单独耗时。
    renderer = model.gs_renderer
    orig_merge = getattr(renderer, "_orig_merge", renderer.merge_voxels)

    def measured_merge(splats, keep):
        stash["raw_splats"] = splats
        stash["keep"] = keep
        # 非叶张量默认不保留 .grad，诊断分支梯度要先挂 retain_grad
        for k in ("quats", "scales", "opacities", "sh", "weights"):
            if splats[k].requires_grad:
                splats[k].retain_grad()
        with chrono("merge_s"):
            out = merge_voxels_functional(splats, keep, renderer.voxel_size)
        stash["merged_n"] = int(out["means"][0].shape[0])
        return out

    renderer.merge_voxels = measured_merge
    try:
        with chrono("gs_render_s"), torch.autocast(device_type=views["img"].device.type,
                                                   enabled=False):
            return renderer(views, geometry, features)
    finally:
        renderer.merge_voxels = orig_merge


def render_splats(preds, dev, hw, bg=0.0, eps2d=0.01, dbg=None):
    """可导光栅化。约定同 03_render_eval.py：inv(c2w) 当 viewmats，colors 传 SH DC。

    hw 必须由调用方从输入图取 —— 不要从 `preds['depth']` 推：DPT 的 depth 是 5 维
    `[B,S,H,W,1]`（末尾还挂一个单通道），`shape[-2:]` 会得到 `(H,1)`。
    """
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

    if dbg is not None:
        dbg.update(depth_shape=list(preds["depth"].shape), means=list(means.shape),
                   c2w=list(c2w.shape), K=list(K.shape), h=h, w=w)

    colors = sh.unsqueeze(0).expand(C, -1, -1, -1).contiguous()
    backgrounds = torch.full((C, 3), float(bg), device=dev, dtype=torch.float32)
    rgb, alpha, _ = rasterization(
        means=means, quats=quats, scales=scales, opacities=opac,
        colors=colors, viewmats=viewmats, Ks=K, width=w, height=h,
        sh_degree=0, eps2d=eps2d, near_plane=0.01, far_plane=400.0,
        backgrounds=backgrounds, render_mode="RGB")
    return rgb, alpha


def img_hw(views):
    return (int(views["img"].shape[-2]), int(views["img"].shape[-1]))


def splat_fields(preds):
    sp = preds["splats"]
    return {k: sp[k][0] for k in ("means", "quats", "scales", "opacities", "sh")}


def param_checksum(mod):
    s = 0.0
    with torch.no_grad():
        for p in mod.parameters():
            s += float(p.detach().float().sum())
    return s


# --------------------------------------------------------------------------------------
# 各阶段
# --------------------------------------------------------------------------------------
def stage_fwd(model, views, geometry, cond_flags, dev, chrono, report):
    """训练模式前向 vs 冻结推理前向。官方 no_grad 路径为基准。"""
    stash = {}
    with torch.no_grad():
        ref = model(views, geometry, cond_flags=cond_flags)
    got = forward_train(model, views, geometry, cond_flags, chrono, stash)
    a, b = splat_fields(ref), splat_fields(got)
    diffs = {}
    for k in a:
        d = (a[k].double() - b[k].double()).abs()
        diffs[k] = dict(max_abs=float(d.max()), mean_abs=float(d.mean()),
                        ref_absmean=float(a[k].double().abs().mean()))
    hw = img_hw(views)
    rgb_ref, _ = render_splats(ref, dev, hw)
    dbg = {}
    rgb_got, _ = render_splats(got, dev, hw, dbg=dbg)
    rgb_mae = float((rgb_ref - rgb_got).abs().mean())
    report["fwd_equiv"] = dict(
        fields=diffs, render_rgb_mae=rgb_mae, render_shapes=[list(rgb_ref.shape), list(rgb_got.shape)],
        render_dbg=dbg, target_shape=list(views["img"].shape),
        max_field_absdiff=max(v["max_abs"] for v in diffs.values()),
        n_merged=int(got["splats"]["means"][0].shape[0]),
    )
    print(f"[fwd] 逐字段 max|Δ| = {report['fwd_equiv']['max_field_absdiff']:.3e}，"
          f"渲染 RGB MAE = {rgb_mae:.3e}，合并后 {report['fwd_equiv']['n_merged']} 颗")
    return got


def stage_sweep(model, trainable, views, geometry, cond_flags, dev, report, lrs,
                n_steps=6, restore=None):
    """找能让 scale 头不炸的 lr：查 loss 是否单调、σ 是否守在 GSN 尺度附近、点数是否归零。"""
    target = views["img"][0].permute(0, 2, 3, 1).float()
    sigma0 = None
    rows = []
    for lr in lrs:
        if restore:
            restore()
        opt = torch.optim.Adam([p for _, p in trainable], lr=lr)
        curve = []
        blew = False
        for step in range(n_steps):
            opt.zero_grad(set_to_none=True)
            stash = {}
            preds = forward_train(model, views, geometry, cond_flags, Chrono(dev), stash)
            rgb, alpha = render_splats(preds, dev, img_hw(views))
            loss = F.mse_loss(rgb, target)
            raw = stash["raw_splats"]
            sig = float(raw["scales"].mean())
            sigma0 = sig if sigma0 is None else sigma0
            curve.append(dict(step=step, loss=float(loss), mean_scale=sig,
                              n_merged=int(stash["merged_n"]),
                              alpha_med=float(alpha.median())))
            print(f"[sweep] lr {lr:g} step {step}  loss {float(loss):.6f}"
                  f"  σ {sig:.5f}  n {stash['merged_n']}")
            if not np.isfinite(float(loss)) or int(stash["merged_n"]) == 0:
                blew = True
                break
            loss.backward()
            opt.step()
        losses = [c["loss"] for c in curve]
        rows.append(dict(lr=lr, curve=curve, diverged=blew,
                         loss_first=losses[0], loss_last=losses[-1],
                         best=min(losses), sigma_final=curve[-1]["mean_scale"]))
    report["lr_sweep"] = dict(sigma_at_step0=sigma0, rows=rows)
    print("[sweep] 小结：" + "；".join(
        f"lr {r['lr']:g} → {r['loss_last']:.5f} σ{r['sigma_final']:.4f}"
        f"{' (炸)' if r['diverged'] else ''}" for r in rows))


def stage_step1(model, teacher, trainable, views, geometry, cond_flags, dev, chrono, report,
                lr=1e-3):
    """1 个 optimizer step：分支梯度、参数 delta、教师不变。"""
    opt = torch.optim.Adam([p for _, p in trainable], lr=lr)
    target = views["img"][0].permute(0, 2, 3, 1).float()
    stash = {}
    before = {n: p.detach().clone() for n, p in trainable}
    enc_before = param_checksum(model.visual_geometry_transformer)
    reset_peak(dev)

    opt.zero_grad(set_to_none=True)
    preds = forward_train(model, views, geometry, cond_flags, chrono, stash)
    with chrono("raster_fwd_s"):
        rgb, alpha = render_splats(preds, dev, img_hw(views))
    loss = F.mse_loss(rgb, target)
    with chrono("bwd_s"):
        loss.backward()

    # 分支梯度：raw_splats 是卷积解码后、合并前的张量
    raw = stash["raw_splats"]
    branch = {}
    for k in ("quats", "scales", "opacities", "sh", "weights"):
        g = raw[k].grad
        branch[k] = None if g is None else dict(
            finite=bool(torch.isfinite(g).all()), absmax=float(g.abs().max()),
            nonzero=int((g != 0).sum()))
    conv = model.gs_renderer.decoder.gs_head[-1]
    gw = conv.weight.grad
    names = ["quat.x", "quat.y", "quat.z", "quat.w",
             "scale.0", "scale.1", "scale.2", "opacity",
             "sh.r", "sh.g", "sh.b", "merge_weight"]
    row_grad = {names[i]: (None if gw is None else float(gw[i].abs().sum())) for i in range(12)}

    gnorms = {n: (None if p.grad is None else float(p.grad.abs().sum()))
              for n, p in trainable}
    frozen_grad_tensors = sum(1 for p in model.visual_geometry_transformer.parameters()
                              if p.grad is not None)
    frozen_checksum_delta = param_checksum(model.visual_geometry_transformer) - enc_before

    with chrono("opt_s"):
        opt.step()

    delta = {}
    for n, p in trainable:
        d = (p.detach() - before[n]).abs()
        delta[n] = dict(max=float(d.max()), mean=float(d.mean()), numel=int(p.numel()))
    changed = sum(1 for n in delta if delta[n]["max"] > 0)

    report["step1"] = dict(
        lr=lr, loss=float(loss), psnr_self=float(10 * np.log10(1.0 / max(float(loss), 1e-12))),
        alpha_median=float(alpha.median()), n_merged=int(stash["merged_n"]),
        branch_grad=branch, conv_row_grad=row_grad,
        trainable_grad_abs_sum=gnorms,
        trainable_params=int(sum(p.numel() for _, p in trainable)),
        trainable_changed=changed, trainable_total=len(trainable),
        frozen_encoder_grad_tensors=frozen_grad_tensors,
        frozen_encoder_checksum_delta=frozen_checksum_delta,
        teacher_checksum_delta=param_checksum(teacher) - report["_teacher_sum"],
        timings=dict(chrono.t), mem=mem_stats(dev),
    )
    print(f"[step1] loss {float(loss):.6f}（自洽 {report['step1']['psnr_self']:.2f} dB）"
          f"｜可训 {report['step1']['trainable_params']} 参数，{changed}/{len(trainable)} 组已更新"
          f"｜encoder 有梯度的张量 {frozen_grad_tensors}，教师 ΣΔ {report['step1']['teacher_checksum_delta']:.3e}")
    print("[step1] 分支梯度 " + "，".join(
        f"{k}={'None' if v is None else '%.2e' % v['absmax']}" for k, v in branch.items()))
    print("[step1] 12 通道行梯度 " + "，".join(
        f"{k}={'-' if v is None else '%.2e' % v}" for k, v in row_grad.items()))


def stage_steps(model, trainable, views, geometry, cond_flags, dev, chrono, report,
                n_steps, lr=1e-3, log_at=(0, 10, 100), log_every=0):
    """连续 n_steps，记录 loss/计时/显存；判据：MSE 较 step 0 降 ≥10% 且无 NaN。"""
    opt = torch.optim.Adam([p for _, p in trainable], lr=lr)
    target = views["img"][0].permute(0, 2, 3, 1).float()
    curve = []
    reset_peak(dev)
    t_start = time.perf_counter()
    for step in range(n_steps + 1):
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        stash = {}
        preds = forward_train(model, views, geometry, cond_flags, chrono, stash)
        rgb, alpha = render_splats(preds, dev, img_hw(views))
        loss = F.mse_loss(rgb, target)
        if step < n_steps:
            loss.backward()
            opt.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        lf = float(loss)
        show = (step in log_at) or (log_every and step % log_every == 0) or step == n_steps
        if show:
            raw = stash["raw_splats"]
            entry = dict(step=step, loss=lf,
                         psnr=float(10 * np.log10(1.0 / max(lf, 1e-12))),
                         step_s=dt, n_merged=int(stash["merged_n"]),
                         alpha_med=float(alpha.median()), rgb_mean=float(rgb.mean()),
                         mean_weight=float(raw["weights"].mean()),
                         mean_opacity=float(raw["opacities"].mean()),
                         mean_scale=float(raw["scales"].mean()),
                         mean_sh=float(raw["sh"].mean()),
                         finite=bool(np.isfinite(lf)))
            curve.append(entry)
            print(f"[steps] step {step:3d}  loss {lf:.6f}  自洽 {entry['psnr']:.2f} dB"
                  f"  {dt*1000:.0f} ms  α{entry['alpha_med']:.3f}"
                  f"  w{entry['mean_weight']:.3f}  σ{entry['mean_scale']:.4f}"
                  f"  n={entry['n_merged']}")
        if not np.isfinite(lf):
            print("❌ loss 出现 NaN/Inf，停止")
            break
    total = time.perf_counter() - t_start
    base = curve[0]["loss"]
    last = curve[-1]["loss"]
    ok = bool(np.isfinite(last) and last <= 0.9 * base)
    report["steps"] = dict(
        n_steps=n_steps, lr=lr, curve=curve,
        loss_step0=base, loss_last=last, drop_pct=100.0 * (base - last) / base,
        criterion_drop_ge_10pct=ok,
        total_wall_s=total, mean_step_s=(total / max(n_steps, 1)),
        timings=dict(chrono.t), mem=mem_stats(dev),
    )
    print(f"[steps] {n_steps} 步：loss {base:.6f} → {last:.6f}（{report['steps']['drop_pct']:.1f}%），"
          f"均值 {report['steps']['mean_step_s']*1000:.0f} ms/step，判据 {'PASS' if ok else 'FAIL'}")
    return opt


def stage_resume(model, trainable, views, geometry, cond_flags, dev, report, opt, ckpt_path,
                 lr=1e-3):
    """保存 → 新优化器加载 → 再走一步，验证可续训。"""
    ck = Path(ckpt_path)
    ck.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "trainable": {n: p.detach().cpu() for n, p in trainable},
        "optimizer": opt.state_dict(),
        "rng": dict(torch=torch.get_rng_state(),
                    cuda=torch.cuda.get_rng_state_all() if dev.type == "cuda" else None,
                    numpy=np.random.get_state(), python=random.getstate()),
    }
    torch.save(state, ck)
    size_MB = ck.stat().st_size / 2 ** 20

    opt2 = torch.optim.Adam([p for _, p in trainable], lr=lr)
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    with torch.no_grad():
        for n, p in trainable:
            p.copy_(blob["trainable"][n].to(p.device))
    opt2.load_state_dict(blob["optimizer"])
    torch.set_rng_state(blob["rng"]["torch"])
    if dev.type == "cuda" and blob["rng"]["cuda"] is not None:
        torch.cuda.set_rng_state_all(blob["rng"]["cuda"])
    np.random.set_state(blob["rng"]["numpy"])
    random.setstate(blob["rng"]["python"])

    for g in opt2.param_groups:
        for p in g["params"]:
            p.grad = None
    target = views["img"][0].permute(0, 2, 3, 1).float()
    preds = forward_train(model, views, geometry, cond_flags, Chrono(dev), {})
    rgb, _ = render_splats(preds, dev, img_hw(views))
    loss = F.mse_loss(rgb, target)
    loss.backward()
    opt2.step()
    lf = float(loss)
    n_state = sum(int(t.numel()) for st in opt2.state.values() for t in st.values()
                  if torch.is_tensor(t))
    report["resume"] = dict(
        ckpt=str(ck), ckpt_MB=size_MB, resumed_step_loss=lf,
        finite=bool(np.isfinite(lf)), optimizer_state_tensors=int(n_state),
        optimizer_state_MB=n_state * 4 / 2 ** 20,
        trainable_MB=sum(p.numel() for _, p in trainable) * 4 / 2 ** 20,
    )
    print(f"[resume] 恢复后一步 loss {lf:.6f}（finite={np.isfinite(lf)}），"
          f"ckpt {size_MB:.1f} MB，优化器状态 {n_state*4/2**20:.1f} MB")


# --------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=str(Path.home() /
                    "output/splatanything_results/_scenes/jiajia_381/images"))
    ap.add_argument("--res", type=int, default=504, help="长边；必须是 14 的倍数")
    ap.add_argument("--pose-mode", action="store_true", help="pose 条件模式（需 prior_camera.json）")
    ap.add_argument("--stages", default="all",
                    help="逗号分隔：fwd,step1,steps,resume / all")
    ap.add_argument("--n-steps", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=0, help="每 N 步打一条诊断（0=只打 log_at）")
    ap.add_argument("--sweep-lr", default="",
                    help="逗号分隔的 lr 列表，配合 --stages sweep 找不炸的学习率")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default=str(Path.home() /
                    "output/splatanything_results/train_probe.json"))
    ap.add_argument("--ckpt", default=str(Path.home() /
                    "output/splatanything_results/train_probe_ckpt.pt"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    if args.res % 14:
        raise SystemExit("❌ --res 必须是 14 的倍数")

    repo = official_repo()
    sys.path.insert(0, str(repo))
    dev = torch.device(args.device)
    stages = set(s.strip() for s in args.stages.split(",") if s.strip())
    if "all" in stages:
        stages = {"fwd", "step1", "steps", "resume"}
    sweep_lrs = [float(x) for x in args.sweep_lr.replace(",", " ").split() if x]

    print(f"[Env] {repo} @ {os.popen(f'git -C {repo} rev-parse --short HEAD').read().strip()}")
    print(f"[Env] torch {torch.__version__} cuda {torch.version.cuda} "
          f"dev {torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'}")

    t_build = time.perf_counter()
    cfg, model, teacher = build_models(repo, dev, pose_mode=args.pose_mode)
    views, scene_name, img_paths = build_inputs(args.scene, args.res, dev,
                                                pose_mode=args.pose_mode)
    trainable = set_trainable(model)
    # 保留官方 bound method：forward_train 会临时替换 merge_voxels 以计时
    model.gs_renderer._orig_merge = model.gs_renderer.merge_voxels
    teacher.eval().requires_grad_(False)
    model.eval()   # 无 BN/dropout；eval 只关掉 drop_path，保证确定性（解码头本身无随机性）
    cond_flags = [1, 0, 1] if args.pose_mode else [0, 0, 0]

    report = dict(
        base_commit=os.popen(f"git -C {repo} rev-parse --short HEAD").read().strip(),
        scene=scene_name, n_views=len(img_paths), resolution=args.res,
        pose_mode=bool(args.pose_mode), cond_flags=cond_flags,
        torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
        trainable=[n for n, _ in trainable],
        trainable_params=int(sum(p.numel() for _, p in trainable)),
        stage_config=dict(lr=args.lr, n_steps=args.n_steps),
        build_s=time.perf_counter() - t_build,
    )
    print(f"[Model] 可训 {report['trainable_params']:,} 参数：{report['trainable']}")

    chrono = Chrono(dev)
    with torch.no_grad():
        geometry = teacher(views, cond_flags=cond_flags)
    report["geometry"] = dict(depth_shape=list(geometry["depth"].shape))

    # 每个阶段都从同一份初始权重出发，避免前面的 step 污染后面的成本读数
    snap = {n: p.detach().clone() for n, p in trainable}

    def restore():
        with torch.no_grad():
            for n, p in trainable:
                p.copy_(snap[n])
                p.grad = None

    if "sweep" in stages:
        stage_sweep(model, trainable, views, geometry, cond_flags, dev, report,
                    sweep_lrs or [1e-3, 3e-4, 1e-4, 3e-5, 1e-5], restore=restore)
    if "fwd" in stages:
        restore()
        chrono.t.clear()
        stage_fwd(model, views, geometry, cond_flags, dev, chrono, report)
    if "step1" in stages:
        restore()
        chrono.t.clear()
        reset_peak(dev)
        report["_teacher_sum"] = param_checksum(teacher)
        stage_step1(model, teacher, trainable, views, geometry, cond_flags, dev, chrono,
                    report, args.lr)
    opt = None
    if "steps" in stages:
        restore()
        chrono.t.clear()
        opt = stage_steps(model, trainable, views, geometry, cond_flags, dev, chrono, report,
                          args.n_steps, args.lr, log_every=args.log_every)
    if "resume" in stages:
        opt = opt or torch.optim.Adam([p for _, p in trainable], lr=args.lr)
        report["_teacher_sum"] = report.get("_teacher_sum", param_checksum(teacher))
        stage_resume(model, trainable, views, geometry, cond_flags, dev, report, opt,
                     args.ckpt, args.lr)

    report.pop("_teacher_sum", None)

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[Done] → {out}")


if __name__ == "__main__":
    main()
