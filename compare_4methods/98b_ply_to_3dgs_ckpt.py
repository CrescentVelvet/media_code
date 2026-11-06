#!/usr/bin/env python3
"""P1-1：把 DA3 前馈高斯 PLY 转成 3DGS 的 `--start_checkpoint` 权重。

## 为什么不能直接给 PLY
原版 3DGS 的 `train.py` 有两个入口能带初值：
  A. `Scene.__init__` 里 `if self.loaded_iter:` → `gaussians.load_ply(...)`，
     但 `train.py` 调用 `Scene(dataset, gaussians)` 时**没有传 load_iteration**，
     所以这条分支永远不走（走的是 COLMAP 点云 `create_from_pcd`）。
  B. `--start_checkpoint` → `torch.load(...)` 得到 `(model_args, first_iter)`，
     再 `gaussians.restore(model_args, opt)`，其中 model_args 是 `capture()` 的
     12 元组（含 optimizer state_dict）。

所以这里造 B 需要的 `.pth`，不动第三方仓一行代码。

## 三处必须对齐的约定（每一处错了都会静默出坏结果）
1. **scale / opacity 的参数空间**：DA3 导出时 `scales.log()`（gsply_helpers.py:81）、
   `opacity` 走 `inverse_sigmoid`（`inv_opacity=True`）。3DGS 的 `_scaling`/`_opacity`
   恰好也是 log / logit 空间，**原样搬运，不要再激活一次**。
2. **SH 阶数**：DA3 的 gs_adapter 是 `sh_degree=2` → 9 个系数（DC + 8 阶）。3DGS
   内部写死 `sh_degree=3` → 需要 15 个 rest 系数，`load_ply` 有
   `assert len(extra_f_names) == 3*(max_sh_degree+1)**2-3`。所以把 8 阶填进前 8 个槽、
   后 7 个补 0（等价于 sh_degree=2 的表达能力，同时留出高阶学习空间）。
   布局：DA3 PLY 的 f_rest 是**通道优先**（`harmonics[...,1:].flatten(start_dim=1)`，
   即 [R 的 8 个, G 的 8 个, B 的 8 个]）；3DGS 的 `_features_rest` 是 (N, 15, 3)，
   `save_ply` 里 `transpose(1,2).flatten()` 后同样是通道优先 → 逐通道对位填充。
3. **optimizer 的 per-group lr**：`restore()` 会 `self.optimizer.load_state_dict(opt_dict)`，
   这会把 `training_setup()` 刚设好的 lr **整体覆盖成 opt_dict 里的值**。
   若这里填 0，除 xyz（每步被 `update_learning_rate` 重设）之外的
   f_dc / f_rest / opacity / scaling / rotation **全部不更新**，训练看起来正常但什么都不学。
   所以按 3DGS 的默认超参填入真实 lr（与官方 checkpoint 的等价行为）。
4. **尺度口径**（E-01）：cameras.npz 是米制，官方 gs_ply 是 DA3 对齐前的坐标系，
   差一个 `scale_factor`。本脚本开头会断言 npz 里 `gs_scale_applied == 1`（由 98/97
   在导出时折算并写入）。老 npz 直接报错退出，不做兼容——静默产出错尺度 ckpt 的代价太大。

用法（WSL，任意装了 torch 的 env；需要在 GPU 上保存，官方 `torch.load` 无 map_location）：
  PLY=~/output/da3_gs_results/佳佳抱枕_p1/gs_ply/0000.ply \
  NPZ=~/output/da3_gs_results/佳佳抱枕_p1/cameras.npz \
  OUT_DIR=~/output/da3_gs_results/佳佳抱枕_p1/ffinit_ckpt \
  ~/miniconda3/envs/querysplat/bin/python compare_4methods/98b_ply_to_3dgs_ckpt.py
"""
import json
import os
import re
from pathlib import Path

import numpy as np
import torch

# 3DGS 的 arguments/__init__.py 默认值。restore 会用它覆盖 training_setup 设的 lr，
# 所以这里必须是真实值而不是 0（见模块 docstring 第 3 条）。
LR_BY_GROUP = {
    "xyz": 0.0,            # 每步被 update_learning_rate 重设，这里填什么都会被覆盖
    "f_dc": 0.0025,
    "f_rest": 0.0025 / 20.0,
    "opacity": 0.025,
    "scaling": 0.005,
    "rotation": 0.001,
}
GROUP_ORDER = ["xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation"]
MAX_SH_DEGREE = 3  # 3DGS 内部写死


def log(msg):
    print(msg, flush=True)


def read_ply(path):
    """读 binary_little_endian PLY，返回 (names, float32 数组 (N,P))。"""
    with open(path, "rb") as f:
        head_raw = f.read(8192)
        end = head_raw.index(b"end_header") + len(b"end_header") + 1
        head = head_raw[:end].decode("ascii", "replace")
        n = int(re.search(r"element vertex (\d+)", head).group(1))
        names = [l.split()[2] for l in head.splitlines() if l.startswith("property")]
        f.seek(end)
        arr = np.fromfile(f, dtype=np.float32).reshape(n, len(names))
    return names, arr


def camera_extent(npz_path):
    """与 3DGS getNerfppNorm 同口径：radius = 1.1 * max(|cam_center - mean|)。"""
    d = np.load(npz_path)
    e = d["extrinsics"]
    C = np.einsum("nij,nj->ni", -e[:, :3, :3].transpose(0, 2, 1), e[:, :3, 3])
    return float(1.1 * np.linalg.norm(C - C.mean(axis=0), axis=1).max())


def build_opt_dict(n):
    """造一个结构正确、state 为空的 Adam state_dict（Adam 首次 step 时惰性初始化）。"""
    # 必须每个 group 一个独立 tensor：`[t] * 6` 会让 6 个 group 共享同一 param，
    # Adam 直接报 "some parameters appear in more than one parameter group"
    dummy = [torch.zeros(1) for _ in GROUP_ORDER]
    opt = torch.optim.Adam([{"params": [p]} for p in dummy], lr=0.0, eps=1e-15)
    d = opt.state_dict()
    for g, name in zip(d["param_groups"], GROUP_ORDER):
        # 必须补回 "name"：torch 的 load_state_dict 会用 saved_groups 的键**完全重建**
        # param_groups，标准 Adam 的 group 没有 name 键 → 3DGS 的
        # `if param_group["name"] == "xyz"` 直接 KeyError
        g["name"] = name
        g["lr"] = LR_BY_GROUP[name]
    d["state"] = {}
    return d


def main():
    ply_path = Path(os.path.expanduser(os.environ["PLY"]))
    npz_path = Path(os.path.expanduser(os.environ["NPZ"]))
    out_dir = Path(os.path.expanduser(os.environ["OUT_DIR"]))
    min_opacity = float(os.environ.get("MIN_OPACITY", "0.005"))
    max_dist = float(os.environ.get("MAX_DIST", "5.0"))   # 0 = 不按距离剪
    max_points = int(os.environ.get("MAX_POINTS", "0"))   # 0 = 不限
    save_ply = os.environ.get("SAVE_PLY", "1") == "1"
    out_dir.mkdir(parents=True, exist_ok=True)

    names, arr = read_ply(ply_path)
    nm = {k: i for i, k in enumerate(names)}
    n_raw = arr.shape[0]
    log(f"📥 {ply_path.name}: {n_raw:,} 点, {len(names)} 属性")

    # ── 尺度口径闸门（E-01，2026-11-06）──────────────────────────────────────
    # cameras.npz 的 extrinsics/depth 是米制，而官方导出的 gs_ply 在 DA3 对齐前的坐标系，
    # 差一个 scale_factor（本例 1.81）。混用会让点云只有真实尺寸的 0.55，
    # 3DGS 5k 步挪不动 0.9 m 的错位，只能长出第二套壳 → 重影。
    # 修法在 98/97（导出时折到米制），这里只做校验：宁可报错，不要静默产出错尺度的 ckpt。
    _d = np.load(npz_path)
    if "gs_scale_applied" not in _d.files:
        raise SystemExit(
            "❌ cameras.npz 是老版本（无 gs_scale_applied 字段），无法判断 PLY 是否为米制。\n"
            "   请用修复后的 98_da3_ff_init.py 重新导出（PLY 与 npz 必须同尺度，见 ERRATA.md E-01）。")
    if int(_d["gs_scale_applied"]) != 1:
        raise SystemExit(
            f"❌ PLY 未折算到米制（gs_scale_applied=0，DA3 scale_factor="
            f"{float(_d['scale_factor']) if 'scale_factor' in _d.files else float('nan'):.4f}）。\n"
            "   重新导出时不要设 GS_SCALE_MODE=raw。")
    if "scale_factor" in _d.files:
        log(f"📏 尺度口径 OK：PLY 已 ×scale_factor={float(_d['scale_factor']):.4f} 折到米制")

    xyz = arr[:, [nm["x"], nm["y"], nm["z"]]].astype(np.float32)
    op_logit = arr[:, nm["opacity"]].astype(np.float32)
    scaling = arr[:, [nm["scale_0"], nm["scale_1"], nm["scale_2"]]].astype(np.float32)
    rotation = arr[:, [nm["rot_0"], nm["rot_1"], nm["rot_2"], nm["rot_3"]]].astype(np.float32)
    f_dc = arr[:, [nm["f_dc_0"], nm["f_dc_1"], nm["f_dc_2"]]].astype(np.float32)

    rest_names = sorted(
        [k for k in names if k.startswith("f_rest_")], key=lambda s: int(s.split("_")[-1])
    )
    d_sh_rest = len(rest_names) // 3  # 每通道的 rest 数（DA3 sh_degree=2 → 8）
    da3_rest = arr[:, [nm[k] for k in rest_names]].astype(np.float32) if rest_names else None
    log(f"🔢 SH: DA3 rest {d_sh_rest}/通道（sh_degree={int(round((d_sh_rest+1) ** 0.5)) - 1}）→ 3DGS 需 15/通道，后 {15-d_sh_rest} 阶补 0")

    # ── 剪枝 ────────────────────────────────────────────────────────────────
    op = 1.0 / (1.0 + np.exp(-op_logit.astype(np.float64)))
    keep = op > min_opacity
    log(f"✂️ opacity>{min_opacity}: {keep.sum():,} / {n_raw:,} ({(keep.sum()/n_raw*100):.1f}%)，"
        f"透明度权重覆盖 {op[keep].sum()/op.sum()*100:.2f}%")

    if max_dist > 0:
        d = np.load(npz_path)
        e = d["extrinsics"]
        C = np.einsum("nij,nj->ni", -e[:, :3, :3].transpose(0, 2, 1), e[:, :3, 3])
        # 只跟最近相机比：离所有相机都远 = 不可见的 floater
        dd = np.sqrt(((xyz[:, None, :] - C[None, :, :]) ** 2).sum(-1)).min(1)
        m = dd < max_dist
        log(f"✂️ 距最近相机<{max_dist}m: 再剪 {(~m & keep).sum():,} 点")
        keep &= m

    if max_points and keep.sum() > max_points:
        # 按 opacity 保序保留 top-K（透明度高的点对成像贡献大）
        idx_keep = np.where(keep)[0]
        order = np.argsort(-op[idx_keep])[:max_points]
        keep = np.zeros(n_raw, dtype=bool)
        keep[idx_keep[order]] = True
        log(f"✂️ 上限 {max_points:,} 点：按 opacity 保留 top-K")

    N = int(keep.sum())
    xyz, op_logit, scaling, rotation, f_dc = (
        xyz[keep], op_logit[keep], scaling[keep], rotation[keep], f_dc[keep]
    )
    if da3_rest is not None:
        da3_rest = da3_rest[keep]
    log(f"✅ 保留 {N:,} 点 ({N/n_raw*100:.1f}%)")

    # ── 组装 3DGS 参数（注意参数空间，见 docstring） ─────────────────────────
    # ⚠️ 必须包成 nn.Parameter(requires_grad=True)：
    #   `restore()` 是**直接赋值** `self._xyz = model_args[1]`，不做任何包装。
    #   官方 checkpoint 里存的就是 nn.Parameter；若这里存普通 tensor
    #   （torch.from_numpy 得到的 requires_grad=False），autograd 不追踪 →
    #   梯度永远 None → Adam 一步也不更新，训练 loss 卡住、存出的 PLY 与初值
    #   逐字节相同（2026-11-06 实测踩坑，四个 iteration 的 PLY md5 全相同）。
    def P(t):
        return torch.nn.Parameter(t.contiguous(), requires_grad=True)

    t_xyz = P(torch.from_numpy(xyz).cuda())
    t_dc = P(torch.from_numpy(f_dc).cuda().unsqueeze(1))               # (N,1,3)
    rest = torch.zeros((N, 15, 3), dtype=torch.float32, device="cuda")
    if da3_rest is not None:
        r = torch.from_numpy(da3_rest).cuda()                          # (N, 3*d_sh) 通道优先
        for c in range(3):
            k = min(d_sh_rest, 15)
            rest[:, :k, c] = r[:, c * d_sh_rest: c * d_sh_rest + k]
    t_rest = P(rest)
    t_scaling = P(torch.from_numpy(scaling).cuda())
    t_rotation = P(torch.from_numpy(rotation).cuda())
    t_opacity = P(torch.from_numpy(op_logit).cuda().unsqueeze(1))      # (N,1) logit

    extent = camera_extent(npz_path)
    # 初始 active_sh_degree。默认 0 = 与 3DGS 原生一致（每 1000 步升一阶，2k 步里
    # SH rest 只有后 1000 步参与渲染）。做「SH 初值值多少」的 A/B 时必须设成 2，
    # 否则测的是"半程 SH"而不是"有/无 SH 初值"（见 P0_EXPERIMENTS.md R-2.0）。
    sh_deg_init = int(os.environ.get("SH_DEGREE_INIT", "0"))
    model_args = (
        sh_deg_init,                                                   # active_sh_degree
        t_xyz,
        t_dc,
        t_rest,
        t_scaling,
        t_rotation,
        t_opacity,
        torch.zeros(N, device="cuda"),                                 # max_radii2D
        torch.zeros((N, 1), device="cuda"),                            # xyz_gradient_accum
        torch.zeros((N, 1), device="cuda"),                            # denom
        build_opt_dict(N),
        extent,                                                        # spatial_lr_scale
    )

    ckpt = out_dir / "chkpnt_ffinit.pth"
    torch.save((model_args, 0), str(ckpt))
    log(f"💾 {ckpt}  ({ckpt.stat().st_size/1024/1024:.1f} MB)  spatial_lr_scale={extent:.3f}")

    # 同时落一份 3DGS 格式 PLY（45 个 f_rest），便于外部工具检查/渲染，训练不用
    if save_ply:
        p = out_dir / "point_cloud" / "iteration_0" / "point_cloud.ply"
        p.parent.mkdir(parents=True, exist_ok=True)
        attrs = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
        attrs += [f"f_rest_{i}" for i in range(45)]
        attrs += ["opacity", "scale_0", "scale_1", "scale_2",
                  "rot_0", "rot_1", "rot_2", "rot_3"]
        dt = np.dtype([(a, "f4") for a in attrs])
        el = np.empty(N, dtype=dt)
        el["x"], el["y"], el["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        el["nx"] = el["ny"] = el["nz"] = 0.0
        el["f_dc_0"], el["f_dc_1"], el["f_dc_2"] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]
        rest_ply = rest.detach().cpu().numpy().transpose(0, 2, 1).reshape(N, 45)  # 通道优先
        for i in range(45):
            el[f"f_rest_{i}"] = rest_ply[:, i]
        el["opacity"] = op_logit
        el["scale_0"], el["scale_1"], el["scale_2"] = scaling[:, 0], scaling[:, 1], scaling[:, 2]
        el["rot_0"], el["rot_1"], el["rot_2"], el["rot_3"] = (
            rotation[:, 0], rotation[:, 1], rotation[:, 2], rotation[:, 3]
        )
        from plyfile import PlyData, PlyElement
        PlyData([PlyElement.describe(el, "vertex")]).write(str(p))
        log(f"💾 {p}  ({p.stat().st_size/1024/1024:.1f} MB)")

    summary = {
        "source_ply": str(ply_path),
        "cameras_npz": str(npz_path),
        "n_raw": n_raw,
        "n_kept": N,
        "keep_ratio": round(N / n_raw, 4),
        "min_opacity": min_opacity,
        "max_dist_m": max_dist,
        "max_points": max_points,
        "da3_sh_rest_per_channel": d_sh_rest,
        "sh_degree_init": sh_deg_init,
        "spatial_lr_scale": round(extent, 4),
        "lr_by_group": LR_BY_GROUP,
        "checkpoint": str(ckpt),
    }
    with open(out_dir / "ckpt_summary.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    log("✅ done")


if __name__ == "__main__":
    main()
