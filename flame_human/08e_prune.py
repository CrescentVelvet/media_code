#!/usr/bin/env python3
"""08e_prune.py — 三分支多视角投票剪枝（治人物周围黑雾 floater / 幽灵点）。

诊断依据（2026-09-10 分支分解，a70cca3）：人物周围黑雾来自三分支的错位
暗色高斯叠加（人物黑衣黑发，错位即呈暗色）：
  - head : 08 用 person mask 全区域监督 + 自由高斯 _free_can → 往身体长、
           并 densify 出放射状细长 floater（环带暗化 21-36%，贡献最大）
  - body : 59 帧静态平均 → 深色夹克/头发在真实轮廓外涂抹（暗化 11-17%）
  - scene: 08d 的 LAMBDA_INSIDE 罚「亮度」→ scene 在 person 区涂黑
           （inside 亮度 0.022）而非变透明

做法（08c 逻辑的推广，纯后处理零训练成本）：
  投票 = 该高斯投影到 59 个训练相机，统计「落在 person mask 内 / 在画面内」
  的比例 score。
  - MODE=body : 保留 score >= KEEP_RATIO（正常身体表面点多数视角在 mask 内；
                悬空 floater 多数视角在 mask 外）；另删头盒内点（头部归 head）
  - MODE=head : 保留 score >= KEEP_RATIO（脸/头发在 mask 内；放射 floater 在外）
  - MODE=scene: 反向——删 score > DROP_RATIO 的点（多数视角在 mask 内 =
                跟着人走的幽灵点；真背景点多数视角在 mask 外）
  三种模式都删「59 帧全无观测」的点（不可见，纯垃圾）。

head 用逐帧 FLAME 形变后的世界坐标投票（head 跟随头部运动，静态坐标无意义）。

Env: MODE(head|body|scene) / PID / RESULTS_DIR / SOURCE_DIR / MASKS_DIR /
     ALIGN_JSON / IN_PLY / IN_CKPT / OUT_PLY / OUT_CKPT /
     KEEP_RATIO / DROP_RATIO / HEAD_BBOX_MARGIN / DROP_UNSEEN
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colmap_io import read_cameras, proj_matrix  # noqa: E402


def log(m):
    print(m, flush=True)


def vote(xyz_per_frame, stems, view_by, masks_dir, pid2):
    """逐帧投影投票 → (inside, front) 计数（长度 N）。"""
    from PIL import Image
    n = xyz_per_frame[0].shape[0]
    inside = np.zeros(n, np.int32)
    front = np.zeros(n, np.int32)
    for s, xyz in zip(stems, xyz_per_frame):
        vv = view_by.get(s)
        if vv is None:
            continue
        mp = masks_dir / f"{s}.p{pid2}.alpha.png"
        if not mp.exists():
            mp = masks_dir / f"{s}.p{pid2}.mask.png"
        if not mp.exists():
            continue
        m = np.asarray(Image.open(mp).convert("L").resize(
            (vv["W"], vv["H"]), Image.LANCZOS), dtype=np.float32) / 255.0
        P, _ = proj_matrix(vv)
        pr = np.c_[xyz, np.ones(len(xyz))] @ P.T          # (N,3)
        z = pr[:, 2]
        ok = z > 1e-6
        u = np.where(ok, pr[:, 0] / np.clip(z, 1e-6, None), -1).astype(int)
        w = np.where(ok, pr[:, 1] / np.clip(z, 1e-6, None), -1).astype(int)
        inimg = ok & (u >= 0) & (u < vv["W"]) & (w >= 0) & (w < vv["H"])
        front += inimg.astype(np.int32)
        idx = np.nonzero(inimg)[0]
        inside[idx] += (m[w[idx], u[idx]] > 0.5).astype(np.int32)
    return inside, front


def main():
    results_dir = Path(os.environ.get("RESULTS_DIR", ""))
    mode = os.environ.get("MODE", "body")
    pid = os.environ.get("PID", "0")
    pid2 = pid.zfill(2)
    source_dir = os.environ.get("SOURCE_DIR", "")
    masks_dir = Path(os.environ.get(
        "MASKS_DIR", f"{results_dir}/01c_sam3_person_masks"))
    align_json = Path(os.environ.get(
        "ALIGN_JSON", f"{results_dir}/05_align/head_align.json"))
    keep_ratio = float(os.environ.get("KEEP_RATIO", "0.5"))
    drop_ratio = float(os.environ.get("DROP_RATIO", "0.6"))
    margin = float(os.environ.get("HEAD_BBOX_MARGIN", "0.15"))
    drop_unseen = os.environ.get("DROP_UNSEEN", "1") == "1"
    log(f"✂️  [08e] prune MODE={mode}  keep_ratio={keep_ratio} "
        f"drop_ratio={drop_ratio}")

    views = read_cameras(source_dir)
    view_by = {vv["stem"]: vv for vv in views}
    stems = json.loads(align_json.read_text())["persons"][pid]["frames"]

    if mode in ("body", "scene"):
        from plyfile import PlyData, PlyElement
        in_ply = Path(os.environ.get("IN_PLY", ""))
        out_ply = Path(os.environ.get("OUT_PLY", ""))
        pd = PlyData.read(str(in_ply))
        v = pd["vertex"]
        names = [p.name for p in v.properties]
        xyz = np.stack([np.asarray(v[k]) for k in "xyz"], 1)
        n = len(xyz)
        log(f"  📦 {in_ply.name}: {n:,} 高斯")
        inside, front = vote([xyz] * len(stems), stems, view_by,
                             masks_dir, pid2)
        score = np.where(front > 0, inside / np.clip(front, 1, None), 1.0)
        if mode == "body":
            keep = (score >= keep_ratio) if drop_unseen \
                else (score >= keep_ratio)
            # 头盒内删除（头部归 head 分支，避免重影）
            from split_body_scene import head_box
            mesh_dir = Path(os.environ.get(
                "MESH_DIR", f"{results_dir}/06_avatar_gs"))
            lo, hi, _, _ = head_box(mesh_dir / f"avatar_mesh_p{pid}.npz",
                                    mesh_dir / f"avatar_bind_p{pid}.npz",
                                    margin)
            in_head = np.all((xyz >= lo) & (xyz <= hi), 1)
            log(f"  🗃️ 头盒内删除 {int(in_head.sum()):,}")
            keep = keep & ~in_head
        else:   # scene：反向，删幽灵点
            keep = score <= drop_ratio
        if drop_unseen:
            keep = keep & (front > 0)
        # 📏 尺度离群剪枝（同 head：巨型高斯糊暗雾）
        scale_max = float(os.environ.get("SCALE_MAX", "0"))
        if scale_max > 0:
            sc = np.exp(np.stack(
                [np.asarray(v[f"scale_{i}"]) for i in range(3)], 1)).max(1)
            big = sc > scale_max
            log(f"  📏 scale>{scale_max}: 删 {int(big.sum()):,} "
                f"({big.mean():.1%}), 最大 scale={sc.max():.4f}")
            keep = keep & ~big
        log(f"  🗳️  保留 {int(keep.sum()):,} / {n:,} "
            f"({keep.mean():.1%}); 无观测 {int((front == 0).sum()):,}")
        arr = np.empty(int(keep.sum()),
                       dtype=[(nm, v[nm].dtype) for nm in names])
        for nm in names:
            arr[nm] = np.asarray(v[nm])[keep]
        out_ply.parent.mkdir(parents=True, exist_ok=True)
        PlyData([PlyElement.describe(arr, "vertex")],
                text=False).write(str(out_ply))
        log(f"  💾 {out_ply}")

    elif mode == "head":
        # head：逐帧 FLAME 形变后的世界坐标投票（静态坐标无意义）
        from composite_check import CkptAvatar
        from train_avatar import N_SHAPE, N_EXPR
        in_ckpt = Path(os.environ.get("IN_CKPT", ""))
        out_ckpt = Path(os.environ.get("OUT_CKPT", ""))
        ck = torch.load(in_ckpt, map_location="cpu")
        dev = torch.device("cpu")
        import smplx
        flame = smplx.create(
            model_path=os.path.dirname(os.environ["FLAME_MODEL"]),
            model_type="flame", num_betas=N_SHAPE,
            num_expression_coeffs=N_EXPR,
            use_face_contour=False).to(dev)
        for p in flame.parameters():
            p.requires_grad_(False)
        head = CkptAvatar(ck, flame, dev)
        n = int(ck["opacity"].shape[0])
        log(f"  📦 {in_ckpt.name}: {n:,} 高斯 "
            f"(free={int(ck['is_free'].sum()):,})")
        xyz_per_frame = []
        for i in range(len(stems)):
            head.set_frame(i)
            xyz_per_frame.append(head.inner.get_xyz.detach().cpu().numpy())
        inside, front = vote(xyz_per_frame, stems, view_by, masks_dir, pid2)
        score = np.where(front > 0, inside / np.clip(front, 1, None), 1.0)
        keep = score >= keep_ratio
        if drop_unseen:
            keep = keep & (front > 0)
        # 📏 尺度离群剪枝（黑雾的真凶）：诊断见 2026-09-10 —— head 位置
        # 全在头盒内（位置投票 100% 保留无效），但 330 个高斯 scale>0.02、
        # max 0.21（头半径才 0.13）且 opacity~0.7 → 巨型高斯糊出大片暗雾。
        scale_max = float(os.environ.get("SCALE_MAX", "0"))
        if scale_max > 0:
            sc = torch.exp(ck["scaling"]).max(1).values.numpy()
            big = sc > scale_max
            log(f"  📏 scale>{scale_max}: 删 {int(big.sum()):,} "
                f"({big.mean():.1%}), 其中最大 scale={sc.max():.4f}")
            keep = keep & ~big
        log(f"  🗳️  保留 {int(keep.sum()):,} / {n:,} ({keep.mean():.1%}); "
            f"无观测 {int((front == 0).sum()):,}")
        # 过滤 ckpt：逐点数组按 keep，free_can 按 keep[is_free]
        keep_t = torch.from_numpy(keep)
        is_free = ck["is_free"].bool()
        out = {}
        for k, val in ck.items():
            if not torch.is_tensor(val):
                out[k] = val
                continue
            if k == "free_can":
                out[k] = val[keep_t[is_free]]
            elif val.shape and val.shape[0] == n:
                out[k] = val[keep_t]
            else:
                out[k] = val
        out_ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out, out_ckpt)
        log(f"  💾 {out_ckpt}")
    else:
        sys.exit(f"❌ 未知 MODE={mode}")


if __name__ == "__main__":
    main()
