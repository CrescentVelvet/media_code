#!/usr/bin/env python3
"""diag_mhr_hand_topology.py — 探针：MHR 的手部到底给了什么（pose 自由度 / identity / 顶点归属）。

背景：`DESIGN_hand_pipeline.md` §6 未决项 1 —— 「MHR 手只有 5 维 identity，做不了手指级形变」
这个说法把 identity 与 pose 混在一起，且没人验证过 MHR 是否暴露手部**顶点索引区间**
（绑定高斯的前提）。本脚本把三件事分开回答：

  A. 参数布局（离线可答，std 库即可）：手部 pose 自由度、identity/scale 分几维、
     2D 观测点够不够解 SRT；
  B. 网格规模（离线可答）：顶点/面/关节数，以及手部解码器输出的是子网格还是整网格；
  C. 手部顶点集合（**需要资产**）：18439 个顶点里哪些属于手 —— 只能靠 skinning 权重
     或 torchscript buffer 反推，没有资产就答不了，本脚本会明确报「缺什么」。

模式：
  # A+B（默认，无需权重，秒级）
  python3 vggt_human/diag_mhr_hand_topology.py --sam3d-dir ~/repos/sam-3d-body

  # C-1：给 torchscript 权重就枚举 buffer（sam_3d_body 的 assets/mhr_model.pt）
  python3 vggt_human/diag_mhr_hand_topology.py --sam3d-dir ... \
      --mhr-model ~/model/sam-3d-body/sam-3d-body-dinov3/assets/mhr_model.pt

  # C-2：给 mhr 包资产目录（lod*.fbx + compact_v6_1.model）就按 skinning 权重算手部顶点
  python3 vggt_human/diag_mhr_hand_topology.py --sam3d-dir ... --mhr-assets ~/model/mhr/assets

输出：终端报告 + `--json <path>` 落盘（供 EXPERIMENTS.md 引用）。

⚠️ 结论口径：A/B 是**源码事实**，可当定论；C 在没资产时是**未验证**，不要外推。
"""
import argparse
import json
import re
import sys
from pathlib import Path

# ── MHR 文档里的常量（sam-3d-body 源码可核对，见 check_static）────────────
IDENTITY_SPLIT = {"body": 20, "head": 20, "hands": 5}   # 45 = 20+20+5
HAND_KEYPOINTS_PER_HAND = 21                            # wrist + 5 指 × 4 关节
SRT_DOF = 6                                             # 位姿对齐只解 R/t（s 全局固定）


def log(msg=""):
    print(msg, flush=True)


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def find_numbers(src: str, name: str, n=1):
    """抓 `name = <num>` 或 `name: int = <num>`（源码里的类属性写法）。"""
    pat = rf"self\.{re.escape(name)}\s*=\s*(\d+)"
    return [int(m) for m in re.findall(pat, src)][:n]


def parse_int_list(src: str, name: str):
    """抓 `name = [1, 2, 3]` 或 `name = torch.tensor([...])`。"""
    pat = rf"{re.escape(name)}\s*=\s*(?:torch\.tensor\()?\[([0-9,\s]+)\]"
    m = re.search(pat, src)
    if not m:
        return None
    return [int(x) for x in re.findall(r"\d+", m.group(1))]


def check_static(sam3d: Path) -> dict:
    """A + B：全部来自源码常量，不需要任何权重。"""
    head_p = sam3d / "sam_3d_body/models/heads/mhr_head.py"
    util_p = sam3d / "sam_3d_body/models/modules/mhr_utils.py"
    meta_p = sam3d / "sam_3d_body/metadata/mhr70.py"
    est_p = sam3d / "sam_3d_body/sam_3d_body_estimator.py"
    for p in (head_p, util_p, meta_p):
        if not p.exists():
            sys.exit(f"❌ 找不到 {p}（--sam3d-dir 指到官方仓根目录）")

    head, util, meta = read(head_p), read(util_p), read(meta_p)
    out = {}

    # ── 参数布局 ──
    out["num_shape_comps"] = find_numbers(head, "num_shape_comps")
    out["num_scale_comps"] = find_numbers(head, "num_scale_comps")
    out["num_hand_comps"] = find_numbers(head, "num_hand_comps")
    out["num_face_comps"] = find_numbers(head, "num_face_comps")
    out["body_cont_dim"] = find_numbers(head, "body_cont_dim")
    # 手部 scale 分量 = 总 scale 分量 - 身体部分（源码写法：self.num_scale_comps - 18）
    m = re.search(r"self\.num_hand_scale_comps\s*=\s*self\.num_scale_comps\s*-\s*(\d+)", head)
    if m and out["num_scale_comps"]:
        out["num_hand_scale_comps"] = out["num_scale_comps"][0] - int(m.group(1))

    hand_param_idxs = parse_int_list(util, "mhr_param_hand_idxs")
    hand_dofs = parse_int_list(util, "hand_dofs_in_order")
    out["hand_param_idxs"] = hand_param_idxs
    out["hand_dofs_in_order"] = hand_dofs
    if hand_param_idxs:
        out["hand_param_dims_total"] = len(hand_param_idxs)
        out["hand_pose_dof_per_hand"] = len(hand_param_idxs) // 2
    if hand_dofs:
        out["hand_joint_count_per_hand"] = len(hand_dofs)
        out["hand_dofs_sum"] = sum(hand_dofs)

    # ── 网格规模 ──
    m = re.search(r"torch\.zeros\((\d+),\s*3\)\.long\(\)", head)   # faces buffer
    if m:
        out["num_faces"] = int(m.group(1))
    m = re.search(r"torch\.zeros\((\d+),\s*(\d+)\s*\+\s*(\d+)\)", head)  # keypoint_mapping
    if m:
        out["num_mhr_keypoints_total"] = int(m.group(1))
        out["num_vertices"] = int(m.group(2))
        out["num_joints"] = int(m.group(3))
    m = re.search(r"torch\.zeros\((\d+),\s*3,\s*3\)", head)        # joint_rotation
    if m:
        out["num_joint_rotations"] = int(m.group(1))
    out["hand_joint_idxs_len"] = [
        len(x) for x in re.findall(r"hand_joint_idxs_(?:left|right)\s*=\s*nn\.Parameter\(\s*torch\.zeros\((\d+)\)", head)]

    # ── 2D 观测：MHR70 里有几只手指关键点 ──
    names = re.findall(r'"(left|right)-([a-z\-]+)"', meta)
    fingers = {"thumb", "index", "middle", "ring", "pinky"}
    per_hand = {s: 0 for s in ("left", "right")}
    for side, nm in names:
        if nm.split("-")[0] in fingers or nm == "wrist":
            per_hand[side] += 1
    out["hand_keypoints_per_hand"] = per_hand
    out["hand_keypoints_note"] = f"文档常量 {HAND_KEYPOINTS_PER_HAND}"

    # ── 手部解码器输出的是整网格还是子网格 ──
    vis_p = sam3d / "tools/vis_utils.py"
    if vis_p.exists():
        mm = re.search(r"\[-2\*(\d+):\]", read(vis_p))
        if mm:
            out["hand_decoder_vertices_stacked"] = int(mm.group(1))
            out["hand_decoder_output_is_full_mesh"] = (
                int(mm.group(1)) == out.get("num_vertices"))

    # ── A 的可解性：21 点 ×2 坐标 vs 6 DOF ──
    obs = per_hand.get("left", HAND_KEYPOINTS_PER_HAND) * 2
    out["obs_per_hand_per_frame"] = obs
    out["obs_per_dof_srt"] = round(obs / SRT_DOF, 2)
    return out


def inspect_torchscript(model_path: Path) -> dict:
    """C-1：枚举 torchscript 里的 buffer，找手部相关项。不猜、只报看到的。"""
    try:
        import torch
    except ImportError:
        return {"error": "torch 不可用（换装了 torch 的 env 再跑）"}
    try:
        m = torch.jit.load(str(model_path), map_location="cpu")
    except Exception as e:  # noqa: BLE001  — 打开失败原因要原样报出来
        return {"error": f"torch.jit.load 失败: {type(e).__name__}: {e}"}

    found = {}
    for name, buf in list(m.named_buffers()):
        if "hand" in name.lower():
            found[name] = list(buf.shape)
    info = {"hand_buffers": found}
    # 常见 buffer 的形状直接报出来，便于判断能不能反推顶点归属
    for key in ("faces", "keypoint_mapping", "joint_rotation", "scale_comps",
                "hand_joint_idxs_left", "hand_joint_idxs_right",
                "hand_pose_mean", "hand_pose_comps"):
        try:
            t = getattr(m, key)
            info[key] = list(t.shape)
        except Exception:  # noqa: BLE001  — 缺这个属性是正常情况
            info[key] = None
    info["verdict"] = ("手部顶点归属**未**由 buffer 直接给出；需按 skinning 权重反推"
                       if not any("skin" in k.lower() or "vert" in k.lower() for k in found)
                       else "存在 skin/vert 相关 buffer，人工确认语义")
    return info


def inspect_pymomentum(assets: Path) -> dict:
    """C-2：用 mhr 包 + pymomentum 载入 FBX，按主导关节把顶点分到左右手。"""
    try:
        import torch  # noqa: F401
        from mhr.mhr import MHR
    except ImportError as e:
        return {"error": f"缺依赖（pip install mhr pymomentum 且需 assets/）: {e}"}
    try:
        model = MHR.from_files(folder=assets, device="cpu", lod=1,
                               wants_pose_correctives=False)
    except Exception as e:  # noqa: BLE001
        return {"error": f"加载 MHR 资产失败: {type(e).__name__}: {e}"}

    char = model.character
    info = {"num_vertices": None, "num_joints": None}
    try:
        info["num_vertices"] = int(char.rest_vertex_positions.shape[0])
    except Exception:  # noqa: BLE001
        pass
    try:
        info["num_joints"] = int(char.skeleton.num_joints())
    except Exception:  # noqa: BLE001
        pass
    # 主导关节 → 顶点集合；手部关节名里含 hand / thumb / index / middle / ring / pinky
    try:
        w = char.skin_weights            # (V, J) 稀疏或稠密
        w = w.to_dense() if hasattr(w, "to_dense") else w
        dom = w.argmax(dim=1).cpu().numpy()
        jnames = [str(n) for n in char.skeleton.joint_names]
        handish = [i for i, n in enumerate(jnames)
                   if any(k in n.lower() for k in
                          ("hand", "thumb", "index", "middle", "ring", "pinky", "wrist"))]
        for side in ("left", "right"):
            js = [i for i in handish if side in jnames[i].lower()]
            idx = sorted(int(v) for v in (dom == js[0]).nonzero()) if js else []
            if js:
                mask = sum((dom == j) for j in js)
                idx = sorted(int(v) for v in mask.nonzero())
            info[f"{side}_hand_vertices"] = {
                "count": len(idx),
                "range": [idx[0], idx[-1]] if idx else None,
                "contiguous": bool(idx) and idx[-1] - idx[0] + 1 == len(idx),
                "joint_names": [jnames[j] for j in js],
            }
    except Exception as e:  # noqa: BLE001
        info["skin_error"] = f"{type(e).__name__}: {e}"
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam3d-dir", default="~/repos/sam-3d-body",
                    help="sam-3d-body 官方仓根目录")
    ap.add_argument("--mhr-model", default="", help="assets/mhr_model.pt（torchscript）")
    ap.add_argument("--mhr-assets", default="", help="mhr 包资产目录（lod*.fbx ...）")
    ap.add_argument("--json", default="", help="报告落盘路径")
    args = ap.parse_args()

    sam3d = Path(args.sam3d_dir).expanduser()
    report = {"sam3d_dir": str(sam3d)}

    log("=" * 72)
    log("🔍 [A+B] 源码常量（离线，可当定论）")
    log("=" * 72)
    A = check_static(sam3d)
    report["static"] = A
    for k in ("num_shape_comps", "num_scale_comps", "num_hand_comps", "num_face_comps",
              "body_cont_dim", "num_vertices", "num_faces", "num_joints",
              "num_joint_rotations", "num_mhr_keypoints_total"):
        if k in A:
            log(f"  {k:28s} = {A[k]}")
    if "hand_pose_dof_per_hand" in A:
        log(f"  手部 pose 自由度/只          = {A['hand_pose_dof_per_hand']}"
            f"（关节数 {A.get('hand_joint_count_per_hand')}，"
            f"doflist {A.get('hand_dofs_in_order')}）")
    if A.get("hand_param_idxs"):
        log(f"  手部参数在 mhr_param 里的下标 = {A['hand_param_idxs'][0]}..{A['hand_param_idxs'][-1]}"
            f"（共 {A['hand_param_dims_total']} 维）")
    log(f"  identity 拆分                = {IDENTITY_SPLIT}（合计 {sum(IDENTITY_SPLIT.values())}）")
    if "num_hand_scale_comps" in A:
        log(f"  手部 scale 分量              = {A['num_hand_scale_comps']}  ← 总 scale - 18（身体）")
    log(f"  MHR70 手部关键点/只          = {A.get('hand_keypoints_per_hand')}")
    log(f"  每帧每手 2D 观测             = {A.get('obs_per_hand_per_frame')}，"
        f"obs/DOF(SRT) = {A.get('obs_per_dof_srt')}")
    if "hand_decoder_output_is_full_mesh" in A:
        log(f"  手部解码器输出               = {A['hand_decoder_vertices_stacked']} 顶点 × 2 只手"
            f"（整网格={A['hand_decoder_output_is_full_mesh']}）")

    if args.mhr_model:
        log("")
        log("=" * 72)
        log("🧱 [C-1] torchscript buffer 枚举")
        log("=" * 72)
        C1 = inspect_torchscript(Path(args.mhr_model).expanduser())
        report["torchscript"] = C1
        for k, v in C1.items():
            log(f"  {k:28s} = {v}")

    if args.mhr_assets:
        log("")
        log("=" * 72)
        log("🦴 [C-2] skinning 权重 → 手部顶点集合")
        log("=" * 72)
        C2 = inspect_pymomentum(Path(args.mhr_assets).expanduser())
        report["pymomentum"] = C2
        for k, v in C2.items():
            log(f"  {k:28s} = {v}")

    if not args.mhr_model and not args.mhr_assets:
        log("")
        log("⚠️ 未给 --mhr-model / --mhr-assets：手部**顶点索引区间**仍未知（这是 C 项）。")
        log("   需要的资产：sam-3d-body gated ckpt 里的 assets/mhr_model.pt，")
        log("   或 mhr 包资产目录（lod1.fbx + compact_v6_1.model，来自同一个 gated 仓）。")

    if args.json:
        p = Path(args.json).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        # 统一 LF，避免 Windows 侧 CRLF
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        log(f"\n💾 报告: {p}")


if __name__ == "__main__":
    main()
