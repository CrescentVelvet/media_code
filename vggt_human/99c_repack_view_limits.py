#!/usr/bin/env python3
"""99c_repack_view_limits.py — 收窄视角重封装：基于 99b 中间产物重打 MP4。

手机端观看视角过大时能看到重建边缘的残缺/伪影。本脚本不动 PLY 编码和视频，
只收窄 view_limits.json 的可视角范围（必要时夹紧初始相机），重跑封装两步：

    [1/2] 99b 的 bin + camera.json + 新 view_limits/init_camera → GLB
    [2/2] 99b 的视频 output.mp4 + 新 GLB → mp4_crop/<task>.mp4

前置条件（缺一报错提示先跑 99b，均由 99b 产出并默认保留在 mp4_work/）:
    mp4_work/<task>/GSCompressed_B*.bin     压缩码流
    mp4_work/<task>/output.mp4              帧视频
    三件套 json: task 目录下（UWA 型）或 mp4_work/<task>/uwa_json/（COLMAP 型）

收窄规则（对 view_limits.json 原值内缩，保持中心不动）:
    Phi  (水平)   [minPhi,   maxPhi]   → [minPhi+LEFT,   maxPhi-RIGHT]
    Theta(垂直)   [minTheta, maxTheta] → [minTheta+BOTTOM, maxTheta-TOP]
    Radius        [minRadius,maxRadius]→ [minR+MARGIN,   maxR-MARGIN]（默认 0 不动）
    初始相机若落到收窄区外 → 自动夹到边界并打印警告
    单侧收窄量 ≥ 半跨度 → 该 task 报错跳过（收没了）

用法:
    python vggt_human/99c_repack_view_limits.py                  # 默认 10° 对称收窄
    PHI_MARGIN=15 THETA_MARGIN=5 python vggt_human/99c_repack_view_limits.py
    PHI_LEFT=5 PHI_RIGHT=25 python vggt_human/99c_repack_view_limits.py  # 不对称
    DRY_RUN=1  python ...   # 只打印收窄前后范围，不执行
    FORCE=1 / ONLY=task_a,task_b   # 同 99b

Env vars:
    TOOL_DIR      UWA 工具链根目录（含 muxer.py / build/gltf_packer）
    SRC_ROOT      批次根目录（同 99b，用于定位 task 目录与三件套 json）
    RESULTS_ROOT  统一结果根（默认 ../../output/recon_human_results）
    OUT_DIR       输出目录，默认 <RESULTS_ROOT>/<批次名>/mp4_crop（与 99b 的
                  mp4/ 平级共存，不覆盖 99b 成品）
    WORK_ROOT     99b 中间产物根，默认 <RESULTS_ROOT>/<批次名>/mp4_work
    PHI_MARGIN    Phi 两侧对称收窄角度（度，默认 10；LEFT/RIGHT 未设时用它）
    THETA_MARGIN  Theta 两侧对称收窄角度（度，默认 10；TOP/BOTTOM 未设时用它）
    PHI_LEFT / PHI_RIGHT / THETA_TOP / THETA_BOTTOM
                  单侧收窄角度（度，未设则取对应 *_MARGIN）
    RADIUS_MARGIN Radius 两侧内缩（米，默认 0 不动）
    ONLY / FORCE / DRY_RUN / PYTHON_BIN / ASTC_BLOCK   含义同 99b
"""
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

# UWA 三件套文件名（与 99b 一致；camera.json 原样透传，另两件是收窄对象）
UWA_JSONS = {
    "init_camera": "init_camera.json",
    "camera": "camera.json",
    "view_params": "view_limits.json",
}
# 收窄版落盘名（放 mp4_work/<task>/crop_json/，不污染 99b 的 uwa_json/）
CROP_JSONS = {
    "init_camera": "init_camera_crop.json",
    "view_params": "view_limits_crop.json",
}


# --------------------------------------------------------------------------
# 基础工具（与 99b 相同）
# --------------------------------------------------------------------------
def _tail(text: str, n: int = 15) -> str:
    lines = [l for l in text.strip().splitlines() if l.strip()]
    return "\n".join(lines[-n:])


def run_cmd(cmd, cwd=None, env=None, log_tail: int = 15) -> int:
    """执行命令，失败时打印输出尾部（不吞异常，返回码交给调用方判断）。"""
    proc = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(cwd) if cwd else None,
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(f"  ❌ 命令失败 (rc={proc.returncode}): {' '.join(str(c) for c in cmd)}")
        if proc.stdout:
            print("  ---- stdout ----")
            print(_tail(proc.stdout, log_tail))
        if proc.stderr:
            print("  ---- stderr ----")
            print(_tail(proc.stderr, log_tail))
    return proc.returncode


# --------------------------------------------------------------------------
# 99b 中间产物探测
# --------------------------------------------------------------------------
def find_bin(work_task: Path) -> Path | None:
    """找 99b 产出的压缩码流 GSCompressed_B*.bin。"""
    bins = sorted(work_task.glob("GSCompressed_B*.bin"))
    return bins[0] if bins else None


def find_jsons(task_dir: Path, work_task: Path, ref_json_dir: Path | None):
    """定位三件套 json，返回 dict 或 None。

    优先级：REF_JSON_DIR > task 目录（UWA 型）> mp4_work/<task>/uwa_json/
    （99b COLMAP 型现场生成的位置）。
    """
    bases = [ref_json_dir, task_dir, work_task / "uwa_json"]
    for base in bases:
        if base is None:
            continue
        hit = {k: base / n for k, n in UWA_JSONS.items()}
        if all(p.is_file() for p in hit.values()):
            if ref_json_dir and base == ref_json_dir:
                print(f"  📎 用 REF_JSON_DIR 的三件套: {base}")
            return hit
    return None


def load_view_limits(jsons: dict) -> dict | None:
    vl = json.loads(jsons["view_params"].read_text())
    need = ["minPhi", "maxPhi", "minTheta", "maxTheta", "minRadius", "maxRadius"]
    miss = [k for k in need if k not in vl]
    if miss:
        print(f"  ❌ view_limits.json 缺字段: {', '.join(miss)}")
        return None
    return vl


# --------------------------------------------------------------------------
# 视角收窄数学
# 位置反解 pitch/yaw 与 get_loc_by_pitch_yaw 重建互为逆变换，
# 公式与 99b generate_uwa_jsons（抄自 pack_ply_to_mp4_v2.sh）完全一致
# --------------------------------------------------------------------------
def pos_to_pitch_yaw(loc, radius):
    """init_camera.position → (pitch, yaw)，单位度。

    99b 正变换: vy=sin(pitch)*R, vx=sin(yaw)*cos(pitch)*R, vz=cos(yaw)*cos(pitch)*R,
    且 loc=[-vx, vy, vz]（x 取负）。反解时先还原 vx=-x。
    """
    x, y, z = float(loc[0]), float(loc[1]), float(loc[2])
    nv = math.sqrt(x * x + y * y + z * z)
    if nv < 1e-10:
        return None
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, y / nv))))
    yaw = math.degrees(math.atan2(-x, z))
    return pitch, yaw


def loc_by_pitch_yaw(pitch, yaw, radius):
    """(pitch, yaw) → position，99b 的 get_loc_by_pitch_yaw 原式。"""
    vy = math.sin(math.radians(pitch)) * radius
    theta = math.cos(math.radians(pitch)) * radius
    vx = math.sin(math.radians(yaw)) * theta
    vz = math.cos(math.radians(yaw)) * theta
    return [-vx, vy, vz]


def rebuild_rotation(loc):
    """朝向原点重建 w2c 旋转（c2w），99b generate_uwa_jsons 原式，返回 3x3 嵌套 list。"""
    import numpy as np
    loc = np.asarray(loc, dtype=float)
    n = np.linalg.norm(loc)
    z_norm = -loc / n if n > 1e-10 else loc * 0.0
    x_norm = np.cross(z_norm, [0.0, 1.0, 0.0])
    nx = np.linalg.norm(x_norm)
    x_norm = x_norm / nx if nx > 1e-10 else x_norm * 0.0
    y_norm = np.cross(z_norm, x_norm)
    x_norm = np.cross(y_norm, z_norm)
    return np.stack([x_norm, y_norm, z_norm]).T.tolist()


def clamp_deg(v, lo, hi):
    return max(lo, min(hi, v))


def crop_view_limits(vl: dict, m: dict) -> dict | None:
    """对 view_limits 原值内缩（保持中心不动），返回新 dict 或 None（收没了）。"""
    phi_span = vl["maxPhi"] - vl["minPhi"]
    theta_span = vl["maxTheta"] - vl["minTheta"]
    r_span = vl["maxRadius"] - vl["minRadius"]

    # 单侧收窄 ≥ 半跨度 → 区间反转/变空，视为参数错误
    for name, cut, span in (("Phi", m["phi_left"], phi_span),
                            ("Phi", m["phi_right"], phi_span),
                            ("Theta", m["theta_bottom"], theta_span),
                            ("Theta", m["theta_top"], theta_span),
                            ("Radius", m["radius_margin"], r_span)):
        if span > 1e-9 and cut >= span / 2 or span <= 1e-9 and cut > 0:
            print(f"  ❌ {name} 单侧收窄 {cut:.2f} ≥ 半跨度 {span / 2:.2f}，区间会收没")
            return None

    new = dict(vl)
    new["minPhi"] = vl["minPhi"] + m["phi_left"]
    new["maxPhi"] = vl["maxPhi"] - m["phi_right"]
    new["minTheta"] = vl["minTheta"] + m["theta_bottom"]
    new["maxTheta"] = vl["maxTheta"] - m["theta_top"]
    new["minRadius"] = vl["minRadius"] + m["radius_margin"]
    new["maxRadius"] = vl["maxRadius"] - m["radius_margin"]
    return new


def crop_init_camera(init_json: list, vl_new: dict, radius: float) -> list | None:
    """初始相机落到收窄区外时夹到边界（就近），返回新 init_camera 列表。

    pitch/yaw 越界 → 夹紧后用 radius 重建 position/rotation（fov 等其余字段保留）。
    radius 越界 → 只改 radius 分量？不行，position 由 (pitch,yaw,r) 三个量决定，
    统一取夹紧后的 (pitch, yaw, r) 重建。
    """
    try:
        import numpy as np  # noqa: F401  # rebuild_rotation 内部用到
    except ImportError:
        print("  ❌ 夹紧初始相机需要 numpy（用 PYTHON_BIN 指定带 numpy 的解释器）")
        return None

    cam = init_json[0]
    py = pos_to_pitch_yaw(cam["position"], radius)
    if py is None:
        print("  ❌ init_camera.position 范数为 0，无法反解角度")
        return None
    pitch, yaw = py

    # pitch/yaw ↔ Theta/Phi 换算: Theta = 90 - pitch, Phi = yaw
    new_pitch = 90.0 - clamp_deg(90.0 - pitch, vl_new["minTheta"], vl_new["maxTheta"])
    new_yaw = clamp_deg(yaw, vl_new["minPhi"], vl_new["maxPhi"])
    new_r = clamp_deg(
        # radius 不在 vl 里单独存，用 position 范数；夹紧到 [minR, maxR]
        math.sqrt(sum(float(v) ** 2 for v in cam["position"])),
        vl_new["minRadius"], vl_new["maxRadius"])

    moved = (abs(new_pitch - pitch) > 1e-6 or abs(new_yaw - yaw) > 1e-6
             or abs(new_r - math.sqrt(sum(float(v) ** 2 for v in cam["position"]))) > 1e-6)
    if moved:
        print(f"  ⚠️ 初始视角越界: pitch {pitch:.2f}→{new_pitch:.2f}, "
              f"yaw {yaw:.2f}→{new_yaw:.2f}, r {math.sqrt(sum(float(v)**2 for v in cam['position'])):.3f}"
              f"→{new_r:.3f}（已自动夹到收窄区边界）")

    new_cam = dict(cam)
    new_cam["position"] = loc_by_pitch_yaw(new_pitch, new_yaw, new_r)
    new_cam["rotation"] = rebuild_rotation(new_cam["position"])
    return [new_cam]


# --------------------------------------------------------------------------
# 单 task 重封装
# --------------------------------------------------------------------------
def repack_one(task_dir: Path, work_task: Path, out_mp4: Path, cfg: dict,
               env: dict) -> str:
    """复用 99b 中间产物重打一个 task，返回 "ok" / "failed"。"""
    print(f"\n================ {task_dir.name} ================")

    # --- 1. 99b 中间产物 ---
    bin_file = find_bin(work_task)
    video_file = work_task / "output.mp4"
    if bin_file is None:
        print(f"❌ {work_task} 下没有 GSCompressed_B*.bin，先跑 99b（KEEP_WORK 默认保留）")
        return "failed"
    if not video_file.is_file():
        print(f"❌ 缺 {video_file}，先跑 99b（KEEP_WORK 默认保留）")
        return "failed"
    print(f"📦 码流: {bin_file.name}  ({bin_file.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"🎬 视频: {video_file.name}  ({video_file.stat().st_size / 1024 / 1024:.1f} MB)")

    jsons = find_jsons(task_dir, work_task, cfg["ref_json_dir"])
    if jsons is None:
        print(f"❌ 找不到三件套 json（找过 task 目录和 {work_task / 'uwa_json'}；"
              f"UWA 型在 task 下，COLMAP 型在 uwa_json/ 下，或用 REF_JSON_DIR 指定）")
        return "failed"

    vl = load_view_limits(jsons)
    if vl is None:
        return "failed"
    print(f"  📐 原: Phi[{vl['minPhi']:.2f},{vl['maxPhi']:.2f}] "
          f"Theta[{vl['minTheta']:.2f},{vl['maxTheta']:.2f}] "
          f"R[{vl['minRadius']:.3f},{vl['maxRadius']:.3f}]")

    # --- 2. 收窄 ---
    vl_new = crop_view_limits(vl, cfg["margins"])
    if vl_new is None:
        return "failed"
    print(f"  ✂️ 新: Phi[{vl_new['minPhi']:.2f},{vl_new['maxPhi']:.2f}] "
          f"Theta[{vl_new['minTheta']:.2f},{vl_new['maxTheta']:.2f}] "
          f"R[{vl_new['minRadius']:.3f},{vl_new['maxRadius']:.3f}]")

    init_json = json.loads(jsons["init_camera"].read_text())
    init_new = crop_init_camera(init_json, vl_new, vl["maxRadius"] / 1.2)
    if init_new is None:
        return "failed"

    if cfg["dry_run"]:
        print(f"💾 输出(预览): {out_mp4}")
        print("⏭️  DRY_RUN=1，跳过执行")
        return "ok"

    # --- 3. 落盘收窄版 json（放 crop_json/，不碰 99b 的 uwa_json/）---
    crop_dir = work_task / "crop_json"
    crop_dir.mkdir(parents=True, exist_ok=True)
    fp_view = crop_dir / CROP_JSONS["view_params"]
    fp_init = crop_dir / CROP_JSONS["init_camera"]
    fp_view.write_text(json.dumps(vl_new))
    fp_init.write_text(json.dumps(init_new))
    print(f"  ✅ 收窄 json: {fp_view.name} / {fp_init.name} → {crop_dir}")

    glb_file = work_task / "3DGS_crop.glb"
    t0 = time.time()

    # --- [1/2] 码流 + 相机 → GLB（gltf_packer，参数顺序同 99b）---
    print("[Step 1/2] 📦 封装为 GLB（收窄视角）...")
    rc = run_cmd([
        str(cfg["gltf_packer"]), str(bin_file), str(glb_file),
        str(fp_init), str(jsons["camera"]), str(fp_view),
    ])
    if rc != 0 or not glb_file.is_file():
        print("❌ Step 1 失败")
        return "failed"
    print(f"  ✅ GLB: {glb_file.name}  ({glb_file.stat().st_size / 1024 / 1024:.1f} MB)")

    # --- [2/2] GLB + 视频 → MP4（muxer.py，复用 99b 的视频）---
    print("[Step 2/2] 📦 封装 GLB 到 MP4...")
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    rc = run_cmd([
        cfg["python_bin"], "muxer.py",
        "--glb_path", str(glb_file),
        "--mp4_path", str(video_file),
        "--output_path", str(out_mp4),
    ], cwd=cfg["tool_dir"], env=env)
    if rc != 0 or not out_mp4.is_file():
        print("❌ Step 2 失败")
        return "failed"

    print(f"  ✅ {out_mp4.name}  ({out_mp4.stat().st_size / 1024 / 1024:.1f} MB)"
          f"  ⏱️ {time.time() - t0:.1f}s")
    return "ok"
