#!/usr/bin/env python3
"""99b_pack_ply_to_mp4.py — 批量把 PLY（3DGS 重建结果）封装成可播放的 MP4。

环境准备
    conda activate xcodec
    conda install -y -c conda-forge ffmpeg

输入是「批次目录」，其下每个子目录是一个 task。脚本**自动判断 task 属于哪种模式**并打印：

    模式 UWA     init_camera.json + camera.json + view_limits.json + recon_result.ply
                → json 现成，直接跑四步链
    模式 COLMAP  cameras.txt + images.txt + points3D.txt + point_cloud_final.ply
                → 现场生成三件套 json（逻辑见 generate_uwa_jsons，抄自
                  pack_ply_to_mp4_v2.sh，已用旧目录数据数值验证过），再跑四步链

四步链:
    [1/4] 图片序列        → H.264 视频（ffmpeg）
    [2/4] PLY             → 压缩码流 GSCompressed_B<n>.bin（工具链 encode.py）
    [3/4] 压缩码流 + 相机 → GLB（工具链 build/gltf_packer）
    [4/4] GLB + 视频      → 最终 MP4（工具链 muxer.py）

用法:
    python vggt_human/99b_pack_ply_to_mp4.py
    DRY_RUN=1  python vggt_human/99b_pack_ply_to_mp4.py   # 只探测不执行
    FORCE=1    python vggt_human/99b_pack_ply_to_mp4.py   # 已存在 mp4 也重跑
    ONLY=task_id_a,task_id_b python vggt_human/99b_pack_ply_to_mp4.py

Env vars（不设则用下方 main() 里的默认值）:
    TOOL_DIR    UWA 工具链根目录（含 encode.py / muxer.py / build/gltf_packer）
    SRC_ROOT    批次根目录（其下每个子目录是一个 task）
    RESULTS_ROOT    统一结果根（默认 ../../output/recon_human_results）
    OUT_DIR     输出目录，默认 <RESULTS_ROOT>/<批次名>/mp4（批次名与源目录同名，
                99a 的 ply 落在同批次 ply/ 子目录；显式给 OUT_DIR 则完全覆盖）
    ONLY        逗号分隔的 task 白名单
    MODE        auto（默认）/ uwa / colmap，强制指定模式
    INSIDEOUT=1 室内朝外视角（人像默认 0），影响 view_limits / init_camera 的角度映射
    GS_FALLBACK=1  缺 gs_camera_params_final.json 时，用 COLMAP 相机位姿推断
                   radius / pitch / yaw 范围（默认关闭：缺该文件即失败）
    REF_JSON_DIR   外部提供三件套 json 的目录，优先级高于 COLMAP 现场生成
    PYTHON_BIN  跑 encode.py / muxer.py 的解释器（默认 python，需带工具链依赖）
    FPS / CRF / PRESET   视频编码参数（默认 30 / 28 / fast）
    ASTC_BLOCK  码流文件名的 ASTC 块大小（默认 4，仅用于拼文件名）
    PLY_NAME / IMAGE_DIR 强制指定 ply 文件名 / 帧序列目录。
                IMAGE_DIR 相对路径的基点按 task 目录 → 批次根 → 结果根 → cwd
                顺序试（第一个存在的胜出），解析后的绝对路径会打印出来。
                默认 ../../code/Reconstruction/dataset/B003_Human_Data_w_pose
    FORCE=1     覆盖已存在的 mp4（默认跳过）
    KEEP_WORK=0 成功后删除中间产物目录（默认保留，方便排查）
    DRY_RUN=1   只打印探测结果，不执行
"""
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

# ply 探测顺序按模式分开：UWA 目录产物叫 recon_result.ply，COLMAP 目录叫 point_cloud_final.ply
PLY_CANDIDATES = {
    "UWA": ["recon_result.ply", "point_cloud_final.ply", "point_cloud.ply", "scene.ply"],
    "COLMAP": ["point_cloud_final.ply", "recon_result.ply", "point_cloud.ply", "scene.ply"],
}
IMAGE_DIR_CANDIDATES = ["image", "images", "input", "frames"]
IMAGE_EXTS = ["jpg", "jpeg", "png"]

# UWA 模式三件套（gltf_packer 的入参）
UWA_JSONS = {
    "init_camera": "init_camera.json",
    "camera": "camera.json",
    "view_params": "view_limits.json",
}
# COLMAP 模式的标志文件
COLMAP_FILES = ["cameras.txt", "images.txt", "points3D.txt"]
# 现场生成 view_limits 需要的参数文件（COLMAP 模式）
GS_PARAMS_NAME = "gs_camera_params_final.json"

MODE_UWA = "UWA"
MODE_COLMAP = "COLMAP"


# --------------------------------------------------------------------------
# 基础工具
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
# 输入探测
# --------------------------------------------------------------------------
def detect_ply(task_dir: Path, mode: str, ply_name: str = "") -> Path | None:
    if ply_name:
        p = task_dir / ply_name
        return p if p.is_file() else None
    for name in PLY_CANDIDATES.get(mode, PLY_CANDIDATES["UWA"]):
        p = task_dir / name
        if p.is_file():
            return p
    # 兜底：目录里任意一个 .ply（多个时取名字排序第一个，保证可复现）
    plies = sorted(task_dir.glob("*.ply"))
    return plies[0] if plies else None


def has_images(d: Path) -> bool:
    """目录（仅一层）里是否有 jpg/png 帧。"""
    try:
        return any(f.is_file() and f.suffix.lower().lstrip(".") in IMAGE_EXTS
                   for f in d.iterdir())
    except OSError:
        return False


def detect_image_dir(task_dir: Path, image_dir: str = "") -> Path | None:
    if image_dir:
        d = task_dir / image_dir
        return d if d.is_dir() else None
    for name in IMAGE_DIR_CANDIDATES:
        d = task_dir / name
        if d.is_dir():
            return d
    return None


def pick_image_dir(cand: Path, task_name: str) -> Path | None:
    """在一个候选基点下挑出真正放帧的目录。

    优先级：① <基点>/<task 名>（数据集按 task 分目录时）② 基点本身 ③ 它们的
    image/ / images/ 子目录。都命中不了但目录存在时仍返回它，让上层按「缺帧」处理
    （COLMAP 的渲染帧可能还没生成，属待补而非失败）。
    """
    sub = cand / task_name
    if sub.is_dir() and has_images(sub):
        return sub
    if cand.is_dir() and has_images(cand):
        return cand
    for d in (sub, cand):
        if not d.is_dir():
            continue
        for name in IMAGE_DIR_CANDIDATES:
            s = d / name
            if s.is_dir() and has_images(s):
                return s
        return d
    return None


def resolve_image_dir(task_dir: Path, image_dir: str, bases: list) -> Path | None:
    """解析 IMAGE_DIR。

    绝对路径直接用。相对路径的基点有歧义——可能是相对 task 目录、批次根、结果根
    或当前工作目录（v2 脚本里是相对 INPUT_DIR，但命令行下也常相对 cwd 写），
    单一语义猜错代价大，所以按 bases 顺序逐个试，第一个真实存在的胜出，
    解析结果由调用方打印供核对。
    """
    p = Path(image_dir)
    if p.is_absolute():
        return pick_image_dir(p, task_dir.name)

    tried, cands = [], []
    for b in bases:
        c = (b / p).resolve()
        if c not in cands:
            cands.append(c)
    for c in cands:
        got = pick_image_dir(c, task_dir.name)
        tried.append(c)
        if got is not None:
            return got
    print("  🔎 IMAGE_DIR 未命中，试过的路径:")
    for c in tried:
        print(f"      - {c}")
    return None


def detect_image_sequence(img_dir: Path):
    """识别图片序列的 扩展名 / 数字位宽 / 起始编号。

    返回 dict(ext, pad, start, count)；文件名不是纯数字时返回 None，改用 glob。
    pad 取众数：个别文件位数不一致时按主流命名走。
    start 必须探测：ffmpeg 的 %0Nd 模式默认从 0 开始找，序列从 1 编号会直接报
    "Could find no file with path ..."，所以必须显式传 -start_number。
    """
    pat = re.compile(r"^(\d+)\.([A-Za-z]+)$")
    hits = []
    for f in img_dir.iterdir():
        if not f.is_file():
            continue
        m = pat.match(f.name)
        if not m:
            continue
        ext = m.group(2).lower()
        if ext not in IMAGE_EXTS:
            continue
        hits.append((int(m.group(1)), ext, len(m.group(1))))

    if not hits:
        return None

    ext = Counter(e for _, e, _ in hits).most_common(1)[0][0]
    same_ext = [h for h in hits if h[1] == ext]
    pad = Counter(p for _, _, p in same_ext).most_common(1)[0][0]
    idxs = sorted(i for i, _, _ in same_ext)
    return {"ext": ext, "pad": pad, "start": idxs[0], "count": len(idxs)}


def detect_glob_ext(img_dir: Path) -> str | None:
    """文件名不是纯数字时，取目录里最多的图片扩展名，供 ffmpeg glob 用。"""
    cnt = Counter(f.suffix.lower().lstrip(".") for f in img_dir.iterdir() if f.is_file())
    for ext, _ in cnt.most_common():
        if ext in IMAGE_EXTS:
            return ext
    return None


def find_json(task_dir: Path, name: str, override: str = "") -> Path | None:
    """找相机参数 json；override 可以是绝对路径或相对 task_dir 的文件名。"""
    if override:
        p = Path(override)
        if not p.is_absolute():
            p = task_dir / override
        return p if p.is_file() else None

    p = task_dir / name
    if p.is_file():
        return p
    # 兜底：同名 json 可能在 task 下的子目录里（一层）
    found = sorted(task_dir.glob(f"*/{name}"))
    return found[0] if found else None


def detect_mode(task_dir: Path, cfg: dict):
    """判断 task 属于 UWA 还是 COLMAP 模式。

    返回 (mode, info)；mode 为 None 表示两种都不完整。
    同时存在时以 UWA 优先（json 齐全，省一步生成）。
    """
    uwa_hit = {k: find_json(task_dir, n, cfg.get(f"{k}_name", ""))
               for k, n in UWA_JSONS.items()}
    uwa_all = all(uwa_hit.values())
    uwa_any = any(uwa_hit.values())

    colmap_hit = {f: (task_dir / f).is_file() for f in COLMAP_FILES}
    # images.txt + cameras.txt 是必需，points3D.txt 有时为空/缺失也认
    colmap_ok = colmap_hit["cameras.txt"] and colmap_hit["images.txt"]

    info = {
        "uwa": uwa_hit,
        "uwa_all": uwa_all,
        "uwa_any": uwa_any,
        "colmap": colmap_hit,
        "colmap_ok": colmap_ok,
    }

    forced = cfg.get("mode", "auto")
    if forced == "uwa":
        return MODE_UWA, info
    if forced == "colmap":
        return MODE_COLMAP, info

    if uwa_all:
        return MODE_UWA, info
    if colmap_ok:
        return MODE_COLMAP, info
    return None, info


# --------------------------------------------------------------------------
# Step 0: COLMAP txt → UWA 三件套 json
# 逻辑抄自 pack_ply_to_mp4_v2.sh（其源头是 GaussianPhoto3D
# gaussian3d/reconstruction/src/module/warping.py），已用旧目录数据数值验证。
# --------------------------------------------------------------------------
def generate_uwa_jsons(task_dir: Path, out_dir: Path | None, insideout: bool,
                       gs_fallback: bool = False):
    """从 cameras.txt + images.txt + gs_camera_params_final.json 生成三件套。

    Args:
        out_dir: 落盘目录；传 None 表示只解析打印、不写文件（DRY_RUN）。
    Returns:
        {"init_camera": Path, "camera": Path, "view_params": Path,
         "image_names": [...]} 或 None（失败）。
    """
    try:
        import numpy as np
    except ImportError:
        print("  ❌ 生成相机 json 需要 numpy，当前解释器里没有（用 PYTHON_BIN 指定）")
        return None

    def qvec2rotmat(q):
        w, x, y, z = q
        return np.array([
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ])

    def safe_normalize(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-10 else v * 0.0

    # ---------- cameras.txt ----------
    cams = {}
    cam_path = task_dir / "cameras.txt"
    try:
        for line in cam_path.read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            if len(p) < 8:
                continue
            if p[1] != "PINHOLE":
                print(f"  ❌ 暂只支持 PINHOLE 相机模型，实际: {p[1]}")
                return None
            cams[int(p[0])] = (int(p[2]), int(p[3]), [float(v) for v in p[4:8]])
    except FileNotFoundError:
        print(f"  ❌ 缺少 {cam_path}")
        return None
    if not cams:
        print("  ❌ cameras.txt 未解析到相机")
        return None

    # ---------- images.txt ----------
    # 位姿行: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME [+3k 个 2D 点] → 字段数 %3==1
    # 观测行(可能缺省): 3m 个字段 → 字段数 %3==0
    positions, img_names, cam_ids, quats = [], [], [], []
    try:
        for line in (task_dir / "images.txt").read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            if len(p) < 10 or len(p) % 3 != 1:
                continue
            q = [float(v) for v in p[1:5]]
            R = qvec2rotmat(q)                                  # w2c 旋转
            t = np.array([float(v) for v in p[5:8]])            # w2c 平移
            positions.append(-R.T @ t)                          # c2w: 相机中心(世界系)
            quats.append(q)
            img_names.append(p[9].rsplit(".", 1)[0])            # 去扩展名
            cam_ids.append(int(p[8]))
    except FileNotFoundError:
        print(f"  ❌ 缺少 {task_dir / 'images.txt'}")
        return None
    if not cam_ids:
        print("  ❌ images.txt 中未解析到图像")
        return None

    w0, h0, (fx0, fy0, cx0, cy0) = cams[cam_ids[0]]
    print(f"  📷 图像 {len(cam_ids)} 张, 相机 {len(cams)} 台, "
          f"主相机 {w0}x{h0} fx={fx0:.1f} fy={fy0:.1f}")

    # ---------- camera.json ----------
    camera_json = []
    for idx, cid in enumerate(cam_ids):
        w, h, (fx, fy, cx, cy) = cams[cid]
        camera_json.append({
            "id": idx, "img_name": img_names[idx], "width": w, "height": h,
            "position": positions[idx].tolist(),                # 相机中心(世界系)
            "rotation": qvec2rotmat(quats[idx]).T.tolist(),     # c2w 旋转
            "fy": fy, "fx": fx,
        })

    # ---------- gs_camera_params_final.json ----------
    gs_path = find_json(task_dir, GS_PARAMS_NAME)
    radius = pitch_min = pitch_max = yaw_min = yaw_max = None
    if gs_path is None:
        if not gs_fallback:
            print(f"  ❌ 缺少 {GS_PARAMS_NAME}（COLMAP 模式必需；"
                  f"或设 GS_FALLBACK=1 用相机位姿推断）")
            return None
        print(f"  ⚠️ 无 {GS_PARAMS_NAME}，GS_FALLBACK=1：用相机位姿推断（未经数值验证）")
        # 推断：radius=相机中心到质心的最大距离；pitch/yaw 取各相机实际角度的范围
        pts = np.stack(positions)
        centroid = pts.mean(axis=0)
        radius = float(np.linalg.norm(pts - centroid, axis=1).max())
        pitches, yaws = [], []
        for pv in positions:
            nv = np.linalg.norm(pv)
            pitches.append(math.degrees(math.asin(float(pv[1] / nv))))
            yaws.append(math.degrees(math.atan2(float(pv[0]), float(pv[2]))))
        pitch_min, pitch_max = min(pitches), max(pitches)
        yaw_min, yaw_max = min(yaws), max(yaws)
        print(f"     radius={radius:.4f} pitch[{pitch_min:.2f},{pitch_max:.2f}] "
              f"yaw[{yaw_min:.2f},{yaw_max:.2f}]")
    else:
        gs = json.loads(gs_path.read_text())
        s = gs["setting"]
        radius = gs["scene"]["radius"]
        pitch_min, pitch_max = s["pitch_min"], s["pitch_max"]
        yaw_min, yaw_max = s["yaw_min"], s["yaw_max"]

    # ---------- view_limits.json ----------
    # 映射(已验证): Phi=yaw, Theta=90-pitch, minR=0.9r, maxR=1.2r
    if insideout:
        pitch_min, pitch_max = -pitch_max, -pitch_min
        yaw_min, yaw_max = yaw_min - 180, yaw_max - 180
        if yaw_min < -180:
            yaw_min += 360
        if yaw_max > 180:
            yaw_max -= 360
        yaw_min, yaw_max = min(yaw_min, yaw_max), max(yaw_min, yaw_max)

    view_limits = {
        "gravityCoordinate": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "maxPhi": yaw_max, "minPhi": yaw_min,
        "maxTheta": 90.0 - pitch_min, "minTheta": 90.0 - pitch_max,
        "maxRadius": 1.2 * radius, "minRadius": 0.9 * radius,
        "maxX": 0.0, "maxY": 0.0, "maxZ": 0.0,
        "minX": 0.0, "minY": 0.0, "minZ": 0.0,
        "target": [0.0, 0.0, 0.0],
    }
    print(f"  📐 view_limits: Phi[{yaw_min:.2f},{yaw_max:.2f}] "
          f"Theta[{view_limits['minTheta']:.2f},{view_limits['maxTheta']:.2f}] "
          f"R[{view_limits['minRadius']:.3f},{view_limits['maxRadius']:.3f}]")

    # ---------- init_camera.json ----------
    # pitch_init/yaw_init 取第一台相机位置（gs json 里的 yaw_init 硬编码 0，不可用）
    p0 = positions[0]
    pitch_init = math.degrees(math.asin(float(p0[1] / np.linalg.norm(p0))))
    yaw_init = math.degrees(math.atan2(float(p0[0]), float(p0[2])))
    if insideout:
        yaw_init = yaw_init - 180 if yaw_init > 0 else yaw_init + 180
        pitch_init = -pitch_init

    vy = math.sin(math.radians(pitch_init)) * radius
    theta = math.cos(math.radians(pitch_init)) * radius
    vx = math.sin(math.radians(yaw_init)) * theta
    vz = math.cos(math.radians(yaw_init)) * theta
    loc = np.array([-vx, vy, vz])

    # 朝向原点构建 w2c 旋转（相机 Z 轴指向场景）
    z_norm = safe_normalize(-loc)
    x_norm = safe_normalize(np.cross(z_norm, [0.0, 1.0, 0.0]))
    y_norm = np.cross(z_norm, x_norm)
    x_norm = np.cross(y_norm, z_norm)
    c2w_r = np.stack([x_norm, y_norm, z_norm]).T

    # v2 原式用 h0 配 fx0（方形像素下等价），保持原样
    init_fov = math.degrees(2 * math.atan(h0 / (2 * fx0))) * 0.8
    name_pos = "".join(f"_{v}" for v in loc)
    init_camera = [{
        "id": 0,
        "img_name": f"novel_{img_names[0]}_pos{name_pos}",
        "width": w0, "height": h0,
        "position": loc.tolist(), "rotation": c2w_r.tolist(),
        "fy": fx0, "fx": fx0, "init_fov": init_fov,
    }]
    print(f"  📍 init_camera: pitch_init={pitch_init:.4f} yaw_init={yaw_init:.4f} "
          f"init_fov={init_fov:.4f}")

    if gs_path is not None and not insideout:
        if abs(pitch_init - json.loads(gs_path.read_text())["setting"]["pitch_init"]) > 1.0:
            print("  ⚠️ pitch_init 与 gs json 偏差 > 1°，请检查")

    if out_dir is None:
        print("  ⏭️  DRY_RUN=1，不落盘")
        return {"init_camera": None, "camera": None, "view_params": None,
                "image_names": img_names}

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, obj, fname in (("camera", camera_json, UWA_JSONS["camera"]),
                            ("view_params", view_limits, UWA_JSONS["view_params"]),
                            ("init_camera", init_camera, UWA_JSONS["init_camera"])):
        fp = out_dir / fname
        fp.write_text(json.dumps(obj))
        paths[key] = fp
    print(f"  ✅ 三件套 json 已生成: {out_dir}")
    paths["image_names"] = img_names
    return paths


def resolve_colmap_image_dir(task_dir: Path, names: list) -> Path | None:
    """COLMAP 的 images.txt 只记相对路径，回推图片实际所在目录。"""
    if not names:
        return None
    base = os.path.basename(names[0])
    for cand in IMAGE_DIR_CANDIDATES:
        d = task_dir / cand
        if d.is_dir() and ((d / names[0]).is_file() or (d / base).is_file()):
            return d
    if (task_dir / names[0]).is_file() or (task_dir / base).is_file():
        return task_dir
    return None


def build_ffmpeg_inputs(img_dir: Path, seq, fps: str) -> list:
    """拼 ffmpeg 输入参数。

    数字命名 → %0Nd 模式（-start_number 必传，否则从 1 编号的序列会找不到文件）；
    其它命名 → glob 按字典序（渲染帧常用）。
    """
    args = ["-framerate", str(fps)]
    if seq is not None:
        args += ["-start_number", str(seq["start"]),
                 "-i", f"{img_dir}/%0{seq['pad']}d.{seq['ext']}"]
        return args
    ext = detect_glob_ext(img_dir)
    args += ["-pattern_type", "glob", "-i", f"{img_dir}/*.{ext or 'jpg'}"]
    return args


# --------------------------------------------------------------------------
# 单个 task 的处理
# --------------------------------------------------------------------------
def pack_one(task_dir: Path, out_mp4: Path, cfg: dict, env: dict) -> str:
    """处理一个 task，返回 "ok" / "pending"（缺输入待补） / "failed"。"""
    print(f"\n================ {task_dir.name} ================")

    # --- 1. 模式判断 ---
    mode, info = detect_mode(task_dir, cfg)
    if mode is None:
        print("❌ 模式未知：既没有完整的 UWA 三件套 json，也没有 cameras.txt+images.txt")
        print(f"   UWA 命中: {[n for k, n in UWA_JSONS.items() if info['uwa'][k]] or '无'}")
        print(f"   COLMAP 命中: {[f for f, ok in info['colmap'].items() if ok] or '无'}")
        return "failed"

    if mode == MODE_UWA:
        print(f"📦 模式: {MODE_UWA}  ({' + '.join(UWA_JSONS.values())} + <ply>)")
    else:
        have = [f for f, ok in info["colmap"].items() if ok]
        print(f"📦 模式: {MODE_COLMAP}  ({' + '.join(have)} + <ply>)")
        if info["uwa_any"]:
            print("  ℹ️ 该目录也有部分 UWA json，但不齐全，按 COLMAP 处理")
        if cfg["insideout"]:
            print("  🔄 INSIDEOUT=1（室内朝外视角）")

    # --- 2. ply ---
    ply = detect_ply(task_dir, mode, cfg["ply_name"])
    if ply is None:
        print(f"❌ 找不到 ply（候选: {PLY_CANDIDATES[mode]} 或任一 *.ply）")
        return "failed"
    print(f"🖼️ ply:    {ply.name}")

    # --- 3. 三件套 json ---
    work_dir = out_mp4.parent / "mp4_work" / task_dir.name
    if mode == MODE_UWA:
        jsons = {k: v for k, v in info["uwa"].items()}
        image_names = []
    else:
        ref = cfg.get("ref_json_dir")
        if ref:
            jsons = {k: (ref / n) for k, n in UWA_JSONS.items()}
            missing = [n for k, n in UWA_JSONS.items() if not jsons[k].is_file()]
            if missing:
                print(f"  ❌ REF_JSON_DIR 缺少: {', '.join(missing)}")
                return "failed"
            print(f"  📎 借用 REF_JSON_DIR 的三件套 json: {ref}")
            image_names = []
        else:
            print("[Step 0/4] 🧮 从 COLMAP txt 生成相机 json...")
            jsons = generate_uwa_jsons(
                task_dir,
                None if cfg["dry_run"] else work_dir / "uwa_json",
                cfg["insideout"], cfg["gs_fallback"])
            if jsons is None:
                return "failed"
            image_names = jsons.get("image_names", [])

    if cfg["dry_run"]:
        print(f"💾 输出(预览): {out_mp4}")
        print("⏭️  DRY_RUN=1，跳过执行")
        return "ok"

    work_dir.mkdir(parents=True, exist_ok=True)
    if mode == MODE_UWA:
        print(f"📐 相机:   {jsons['init_camera'].name} / {jsons['camera'].name} "
              f"/ {jsons['view_params'].name}")

    # --- 4. 图片序列 ---
    out_root = out_mp4.parent
    if cfg["image_dir"]:
        # 相对 IMAGE_DIR 的基点按「task → 批次根 → 结果根 → cwd」依次试，命中即停
        bases = [task_dir, cfg.get("src_root", task_dir), out_root,
                 out_root.parent, out_root.parent.parent, Path.cwd()]
        img_dir = resolve_image_dir(task_dir, cfg["image_dir"], bases)
    elif mode == MODE_COLMAP:
        img_dir = (resolve_colmap_image_dir(task_dir, image_names)
                   or detect_image_dir(task_dir))
    else:
        img_dir = detect_image_dir(task_dir)

    no_frames = False
    if img_dir is None:
        # COLMAP 模式的帧常常是 novel 视角渲染帧，缺帧是「待补」而非出错：
        # 三件套 json 已经生成好了，补完帧直接重跑即可
        if cfg["image_dir"]:
            print(f"⏸️ IMAGE_DIR={cfg['image_dir']} 在所有基点下都未命中"
                  f"（三件套 json 已生成，可直接复用）")
        elif mode == MODE_COLMAP:
            print(f"⏸️ 没有帧目录（候选: {IMAGE_DIR_CANDIDATES}）。novel 视角渲染帧需先渲染，"
                  f"或用 IMAGE_DIR=/path/to/frames 指定（三件套 json 已生成，可直接复用）")
        else:
            print(f"❌ 找不到图片目录（候选: {IMAGE_DIR_CANDIDATES}）")
        if mode == MODE_COLMAP:
            return "pending"
        return "failed"

    seq = detect_image_sequence(img_dir)
    # IMAGE_DIR 的基点是试出来的，必须打全路径供核对；自动探测的就在 task 下，打目录名即可
    shown = str(img_dir) if cfg["image_dir"] else img_dir.name
    if seq is None:
        print(f"🖼️ 图片:   {shown}（非数字命名，glob 按字典序）")
    else:
        print(f"🖼️ 图片:   {shown}/%0{seq['pad']}d.{seq['ext']}"
              f"  (起始 {seq['start']}, 共 {seq['count']} 张)")
    if not any(f.is_file() and f.suffix.lower().lstrip(".") in IMAGE_EXTS
               for f in img_dir.iterdir()):
        print(f"⏸️ {img_dir} 下没有 jpg/png 帧。novel 视角渲染帧需先渲染，"
              f"或用 IMAGE_DIR=/path/to/frames 指定（三件套 json 已生成，可直接复用）")
        return "pending"
    print(f"💾 输出:   {out_mp4}")

    video_file = work_dir / "output.mp4"
    glb_file = work_dir / "3DGS.glb"
    t0 = time.time()

    # --- Step 1/4: 图片序列 → 视频 ---
    print("[Step 1/4] 🎬 生成 H.264 视频...")
    # pad=ceil(iw/2)*2: yuv420p 要求宽高为偶数，奇数分辨率会直接编码失败
    rc = run_cmd([
        "ffmpeg", "-y",
        *build_ffmpeg_inputs(img_dir, seq, cfg["fps"]),
        "-vcodec", "libx264",
        "-crf", str(cfg["crf"]),
        "-preset", cfg["preset"],
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt", "yuv420p",
        str(video_file),
    ])
    if rc != 0 or not video_file.is_file():
        print("❌ Step 1 失败")
        return "failed"
    print(f"  ✅ 视频: {video_file}  ({video_file.stat().st_size / 1024 / 1024:.1f} MB)")

    # --- Step 2/4: PLY → 压缩码流 ---
    print("[Step 2/4] 📦 编码 PLY 为压缩码流...")
    rc = run_cmd(
        [cfg["python_bin"], "encode.py", "--ply-path", str(ply), "--save-dir", str(work_dir)],
        cwd=cfg["tool_dir"], env=env,
    )
    bin_file = work_dir / f"GSCompressed_B{cfg['astc_block']}.bin"
    if not bin_file.is_file():
        # encode.py 的块大小由它内部决定，环境变量只是我们的预期值；
        # 对不上时按实际产物 glob，避免整批误判失败
        alts = sorted(work_dir.glob("GSCompressed_B*.bin"))
        if alts:
            print(f"⚠️ 期望 {bin_file.name} 不存在，改用实际产物 {alts[0].name}")
            bin_file = alts[0]
        else:
            print(f"❌ Step 2 失败：{work_dir} 下没有 GSCompressed_B*.bin")
            return "failed"
    print(f"  ✅ 码流: {bin_file}  ({bin_file.stat().st_size / 1024 / 1024:.1f} MB)")

    # --- Step 3/4: 码流 + 相机 → GLB ---
    print("[Step 3/4] 📦 封装为 GLB...")
    rc = run_cmd([
        str(cfg["gltf_packer"]), str(bin_file), str(glb_file),
        str(jsons["init_camera"]), str(jsons["camera"]), str(jsons["view_params"]),
    ])
    if rc != 0 or not glb_file.is_file():
        print("❌ Step 3 失败")
        return "failed"
    print(f"  ✅ GLB: {glb_file}  ({glb_file.stat().st_size / 1024 / 1024:.1f} MB)")

    # --- Step 4/4: GLB + 视频 → MP4 ---
    print("[Step 4/4] 📦 封装 GLB 到 MP4...")
    rc = run_cmd([
        cfg["python_bin"], "muxer.py",
        "--glb_path", str(glb_file),
        "--mp4_path", str(video_file),
        "--output_path", str(out_mp4),
    ], cwd=cfg["tool_dir"], env=env)
    if rc != 0 or not out_mp4.is_file():
        print("❌ Step 4 失败")
        return "failed"

    print(f"  ✅ {out_mp4.name}  ({out_mp4.stat().st_size / 1024 / 1024:.1f} MB)"
          f"  ⏱️ {time.time() - t0:.1f}s")

    if not cfg["keep_work"]:
        shutil.rmtree(work_dir, ignore_errors=True)
        print(f"  🧹 已清理中间产物: {work_dir}")
    return "ok"


# --------------------------------------------------------------------------
# 批量入口
# --------------------------------------------------------------------------
def pack_batch(src_root: Path, out_root: Path, cfg: dict, only: list[str]) -> None:
    if not src_root.is_dir():
        sys.exit(f"❌ 源目录不存在: {src_root}")
    if not cfg["dry_run"]:
        out_root.mkdir(parents=True, exist_ok=True)
    # 供 IMAGE_DIR 相对路径解析用（相对批次根 / 结果根都支持）
    cfg["src_root"] = src_root

    # 工具链 PYTHONPATH：muxer.py 依赖 pymp4，必须把 thirdparty 和工具链根都加进去
    env = os.environ.copy()
    pymp4 = cfg["tool_dir"] / "thirdparty/pymp4-1.4.0/src"
    parts = [str(cfg["tool_dir"])]
    if pymp4.is_dir():
        parts.append(str(pymp4))
    old = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(parts + ([old] if old else []))

    print(f"🔍 源目录:   {src_root}")
    print(f"📦 批次名:   {src_root.resolve().name}")
    print(f"📁 输出目录: {out_root}")
    print(f"🛠️ 工具链:   {cfg['tool_dir']}")
    if cfg.get("ref_json_dir"):
        print(f"📎 外部 json: {cfg['ref_json_dir']}")
    if cfg["insideout"]:
        print("🔄 INSIDEOUT=1")
    if cfg["dry_run"]:
        print("⏭️  DRY_RUN=1（只探测，不执行）")
    print()

    subdirs = sorted([d for d in src_root.iterdir() if d.is_dir()], key=lambda p: p.name)
    # 中间产物目录 mp4_work 在输出目录里，不是 task，排除掉
    subdirs = [d for d in subdirs if d.name != "mp4_work"]
    if only:
        subdirs = [d for d in subdirs if d.name in only]
    if not subdirs:
        print("⚠️ 没有待处理的 task 目录")
        return

    ok = skip = fail = pend = 0
    failed, pending = [], []
    mode_count = Counter()
    for sub in subdirs:
        # 先判模式，用于统计（即便后面失败/跳过也算进去）
        mode, _ = detect_mode(sub, cfg)
        mode_count[mode or "未知"] += 1

        out_mp4 = out_root / f"{sub.name}.mp4"
        if out_mp4.is_file() and not cfg["force"] and not cfg["dry_run"]:
            print(f"\n================ {sub.name} ================")
            print(f"📦 模式: {mode or '未知'}")
            print(f"⏭️  skip  {sub.name}  (已存在 {out_mp4.name}，FORCE=1 可覆盖)")
            skip += 1
            continue

        status = pack_one(sub, out_mp4, cfg, env)
        if status == "ok":
            ok += 1
        elif status == "pending":
            pend += 1
            pending.append(sub.name)
        else:
            fail += 1
            failed.append(sub.name)

    print()
    print(f"🎉 Done.  ✅ {ok}  ⏸️ {pend}  ⏭️ {skip}  ❌ {fail}")
    print("📊 模式统计: " + "  ".join(f"{k} x{v}" for k, v in mode_count.items()))
    if pending:
        print(f"⏸️ 待补输入: {', '.join(pending)}")
    if failed:
        print(f"❌ 失败列表: {', '.join(failed)}")
    print(f"📁 结果: {out_root}")


def main():
    # ===== 在这里直接改路径（或用环境变量覆盖）=====
    TOOL_DIR = Path(os.environ.get(
        "TOOL_DIR", "../../model/UWA_Sample_Tool_v3"))
    # 批次根目录：其下每个子目录是一个 task（UWA 型或 COLMAP 型混着也行）
    SRC_ROOT = Path(os.environ.get(
        "SRC_ROOT",
        "../../code/Reconstruction/output/"
        "B003_Human_Data_w_pose-脸红优化+外插视角增强"))
    # 统一结果根：与 99a_collect_ply.py 同源，便于 ply 和 mp4 一起找
    RESULTS_ROOT = Path(os.environ.get(
        "RESULTS_ROOT", "../../output/recon_human_results"))
    # 输出目录：默认 <RESULTS_ROOT>/<批次名>/mp4（与源目录同名）。
    # ply 收集在同级 ply/ 子目录（99a 产出），同批次目录下按产物类型分开。
    # 中间产物在 <OUT_DIR>/mp4_work/<task>/，最终 mp4 为 <OUT_DIR>/<task>.mp4
    OUT_DIR = Path(os.environ.get(
        "OUT_DIR", str(RESULTS_ROOT / SRC_ROOT.resolve().name / "mp4")))
    # 帧序列目录。留空则自动探测 task 下的 image/ images/ input/ frames/；
    # 填了就对所有 task 生效（UWA / COLMAP 都会用它，不再走自动探测）。
    # 相对路径的基点依次试 task 目录 → 批次根 → 结果根 → cwd，命中即停并打印。
    IMAGE_DIR = os.environ.get(
        "IMAGE_DIR", "../../code/Reconstruction/dataset/B003_Human_Data_w_pose")
    # ==============================================

    ref = os.environ.get("REF_JSON_DIR", "")
    cfg = {
        "tool_dir": TOOL_DIR,
        "gltf_packer": TOOL_DIR / "build/gltf_packer",
        "python_bin": os.environ.get("PYTHON_BIN", "python"),
        "fps": os.environ.get("FPS", "30"),
        "crf": os.environ.get("CRF", "28"),
        "preset": os.environ.get("PRESET", "fast"),
        "astc_block": os.environ.get("ASTC_BLOCK", "4"),
        "ply_name": os.environ.get("PLY_NAME", ""),
        "image_dir": IMAGE_DIR,
        "mode": os.environ.get("MODE", "auto").lower(),
        "ref_json_dir": Path(ref) if ref else None,
        "insideout": os.environ.get("INSIDEOUT", "0") == "1",
        "gs_fallback": os.environ.get("GS_FALLBACK", "0") == "1",
        "init_camera_name": os.environ.get("INIT_CAMERA_NAME", ""),
        "camera_name": os.environ.get("CAMERA_NAME", ""),
        "view_params_name": os.environ.get("VIEW_PARAMS_NAME", ""),
        "force": os.environ.get("FORCE", "0") == "1",
        # 默认保留中间产物：这四步任一步失败都要靠 work 目录里的东西定位
        "keep_work": os.environ.get("KEEP_WORK", "1") == "1",
        "dry_run": os.environ.get("DRY_RUN", "0") == "1",
    }
    only = [s for s in os.environ.get("ONLY", "").split(",") if s]

    # --- 前置检查 ---
    if not TOOL_DIR.is_dir():
        sys.exit(f"❌ 工具链目录不存在: {TOOL_DIR}")
    for f in ("encode.py", "muxer.py"):
        if not (TOOL_DIR / f).is_file():
            sys.exit(f"❌ 工具链缺少 {f}: {TOOL_DIR / f}")
    if not cfg["gltf_packer"].is_file():
        sys.exit(f"❌ gltf_packer 不存在: {cfg['gltf_packer']}")
    if not os.access(cfg["gltf_packer"], os.X_OK):
        sys.exit(f"❌ gltf_packer 不可执行（需 chmod +x）: {cfg['gltf_packer']}")
    if shutil.which("ffmpeg") is None:
        sys.exit("❌ PATH 里找不到 ffmpeg")
    if shutil.which(cfg["python_bin"]) is None:
        sys.exit(f"❌ PATH 里找不到解释器: {cfg['python_bin']}（用 PYTHON_BIN 指定）")
    if cfg["ref_json_dir"] and not cfg["ref_json_dir"].is_dir():
        sys.exit(f"❌ REF_JSON_DIR 不存在: {cfg['ref_json_dir']}")
    if cfg["mode"] not in ("auto", "uwa", "colmap"):
        sys.exit(f"❌ MODE 只能是 auto/uwa/colmap，当前: {cfg['mode']}")

    pack_batch(SRC_ROOT, OUT_DIR, cfg, only)


if __name__ == "__main__":
    main()
