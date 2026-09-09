#!/usr/bin/env python3
"""99b_pack_ply_to_mp4.py — 批量把 PLY（3DGS 重建结果）封装成可播放的 MP4。

输入是「批次目录」，其下每个子目录是一个 task。脚本**自动判断 task 属于哪种模式**并打印：

    模式 UWA     init_camera.json + camera.json + view_limits.json + recon_result.ply
                → 直接跑完整四步链，得到 <task_id>.mp4
    模式 COLMAP  cameras.txt + images.txt + points3D.txt + processing.txt + point_cloud_final.ply
                → 缺 UWA 三件套 json。解析 COLMAP 打印相机/场景统计，并把算出来的数
                  写成待校验的 json 骨架；拿到真样本后用 REF_JSON_DIR 覆盖即可继续封装。

四步链（仅 UWA 模式或有 json 时执行）:
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
    OUT_DIR     输出目录（默认 <SRC_ROOT>_mp4），产出 <task_id>.mp4
    ONLY        逗号分隔的 task 白名单
    MODE        auto（默认）/ uwa / colmap，强制指定模式
    REF_JSON_DIR  含真实的 init_camera.json / camera.json / view_limits.json 的目录，
                  给 COLMAP 模式的 task 借用（优先级最高，不再走推断）
    PYTHON_BIN  跑 encode.py / muxer.py 的解释器（默认 python，需带工具链依赖）
    FPS / CRF / PRESET   视频编码参数（默认 30 / 28 / fast）
    ASTC_BLOCK  码流文件名的 ASTC 块大小（默认 4，仅用于拼文件名）
    PLY_NAME    指定 ply 文件名，不给则按候选列表自动探测
    IMAGE_DIR   指定图片子目录名，不给则自动探测
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

# ply 自动探测顺序：不同流水线产物名不一样，按常见程度排
PLY_CANDIDATES = [
    "recon_result.ply",
    "point_cloud_final.ply",
    "point_cloud.ply",
    "scene.ply",
]
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
def detect_ply(task_dir: Path, ply_name: str = "") -> Path | None:
    if ply_name:
        p = task_dir / ply_name
        return p if p.is_file() else None
    for name in PLY_CANDIDATES:
        p = task_dir / name
        if p.is_file():
            return p
    # 兜底：目录里任意一个 .ply（多个时取名字排序第一个，保证可复现）
    plies = sorted(task_dir.glob("*.ply"))
    return plies[0] if plies else None


def detect_image_dir(task_dir: Path, image_dir: str = "") -> Path | None:
    if image_dir:
        d = task_dir / image_dir
        return d if d.is_dir() else None
    for name in IMAGE_DIR_CANDIDATES:
        d = task_dir / name
        if d.is_dir():
            return d
    return None


def detect_image_sequence(img_dir: Path):
    """识别图片序列的 扩展名 / 数字位宽 / 起始编号。

    返回 dict(ext, pad, start, count) 或 None。
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
    同时存在时以 UWA 优先（json 齐全才能直接封装）。
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
# COLMAP 解析（纯 python，不依赖 numpy）
# --------------------------------------------------------------------------
def qvec2rotmat(q):
    """COLMAP qvec = [qw, qx, qy, qz] → 3x3 旋转矩阵（行优先 list）。"""
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]


def _intrinsics_from_params(model: str, params):
    """从 COLMAP 相机参数里取 fx/fy/cx/cy。"""
    m = model.upper()
    if m in ("PINHOLE", "OPENCV", "FULL_OPENCV", "SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        if m in ("PINHOLE", "OPENCV", "FULL_OPENCV"):
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        else:
            # SIMPLE_* / RADIAL：第 1 个参数是共享焦距 f
            fx = fy = params[0]
            cx, cy = params[1], params[2]
        return fx, fy, cx, cy
    # 未知模型：退化处理，按 PINHOLE 前 4 个参数猜
    if len(params) >= 4:
        return params[0], params[1], params[2], params[3]
    return None


def read_colmap_cameras(path: Path) -> dict:
    """cameras.txt → {camera_id: {model,width,height,fx,fy,cx,cy}}"""
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        cid = int(parts[0])
        model = parts[1]
        w, h = int(parts[2]), int(parts[3])
        params = [float(p) for p in parts[4:]]
        intr = _intrinsics_from_params(model, params)
        if intr is None:
            continue
        fx, fy, cx, cy = intr
        out[cid] = {"model": model, "width": w, "height": h,
                    "fx": fx, "fy": fy, "cx": cx, "cy": cy}
    return out


def read_colmap_images(path: Path) -> list:
    """images.txt → [{image_id, qvec, tvec, camera_id, name}]。

    位姿行固定 10 列（id qw qx qy qz tx ty tz camera_id name），其后紧跟一行
    points2D（**可能是空行**）。所以不能先滤空行再按两行跳读——空行一滤，
    张数就少一半。判定规则：列数 >=10 且不是 3 的倍数（points2D 行恒为 3k 列）。
    """
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10 or len(parts) % 3 == 0:
            continue  # points2D 行（3k 列）或异常行
        out.append({
            "image_id": int(parts[0]),
            "qvec": [float(v) for v in parts[1:5]],
            "tvec": [float(v) for v in parts[5:8]],
            "camera_id": int(parts[8]),
            "name": parts[9],
        })
    return out


def colmap_summary(task_dir: Path):
    """汇总 COLMAP 信息：相机内参、图像数、场景中心/半径、首帧位姿。"""
    cams = read_colmap_cameras(task_dir / "cameras.txt")
    imgs = read_colmap_images(task_dir / "images.txt")
    if not imgs or not cams:
        return None

    # 主相机：被最多图像引用的那台
    cid_count = Counter(im["camera_id"] for im in imgs)
    main_cid = cid_count.most_common(1)[0][0]
    if len(cams) > 1:
        print(f"  ⚠️ 有 {len(cams)} 台相机，取用得最多的 #{main_cid}"
              f"（各台数量: {dict(cid_count)}）")
    cam = cams[main_cid]

    # 相机中心 C = -R^T t；场景中心和半径用相机中心估（相机绕物体，够用且便宜）
    centers = []
    first_pose = None
    for im in sorted(imgs, key=lambda d: d["name"]):
        R = qvec2rotmat(im["qvec"])
        t = im["tvec"]
        C = [-sum(R[k][j] * t[k] for k in range(3)) for j in range(3)]
        centers.append(C)
        if first_pose is None:
            first_pose = C

    n = len(centers)
    center = [sum(c[j] for c in centers) / n for j in range(3)]
    radius = max(math.dist(c, center) for c in centers)

    fovx = 2 * math.degrees(math.atan(cam["width"] / (2 * cam["fx"]))) if cam["fx"] else 0.0
    fovy = 2 * math.degrees(math.atan(cam["height"] / (2 * cam["fy"]))) if cam["fy"] else 0.0

    return {
        "num_images": len(imgs),
        "num_cameras": len(cams),
        "camera": cam,
        "fovx": fovx,
        "fovy": fovy,
        "center": center,
        "radius": radius,
        "first_position": first_pose,
        "image_names": [im["name"] for im in sorted(imgs, key=lambda d: d["name"])],
    }


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


def build_uwa_json_skeleton(summary: dict) -> dict:
    """按推断的字段名生成三件套骨架。

    ⚠️ 字段名是推断的，没有真实样本校验过。真样本到手后：
       ① 改本函数的 key 名（集中在这一个地方）；
       ② 或用 REF_JSON_DIR 直接提供已知可用的 json 跳过本函数。
    """
    cam = summary["camera"]
    note = ("⚠️ 由 COLMAP 推断生成，字段名未经真实样本校验；"
            "确认后删掉 _note 并用 REF_JSON_DIR 或改 build_uwa_json_skeleton()")
    return {
        "init_camera.json": {
            "_note": note,
            "position": [round(v, 6) for v in summary["first_position"]],
            "target": [round(v, 6) for v in summary["center"]],
            # COLMAP 世界系 Y 朝下，与常见 Y-up 查看器相反，故给 -Y
            "up": [0, -1, 0],
            "fov": round(summary["fovx"], 3),
        },
        "camera.json": {
            "_note": note,
            "camera_model": cam["model"],
            "width": cam["width"],
            "height": cam["height"],
            "fx": round(cam["fx"], 4),
            "fy": round(cam["fy"], 4),
            "cx": round(cam["cx"], 4),
            "cy": round(cam["cy"], 4),
            "fovx": round(summary["fovx"], 3),
            "fovy": round(summary["fovy"], 3),
            "num_images": summary["num_images"],
        },
        "view_limits.json": {
            "_note": note,
            "center": [round(v, 6) for v in summary["center"]],
            "radius": round(summary["radius"], 6),
            "min_distance": round(summary["radius"] * 0.3, 6),
            "max_distance": round(summary["radius"] * 4.0, 6),
            "min_pitch": -60.0,
            "max_pitch": 60.0,
        },
    }


def write_json_skeleton(out_dir: Path, skeleton: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, obj in skeleton.items():
        (out_dir / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False))


# --------------------------------------------------------------------------
# 单个 task 的处理
# --------------------------------------------------------------------------
def prepare_jsons(task_dir: Path, mode: str, info: dict, work_dir: Path, cfg: dict):
    """拿到 gltf_packer 需要的三件套 json。

    返回 (jsons|None, status)：
      ("ok")      jsons 可用
      ("pending") COLMAP 模式暂无 json，已打印统计 + 落骨架，等真样本
      ("failed")  真出错
    """
    if mode == MODE_UWA:
        return {k: v for k, v in info["uwa"].items()}, "ok"

    # ---- COLMAP 模式 ----
    # 优先级 1：外部提供已知可用的 json
    ref = cfg.get("ref_json_dir")
    if ref:
        got = {k: (ref / n) for k, n in UWA_JSONS.items()}
        missing = [n for k, n in UWA_JSONS.items() if not got[k].is_file()]
        if missing:
            print(f"  ❌ REF_JSON_DIR 缺少: {', '.join(missing)}")
            return None, "failed"
        print(f"  📎 借用 REF_JSON_DIR 的三件套 json: {ref}")
        return got, "ok"

    # 优先级 2：解析 COLMAP，打印统计 + 落骨架文件，然后停（不擅自封装）
    summary = colmap_summary(task_dir)
    skel_dir = work_dir / "uwa_json"
    if summary is None:
        print("  ❌ COLMAP 解析失败（cameras.txt / images.txt 为空或格式异常）")
        return None, "failed"

    cam = summary["camera"]
    print(f"  📷 相机 {summary['num_cameras']} 台, 主相机 {cam['model']} "
          f"{cam['width']}x{cam['height']}, fx={cam['fx']:.1f} fy={cam['fy']:.1f} "
          f"cx={cam['cx']:.1f} cy={cam['cy']:.1f}")
    print(f"  🖼️ 图像 {summary['num_images']} 张, fovx={summary['fovx']:.1f}° "
          f"fovy={summary['fovy']:.1f}°")
    print(f"  📐 场景中心 [{summary['center'][0]:.3f}, {summary['center'][1]:.3f}, "
          f"{summary['center'][2]:.3f}], 半径 {summary['radius']:.3f}")
    print(f"  📍 首帧相机位置 [{summary['first_position'][0]:.3f}, "
          f"{summary['first_position'][1]:.3f}, {summary['first_position'][2]:.3f}]")
    write_json_skeleton(skel_dir, build_uwa_json_skeleton(summary))
    print(f"  📝 推断值已写入（字段名未校验，仅供参考）: {skel_dir}")
    print("  ⏸️ 无 UWA 三件套 json 真实样本，不做推断封装。"
          "给 REF_JSON_DIR，或按真样本改 build_uwa_json_skeleton() 后重跑。")
    return None, "pending"


def pack_one(task_dir: Path, out_mp4: Path, cfg: dict, env: dict) -> str:
    """处理一个 task，返回 "ok" / "pending"（缺 json 待补） / "failed"。"""
    print(f"\n================ {task_dir.name} ================")

    # --- 1. 模式判断 ---
    mode, info = detect_mode(task_dir, cfg)
    if mode is None:
        print(f"❌ 模式未知：既没有完整的 UWA 三件套 json，也没有 cameras.txt+images.txt")
        print(f"   UWA 命中: {[n for k, n in UWA_JSONS.items() if info['uwa'][k]] or '无'}")
        print(f"   COLMAP 命中: {[f for f, ok in info['colmap'].items() if ok] or '无'}")
        return "failed"

    if mode == MODE_UWA:
        print(f"📦 模式: {MODE_UWA}  "
              f"({' + '.join(UWA_JSONS.values())} + <ply>)")
    else:
        have = [f for f, ok in info["colmap"].items() if ok]
        print(f"📦 模式: {MODE_COLMAP}  ({' + '.join(have)} + <ply>)")
        if info["uwa_any"]:
            print("  ℹ️ 该目录也有部分 UWA json，但不齐全，按 COLMAP 处理")

    # --- 2. ply / 图片 ---
    ply = detect_ply(task_dir, cfg["ply_name"])
    if ply is None:
        print(f"❌ 找不到 ply（候选: {PLY_CANDIDATES} 或任一 *.ply）")
        return "failed"
    print(f"🖼️ ply:    {ply.name}")

    if cfg["image_dir"]:
        img_dir = detect_image_dir(task_dir, cfg["image_dir"])
    elif mode == MODE_COLMAP:
        names = []
        try:
            names = [im["name"] for im in read_colmap_images(task_dir / "images.txt")]
        except Exception as e:
            print(f"⚠️ 读 images.txt 失败: {e}")
        img_dir = resolve_colmap_image_dir(task_dir, names) or detect_image_dir(task_dir)
    else:
        img_dir = detect_image_dir(task_dir)

    if img_dir is None:
        print(f"❌ 找不到图片目录（候选: {IMAGE_DIR_CANDIDATES}）")
        return "failed"
    seq = detect_image_sequence(img_dir)
    if seq is None:
        print(f"❌ {img_dir} 下没有 <数字>.jpg/png 形式的图片序列")
        return "failed"
    print(f"🖼️ 图片:   {img_dir.name}/%0{seq['pad']}d.{seq['ext']}"
          f"  (起始 {seq['start']}, 共 {seq['count']} 张)")

    # --- 3. 三件套 json ---
    work_dir = out_mp4.parent / "mp4_work" / task_dir.name
    if cfg["dry_run"]:
        print(f"💾 输出(预览): {out_mp4}")
        print("⏭️  DRY_RUN=1，跳过执行")
        return "ok"

    work_dir.mkdir(parents=True, exist_ok=True)
    jsons, status = prepare_jsons(task_dir, mode, info, work_dir, cfg)
    if jsons is None:
        return status
    print(f"📐 相机:   {jsons['init_camera'].name} / {jsons['camera'].name} / {jsons['view_params'].name}")
    print(f"💾 输出:   {out_mp4}")

    video_file = work_dir / "output.mp4"
    glb_file = work_dir / "3DGS.glb"
    t0 = time.time()

    # --- Step 1/4: 图片序列 → 视频 ---
    print("[Step 1/4] 🎬 生成 H.264 视频...")
    # pad=ceil(iw/2)*2: yuv420p 要求宽高为偶数，奇数分辨率会直接编码失败
    rc = run_cmd([
        "ffmpeg", "-y",
        "-framerate", str(cfg["fps"]),
        "-start_number", str(seq["start"]),
        "-i", f"{img_dir}/%0{seq['pad']}d.{seq['ext']}",
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

    # 工具链 PYTHONPATH：muxer.py 依赖 pymp4，必须把 thirdparty 和工具链根都加进去
    env = os.environ.copy()
    pymp4 = cfg["tool_dir"] / "thirdparty/pymp4-1.4.0/src"
    parts = [str(cfg["tool_dir"])]
    if pymp4.is_dir():
        parts.append(str(pymp4))
    old = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(parts + ([old] if old else []))

    print(f"🔍 源目录:   {src_root}")
    print(f"📁 输出目录: {out_root}")
    print(f"🛠️ 工具链:   {cfg['tool_dir']}")
    if cfg.get("ref_json_dir"):
        print(f"📎 外部 json: {cfg['ref_json_dir']}")
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
        print(f"⏸️ 待补 json: {', '.join(pending)}")
    if failed:
        print(f"❌ 失败列表: {', '.join(failed)}")
    print(f"📁 结果: {out_root}")


def main():
    # ===== 在这里直接改路径（或用环境变量覆盖）=====
    TOOL_DIR = Path(os.environ.get(
        "TOOL_DIR", "/data_3d/w00950754/model/UWA_Sample_Tool_v3"))
    # 批次根目录：其下每个子目录是一个 task（UWA 型或 COLMAP 型混着也行）
    SRC_ROOT = Path(os.environ.get(
        "SRC_ROOT",
        "/data_3d/w00950754/code/Reconstruction/output/"
        "B003_Human_Data_w_pose-脸红优化+外插视角增强+互补双监督"))
    # 输出目录：默认放在源目录旁边的 <批次名>_mp4
    OUT_DIR = Path(os.environ.get("OUT_DIR", str(SRC_ROOT) + "_mp4"))
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
        "image_dir": os.environ.get("IMAGE_DIR", ""),
        "mode": os.environ.get("MODE", "auto").lower(),
        "ref_json_dir": Path(ref) if ref else None,
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
