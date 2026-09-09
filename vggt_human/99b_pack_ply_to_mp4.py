#!/usr/bin/env python3
"""99b_pack_ply_to_mp4.py — 批量把 PLY（3DGS 重建结果）封装成可播放的 MP4。

输入是「批次目录」，其下每个子目录是一个 task，里面同时放着 ply、图片序列和相机参数；
本脚本逐个 task 跑完 4 步流水线，在输出目录得到 <task_id>.mp4：

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
    PYTHON_BIN  跑 encode.py / muxer.py 的解释器（默认 python，需带工具链依赖）
    FPS / CRF / PRESET   视频编码参数（默认 30 / 28 / fast）
    ASTC_BLOCK  码流文件名的 ASTC 块大小（默认 4，仅用于拼文件名）
    PLY_NAME    指定 ply 文件名，不给则按候选列表自动探测
    IMAGE_DIR   指定图片子目录名，不给则按 image/images/input/frames 自动探测
    FORCE=1     覆盖已存在的 mp4（默认跳过）
    KEEP_WORK=0 成功后删除中间产物目录 <task>/mp4_work（默认保留，方便排查）
    DRY_RUN=1   只打印探测结果，不执行
"""
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


# --------------------------------------------------------------------------
# 工具函数
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


# --------------------------------------------------------------------------
# 单个 task 的封装流程
# --------------------------------------------------------------------------
def pack_one(task_dir: Path, out_mp4: Path, cfg: dict, env: dict) -> bool:
    print(f"\n================ {task_dir.name} ================")

    # --- 探测输入 ---
    ply = detect_ply(task_dir, cfg["ply_name"])
    if ply is None:
        print(f"❌ 找不到 ply（候选: {PLY_CANDIDATES} 或任一 *.ply）")
        return False

    img_dir = detect_image_dir(task_dir, cfg["image_dir"])
    if img_dir is None:
        print(f"❌ 找不到图片目录（候选: {IMAGE_DIR_CANDIDATES}）")
        return False

    seq = detect_image_sequence(img_dir)
    if seq is None:
        print(f"❌ {img_dir} 下没有 <数字>.jpg/png 形式的图片序列")
        return False

    jsons = {}
    for key, default_name in (
        ("init_camera", "init_camera.json"),
        ("camera", "camera.json"),
        ("view_params", "view_limits.json"),
    ):
        p = find_json(task_dir, default_name, cfg.get(f"{key}_name", ""))
        if p is None:
            print(f"❌ 找不到 {default_name}（task 目录或其一层子目录）")
            return False
        jsons[key] = p

    print(f"🖼️ ply:    {ply.name}")
    print(f"🖼️ 图片:   {img_dir.name}/%0{seq['pad']}d.{seq['ext']}"
          f"  (起始 {seq['start']}, 共 {seq['count']} 张)")
    print(f"📐 相机:   {jsons['init_camera'].name} / {jsons['camera'].name} / {jsons['view_params'].name}")
    print(f"💾 输出:   {out_mp4}")

    if cfg["dry_run"]:
        print("⏭️  DRY_RUN=1，跳过执行")
        return True

    work_dir = out_mp4.parent / "mp4_work" / task_dir.name
    work_dir.mkdir(parents=True, exist_ok=True)
    video_file = work_dir / "output.mp4"
    glb_file = work_dir / "3DGS.glb"
    t0 = time.time()

    # --- Step 1/4: 图片序列 → 视频 ---
    print("[Step 1/4] 🎬 生成 H.264 视频...")
    # pad=ceil(w/2)*2: yuv420p 要求宽高为偶数，奇数分辨率会直接编码失败
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
        return False
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
            return False
    print(f"  ✅ 码流: {bin_file}  ({bin_file.stat().st_size / 1024 / 1024:.1f} MB)")

    # --- Step 3/4: 码流 + 相机 → GLB ---
    print("[Step 3/4] 📦 封装为 GLB...")
    rc = run_cmd([
        str(cfg["gltf_packer"]), str(bin_file), str(glb_file),
        str(jsons["init_camera"]), str(jsons["camera"]), str(jsons["view_params"]),
    ])
    if rc != 0 or not glb_file.is_file():
        print("❌ Step 3 失败")
        return False
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
        return False

    print(f"  ✅ {out_mp4.name}  ({out_mp4.stat().st_size / 1024 / 1024:.1f} MB)"
          f"  ⏱️ {time.time() - t0:.1f}s")

    if not cfg["keep_work"]:
        shutil.rmtree(work_dir, ignore_errors=True)
        print(f"  🧹 已清理中间产物: {work_dir}")
    return True


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

    ok = skip = fail = 0
    failed = []
    for sub in subdirs:
        out_mp4 = out_root / f"{sub.name}.mp4"
        if out_mp4.is_file() and not cfg["force"] and not cfg["dry_run"]:
            print(f"⏭️  skip  {sub.name}  (已存在 {out_mp4.name}，FORCE=1 可覆盖)")
            skip += 1
            continue
        if pack_one(sub, out_mp4, cfg, env):
            ok += 1
        else:
            fail += 1
            failed.append(sub.name)

    print()
    print(f"🎉 Done.  ✅ {ok}  ⏭️ {skip}  ❌ {fail}")
    if failed:
        print(f"❌ 失败列表: {', '.join(failed)}")
    print(f"📁 结果: {out_root}")


def main():
    # ===== 在这里直接改路径（或用环境变量覆盖）=====
    TOOL_DIR = Path(os.environ.get(
        "TOOL_DIR", "/data_3d/w00950754/model/UWA_Sample_Tool_v3"))
    # 批次根目录：其下每个子目录是一个 task（含 ply + image/ + 三个 json）
    SRC_ROOT = Path(os.environ.get(
        "SRC_ROOT",
        "/data_3d/w00950754/code/Reconstruction/output/"
        "B003_Human_Data_w_pose-脸红优化+外插视角增强+互补双监督"))
    # 输出目录：默认放在源目录旁边的 <批次名>_mp4
    OUT_DIR = Path(os.environ.get("OUT_DIR", str(SRC_ROOT) + "_mp4"))
    # ==============================================

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

    pack_batch(SRC_ROOT, OUT_DIR, cfg, only)


if __name__ == "__main__":
    main()
