#!/usr/bin/env python3
"""99e_unpack_mp4_to_ply.py — 解封装单个 MP4 回 PLY（99b 的严格镜像）。

99b 正向四步：图片→视频 | PLY→bin | bin+相机→GLB | GLB+视频→MP4
本脚本反向三步（单样例，不批量，始终走 demux）：
  [1/3] demuxer.py      MP4  → GLB + 视频
  [2/3] gltf_unpacker   GLB  → bin + 三件套 json
  [3/3] decode.py       bin  → PLY

⚠️ 已知局限（务必先读）
  - 还原出的 view_limits.json 必然是残缺版：gltf_packer 只往
    UWA_viewing_parameters 写 longitude/latitude/distance/gravity/target/
    boundingbox 六项白名单（详见 NOTES.md 第 9 条），其余字段打包时已丢弃。
  - PLY 未必位级无损：encode/decode 若含量化，往返会有数值偏差。
  - 反向产出的三件套 json 不可当原始 json 回灌 99b/99c。

用法:
    python vggt_human/99e_unpack_mp4_to_ply.py /path/to/taskA.mp4
    MP4=/path/to/taskA.mp4 python vggt_human/99e_unpack_mp4_to_ply.py
    PROBE=1 TOOL_DIR=../../model/UWA_Sample_Tool_v3 \
        python vggt_human/99e_unpack_mp4_to_ply.py     # 只看三个工具的 usage

Env vars:
    MP4          输入 MP4（也可用命令行第一个位置参数）
    TOOL_DIR     工具链根（含 demuxer.py / decode.py / build/gltf_unpacker）
    PYTHON_BIN   跑 demuxer.py / decode.py 的解释器（默认 python）
    OUT_DIR      输出目录（默认 <MP4 所属批次>/unpack_ply，产出 <task>.ply）
    WORK_DIR     中间产物目录（默认 <批次>/unpack_work/<task>）
    KEEP_WORK    默认 1 保留中间产物（GLB/bin/视频/json）
    FORCE        默认 0；已存在 PLY 时跳过，置 1 覆盖
    DRY_RUN      默认 0；只打印三步命令不执行
    PROBE        默认 0；置 1 只跑三个工具的 usage 后退出
    ASTC_BLOCK   期望的 bin 块大小（默认 4，仅用于拼文件名；对不上自动 glob 兜底）
    DEMUX_ARGS / UNPACK_ARGS / DECODE_ARGS
                 完整参数模板覆盖（留空用下方默认镜像参数）。占位符：
                   DEMUX_ARGS   {mp4} {glb} {video}
                   UNPACK_ARGS  {glb} {bin} {init} {camera} {view} {outdir}
                   DECODE_ARGS  {bin} {ply} {outdir}
                 例：DEMUX_ARGS='--in {mp4} --glb {glb} --video {video}'
"""
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

# 工具链内三个反向工具的相对路径（与 99b 正向一一对应）
DEMUXER = "demuxer.py"
UNPACKER = "build/gltf_unpacker"
DECODER = "decode.py"

# 反向三件套 json 名（gltf_unpacker 的产出）
UWA_JSONS = {
    "init_camera": "init_camera.json",
    "camera": "camera.json",
    "view_params": "view_limits.json",
}

# 默认参数模板：按 99b 正向调用的「镜像推测」。参数名对不上时用 PROBE=1 查真名，
# 再用对应 *_ARGS 覆盖，无需改脚本。
DEMUX_ARGS_DEFAULT = "--mp4_path {mp4} --glb_path {glb} --output_path {video}"
UNPACK_ARGS_DEFAULT = "{glb} {bin} {init} {camera} {view}"
DECODE_ARGS_DEFAULT = "--bin-path {bin} --save-dir {outdir}"


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


def build_args(env_name: str, default: str, **kw) -> list:
    """把模板渲染成参数列表：env 覆盖优先，占位符用 kw 里的路径填充。"""
    tpl = os.environ.get(env_name, "").strip() or default
    try:
        return shlex.split(tpl.format(**{k: str(v) for k, v in kw.items()}))
    except KeyError as e:
        sys.exit(f"❌ {env_name} 里的占位符未知: {e}（可用: {', '.join(kw)}）")


def make_env(tool_dir: Path) -> dict:
    """工具链 PYTHONPATH：demuxer/muxer 依赖 pymp4，thirdparty 与根目录都要加进去。"""
    env = os.environ.copy()
    parts = [str(tool_dir)]
    for p in sorted(tool_dir.glob("thirdparty/pymp4*/src")):
        parts.append(str(p))
    old = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(parts + ([old] if old else []))
    return env


# --------------------------------------------------------------------------
# 输入 / 路径解析
# --------------------------------------------------------------------------
def resolve_mp4() -> Path:
    """输入 MP4：命令行第一个位置参数优先，其次 env MP4。"""
    arg = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else ""
    raw = arg or os.environ.get("MP4", "").strip()
    if not raw:
        sys.exit("❌ 未指定输入 MP4。用法: python vggt_human/99e_unpack_mp4_to_ply.py /path/to/x.mp4"
                 "（或先用 MP4=/path/to/x.mp4）")
    mp4 = Path(raw).expanduser()
    if not mp4.is_file():
        sys.exit(f"❌ MP4 不存在: {mp4}")
    return mp4.resolve()


def resolve_dirs(mp4: Path):
    """推导 task 名 / 批次根 / 输出目录 / 工作目录。

    99b 的产物布局是 <批次>/mp4/<task>.mp4（99c 是 mp4_crop/），命中则批次根上移一层；
    否则（别人给的孤立 MP4）就把 MP4 所在目录当批次根。
    """
    task = mp4.stem
    batch = mp4.parent.parent if mp4.parent.name in ("mp4", "mp4_crop") else mp4.parent
    out_dir = Path(os.environ.get("OUT_DIR", str(batch / "unpack_ply")))
    work_dir = Path(os.environ.get("WORK_DIR", str(batch / "unpack_work" / task)))
    return task, batch, out_dir, work_dir


# --------------------------------------------------------------------------
# PROBE：打印三个反向工具的 usage，便于核对参数名
# --------------------------------------------------------------------------
def probe(tool_dir: Path, python_bin: str, env: dict) -> None:
    print(f"🔎 PROBE: 工具链 {tool_dir}\n")
    targets = [
        ("demuxer.py", tool_dir / DEMUXER, [python_bin, DEMUXER]),
        ("gltf_unpacker", tool_dir / UNPACKER, [str(tool_dir / UNPACKER)]),
        ("decode.py", tool_dir / DECODER, [python_bin, DECODER]),
    ]
    for name, path, base in targets:
        print(f"================ {name} ================")
        if not path.is_file():
            print(f"  ⚠️ 工具不存在: {path}")
            print()
            continue
        shown = False
        for extra in (["--help"], ["-h"], []):
            proc = subprocess.run([str(c) for c in base + extra], cwd=str(tool_dir),
                                  env=env, capture_output=True, text=True)
            out = (proc.stdout or "") + (proc.stderr or "")
            if out.strip():
                print(f"$ {' '.join(str(c) for c in base + extra)}")
                print(_tail(out, 40))
                shown = True
                break
        if not shown:
            print("  （无输出；工具可能必须带参数才能打印用法）")
        print()


# --------------------------------------------------------------------------
# 解封装（单样例三步）
# --------------------------------------------------------------------------
def find_newest(d: Path, pattern: str) -> Path | None:
    cands = sorted(d.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def unpack_one(mp4: Path, out_dir: Path, work_dir: Path, cfg: dict, env: dict) -> str:
    """单样例解包：MP4 → GLB+视频 → bin+json → PLY。返回 ok/skip/failed。"""
    task = mp4.stem
    out_ply = out_dir / f"{task}.ply"
    if out_ply.is_file() and not cfg["force"] and not cfg["dry_run"]:
        print(f"⏭️ 已存在，跳过: {out_ply}（FORCE=1 可覆盖）")
        return "skip"

    work_dir.mkdir(parents=True, exist_ok=True)
    if not cfg["dry_run"]:
        out_dir.mkdir(parents=True, exist_ok=True)

    glb = work_dir / "3DGS.glb"
    video = work_dir / "output.mp4"
    bin_file = work_dir / f"GSCompressed_B{cfg['astc_block']}.bin"
    j_init = work_dir / UWA_JSONS["init_camera"]
    j_cam = work_dir / UWA_JSONS["camera"]
    j_view = work_dir / UWA_JSONS["view_params"]

    # 三步命令统一走 *_ARGS 模板（参数名与镜像推测不符时可 env 覆盖，无需改脚本）
    demux_cmd = [cfg["python_bin"], DEMUXER] + build_args(
        "DEMUX_ARGS", DEMUX_ARGS_DEFAULT, mp4=mp4, glb=glb, video=video)
    unpack_cmd = [str(cfg["tool_dir"] / UNPACKER)] + build_args(
        "UNPACK_ARGS", UNPACK_ARGS_DEFAULT, glb=glb, bin=bin_file,
        init=j_init, camera=j_cam, view=j_view, outdir=work_dir)
    decode_cmd = [cfg["python_bin"], DECODER] + build_args(
        "DECODE_ARGS", DECODE_ARGS_DEFAULT, bin=bin_file, ply=out_ply, outdir=work_dir)

    print("[Step 1/3] 🎬 解复用 MP4 → GLB + 视频")
    print(f"  $ {' '.join(str(c) for c in demux_cmd)}")
    print("[Step 2/3] 📦 解包 GLB → 码流 + 三件套 json")
    print(f"  $ {' '.join(str(c) for c in unpack_cmd)}")
    print("[Step 3/3] 📦 解码码流 → PLY")
    print(f"  $ {' '.join(str(c) for c in decode_cmd)}")

    if cfg["dry_run"]:
        print("⏭️ DRY_RUN=1，以上命令未执行")
        return "ok"

    t0 = time.time()

    # --- Step 1/3: MP4 → GLB + 视频 ---
    rc = run_cmd(demux_cmd, cwd=cfg["tool_dir"], env=env)
    if rc != 0 or not glb.is_file():
        print(f"❌ Step 1 失败：没得到 GLB（{glb}）")
        print("   若参数名不符，先 PROBE=1 查真名，再用 DEMUX_ARGS 覆盖")
        return "failed"
    print(f"  ✅ GLB: {glb.name}  ({glb.stat().st_size / 1024 / 1024:.1f} MB)")

    # --- Step 2/3: GLB → bin + 三件套 json ---
    rc = run_cmd(unpack_cmd, cwd=cfg["tool_dir"], env=env)
    if not bin_file.is_file():
        # 块大小由工具内部决定，文件名对不上时按实际产物兜底
        alt = find_newest(work_dir, "GSCompressed_B*.bin") or find_newest(work_dir, "*.bin")
        if alt:
            print(f"⚠️ 期望 {bin_file.name} 不存在，改用实际产物 {alt.name}")
            bin_file = alt
        else:
            print(f"❌ Step 2 失败：{work_dir} 下没有码流 (*.bin)")
            return "failed"
    if rc != 0:
        print("⚠️ gltf_unpacker 返回非 0，但码流已产出，继续")
    print(f"  ✅ 码流: {bin_file.name}  ({bin_file.stat().st_size / 1024 / 1024:.1f} MB)")

    got = [j.name for j in (j_init, j_cam, j_view) if j.is_file()]
    if len(got) < 3:
        others = sorted(p.name for p in work_dir.glob("*.json"))
        print(f"  ⚠️ 三件套只拿到 {len(got)}/3；work 下现有 json: {others or '无'}")
    else:
        print("  ✅ 三件套 json: " + ", ".join(got))

    # --- Step 3/3: bin → PLY ---
    before = {p.resolve() for p in work_dir.glob("*.ply")}
    rc = run_cmd(decode_cmd, cwd=cfg["tool_dir"], env=env)
    produced = out_ply if out_ply.is_file() else None
    if produced is None:
        # 只认本次新产出的 ply，避免捡到上一轮的残留
        fresh = [p for p in sorted(work_dir.glob("*.ply"),
                                   key=lambda q: q.stat().st_mtime, reverse=True)
                 if p.resolve() not in before]
        produced = fresh[0] if fresh else None
    if produced is None or not produced.is_file():
        print(f"❌ Step 3 失败：没找到解出的 PLY（既不在 {out_ply}，也不在 {work_dir}/*.ply）")
        return "failed"
    if produced != out_ply:
        shutil.move(str(produced), str(out_ply))
        print(f"  ↩️ 重命名 {produced.name} → {out_ply.name}")

    print(f"  ✅ {out_ply}  ({out_ply.stat().st_size / 1024 / 1024:.1f} MB)"
          f"  ⏱️ {time.time() - t0:.1f}s")
    return "ok"


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------
def main():
    # ===== 在这里直接改路径（或用环境变量覆盖）=====
    TOOL_DIR = Path(os.environ.get("TOOL_DIR", "../../model/UWA_Sample_Tool_v3"))
    # ==============================================
    cfg = {
        "tool_dir": TOOL_DIR,
        "python_bin": os.environ.get("PYTHON_BIN", "python"),
        "astc_block": os.environ.get("ASTC_BLOCK", "4"),
        "force": os.environ.get("FORCE", "0") == "1",
        # 默认保留中间产物：GLB/bin/视频/json 是排查和复用的关键
        "keep_work": os.environ.get("KEEP_WORK", "1") == "1",
        "dry_run": os.environ.get("DRY_RUN", "0") == "1",
    }

    # --- 前置检查 ---
    if not TOOL_DIR.is_dir():
        sys.exit(f"❌ 工具链目录不存在: {TOOL_DIR}")
    missing = [f for f in (DEMUXER, UNPACKER, DECODER) if not (TOOL_DIR / f).is_file()]
    if missing:
        sys.exit(f"❌ 工具链缺少 {', '.join(missing)}（在 {TOOL_DIR} 下）")
    if not os.access(TOOL_DIR / UNPACKER, os.X_OK):
        sys.exit(f"❌ gltf_unpacker 不可执行（需 chmod +x）: {TOOL_DIR / UNPACKER}")
    if shutil.which(cfg["python_bin"]) is None:
        sys.exit(f"❌ PATH 里找不到解释器: {cfg['python_bin']}（用 PYTHON_BIN 指定）")

    env = make_env(TOOL_DIR)

    if os.environ.get("PROBE", "0") == "1":
        probe(TOOL_DIR, cfg["python_bin"], env)
        return

    mp4 = resolve_mp4()
    task, batch, out_dir, work_dir = resolve_dirs(mp4)

    print(f"🔍 输入 MP4: {mp4}")
    print(f"📦 批次:     {batch}")
    print(f"💾 输出 PLY: {out_dir / (task + '.ply')}")
    print(f"🧰 工作目录: {work_dir}")
    print(f"🛠️ 工具链:   {TOOL_DIR}")
    if cfg["dry_run"]:
        print("⏭️  DRY_RUN=1（只打印命令）")
    print()

    status = unpack_one(mp4, out_dir, work_dir, cfg, env)

    if status == "failed":
        sys.exit("❌ 解封装失败")
    if not cfg["dry_run"] and not cfg["keep_work"] and work_dir.is_dir():
        shutil.rmtree(work_dir, ignore_errors=True)
        print(f"🧹 已清理中间产物: {work_dir}")
    print("🎉 Done.")


if __name__ == "__main__":
    main()
