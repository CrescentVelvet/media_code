#!/usr/bin/env python3
"""99e_unpack_mp4_to_ply.py — 解封装单个 MP4 回 PLY（99b 的严格镜像）。

99b 正向四步：图片→视频 | PLY→bin | bin+相机→GLB | GLB+视频→MP4
本脚本反向三步（单样例，不批量，始终走 demux）：
  [1/3] demuxer.py      MP4  → GLB
  [2/3] gltf_unpacker   GLB  → bin + 三件套 json
  [3/3] decode.py       bin  → PLY

⚠️ 已知局限（务必先读）
  - 三件套 json 的完整性取决于工具链版本：
      · 原版 gltf_packer 只往 UWA_viewing_parameters 写 longitude/latitude/
        distance/gravity/target/boundingbox 六项白名单（NOTES.md 第 9 条），
        此时反向拿不到三件套；
      · cgltf 打了「补丁输出 json」后可以拿到三件套，但内容是否等价于打包前的
        原始 json 尚未验证——**回灌 99b/99c 前务必先比对字段**。
  - PLY 未必位级无损：encode/decode 若含量化，往返会有数值偏差。
  - 三件套命名在不同版本/补丁下不一致（如 cameras.json / init_cam.json /
    view_limit.json），脚本按 UWA_JSONS 候选名 + 名字子串两级容错识别；
    传参给 gltf_unpacker 时统一用官方名，便于回灌 99b/99c。
  - decode.py 依赖 astcenc 把 .astc 转 .bmp，且把它的报错重定向到 /dev/shm 临时日志
    （退出即删）——缺执行位时错误被吞掉，最终伪装成 PIL 的 FileNotFoundError:
    image0.bmp，极难定位。本脚本已在前置检查里拦这一项（见 check_astcenc）。

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
    ASTCENC_PATH 显式指定 astcenc 可执行文件（默认在 TOOL_DIR 下自动找）
    ASTCENC_AUTOFIX
                 默认 0：发现 astcenc 缺执行位时报错并给出 chmod 命令；
                 置 1 则自动补 chmod +x 后继续
    DEMUX_ARGS / UNPACK_ARGS / DECODE_ARGS
                 完整参数模板覆盖（留空用实测/镜像的默认参数）。占位符：
                   DEMUX_ARGS   {mp4} {glb} {video} {outdir}
                   UNPACK_ARGS  {glb} {bin} {init} {camera} {view} {outdir}
                   DECODE_ARGS  {bin} {ply} {outdir}
                 例：UNPACK_ARGS='{glb} {outdir}'
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

# 反向三件套 json 名。**首个是官方名**（99b 正向生成、99c 回灌、UWA 样本都用它），
# 传参给 gltf_unpacker 时用它，便于解出来的 json 直接回灌 99b/99c；
# 其余为已知别名（cgltf 补丁实测输出 cameras.json / init_cam.json / view_limit.json）。
UWA_JSONS = {
    "init_camera": ("init_camera.json", "init_cam.json"),
    "camera": ("camera.json", "cameras.json"),
    "view_params": ("view_limits.json", "view_limit.json"),
}
# 候选名全不命中时按文件名子串兜底归类。顺序不能反：init_cam.json 同时含
# "init" 和 "cam"，必须先匹配 init。
JSON_HINTS = (("init_camera", "init"), ("view_params", "view"), ("camera", "cam"))

# decode.py 实测产出名（--save-dir 下）。仍按 mtime 发现真实文件，这里只用于报错提示。
DECODE_PLY_NAME = "decode_point_cloud.ply"

# 默认参数模板。demuxer.py / decode.py 的 CLI 已于 2026-09-14 用 PROBE 实测：
#   demuxer.py     --mp4_path <mp4> --output_dir <dir>   # GLB 落到 dir，文件名由工具定
#   decode.py      --bitstream-path <bin> --save-dir <dir>
#   gltf_unpacker  --help 直接崩（C++ 二进制不处理 help）。5 位置参数模板实测可用：
#                  {bin} 位被正确尊重（码流落到了指定路径），但 3 个 json 位不产出文件
#                  —— 反向 GLB 只带 6 项白名单（NOTES 第 9 条），三件套拿不齐属常态。
DEMUX_ARGS_DEFAULT = "--mp4_path {mp4} --output_dir {outdir}"
UNPACK_ARGS_DEFAULT = "{glb} {bin} {init} {camera} {view}"
DECODE_ARGS_DEFAULT = "--bitstream-path {bin} --save-dir {outdir}"


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
    """把模板渲染成参数列表：env 覆盖优先，占位符用 kw 里的路径填充。

    先 split 模板再代入路径——若先代入后 split，shlex 会把 Windows 路径里的
    反斜杠当转义符吃掉（C:\\a\\b → C:ab）；含空格的路径也不会被二次拆开。
    """
    tpl = os.environ.get(env_name, "").strip() or default
    subs = {k: str(v) for k, v in kw.items()}
    out = []
    for part in shlex.split(tpl):
        try:
            out.append(part.format(**subs))
        except KeyError as e:
            sys.exit(f"❌ {env_name} 里的占位符未知: {e}（可用: {', '.join(kw)}）")
    return out


def make_env(tool_dir: Path) -> dict:
    """工具链 PYTHONPATH：demuxer/muxer 依赖 pymp4，thirdparty 与根目录都要加进去。"""
    env = os.environ.copy()
    parts = [str(tool_dir)]
    for p in sorted(tool_dir.glob("thirdparty/pymp4*/src")):
        parts.append(str(p))
    old = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(parts + ([old] if old else []))
    return env


# decode.py 内部经 shell 调用 astcenc 把 .astc 转 .bmp，且把它的报错重定向到 /dev/shm
# 临时日志（退出即删）。缺执行位时 Permission denied 被吞掉，只表现为 PIL 打不开
# image0.bmp——所以这里必须前置拦，不能等 Step 3 报错再猜。
ASTCENC_GLOBS = ("src/xencode/tools/astcenc*", "**/astcenc*")


def check_astcenc(tool_dir: Path) -> list:
    """检查 astcenc 可执行位。返回「存在但不可执行」的列表（空 = 通过）。

    ASTCENC_AUTOFIX=1 时直接补 chmod +x；找不到文件不报错（不同版本路径可能不同）。
    """
    explicit = os.environ.get("ASTCENC_PATH", "").strip()
    if explicit:
        cands = [Path(explicit)]
    else:
        cands = sorted({p for pat in ASTCENC_GLOBS for p in tool_dir.glob(pat)
                        if p.is_file()})
    if not cands:
        print("  ℹ️ 未找到 astcenc（版本差异，跳过；若 Step 3 报 image0.bmp 缺失，"
              "用 ASTCENC_PATH 指定）")
        return []

    autofix = os.environ.get("ASTCENC_AUTOFIX", "0") == "1"
    bad = []
    for p in cands:
        if os.access(p, os.X_OK):
            print(f"  ✅ astcenc 可执行: {p}")
            continue
        if autofix:
            try:
                p.chmod(p.stat().st_mode | 0o111)
                print(f"  🔧 已补执行位: {p}")
                continue
            except OSError as e:
                print(f"  ⚠️ 自动 chmod 失败: {e}")
        bad.append(p)
    return bad


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
        # C++ 二进制遇 --help 可能直接 terminate，所以无参也试一遍；
        # 三种尝试逐个打印，不因某一个有输出就停（单次尝试看不出参数形态）
        attempts = [[], ["--help"], ["-h"]] if name == "gltf_unpacker" \
            else [["--help"], ["-h"], []]
        for extra in attempts:
            shown = "  (无参数)" if not extra else ""
            proc = subprocess.run([str(c) for c in base + extra], cwd=str(tool_dir),
                                  env=env, capture_output=True, text=True)
            out = ((proc.stdout or "") + (proc.stderr or "")).strip()
            print(f"$ {' '.join(str(c) for c in base + extra)}{shown}   [rc={proc.returncode}]")
            print(_tail(out, 20) if out else "  （无输出）")
        print()


# --------------------------------------------------------------------------
# 解封装（单样例三步）
# --------------------------------------------------------------------------
def find_fresh(d: Path, pattern: str, since: float) -> Path | None:
    """按 mtime 找 since 之后新写出的文件。

    用 mtime 而不是「执行前后的文件集合差」：重跑时工具常覆盖同名文件，
    集合差看不出它是新的，会误判成没产出。
    """
    cands = [p for p in d.glob(pattern) if p.stat().st_mtime >= since]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def pick_glb(work_dir: Path, since: float) -> Path | None:
    """挑 demuxer 产出的 GLB：优先精确名 3DGS.glb，否则取本次最新的 *.glb。

    补丁版 unpacker 会额外写出 3DGS_1.glb 之类的副本，用名字偏好保证结果确定。
    """
    exact = work_dir / "3DGS.glb"
    if exact.is_file() and exact.stat().st_mtime >= since:
        return exact
    return find_fresh(work_dir, "*.glb", since)


def find_jsons(work_dir: Path, tool_dir: Path) -> dict:
    """收集三件套 json，返回 {key: Path}。

    两级容错：① 候选名逐个试（官方名优先）；② 全不命中时按文件名子串猜。
    工具若忽略给出的输出路径、把 json 写进 CWD（我们以 tool_dir 为 CWD），自动归位。
    """
    found = {}
    for key, cands in UWA_JSONS.items():
        for base in (work_dir, tool_dir):
            hit = next((base / n for n in cands if (base / n).is_file()), None)
            if hit is None:
                continue
            if base != work_dir:
                shutil.move(str(hit), str(work_dir / hit.name))
                print(f"  ↩️ 从 {base} 归位 {hit.name}")
                hit = work_dir / hit.name
            found[key] = hit
            break

    if len(found) < len(UWA_JSONS):
        for p in sorted(work_dir.glob("*.json")):
            low = p.name.lower()
            for key, hint in JSON_HINTS:
                if key not in found and hint in low:
                    found[key] = p
                    print(f"  ℹ️ 候选名未命中，按名字猜出 {key} ← {p.name}")
                    break
    return found


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

    glb = work_dir / "3DGS.glb"          # 期望名；demuxer 实际产出名未知，见 Step 1 兜底
    video = work_dir / "output.mp4"
    bin_file = work_dir / f"GSCompressed_B{cfg['astc_block']}.bin"
    # 传参用官方名（各候选名元组的首项），拿到别名时由 find_jsons 兜底识别
    j_init = work_dir / UWA_JSONS["init_camera"][0]
    j_cam = work_dir / UWA_JSONS["camera"][0]
    j_view = work_dir / UWA_JSONS["view_params"][0]

    # 三步命令统一走 *_ARGS 模板（参数名不符时可 env 覆盖，无需改脚本）。
    # 做成闭包：GLB / 码流的真实文件名要等上一步跑完才知道，届时重建命令。
    def mk_demux():
        return [cfg["python_bin"], DEMUXER] + build_args(
            "DEMUX_ARGS", DEMUX_ARGS_DEFAULT, mp4=mp4, glb=glb, video=video, outdir=work_dir)

    def mk_unpack(g):
        return [str(cfg["tool_dir"] / UNPACKER)] + build_args(
            "UNPACK_ARGS", UNPACK_ARGS_DEFAULT, glb=g, bin=bin_file,
            init=j_init, camera=j_cam, view=j_view, outdir=work_dir)

    def mk_decode(b):
        return [cfg["python_bin"], DECODER] + build_args(
            "DECODE_ARGS", DECODE_ARGS_DEFAULT, bin=b, ply=out_ply, outdir=work_dir)

    demux_cmd, unpack_cmd, decode_cmd = mk_demux(), mk_unpack(glb), mk_decode(bin_file)
    print("[Step 1/3] 🎬 解复用 MP4 → GLB")
    print(f"  $ {' '.join(str(c) for c in demux_cmd)}")
    print("[Step 2/3] 📦 解包 GLB → 码流 + 三件套 json")
    print(f"  $ {' '.join(str(c) for c in unpack_cmd)}")
    print("[Step 3/3] 📦 解码码流 → PLY")
    print(f"  $ {' '.join(str(c) for c in decode_cmd)}")

    if cfg["dry_run"]:
        print("⏭️ DRY_RUN=1，以上命令未执行")
        return "ok"

    t0 = time.time()

    # --- Step 1/3: MP4 → GLB（demuxer 只收 --output_dir，产出名由它自己定）---
    t_step = time.time()
    rc = run_cmd(demux_cmd, cwd=cfg["tool_dir"], env=env)
    fresh = pick_glb(work_dir, t_step - 1.0)              # -1s 容错文件系统时间精度
    if fresh is not None:
        if fresh != glb:
            print(f"ℹ️ 未得到 {glb.name}，改用实际产物 {fresh.name}")
        glb = fresh
    elif glb.is_file() and rc == 0:
        print(f"⚠️ 没检测到新写出的 GLB，沿用已存在的 {glb.name}")
    else:
        print(f"❌ Step 1 失败：{work_dir} 下没得到 GLB（rc={rc}）")
        print("   若参数名不符，先 PROBE=1 查真名，再用 DEMUX_ARGS 覆盖")
        return "failed"
    print(f"  ✅ GLB: {glb.name}  ({glb.stat().st_size / 1024 / 1024:.1f} MB)")

    # --- Step 2/3: GLB → bin + 三件套 json ---
    unpack_cmd = mk_unpack(glb)          # GLB 名可能刚被改写，重建命令
    t_step = time.time()
    rc = run_cmd(unpack_cmd, cwd=cfg["tool_dir"], env=env)
    if not bin_file.is_file():
        # 块大小由工具内部决定，文件名对不上时按本次新产出的码流兜底
        alt = find_fresh(work_dir, "GSCompressed_B*.bin", t_step - 1.0) \
            or find_fresh(work_dir, "*.bin", t_step - 1.0)
        if alt:
            print(f"⚠️ 期望 {bin_file.name} 不存在，改用实际产物 {alt.name}")
            bin_file = alt
        else:
            print(f"❌ Step 2 失败：{work_dir} 下没有码流 (*.bin)（rc={rc}）")
            print("   gltf_unpacker 的参数形态未实测，可试 UNPACK_ARGS='{glb} {outdir}'")
            return "failed"
    if rc != 0:
        print("⚠️ gltf_unpacker 返回非 0，但码流已产出，继续")
    print(f"  ✅ 码流: {bin_file.name}  ({bin_file.stat().st_size / 1024 / 1024:.1f} MB)")

    js = find_jsons(work_dir, cfg["tool_dir"])
    if len(js) == len(UWA_JSONS):
        print("  ✅ 三件套 json: " + ", ".join(p.name for p in js.values()))
    elif js:
        print(f"  ⚠️ 三件套 {len(js)}/3: "
              + ", ".join(f"{k}={p.name}" for k, p in js.items()))
        missing = [k for k in UWA_JSONS if k not in js]
        print(f"     缺: {', '.join(missing)}（可能是该维本来就没进 GLB）")
    else:
        others = sorted(p.name for p in work_dir.glob("*.json"))
        if others:
            print(f"  ⚠️ 三件套 0/3，但 work 下有 json: {others}")
            print("     像是命名不匹配，把实际名字加进 UWA_JSONS 的候选元组即可")
        else:
            print("  ℹ️ 三件套 0/3（work 下无 json）——旧版 gltf_packer 只写 6 项白名单"
                  "（NOTES 第 9 条），属预期；打过补丁的 cgltf 应能吐出三件套")

    # --- Step 3/3: bin → PLY（decode 输出名未知，按本次新产出识别）---
    t_step = time.time()
    rc = run_cmd(mk_decode(bin_file), cwd=cfg["tool_dir"], env=env)
    produced = find_fresh(work_dir, "*.ply", t_step - 1.0)
    if produced is None and out_ply.is_file():
        produced = out_ply      # decode 直接写到了目标路径（DECODE_ARGS 给了 {ply}）
    if produced is None or not produced.is_file():
        print(f"❌ Step 3 失败：没找到解出的 PLY（既不在 {out_ply}，也不在 {work_dir}/*.ply）")
        print(f"   decode.py 实测产出名为 {DECODE_PLY_NAME}；若它写到了别处，用 DECODE_ARGS 覆盖")
        print("   若日志里有 FileNotFoundError: .../image0.bmp → astcenc 缺执行位"
              "（chmod +x <astcenc> 或 ASTCENC_AUTOFIX=1）")
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

    # decode.py 的隐式依赖：astcenc 缺执行位会静默失败，必须前置拦（见 check_astcenc）
    bad_astc = check_astcenc(TOOL_DIR)
    if bad_astc:
        sys.exit("❌ astcenc 存在但没有可执行位，decode.py 会静默失败"
                 "（伪装成 FileNotFoundError: image0.bmp）。修复：\n   "
                 + "\n   ".join(f"chmod +x {p}" for p in bad_astc)
                 + "\n   或加 ASTCENC_AUTOFIX=1 自动补")

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
