#!/usr/bin/env python3
"""99n_batch_traj_render.py — 批量场景运镜编排：发现场景 → 99l 生成六种运镜 → 99m 渲染 MP4。

输入：BATCH_ROOT 目录，遍历其「自身 + 一层子目录」，找出同时包含
      point_cloud/ 与 refined_camera/ 的场景目录。对每个场景：
      1. 99l 生成 <场景>/traj_camera/<效果名>/（默认 6 种运镜三件套）；
      2. 99m 把 <场景>/point_cloud/iteration_*/point_cloud.ply（取最大 iteration，
         没有迭代目录则退回 point_cloud/point_cloud.ply）逐条轨迹渲成
         <场景>/traj_videos/<效果名>.mp4。

99l / 99m 用子进程调用（env 传参），与单场景手工跑完全同一条代码路径——
99n 只做编排、进度汇报与失败汇总，本身零第三方依赖（纯标准库），
在渲染 env 里跑即可（99m 子进程仍需 GPU + gsplat）。

用法:
    python tools_99/99n_batch_traj_render.py                 # 预设 BATCH_ROOT 批量
    BATCH_ROOT=/path/to/batch MAX_SCENES=1 python ...        # 先小规模试跑
    EFFECTS=orbit_left,grammy python ...                     # 只跑部分运镜
    NO_GEN=1 python ...                                      # 跳过 99l（轨迹已生成）
    SKIP_EXISTING=1 python ...                               # 视频齐了的场景跳过渲染
    DRY_RUN=1 python ...                                     # 只列出将处理哪些场景
    STOP_ON_ERROR=1 python ...                               # 遇错即停（默认记录后跳过）

Env vars（不设则用下方 main() 预设）:
  编排  BATCH_ROOT, MAX_SCENES（0=不限）, STOP_ON_ERROR（=1 遇错即停）
  生成  NO_GEN（=1 不调 99l）, EFFECTS, N, CENTER, PITCH, ANGLE, PARAM, GROUP, INIT_INDEX
  渲染  SKIP_EXISTING（=1 六条视频都在则跳过渲染）；FPS / SSAA / SH_LAYOUT /
        MAX_FRAMES 等其余渲染参数直接透传给 99m 子进程（在环境里设了就生效）

退出码：全部场景成功 = 0；有场景失败 = 1；DRY_RUN = 0。
"""
import os
import subprocess
import sys
import time
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
GEN_SCRIPT = TOOLS_DIR / "99l_gen_colmap_traj.py"
RENDER_SCRIPT = TOOLS_DIR / "99m_render_trajectory.py"
EFFECTS_ALL = ("orbit_left", "orbit_right", "orbit_up",
               "dolly_out", "hitchcock", "grammy")
VIDEOS_DIR_NAME = "traj_videos"


def die(msg):
    sys.exit(f"❌ {msg}")


def fmt_dur(sec):
    """秒 → m:ss.s，供进度打印。"""
    m, s = divmod(float(sec), 60.0)
    return f"{int(m)}:{s:04.1f}"


def find_scenes(root):
    """BATCH_ROOT 自身 + 一层子目录中，同时含 point_cloud/ 与 refined_camera/ 的。"""
    root = Path(root)
    if not root.is_dir():
        die(f"BATCH_ROOT 不是目录: {root}")
    candidates = [root] + sorted(p for p in root.iterdir() if p.is_dir())
    scenes = [p for p in candidates
              if (p / "point_cloud").is_dir() and (p / "refined_camera").is_dir()]
    return scenes


def find_ply(scene):
    """<场景>/point_cloud/iteration_*/point_cloud.ply 取最大 iteration；
    没有迭代目录则退回 point_cloud/point_cloud.ply。找不到返回 None。"""
    pc = scene / "point_cloud"
    iters = []
    for p in pc.glob("iteration_*/point_cloud.ply"):
        try:
            iters.append((int(p.parent.name.split("_", 1)[1]), p))
        except (ValueError, IndexError):
            iters.append((-1, p))
    if iters:
        iters.sort()
        return iters[-1][1]
    direct = pc / "point_cloud.ply"
    return direct if direct.exists() else None


def run_step(cmd, extra_env, tag):
    """子进程跑 99l / 99m，输出原样透传。返回 (是否成功, 耗时秒)。"""
    env = dict(os.environ, PYTHONIOENCODING="utf-8", **extra_env)
    t0 = time.time()
    print(f"  ▶️ {tag}", flush=True)
    ret = subprocess.run(cmd, env=env).returncode
    return ret == 0, time.time() - t0


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # ===== 预设路径与参数，改批次时只改这几行 =====
    BATCH_ROOT = Path(
        "../../output/2025.11.27-HMSD-Lisidani_out/action_01/result_gs"
    )
    EFFECTS = list(EFFECTS_ALL)   # 每个场景跑哪些运镜（env EFFECTS 可覆盖）
    NO_GEN = False                # True 不调 99l（traj_camera/ 已有轨迹）
    SKIP_EXISTING = False         # True 时六条 MP4 都在的场景跳过渲染
    STOP_ON_ERROR = False         # True 遇错即停；默认记录后继续下一个场景
    MAX_SCENES = 0                # >0 只处理前 N 个场景（试跑用）
    # ======================================================

    def env(k, d):
        v = os.environ.get(k, "").strip()
        return d if v == "" else v

    root = Path(env("BATCH_ROOT", str(BATCH_ROOT)))
    effects = [e.strip() for e in env("EFFECTS", ",".join(EFFECTS)).split(",")
               if e.strip()]
    for e in effects:
        if e not in EFFECTS_ALL:
            die(f"未知效果 {e!r}，可选: {'/'.join(EFFECTS_ALL)}")
    no_gen = env("NO_GEN", "1" if NO_GEN else "0") == "1"
    skip_existing = env("SKIP_EXISTING", "1" if SKIP_EXISTING else "0") == "1"
    stop_on_error = env("STOP_ON_ERROR", "1" if STOP_ON_ERROR else "0") == "1"
    max_scenes = int(env("MAX_SCENES", str(MAX_SCENES)))
    dry_run = env("DRY_RUN", "0") == "1"

    scenes = find_scenes(root)
    if max_scenes > 0:
        scenes = scenes[:max_scenes]

    print(f"🚀 [99n] BATCH_ROOT={root}")
    print(f"   发现 {len(scenes)} 个场景（同时含 point_cloud/ 与 refined_camera/）")
    for scene in scenes:
        ply = find_ply(scene)
        print(f"   • {scene.name} → {ply if ply else '（无 ply）'}")
    if not scenes:
        print("🏁 没有可处理的场景")
        return
    if dry_run:
        print("⏭️ DRY_RUN=1，只列出不执行")
        return

    # 99l / 99m 的生成侧参数走 env 透传：在环境里设了 N/CENTER/... 就自动生效
    results = []   # [(场景名, 状态, 说明)]
    t_all = time.time()
    for i, scene in enumerate(scenes, 1):
        ply = find_ply(scene)
        print(f"\n{'=' * 60}\n📦 [{i}/{len(scenes)}] {scene.name}（累计 "
              f"{fmt_dur(time.time() - t_all)}）\n{'=' * 60}", flush=True)
        if ply is None:
            print(f"  ⏭️ 跳过：{scene}/point_cloud 下没有 point_cloud.ply")
            results.append((scene.name, "跳过", "无 PLY"))
            continue

        ok = True
        note = []
        if not no_gen:
            ok_gen, dt = run_step(
                [sys.executable, str(GEN_SCRIPT)],
                {"SCENE": str(scene)}, f"99l 生成 {len(effects)} 条轨迹")
            note.append(f"99l {'✅' if ok_gen else '❌'} {fmt_dur(dt)}")
            ok = ok_gen
        else:
            print("  ⏭️ NO_GEN=1，跳过 99l")

        traj_dirs = [scene / "traj_camera" / e for e in effects]
        missing = [d.name for d in traj_dirs if not (d / "images.txt").exists()]
        out_dir = scene / VIDEOS_DIR_NAME
        done_mp4 = [out_dir / f"{e}.mp4" for e in effects]
        if ok and missing:
            print(f"  ⏭️ 跳过渲染：轨迹缺失 {missing}（99l 失败或未跑）")
            note.append(f"轨迹缺失 {missing}")
            ok = False
        if ok and skip_existing and all(p.exists() for p in done_mp4):
            print(f"  ⏭️ SKIP_EXISTING：{len(done_mp4)} 条视频已齐")
            note.append("视频已存在")

        if ok and not (skip_existing and all(p.exists() for p in done_mp4)):
            ok_r, dt = run_step(
                [sys.executable, str(RENDER_SCRIPT)],
                {"PLY": str(ply),
                 "TRAJ_DIR": ",".join(str(d) for d in traj_dirs),
                 "OUT_DIR": str(out_dir)},
                f"99m 渲染 {len(traj_dirs)} 条 → {out_dir}")
            note.append(f"99m {'✅' if ok_r else '❌'} {fmt_dur(dt)}")
            ok = ok_r

        results.append((scene.name, "✅" if ok else "❌", "；".join(note)))
        if not ok and stop_on_error:
            print("🛑 STOP_ON_ERROR=1，终止批次")
            break

    n_ok = sum(1 for _, s, _ in results if s == "✅")
    print(f"\n{'=' * 60}\n🏁 批次完成：{n_ok}/{len(results)} 成功，"
          f"总耗时 {fmt_dur(time.time() - t_all)}\n{'=' * 60}")
    for name, status, note in results:
        print(f"  {status} {name}  {note}")
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
