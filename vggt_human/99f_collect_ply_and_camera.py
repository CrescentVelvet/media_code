#!/usr/bin/env python3
"""99f_collect_ply_and_camera.py — 批量收集各 task 子目录下的
point_cloud_final.ply + gs_camera_params_final.json，成对复制并改名为
point_cloud.ply / gs_camera_params.json，按 task 名建子目录集中放到一个输出目录。

环境准备
    纯标准库，任意 python3 即可（无需 conda）

与 99a 的区别：99a 把 ply **压平**成 <批次>/ply/<task_id>.ply（适合单独看/上传），
本脚本保留 task 目录层级、并把同 task 的相机参数一并带上，产出「一目录一 task」
的成对形态，供下游按目录整体加载：

    <RESULTS_ROOT>/<批次名>/ply_viewlimit/<task_id>/point_cloud.ply
    <RESULTS_ROOT>/<批次名>/ply_viewlimit/<task_id>/gs_camera_params.json

输入输出的预设路径与 99a 保持一致（改批次时改 main() 里那几行即可）。

用法:
    python vggt_human/99f_collect_ply_and_camera.py
    DRY_RUN=1 python ...            # 预览，不实际复制
    ONLY=task_a,task_b python ...   # 只处理白名单里的 task

Env vars（不设则用下方 main() 里的默认值）:
    SRC_ROOT      批次根目录（其下每个子目录是一个 task）
    RESULTS_ROOT  统一结果根（默认 ../../output/recon_human_results）
    OUT_DIR       输出目录，默认 <RESULTS_ROOT>/<批次名>/ply_viewlimit
    PLY_NAME      源 ply 文件名（默认 point_cloud_final.ply）
    GS_NAME       源相机 json 文件名（默认 gs_camera_params_final.json）
    PLY_OUT_NAME  复制后的 ply 名（默认 point_cloud.ply）
    GS_OUT_NAME   复制后的 json 名（默认 gs_camera_params.json）
    ONLY          逗号分隔的 task 白名单（不设=全部）
    DRY_RUN=1     只打印不复制
"""
import os
import shutil
import sys
from pathlib import Path


def _fmt_size(nbytes: int) -> str:
    """字节数 → 人类可读（MB 保留 1 位小数，小于 1MB 显示 KB）。"""
    if nbytes >= 1024 * 1024:
        return f"{nbytes / 1024 / 1024:.1f} MB"
    return f"{nbytes / 1024:.1f} KB"


def collect_task_pairs(
    src_root: Path,
    dst_root: Path,
    src_names: list,
    dst_names: list,
    only=None,
    dry_run: bool = False,
) -> dict:
    """遍历 src_root 下每个子目录（task），把其中的 src_names 成对复制到
    dst_root/<task>/ 下并改名为 dst_names。

    同一 task 的多个文件**要么全成功要么不复制**：先整体检查是否齐全，
    缺任一文件就跳过该 task（避免下游拿到 ply 却没有相机参数的半成品）。

    Args:
        src_root:  批次根目录（其下每个子目录是一个 task）。
        dst_root:  输出目录（自动创建）。
        src_names: 要收集的源文件名列表。
        dst_names: 复制后的目标文件名列表（与 src_names 一一对应）。
        only:      task 名白名单（None = 全部）。
        dry_run:   True 只打印不复制。

    Returns:
        {"ok": n, "skip": n, "fail": n, "tasks": [...]}
    """
    if len(src_names) != len(dst_names):
        sys.exit("❌ src_names 与 dst_names 长度不一致（内部错误）")
    if not src_root.is_dir():
        sys.exit(f"❌ 源目录不存在: {src_root}")

    only_set = set(only) if only else None

    if dry_run:
        print(f"📁 输出目录(预览): {dst_root}")
    else:
        dst_root.mkdir(parents=True, exist_ok=True)
        print(f"📁 输出目录: {dst_root}")
    print(f"🔍 源目录: {src_root}")
    for s, d in zip(src_names, dst_names):
        print(f"🖼️ 收集: {s}  ->  <task_id>/{d}")
    if only_set:
        print(f"🎯 ONLY: {', '.join(sorted(only_set))}")
    if dry_run:
        print("⏭️  DRY_RUN=1（预览，不实际复制）")
    print()

    # 遍历每个 task 子目录（仅一层，与 99a 一致）
    subdirs = sorted(
        [d for d in src_root.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )
    if only_set is not None:
        subdirs = [d for d in subdirs if d.name in only_set]
    if not subdirs:
        print("⚠️ 源目录下没有可处理的子目录")
        return {"ok": 0, "skip": 0, "fail": 0, "tasks": []}

    ok = skip = fail = 0
    done_tasks = []
    for sub in subdirs:
        missing = [n for n in src_names if not (sub / n).exists()]
        if missing:
            print(f"⏭️  skip  {sub.name}  (缺 {' + '.join(missing)})")
            skip += 1
            continue

        if dry_run:
            print(f"📋 would  {sub.name}/  ->  {', '.join(dst_names)}")
            ok += 1
            continue

        dst_dir = dst_root / sub.name
        try:
            dst_dir.mkdir(parents=True, exist_ok=True)
            sizes = []
            for s, d in zip(src_names, dst_names):
                # copy2 保留元数据；同名已存在直接覆盖（保证最新，与 99a 一致）
                shutil.copy2(sub / s, dst_dir / d)
                sizes.append(_fmt_size((dst_dir / d).stat().st_size))
            print(f"✅ {sub.name}  ({', '.join(sizes)})")
            ok += 1
            done_tasks.append(sub.name)
        except Exception as e:
            print(f"❌ {sub.name}: {e}")
            fail += 1

    print()
    print(f"🎉 Done.  ✅ {ok}  ⏭️ {skip}  ❌ {fail}")
    if not dry_run:
        print(f"📁 结果: {dst_root}")
    return {"ok": ok, "skip": skip, "fail": fail, "tasks": done_tasks}


def main():
    # Windows 控制台默认 cp936，直接打印 emoji 会 UnicodeEncodeError；
    # WSL/Linux 下这个调用无害
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # ===== 预设路径与 99a 保持一致，改批次时只改这几行 =====
    # 源：批次根目录（其下每个子目录是一个 task）
    SRC_ROOT = Path(
        "../../code/Reconstruction/output/"
        "B003_Human_Data_w_pose-脸红优化+外插视角增强"
    )
    # 输出：会自动新建。批次名与源同名；ply_viewlimit/ 与 99a 的 ply/、99b 的 mp4/ 平级，
    # 同一批次目录下按产物形态分开，互不覆盖
    RESULTS_ROOT = Path("../../output/recon_human_results")
    DST_ROOT = RESULTS_ROOT / SRC_ROOT.resolve().name / "ply_viewlimit"
    # 要收集的文件（流水线产物固定名）及其改名后的名字
    SRC_NAMES = ["point_cloud_final.ply", "gs_camera_params_final.json"]
    DST_NAMES = ["point_cloud.ply", "gs_camera_params.json"]
    # ======================================================

    # env 覆盖（不设则用上面的预设）
    SRC_ROOT = Path(os.environ.get("SRC_ROOT", str(SRC_ROOT)))
    RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", str(RESULTS_ROOT)))
    DST_ROOT = Path(os.environ.get("OUT_DIR", str(DST_ROOT)))
    SRC_NAMES[0] = os.environ.get("PLY_NAME", SRC_NAMES[0])
    SRC_NAMES[1] = os.environ.get("GS_NAME", SRC_NAMES[1])
    DST_NAMES[0] = os.environ.get("PLY_OUT_NAME", DST_NAMES[0])
    DST_NAMES[1] = os.environ.get("GS_OUT_NAME", DST_NAMES[1])

    only = [s.strip() for s in os.environ.get("ONLY", "").split(",") if s.strip()]
    dry_run = os.environ.get("DRY_RUN", "0") == "1"

    collect_task_pairs(SRC_ROOT, DST_ROOT, SRC_NAMES, DST_NAMES, only=only, dry_run=dry_run)


if __name__ == "__main__":
    main()
