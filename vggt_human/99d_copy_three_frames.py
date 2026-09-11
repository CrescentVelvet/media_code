#!/usr/bin/env python3
"""99d_copy_three_frames.py — 从数据集每个 task 里抽 3 帧，复制并按 task 名重命名，
集中拉平到一个输出目录。

环境准备
    conda activate xcodec   （本脚本只用标准库，任意 python3 都能跑）

典型场景：B003_Human_Data_w_pose 下每个子目录是一个 task（人），帧放在
<task>/image/01000000.jpg（按帧号命名，01000000 起步）。做并排对比图/人工巡检时
只需要每人 3 帧（首帧 / 中间 / 靠后各一张），本脚本把它们拉平成
<task>_<帧号>.jpg，集中放到一个目录，方便一眼扫完。

用法:
    python vggt_human/99d_copy_three_frames.py
    DRY_RUN=1 python vggt_human/99d_copy_three_frames.py       # 预览，不实际复制
    ONLY=a1b2c3,d4e5f6 python ...                              # 只处理指定 task
    SUFFIX_STYLE=seq python ...                                # 用 1/2/3 序号代替原帧号

Env vars（不设则用下方 main() 里的默认值）:
    SRC_ROOT      批次根目录（其下每个子目录是一个 task）
    RESULTS_ROOT  统一结果根（与 99a/99b 同源）
    DST_ROOT      输出目录，默认 <RESULTS_ROOT>/img_three
    FRAMES        逗号分隔的帧文件名，默认 01000000.jpg,01000050.jpg,01000100.jpg
    IMAGE_SUBDIR  task 下放帧的子目录名（默认 image）
    SUFFIX_STYLE  orig（默认，保留原帧号 → <task>_01000000.jpg）
                  seq （按抽取顺序编号 → <task>_1.jpg / <task>_2.jpg / <task>_3.jpg）
    NAME_SEP      连接符（默认下划线 _）
    ONLY          逗号分隔的 task 白名单
    DRY_RUN=1     只打印不复制
"""
import os
import shutil
import sys
from pathlib import Path

# 默认抽的 3 帧：首帧 / 中间 / 靠后各一张（帧号即时间顺序，000000 起步）
DEFAULT_FRAMES = ["01000000.jpg", "01000050.jpg", "01000100.jpg"]


def dst_name(task_name: str, src_name: str, idx: int, sep: str, style: str) -> str:
    """拼目标文件名：<task 名><sep><序号后缀><扩展名>。

    orig: 序号后缀 = 原帧号（如 01000050）—— 默认。无损：事后能从文件名反查源帧，
          且不依赖 FRAMES 的书写顺序。
    seq : 序号后缀 = 抽取顺序（从 1 起）。纯粹按「第几张」编号，与源帧号无关。
    """
    ext = Path(src_name).suffix
    tag = Path(src_name).stem if style == "orig" else str(idx + 1)
    return f"{task_name}{sep}{tag}{ext}"


def collect_frames(src_root: Path, dst_root: Path, frames: list, image_subdir: str,
                   sep: str, style: str, only: list, dry_run: bool = False) -> None:
    """遍历 src_root 下每个 task 子目录，抽 frames 里的帧复制到 dst_root。

    Args:
        src_root: 批次根目录（其下每个子目录是一个 task）。
        dst_root: 输出目录（自动创建）。
        frames: 要抽的帧文件名列表（按给定顺序）。
        image_subdir: task 下放帧的子目录名（如 image）。
        sep: 目标文件名里 task 名与序号后缀之间的连接符。
        style: orig / seq，见 dst_name。
        only: task 白名单（空列表 = 全部）。
        dry_run: True 只打印不复制。
    """
    if not src_root.is_dir():
        sys.exit(f"❌ 源目录不存在: {src_root}")

    # 输出目录：自动创建（含父目录）
    if not dry_run:
        dst_root.mkdir(parents=True, exist_ok=True)
    print(f"📁 输出目录: {dst_root}")
    print(f"🔍 源目录:   {src_root}")
    print(f"🖼️ 抽取帧:   {', '.join(frames)}")
    print(f"🏷️ 命名规则: " + (
        f"<task>{sep}<原帧号>" if style == "orig" else f"<task>{sep}<1..{len(frames)}>"))
    if dry_run:
        print("⏭️  DRY_RUN=1（预览，不实际复制）")
    print()

    # 遍历每个 task 子目录（仅一层）
    subdirs = sorted([d for d in src_root.iterdir() if d.is_dir()], key=lambda p: p.name)
    if only:
        subdirs = [d for d in subdirs if d.name in only]
    if not subdirs:
        print("⚠️ 没有待处理的 task 目录")
        return

    ok = skip = fail = 0
    # 记下每个 task 缺哪几帧，最后汇总打印，避免刷屏时被淹没
    missing_log = []
    for sub in subdirs:
        img_dir = sub / image_subdir
        if not img_dir.is_dir():
            print(f"⏭️  skip  {sub.name}  (无 {image_subdir}/)")
            missing_log.append(f"{sub.name} (无 {image_subdir}/)")
            skip += 1
            continue

        miss = [n for n in frames if not (img_dir / n).is_file()]
        if miss:
            print(f"⚠️  {sub.name}  缺 {len(miss)}/{len(frames)} 帧: {', '.join(miss)}")
            missing_log.append(f"{sub.name} (缺 {', '.join(miss)})")

        hit = 0
        for idx, name in enumerate(frames):
            src = img_dir / name
            if not src.is_file():
                continue
            out = dst_root / dst_name(sub.name, name, idx, sep, style)

            if dry_run:
                print(f"📋 would  {sub.name}/{image_subdir}/{name}  ->  {out.name}")
                ok += 1
                continue

            try:
                # copy2 保留元数据（mtime 对看「数据多新」有用）
                shutil.copy2(src, out)
                hit += 1
                ok += 1
            except Exception as e:
                print(f"❌ {sub.name}/{name}: {e}")
                fail += 1
        if hit and not dry_run:
            # 一行汇总该 task 实拷了几张，逐帧行太多时便于回看
            print(f"✅ {sub.name}  ({hit}/{len(frames)} 帧)")

    print()
    print(f"🎉 Done.  ✅ {ok}  ⏭️ {skip}  ❌ {fail}")
    if missing_log:
        print(f"⚠️ 不完整: {'; '.join(missing_log)}")
    if not dry_run:
        print(f"📁 结果: {dst_root}")


def main():
    # ===== 在这里直接改路径（或用环境变量覆盖）=====
    # 源：批次根目录（其下每个子目录是一个 task，帧在 <task>/image/）
    #     与 99b_pack_ply_to_mp4.py 的 IMAGE_DIR 同一份数据
    SRC_ROOT = Path(os.environ.get(
        "SRC_ROOT",
        "../../code/Reconstruction/dataset/B003_Human_Data_w_pose"))
    # 统一结果根：与 99a/99b 一致，便于几种产物一起找
    RESULTS_ROOT = Path(os.environ.get(
        "RESULTS_ROOT", "../../output/recon_human_results"))
    # 输出目录：三个平铺的帧文件都落这里
    DST_ROOT = Path(os.environ.get(
        "DST_ROOT", str(RESULTS_ROOT / "img_three")))
    # 要抽的帧文件名
    FRAMES = [s for s in os.environ.get(
        "FRAMES", ",".join(DEFAULT_FRAMES)).split(",") if s]
    # task 下放帧的子目录名
    IMAGE_SUBDIR = os.environ.get("IMAGE_SUBDIR", "image")
    # 命名风格 / 连接符
    SUFFIX_STYLE = os.environ.get("SUFFIX_STYLE", "orig").lower()
    NAME_SEP = os.environ.get("NAME_SEP", "_")
    # ==============================================

    if not FRAMES:
        sys.exit("❌ FRAMES 为空")
    if SUFFIX_STYLE not in ("orig", "seq"):
        sys.exit(f"❌ SUFFIX_STYLE 只能是 orig/seq，当前: {SUFFIX_STYLE}")

    only = [s for s in os.environ.get("ONLY", "").split(",") if s]
    dry_run = os.environ.get("DRY_RUN", "0") == "1"

    collect_frames(SRC_ROOT, DST_ROOT, FRAMES, IMAGE_SUBDIR, NAME_SEP,
                   SUFFIX_STYLE, only, dry_run=dry_run)


if __name__ == "__main__":
    main()
