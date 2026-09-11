#!/usr/bin/env python3
"""99d_copy_three_frames.py — 从数据集每个 task 里抽 3 帧（首 / 中 / 尾），
复制并按 task 名重命名，集中拉平到一个输出目录。

环境准备
    conda activate xcodec   （本脚本只用标准库，任意 python3 都能跑）

典型场景：B003_Human_Data_w_pose 下每个子目录是一个 task（人），帧放在
<task>/image/ 下、按帧号命名（01000000.jpg 这种）。做并排对比图 / 人工巡检时
每人只要 3 帧，本脚本自动取该 task 的**第一帧 / 中间一帧 / 最后一帧**，
拉平成 <task>_<帧号>.jpg，集中放到一个目录，方便一眼扫完。

帧列表按帧号数值排序后取首/中/尾，所以各 task 帧数不同也没关系；
帧数不足 3（重合）时按实际去重后输出，并打印提醒。

用法:
    python vggt_human/99d_copy_three_frames.py
    DRY_RUN=1 python vggt_human/99d_copy_three_frames.py       # 预览，不实际复制
    ONLY=a1b2c3,d4e5f6 python ...                              # 只处理指定 task
    SUFFIX_STYLE=seq python ...                                # 用 1/2/3 序号代替原帧号
    FRAMES=01000000.jpg,01000100.jpg python ...                # 改回指定帧号（覆盖自动抽取）

Env vars（不设则用下方 main() 里的默认值）:
    SRC_ROOT      批次根目录（其下每个子目录是一个 task）
    RESULTS_ROOT  统一结果根（与 99a/99b 同源）
    DST_ROOT      输出目录，默认 <RESULTS_ROOT>/img_three
    FRAMES        可选：逗号分隔的帧文件名，显式指定要抽哪几帧
                  （留空 = 默认行为，自动取首 / 中 / 尾）
    IMAGE_SUBDIR  task 下放帧的子目录名（默认 image）
    SUFFIX_STYLE  orig（默认，保留原帧号 → <task>_01000000.jpg）
                  seq （按抽取顺序编号 → <task>_1.jpg / <task>_2.jpg / <task>_3.jpg）
    NAME_SEP      连接符（默认下划线 _）
    ONLY          逗号分隔的 task 白名单
    DRY_RUN=1     只打印不复制
"""
import os
import re
import shutil
import sys
from pathlib import Path

IMAGE_EXTS = ("jpg", "jpeg", "png")
# 帧文件名形如 01000000.jpg；用数值排序避免位宽不一致时字符串序出错
NUM_FRAME_RE = re.compile(r"^(\d+)\.([A-Za-z]+)$")


def dst_name(task_name: str, src_name: str, idx: int, sep: str, style: str) -> str:
    """拼目标文件名：<task 名><sep><序号后缀><扩展名>。

    orig: 序号后缀 = 原帧号（如 01000050）—— 默认。无损：事后能从文件名反查源帧，
          且不依赖抽取顺序。
    seq : 序号后缀 = 抽取顺序（从 1 起，即 1=首帧 2=中帧 3=尾帧）。
    """
    ext = Path(src_name).suffix
    tag = Path(src_name).stem if style == "orig" else str(idx + 1)
    return f"{task_name}{sep}{tag}{ext}"


def list_frames(img_dir: Path) -> list:
    """列出 img_dir 下的帧文件名，按时间（帧号）顺序返回。

    优先按纯数字帧号数值排序；目录里没有任何数字命名文件时才退回字典序
    （两种情况混在一起时以数字命名为准，并提示忽略了几个非数字名）。
    """
    num, other = [], []
    for f in img_dir.iterdir():
        if not f.is_file():
            continue
        m = NUM_FRAME_RE.match(f.name)
        if m and m.group(2).lower() in IMAGE_EXTS:
            num.append((int(m.group(1)), f.name))
        elif f.suffix.lower().lstrip(".") in IMAGE_EXTS:
            other.append(f.name)

    if num:
        if other:
            print(f"   ℹ️ {img_dir.parent.name}/{img_dir.name}: 忽略 {len(other)} 个非数字命名文件")
        return [name for _, name in sorted(num)]
    return sorted(other)


def pick_three(frames: list) -> list:
    """从有序帧列表取首 / 中 / 尾，返回 [(标签, 帧文件名), ...]（已去重）。

    中间一帧取 0-based 的 n//2：奇数帧数时是正中间，偶数时偏后半。
    帧数 <3 时首/中/尾会重合，去重后按首→中→尾的顺序返回更少的项。
    """
    n = len(frames)
    if n == 0:
        return []
    picks = [("first", frames[0]), ("mid", frames[n // 2]), ("last", frames[-1])]
    seen, out = set(), []
    for tag, name in picks:
        if name in seen:
            continue
        seen.add(name)
        out.append((tag, name))
    return out


def resolve_frames(img_dir: Path, frames_override: list):
    """决定一个 task 要抽哪些帧。

    frames_override 非空 → 按给定文件名抽（不存在的跳过）；
    否则自动取该 task 的首 / 中 / 尾。

    Returns:
        (picks, warn)：picks = [(标签, 帧文件名), ...]（空 = 无可用帧）；
        warn = 需要提醒的问题描述（无问题则空串）。
    """
    if frames_override:
        miss = [n for n in frames_override if not (img_dir / n).is_file()]
        picks = [(f"#{i + 1}", n) for i, n in enumerate(frames_override)
                 if (img_dir / n).is_file()]
        warn = f"缺 {len(miss)}/{len(frames_override)} 帧: {', '.join(miss)}" if miss else ""
        return picks, warn

    all_frames = list_frames(img_dir)
    picks = pick_three(all_frames)
    if not picks:
        return [], ""
    # 帧数不足 3 时首/中/尾重合，输出会比预期少几条，明确提示
    warn = f"仅 {len(all_frames)} 帧，首/中/尾重合" if len(picks) < 3 else ""
    return picks, warn


def collect_frames(src_root: Path, dst_root: Path, frames: list, image_subdir: str,
                   sep: str, style: str, only: list, dry_run: bool = False) -> None:
    """遍历 src_root 下每个 task 子目录，抽首/中/尾帧复制到 dst_root。

    Args:
        src_root: 批次根目录（其下每个子目录是一个 task）。
        dst_root: 输出目录（自动创建）。
        frames: 显式指定的帧文件名列表；空列表 = 自动取首/中/尾。
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
    print("🖼️ 抽取帧:   " + ("首 / 中 / 尾（自动）" if not frames else ", ".join(frames)))
    print(f"🏷️ 命名规则: " + (
        f"<task>{sep}<原帧号>" if style == "orig" else f"<task>{sep}<1..N>"))
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
    # 记下每个 task 的问题（缺帧 / 无帧目录），最后汇总打印，避免刷屏时被淹没
    missing_log = []
    for sub in subdirs:
        img_dir = sub / image_subdir
        if not img_dir.is_dir():
            print(f"⏭️  skip  {sub.name}  (无 {image_subdir}/)")
            missing_log.append(f"{sub.name} (无 {image_subdir}/)")
            skip += 1
            continue

        picks, warn = resolve_frames(img_dir, frames)
        if warn:
            print(f"⚠️  {sub.name}  {warn}")
            missing_log.append(f"{sub.name} ({warn})")
        if not picks:
            print(f"⏭️  skip  {sub.name}  ({image_subdir}/ 下没有可用帧)")
            if not warn:
                missing_log.append(f"{sub.name} (无可用帧)")
            skip += 1
            continue

        hit = 0
        for idx, (tag, name) in enumerate(picks):
            src = img_dir / name
            out = dst_root / dst_name(sub.name, name, idx, sep, style)

            if dry_run:
                print(f"📋 would  {sub.name}  {tag:<5}  {image_subdir}/{name}  ->  {out.name}")
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
            # 一行汇总该 task 实拷了哪几帧，逐帧行太多时便于回看
            detail = " ".join(f"{tag}={name}" for tag, name in picks)
            print(f"✅ {sub.name}  ({hit} 帧: {detail})")

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
    # 显式指定帧名（留空 = 自动取首 / 中 / 尾）
    FRAMES = [s for s in os.environ.get("FRAMES", "").split(",") if s]
    # task 下放帧的子目录名
    IMAGE_SUBDIR = os.environ.get("IMAGE_SUBDIR", "image")
    # 命名风格 / 连接符
    SUFFIX_STYLE = os.environ.get("SUFFIX_STYLE", "orig").lower()
    NAME_SEP = os.environ.get("NAME_SEP", "_")
    # ==============================================

    if SUFFIX_STYLE not in ("orig", "seq"):
        sys.exit(f"❌ SUFFIX_STYLE 只能是 orig/seq，当前: {SUFFIX_STYLE}")

    only = [s for s in os.environ.get("ONLY", "").split(",") if s]
    dry_run = os.environ.get("DRY_RUN", "0") == "1"

    collect_frames(SRC_ROOT, DST_ROOT, FRAMES, IMAGE_SUBDIR, NAME_SEP,
                   SUFFIX_STYLE, only, dry_run=dry_run)


if __name__ == "__main__":
    main()
