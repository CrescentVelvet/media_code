#!/usr/bin/env python3
"""99a_collect_ply.py — 批量收集各 task 子目录下的 point_cloud_final.ply，
复制并重命名为「子目录名.ply」，集中放到一个输出目录。

典型场景：重建流水线跑完后，每个 task 产出各自的 point_cloud_final.ply，
散落在 <batch_dir>/<task_id>/point_cloud_final.ply。本脚本把它们归集到
<output_root>/<batch_name>/<task_id>.ply，方便统一查看/上传。

用法:
    python vggt_human/99a_collect_ply.py
    DRY_RUN=1 python vggt_human/99a_collect_ply.py    # 预览，不实际复制

路径都在下方 main() 的变量里写死，改其它批次时直接改这几行即可。
"""
import os
import shutil
import sys
from pathlib import Path


def collect_ply(src_root: Path, dst_root: Path, ply_name: str, dry_run: bool = False) -> None:
    """遍历 src_root 下每个子目录，把 <sub>/ply_name 复制到 dst_root/<sub>.ply。

    Args:
        src_root: 批次根目录（其下每个子目录是一个 task）。
        dst_root: 输出目录（自动创建）。
        ply_name: 要收集的 ply 文件名（如 point_cloud_final.ply）。
        dry_run: True 只打印不复制。
    """
    if not src_root.is_dir():
        sys.exit(f"❌ 源目录不存在: {src_root}")

    # 输出目录：自动创建（含父目录）。dry_run 下也检查源即可。
    if not dry_run:
        dst_root.mkdir(parents=True, exist_ok=True)
        print(f"📁 输出目录: {dst_root}")
    else:
        print(f"📁 输出目录(预览): {dst_root}")

    print(f"🔍 源目录: {src_root}")
    print(f"🖼️ 目标文件名: <task_id>.ply  (源文件: {ply_name})")
    if dry_run:
        print("⏭️  DRY_RUN=1（预览，不实际复制）")
    print()

    # 遍历每个 task 子目录（仅一层）
    subdirs = sorted(
        [d for d in src_root.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )
    if not subdirs:
        print("⚠️ 源目录下没有子目录")
        return

    ok = skip = fail = 0
    for sub in subdirs:
        src_ply = sub / ply_name
        dst_ply = dst_root / f"{sub.name}.ply"

        if not src_ply.exists():
            print(f"⏭️  skip  {sub.name}  (无 {ply_name})")
            skip += 1
            continue

        # 已存在则覆盖（保证最新）；dry_run 下只打印意图
        if dry_run:
            print(f"📋 would  {sub.name}  ->  {dst_ply.name}")
            ok += 1
            continue

        try:
            # copy2 保留元数据；同文件系统 copy 快，跨文件系统也安全
            shutil.copy2(src_ply, dst_ply)
            size_mb = dst_ply.stat().st_size / 1024 / 1024
            print(f"✅ {sub.name}  ({size_mb:.1f} MB)")
            ok += 1
        except Exception as e:
            print(f"❌ {sub.name}: {e}")
            fail += 1

    print()
    print(f"🎉 Done.  ✅ {ok}  ⏭️ {skip}  ❌ {fail}")
    if not dry_run:
        print(f"📁 结果: {dst_root}")


def main():
    # ===== 在这里直接改路径 =====
    # 源：批次根目录（其下每个子目录是一个 task，含 point_cloud_final.ply）
    SRC_ROOT = Path(
        "/data_3d/w00950754/code/Reconstruction/output/"
        "B003_Human_Data_w_pose-脸红优化+外插视角增强+互补双监督"
    )
    # 输出：会自动新建。批次名与源同名，放在统一的 recon_human_results 下
    DST_ROOT = Path(
        "/data_3d/w00950754/output/recon_human_results/"
        "B003_Human_Data_w_pose-脸红优化+外插视角增强+互补双监督"
    )
    # 要收集的 ply 文件名（流水线产物固定名）
    PLY_NAME = "point_cloud_final.ply"
    # ===========================

    dry_run = os.environ.get("DRY_RUN", "0") == "1"
    collect_ply(SRC_ROOT, DST_ROOT, PLY_NAME, dry_run=dry_run)


if __name__ == "__main__":
    main()
