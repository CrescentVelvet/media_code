#!/usr/bin/env python3
"""01h_build_person_scenes.py — 按人（SAM3 track pid）拆出独立训练场景。

前提：01g_person_masks.sh 已产出 `<scene>/persons/masks/`（SAM3 video 模式，
obj_id 跨帧一致）。本脚本为每个 pid 建一个**可独立训练的 COLMAP 场景**：

    <scene>/persons/p{pid}/
      images -> ../../images            （目录软链，不复制原图）
      sparse -> ../../sparse            （位姿/内参共用，不复制）
      masks/<stem>.png                  （该 pid 的软 alpha，缺失帧写全 0）
      person.json                       （pid / 帧数 / 覆盖率 / 缺失帧）

为什么不复制图：01c 的 images 本身已是软链，训练只读；160 帧 × N 人复制
既慢又占空间。软链目录对 os.listdir / PIL 读图完全透明。

为什么缺失帧写全 0 而不是不写：训练要区分「此人不在画面」与「mask 找不到」
——前者该帧 loss 应为 0（跳过），后者是配置错误应报警。写 0 图把前者变显式。

Env / CLI:
  --scene        COLMAP 场景目录（含 images/、sparse/、persons/masks/）
  --masks-dir    默认 <scene>/persons/masks
  --out-root     默认 <scene>/persons
  --min-frames   某人至少被跟踪到的帧数（默认 10，丢 SAM3 碎片 track）
  --min-coverage 平均占画面百分比下限（默认 0.5%，丢误检小块）
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
from PIL import Image


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--masks-dir", default=None)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--min-frames", type=int, default=10)
    ap.add_argument("--min-frame-ratio", type=float, default=0.5,
                    help="至少被跟踪到的帧占比（默认 0.5）。SAM3 text prompt 会把背景"
                         "物体误检成低覆盖 person（实测 25/27 个 track 面积 0.01~1.3%%、"
                         "与真人 IoU=0），必须靠帧覆盖率把真人（全程可见）筛出来。")
    ap.add_argument("--min-coverage", type=float, default=0.5)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    scene = os.path.abspath(args.scene)
    masks_src = args.masks_dir or os.path.join(scene, "persons", "masks")
    out_root = args.out_root or os.path.join(scene, "persons")

    mf_path = os.path.join(masks_src, "masks_manifest.json")
    if not os.path.isfile(mf_path):
        sys.exit("❌ 找不到 {}（先跑 01g_person_masks.sh）".format(mf_path))

    mf = load_manifest(mf_path)
    frames = mf.get("frames", {})
    if not frames:
        sys.exit("❌ manifest 里没有任何帧")

    images_dir = os.path.join(scene, "images")
    if not os.path.isdir(images_dir):
        sys.exit("❌ 找不到 {}".format(images_dir))
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(images_dir)
                   if os.path.splitext(f)[1].lower() in
                   {".png", ".jpg", ".jpeg", ".bmp", ".webp"})
    if not stems:
        sys.exit("❌ {} 里没有图片".format(images_dir))

    # ── 汇总每个 pid ────────────────────────────────────────────────────────
    per_pid = {}
    for stem in stems:
        for inst in frames.get(stem, []):
            pid = inst["pid"]
            d = per_pid.setdefault(pid, {"stems": [], "cov": []})
            d["stems"].append(stem)
            d["cov"].append(float(inst.get("coverage", 0.0)))

    print("📊 SAM3 原始 track: {} 个 pid, 共 {} 帧".format(len(per_pid), len(stems)))
    kept = []
    for pid in sorted(per_pid):
        d = per_pid[pid]
        n = len(d["stems"])
        cov = float(np.mean(d["cov"])) if d["cov"] else 0.0
        ratio = n / max(len(stems), 1)
        ok = (n >= args.min_frames and cov >= args.min_coverage
              and ratio >= args.min_frame_ratio)
        print("  p{:02d}: {:4d} 帧 ({:5.1%}), 平均占画面 {:5.2f}%  {}".format(
            pid, n, ratio, cov, "✅" if ok else "❌ 丢弃（覆盖率/面积不足）"))
        if ok:
            kept.append((pid, n, cov))
    if not kept:
        sys.exit("❌ 没有 pid 通过过滤（调小 --min-frames / --min-coverage）")

    # ── 逐人建场景 ──────────────────────────────────────────────────────────
    made = []
    for pid, n, cov in kept:
        dst = os.path.join(out_root, "p{:02d}".format(pid))
        if os.path.isdir(dst) and not args.force:
            print("⏭️ 已存在，跳过: {}".format(dst))
            made.append((pid, dst, n, cov, False))
            continue
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        os.makedirs(os.path.join(dst, "masks"), exist_ok=True)

        # images / sparse 目录软链（相对路径，整目录可搬）
        for name in ("images", "sparse"):
            link = os.path.join(dst, name)
            target = os.path.join("..", "..", name)
            os.symlink(target, link)

        # 逐帧软 alpha：有则复制，无则写全 0
        hw = None
        n_hit = 0
        for stem in stems:
            src = os.path.join(masks_src, "{}.p{:02d}.alpha.png".format(stem, pid))
            out = os.path.join(dst, "masks", "{}.png".format(stem))
            if os.path.isfile(src):
                shutil.copyfile(src, out)
                n_hit += 1
                if hw is None:
                    with Image.open(src) as im:
                        hw = im.size[::-1]
            else:
                if hw is None:      # 首帧就缺 → 从任意一张真实图片取尺寸兜底
                    for cand in stems:
                        cp = os.path.join(images_dir, cand)
                        for ext in ("", ".png", ".jpg", ".jpeg"):
                            if os.path.isfile(cp + ext):
                                with Image.open(cp + ext) as im:
                                    hw = im.size[::-1]
                                break
                        if hw is not None:
                            break
                if hw is None:
                    sys.exit("❌ 无法确定图像尺寸（images 目录不可读）")
                Image.fromarray(np.zeros(hw, dtype=np.uint8)).save(out)

        meta = {
            "pid": pid,
            "scene": scene,
            "n_images": len(stems),
            "n_frames_with_person": n_hit,
            "mean_coverage_pct": round(cov, 3),
            "mask_hw": list(hw),
            "masks_dir": masks_src,
            "manifest": mf_path,
        }
        with open(os.path.join(dst, "person.json"), "w", encoding="utf-8",
                  newline="\n") as f:
            json.dump(meta, f, indent=4, ensure_ascii=False)

        print("  ✅ p{:02d}: {} 帧有 mask / {} 帧总计 → {}".format(pid, n_hit, len(stems), dst))
        made.append((pid, dst, n_hit, cov, True))

    print("\n🎉 完成，{} 个场景:".format(len(made)))
    for pid, dst, n, cov, _ in made:
        print("  p{:02d}  {:4d} 帧  {:5.2f}%  {}".format(pid, n, cov, dst))
    print("\n下一步（每人一个模型，SCENE_NAME 用绝对路径避免歧义）:")
    for pid, dst, _, _, _ in made:
        print("  SCENE_NAME=hand_p{:02d} SOURCE_PATH={} MODEL_PATH=<out>/model_p{:02d} \\".format(pid, dst, pid))
        print("    bash som_human/02_train.sh")


if __name__ == "__main__":
    main()
