#!/usr/bin/env python3
# eval_compare.py — 单模型 test 视角评测 + GT|render 对比图（按运动量排序）。
#
# 源自 wave2p 的临时脚本，参数化后入库。配合 03_render.sh 使用：
# 03 产出 $MODEL_PATH/test/ours_<iter>/renders/*.png，本脚本对齐 GT（llffhold=8
# 的 test split），逐视角算全图/前景 PSNR，前景区域裁剪拼 GT|render 对比图。
#
# Env (all optional):
#   SCENE_NAME=human_seq
#   RESULTS_DIR=        # 与 01/02/03 同一个
#   SOURCE_PATH=        # 默认 $RESULTS_DIR/$SCENE_NAME/colmap_scene
#   MODEL_PATH=         # 默认 $RESULTS_DIR/$SCENE_NAME/model
#   OUT_DIR=            # 默认 /mnt/d/output/${SCENE_NAME}_compare
#   LLFFHOLD=8          # test split 间隔（须与训练 --eval 一致）
import math
import os
import re
import sys

import numpy as np
from PIL import Image

SCENE_NAME = os.environ.get("SCENE_NAME", "human_seq")
RESULTS_DIR = os.path.expanduser(os.environ.get(
    "RESULTS_DIR", "~/output/deformable_human_results"))
SOURCE_PATH = os.environ.get(
    "SOURCE_PATH", os.path.join(RESULTS_DIR, SCENE_NAME, "colmap_scene"))
MODEL_PATH = os.environ.get(
    "MODEL_PATH", os.path.join(RESULTS_DIR, SCENE_NAME, "model"))
OUT_DIR = os.environ.get(
    "OUT_DIR", f"/mnt/d/output/{SCENE_NAME}_compare")
LLFFHOLD = int(os.environ.get("LLFFHOLD", "8"))


def load_rgb(path, size=None):
    img = Image.open(path).convert("RGB")
    if size is not None and img.size != size:
        img = img.resize(size, Image.LANCZOS)
    return np.asarray(img, dtype=np.float64) / 255.0


def psnr(a, b, mask=None):
    diff = (a - b) ** 2
    if mask is not None:
        m = mask[..., None]
        mse = (diff * m).sum() / max(m.sum() * diff.shape[2], 1.0)
    else:
        mse = diff.mean()
    return float("inf") if mse <= 1e-12 else -10.0 * math.log10(mse)


def find_img(images_dir, stem):
    for ext in (".jpg", ".png"):
        p = os.path.join(images_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


def latest_ours_dir(model_path):
    test_dir = os.path.join(model_path, "test")
    cands = []
    for d in os.listdir(test_dir):
        m = re.fullmatch(r"ours_(-?\d+)", d)
        if m and os.path.isdir(os.path.join(test_dir, d, "renders")):
            cands.append((int(m.group(1)), d))
    if not cands:
        sys.exit(f"❌ ERROR: {test_dir} 下没有 ours_*/renders（03 没跑？）")
    # -1 是官方「最新」软链语义，实际目录名是具体数字；取最大 iteration
    return sorted(cands)[-1][1]


def main():
    images_dir = os.path.join(SOURCE_PATH, "images")
    masks_dir = os.path.join(SOURCE_PATH, "masks")
    ours = latest_ours_dir(MODEL_PATH)
    renders = os.path.join(MODEL_PATH, "test", ours, "renders")
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(images_dir)
                   if f.lower().endswith((".jpg", ".png")))
    test_stems = stems[::LLFFHOLD]
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"📂 scene={SCENE_NAME} renders={ours} test视角={len(test_stems)}")

    rows = []
    for i, stem in enumerate(test_stems):
        rpath = os.path.join(renders, f"{i:05d}.png")
        gpath = find_img(images_dir, stem)
        if not (os.path.exists(rpath) and gpath):
            print(f"[skip] {stem}")
            continue
        size = Image.open(rpath).size
        gt = load_rgb(gpath, size)
        r = load_rgb(rpath)
        mpath = os.path.join(masks_dir, stem + ".png")
        mask = None
        if os.path.exists(mpath):
            m = Image.open(mpath).convert("L").resize(size, Image.NEAREST)
            mask = np.asarray(m, dtype=np.float64) > 127
        rows.append(dict(i=i, stem=stem, gt=gt, r=r, mask=mask,
                         full=psnr(r, gt),
                         fg=psnr(r, gt, mask) if mask is not None else float("nan")))

    # 运动量排序：相邻 test GT 的 mean abs diff（test 每 LLFFHOLD 取 1，差分近似运动幅度）
    motion = {}
    for a, b in zip(rows[:-1], rows[1:]):
        d = float(np.abs(a["gt"] - b["gt"]).mean())
        motion[b["stem"]] = d
        motion[a["stem"]] = max(motion.get(a["stem"], 0.0), d)

    for row in rows:
        gt, r, mask = row["gt"], row["r"], row["mask"]
        if mask is not None and mask.any():
            ys, xs = np.where(mask)
            h, w = mask.shape
            y0, y1 = ys.min(), ys.max()
            x0, x1 = xs.min(), xs.max()
            py, px = int((y1 - y0) * 0.15), int((x1 - x0) * 0.15)
            y0, y1 = max(0, y0 - py), min(h, y1 + py)
            x0, x1 = max(0, x0 - px), min(w, x1 + px)
            strip = np.concatenate([gt[y0:y1, x0:x1], r[y0:y1, x0:x1]], axis=1)
        else:
            strip = np.concatenate([gt, r], axis=1)
        name = f"cmp_{row['stem']}.jpg"
        Image.fromarray((strip * 255).astype(np.uint8)).save(
            os.path.join(OUT_DIR, name), quality=92)
        row["cmp"] = name

    valid = [x for x in rows if not math.isnan(x["fg"])]
    print("=== 逐视角（按运动量降序）===")
    for row in sorted(rows, key=lambda x: -motion.get(x["stem"], 0)):
        print(f"{row['stem']}: full {row['full']:.2f} | fg {row['fg']:.2f} | "
              f"motion {motion.get(row['stem'], 0):.4f} | {row['cmp']}")
    if valid:
        print("\n=== 汇总（%d 个 test 视角）===" % len(valid))
        print(f"全图 PSNR : {np.mean([x['full'] for x in valid]):.3f}")
        print(f"前景 PSNR: {np.mean([x['fg'] for x in valid]):.3f}")
    print(f"\n对比图: {OUT_DIR}")


if __name__ == "__main__":
    main()
