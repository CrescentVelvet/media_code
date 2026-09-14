"""_debug_temporal_check.py — 量化 09 渲染后增强的「帧间闪烁」。

思路：逐帧独立 2D 增强不保证时序一致 → 相邻帧的差异可能被放大。
指标（都在人脸框内统计，避免背景干扰）：
  D_raw(k) = mean|raw[k] - raw[k-1]|     原渲染的帧间差
  D_enh(k) = mean|enh[k] - enh[k-1]|     增强后的帧间差
  ratio    = D_enh / D_raw                >1 说明增强放大了时间抖动
另输出并排对比视频（raw | enhanced）供目检。

Env: RESULTS_DIR / MATCH_JSON / PID / FPS / OUT_VIDEO
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

R = Path(os.environ["RESULTS_DIR"])
match_json = Path(os.environ.get(
    "MATCH_JSON", R / "03_match/face_person_match.json"))
PID = os.environ.get("PID", "0")
FPS = int(os.environ.get("FPS", "12"))
raw_dir = R / "09_render/raw"
enh_dir = R / "09_render/images"
out_video = Path(os.environ.get("OUT_VIDEO", R / "09_render/cmp_raw_vs_enh.gif"))


def main():
    match = json.loads(match_json.read_text())["frames"]
    files = sorted(raw_dir.glob("*.png"))
    print(f"📼 {len(files)} 帧")

    raws, enhs, boxes = [], [], []
    for p in files:
        ep = enh_dir / p.name
        if not ep.exists():
            continue
        stem = p.stem
        key = stem.split("_", 1)[1] if "_" in stem and \
            stem.split("_", 1)[0].isdigit() else stem
        rec = match.get(key, {}).get("persons", {}).get(PID)
        if not rec:
            continue
        a = np.asarray(Image.open(p).convert("RGB"), np.float32)
        b = np.asarray(Image.open(ep).convert("RGB"), np.float32)
        raws.append(a)
        enhs.append(b)
        boxes.append(rec["bbox"])
    n = len(raws)
    print(f"📊 有效 {n} 帧")

    def crop(arr, bb, pad=0.15):
        H, W = arr.shape[:2]
        x1, y1, x2, y2 = bb
        bw, bh = x2 - x1, y2 - y1
        x1 = max(0, int(x1 - bw * pad)); y1 = max(0, int(y1 - bh * pad))
        x2 = min(W, int(x2 + bw * pad)); y2 = min(H, int(y2 + bh * pad))
        return arr[y1:y2, x1:x2]

    d_raw, d_enh = [], []
    for k in range(1, n):
        # 每帧用自己的脸框裁（脸在动，用各自框保证裁的是同一部位）
        ca0, ca1 = crop(raws[k - 1], boxes[k - 1]), crop(raws[k], boxes[k])
        cb0, cb1 = crop(enhs[k - 1], boxes[k - 1]), crop(enhs[k], boxes[k])
        h = min(ca0.shape[0], ca1.shape[0], cb0.shape[0], cb1.shape[0])
        w = min(ca0.shape[1], ca1.shape[1], cb0.shape[1], cb1.shape[1])
        ca0, ca1 = ca0[:h, :w], ca1[:h, :w]
        cb0, cb1 = cb0[:h, :w], cb1[:h, :w]
        d_raw.append(np.abs(ca1 - ca0).mean())
        d_enh.append(np.abs(cb1 - cb0).mean())

    d_raw = np.array(d_raw); d_enh = np.array(d_enh)
    ratio = d_enh / np.maximum(d_raw, 1e-6)
    print(f"\n帧间差（人脸框内, 0-255 尺度）:")
    print(f"  raw   p50={np.median(d_raw):6.2f}  mean={d_raw.mean():6.2f}")
    print(f"  enh   p50={np.median(d_enh):6.2f}  mean={d_enh.mean():6.2f}")
    print(f"  ratio p50={np.median(ratio):6.3f}  mean={ratio.mean():6.3f}  "
          f"p90={np.percentile(ratio,90):6.3f}  max={ratio.max():6.3f}")
    worse = int((ratio > 1.5).sum())
    print(f"  ratio>1.5 的相邻帧对: {worse}/{len(ratio)}")

    # 并排视频（每帧 raw | enhanced，缩到 1/2 便于查看）
    try:
        import imageio.v2 as imageio
        half = lambda a: np.asarray(
            Image.fromarray(a.astype(np.uint8)).resize(
                (a.shape[1] // 2, a.shape[0] // 2), Image.LANCZOS))
        frames = [np.concatenate([half(raws[i]), half(enhs[i])], axis=1)
                  for i in range(min(n, 200))]
        dur = 1.0 / max(FPS, 1)
        if out_video.suffix.lower() == ".gif":
            imageio.mimsave(str(out_video), frames, duration=dur, loop=0)
        else:
            imageio.mimsave(str(out_video), frames, fps=FPS)
        print(f"\n🎬 并排视频 → {out_video}")
    except Exception as e:
        print(f"⚠️ 视频生成失败: {e}")


if __name__ == "__main__":
    main()
