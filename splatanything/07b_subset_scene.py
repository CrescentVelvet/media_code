#!/usr/bin/env python3
"""07b_subset_scene.py — 从已建好的场景目录抽 N 个视角，生成子场景（3v/8v/13v 控制面板用）。

为什么抽子集而不是重新跑 07：07 的产物（去畸变后的 PNG + prior_camera.json）就是推理真正
吃进去的东西，抽子集是**精确复制**，不会再过一次去畸变/缩放，前后对比只差"视角数"这一个变量。

选视角策略：相机中心最远点采样（greedy max-min 夹角）。默认"等间隔抽帧"在环绕轨迹上会
抽到相邻视角扎堆，3v 时基线几乎为 0，推理退化成单目 —— 这正是要避开的。

Args:
  --scene   源场景目录（含 images/ 和 prior_camera.json）
  --n       抽几个视角（默认 3）
  --out     输出场景目录（默认 源目录_<n>v）
  --idxs    显式指定源视角序号（逗号分隔），优先级高于 --n
  --hardlink  默认 1：图片用硬链接省空间（同盘才行），0 = 复制
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np


def load_cams(scene):
    """读相机；只保留**图片真实存在**的 id。

    官方 `examples/object/prior_camera.json` 有 24 条 extrinsics 但只有 23 张图
    （`frame_0023.png` 缺失），直接用相机 id 会抽到不存在的帧。
    """
    data = json.loads((scene / 'prior_camera.json').read_text())
    extr = {e['camera_id']: np.asarray(e['matrix'], np.float64) for e in data['extrinsics']}
    intr = {e['camera_id']: np.asarray(e['matrix'], np.float64) for e in data['intrinsics']}
    ids = [e['camera_id'] for e in data['extrinsics'] if e['camera_id'] in intr]
    missing = [i for i in ids if not (scene / 'images' / f'{i}.png').exists()]
    if missing:
        print(f'⚠️  {len(missing)} 个相机没有对应图片，已剔除：{missing[:4]}'
              f'{" …" if len(missing) > 4 else ""}')
        ids = [i for i in ids if i not in set(missing)]
    if not ids:
        raise SystemExit(f'❌ {scene} 的相机与 images/ 完全对不上')
    centers = np.stack([extr[i][:3, 3] for i in ids])
    return data, ids, centers, extr, intr


def max_min_pick(centers, n):
    """最远点采样：先取离质心最远的点，之后每次取到已选集合最小夹角最大的点。"""
    if n >= len(centers):
        return list(range(len(centers)))
    centroid = centers.mean(0)
    dirs = centers - centroid
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
    picked = [int(np.argmax(np.linalg.norm(centers - centroid, axis=1)))]
    while len(picked) < n:
        sim = dirs @ dirs[picked].T                      # [N, k] cos
        worst = np.clip(sim.max(axis=1), -1, 1)           # 与已选最近的那个
        worst[picked] = 1.0                               # 已选的排除
        picked.append(int(np.argmin(worst)))              # 夹角最大 = cos 最小
    return sorted(picked)


def angles_deg(centers, sel):
    centroid = centers.mean(0)
    d = centers[sel] - centroid
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    cos = np.clip(d @ d.T, -1, 1)
    iu = np.triu_indices(len(sel), 1)
    ang = np.degrees(np.arccos(cos[iu]))
    return dict(min=float(ang.min()), median=float(np.median(ang)), max=float(ang.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True)
    ap.add_argument('--n', type=int, default=3)
    ap.add_argument('--out', default=None)
    ap.add_argument('--idxs', default=None)
    ap.add_argument('--hardlink', type=int, default=1)
    args = ap.parse_args()

    src = Path(args.scene).expanduser().resolve()
    if not (src / 'images').is_dir() or not (src / 'prior_camera.json').is_file():
        raise SystemExit(f'❌ {src} 缺少 images/ 或 prior_camera.json')
    out = Path(args.out).expanduser().resolve() if args.out else src.parent / f'{src.name}_{args.n}v'
    if out.exists():
        raise SystemExit(f'❌ {out} 已存在（先删或换 --out）')

    data, ids, centers, extr, intr = load_cams(src)
    if args.idxs:
        sel = [int(i) for i in args.idxs.split(',')]
        if max(sel) >= len(ids):
            raise SystemExit(f'❌ --idxs 越界（共 {len(ids)} 个视角）')
        sel = sorted(sel)
    else:
        sel = max_min_pick(centers, args.n)

    (out / 'images').mkdir(parents=True)
    for i in sel:
        name = f'{ids[i]}.png'
        s, d = src / 'images' / name, out / 'images' / name
        if args.hardlink:
            try:
                os.link(s, d)
                continue
            except OSError:
                pass
        shutil.copy2(s, d)

    sub = {
        'extrinsics': [{'camera_id': ids[i], 'matrix': extr[ids[i]].tolist()} for i in sel],
        'intrinsics': [{'camera_id': ids[i], 'matrix': intr[ids[i]].tolist()} for i in sel],
    }
    (out / 'prior_camera.json').write_text(json.dumps(sub, indent=2) + '\n')

    ang = angles_deg(centers, sel)
    meta = {
        'source': str(src), 'n_source_views': len(ids), 'n_views': len(sel),
        'picked': [int(i) for i in sel],
        'picked_ids': [ids[i] for i in sel],
        'held_out': [int(i) for i in range(len(ids)) if i not in sel],
        'pairwise_deg': ang,
    }
    (out / 'meta.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False) + '\n')

    print(f'✅ {len(sel)}/{len(ids)} 视角 → {out}')
    print(f'   源序号 {sel}')
    print(f'   两两夹角 min {ang["min"]:.1f}° / median {ang["median"]:.1f}° / max {ang["max"]:.1f}°')
    print(f'   留出 {len(meta["held_out"])} 个视角（WB-2 用）')


if __name__ == '__main__':
    main()
