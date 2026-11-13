#!/usr/bin/env python3
"""10b_verify_sheet.py — 指认包的独立体检（不看图，只量数）。

为什么需要：`10_nose_sheet.py` 的裁窗是"检测框 × 倍数"推出来的，构造本身不能证明框对了。
这里用**独立于构造**的量反向验证：

  V1 face_in_crop   把 GT 块再检测一次，能检出且框落在块内 → 窗里确实有脸。
  V2 skin_ratio     块内肤色像素占比。正脸应显著高于 0.15；接近 0 就是框空/框到背景。
  V3 skin_align     GT 与 RD 的肤色重心差。**这是最关键的一条**——两图同框才可比，
                    若重心差 > 0.18 说明裁窗错位，上下对比无效。
  V4 lap_var        GT 块的 Laplacian 方差。过低 = 纯色块（背景/黑屏），不是人脸。

三分类（供人工指认时区分）：
  face     检测成功 → 正常可指认鼻宽
  profile  检测失败但肤色占比高 → 侧面有脸，检测器漏检，**鼻宽 N/A 但仍可指认"糊/重影"**
  empty    检测失败且肤色极低 → 背面/无脸，**不可指认**

用法：
  python splatanything/10b_verify_sheet.py --dir /mnt/d/tmp/r2n/sheets
  python splatanything/10b_verify_sheet.py --dir ... --rewrite   # 回写 manifest
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _load_detector():
    import mediapipe as mp
    # 只用 sel=1：sel=0 在本数据上误检（见 10_nose_sheet.detect_face 注释）
    return mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.3)


def face_in_crop(det, bgr):
    """块内能检出脸吗？返回 (hit, shrink_norm, conf)。shrink: 0=正中心, 1=贴边。"""
    arr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    r = det.process(arr)
    if not r.detections:
        return False, None, None
    d = max(r.detections, key=lambda x: float(x.score[0]))
    bb = d.location_data.relative_bounding_box
    cx, cy = bb.xmin + bb.width / 2, bb.ymin + bb.height / 2
    sh = max(abs(cx - 0.5), abs(cy - 0.5)) * 2
    return True, round(float(sh), 3), round(float(d.score[0]), 3)


def skin_mask(bgr):
    """YCrCb 经典肤色区间 ∩ HSV 排除过暗。"""
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return cv2.bitwise_and(cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127)),
                           cv2.inRange(hsv, (0, 30, 40), (25, 200, 255)))


def skin_centroid(m):
    ys, xs = np.nonzero(m)
    if len(xs) < 50:
        return None, 0.0
    h, w = m.shape
    return (float(xs.mean() / w), float(ys.mean() / h)), float((m > 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='/mnt/d/tmp/r2n/sheets')
    ap.add_argument('--rewrite', action='store_true',
                    help='把 view_class / 体检读数回写 nose_case_manifest.json')
    ap.add_argument('--skin-min', type=float, default=0.06,
                    help='profile 判定的肤色下限')
    ap.add_argument('--align-tol', type=float, default=0.18,
                    help='GT 与 RD 肤色重心差上限，超过=裁窗错位')
    args = ap.parse_args()

    root = Path(args.dir)
    mp_path = root / 'nose_case_manifest.json'
    man = json.loads(mp_path.read_text(encoding='utf-8'))
    det = _load_detector()

    print(f'{"run":24s} {"vw":4s} {"kind":5s} {"cls":8s} {"skin":>6s} {"rdsk":>6s} '
          f'{"dxy":>6s} {"lapvar":>7s} {"shrink":>7s}')
    print('-' * 92)
    stat, kinds = {}, {}
    bad = []
    for bucket in ('runs', 'controls'):
        for run, rec in man[bucket].items():
            tdir = root / 'tiles' / rec['sub']
            for v in rec['views']:
                key = v['view']
                gp, rp = tdir / f'{key}_gt.png', tdir / f'{key}_rd.png'
                if not gp.exists():
                    continue
                g = cv2.imread(str(gp))
                r = cv2.imread(str(rp))
                kind = v.get('roi_kind', 'nose')
                kinds[kind] = kinds.get(kind, 0) + 1
                # V1 / V2 / V4
                hit, shrink, conf = face_in_crop(det, g)
                mg, rg = skin_mask(g), skin_mask(r)
                cg, sg = skin_centroid(mg)
                cr, sr = skin_centroid(rg)
                lap = float(cv2.Laplacian(cv2.cvtColor(g, cv2.COLOR_BGR2GRAY),
                                          cv2.CV_64F).var())
                # V3 对齐
                if cg and cr:
                    dxy = max(abs(cr[0] - cg[0]), abs(cr[1] - cg[1]))
                else:
                    dxy = None

                if hit:
                    cls = 'face'
                elif sg >= args.skin_min:
                    cls = 'profile'
                else:
                    cls = 'empty'
                v['view_class'] = cls
                v['verify'] = {
                    'face_in_crop': bool(hit), 'crop_shrink': shrink,
                    'crop_conf': conf, 'skin_gt': round(sg, 4),
                    'skin_rd': round(sr, 4),
                    'skin_dxy': None if dxy is None else round(dxy, 4),
                    'lap_var': round(lap, 1)}
                stat[cls] = stat.get(cls, 0) + 1

                if kind == 'nose' and cls == 'empty':
                    bad.append((run, key, f'标为鼻特写但肤色仅 {sg:.3f}，框空了'))
                if kind == 'nose' and dxy is not None and dxy > args.align_tol:
                    bad.append((run, key, f'GT/RD 肤色重心差 {dxy:.3f} > {args.align_tol}，错位'))
                if kind == 'nose' and cls == 'face' and shrink is not None and shrink > 0.5:
                    bad.append((run, key, f'脸偏出块 shrink={shrink}'))
                if kind == 'nose' and lap < 20:
                    bad.append((run, key, f'GT 块 lapvar={lap:.0f}，疑似纯色'))

                print(f'{run:24s} {key[:3]:4s} {kind:5s} {cls:8s} {sg:6.3f} {sr:6.3f} '
                      f'{"-" if dxy is None else f"{dxy:.3f}":>6s} {lap:7.1f} '
                      f'{"-" if shrink is None else f"{shrink:.3f}":>7s}')

    print(f'\nroi_kind 分布：{kinds}')
    print(f'三分类：{stat}')
    if bad:
        print(f'\n⚠️ 需注意 {len(bad)} 项：')
        for b in bad:
            print(f'  - {b[0]}/{b[1]}: {b[2]}')
    else:
        print('\n✓ 无异常')

    if args.rewrite:
        for bucket in ('runs', 'controls'):
            for run, rec in man[bucket].items():
                cs = [v.get('view_class') for v in rec['views']]
                rec['n_face'] = sum(1 for c in cs if c == 'face')
                rec['n_profile'] = sum(1 for c in cs if c == 'profile')
                rec['n_empty'] = sum(1 for c in cs if c == 'empty')
        man['_verify'] = {
            'face': stat.get('face', 0), 'profile': stat.get('profile', 0),
            'empty': stat.get('empty', 0),
            'roi_kind': kinds,
            'n_bad': len(bad),
            'rule': ('face=块内能检出人脸（可指认鼻宽）；'
                     f'profile=检出失败但肤色≥{args.skin_min}（侧面，鼻宽 N/A，'
                     '仍可指认糊/重影）；empty=肤色极低（背面，不可指认）'),
        }
        mp_path.write_text(json.dumps(man, ensure_ascii=False, indent=2),
                           encoding='utf-8')
        print(f'\n已回写 {mp_path}')


if __name__ == '__main__':
    main()
