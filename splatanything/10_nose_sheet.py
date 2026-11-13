#!/usr/bin/env python3
"""10_nose_sheet.py — R2-N 鼻部专项的"坏例指认包"生成器。

为什么需要这个脚本
------------------
用户目视反馈"SplatAnything 物体好、人物鼻子变宽，疑似重影"，但没有指出具体是哪个 run、
哪个视角、哪张图。EXPERIMENT_PLAN §0.0 R2-N 明确要求先定位**用户查看的实际输出**，否则
后续 N0~N4 五个假设（相机不一致 / 相机误差 / 深度不一致 / 合并 footprint / 域预算不足）
没有共同靶子，量出的数字也无法判读。

所以本脚本不产生任何质量结论，只做一件事：把**所有可指认的候选（run × 视角）**按统一样式
裁好、编号、铺成一张 sheet，让用户能直接说"就是 B03 这张"。

口径（踩过的坑，别改）
--------------------
1. **GT 与渲染必须走同一个 fit_square**。GT 源图是 1439×1919 竖图，渲染是已裁方的 504²。
   直接 resize 会把竖图压扁，人脸位置与渲染完全错位（fig01b 的注释也强调了这点）。
2. **人脸框逐视角独立检测，不能全图共用一个框**。compare_4methods 的历史事故：固定人脸框
   在 view 8 落到书架上，把锐度虚高 +180。
3. **检测器会漏侧脸**（mediapipe face_detection 在 view 008/022 上 GT 与渲染都 MISS）。
   漏检不算失败，退回到"整图中心偏上"的保守窗，并在 manifest 里标 `roi_source=fallback`，
   让用户知道这个窗没有检测支撑。**不要因为漏检就把该视角从指认包里删掉**——侧面正是
   看"鼻子变宽"最要紧的角度。
4. **放大的只是给人看，不改评分像素**。sheet 里的裁剪窗尺寸和缩放倍数都记进 manifest，
   后续正式量测（鼻翼宽 / 眼间距归一化）重新按 manifest 的框算，不复用这张 PNG。

输出
----
  <out>/sheet_<run>.png      每个 run 一张 contact sheet（编号 A01/B03…）
  <out>/tiles/<run>/<id>.png 单块原尺寸裁图（不含 sheet 上的编号与文字）
  <out>/nose_case_manifest.json  全部候选的元信息 + 待用户填写的 `pick` 字段

用法（WSL，flame_human env 有 mediapipe）：
  python splatanything/10_nose_sheet.py --out /mnt/d/tmp/r2n/sheets
  python splatanything/10_nose_sheet.py --only jiajia_hold_3v --no-fallback
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PAIRS = Path(os.environ.get('SA_PAIRS', 'D:/splatanything_figures/pairs'))
FONT_DIR = Path(os.environ.get('SA_FONT_DIR', 'C:/Windows/Fonts'))

# 候选 run：sub 是 pairs 下的子目录名，(gt_pre, rd_pre) 是文件前缀
RUNS = [
    # name                  sub                     note
    ('jiajia_hold_3v',      'jiajia3_held',         '3v 建云 → 21 个 held-out（R1 主线档）'),
    ('jiajia_hold_3v_self',  'jiajia3_self',         '3v 建云 → 3 个输入视角自洽'),
    ('jiajia_hold_8v',      'jiajia8_held',         '8v 建云 → 16 个 held-out'),
    ('jiajia_hold_13v',     'jiajia13_held',        '13v 建云 → 11 个 held-out'),
    ('jiajia_hold_24v_self', 'jiajia24_self',       '24v 自洽（fig01b 人脸 3× 放大的来源）'),
    ('jiajia_hold_24v_vox001', 'jiajia24_self_vox001', '24v 自洽 · voxel=0.001 加细'),
]

# 物体包：**默认不跑**。物体上没有人脸，mediapipe 在物体上只会误检，13 个"鼻特写"块
# 全是假阳性，会污染指认表。物体控制组的作用（证明重建链路本身没问题）已由
# fig 系列的区域指标覆盖，不需要用户在这里看。需要时用 --with-controls 打开。
CONTROL_RUNS = [
    ('object_3v_held', 'obj3_held', '物体 3v held-out（链路控制，无人脸）'),
    ('object_8v_held', 'obj8_held', '物体 8v held-out（链路控制，无人脸）'),
]


def _font(size, bold=False):
    names = ['msyhbd.ttc', 'msyh.ttc'] if bold else ['msyh.ttc']
    for n in names:
        p = FONT_DIR / n
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size, index=0)
            except Exception:
                pass
    return ImageFont.load_default()


def fit_square(im, size):
    """等比缩放 + 居中裁成正方形。size=None 时只裁不缩放（保留原生分辨率）。"""
    sw, sh = im.size
    s = min(sw, sh)
    im = im.crop(((sw - s) // 2, (sh - s) // 2, (sw + s) // 2, (sh + s) // 2))
    return im if size is None else im.resize((size, size), Image.LANCZOS)


_DET = {}


def _detector(sel):
    if sel not in _DET:
        import mediapipe as mp
        _DET[sel] = mp.solutions.face_detection.FaceDetection(
            model_selection=sel, min_detection_confidence=0.3)
    return _DET[sel]


def detect_face(sq):
    """返回 (cx, cy, w, conf, sel) 归一化；检不到返回 None。

    **只用 model_selection=1（full-range）**。实测（134 视角逐个对照全分辨率结果）：
      - sel=0 在 view 001 上给出 cy=0.297 的框，而全分辨率高置信度(0.94)框在 cy=0.127，
        sel=0 这个框 conf 只有 0.52 —— 是误检（框到胸口/衣服），会把鼻窗整个带偏。
      - sel=1 与全分辨率结论逐视角一致（|Δcy| < 0.01），是本数据唯一可信档位。
    阈值用 0.3：低于 0.3 的检测在本数据上一次也没命中真脸，只会引入噪声。
    """
    arr = np.asarray(sq)
    try:
        r = _detector(1).process(arr)
    except Exception:
        return None
    if not r.detections:
        return None
    d = max(r.detections, key=lambda x: float(x.score[0]))
    bb = d.location_data.relative_bounding_box
    return (float(bb.xmin + bb.width / 2), float(bb.ymin + bb.height / 2),
            float(bb.width), float(d.score[0]), 1)


def nose_window(sq, det, span_mult=2.6, y_shift=0.06):
    """由人脸框推鼻部裁窗。返回 (x0, y0, side, roi_source, kind)。

    `kind` 只有两种：
      nose  检测到脸 → 鼻部特写窗（边长 = bbox_w × span_mult）
      full  检测失败 → **整幅方图**，不做任何放大

    为什么漏检时给整幅而不是"保守窗"：本数据里漏检的视角是纯侧脸/背面，脸根本不在
    图像中心偏上那个位置（实测 007/023 的肤色重心在 y≈0.6~0.75），任何固定保守窗都会
    框空。显示整幅至少能让用户看清"这是侧面"，不会误以为框错了。**宁可给全景也绝不
    给一个假装是鼻部的空窗。**

    贴边处理：检测框靠边时整窗内移，不缩窗（缩了看不清鼻翼）。
    """
    W = sq.width
    if det is None:
        return 0, 0, W, 'fallback→full', 'full'
    cx, cy, w, conf, sel = det
    cx *= W
    cy = cy * W + y_shift * w * W
    side = w * W * span_mult
    src = f'detect(sel={sel},conf={conf:.2f})'
    half = side / 2.0
    if half <= W / 2:
        cx = min(max(cx, half), W - half)
    else:
        cx = W / 2
    side = int(max(48, min(W, round(side))))
    x0 = int(round(cx - side / 2))
    y0 = int(round(cy - side / 2))
    x0 = max(0, min(W - side, x0))
    y0 = max(0, min(W - side, y0))
    return x0, y0, side, src, 'nose'


def _view_key(fname):
    """从配对文件名取视角 id。

    人物包是 `gt_001_5093310374431078.png` → `001`；
    物体包是 `gt_frame_0000.png` → `frame_0000`（不要按 '_' 切第二段，会全塌成 'frame'）。
    """
    stem = fname.rsplit('.', 1)[0]
    for pre in ('gt_', 'rd_'):
        if stem.startswith(pre):
            return stem[len(pre):]
    return stem


def refine_window(gt_sq, x0, y0, side, rounds=2, tol=0.32):
    """自纠错：裁出来再检一次，脸还偏就按新检测重定窗。

    为什么需要：mediapipe 在整图 504² 上给的是粗框，直接乘倍数裁，小脸/侧脸会偏。
    实测 134 个视角里有 8 个 shrink>0.6（脸跑到窗角）。这里把窗内图放大到 512 再检测
    ——检测器对放大后的脸更准——用新框中心反推回 504 坐标重定窗，最多两轮。

    返回 (x0, y0, side, extra_src, shrink)，shrink 为最后一次窗内检测的偏心度。
    """
    src = ''
    shrink = None
    for r in range(rounds):
        crop = gt_sq.crop((x0, y0, x0 + side, y0 + side)).resize(
            (512, 512), Image.LANCZOS)
        det = detect_face(crop)
        if det is None:
            shrink = None      # 窗内检不到（纯侧脸/糊），无法评估偏心
            break
        ncx, ncy, nw, _conf, _sel = det
        shrink = max(abs(ncx - 0.5), abs(ncy - 0.5)) * 2
        if shrink <= tol:
            break
        if r == rounds - 1:
            break              # 最后一轮不再重定，把实测 shrink 留给体检
        # 窗内归一化坐标 → 504 原图坐标
        gx = x0 + ncx * side
        gy = y0 + ncy * side
        new_side = int(max(side, nw * side * 2.6))
        new_side = max(48, min(gt_sq.width, new_side))
        half = new_side / 2.0
        gx = min(max(gx, half), gt_sq.width - half)
        gy = min(max(gy, half), gt_sq.width - half)
        x0 = int(round(gx - half))
        y0 = int(round(gy - half))
        side = new_side
        x0 = max(0, min(gt_sq.width - side, x0))
        y0 = max(0, min(gt_sq.width - side, y0))
        src = '+refined'
    return x0, y0, side, src, shrink


def build_run(sub, tile=300, scale_up=1.0, no_fallback=False):
    """一个 run 的所有视角 → [(id, gt_crop, rd_crop, meta)]。"""
    d = PAIRS / sub
    if not d.is_dir():
        return []
    gts = {}
    for p in sorted(d.glob('gt_*.png')):
        gts.setdefault(_view_key(p.name), p)
    rds = {}
    for p in sorted(d.glob('rd_*.png')):
        rds.setdefault(_view_key(p.name), p)

    out = []
    for key in sorted(set(gts) & set(rds)):
        try:
            gt_im = Image.open(gts[key]).convert('RGB')
            rd_im = Image.open(rds[key]).convert('RGB')
        except Exception as e:
            print(f'  ! {sub}/{key} 读图失败: {e}', file=sys.stderr)
            continue
        # ⚠️ ROI 全程用**归一化坐标**（0~1），只在裁剪时乘各自原生边长。
        # GT 是 1439×1919、RD 是 504²。早期版本把 GT 也先降采样到 504 再裁 133px，
        # 等于把 GT 的 380 个原生像素丢成 133 个，再放大 2.7× —— 用户看到的 GT 是糊的，
        # 会把插值模糊错记成"模型模糊"。**GT 必须走原生分辨率**。
        gt_nat = fit_square(gt_im, None)      # 1439²（只裁方，不缩放）
        sq_g = gt_nat.resize((504, 504), Image.LANCZOS)   # 仅供检测用，不进成品
        rd_sq = fit_square(rd_im, 504)
        det_g = detect_face(sq_g)
        det_r = detect_face(rd_sq)
        # 裁窗**只由 GT 决定**，绝不借渲染侧框。原因：同一视角的 GT 在 3v/8v/13v/24v
        # 里是同一张图，若某个 run 借了渲染框，同一张 GT 就会在不同 run 裁成不同窗，
        # 用户看到的差异里混进裁窗成分。渲染检测只作旁证记录（det_rd）。
        if det_g is None and no_fallback:
            continue
        x0, y0, side, src, kind = nose_window(sq_g, det_g)
        if kind == 'nose':
            x0, y0, side, extra, shrink = refine_window(sq_g, x0, y0, side)
            src += extra
        else:
            shrink = None
        # 归一化 ROI（相对 504 方图）→ 按各自原生边长裁
        nx0, ny0 = x0 / 504.0, y0 / 504.0
        ns = side / 504.0
        t = int(round(tile * scale_up))

        def _cut(sq_img):
            Wn = sq_img.width
            ax0, ay0 = int(round(nx0 * Wn)), int(round(ny0 * Wn))
            asd = max(1, int(round(ns * Wn)))
            ax1, ay1 = min(Wn, ax0 + asd), min(Wn, ay0 + asd)
            return sq_img.crop((ax0, ay0, ax1, ay1)).resize((t, t), Image.LANCZOS)

        g = _cut(gt_nat)
        r = _cut(rd_sq)
        out.append((key, g, r, {
            'view': key,
            'gt_file': gts[key].name,
            'rd_file': rds[key].name,
            # 归一化 ROI 是正式口径；像素框随源分辨率不同而不同，两者都记，避免歧义
            'roi_norm': [round(nx0, 5), round(ny0, 5), round(ns, 5)],
            'roi_box_504': [x0, y0, x0 + side, y0 + side],
            'roi_box_gt_native': [int(round(nx0 * gt_nat.width)),
                                  int(round(ny0 * gt_nat.width)),
                                  int(round((nx0 + ns) * gt_nat.width)),
                                  int(round((ny0 + ns) * gt_nat.width))],
            'roi_side': side,
            'gt_native_px': gt_nat.width,
            'rd_native_px': rd_sq.width,
            'roi_source': src,
            'roi_kind': kind,
            'shrink': None if shrink is None else round(shrink, 3),
            'det_gt': None if det_g is None else {
                'cx': round(det_g[0], 4), 'cy': round(det_g[1], 4),
                'w': round(det_g[2], 4), 'conf': round(det_g[3], 3),
                'model_selection': det_g[4]},
            'det_rd': None if det_r is None else {
                'cx': round(det_r[0], 4), 'cy': round(det_r[1], 4),
                'w': round(det_r[2], 4), 'conf': round(det_r[3], 3),
                'model_selection': det_r[4]},
        }))
    return out


def render_sheet(name, note, items, out_path, tile=300, cols=6):
    """编号 A01.. 的 contact sheet：每块**竖排** GT 在上、RD 在下。

    块宽只有 tile（不是 2×tile）——早期版本按左右并排算了 pair_w，造成 6 列 sheet
    宽 3754 但只用了一半，全是大片空白。
    """
    if not items:
        return None
    pad, gut, hdr, tag_h, gap_gr = 18, 14, 96, 30, 46
    blk_w = tile
    blk_h = tile * 2 + 8 + tag_h
    rows = (len(items) + cols - 1) // cols
    W = pad * 2 + cols * blk_w + (cols - 1) * gut
    H = hdr + pad + rows * blk_h + (rows - 1) * gap_gr + pad + 30
    im = Image.new('RGB', (W, H), (252, 253, 255))
    d = ImageDraw.Draw(im)
    INK, INK2, RULE = (24, 30, 38), (104, 116, 132), (214, 221, 230)
    INDIGO, RED, GREEN, AMBER = (58, 84, 160), (186, 60, 54), (32, 132, 96), (176, 118, 22)

    n_nose = sum(1 for x in items if x[3].get('roi_kind') == 'nose')
    n_full = len(items) - n_nose
    d.text((pad, 20), f'{name} — 鼻部裁窗指认表', font=_font(30, True), fill=INK)
    d.text((pad, 58), note, font=_font(17), fill=INK2)
    legend = (f'{n_nose} 块 = 鼻部特写（绿号，可指认）    '
              f'{n_full} 块 = 无检测·整幅全景（橙号，多为侧面/背面，不可指认鼻宽）')
    d.text((pad, 84), legend, font=_font(15), fill=INK2)
    d.line([(pad, hdr - 12), (W - pad, hdr - 12)], fill=RULE, width=1)

    for i, (key, g, r, meta) in enumerate(items):
        rr, cc = divmod(i, cols)
        x = pad + cc * (blk_w + gut)
        y = hdr + pad + rr * (blk_h + gap_gr)
        kind = meta.get('roi_kind', 'nose')
        im.paste(g, (x, y))
        im.paste(r, (x, y + tile + 8))
        d.rectangle([x, y, x + tile - 1, y + 2 * tile + 7],
                    outline=AMBER if kind == 'full' else RULE, width=2 if kind == 'full' else 1)
        d.line([(x, y + tile + 3), (x + tile - 1, y + tile + 3)], fill=RULE, width=1)
        tag = f'{chr(65 + rr)}{cc + 1:02d}'
        d.text((x + 4, y + 4), tag, font=_font(22, True),
               fill=AMBER if kind == 'full' else GREEN)
        d.text((x + tile - 4, y + 6), key[:3], font=_font(15), fill=INDIGO, anchor='ra')
        base = y + 2 * tile + 12
        d.text((x, base), 'GT', font=_font(14, True), fill=INK2)
        d.text((x + tile, base), 'RD', font=_font(14, True),
               fill=AMBER if kind == 'full' else INDIGO, anchor='ra')
        if kind == 'full':
            d.text((x, base + tag_h - 10), 'no-face·全景', font=_font(12), fill=AMBER)

    im.save(out_path, optimize=True)
    print(f'  ✓ {out_path.name}  {W}x{H}  {len(items)} 块  '
          f'（鼻特写 {n_nose} / 全景 {n_full}）  '
          f'{os.path.getsize(out_path) / 1e6:.2f} MB')
    return {'file': out_path.name, 'w': W, 'h': H, 'n': len(items),
            'n_nose': n_nose, 'n_full': n_full}


def main():
    global PAIRS
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='D:/tmp/r2n/sheets')
    ap.add_argument('--pairs', default=str(PAIRS))
    ap.add_argument('--tile', type=int, default=360,
                    help='单块像素。实测鼻窗约 135 方图像素，360 已是 2.7× 上采样，'
                         '再小看不出鼻翼')
    ap.add_argument('--cols', type=int, default=6)
    ap.add_argument('--scale-up', type=float, default=1.0,
                    help='单块放大倍数（只影响出图观感，manifest 记的是原始框）')
    ap.add_argument('--only', default=None, help='只跑某个 sub')
    ap.add_argument('--no-fallback', action='store_true',
                    help='检测不到脸的视角直接跳过（默认保留并出整幅全景，标橙号）')
    ap.add_argument('--with-controls', action='store_true',
                    help='同时跑物体控制组（默认关：物体无人脸，检测全是假阳性）')
    args = ap.parse_args()

    PAIRS = Path(args.pairs)
    out = Path(args.out)
    (out / 'tiles').mkdir(parents=True, exist_ok=True)

    manifest = {
        '_readme': __doc__.strip().split('\n')[0],
        '_how_to_pick': (
            '每个块竖排两张：上 = GT 原图，下 = 重建渲染（RD），同一裁窗、同一缩放。'
            '看 RD 相对 GT：鼻子是否更宽 / 边缘是否发虚 / 是否有双层轮廓（重影）。'
            '把该块左上角的编号填进对应视角的 pick 字段（默认 null），'
            '并把 symptom 填成 wide（只变宽）/ ghost（可见双层）/ blur（糊但不变宽）/ '
            'ok（正常对照）。一个 run 可以指多个，跨 run 都可指。'
            '⚠️ 橙号块（roi_kind=full）是无检测的整幅全景，多为纯侧面或背面，'
            '**不能用来判断鼻宽**，仅用于确认该视角确实拍不到正脸。'),
        '_repro_note': (
            '同一视角的 GT 在 3v/8v/13v/24v 是同一张图。裁窗只由 GT 决定，'
            '因此同一视角 cross-run 的裁窗完全相同（manifest 里 roi_box_504 可直接比对）。'
            '若发现不一致就是 bug。'),
        'runs': {},
        'controls': {},
        'pairs_root': str(PAIRS),
        'tile_px': args.tile,
    }

    all_specs = [(n, s, note, 'runs') for n, s, note in RUNS]
    if args.with_controls:
        all_specs += [(n, s, note, 'controls') for n, s, note in CONTROL_RUNS]

    for name, sub, note, bucket in all_specs:
        if args.only and sub != args.only and name != args.only:
            continue
        print(f'[{name}] sub={sub}')
        items = build_run(sub, tile=args.tile, scale_up=args.scale_up,
                          no_fallback=args.no_fallback)
        if not items:
            print('  (跳过：无配对图)')
            continue
        # 单块也存一份原尺寸裁图，方便用户放大看
        tdir = out / 'tiles' / sub
        tdir.mkdir(parents=True, exist_ok=True)
        for key, g, r, meta in items:
            g.save(tdir / f'{key}_gt.png', optimize=True)
            r.save(tdir / f'{key}_rd.png', optimize=True)
        info = render_sheet(name, note, items, out / f'sheet_{name}.png',
                            tile=args.tile, cols=args.cols)
        recs = []
        for i, (key, g, r, meta) in enumerate(items):
            rr, cc = divmod(i, args.cols)
            m = dict(meta)
            m['block_id'] = f'{chr(65 + rr)}{cc + 1:02d}'   # 页面上那个编号
            m['pick'] = None                                 # ← 用户填
            m['symptom'] = None                              # ← 用户填
            recs.append(m)
        manifest[bucket][name] = {
            'sub': sub, 'note': note, 'sheet': info['file'] if info else None,
            'n_views': len(items),
            'n_nose': info['n_nose'] if info else 0,
            'n_full': info['n_full'] if info else 0,
            'views': recs,
        }

    mp_path = out / 'nose_case_manifest.json'
    mp_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                       encoding='utf-8')
    n_run = len(manifest['runs']) + len(manifest['controls'])
    n_view = sum(v['n_views'] for b in ('runs', 'controls')
                 for v in manifest[b].values())
    n_nose = sum(v['n_nose'] for b in ('runs', 'controls')
                 for v in manifest[b].values())
    n_full = sum(v['n_full'] for b in ('runs', 'controls')
                 for v in manifest[b].values())
    print(f'\n完成：{n_run} 个 run / {n_view} 个视角')
    print(f'      鼻部特写 {n_nose} 块（可指认鼻宽） / 全景 {n_full} 块（无检测，仅供确认是侧面）')
    print(f'清单：{mp_path}')


if __name__ == '__main__':
    main()
