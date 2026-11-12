#!/usr/bin/env python3
"""08b_nvs_compare.py — 把多份 nvs_summary.json 汇成一张可比的表。

关键点：不同 source 配置的"留出集合"大小不同（24−N），直接比各自均值仍不公平。
子集是嵌套的（held_out(13v) ⊂ held_out(8v) ⊂ held_out(3v)），所以取**交集视角**
重算一遍，才是真正同视角对比的两列。

用法：
  python 08b_nvs_compare.py --runs a/nvs_summary.json b/nvs_summary.json ...
                             [--out compare.json] [--csv compare.csv]
"""
import argparse
import json
from pathlib import Path

KEYS = ('psnr', 'psnr_fg', 'psnr_bg', 'ssim', 'alpha_mean', 'sharp_ratio')


def mean_of(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None and r[key] == r[key]]
    return sum(vals) / len(vals) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--csv', default=None)
    args = ap.parse_args()

    runs = []
    for p in args.runs:
        d = json.loads(Path(p).read_text())
        d['_label'] = f'source {d["n_source"]}v'
        d['_path'] = str(Path(p).parent)
        runs.append(d)

    # 交集：所有 run 都留出过的视角
    common = {v['id'] for v in runs[0]['views_detail']}
    for r in runs[1:]:
        common &= {v['id'] for v in r['views_detail']}
    common = sorted(common)

    print(f'{"配置":<14s} {"留出":>4s} {"PSNR":>7s} {"前景":>7s} {"SSIM":>7s} '
          f'{"alpha":>7s} {"锐度":>6s} | {"交集n":>5s} {"PSNR":>7s} {"前景":>7s} {"SSIM":>7s}')
    print('-' * 108)
    out = {'common_ids': common, 'runs': []}
    for r in runs:
        a = r['views_detail']
        b = [v for v in a if v['id'] in common]

        def f(x):
            return '  n/a ' if x is None else f'{x:7.2f}'

        print(f'{r["_label"]:<14s} {len(a):>4d} {f(mean_of(a, "psnr"))} {f(mean_of(a, "psnr_fg"))} '
              f'{f(mean_of(a, "ssim"))} {f(mean_of(a, "alpha_mean"))} '
              f'{f(None if mean_of(a, "sharp_ratio") is None else 100 * mean_of(a, "sharp_ratio"))} | '
              f'{len(b):>5d} {f(mean_of(b, "psnr"))} {f(mean_of(b, "psnr_fg"))} {f(mean_of(b, "ssim"))}')
        out['runs'].append(dict(
            label=r['_label'], path=r['_path'], n_source=r['n_source'],
            n_held_out=len(a),
            all={k: mean_of(a, k) for k in KEYS},
            common={k: mean_of(b, k) for k in KEYS},
        ))

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f'\n台账: {args.out}')
    if args.csv:
        lines = ['config,n_source,n_held,subset,' + ','.join(KEYS)]
        for run in out['runs']:
            for subset in ('all', 'common'):
                vals = ','.join('' if run[subset][k] is None else f'{run[subset][k]:.4f}' for k in KEYS)
                n = run['n_held_out'] if subset == 'all' else len(common)
                lines.append(f'{run["label"]},{run["n_source"]},{n},{subset},{vals}')
        Path(args.csv).write_text('\n'.join(lines) + '\n')
        print(f'CSV: {args.csv}')


if __name__ == '__main__':
    main()
