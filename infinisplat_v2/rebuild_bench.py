#!/usr/bin/env python3
"""从已有结果目录重建 bench 汇总（CSV + markdown 表），不重跑推理。

用途：推理已经把 scene.ply 落盘后，若指标口径/字段名改了，不必重跑 GPU，
直接从每个 <prefix>_<case>/ 目录里的 result.json + eval/metrics_*.json 重算。

用法:
  python rebuild_bench.py --results ~/output/infinisplat_v2_results --prefix threeview
"""
import argparse
import json
from pathlib import Path

COLS = [("case", "物体"), ("views", "视角"), ("sec", "端到端(s)"),
        ("peak_mem_mb", "显存峰值(MB)"), ("gaussians", "高斯数"),
        ("psnr", "自洽PSNR"), ("ssim", "自洽SSIM")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(Path.home() / "output/infinisplat_v2_results"))
    ap.add_argument("--prefix", default="threeview")
    ap.add_argument("--csv", default=None, help="已有 CSV 路径（读它取耗时/显存列）")
    args = ap.parse_args()

    res = Path(args.results)
    csv_path = Path(args.csv) if args.csv else res / f"{args.prefix}_bench.csv"
    old = {}
    if csv_path.exists():
        import csv as csvmod
        for r in csvmod.DictReader(open(csv_path, encoding="utf-8")):
            old[r["case"]] = r

    rows = []
    for d in sorted(res.glob(f"{args.prefix}_*")):
        if not d.is_dir() or d.name.endswith("_logs"):
            continue
        rj = d / "result.json"
        if not rj.exists():
            continue
        case = d.name[len(args.prefix) + 1:]
        meta = json.loads(rj.read_text(encoding="utf-8"))
        stats = meta.get("stats", {})
        row = dict(old.get(case, {}))
        row.update({"case": case, "views": len(meta.get("inputs", [])),
                    "gaussians": stats.get("final_count", 0),
                    "pose_mode": meta.get("pose_mode", "")})
        row.setdefault("sec", "")
        row.setdefault("peak_mem_mb", "")
        m = d / "eval" / "metrics_rerender.json"
        if m.exists():
            mm = json.loads(m.read_text(encoding="utf-8"))
            # render_check.py 字段名是 psnr_mean/ssim_mean
            row["psnr"] = f"{mm.get('psnr_mean', float('nan')):.3f}"
            row["ssim"] = f"{mm.get('ssim_mean', float('nan')):.4f}"
        rows.append(row)

    keys = ["case", "views", "long_side", "pose_mode", "sec", "peak_mem_mb",
            "base_mem_mb", "gaussians", "psnr", "ssim"]
    import csv as csvmod
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csvmod.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    lines = ["| " + " | ".join(t for _, t in COLS) + " |", "|" + "---|" * len(COLS)]
    for r in rows:
        cells = []
        for k, _ in COLS:
            v = str(r.get(k, ""))
            if k == "gaussians" and v.isdigit():
                v = f"{int(v):,}"
            cells.append(v)
        lines.append("| " + " | ".join(cells) + " |")
    md = "\n".join(lines)
    (res / f"{args.prefix}_bench.md").write_text(md + "\n", encoding="utf-8", newline="\n")
    print(md)
    print(f"\n✅ {csv_path}")


if __name__ == "__main__":
    main()
