"""Summarize frozen paired geometry gates for each experiment pack."""
import argparse
import json
from pathlib import Path
from statistics import mean

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", type=Path)
    ap.add_argument("names", nargs="+")
    args = ap.parse_args()
    summary = {}
    for name in args.names:
        rows = json.loads((args.out/name/"eval.json").read_text())
        valid = [r["gaussians"] for r in rows if "gaussians" in r]
        report = {"targets": len(rows), "valid_faces": len(valid)}
        for arm in ("R0", "Rtri"):
            stats = {"worst_coverage_delta_pp": 100*min(f[arm]["coverage"]-f["A"]["coverage"] for f in valid)}
            if "nose_tip" in valid[0][arm]:
                regions = ("nose_tip", "nose_ala")
                gain = 100*mean(1-f[arm][k]["p90"]/f["A"][k]["p90"] for f in valid for k in regions)
                worst = max(f[arm]["face_tight_landmark"]["mae"]-f["A"]["face_tight_landmark"]["mae"] for f in valid)
                support = min(f[arm][k]["coverage"] for f in valid for k in (*regions,"face_tight_landmark"))
                stats.update(nose_reduction_pct=gain, worst_face_increase_mm=worst, min_support=support,
                             static_pass=gain>=10 and worst<=1 and support>=.5)
            report[arm] = stats
        summary[name] = report
    (args.out/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
