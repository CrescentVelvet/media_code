"""Summarize the paired acceptance tests on the evaluated packs."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    parser.add_argument("names",nargs="+")
    args=parser.parse_args()
    result={}
    for name in args.names:
        frames=json.loads((args.output/name/"eval.json").read_text())
        valid=[f["arms"] for f in frames if "A" in f["arms"]]
        entry={"scored":len(valid),"targets":len(frames)}
        for arm in "BC":
            stats=dict(worst_coverage_delta_pp=100*min(f[arm]["coverage"]-f["A"]["coverage"] for f in valid),
                       max_outside_delta_255=max(f[arm]["outside_delta_255"] for f in valid))
            if "nose_tip" in valid[0][arm]:
                vals=[1-f[arm][region]["p90"]/f["A"][region]["p90"] for f in valid
                      for region in ("nose_tip","nose_ala")]
                stats.update(nose_p90_reduction_pct=float(100*np.mean(vals)),
                    worst_face_mae_increase_mm=max(f[arm]["face_tight_landmark"]["mae"]-
                        f["A"]["face_tight_landmark"]["mae"] for f in valid),
                    min_common_support=min(f[arm][region]["common_fraction"] for f in valid
                        for region in ("nose_tip","nose_ala","face_tight_landmark")))
                stats["static_pass"]=bool(stats["nose_p90_reduction_pct"]>=10 and
                    stats["worst_face_mae_increase_mm"]<=1 and stats["min_common_support"]>=.5)
            entry[arm]=stats
        result[name]=entry
    (args.output/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
