#!/usr/bin/env python3
"""Plot training JSONL as dependency-light SVG and a readable metrics CSV.

matplotlib is deliberately not required. SVG is an actual loss curve, not a
placeholder. Non-finite values and empty/non-step event rows are excluded.
"""
import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image

CURVES = (("loss", "Total loss", "#bf4b35"),
          ("rgb_person", "Person RGB MSE", "#4569a8"),
          ("depth_loss_m2", "Valid person Z MSE (m^2)", "#388b7d"),
          ("alpha_loss", "Person alpha MSE", "#a57a27"))


def rows(path):
    with open(path, encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    return [r for r in data if r.get("event") == "step" and np.isfinite(r.get("step_s", np.nan))]


def panel(box, x, y, width, height, label, color):
    px, py, pw, ph = 64, 34, width - 90, height - 74
    vals = [r.get(label) for r in box if r.get(label) is not None and np.isfinite(r[label])]
    if not vals:
        return ""
    xmin, xmax = box[0]["step"], box[-1]["step"]
    lo, hi = min(vals + [0.]), max(vals)
    hi = hi * 1.05 if hi > lo else lo + 1e-9
    coords = []
    for r in box:
        value = r.get(label)
        if value is None or not np.isfinite(value):
            continue
        xx = px + (r["step"] - xmin) / max(xmax - xmin, 1) * pw
        yy = py + (hi - value) / (hi - lo) * ph
        coords.append((xx, yy))
    title = {key: label for key, label, _ in CURVES}[label]
    s = ['<rect x="0" y="0" width="%d" height="%d" fill="#ffffff" stroke="#d5d9df"/>' % (width, height),
         '<text x="%d" y="22" class="title">%s</text>' % (16, html.escape(title)),
         '<text x="%d" y="%d" class="small">step %d..%d; max %.6g</text>' %
         (px, height - 10, xmin, xmax, max(vals))]
    for i in range(5):
        yy = py + i * ph / 4
        v = hi - i * (hi - lo) / 4
        s.append('<line x1="%d" y1="%.2f" x2="%d" y2="%.2f" stroke="#edf0f2"/>' %
                 (px, yy, px + pw, yy))
        s.append('<text x="%d" y="%.2f" text-anchor="end" class="small">%.3g</text>' %
                 (px - 6, yy + 4, v))
    s.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="1.6"/>' %
             (" ".join("%.2f,%.2f" % p for p in coords), color))
    return '<g transform="translate(%d,%d)">%s</g>' % (x, y, "".join(s))


def plot_svg(data, path, run_label=""):
    width, height = 620, 260
    chunks = [panel(data, (i % 2) * width, (i // 2) * height, width, height, k, c)
              for i, (k, _, c) in enumerate(CURVES)]
    if data:
        rows_tail = data[-min(20, len(data)):]
        median = float(np.median([r["step_s"] for r in rows_tail]))
        mean = float(np.mean([r["step_s"] for r in rows_tail]))
        footer = ("Steps %d; final %.7g; last-%d mean %.3fs, median %.3fs/step; "
                  "measured full-step 2000-step ETA %.2fh; includes source cache, "
                  "target preprocessing and source replay; model build/eval excluded") % (
                      len(data), data[-1]["loss"], len(rows_tail), mean, median, mean * 2000 / 3600)
    else:
        footer = "No completed steps; no ETA inferred"
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d">'
           '<style>text{font-family:Arial,sans-serif;fill:#1e2935}.title{font-size:14px}'
           '.small{font-size:11px;fill:#596674}.head{font-size:18px;font-weight:bold}</style>'
           '<rect width="100%%" height="100%%" fill="#f3f5f7"/>'
           '<text x="16" y="28" class="head">SA portrait short training %s</text>'
           '<g transform="translate(0,38)">%s</g>'
           '<text x="16" y="%d" class="small">%s</text></svg>') % (
               2 * width, 2 * height + 88, 2 * width, 2 * height + 88,
               html.escape(run_label), "".join(chunks), 2 * height + 64, html.escape(footer))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(svg)


def png(data, path):
    width, height = 1240, 608
    im = Image.new("RGB", (width, height), "#f3f5f7")
    for i, (key, _, color) in enumerate(CURVES):
        vals = [r.get(key) for r in data if r.get(key) is not None and np.isfinite(r[key])]
        if not vals:
            continue
        ox, oy = (i % 2) * 620 + 64, (i // 2) * 260 + 72
        hi, lo = max(vals) * 1.05, min(vals + [0.])
        points = []
        for r in data:
            if r.get(key) is None or not np.isfinite(r[key]):
                continue
            x = ox + (r["step"] - data[0]["step"]) / max(data[-1]["step"] - data[0]["step"], 1) * 530
            y = oy + (hi - r[key]) / max(hi - lo, 1e-15) * 186
            points.append((int(x), int(y)))
        from PIL import ImageDraw
        d = ImageDraw.Draw(im)
        d.rectangle((ox, oy, ox + 530, oy + 186), fill="white", outline="#d5d9df")
        if len(points) > 1:
            d.line(points, fill=color, width=2)
        d.text((ox, oy - 30), {key: label for key, label, _ in CURVES}[key], fill="#1e2935")
        d.text((ox, oy + 190), "steps %d..%d; max %.5g" % (data[0]["step"], data[-1]["step"], max(vals)), fill="#596674")
    d = ImageDraw.Draw(im)
    d.text((16, height - 22), "Loss SVG has axes/ETA; PNG is the same curves for quick viewing.", fill="#596674")
    im.save(path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    args = p.parse_args(argv)
    run = Path(args.run)
    data = rows(run / "metrics.jsonl")
    plot_svg(data, run / "loss_curves.svg", run.name)
    png(data, run / "loss_curves.png")
    print("LOSS_PLOT", run / "loss_curves.svg", "rows", len(data))


if __name__ == "__main__":
    main()
