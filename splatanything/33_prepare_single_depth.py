"""Stage frozen camera manifests and the same three RGB files for independent DP2 calls."""
import argparse
import json
import shutil
from pathlib import Path

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    manifest = json.loads((args.source/"manifest.json").read_text())
    (args.out/"rgb").mkdir(parents=True, exist_ok=True)
    assert len(manifest["frames"]) == 3
    for i, frame in enumerate(manifest["frames"]):
        frame["index"] = i
        name = f"{i:02d}.png"
        shutil.copyfile(args.source/"rgb"/name, args.out/"rgb"/name)
    (args.out/"manifest.json").write_text(json.dumps(manifest, indent=2))
