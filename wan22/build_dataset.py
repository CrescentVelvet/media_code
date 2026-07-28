import os, csv, glob

data_dir = os.environ["DATA_DIR"]
txt_dir = os.environ.get("TXT_DIR", "")
fixed_prompt = os.environ.get("PROMPT", "")
base_path = os.environ.get("DATASET_BASE_PATH", data_dir)
metadata_out = os.environ["METADATA_OUT"]

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".webm", ".mkv", ".flv")

# Collect video files (recursive).
videos = []
for root, _, files in os.walk(data_dir):
    for f in sorted(files):
        if f.lower().endswith(VIDEO_EXTS):
            videos.append(os.path.join(root, f))

if not videos:
    raise SystemExit(f"ERROR: no video files found in {data_dir}")

print(f"[*] found {len(videos)} videos in {data_dir}")

# Build metadata rows.
rows = []
missing_txt = 0
for vpath in videos:
    # Relative path from base_path (so training finds the file regardless of cwd).
    try:
        rel = os.path.relpath(vpath, base_path)
    except ValueError:
        rel = vpath

    # Prompt: per-video .txt if TXT_DIR is set, else fixed prompt.
    prompt = fixed_prompt
    if txt_dir:
        stem = os.path.splitext(os.path.basename(vpath))[0]
        # Try matching the relative structure under txt_dir.
        rel_txt = os.path.splitext(rel)[0] + ".txt"
        txt_path = os.path.join(txt_dir, rel_txt)
        if not os.path.isfile(txt_path):
            # Fallback: just the stem .txt at the txt_dir root.
            txt_path = os.path.join(txt_dir, stem + ".txt")
        if os.path.isfile(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
        else:
            missing_txt += 1

    rows.append({"video": rel, "prompt": prompt})

if txt_dir and missing_txt:
    print(f"[*] WARNING: {missing_txt}/{len(rows)} videos had no matching .txt — using fixed/empty prompt")

# Write metadata.csv.
os.makedirs(os.path.dirname(os.path.abspath(metadata_out)) or ".", exist_ok=True)
with open(metadata_out, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["video", "prompt"])
    writer.writeheader()
    writer.writerows(rows)

print(f"[*] wrote {len(rows)} rows -> {metadata_out}")
print(f"    columns: video (relative to {base_path}), prompt")
print(f"    sample:  {rows[0]}")
