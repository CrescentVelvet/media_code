"""
sam3_track_dedup.py — Detect and merge duplicate / ID-switched tracks in SAM3 manifest.

Tracks shorter than MIN_HITS frames are removed unconditionally.

For every pair of surviving tracks that share a consecutive run of
>= MIN_LOCK_FRAMES frames with IoM (intersection over min area) > IOM_THRESHOLD,
they are declared the same person. The longer track is master; the shorter is
slave. The slave is then classified frame-by-frame:

  * Locked frames (in a high-IoM run): slave's mask files duplicate master's
    and are deleted; slave's manifest entries for those frames are dropped.
  * Common-but-low-IoM frames (same stem, different box): pick the box with the
    higher face-likeness score (aspect ratio close to 0.75 + sane area) and let
    it overwrite the other's mask + manifest box.
  * Slave-only frames (master absent): rename slave's mask files to master's
    pid (no overwrite needed).

If slave's retained (replace + rename) frame count is < MIN_HITS, those frames
are also dropped (slave becomes a pure duplicate).

The script defaults to dry-run; pass --apply to actually rewrite the manifest
and rename/overwrite/delete mask files.

Usage:
  python sam3_track_dedup.py --manifest M.json --masks_dir D [--apply] \\
      [--report R.json] [--min_hits 15] [--iom_threshold 0.8] \\
      [--min_lock_frames 5] [--likeness_margin 0.2] [--out_dir OUT_DIR]
"""
import argparse, json, os, re, shutil, time
from collections import defaultdict

MASK_TAG = re.compile(r"^(.+)\.p(\d+)\.(mask|alpha)\.png$")


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(path, mf):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mf, f, ensure_ascii=False, indent=2)


def box_xyxy(b):
    x, y, w, h = b
    return (x, y, x + w, y + h)


def iom(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    am = (ax2 - ax1) * (ay2 - ay1)
    bm = (bx2 - bx1) * (by2 - by1)
    m = min(am, bm)
    return inter / m if m > 1e-12 else 0.0


def face_likeness(box_xywh):
    """Score 0-1: how face-like is a 2D box? Ideal face aspect W/H ~= 0.75
    (portrait). Linear falloff to 0 at ar>=1.75 or ar<=0.1. Penalty for absurd
    areas (too small / too large)."""
    _, _, w, h = box_xywh
    if w <= 0 or h <= 0:
        return 0.0
    ar = w / h
    ar_score = max(0.0, 1.0 - abs(ar - 0.75) / 0.75)
    area = w * h
    if area < 0.0005 or area > 0.15:
        ar_score *= 0.1
    return ar_score


def find_locked_segments(common, insts_a, insts_b, iom_th, min_len):
    """Walk the common frames in time order. Return list of (start_idx,
    end_idx) inclusive contiguous runs where every consecutive IoM >= iom_th and
    run length >= min_len. Indices are into `common`."""
    if len(common) < min_len:
        return []
    iom_seq = [iom(box_xyxy(insts_a[k]), box_xyxy(insts_b[k])) for k in common]
    runs = []
    i = 0
    while i < len(common) - min_len + 1:
        if iom_seq[i] >= iom_th:
            j = i + 1
            while j < len(common) and iom_seq[j] >= iom_th:
                j += 1
            if j - i >= min_len:
                runs.append((i, j - 1))
            i = j
        else:
            i += 1
    return runs


def collect_track_insts(mf, pid):
    insts = {}
    for nm, lst in mf["frames"].items():
        for it in lst:
            if int(it["pid"]) == pid:
                insts[nm] = it["box_xywh"]
    return insts


def apply_renames(masks_dir, src_pid, dst_pid, frames, dry_run, overwrite):
    actions = []
    for stem in frames:
        for kind in ("mask", "alpha"):
            old = os.path.join(masks_dir, f"{stem}.p{src_pid:02d}.{kind}.png")
            new = os.path.join(masks_dir, f"{stem}.p{dst_pid:02d}.{kind}.png")
            if not os.path.exists(old):
                actions.append({"kind": "skip_missing", "path": old})
                continue
            if overwrite and os.path.exists(new):
                actions.append({"kind": "overwrite", "path": new})
                if not dry_run:
                    os.remove(new)
            actions.append({"kind": "rename", "from": old, "to": new})
            if not dry_run:
                os.rename(old, new)
    return actions


def apply_deletes(masks_dir, pid, frames, dry_run):
    actions = []
    for stem in frames:
        for kind in ("mask", "alpha"):
            p = os.path.join(masks_dir, f"{stem}.p{pid:02d}.{kind}.png")
            if not os.path.exists(p):
                continue
            actions.append({"kind": "delete", "path": p})
            if not dry_run:
                os.remove(p)
    return actions


def rewrite_manifest(mf, removed_pids, merges, replace_box_overrides):
    """Apply removals and merges to a manifest dict in place.

    replace_box_overrides: {(master_pid, stem): box_xywh} — for low-IoM common
    frames, override the chosen winner's box.
    """
    new_frames = {}
    pid_map = {}
    for m in merges:
        pid_map[m["slave_pid"]] = m["master_pid"]
    removed = set(removed_pids)

    for nm, lst in mf["frames"].items():
        new_lst = []
        for it in lst:
            p = int(it["pid"])
            in_drop = any(p == mg["slave_pid"] and nm in mg["drop_frames"]
                          for mg in merges)
            if p in removed or in_drop:
                continue
            new_it = dict(it)
            if p in pid_map:
                new_it["pid"] = pid_map[p]
                new_pid = pid_map[p]
            else:
                new_pid = p
            if (new_pid, nm) in replace_box_overrides:
                new_it["box_xywh"] = replace_box_overrides[(new_pid, nm)]
            new_lst.append(new_it)
        if new_lst:
            new_frames[nm] = new_lst
    mf["frames"] = new_frames
    mf["dedup"] = {
        "removed_pids": sorted(removed),
        "merges": [
            {
                "master_pid": m["master_pid"],
                "slave_pid": m["slave_pid"],
                "drop_frames_n": len(m["drop_frames"]),
                "rename_frames_n": len(m["rename_frames"]),
                "replace_frames_n": len(m["replace_frames"]),
            } for m in merges
        ],
        "n_box_overrides": len(replace_box_overrides),
    }
    return mf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--masks_dir", required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--report", default="None")
    ap.add_argument("--min_hits", type=int, default=15)
    ap.add_argument("--iom_threshold", type=float, default=0.8)
    ap.add_argument("--min_lock_frames", type=int, default=5)
    ap.add_argument("--likeness_margin", type=float, default=0.2,
                    help="Min score gap between slave/master box to prefer slave")
    ap.add_argument("--out_dir", default="None",
                    help="Where to write the cleaned manifest (default: alongside input)")
    args = ap.parse_args()

    mf = load_manifest(args.manifest)
    all_pids = sorted({int(it["pid"]) for lst in mf["frames"].values() for it in lst})
    track_frames = {p: sorted([nm for nm, lst in mf["frames"].items()
                               if any(int(it["pid"]) == p for it in lst)])
                    for p in all_pids}
    insts = {p: collect_track_insts(mf, p) for p in all_pids}

    report = {
        "input_manifest": args.manifest,
        "input_n_tracks": len(all_pids),
        "min_hits": args.min_hits,
        "iom_threshold": args.iom_threshold,
        "min_lock_frames": args.min_lock_frames,
        "likeness_margin": args.likeness_margin,
        "dry_run": not args.apply,
        "removed_pids": [],
        "merges": [],
        "replace_overrides": [],
        "actions": [],
    }

    # Step 1: min_hits filter
    survivors = [p for p in all_pids if len(track_frames[p]) >= args.min_hits]
    removed_short = [p for p in all_pids if p not in survivors]
    report["removed_pids"].extend(
        [{"pid": p, "reason": f"frames<{args.min_hits}",
          "n_frames": len(track_frames[p])} for p in removed_short])

    # Step 2: pairwise merge detection with union-find for transitive cases
    union_find = {p: p for p in survivors}

    def find(x):
        while union_find[x] != x:
            union_find[x] = union_find[union_find[x]]
            x = union_find[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            union_find[rb] = ra

    raw_merges = []
    for i in range(len(survivors)):
        for j in range(i + 1, len(survivors)):
            a, b = survivors[i], survivors[j]
            common = sorted(set(track_frames[a]) & set(track_frames[b]))
            if len(common) < args.min_lock_frames:
                continue
            runs = find_locked_segments(common, insts[a], insts[b],
                                        args.iom_threshold, args.min_lock_frames)
            if not runs:
                continue
            union(a, b)
            if len(track_frames[a]) >= len(track_frames[b]):
                master, slave = a, b
            else:
                master, slave = b, a
            drop = []
            for (s, e) in runs:
                drop.extend(common[s:e + 1])
            drop_set = set(drop)
            rename = []
            replace = []
            for k in track_frames[slave]:
                if k in drop_set:
                    continue
                if k not in track_frames[master]:
                    rename.append(k)
                else:
                    s_score = face_likeness(insts[slave][k])
                    m_score = face_likeness(insts[master][k])
                    if s_score - m_score > args.likeness_margin:
                        replace.append({"frame": k,
                                       "winner": "slave",
                                       "slave_score": round(s_score, 3),
                                       "master_score": round(m_score, 3),
                                       "slave_box": insts[slave][k],
                                       "master_box": insts[master][k]})
            raw_merges.append({
                "master_pid": master, "slave_pid": slave,
                "drop_frames": sorted(drop_set),
                "rename_frames": sorted(rename),
                "replace_frames": replace,
                "locked_runs": runs,
            })

    # Collapse transitive merges
    final = {}
    for m in raw_merges:
        root = find(m["master_pid"])
        m["master_pid"] = root
        key = (root, m["slave_pid"])
        if key not in final:
            final[key] = {"master_pid": root, "slave_pid": m["slave_pid"],
                          "drop_frames": [], "rename_frames": [],
                          "replace_frames": [], "locked_runs": []}
        final[key]["drop_frames"].extend(m["drop_frames"])
        final[key]["rename_frames"].extend(m["rename_frames"])
        final[key]["replace_frames"].extend(m["replace_frames"])
        final[key]["locked_runs"].extend(m["locked_runs"])
    merges = list(final.values())

    # Track replace box overrides separately
    replace_overrides = {}  # (master_pid, frame) -> new box
    for m in merges:
        for r in m["replace_frames"]:
            replace_overrides[(m["master_pid"], r["frame"])] = r["slave_box"]
            report["replace_overrides"].append({
                "master_pid": m["master_pid"], "frame": r["frame"],
                "winner": r["winner"],
                "slave_score": r["slave_score"],
                "master_score": r["master_score"],
            })

    # Drop merges whose slave's truly unique (rename) frame count is below
    # min_hits — replace_frames (low-IoM common frames) carry unique correct
    # data and should be retained regardless of count.
    kept_merges = []
    for m in merges:
        unique = len(m["rename_frames"])
        if unique < args.min_hits:
            report["removed_pids"].append(
                {"pid": m["slave_pid"], "reason": "id_switch_below_min_hits",
                 "n_frames_kept_unique": unique, "merged_into": m["master_pid"]})
            # Demote unique frames into drop; keep replace_frames (they carry
            # unique data even if rare).
            m["drop_frames"].extend(m["rename_frames"])
            m["rename_frames"] = []
        kept_merges.append(m)
    merges = kept_merges

    # Apply filesystem actions
    if not args.apply:
        for p in removed_short:
            report["actions"].extend(
                apply_deletes(args.masks_dir, p, track_frames[p], dry_run=True))
        for m in merges:
            if m["rename_frames"]:
                report["actions"].extend(
                    apply_renames(args.masks_dir, m["slave_pid"], m["master_pid"],
                                  m["rename_frames"], dry_run=True, overwrite=False))
            replace_frames = [r["frame"] for r in m["replace_frames"]]
            if replace_frames:
                report["actions"].extend(
                    apply_renames(args.masks_dir, m["slave_pid"], m["master_pid"],
                                  replace_frames, dry_run=True, overwrite=True))
            if m["drop_frames"]:
                report["actions"].extend(
                    apply_deletes(args.masks_dir, m["slave_pid"], m["drop_frames"],
                                  dry_run=True))
    else:
        # Auto-backup: copy all slave pid's mask/alpha files and the original
        # manifest into a timestamped subdirectory so the user can roll back.
        backup_root = os.path.join(args.masks_dir,
                                   "_backup_pre_dedup_" + time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(backup_root, exist_ok=True)
        shutil.copy2(args.manifest, os.path.join(backup_root, "masks_manifest.json"))
        for p in removed_short + [m["slave_pid"] for m in merges]:
            for stem in track_frames[p]:
                for kind in ("mask", "alpha"):
                    src = os.path.join(args.masks_dir, f"{stem}.p{p:02d}.{kind}.png")
                    if os.path.exists(src):
                        shutil.copy2(src, os.path.join(backup_root, os.path.basename(src)))
        report["backup_dir"] = backup_root

        for p in removed_short:
            report["actions"].extend(
                apply_deletes(args.masks_dir, p, track_frames[p], dry_run=False))
        for m in merges:
            if m["rename_frames"]:
                report["actions"].extend(
                    apply_renames(args.masks_dir, m["slave_pid"], m["master_pid"],
                                  m["rename_frames"], dry_run=False, overwrite=False))
            replace_frames = [r["frame"] for r in m["replace_frames"]]
            if replace_frames:
                report["actions"].extend(
                    apply_renames(args.masks_dir, m["slave_pid"], m["master_pid"],
                                  replace_frames, dry_run=False, overwrite=True))
            if m["drop_frames"]:
                report["actions"].extend(
                    apply_deletes(args.masks_dir, m["slave_pid"], m["drop_frames"],
                                  dry_run=False))
        out_dir = args.out_dir or os.path.dirname(args.manifest)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "masks_manifest_cleaned.json")
        rewrite_manifest(mf, removed_short, merges, replace_overrides)
        save_manifest(out_path, mf)
        report["cleaned_manifest"] = out_path

    # Human summary
    print("=" * 70)
    print("SAM3 track dedup %s" % ("DRY-RUN" if not args.apply else "APPLY"))
    print("=" * 70)
    print("input tracks: %d  (pids: %s)" % (len(all_pids), all_pids))
    print("survivors by min_hits>=%d: %d  (pids: %s)"
          % (args.min_hits, len(survivors), survivors))
    if report["removed_pids"]:
        print("\nremoved:")
        for r in report["removed_pids"]:
            n = r.get("n_frames", r.get("n_frames_kept_unique", 0))
            extra = ""
            if "merged_into" in r:
                extra = "  merged_into=%d" % r["merged_into"]
                if r.get("reason") == "id_switch_below_min_hits":
                    # See if any replace_frames for this master reused slave data
                    re_used = sum(1 for ov in report["replace_overrides"]
                                  if ov["master_pid"] == r["merged_into"])
                    if re_used:
                        extra += "  (%d box/mask overrides kept from this track)" % re_used
            print("  pid=%d  reason=%s  frames=%d%s"
                  % (r["pid"], r["reason"], n, extra))
    if report.get("backup_dir"):
        print("\nbackup dir: %s" % report["backup_dir"])
    if merges:
        print("\nmerges:")
        for m in merges:
            print("  pid=%d -> pid=%d  locked=%d  drop=%d  rename=%d  replace(overwrite)=%d"
                  % (m["slave_pid"], m["master_pid"],
                     len(m["locked_runs"]),
                     len(m["drop_frames"]),
                     len(m["rename_frames"]),
                     len(m["replace_frames"])))
    if report["replace_overrides"]:
        print("\nbox overrides (low-IoM common frames; master.box -> slave.box):")
        for r in report["replace_overrides"][:8]:
            print("  pid=%d  frame=%s  scores master=%.3f slave=%.3f"
                  % (r["master_pid"], r["frame"][-6:], r["master_score"], r["slave_score"]))
        if len(report["replace_overrides"]) > 8:
            print("  ... and %d more" % (len(report["replace_overrides"]) - 8))
    print("\nfilesystem actions: %d" % len(report["actions"]))

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print("report saved:", args.report)


if __name__ == "__main__":
    main()