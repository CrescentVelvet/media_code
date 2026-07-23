#!/usr/bin/env python3
"""Generate a derived F3G-Avatar test config from the example YAML.

Loads configs/avatarrex_zzr/avatar.yaml, overrides the `test` (and the
`train.data` that test() reuses for smpl_shape/PCA) sections from env vars,
and writes a derived YAML. The official main_avatar.py then runs:
    python main_avatar.py -c <derived.yaml> -m test

Why a derived config (not editing the official file): main_avatar.py has no
CLI flags for test paths / view setting — everything comes from the YAML. This
script fills the YAML from env (set by 02_run_inference.sh) without touching the
cloned repo. Mirrors hypir's gen_train_config.py approach.

test() ALWAYS builds `training_dataset = MvRgbDataset(**train.data, training=False)`
(to read SMPL betas + PCA), so train.data.data_dir MUST point at the prepared
capture (has smpl_params.npz + smpl_pos_map/). test.pose_data (optional) drives
free-view animation; if absent, test.data renders training frames.

Env vars (set by 02_run_inference.sh):
  BASE_CONFIG, F3G_DIR, DATA_DIR, SUBJECT_NAME, DATA_FRAME_RANGE,
  PREV_CKPT, POSE_DATA, POSE_FRAME_RANGE, VIEW_SETTING, RENDER_VIEW_IDX,
  IMG_SCALE, SAVE_PLY, SAVE_TEX_MAP, N_PCA, SIGMA_PCA, GLOBAL_ORIENT,
  OUTPUT_DIR, OUT_CONFIG
"""
import os
import sys
import yaml


def parse_range(s):
    """'0,2001,1' -> [0, 2001, 1]; '2000,2500' -> [2000, 2500]."""
    parts = [int(x.strip()) for x in s.split(",") if x.strip() != ""]
    return parts


def bval(name, default=None):
    """Read an env var as a bool-ish (SAVE_PLY=1 -> True)."""
    v = os.environ.get(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "on")


def main():
    base = os.environ.get("BASE_CONFIG")
    if not base or not os.path.isfile(base):
        sys.exit(f"ERROR: BASE_CONFIG not found: {base!r}")
    data_dir = os.environ.get("DATA_DIR")
    prev_ckpt = os.environ.get("PREV_CKPT")
    if not data_dir:
        sys.exit("ERROR: DATA_DIR not set (prepared multiview dataset with smpl_params + smpl_pos_map).")
    if not prev_ckpt:
        sys.exit("ERROR: PREV_CKPT not set (a checkpoint dir containing net.pt).")

    with open(base, encoding="UTF-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader) or {}

    subject = os.environ.get("SUBJECT_NAME") or os.path.basename(os.path.normpath(data_dir))

    # --- train.data: test() reuses it for smpl_shape + PCA; must be the capture ---
    cfg.setdefault("train", {}).setdefault("data", {})
    cfg["train"]["data"]["data_dir"] = data_dir
    cfg["train"]["data"]["subject_name"] = subject
    if os.environ.get("DATA_FRAME_RANGE"):
        cfg["train"]["data"]["frame_range"] = parse_range(os.environ["DATA_FRAME_RANGE"])
    cfg["train"]["data"].setdefault("load_smpl_pos_map", True)

    # --- test section ---
    cfg.setdefault("test", {})
    test = cfg["test"]
    test["prev_ckpt"] = prev_ckpt
    if os.environ.get("OUTPUT_DIR"):
        test["output_dir"] = os.environ["OUTPUT_DIR"]
    test["view_setting"] = os.environ.get("VIEW_SETTING", test.get("view_setting", "free"))
    if os.environ.get("RENDER_VIEW_IDX"):
        test["render_view_idx"] = int(os.environ["RENDER_VIEW_IDX"])
    if os.environ.get("IMG_SCALE"):
        test["img_scale"] = float(os.environ["IMG_SCALE"])
    test["save_ply"] = bval("SAVE_PLY", test.get("save_ply", False))
    test["save_tex_map"] = bval("SAVE_TEX_MAP", test.get("save_tex_map", False))
    if os.environ.get("N_PCA") is not None:
        test["n_pca"] = int(os.environ["N_PCA"])
    if os.environ.get("SIGMA_PCA") is not None:
        test["sigma_pca"] = float(os.environ["SIGMA_PCA"])
    if os.environ.get("GLOBAL_ORIENT") is not None:
        test["global_orient"] = bval("GLOBAL_ORIENT", test.get("global_orient", True))

    pose_data = os.environ.get("POSE_DATA")
    if pose_data:
        # Free-view animation from an external pose sequence.
        pd = {"data_path": pose_data}
        if os.environ.get("POSE_FRAME_RANGE"):
            pd["frame_range"] = parse_range(os.environ["POSE_FRAME_RANGE"])
        elif "frame_range" in test.get("pose_data", {}):
            pd["frame_range"] = test["pose_data"]["frame_range"]
        test["pose_data"] = pd
    else:
        # Render frames from the training capture (default camera view etc.).
        # Drop any pose_data left in the base YAML so test() uses test.data.
        test.pop("pose_data", None)
        test.setdefault("data", {})
        test["data"]["data_dir"] = data_dir
        test["data"]["subject_name"] = subject
        if os.environ.get("DATA_FRAME_RANGE"):
            test["data"]["frame_range"] = parse_range(os.environ["DATA_FRAME_RANGE"])
        elif "frame_range" not in test["data"]:
            test["data"]["frame_range"] = [0, 500]

    cfg["mode"] = "test"

    out = os.environ.get("OUT_CONFIG")
    if not out:
        out = os.path.join(os.path.dirname(base), "_test_derived.yaml")
    with open(out, "w", encoding="UTF-8") as f:
        yaml.dump(cfg, f, sort_keys=False, allow_unicode=True)
    print(f"[*] derived test config written: {out}")
    print(f"    mode={cfg['mode']}  prev_ckpt={test['prev_ckpt']}")
    print(f"    train.data.data_dir={cfg['train']['data']['data_dir']}")
    print(f"    view_setting={test['view_setting']}  n_pca={test.get('n_pca')}  sigma_pca={test.get('sigma_pca')}")
    if "pose_data" in test:
        print(f"    pose_data={test['pose_data']}")
    else:
        print(f"    test.data.data_dir={test['data']['data_dir']}")


if __name__ == "__main__":
    main()
