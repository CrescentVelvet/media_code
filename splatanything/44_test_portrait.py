#!/usr/bin/env python3
"""Offline contract tests for portrait preparation, camera and summary code."""
import argparse
import csv
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import importlib
C = importlib.import_module("39_portrait_common")
E = importlib.import_module("42_eval_portrait")


def test_depth_camera():
    radial = np.full((3, 3), 2., np.float32)
    K = np.array([[2., 0., 1.], [0., 2., 1.], [0., 0., 1.]])
    expected = radial / np.sqrt(
        1. + (np.array([-1., 0., 1.]) / 2.) ** 2 +
        (np.array([-1., 0., 1.])[:, None] / 2.) ** 2)
    got = C.radial_to_axial(radial, K)
    assert np.allclose(got, expected)
    assert np.isclose(got[1, 2], 2. / np.sqrt(1.25))
    # Nonzero rotation and scale ensure [sR|T] is not accepted as a camera.
    angle = .31
    R = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                  [-np.sin(angle), 0., np.cos(angle)]])
    fit = {"s": 1.73, "R": R, "T": np.array([.2, -.3, .7])}
    source_pose = np.eye(4)
    source_pose[:3, :3] = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    source_pose[:3, 3] = np.array([3., -2., 5.])
    source_first_w2c = np.linalg.inv(source_pose.astype(np.float64))
    source_views = {"camera_poses": torch.eye(4)[None, None],
                    "source_first_w2c": torch.from_numpy(
                        source_first_w2c.astype(np.float32))}
    # Also accept the model's actual [B,4,4] saved first-source inverse.
    target = np.eye(4)
    target[:3, :3] = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    target[:3, 3] = np.array([1.1, -.4, 2.3])
    raw = C.package_to_source_camera(target, source_views).numpy()
    assert np.allclose(raw, np.linalg.inv(source_pose) @ target, atol=2e-6)
    assert np.allclose(source_first_w2c @ source_pose, np.eye(4))
    # Nontrivial source alignment regression: R is source->model. Camera
    # rotation composition is R_model=R@R_source, not the inverse operation.
    exported = np.tile(np.eye(4), (3, 1, 1))
    q2 = np.array([[np.cos(.47), -np.sin(.47), 0.],
                   [np.sin(.47), np.cos(.47), 0.], [0., 0., 1.]])
    prior_cameras = []
    for rotation, center in zip((np.eye(3), R.T, q2),
                                ([0., 0., 0.], [1., 0., 0.], [.2, 2., .3])):
        source = np.eye(4)
        source[:3, :3], source[:3, 3] = rotation, center
        prior_cameras.append(source)
        expected = C.target_to_model(source, fit, "cpu").numpy()
        exported[len(prior_cameras) - 1] = expected
    source_views["camera_poses"] = torch.from_numpy(np.stack(prior_cameras))[None]
    recovered_fit = C.fit_source_sim3(source_views, torch.from_numpy(exported))
    assert np.allclose(recovered_fit["R"], R, atol=2e-8), (R, recovered_fit["R"])
    assert abs(recovered_fit["s"] - fit["s"]) < 1e-6, (fit["s"], recovered_fit["s"])
    assert np.allclose(recovered_fit["T"], fit["T"], atol=1e-6), (fit["T"], recovered_fit["T"])
    source_views["camera_poses"] = torch.eye(4)[None, None]
    model = C.target_to_model(raw, fit, "cpu").numpy()
    assert np.allclose(model[:3, :3], R @ raw[:3, :3])
    assert np.allclose(model[:3, :3] @ model[:3, :3].T, np.eye(3))
    assert np.allclose(model[:3, 3], fit["s"] * R @ raw[:3, 3] + fit["T"])
    inv = C.model_to_package_camera(model, fit, "cpu").numpy()
    assert np.allclose(inv, raw, atol=2e-6)
    recovered_raw = np.linalg.inv(source_views["source_first_w2c"].numpy().astype(np.float64)) @ inv
    assert np.allclose(recovered_raw, target, atol=2e-6)
    batch = C.target_camera_batch({"c2w": torch.from_numpy(target)[None]},
                                  fit, "cpu", source_views)[0].numpy()
    assert np.allclose(batch, model, atol=2e-6)
    # Actual batched tensor input and saved [B,4,4] source inverse.
    assert np.allclose(
        C.package_to_source_camera(target, {"source_first_w2c": torch.from_numpy(
            source_first_w2c.astype(np.float32))[None]}).numpy(),
        raw, atol=2e-6)


def test_manifest():
    # Real-protocol adapter is checked separately when frozen assets exist.
    proto = Path("/mnt/d/output/results/splatanything_results/n1_real_20261122/protocol_v1/protocol.json")
    report = proto.parent.parent / "real_eval_v18/n1_real_eval.json"
    if proto.is_file() and report.is_file():
        rows = E.real_records(C.read_json(proto), C.read_json(report))
        assert len(rows) == 2 and all(len(r["targets"]) == 2 for r in rows)
        for row in rows:
            for t in row["targets"]:
                assert t["c2w_np"].shape == (4, 4)
                assert t["gt"].shape[:2] == (756, 756) and t["gt"].shape[-1] == 3


def test_masked_loss_backward():
    person = torch.ones((1, 4, 4), dtype=torch.bool)
    rd = {"rgb": torch.ones((1, 4, 4, 3), requires_grad=True),
          "alpha": torch.ones((1, 4, 4), requires_grad=True),
          "depth": torch.ones((1, 4, 4), requires_grad=True)}
    target = {"gt": torch.ones_like(rd["rgb"]), "person": person,
              "valid_depth": person, "depth_z_m": torch.ones_like(rd["depth"])}
    fit = {"s": 1.}
    out = C.masked_losses(rd, target, fit)
    assert person.any() and torch.allclose(target["depth_z_m"], rd["depth"].detach())
    assert out["loss"].requires_grad
    out["loss"].backward()
    assert rd["rgb"].grad is not None and torch.isfinite(rd["rgb"].grad).all()


def test_source_cache_isolation(out):
    class Teacher:
        def __call__(self, views, cond_flags):
            return {"depth": torch.ones((1, 3, 2, 2)), "views_seen": len(views["img"][0])}

    class Model:
        def encode(self, views, flags):
            return views["img"].reshape(1, 3, -1), 0

    res = (8, 8)
    records = []
    for i in range(3):
        path = out / ("source_%d.png" % i)
        Image.new("RGB", res, (i * 50 + 20, 70, 100)).save(path)
        records.append({"cid": str(i), "rgb_path": path,
                        "K_np": np.array([[5., 0., 3.5], [0., 5., 3.5], [0., 0., 1.]], np.float32),
                        "c2w_np": np.eye(4, dtype=np.float32), "resolution_wh": res})
    cache = C.source_cache(Model(), Teacher(), records, "cpu")
    assert set(cache) == cache["allowed_keys"]
    assert not any(k.startswith("target") or k in {"gt", "person", "valid_depth"} for k in cache)
    assert cache["source_ids"] == ["0", "1", "2"]
    assert cache["geometry"]["views_seen"] == 3
    assert cache["tokens"].shape[1] == 3


def test_checkpoint_restore_and_order(out):
    args = types.SimpleNamespace(lr=1e-4, res=504)
    p = torch.nn.Parameter(torch.tensor([1.25]))
    opt = torch.optim.AdamW([p], lr=1e-4)
    p.grad = torch.ones_like(p)
    opt.step()
    with torch.no_grad():
        p.copy_(torch.tensor([3.5]))
    C.save_checkpoint(out / "step_0001.pt", 1, [("head.weight", p)], opt, args, "manifest.json")
    (out / "B0.pt").write_bytes((out / "step_0001.pt").read_bytes())
    assert [p.name for p in C.checkpoint_names(out)] == ["B0.pt", "step_0001.pt"]
    with torch.no_grad():
        p.zero_()
    class Head(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.head = torch.nn.Module()
            self.head.weight = weight
    blob = C.load_trainable(Head(p), out / "step_0001.pt")
    assert blob["step"] == 1 and float(p) == 3.5
    opt2 = torch.optim.AdamW([p], lr=1e-4)
    opt2.load_state_dict(blob["optimizer"])
    assert opt2.param_groups[0]["lr"] == args.lr
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--official-repo", default=os.environ.get("SPLATANYTHING_DIR", ""))
    p.add_argument("--out", required=True)
    p.add_argument("--plot-smoke", action="store_true")
    args = p.parse_args(argv)
    if args.official_repo and args.official_repo not in sys.path:
        sys.path.insert(0, args.official_repo)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    print("TEST depth/camera", flush=True)
    test_depth_camera()
    print("TEST masked-loss backward", flush=True)
    test_masked_loss_backward()
    print("TEST source-cache isolation", flush=True)
    test_source_cache_isolation(out)
    print("TEST checkpoint restore/order", flush=True)
    test_checkpoint_restore_and_order(out)
    print("TEST real-target adapter", flush=True)
    test_manifest()
    result = {"schema": "sa_portrait_offline_contract_v1", "status": "passed",
              "radial_to_axial": "analytic 3x3 test passed",
              "first_source_alignment": "P0^-1 target with [4,4] and [B,4,4] saved inverse",
              "sim3": "source->model camera fit, orthonormal rotation and inverse roundtrip passed",
              "source_cache_isolation": "three source views only; no target RGB/depth/mask keys",
              "checkpoint": "B0-first order and trainable/optimizer restoration",
              "real_target_adapter": "four targets parsed if local frozen assets exist",
              "no_training_or_weight_update": True}
    if args.plot_smoke:
        with open(out / "metrics.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for i in range(1, 11):
                f.write(json.dumps({"event": "step", "step": i, "step_s": 2. + i * .01,
                                    "loss": 1. / i, "rgb_person": .5 / i,
                                    "depth_loss_m2": .02 / i, "alpha_loss": .1 / i}) + "\n")
        import subprocess
        subprocess.check_call([sys.executable, str(HERE / "41_plot_portrait.py"), "--run", str(out)])
        for name in ("loss_curves.svg", "loss_curves.png"):
            assert (out / name).stat().st_size > 1000
        result["plot"] = "ten synthetic JSONL rows rendered as SVG and PNG; not training evidence"
    C.dump_json(out / "contracts.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
