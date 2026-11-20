#!/usr/bin/env python3
"""生成同一佳佳三输入、512² 留出协议的 ZipSplat 对照配置。

Env vars:
  OUT_DIR（结果根，默认 ~/output/zipsplat_results）
  CONFIG_OUT（默认结果根/eval_3view_r512_black/extra_methods.json）
"""
import json
import hashlib
import os
from pathlib import Path

ROOT = Path(os.environ.get("OUT_DIR", "~/output/zipsplat_results")).expanduser()
OUT = Path(os.environ.get("CONFIG_OUT", ROOT / "eval_3view_r512_black" / "extra_methods.json"))
BASE3 = os.path.expanduser("~/output/infinisplat_v2_results/_captures/佳佳抱枕_3v/cameras.npz")
REFERENCE_IMAGES = Path(BASE3).parent / "images"


def main():
    expected = [
        "000_5093309607791922.png",
        "001_5093317707502922.png",
        "002_5093325907213922.png",
    ]
    cases = []
    # 不只凭同名 PNG 认定同输入：历史转换与本轮转换必须逐字节一致。
    for name in expected:
        converted = ROOT / "_captures/jiajia_3v_compare/images" / name
        reference = REFERENCE_IMAGES / name
        assert hashlib.sha256(converted.read_bytes()).digest() == hashlib.sha256(
            reference.read_bytes()).digest(), f"三输入内容不一致: {name}"
    for label, scene, ply_name in [
        ("ZipSplat FF", "jiajia_3v_compare_feedforward", "gaussians_feedforward.ply"),
        ("ZipSplat TokenBA", "jiajia_3v_compare", "gaussians.ply"),
    ]:
        result = ROOT / scene
        run = json.loads((result / "metrics.json").read_text(encoding="utf-8"))
        assert run["image_files"] == expected, (scene, run["image_files"])
        assert run["num_views"] == 3 and run["num_gaussians"] == 46656
        assert run["token_ba"] == (label.endswith("TokenBA"))
        cameras = result / "predicted_cameras.npz"
        ply = result / ply_name
        assert cameras.is_file() and ply.is_file()
        assert (ROOT / "_captures/jiajia_3v_compare/cameras.npz").is_file()
        assert Path(BASE3).is_file()
        cases.append(dict(name=label, ply=str(ply), cams=["c2w", str(cameras)]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    print(f"✅ {OUT}: {len(cases)} 分支，输入 0/40/81，ZipSplat 无位姿输入")


if __name__ == "__main__":
    main()
