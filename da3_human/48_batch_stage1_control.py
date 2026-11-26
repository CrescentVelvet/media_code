#!/usr/bin/env python3
"""单模型串行运行旧17例8源2留出，避免多进程争抢24GB显存。

Env vars: ROOT, DA3_MODEL_DIR, RESULTS_ROOT, PROCESS_RES=704, CASE_FILTER
"""
import importlib.util
import os
from pathlib import Path


def main():
    root = Path(os.environ["ROOT"])
    selected = os.environ.get("CASE_FILTER", "")
    cases = [root / selected] if selected else sorted(root.glob("case*_nopose_recentered"),
             key=lambda p: int(p.name.split("_")[0].removeprefix("case")))
    if not cases:
        raise FileNotFoundError(f"无旧 stage1 案例: {root}")
    out_root = Path(os.environ["RESULTS_ROOT"])
    if out_root.exists() or out_root.is_symlink():
        raise FileExistsError(out_root)
    out_root.mkdir(parents=True)
    spec = importlib.util.spec_from_file_location("control", Path(__file__).with_name("47_run_stage1_control.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    os.environ["HOLDOUT"] = "1"
    os.environ["BRANCHES"] = "rgb,pose"
    for case in cases:
        os.environ["CASE_FILTER"] = case.name
        os.environ["RESULTS_DIR"] = str(out_root / case.name)
        module.main()


if __name__ == "__main__":
    main()
