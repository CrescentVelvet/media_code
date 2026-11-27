#!/usr/bin/env python3
"""DA3 相机条件的矩阵级诊断；只读输入，诊断输出写 audit 子目录。"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from importlib.util import module_from_spec, spec_from_file_location  # noqa: E402

_spec = spec_from_file_location("camera", SCRIPT_DIR / "36_run_hisilicon_cameras.py")
_cam = module_from_spec(_spec)
_spec.loader.exec_module(_cam)


def main():
    root = Path(os.environ.get("ROOT", "/mnt/d/dataset/测试数据/海思端侧重建拍摄数据"))
    name = os.environ.get("CASE_FILTER", "model_man")
    case = root / name
    out = case / "depthanything3_camera_audit/matrix"
    out.mkdir(parents=True, exist_ok=True)
    files, ext, ixt, _, _ = _cam.load_gltf_inputs(case)
    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.utils import gsply_helpers as gh
    gh.export_ply = _cam._io.fast_export_ply
    model = DepthAnything3.from_pretrained(
        os.environ.get("DA3_MODEL_DIR", "/mnt/d/wheel/da3_human_ms")).to("cuda").eval()
    imgs_cpu, ext_cpu, ixt_cpu = model._preprocess_inputs(
        [str(p) for p in files], extrinsics=ext, intrinsics=ixt,
        process_res=504, process_res_method="upper_bound_resize")
    imgs, ext_t, ixt_t = model._prepare_model_inputs(imgs_cpu, ext_cpu, ixt_cpu)
    # 从 DA3 自身工具构造等价的 w2c，避免 shape 或手动转置歧义。
    from depth_anything_3.utils.geometry import affine_inverse
    c2w = affine_inverse(ext_t)
    w2c_from_c2w = affine_inverse(c2w)
    branches = {
        "rgb_only": (None, None),
        "current_w2c": (ext_t, ixt_t),
        # 仅诊断模型是否实际期待 c2w；由工具互逆，数值闭环必过。
        "c2w_alternative": (c2w, ixt_t),
    }
    results = []
    for label, (ex, ix) in branches.items():
        print("RUN", label, "EXT", None if ex is None else ex.shape, flush=True)
        norm = model._normalize_extrinsics(ex.clone() if ex is not None else None)
        raw = model._run_model_forward(imgs, norm, ix, [], True, False, "saddle_balanced")
        pred = model._convert_to_prediction(raw)
        print("RAW", label, "depth", np.quantile(pred.depth, [.1, .5, .9]).tolist(),
              "ext_shape", pred.extrinsics.shape, "scale", pred.scale_factor, flush=True)
        path = out / f"{label}.ply"
        gh.save_gaussian_ply(
            gaussians=pred.gaussians, save_path=str(path),
            ctx_depth=torch.from_numpy(pred.depth).unsqueeze(-1).to(pred.gaussians.means),
            shift_and_scale=False, save_sh_dc_only=False, gs_views_interval=1,
            inv_opacity=True, prune_by_depth_percent=1.0, prune_border_gs=False,
            match_3dgs_mcmc_dev=False)
        np.savez_compressed(out / f"{label}.npz", extrinsics=pred.extrinsics,
                            intrinsics=pred.intrinsics, depth=pred.depth,
                            image_names=np.asarray([p.name for p in files]))
        flat_means = pred.gaussians.means.detach().reshape(-1, 3)
        results.append({"name": label, "depth_quantiles": np.quantile(pred.depth, [.1, .5, .9]).tolist(),
                        "scale_factor": pred.scale_factor,
                        "mean_bounds": [flat_means.amin(dim=0).tolist(),
                                        flat_means.amax(dim=0).tolist()]})
        del raw, pred, norm
        torch.cuda.empty_cache()
    _cam._io.dump_json(out / "summary.json", {"case": name, "branches": results})


if __name__ == "__main__":
    main()
