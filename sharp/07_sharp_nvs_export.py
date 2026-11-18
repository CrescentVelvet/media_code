#!/usr/bin/env python3
"""07_sharp_nvs_export.py — 把 SHARP 前馈结果导出成 SA `08_nvs_eval.py` 能直接吃的格式。

**为什么需要它（v4 §4 第 3 条，2026-11-18）**

v4 要求「SA 相同 target 配对」。要判「SHARP 与 SA 谁在**同一个留出视角**上更好」，
唯一干净的做法是**让两者走同一个评测器**，而不是各写一套再比数字。此前文档里那句
「SHARP 11~17 dB，SA 自渲 26~27 dB，差一个量级」是错的两次：26~27 其实是 SHARP
**自己的同图自渲**（`sharp/EXPERIMENTS.md:472`），而 SA 的真实 held-out NVS 是
15.23 / 17.81 / 18.39 dB。两者分辨率、裁剪、聚合、底色全不同，根本不能相减。

本脚本只做一件事：**把 SHARP 的高斯摆到 SA 导出的那个坐标系里，并产出同款
`cameras.json`**，之后交给 `08_nvs_eval.py` 用完全相同的代码路径打分。

**为什么不吃 `02_run_inference.sh` 的现成 ply（两个坑，都会静默毁掉对照）**

1. **焦距**：`02` 走官方 CLI，**不传焦距**，走 30mm 兜底 → 该分支的 `f_px = 1663.13`；
   而 `prior_camera.json` 的真值是 `fx=1380.82`，**偏大 20.4%**。焦距直接决定深度尺度，
   偏 20% 就是整片云整体缩放到另一个尺度，PSNR 没有任何意义。
2. **主点**：官方 `save_ply` 的 `intrinsic` element 恒写 `w/2, h/2`，
   而真实主点是 `cx=719.5, cy=959.5`（图 1439×1919，即 `w/2=719.5, h/2=959.5`——
   本场景恰好相等，但不能依赖这一点）。

所以本脚本**直接复用 `05_cross_view_consistency.predict()`**（真 `(fx,fy,cx,cy)` 分离、
`disparity_factor` 仍喂 fx 以保持网络行为），保证与 SH1a 完全同一口径。

**坐标准一怎么保证（最容易错，所以显式做，不靠假设）**

SHARP 单图前馈的输出住在「该图自己的相机系」（OpenCV: x 右 y 下 z 前）。
`prior_camera.json` 的 `extrinsics` 是**真实 c2w**（见 `05:97` 与
`SplatAnything/src/utils/camera_io.py:49 "Read OpenCV c2w poses"`）。
取第 0 个 source 视角的相机系为公共系，则：

    p_a = inv(c2w_a) @ c2w_i @ p_i        # 与 05 的 `T = inv(c2w_a) @ c2w_i` 同一式

对 i=a 就是恒等，自动对齐——**不需要拟合任何 M/s/u**，也没有尺度自由度可调。
这正是 v4「不得用 target 校准尺度」的要求：本脚本从头到尾**只读 source 视角**，
`cameras.json` 的内参直接取 `prior_camera.json` 的真 K，不看 target 一眼。

**颜色空间：必须做 `linearRGB2sRGB`（首版判错了，2026-11-18 实测纠正）**

SHARP 的高斯颜色是**线性 RGB**：CLI 里 `SceneMetaData(..., "linearRGB")` 写得很清楚，
官方 `save_ply` 因此在导出前先 `linearRGB2sRGB(colors)` 再编码 SH。

首版曾以「SA 的 `03_render_eval.render()` 不做 sRGB 转换」为由**跳过**这一步，
理由是错的：关键不在「谁做转换」，而在 **SHARP 与 SA 的颜色本来就不在同一空间**。
SA 直接对 sRGB 图像做优化，它的 `f_dc` 编码的就是 sRGB 值；把 SHARP 的 linear 值
原样塞进 `f_dc`，渲染必然整体偏暗，而 GT 是 sRGB → 全图误差 >0.25，diff 图整片饱和。

实测（`diag_color.py`，4 个留出视角）：

| | GT 均值 | 渲染均值 | PSNR |
|---|---:|---:|---:|
| 原样写 linear | 0.471 | **0.283** | 11.38 |
| 做 linearRGB2sRGB | 0.471 | 0.521 | **14.31** |

修后每个视角 **+2.5~+2.9 dB**。所以本脚本做 `linearRGB2sRGB` 后编码 SH，
与官方 `save_ply` 一致，也与 SA 的 sRGB 口径对齐。

⚠️ 严格说 `save_ply` 的做法（先在 sRGB 空间编码 SH、再由光栅器线性混合）不是
物理正确的合成顺序（应先线性混合再 gamma）。但**SA 自己就是这么做的**，
所以为「同一口径对比」只能跟它一致；要改就得两边一起改，属另一个实验。

**输出**

    OUT/
      gaussians.ply     # 所有 source 视角的 SHARP 高斯并联，落在 c2w_a 相机系
      cameras.json      # source 视角在公共系的 c2w + 真 K（供 08 反算对齐、核验零残差）
      export_audit.json # 每视角点数 / 变换 / 尺度核验 + 焦距口径，供人工复核

之后（**同一个评测器**，这才叫同口径）：

    RESULT=OUT SOURCE_SCENE=<scenes>/jiajia_hold_3v ALL_SCENE=<scenes>/jiajia_hold \\
      bash splatanything/08_nvs_eval.sh

⚠️ **两个必须写明的局限**

1. SHARP 是单图模型，N 张图是 N 份**互不相干**的局部云，并联不会有跨视角一致性
   （这正是 SH1a 测的东西）。本脚本并联只是为了让它被**同一把尺子**量，
   **不声称 SHARP 具备多图融合能力**。
2. `export_audit.json` 记了各视角的 `z_median_cam`，`z_median_spread_ratio`
   就是并联云的固有错位量。它越大，PSNR 越低——**这个量必须与 PSNR 一起报**，
   否则会把「尺度没对上」误读成「SHARP 质量差」。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def load_camera_json(path: Path):
    """prior_camera.json -> (ids, c2w[N,4,4], K[N,3,3])。与 05 的同名函数同语义。

    `extrinsics` 是 OpenCV **c2w**（不是 w2c），`intrinsics` 是 3x3 像素内参。
    """
    data = json.loads(Path(path).read_text())
    ext = {e["camera_id"]: np.asarray(e["matrix"], np.float64) for e in data["extrinsics"]}
    itr = {e["camera_id"]: np.asarray(e["matrix"], np.float64) for e in data["intrinsics"]}
    ids = [e["camera_id"] for e in data["extrinsics"] if e["camera_id"] in itr]
    return ids, np.stack([ext[i] for i in ids]), np.stack([itr[i] for i in ids])


def pick_views(ids, want: str):
    """把 `000,004,022` 这类唯一前缀解析成全 camera_id；空则返回 None。"""
    if not want:
        return None
    out = []
    for w in [v.strip() for v in want.split(",") if v.strip()]:
        hit = [cid for cid in ids if cid == w or cid.startswith(w)]
        if len(hit) != 1:
            raise SystemExit(f"--views 的 {w!r} 匹配到 {len(hit)} 个：{hit}；请给更长的前缀或全名")
        out.append(hit[0])
    return out


def gaussians_to_arrays(g):
    """Gaussians3D -> 3DGS PLY 的五个数组（语义与官方 save_ply 逐字段一致）。

    `singular_values` 是**真实尺度**（不是 log），`opacities` 是**概率**（不是 logit），
    写盘时分别取 log 与 logit。`colors` 是 **linearRGB**，必须先 `linearRGB2sRGB`
    再编码 SH 直流（理由见模块 docstring「颜色空间」一节，实测 +2.5~2.9 dB）：

        f_dc = (sRGB(rgb) - 0.5) / 0.28209479177387814
    """
    from sharp.utils.color_space import linearRGB2sRGB
    from sharp.utils.gaussians import convert_rgb_to_spherical_harmonics

    xyz = g.mean_vectors.reshape(-1, 3).float()
    scales = g.singular_values.reshape(-1, 3).float()
    quats = g.quaternions.reshape(-1, 4).float()
    opac = g.opacities.reshape(-1).float()
    rgb_srgb = linearRGB2sRGB(g.colors.reshape(-1, 3).float())
    sh = convert_rgb_to_spherical_harmonics(rgb_srgb)
    return (xyz.cpu().numpy(), scales.cpu().numpy(), quats.cpu().numpy(),
            opac.cpu().numpy(), sh.cpu().numpy())


def write_gs_ply(path: Path, means, scales, quats, opac, sh):
    """写标准 3DGS PLY（14 字段，与 `03_render_eval.load_ply` 逐项对齐）。

    只写 SH degree 0（`f_dc_*`）：SA 的 `render()` 默认 `sh_degree=0`，
    写 `f_rest` 会被忽略且让文件大一倍。
    """
    from plyfile import PlyData, PlyElement

    n = len(means)
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"),
             ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
             ("opacity", "f4"),
             ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
             ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4")]
    arr = np.empty(n, dtype=dtype)
    arr["x"], arr["y"], arr["z"] = means[:, 0], means[:, 1], means[:, 2]
    for i in range(3):
        arr[f"f_dc_{i}"] = sh[:, i]
    # opacity 存 logit，scale 存 log——与 load_ply 的 sigmoid / exp 严格互逆
    p = np.clip(opac, 1e-6, 1.0 - 1e-6)
    arr["opacity"] = np.log(p / (1.0 - p))
    for i in range(3):
        arr[f"scale_{i}"] = np.log(np.clip(scales[:, i], 1e-12, None))
    for i in range(4):
        arr[f"rot_{i}"] = quats[:, i]
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(path))


def main() -> int:
    ap = argparse.ArgumentParser(description="SHARP 前馈 -> SA 08 NVS 可消费格式（真焦距口径）")
    ap.add_argument("--scene", required=True, type=Path,
                    help="场景目录，需含 images/ 与 prior_camera.json")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--views", default="",
                    help="只导出这些视角（逗号分隔，支持唯一前缀）。空=prior_camera 全部")
    ap.add_argument("--anchor", default="",
                    help="锚定视角（公共系取它的相机系）。空=视角列表第一个。必须属于 source")
    ap.add_argument("--ckpt", type=Path, default=None, help="权重；空=官方 DEFAULT_MODEL_URL")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    scene = args.scene.expanduser().resolve()

    import torch

    # 复用 05 的 predict（真焦距分离），保证与 SH1a 同口径
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from importlib import util as _util
    _spec = _util.spec_from_file_location(
        "sharp05", Path(__file__).resolve().parent / "05_cross_view_consistency.py")
    _m05 = _util.module_from_spec(_spec)
    _spec.loader.exec_module(_m05)
    predict = _m05.predict
    load_rgb = _m05.load_rgb

    from sharp.models import PredictorParams, create_predictor
    from sharp.utils.gaussians import apply_transform

    ids, c2w, K = load_camera_json(scene / "prior_camera.json")
    picked = pick_views(ids, args.views)
    stems = picked or ids
    if not stems:
        raise SystemExit("❌ 视角列表为空")

    anchor = args.anchor
    if anchor:
        cand = [cid for cid in stems if cid == anchor or cid.startswith(anchor)]
        if len(cand) != 1:
            raise SystemExit(f"--anchor {anchor!r} 在导出视角里匹配到 {len(cand)} 个")
        anchor = cand[0]
    else:
        anchor = stems[0]
    c2w_a = c2w[ids.index(anchor)]

    print(f"🚀 SHARP -> NVS 导出：{len(stems)} 视角，锚定 {anchor}")
    print(f"   场景 {scene}")
    for s in stems:
        Kh = K[ids.index(s)]
        print(f"   {s}  fx={Kh[0, 0]:.2f} fy={Kh[1, 1]:.2f} cx={Kh[0, 2]:.1f} cy={Kh[1, 2]:.1f}")
    print(f"   公共系 = 相机 {anchor} 的相机系；T = inv(c2w_anchor) @ c2w_i（不拟合任何 M/s/u）")

    device = torch.device(args.device)
    t0 = time.time()
    predictor = create_predictor(PredictorParams())
    if args.ckpt:
        state = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    else:
        import sharp.cli.predict as _p
        print(f"⬇️  加载权重 {_p.DEFAULT_MODEL_URL}")
        state = torch.hub.load_state_dict_from_url(_p.DEFAULT_MODEL_URL, progress=True)
    predictor.load_state_dict(state)
    predictor.eval().to(device)
    print(f"🏋️  权重 {time.time() - t0:.1f}s")

    audit = {"anchor": anchor, "scene": str(scene), "views": [], "warnings": []}
    all_m, all_s, all_q, all_o, all_c = [], [], [], [], []
    for s in stems:
        i = ids.index(s)
        Kh = K[i]
        img = load_rgb(scene / "images" / f"{s}.png")   # uint8 0..255
        h, w = img.shape[:2]
        t1 = time.time()
        pred = predict(predictor, img, float(Kh[0, 0]), float(Kh[1, 1]),
                       float(Kh[0, 2]), float(Kh[1, 2]), device)
        g = pred["gaussians"]

        # 相机 i 系 -> 锚定相机系。用官方 apply_transform（会正确搬运协方差），
        # 传 **3x4**：05:506 记过 1x3x4 会把 `.T` 变成整体倒序导致错维。
        T = np.linalg.inv(c2w_a) @ c2w[i]
        Tt = torch.from_numpy(T[:3, :].astype(np.float32)).to(g.mean_vectors.device)
        g2 = apply_transform(g, Tt)

        means, scales, quats, opac, sh = gaussians_to_arrays(g2)
        z_med = float(np.median(g2.mean_vectors.reshape(-1, 3)[:, 2].cpu().numpy()))
        audit["views"].append({
            "camera_id": s, "n_gaussians": int(len(means)),
            "transform_anchor_from_this": T.tolist(),
            "z_median_cam": z_med,
            "gauss_scale_median": float(np.median(scales.mean(axis=1))),
            "fx": float(Kh[0, 0]), "fy": float(Kh[1, 1]),
            "cx": float(Kh[0, 2]), "cy": float(Kh[1, 2]),
            "image_size_hw": [int(h), int(w)],
            "fy_audit": pred["fy_audit"],          # 真 fy vs 旧口径的点位差（审计用）
            "forward_seconds": round(time.time() - t1, 2),
        })
        all_m.append(means); all_s.append(scales); all_q.append(quats)
        all_o.append(opac); all_c.append(sh)
        print(f"   ✅ {s}  {len(means):,} 点  z中位 {z_med:.4f}  {time.time() - t1:.1f}s")

    means = np.concatenate(all_m, 0); scales = np.concatenate(all_s, 0)
    quats = np.concatenate(all_q, 0); opac = np.concatenate(all_o, 0)
    sh = np.concatenate(all_c, 0)
    write_gs_ply(out / "gaussians.ply", means, scales, quats, opac, sh)
    print(f"  ✅ gaussians.ply：{len(means):,} 高斯（{len(stems)} 视角并联）")

    # cameras.json：source 视角在公共系的 c2w + 真 K。
    # 08 会用它反算 M/s/u 并核验残差；因这里是解析搬的，残差应当**精确 0**。
    # 若 08 报残差非零 → 说明本脚本的变换或 08 的拟合假设有一方错了，
    # 这是一次免费的自检，不要去调 max_residual 掩盖。
    kamap = {cid: K[i] for i, cid in enumerate(ids)}
    cams = {
        "extrinsics": [{"camera_id": cid,
                        "matrix": (np.linalg.inv(c2w_a) @ c2w[ids.index(cid)]).tolist()}
                       for cid in stems],
        "intrinsics": [{"camera_id": cid, "matrix": kamap[cid].tolist()} for cid in stems],
    }
    (out / "cameras.json").write_text(
        json.dumps(cams, indent=2, ensure_ascii=False), encoding="utf-8")

    zs = [v["z_median_cam"] for v in audit["views"]]
    audit["z_median_spread_ratio"] = float(max(zs) / max(min(zs), 1e-9))
    audit["scale_median_spread_ratio"] = float(
        max(v["gauss_scale_median"] for v in audit["views"]) /
        max(min(v["gauss_scale_median"] for v in audit["views"]), 1e-30))
    audit["focal_note"] = (
        "本脚本喂给网络的是 prior_camera.json 的真 (fx,fy,cx,cy)，"
        "不是 02_run_inference.sh 的 30mm 兜底 f_px；disparity_factor 仍用 fx 以保持网络行为。"
    )
    audit["color_note"] = (
        "colors 是 SHARP 的 linearRGB，导出前做了 linearRGB2sRGB 再编码 SH 直流，"
        "与官方 save_ply 一致，也与 SA 的 sRGB 口径对齐。首版曾跳过该步并以 SA 为理由，"
        "实测导致整幅偏暗、PSNR 低 2.5~2.9 dB（GT 0.471 vs 渲染 0.283）。"
    )
    audit["note"] = (
        "SHARP 是单图模型：N 个视角是 N 份互不相干的局部云，并联只为在同一条留出视角上"
        "被同一把尺子量，不代表 SHARP 具备多图融合能力。z_median_spread_ratio 越大，"
        "并联云错位越重、PSNR 越低——该量必须与 PSNR 一起报。"
    )
    (out / "export_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"  ✅ cameras.json：{len(stems)} 视角（真 K，c2w 已搬到锚定系）")
    print(f"  📏 z 中位极差比 {audit['z_median_spread_ratio']:.4f}"
          f"｜尺度中位极差比 {audit['scale_median_spread_ratio']:.4f}")
    print(f"  📄 台账 {out / 'export_audit.json'}")
    print(f"\n下一步（同一个评测器，这才叫同口径）：")
    print(f"  RESULT={out} \\")
    print(f"  SOURCE_SCENE={scene} \\")
    print(f"  ALL_SCENE=<含全部视角的场景，留出=全部-source> \\")
    print(f"    bash splatanything/08_nvs_eval.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
