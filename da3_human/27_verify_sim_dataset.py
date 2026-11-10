#!/usr/bin/env python3
"""仿真数据集验收脚本（渲染器无关，UE / Blender 产物通用）。

对应 24_ue_render_spec.md §5 验收项 + 深度几何自检：
  1. cameras.json 键 ↔ 图像/深度文件一一对应、数量对
  2. K 抽查（tele 长焦 fx 应是主摄的 4 倍；cx/cy 居中）
  3. 深度值域 sanity（最小值 ≈ 人物距离，不该出现 0 大面积或离群巨大值）
  4. 跨视角深度一致性：A 的深度点反投影到世界 → 投影进 B → 与 B 深度比较。
     手系写反 / 深度行序翻转 / w2c-K 不配会立刻爆（中位误差 >0.1m），
     正确几何下 Eevee 数据应 <1cm（受 1mm 深度量化 + 像素舍入限制）

用法:
  python 27_verify_sim_dataset.py <dataset_dir>
"""
import json
import os
import sys

import numpy as np
from PIL import Image


def log(m):
    print(m, flush=True)


def resolve_depth_dir(root):
    """深度目录：V1.1 规格为 depth_radial/，本地 Blender 验证集为 depth/，两者都认。"""
    for s in ("depth_radial", "depth"):
        if os.path.isdir(os.path.join(root, s)):
            return s
    sys.exit(f"❌ {root} 找不到深度目录 depth_radial/ 或 depth/")


def load_depth(path):
    """16-bit PNG（毫米）→ float32 米。PIL 'I' 模式按 uint32 读，我们写的是 uint16 值。"""
    a = np.array(Image.open(path), np.int32)
    return a.astype(np.float32) / 1000.0


def unproject(depth_radial_m, K, w2c, step=1):
    """径向深度 + K + w2c → 世界系点云（HxWx3）与有效 mask。

    step>1 用于跳采样加速：像素坐标必须带全分辨率步长（arange(0,W,step)），
    否则采回来的 (u,v) 直接索引全分辨率深度会错位（自伤过一次）。
    """
    H, W = depth_radial_m.shape
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    u, v = np.meshgrid(np.arange(0, W * step, step), np.arange(0, H * step, step))
    d_cam = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=-1)  # (H,W,3)
    norm = np.linalg.norm(d_cam, axis=-1, keepdims=True)
    pts_cam = d_cam / norm * depth_radial_m[..., None]                          # radial 沿射线
    c2w = np.linalg.inv(np.asarray(w2c, np.float64))
    R, t = c2w[:3, :3], c2w[:3, 3]
    pts_world = pts_cam @ R.T + t
    valid = depth_radial_m > 1e-3
    return pts_world, valid


def z_consistency(pair_a, pair_b, cams, dataset_dir, step=4):
    """A 的深度点 → 世界 → B 相机 → 与 B 的 z-depth 比较，返回误差中位（米）。"""
    ka, kb = pair_a, pair_b
    da = load_depth(os.path.join(dataset_dir, resolve_depth_dir(dataset_dir), f"{ka}.png"))[::step, ::step]
    db = load_depth(os.path.join(dataset_dir, resolve_depth_dir(dataset_dir), f"{kb}.png"))
    pa, va = unproject(da, cams[ka]["K"], cams[ka]["w2c"], step=step)
    Hb, Wb = db.shape
    fxb, fyb = cams[kb]["K"][0][0], cams[kb]["K"][1][1]
    cxb, cyb = cams[kb]["K"][0][2], cams[kb]["K"][1][2]
    w2cb = np.asarray(cams[kb]["w2c"], np.float64)
    pb = pa @ w2cb[:3, :3].T + w2cb[:3, 3]
    zb = pb[..., 2]
    ub = np.round(fxb * pb[..., 0] / zb + cxb).astype(int)
    vb = np.round(fyb * pb[..., 1] / zb + cyb).astype(int)
    inb = (zb > 0.05) & (ub >= 0) & (ub < Wb) & (vb >= 0) & (vb < Hb) & va
    if inb.sum() < 100:
        return None, 0
    # B 存的是径向深度 → 转 z 再比
    u, v = np.meshgrid(np.arange(Wb), np.arange(Hb))
    mult_b = np.sqrt(1.0 + ((u - cxb) / fxb) ** 2 + ((v - cyb) / fyb) ** 2)
    z_of_b = (db / mult_b)[vb[inb], ub[inb]]
    z_of_pts = zb[inb]
    err = np.abs(z_of_pts - z_of_b)
    return float(np.median(err)), int(inb.sum())


def main():
    if len(sys.argv) < 2:
        sys.exit("❌ 用法: python 27_verify_sim_dataset.py <dataset_dir>")
    root = sys.argv[1]
    cams = json.load(open(os.path.join(root, "cameras.json"), encoding="utf-8"))
    meta = json.load(open(os.path.join(root, "meta.json"), encoding="utf-8"))
    log(f"🚀 验收 {root}（{meta.get('seq_id', '?')}）")

    # 1) 文件对应
    imgs = {}
    for sub in ("images_input", "images_eval"):
        d = os.path.join(root, sub)
        if os.path.isdir(d):
            for f in os.listdir(d):
                imgs[os.path.splitext(f)[0]] = os.path.join(sub, f)
    depths_dir = resolve_depth_dir(root)
    depths = {os.path.splitext(f)[0]: f for f in os.listdir(os.path.join(root, depths_dir))}
    miss_img = sorted(set(cams) - set(imgs))
    miss_dep = sorted(set(cams) - set(depths))
    extra = sorted((set(imgs) | set(depths)) - set(cams))
    ok1 = not (miss_img or miss_dep or extra)
    log(f"   [{'✅' if ok1 else '❌'}] 键对应: cameras {len(cams)} / 图 {len(imgs)} / 深度 {len(depths)}"
        f"（{depths_dir}/）"
        + (f" 缺图{miss_img} 缺深{miss_dep} 多余{extra}" if not ok1 else ""))

    # 2) K 抽查
    fx_main = cams["main"]["K"][0][0] if "main" in cams else None
    fx_tele = cams["tele"]["K"][0][0] if "tele" in cams else None
    msgs = []
    ok2 = True
    if fx_main and fx_tele:
        ratio = fx_tele / fx_main
        ok2 = abs(ratio - 4.0) < 0.05
        msgs.append(f"tele/main fx 比 {ratio:.3f}（应≈4）")
    for k in ("s06", "main"):
        if k in cams:
            cx, cy = cams[k]["K"][0][2], cams[k]["K"][1][2]
            W, H = meta.get("W", 0), meta.get("H", 0)
            if W:
                okc = abs(cx - (W - 1) / 2) < 1 and abs(cy - (H - 1) / 2) < 1
                ok2 = ok2 and okc
                msgs.append(f"{k} 主点 ({cx:.1f},{cy:.1f})")
    log(f"   [{'✅' if ok2 else '❌'}] K 抽查: {'; '.join(msgs)}")

    # 3) 深度值域
    ok3 = True
    for k in ("s06", "wide", "tele", "e00_p00"):
        if k not in cams:
            continue
        d = load_depth(os.path.join(root, depths_dir, f"{k}.png"))
        finite = d[(d > 0.001) & (d < 65.0)]
        if len(finite) < d.size * 0.5:
            ok3 = False
        log(f"   [{'✅' if ok3 else '❌'}] {k} 深度: 有效 {len(finite)/d.size:.1%} "
            f"min {finite.min():.3f}m 中位 {np.median(finite):.3f}m max {finite.max():.2f}m")

    # 4) 跨视角一致性（同主摄、相邻方位的评估环最严格）
    pairs = [("e00_p00", "e01_p00"), ("e00_p00", "s06"), ("s06", "e24_p00"),
             ("main", "wide"), ("main", "tele")]
    ok4 = True
    for a, b in pairs:
        if a in cams and b in cams:
            med, n = z_consistency(a, b, cams, root)
            if med is None:
                log(f"   ⚠️ {a}→{b} 重叠点不足，跳过")
                continue
            good = med < 0.02
            ok4 = ok4 and good
            log(f"   [{'✅' if good else '❌'}] {a}→{b}: 中位 z 误差 {med*1000:.1f} mm（n={n}）"
                f"{' —— 手系/深度翻转嫌疑！' if not good else ''}")

    allok = ok1 and ok2 and ok3 and ok4
    log(f"🎉 验收{'通过' if allok else '未通过，看上面 ❌ 项'}")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
