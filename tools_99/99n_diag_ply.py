#!/usr/bin/env python3
"""99n：诊断一个 PLY 高斯文件的字段约定，判断 99m 该用哪套 activation。

渲染出"整屏红绿蓝色块/光影"时先跑本脚本定位：是 activation 二次施加、
SH 系数约定不符、还是 NaN/Inf 污染。纯只读，不渲染。

环境准备：
  渲染 conda env（有 plyfile + numpy），与 99m 同环境。

用法：
  PLY=<ply路径> python tools_99/99n_diag_ply.py
  # 或改 main() 里的 PLY 预设后零参数直跑

Env vars（不设则用 main() 里的默认值）：
  PLY         要诊断的 PLY 路径（必填）
"""
import math
import os
import sys


def die(msg):
    sys.exit(f"❌ {msg}")


def fmt(v):
    """紧凑格式化数值：很大/很小的数用科学计数法。"""
    if abs(v) >= 1000 or (abs(v) < 0.001 and v != 0):
        return f"{v:.3e}"
    return f"{v:.4f}"


def diag_ply(ply_path):
    try:
        import numpy as np
        from plyfile import PlyData
    except ImportError:
        die("需要 plyfile + numpy，请用 99m 的 conda env 跑")

    if not os.path.isfile(ply_path):
        die(f"PLY 不存在: {ply_path}")
    print(f"🔍 诊断 PLY: {ply_path}")

    ply = PlyData.read(ply_path)
    names = [p.name for p in ply["vertex"].properties]
    v = ply["vertex"]
    n = len(v)
    print(f"  顶点数: {n}")
    print(f"  字段({len(names)}): {names}")

    # 每字段范围 + NaN/Inf 计数
    print("\n=== 逐字段统计 ===")
    stats = {}
    for p in ply["vertex"].properties:
        name = p.name
        try:
            arr = np.asarray(v[name])
        except Exception as e:
            print(f"  ⚠️ {name}: 读取失败 {e}")
            continue
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        a = arr.astype(np.float64)
        nan_cnt = int(np.isnan(a).sum())
        inf_cnt = int(np.isinf(a).sum())
        finite = a[np.isfinite(a)]
        if finite.size == 0:
            print(f"  {name:10s} dtype={arr.dtype}  全部非有限值! NaN={nan_cnt} Inf={inf_cnt}")
            stats[name] = None
            continue
        print(f"  {name:10s} dtype={arr.dtype} min={fmt(finite.min())} "
              f"max={fmt(finite.max())} mean={fmt(finite.mean())}"
              + (f"  ⚠️NaN={nan_cnt} Inf={inf_cnt}" if nan_cnt or inf_cnt else ""))
        stats[name] = (float(finite.min()), float(finite.max()))

    # ---- 约定判据 ----
    print("\n=== 约定判据（99m 当前假设 vs 你的数据） ===")

    # 1) scale：标准 3DGS 训练端存 log-scale（激活前），典型范围 [-10, +3]；
    #    若已是米制激活值（max 数十上百），再 exp 会爆成巨块。
    for key in ("scale_0", "scale0", "scales"):
        if key in stats and stats[key]:
            lo, hi = stats[key]
            verdict = ("✅ 像 log-scale（激活前，99m 现状 exp 正确）" if hi < 10
                       else "⚠️ 像已激活米制值（99m 再 exp 会爆成巨型高斯 → 整屏色块）")
            print(f"  scale ({key}): max={fmt(hi)}  {verdict}")

    # 2) opacity：激活前 ~N(0, 小)；激活后 0~1
    for key in ("opacity",):
        if key in stats and stats[key]:
            lo, hi = stats[key]
            verdict = ("✅ 像激活前（sigmoid 正确）" if lo < 0 or hi > 1
                       else "⚠️ 像已激活 0~1（再 sigmoid 会全趋 0.62，影响小）")
            print(f"  opacity: [{fmt(lo)}, {fmt(hi)}]  {verdict}")

    # 3) f_dc：激活前典型 [-3, 3]；激活后 [0, 1]（甚至 0~0.5 若乘过 0.5+0.5）
    for key in ("f_dc_0", "red", "r"):
        if key in stats and stats[key]:
            lo, hi = stats[key]
            if key == "f_dc_0":
                verdict = ("✅ 像激活前（sigmoid×0.5+0.5 正确）" if lo < 0 or hi > 1
                           else "⚠️ 像已激活 0~1（再激活会发灰）")
            else:
                verdict = "⚠️ 这是普通 RGB 颜色字段而非 SH DC 项"
            print(f"  {key}: [{fmt(lo)}, {fmt(hi)}]  {verdict}")

    # 4) SH 系数：f_rest 按 3 通道展平，每通道 (d+1)^2-1 个 → 阶数与磁盘排布
    #    （排布判据与 99m detect_sh_layout 同源：通道主序每块尾部趋零）
    sh_cols = sorted((c for c in names if c.startswith("f_rest")),
                     key=lambda s: int(s.split("_")[-1]))
    if sh_cols:
        k = len(sh_cols)
        per = k // 3
        if k % 3 == 0 and math.isqrt(per + 1) ** 2 == per + 1:
            print(f"  f_rest 系数: {k} 个（3 通道 × {per}）→ SH {math.isqrt(per + 1) - 1} 阶（完整）")
            # 排布判据与 99m detect_sh_layout 同源：取各块尾/头能量比的最大值，
            # 通道主序每块都衰减，系数主序至少有一块（同阶）近似水平。
            # 双统计量取「或」（中位 0.30 / 均值 0.25），抗真实数据的重尾与异常点
            col_abs = np.abs(np.stack([np.asarray(v[c]) for c in sh_cols], axis=1))

            def _ratios(a):
                out = []
                for b in range(3):
                    head = float(a[b * per: b * per + 3].mean())
                    tail = float(a[(b + 1) * per - 3: (b + 1) * per].mean())
                    out.append(tail / head if head > 0 else 1.0)
                return out

            r_med = _ratios(np.median(col_abs, axis=0))
            r_mean = _ratios(col_abs.mean(axis=0))
            worst = min(max(r_med), max(r_mean))
            print(f"  SH 排布: 块尾/头能量比 中位 {['%.3f' % r for r in r_med]} "
                  f"均值 {['%.3f' % r for r in r_mean]} → " +
                  ("⚠️ 通道主序（部分 fork 导出，99m SH_LAYOUT=auto 可识别，"
                   "症状=DC 颜色正常但一动相机整屏彩虹）" if worst < 0.30
                   else "✅ 系数主序（官方 3DGS，99m 默认即可）"))
        else:
            print(f"  f_rest 系数: {k} 个 → ⚠️ 个数不是 3×((d+1)^2-1)，排布可能非标")
    elif "f_dc_0" in names:
        print("  无 f_rest → 纯 DC 颜色（视角无关），排除了 SH 约定问题")

    # 5) rot：四元数模长应≈1
    if "rot_0" in stats and stats["rot_0"]:
        import numpy as _np
        norms = _np.sqrt(sum(_np.asarray(v[f"rot_{i}"], dtype=_np.float64) ** 2
                             for i in range(4)))
        print(f"  rot 四元数模长: min={fmt(norms.min())} max={fmt(norms.max())} "
              f"（≈1 则归一化良好；分布乱则可能是 xyzw 顺序，99m 需 ROT_ORDER=xyzw）")

    print("\n📋 把以上输出原样贴回来，据此定 99m 的开关组合。")


def main():
    # ===== 改模型时只改这几行 =====
    PLY = ""  # 必填：要诊断的 PLY 路径
    cfg = {"ply": os.environ.get("PLY", PLY).strip()}
    if not cfg["ply"]:
        die("PLY 未设置（env 或 main() 预设二选一）")
    diag_ply(cfg["ply"])


if __name__ == "__main__":
    main()
