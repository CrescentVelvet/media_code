"""按真实时间分段算相机方位覆盖：垂手段 vs 抬手段，判断删帧代价。"""
import os
import json
import numpy as np

ROOT = "/home/velvet/output/som_human_results/hand10_shuangren_zhaoshou"
SPLIT = 130  # 从 sheet 看到的抬手起点


def main():
    for pid in ("p02", "p07"):
        cj = os.path.join(ROOT, f"model_{pid}", "cameras.json")
        with open(cj, encoding="utf-8") as f:
            cams = sorted(json.load(f), key=lambda c: c["img_name"])
        pos = np.array([c["position"] for c in cams], dtype=float)
        n = len(pos)
        az = np.degrees(np.arctan2(pos[:, 1], pos[:, 0]))
        azu = np.degrees(np.unwrap(np.radians(az)))
        el = np.degrees(np.arctan2(pos[:, 2], np.linalg.norm(pos[:, :2], axis=1)))

        def span(a):
            return float(a.max() - a.min())

        print(f"\n===== {pid} n={n} =====")
        print(f"  全帧 方位角 {span(azu):.1f} deg | 俯仰角 {span(el):.1f} deg")
        for name, sl in (("垂手段", slice(0, SPLIT)), ("抬手段", slice(SPLIT, n))):
            a = azu[sl]
            print(f"  {name}[{sl.start}:{sl.stop}] n={a.size} 方位 {span(a):.1f} deg "
                  f"范围 [{a.min():.1f}, {a.max():.1f}] | 俯仰 {span(el[sl]):.1f} deg")

        # 删抬手段后损失的方位角度
        keep = azu[:SPLIT]
        drop = azu[SPLIT:]
        # 简化：落入「全帧范围」但不在 keep 范围的角度
        print(f"  删抬手段({n-SPLIT}帧) 后覆盖 {span(keep):.1f} deg (vs 全 {span(azu):.1f})")
        # 每 10° bin 的占用
        bins = np.arange(-180, 181, 10)
        hk, _ = np.histogram(np.radians(azu[:SPLIT]) * 0 + ((azu[:SPLIT] % 360 + 540) % 360 - 180), bins=bins)
        hd, _ = np.histogram(((azu[SPLIT:] % 360 + 540) % 360 - 180), bins=bins)
        lost = sum(1 for k, d in zip(hk, hd) if k == 0 and d > 0)
        print(f"  仅抬手段独有、垂手段没有的 10deg 方位 bin: {lost} / 36")


if __name__ == "__main__":
    main()
