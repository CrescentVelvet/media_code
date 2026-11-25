#!/usr/bin/env python3
"""汇总各 case 的 COLMAP 结果，生成离线总览和 CSV。

Env: DATASET_ROOT, OUTPUT_DIRNAME.
"""
import csv
import html as html_lib
import json
import os
from pathlib import Path

ROOT = Path(os.environ["DATASET_ROOT"])
OUT_NAME = os.environ.get("OUTPUT_DIRNAME", "colmap_sfm_results")


def main():
    rows = []
    for path in sorted(ROOT.glob(f"case*/{OUT_NAME}/summary.json"), key=lambda p: int(p.parts[-3].split("_")[0][4:])):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["relative_path"] = str(path.relative_to(ROOT)).replace("\\", "/")
        data["viewer_path"] = str((path.parent / "index.html").relative_to(ROOT)).replace("\\", "/")
        data["overview_path"] = str((path.parent / "overview.png").relative_to(ROOT)).replace("\\", "/")
        rows.append(data)
    with (ROOT / f"{OUT_NAME}_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "registered", "input", "points3D", "observations", "median_reprojection_px", "rms_reprojection_px", "seconds", "result_dir"])
        for r in rows:
            writer.writerow([r["case"], r["registered_count"], r["input_count"], r["points3D_count"], r["observation_count"],
                             r["median_reprojection_error_px"], r["rms_reprojection_error_px"], r["elapsed_seconds"], r["relative_path"]])
    summary = {"successful_cases": len(rows), "total_input_images": sum(r["input_count"] for r in rows),
               "registered_images": sum(r["registered_count"] for r in rows),
               "all_registered_cases": sum(r["registered_count"] == r["input_count"] for r in rows), "cases": rows,
               "scale": "Each case has an independent arbitrary SfM scale; do not merge or compare translations numerically.",
               "accuracy": "Reprojection residuals are self-consistency, not ground-truth trajectory accuracy."}
    (ROOT / f"{OUT_NAME}_batch_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    items = []
    for index, r in enumerate(rows):
        ok = r["registered_count"] == r["input_count"]
        missing = ("<p class='missing'>未注册: " + ", ".join(r["unregistered_names"]) + "</p>") if r["unregistered_names"] else ""
        # srcdoc 保留每组原页面的 JS/CSS 作用域，避免合并 17 份脚本后变量互相覆盖。
        # 不依赖 iframe 子文件 URL，因此离线双击和预览代理下都能展开交互。
        viewer = html_lib.escape((ROOT / r["viewer_path"]).read_text(encoding="utf-8"), quote=True)
        items.append(f"""<article class='card'><img src='{r['overview_path']}' loading='lazy' alt='{r['case']} SfM 总览'><div class='body'><h2>{r['case']}</h2><p><span class='{"ok" if ok else "warn"}'>{r['registered_count']}/{r['input_count']} 帧注册</span> · {r['points3D_count']:,} 点</p><p>中位重投影误差 <b>{r['median_reprojection_error_px']:.3f} px</b> · RMS {r['rms_reprojection_error_px']:.3f} px</p>{missing}<p><a href='{r['overview_path']}'>查看总览图</a> · <a href='{r['relative_path']}camera_poses.csv'>位姿 CSV</a></p><details class='interactive'{' open' if index == 0 else ''}><summary>展开交互位姿（拖动旋转 / 滚轮缩放）</summary><iframe title='{r['case']} 交互位姿' loading='lazy' srcdoc="{viewer}"></iframe></details></div></article>""")
    html = """<!doctype html><html lang='zh'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>海思端侧拍摄 — 十图 COLMAP SfM 批量结果</title><style>*{box-sizing:border-box}body{font:15px/1.6 system-ui,'Microsoft YaHei',sans-serif;max-width:1500px;margin:auto;padding:28px;background:#f3f6fa;color:#162438}h1{font-size:27px;margin:0 0 12px}h2{font-size:17px;margin:12px 0 4px}a{color:#086894;text-decoration:none}a:hover{text-decoration:underline}.notice{padding:14px 18px;background:#fff4d8;border-left:4px solid #c58c13;margin:16px 0}.stats{display:flex;gap:12px;flex-wrap:wrap}.stat{padding:12px 20px;background:white;border:1px solid #dbe5f0;border-radius:8px}.stat b{font-size:23px;color:#17648a;display:block}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(390px,1fr));gap:16px}.card{background:#fff;border:1px solid #d8e2ed;border-radius:10px;overflow:hidden}.card img{display:block;width:100%;aspect-ratio:16/10;object-fit:contain;background:#f5f7fb}.body{padding:0 16px 14px}.interactive{margin-top:12px;border-top:1px solid #e4eaf2;padding-top:10px}.interactive summary{cursor:pointer;font-weight:600;color:#086894;padding:5px 0}.interactive iframe{display:block;width:100%;height:900px;margin-top:10px;border:1px solid #d8e2ed;border-radius:6px;background:#f4f7fb}.body p{margin:5px 0}.ok{color:#087a4b}.warn{color:#b46700}.missing{background:#fff0f0;color:#a32121;padding:8px;border-left:3px solid #c23b3b}code{background:#edf1f5;padding:2px 5px}</style></head><body><h1>海思端侧拍摄 · 十图 COLMAP SfM 批量结果</h1><p>每个 case 的 <code>images/</code> 独立重建。相机为 OPENCV 单相机；原图 SIFT → 45 对穷举匹配 → 增量 SfM → 全局 BA。case7/14 匹配比值为 0.9，其余为 0.8，详见每组日志。</p><div class='stats'>__STATS__</div><div class='notice'><b>精度口径：</b>重投影误差是图像观测的自洽性，不等于绝对轨迹精度。SfM 的尺度、世界原点与方向任意；不同 case 的平移数值不可直接比较或拼接。<br><b>case7：</b>仅 view_00/01/02/07/08/09 注册。view_03–06 为侧面/后脑近景，与正面段的稳定共享特征不足；普通/增强匹配均无法闭合。不要把这四张设为单位阵或复制邻帧位姿；需补拍侧面到背面的连续过渡视角，保持相同对焦距离、曝光和静态姿态。</div><div class='grid'>__ITEMS__</div><h2>每个 case 里的文件</h2><ul><li><code>colmap_sfm_results/index.html</code> — 离线交互视图：相机视锥、点云、位姿、匹配矩阵，可拖动旋转。</li><li><code>colmap_sfm_results/overview.png</code> — 直接查看的位姿与质量总览。</li><li><code>colmap_sfm_results/cameras.npz</code> — 供 Python 读取：names / c2w / w2c / intrinsics / distortion / points3D。</li><li><code>colmap_sfm_results/camera_poses.csv</code> — 逐帧 R、t、相机中心 C、K、畸变与误差；可用 Excel 打开。</li><li><code>colmap_sfm_results/sparse/0/</code> — 标准 COLMAP BIN + TXT 模型；<code>sparse_with_cameras.ply</code> 可在 CloudCompare / MeshLab 打开（红色为相机中心）。</li></ul></body></html>"""
    stats = f"<div class='stat'><b>{len(rows)}</b>组已完成</div><div class='stat'><b>{summary['all_registered_cases']}</b>组十帧全注册</div><div class='stat'><b>{summary['registered_images']}/{summary['total_input_images']}</b>图像注册</div>"
    (ROOT / f"{OUT_NAME}_summary.html").write_text(html.replace("__STATS__", stats).replace("__ITEMS__", "".join(items)), encoding="utf-8", newline="\n")
    print(f"✅ 汇总 {len(rows)} 组: {summary['all_registered_cases']} 组全部注册，{summary['registered_images']}/{summary['total_input_images']} 帧")
    for r in rows:
        print(f"{r['case']}: {r['registered_count']}/{r['input_count']} | {r['points3D_count']} pts | median={r['median_reprojection_error_px']:.3f}px | RMS={r['rms_reprojection_error_px']:.3f}px")


if __name__ == "__main__":
    main()
