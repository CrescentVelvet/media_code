#!/usr/bin/env python3
"""从 COLMAP TXT 导出可审计位姿、匹配报告和离线可视化。"""
import csv
import json
import os
import sqlite3
from pathlib import Path

import numpy as np


def read_model(path):
    cameras = {}
    for line in (path / "cameras.txt").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            p = line.split()
            cameras[int(p[0])] = {"model": p[1], "width": int(p[2]), "height": int(p[3]),
                                  "params": np.array([float(x) for x in p[4:]])}
    images = []
    lines = [s for s in (path / "images.txt").read_text(encoding="utf-8").splitlines() if s and not s.startswith("#")]
    if len(lines) % 2:
        raise ValueError("images.txt observation line missing")
    for i in range(0, len(lines), 2):
        p = lines[i].split()
        images.append({"image_id": int(p[0]), "q": np.array([float(x) for x in p[1:5]]),
                       "t": np.array([float(x) for x in p[5:8]]), "camera_id": int(p[8]),
                       "name": p[9], "obs": np.array([float(x) for x in lines[i + 1].split()]).reshape(-1, 3)})
    points = {}
    for line in (path / "points3D.txt").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            p = line.split()
            points[int(p[0])] = {"xyz": np.array([float(x) for x in p[1:4]]),
                                 "rgb": [int(x) for x in p[4:7]], "error": float(p[7]),
                                 "track_length": (len(p) - 8) // 2}
    return cameras, images, points


def rotation(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def distortion(cam):
    name, p = cam["model"], cam["params"]
    if name in ("SIMPLE_PINHOLE", "PINHOLE"):
        return [0.0] * 4
    if name in ("SIMPLE_RADIAL", "RADIAL"):
        return [float(p[-1]), 0.0, 0.0, 0.0]
    if name == "OPENCV":
        return [float(x) for x in p[4:8]]
    raise ValueError(f"Unsupported camera: {name}")


def project(xyz, cam, R, t):
    p = cam["params"]
    x, y, z = R @ xyz + t
    xn, yn = x / z, y / z
    k1, k2, p1, p2 = distortion(cam)
    r2 = xn * xn + yn * yn
    radial = 1 + k1 * r2 + k2 * r2 * r2
    xd = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
    yd = yn * radial + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
    if cam["model"] in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        fx, fy, cx, cy = p[0], p[0], p[1], p[2]
    else:
        fx, fy, cx, cy = p[:4]
    return np.array([fx * xd + cx, fy * yd + cy])


def export_result(case, files, out, model, db, elapsed):
    cameras, images, points = read_model(model / "txt")
    if not images:
        raise RuntimeError("最大子模型没有注册图像")
    with sqlite3.connect(str(db)) as conn:
        name_to_id = {n: i for i, n in conn.execute("SELECT image_id, name FROM images")}
        pairs = conn.execute("SELECT pair_id, rows FROM two_view_geometries WHERE rows > 0").fetchall()
    id_to_name = {v: k for k, v in name_to_id.items()}
    matches = []
    for pair, count in pairs:
        a, b = sorted((pair // 2147483647, pair % 2147483647))
        if a in id_to_name and b in id_to_name:
            matches.append({"a": id_to_name[a], "b": id_to_name[b], "inliers": count})
    rows, residuals = [], []
    for im in sorted(images, key=lambda x: x["name"]):
        cam, R, t = cameras[im["camera_id"]], rotation(im["q"]), im["t"]
        errors = [float(np.linalg.norm(project(points[int(pid)]["xyz"], cam, R, t) - [x, y]))
                  for x, y, pid in im["obs"] if int(pid) in points]
        if not errors:
            raise RuntimeError(f"{im['name']} 没有有效三维观测")
        residuals.extend(errors)
        c = -R.T @ t
        K = np.eye(3)
        if cam["model"] in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            K[0, 0] = K[1, 1] = cam["params"][0]
            K[:2, 2] = cam["params"][1:3]
        else:
            K[0, 0], K[1, 1] = cam["params"][:2]
            K[:2, 2] = cam["params"][2:4]
        w2c, c2w = np.eye(4), np.eye(4)
        w2c[:3, :3], w2c[:3, 3], c2w[:3, :3], c2w[:3, 3] = R, t, R.T, c
        rows.append({"name": im["name"], "registered": True, "observations": len(errors),
            "rms_error_px": float(np.sqrt(np.mean(np.square(errors)))), "median_error_px": float(np.median(errors)),
            "R": R.tolist(), "t": t.tolist(), "center": c.tolist(), "K": K.tolist(),
            "w2c": w2c.tolist(), "c2w": c2w.tolist(), "distortion": distortion(cam), "camera_model": cam["model"],
            "fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2])})
    registered = {r["name"] for r in rows}
    rows += [{"name": p.name, "registered": False} for p in files if p.name not in registered]
    median = float(np.median(residuals))
    rms = float(np.sqrt(np.mean(np.square(residuals))))
    valid = [r for r in rows if r["registered"]]
    np.savez_compressed(out / "cameras.npz", names=np.array([r["name"] for r in valid]),
        c2w=np.array([r["c2w"] for r in valid], dtype=np.float32), w2c=np.array([r["w2c"] for r in valid], dtype=np.float32),
        intrinsics=np.array([r["K"] for r in valid], dtype=np.float32),
        distortion=np.array([r["distortion"] for r in valid], dtype=np.float32),
        camera_model=np.array([r["camera_model"] for r in valid]),
        points3D=np.array([p["xyz"] for p in points.values()], dtype=np.float32),
        convention=np.array("OpenCV x-right y-down z-forward; SfM arbitrary scale, not meters"))
    export_csv(out, rows)
    export_ply(out, images, points)
    payload = {"case": case.name, "input_count": len(files), "registered": len(images),
        "num_points": len(points), "median_reproj_error_px": median, "rms_reproj_error_px": rms,
        "num_observations": len(residuals), "input_names": [p.name for p in files], "per_image": rows,
        "matches": matches, "point_cloud": [[p["xyz"].tolist(), p["rgb"]] for p in points.values()]}
    if len(payload["point_cloud"]) > 60000:
        rng = np.random.default_rng(0)
        payload["point_cloud"] = [payload["point_cloud"][i] for i in rng.choice(len(payload["point_cloud"]), 60000, replace=False)]
    export_html(out, payload)
    valid_cameras = [im["camera_id"] for im in images]
    summary = {"case": case.name, "input_count": len(files), "registered_count": len(images),
        "registered_names": [r["name"] for r in valid], "unregistered_names": [r["name"] for r in rows if not r["registered"]],
        "points3D_count": len(points), "observation_count": len(residuals),
        "median_reprojection_error_px": median, "rms_reprojection_error_px": rms,
        "geometric_inlier_matches": sum(p["inliers"] for p in matches),
        "sift_max_ratio": float(os.environ.get("SIFT_RATIO", "0.8")),
        "colmap_version": "4.2.1 CUDA",
        "image_size": [cameras[valid_cameras[0]]["width"], cameras[valid_cameras[0]]["height"]],
        "intrinsics_per_camera": {r["name"]: r["K"] for r in valid},
        "distortion_per_camera": {r["name"]: r["distortion"] for r in valid},
        "camera_models": sorted(set(r["camera_model"] for r in valid)), "scale": "SfM arbitrary units; not meters",
        "elapsed_seconds": elapsed, "scope": "Image-space residual, not ground-truth trajectory accuracy."}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    (out / "quality_report.txt").write_text(
        f"{case.name}\n注册图像: {len(images)}/{len(files)}\n三维点: {len(points)}\n观测数: {len(residuals)}\n"
        f"中位重投影误差: {median:.6f} px\nRMS 重投影误差: {rms:.6f} px\n"
        "模型: 单 OPENCV 相机，SIFT + exhaustive matching + incremental SfM + global BA。\n"
        "位姿单位: SfM 任意尺度，不是米。\n注意: 无拍摄真值，误差不是绝对位姿精度。\n", encoding="utf-8", newline="\n")
    plot_overview(out, payload)
    try:
        plot_reprojection(case / "images", out, cameras, images, points)
    except Exception as exc:
        # QA overlay must not discard an otherwise valid COLMAP model.
        print(f"⚠️ 重投影 QA 图生成失败: {exc}", flush=True)
    return summary


def plot_reprojection(images_dir, out, cameras, images, points):
    from PIL import Image, ImageDraw, ImageFont
    selected = [images[len(images) // 2], images[-1]]
    canvas = Image.new("RGB", (1408, 756), "#f5f7fb")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/mnt/c/Windows/Fonts/msyh.ttc", 16)
    for k, im in enumerate(selected):
        image = Image.open(images_dir / im["name"]).convert("RGB")
        cam, R, t = cameras[im["camera_id"]], rotation(im["q"]), im["t"]
        layer = ImageDraw.Draw(image)
        for x, y, pid in im["obs"]:
            if int(pid) not in points:
                continue
            u, v = project(points[int(pid)]["xyz"], cam, R, t)
            layer.line((x-3, y, x+3, y), fill=(0, 235, 255), width=1)
            layer.line((x, y-3, x, y+3), fill=(0, 235, 255), width=1)
            layer.ellipse((u-1.5, v-1.5, u+1.5, v+1.5), outline=(255, 40, 80), width=1)
        canvas.paste(image, (704*k, 0))
        draw.text((704*k+8, 712), im["name"] + " | 青十字=观测，红圈=回投", fill="#152b43", font=font)
    canvas.save(out / "reprojection_check.png")


def export_csv(out, rows):
    header = ["image", "registered", "observations", "rms_reprojection_px", "median_reprojection_px",
              "fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "tx", "ty", "tz", "Cx", "Cy", "Cz",
              "R00", "R01", "R02", "R10", "R11", "R12", "R20", "R21", "R22"]
    with (out / "camera_poses.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            if not r["registered"]:
                w.writerow([r["name"], 0])
                continue
            w.writerow([r["name"], 1, r["observations"], r["rms_error_px"], r["median_error_px"],
                        r["fx"], r["fy"], r["cx"], r["cy"]] + r["distortion"] + r["t"] + r["center"] +
                       [x for row in r["R"] for x in row])


def export_ply(out, images, points):
    vertices = [(p["xyz"], p["rgb"]) for p in points.values()]
    vertices += [(-rotation(im["q"]).T @ im["t"], [220, 35, 60]) for im in images]
    with (out / "sparse_with_cameras.ply").open("w", encoding="utf-8", newline="\n") as f:
        f.write("ply\nformat ascii 1.0\ncomment SfM arbitrary units; red vertices are camera centers\n")
        f.write(f"element vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for xyz, rgb in vertices:
            f.write(f"{xyz[0]:.9g} {xyz[1]:.9g} {xyz[2]:.9g} {rgb[0]} {rgb[1]} {rgb[2]}\n")


def export_html(out, payload):
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>__TITLE__</title>
<style>*{box-sizing:border-box}body{margin:0;font:14px/1.5 system-ui,'Microsoft YaHei',sans-serif;background:#f4f7fb;color:#17253b}header{padding:20px 28px;background:#fff;border-bottom:1px solid #dce4ee}h1{font-size:22px;margin:0 0 6px}h2{font-size:17px;margin:0 0 12px}.muted,.small{color:#5b6d83}.layout{display:grid;grid-template-columns:1.2fr 1fr;gap:18px;padding:20px 28px}.panel{background:#fff;padding:18px;border:1px solid #dce4ee;border-radius:10px;min-width:0}#viewer{width:100%;height:530px;background:#f8fafc;position:relative;overflow:hidden;border-radius:6px}canvas{display:block;touch-action:none}table{border-collapse:collapse;width:100%;font-size:12px;white-space:nowrap}th,td{border-bottom:1px solid #e4eaf2;padding:7px 8px;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#edf3f8;position:sticky;top:0}.scroll{max-height:560px;overflow:auto}.warn{background:#fff5dc;border-left:4px solid #dc9400;padding:10px 14px;margin-top:12px}button{border:1px solid #b9c9d9;background:#fff;padding:5px 10px;border-radius:5px;cursor:pointer;margin:4px}@media(max-width:1000px){.layout{grid-template-columns:1fr;padding:12px}}</style></head>
<body><header><h1>__TITLE__</h1><div id="summary" class="muted"></div></header><main class="layout"><section class="panel"><h2>相机位姿与稀疏点云</h2><div id="viewer"></div><button id="reset">重置视角</button><button id="top">俯视</button><button id="side">侧视</button><label class="small"><input id="points" type="checkbox" checked> 点云</label><label class="small"><input id="frustums" type="checkbox" checked> 相机视锥</label><p class="small">拖动旋转 · 滚轮缩放。红色为相机位置与视锥；OpenCV 坐标（x 右、y 下、z 前）。</p><div class="warn">SfM 只恢复相对位姿：尺度、世界原点与坐标轴方向不是绝对真值。重投影误差衡量图像观测自洽性，不等于拍摄轨迹绝对精度。</div></section><section class="panel"><h2>逐帧重投影误差与内参</h2><div id="perimage" class="scroll"></div></section><section class="panel"><h2>位姿数值（world → camera）</h2><p class="small">C = -Rᵀt 是相机中心。CSV / NPZ 中另有 camera → world；长度单位任意。</p><div id="poses" class="scroll"></div></section><section class="panel"><h2>图像对有效匹配数</h2><p class="small">几何验证后的 inlier 数；0 表示未形成可靠匹配。</p><div id="matches" class="scroll"></div></section></main>
<script>
const D=__DATA__;
document.getElementById('summary').textContent=`${D.registered}/${D.input_count} 帧注册 · ${D.num_points} 个三维点 · 中位重投影误差 ${D.median_reproj_error_px.toFixed(3)} px · OPENCV 单相机 · 全局 BA`;
function table(el,head,rows){document.getElementById(el).innerHTML='<table><thead><tr>'+head.map(x=>'<th>'+x+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(x=>'<td>'+x+'</td>').join('')+'</tr>').join('')+'</tbody></table>';}
table('perimage',['图像','观测数','RMS(px)','中位(px)','fx / fy','cx / cy'],D.per_image.filter(x=>x.registered).map(x=>[x.name,x.observations,x.rms_error_px.toFixed(3),x.median_error_px.toFixed(3),x.fx.toFixed(1)+' / '+x.fy.toFixed(1),x.cx.toFixed(1)+' / '+x.cy.toFixed(1)]));
table('poses',['图像','tx','ty','tz','Cx','Cy','Cz'],D.per_image.filter(x=>x.registered).map(x=>[x.name,...x.t.map(n=>n.toFixed(5)),...x.center.map(n=>n.toFixed(5))]));
let mat={};for(let m of D.matches){mat[m.a+'|'+m.b]=m.inliers;mat[m.b+'|'+m.a]=m.inliers;}let N=D.input_names;table('matches',['图像',...N.map(n=>n.replace(/_source_.*/,'').replace('view_',''))],N.map(a=>[a.replace(/_source_.*/,''),...N.map(b=>a===b?'—':(mat[a+'|'+b]||0))]));
const host=document.getElementById('viewer'),canvas=document.createElement('canvas'),ctx=canvas.getContext('2d');host.appendChild(canvas);let yaw=.65,pitch=-.7,drag=null;
const rows=D.per_image.filter(x=>x.registered),pc=rows.map(x=>x.center),all=pc.concat(D.point_cloud.map(x=>x[0])),mid=[0,1,2].map(j=>(Math.min(...all.map(x=>x[j]))+Math.max(...all.map(x=>x[j])))/2),extent=Math.max(...[0,1,2].map(j=>Math.max(...all.map(x=>x[j]))-Math.min(...all.map(x=>x[j]))),.001);let dist=extent*2.2;
function proj(p){let x=p[0]-mid[0],y=p[1]-mid[1],z=p[2]-mid[2],x1=Math.cos(yaw)*x+Math.sin(yaw)*z,z1=-Math.sin(yaw)*x+Math.cos(yaw)*z,y1=Math.cos(pitch)*y-Math.sin(pitch)*z1,z2=Math.sin(pitch)*y+Math.cos(pitch)*z1,s=canvas.height/(dist+Math.max(0,z2));return [canvas.width/2+x1*s,canvas.height/2-y1*s];}
function line(a,b,color){let p=proj(a),q=proj(b);ctx.strokeStyle=color;ctx.lineWidth=devicePixelRatio;ctx.beginPath();ctx.moveTo(...p);ctx.lineTo(...q);ctx.stroke();}
function draw(){canvas.width=host.clientWidth*devicePixelRatio;canvas.height=host.clientHeight*devicePixelRatio;canvas.style.width=host.clientWidth+'px';canvas.style.height=host.clientHeight+'px';if(document.getElementById('points').checked){for(let [p,c] of D.point_cloud){let v=proj(p);ctx.fillStyle=`rgb(${c})`;ctx.fillRect(v[0],v[1],2*devicePixelRatio,2*devicePixelRatio);}}rows.forEach((im,i)=>{let c=im.center,R=im.R,s=extent*.05,pp=[];for(let [x,y] of [[-1,-1],[1,-1],[1,1],[-1,1]]){let v=[s*x,s*y,s*1.5];pp.push([0,1,2].map(k=>c[k]+R[0][k]*v[0]+R[1][k]*v[1]+R[2][k]*v[2]));}if(document.getElementById('frustums').checked){pp.forEach(p=>line(c,p,'#e24765'));for(let j=0;j<4;j++)line(pp[j],pp[(j+1)%4],'#e24765');}let v=proj(c);ctx.fillStyle='#b51f43';ctx.beginPath();ctx.arc(v[0],v[1],5*devicePixelRatio,0,Math.PI*2);ctx.fill();ctx.font=`${12*devicePixelRatio}px sans-serif`;ctx.fillText(im.name.replace(/_source_.*/,''),v[0]+7*devicePixelRatio,v[1]-7*devicePixelRatio);});}
canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)};canvas.onpointermove=e=>{if(drag){yaw+=(e.clientX-drag[0])*.007;pitch+=(e.clientY-drag[1])*.007;drag=[e.clientX,e.clientY];draw();}};canvas.onpointerup=()=>drag=null;canvas.onwheel=e=>{e.preventDefault();dist=Math.max(.0001,dist*Math.exp(e.deltaY*.001));draw();};document.getElementById('reset').onclick=()=>{yaw=.65;pitch=-.7;dist=extent*2.2;draw();};document.getElementById('top').onclick=()=>{pitch=-1.55;draw();};document.getElementById('side').onclick=()=>{pitch=-.05;draw();};document.getElementById('points').onchange=draw;document.getElementById('frustums').onchange=draw;new ResizeObserver(draw).observe(host);
</script></body></html>"""
    (out / "index.html").write_text(html.replace("__TITLE__", payload["case"]).replace("__DATA__", data), encoding="utf-8", newline="\n")


def plot_overview(out, payload):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for font in ["/mnt/c/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/msyh.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if Path(font).exists():
            font_manager.fontManager.addfont(font)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    rows = [r for r in payload["per_image"] if r["registered"]]
    fig = plt.figure(figsize=(16, 10), facecolor="#f5f7fb")
    ax = fig.add_subplot(221, projection="3d")
    xyz = np.array([p[0] for p in payload["point_cloud"]])
    rgb = np.array([p[1] for p in payload["point_cloud"]]) / 255
    if len(xyz):
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=1, c=rgb, alpha=.5, depthshade=False)
    for r in rows:
        c, R = np.array(r["center"]), np.array(r["R"])
        s = max(np.ptp(xyz, axis=0).max() * .065, 1e-5) if len(xyz) else .1
        for u, v in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
            corner = c + R.T @ np.array([s * u, s * v, s * 1.4])
            ax.plot(*zip(c, corner), color="#dd3b57", lw=.7)
        ax.text(*c, r["name"].split("_source")[0], fontsize=8)
    centers = np.array([r["center"] for r in rows])
    ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], "--", color="#1c74a8", lw=1)
    ax.set_title("相机位置 + SfM 稀疏点（任意尺度）")
    ax.set_box_aspect(np.ptp(np.vstack([xyz, centers]), axis=0) + 1e-9)
    ax2 = fig.add_subplot(222)
    names = [r["name"] for r in rows]
    idx = {n: i for i, n in enumerate(names)}
    grid = np.zeros((len(names), len(names)))
    for p in payload["matches"]:
        if p["a"] in idx and p["b"] in idx:
            grid[idx[p["a"]], idx[p["b"]]] = grid[idx[p["b"]], idx[p["a"]]] = p["inliers"]
    im = ax2.imshow(grid, cmap="viridis", vmin=0, vmax=max(grid.max(), 1))
    ax2.set_xticks(range(len(rows)), [r["name"].split("_source")[0].replace("view_", "") for r in rows])
    ax2.set_yticks(range(len(rows)), [r["name"].split("_source")[0] for r in rows])
    ax2.set_title("图像对几何验证 inlier 匹配数")
    for i in range(len(rows)):
        for j in range(len(rows)):
            if i != j:
                ax2.text(j, i, str(int(grid[i, j])), ha="center", va="center", fontsize=7,
                         color="white" if grid[i, j] < grid.max() * .6 else "black")
    fig.colorbar(im, ax=ax2, fraction=.046, pad=.04)
    ax3 = fig.add_subplot(223)
    ax3.bar(range(len(rows)), [r["median_error_px"] for r in rows], color="#2b7ea1")
    ax3.set_xticks(range(len(rows)), [r["name"].split("_source")[0] for r in rows], rotation=45, ha="right")
    ax3.set_ylabel("像素")
    ax3.set_title("每帧重投影中位误差")
    ax3.grid(axis="y", alpha=.25)
    ax4 = fig.add_subplot(224)
    ax4.axis("off")
    text = (f"case: {payload['case']}\n\n注册图像: {payload['registered']}/{payload['input_count']}\n"
            f"三维点: {payload['num_points']:,}\n观测数: {payload['num_observations']:,}\n"
            f"中位重投影误差: {payload['median_reproj_error_px']:.4f} px\n"
            f"RMS 重投影误差: {payload['rms_reproj_error_px']:.4f} px\n\n"
            "单 OPENCV 相机；原图 SIFT，45 对穷举匹配；\n增量 SfM + 全局 BA。\n\n"
            "位姿为任意尺度；无绝对真值，不宣称\n轨迹的绝对精度。交互视图见 index.html。")
    ax4.text(.04, .97, text, va="top", fontsize=12, linespacing=1.7)
    fig.suptitle("COLMAP SfM 位姿重建质量检查", fontsize=20)
    fig.tight_layout(rect=[0, 0, 1, .95])
    fig.savefig(out / "overview.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
