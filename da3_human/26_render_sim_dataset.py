#!/usr/bin/env python3
"""Blender 仿真数据集渲染脚本（规格 = 24_ue_render_spec.md，本地验证用）。

用法（模型是 .blend 时直接当输入文件；是 fbx/obj 时用 MODEL_PATH 导入）：
  blender -b "Asian Girl 5.0.blend" -P 26_render_sim_dataset.py
  blender -b --python 26_render_sim_dataset.py   # 配合 MODEL_PATH=<fbx/obj>

输出（目录结构 / cameras.json / meta.json 与交付规格 §3 完全一致）：
  A 稀疏 13 张（s00~s12, az=-90+15i, r=1.2m, h=1.45m）
  B 三摄 3 张（wide/main/tele, 13/24/96mm 等效, 光心有小偏移）
  C 评估环 125 张（25az×14.4° × 5el, r=1.2m）
  depth/ 16-bit PNG（毫米，uint16；radial 深度）+ cameras.json（w2c 4x4 + K）

坐标转换（Blender z-up 右手 → 本仓 y-up 右手 + OpenCV 相机）：
  w2c = inv(D @ M_bl @ C_bl)，D 同 UE 版规格，C_bl = diag(1,-1,-1)

Env vars:
  MODEL_PATH   fbx/obj 路径（不传则用 -b 已打开的 .blend）
  OUT_DIR      输出目录（必填）
  TARGET_H     人物目标身高 m，默认 1.70
  FACING_DEG   模型绕 Z 旋转角（smoke test 看图校准朝向），默认 0
  W, H         渲染分辨率，默认 3072x4096
  ENGINE       eevee|cyles，默认 eevee
  QUICK        置 1 只渲 s06/wide/tele/e00_m00 四张做冒烟测试
"""
import json
import math
import os
import shutil
import sys
import time
import zlib

import bpy
import numpy as np
from mathutils import Matrix, Vector

# ---------------------------------------------------------------- 规格常量
AZ_A = [-90.0 + 15.0 * i for i in range(13)]          # A: 稀疏 13 视角
AZ_EVAL_STEP, AZ_EVAL_N = 14.4, 25                     # C: 评估环 25 方位
ELS = [-20.0, -10.0, 0.0, 10.0, 20.0]                  # C: 5 俯仰
R_MAIN, H_MAIN = 1.2, 1.45                             # A/C: r 与相机高度
B_CAMS = [                                             # B: 三摄（文件名, 焦距mm, 光心偏移 m）
    ("wide", 13.0, (0.014, 0.004, 0.002)),
    ("main", 24.0, (0.0, 0.0, 0.0)),
    ("tele", 96.0, (-0.012, 0.006, 0.003)),
]
FOCUS_FACE = 1.55                                      # B: look-at 面部中心高度
CX_D, CY_D = 36.0, 36.0                                # 传感器竖边 36mm（竖幅长边=36mm 基准）

D = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], np.float64)  # 世界换基
C_BL = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], np.float64)  # 相机换基


# ---------------------------------------------------------------- 基础工具
def log(m):
    print(m, flush=True)


def write_png16(path, arr_mm):
    """手写 16-bit 灰度 PNG（值=毫米），绕开 Blender 色彩管理，精确可控。"""
    h, w = arr_mm.shape
    raw = b"".join(b"\x00" + arr_mm[i].astype(">u2").tobytes() for i in range(h))

    def chunk(tag, data):
        c = tag + data
        return len(data).to_bytes(4, "big") + c + zlib.crc32(c).to_bytes(4, "big")

    ihdr = (w.to_bytes(4, "big") + h.to_bytes(4, "big") + b"\x10\x00\x00\x00\x00")
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
                + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def look_at(obj, target):
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


_COMP = {"ng": None, "rl": None, "comp": None}


def setup_compositor(scene):
    """合成器把指定 pass 接到组输出（5.x 的 Composite 节点已移除，走 group interface）。

    Blender 5.x 踩坑实录：Render Result 不暴露 passes、后台模式 .pixels 恒为空、
    File Output 只支持 EXR MULTILAYER（读回 size=(0,0)）、CompositorNodeComposite 移除。
    最终方案：RGB/深度两次渲染，深度那次把 Depth 接到组输出，save_render 存
    **单层 EXR**（scene-linear float32 无损），bpy 读回转径向后写 PNG16。
    """
    ng = bpy.data.node_groups.new("sim_comp", 'CompositorNodeTree')
    scene.compositing_node_group = ng
    scene.render.use_compositing = True
    ng.interface.new_socket(name='Image', in_out='OUTPUT', socket_type='NodeSocketColor')
    rl = ng.nodes.new('CompositorNodeRLayers')
    outn = ng.nodes.new('NodeGroupOutput')
    _COMP.update({"ng": ng, "rl": rl, "outn": outn})
    set_comp_mode("rgb")
    return ng


def set_comp_mode(mode):
    ng, rl, outn = _COMP["ng"], _COMP["rl"], _COMP["outn"]
    for l in list(ng.links):
        if l.to_node == outn:
            ng.links.remove(l)
    src = rl.outputs["Image"] if mode == "rgb" else rl.outputs["Depth"]
    ng.links.new(src, outn.inputs["Image"])


def read_exr_depth(exr_path, H, W):
    """读单层 EXR（z-depth，米，top-down HxW）。"""
    img = bpy.data.images.load(exr_path)
    img.colorspace_settings.name = 'Non-Color'    # 防任何传递函数碰数值
    buf = np.array(img.pixels[:], np.float32).reshape(H, W, 4)[..., 0]
    bpy.data.images.remove(img)
    return np.flipud(buf)   # bpy 行序 bottom-up → 翻成 top-down 与 RGB 对齐


def z_to_radial(z_m, H, W, fx, fy, cx, cy):
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    mult = np.sqrt(1.0 + ((u - cx) / fx) ** 2 + ((v - cy) / fy) ** 2)
    return z_m * mult


def reset_pose():
    """ armature 姿态归零（回 rest pose）；网格烘焙了姿态则保持原样。"""
    n = 0
    for o in bpy.context.scene.objects:
        if o.type == 'ARMATURE' and o.pose:
            for pb in o.pose.bones:
                pb.matrix_basis = Matrix.Identity(4)
                n += 1
    if n:
        bpy.context.view_layer.update()
        log(f"🧍 姿态归零: {n} 根骨骼（rest pose）")


# ---------------------------------------------------------------- 场景构建
def clean_scene():
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()          # 删光（cameras/lights/ground 都不要模型的）


def import_model(model_path):
    ext = os.path.splitext(model_path)[1].lower()
    if ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=model_path)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=model_path)
    else:
        sys.exit(f"❌ 不支持的模型格式: {ext}")


def model_roots():
    return [o for o in bpy.context.scene.objects if o.parent is None]


def normalize_model(target_h, facing_deg):
    """缩放至目标身高、脚底 z=0、水平居中、按 FACING_DEG 绕 Z 旋转。"""
    roots = model_roots()
    if not roots:
        sys.exit("❌ 场景里没有模型对象")
    bpy.context.view_layer.update()

    def bbox_world():
        lo = np.array([np.inf] * 3)
        hi = np.array([-np.inf] * 3)
        for o in roots:
            if o.type != 'MESH':
                continue
            for c in o.bound_box:
                p = o.matrix_world @ Matrix.Translation(c).to_translation()
                lo = np.minimum(lo, np.array(p))
                hi = np.maximum(hi, np.array(p))
        return lo, hi

    lo, hi = bbox_world()
    h0 = hi[2] - lo[2]
    if h0 < 1e-3:
        sys.exit("❌ 模型包围盒高度异常")
    s = target_h / h0
    rz = Matrix.Rotation(math.radians(facing_deg), 4, 'Z')
    for r in roots:
        r.matrix_world = rz @ Matrix.Scale(s, 4) @ r.matrix_world
    bpy.context.view_layer.update()
    lo, hi = bbox_world()
    shift = Matrix.Translation((-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]))
    for r in roots:
        r.matrix_world = shift @ r.matrix_world
    bpy.context.view_layer.update()
    lo, hi = bbox_world()
    log(f"📐 模型归一化: 身高 {hi[2] - lo[2]:.3f} m, 脚 z={lo[2]:.4f}, 中心=({(lo[0]+hi[0])/2:.3f}, {(lo[1]+hi[1])/2:.3f})")


def add_env():
    """地板 + 背景墙(+Y 向 4m) + 太阳灯 + 前侧 Area 补光。"""
    bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, 0))
    floor = bpy.context.object
    floor.name = "floor"
    fm = bpy.data.materials.new("floor_mat")
    fm.diffuse_color = (0.55, 0.55, 0.55, 1.0)
    floor.data.materials.append(fm)

    bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 4.0, 4.0), rotation=(math.radians(90), 0, 0))
    wall = bpy.context.object
    wall.name = "wall"
    wm = bpy.data.materials.new("wall_mat")
    wm.diffuse_color = (0.5, 0.48, 0.46, 1.0)
    wall.data.materials.append(wm)

    bpy.ops.object.light_add(type='SUN', location=(3, -4, 6))
    sun = bpy.context.object
    sun.data.energy = 2.0
    sun.rotation_euler = (math.radians(35), math.radians(-15), math.radians(30))

    bpy.ops.object.light_add(type='AREA', location=(-1.5, -2.5, 2.6))
    area = bpy.context.object
    area.data.energy = 120
    area.data.shape = 'DISK'
    area.data.size = 3.0
    area.rotation_euler = (Vector((0, 0, 1.3)) - area.location).to_track_quat("-Z", "Y").to_euler()


def make_camera(name, focal_mm):
    cam_data = bpy.data.cameras.new(name)
    cam_data.lens = focal_mm
    cam_data.sensor_fit = 'VERTICAL'      # 竖幅：长边对应传感器 36mm
    cam_data.sensor_height = CY_D
    cam_data.sensor_width = CX_D
    cam = bpy.data.objects.new(name, cam_data)
    bpy.context.scene.collection.objects.link(cam)
    return cam


# ---------------------------------------------------------------- 视角定义（blender 世界坐标）
def pos_bl(az_deg, el_deg, r, h_base):
    """本仓规格角度（ours: y-up, az 绕 y, el 抬升）→ blender 坐标。

    ours 相机位 = (r·cosφ·sin az, h_base + r·sinφ, r·cosφ·cos az)，模型面朝 ours +Z。
    ours → bl: (x, y, z)_bl = (x_ours, -z_ours, y_ours)。
    """
    a, e = math.radians(az_deg), math.radians(el_deg)
    x = r * math.cos(e) * math.sin(a)
    y_ = h_base + r * math.sin(e)
    z = r * math.cos(e) * math.cos(a)
    return (x, -z, y_)


def build_views():
    views = []  # (key, subdir, az, el, r, h_base, focal, target_bl, cam_tag)
    for i, az in enumerate(AZ_A):
        views.append((f"s{i:02d}", "images_input", az, 0.0, R_MAIN, H_MAIN, 24.0,
                      (0.0, 0.0, H_MAIN), None))
    for name, focal, off in B_CAMS:
        az0, el0 = 0.0, math.degrees(math.atan2(1.5 - H_MAIN, R_MAIN))  # 正面 1.5m 高
        r_eff = math.hypot(R_MAIN, 1.5 - H_MAIN)
        p = pos_bl(az0, el0, r_eff, H_MAIN)
        p = (p[0] + off[0], p[1] + off[2], p[2] + off[1])  # 偏移: (横向x, 纵深-bl-y, 高度z)
        views.append((name, "images_input", az0, el0, r_eff, H_MAIN, focal,
                      (0.0, 0.0, FOCUS_FACE), name))
        views[-1] = (name, "images_input", az0, el0, r_eff, H_MAIN, focal, (0.0, 0.0, FOCUS_FACE), name, p)
    for i in range(AZ_EVAL_N):
        for el in ELS:
            key = f"e{i:02d}_{'m' if el < 0 else 'p'}{abs(int(el)):02d}"
            views.append((key, "images_eval", i * AZ_EVAL_STEP, el, R_MAIN, H_MAIN, 24.0,
                          (0.0, 0.0, H_MAIN), None))
    return views


# ---------------------------------------------------------------- 渲染
def render_all(out_dir, W, H, engine, quick):
    os.makedirs(os.path.join(out_dir, "images_input"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "images_eval"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "depth"), exist_ok=True)

    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = W, H
    scene.render.resolution_percentage = 100
    for eng in (["BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"] if engine == "eevee" else ["CYCLES"]):
        try:
            scene.render.engine = eng
            break
        except Exception:
            continue
    log(f"🎮 引擎: {scene.render.engine}  {W}x{H}")
    scene.render.image_settings.file_format = 'JPEG'
    scene.render.image_settings.quality = 95
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
    scene.render.film_transparent = False
    scene.world.color = (0.05, 0.05, 0.05)
    scene.view_layers[0].use_pass_z = True
    cam_main = make_camera("cam_main", 24.0)
    scene.camera = cam_main
    # 热身渲染：低分辨率渲 1 帧，让 RLayers 的 Depth 输出实体化（Blender 5.x 行为）
    scene.render.resolution_percentage = 5
    bpy.ops.render.render()
    scene.render.resolution_percentage = 100
    setup_compositor(scene)

    views = build_views()
    if quick:
        keep = {"s06", "wide", "main", "tele", "e00_p00"}
        views = [v for v in views if v[0] in keep]
        log(f"🔥 QUICK 模式: 只渲 {[v[0] for v in views]}")

    cams = {}

    total = len(views)
    for vi, v in enumerate(views):
        key, sub, az, el, r, h_base, focal, target_bl, cam_tag = v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]
        pos = v[9] if len(v) > 9 else pos_bl(az, el, r, h_base)

        cam_main.data.lens = focal
        cam_main.location = pos
        look_at(cam_main, Vector(target_bl))
        bpy.context.view_layer.update()

        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
        fpx = focal * H / CY_D
        K = [[fpx, 0.0, cx], [0.0, fpx, cy], [0.0, 0.0, 1.0]]

        # pass 1: RGB
        set_comp_mode("rgb")
        scene.render.image_settings.file_format = 'JPEG'
        scene.render.image_settings.quality = 95
        scene.render.filepath = os.path.join(out_dir, sub, f"{key}.jpg")
        bpy.ops.render.render(write_still=True)

        # pass 2: 深度（Depth 接组输出，save_render 存单层 EXR，读回转径向 → PNG16）
        set_comp_mode("depth")
        scene.render.image_settings.file_format = 'OPEN_EXR'
        scene.render.image_settings.color_depth = '32'
        exr_path = os.path.join(out_dir, "_depth_tmp.exr")
        bpy.ops.render.render()
        bpy.data.images["Render Result"].save_render(exr_path, scene=scene)
        z = read_exr_depth(exr_path, H, W)
        os.remove(exr_path)
        d = z_to_radial(z, H, W, fpx, fpx, cx, cy)
        write_png16(os.path.join(out_dir, "depth", f"{key}.png"),
                    np.clip(d * 1000.0, 0, 65535).astype(">u2"))

        M_bl = np.array(cam_main.matrix_world, np.float64)
        w2c = np.linalg.inv(D @ M_bl @ C_BL)
        entry = {"w2c": np.round(w2c, 8).tolist(), "K": K, "role": "input" if sub == "images_input" else "eval"}
        if cam_tag:
            entry["cam"] = cam_tag
        cams[key] = entry
        log(f"   [{vi + 1}/{total}] {key}: f={focal:.0f}mm pos=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f}) "
            f"深度中位 {np.median(d):.3f}m")

    with open(os.path.join(out_dir, "cameras.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(cams, f)
    meta = {"seq_id": os.path.basename(out_dir.rstrip("/\\")), "model": os.environ.get("MODEL_NAME", ""),
            "env": "blender_local",
            "W": W, "H": H, "depth_type": "radial", "depth_unit": "mm",
            "engine": scene.render.engine, "facing_deg": float(os.environ.get("FACING_DEG", "0")),
            "ue_version": f"blender-{bpy.app.version_string}", "comment": "local validation render"}
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, ensure_ascii=False)
    log(f"🎉 完成: {total} 张 → {out_dir}（cameras.json {len(cams)} 键）")


def main():
    out_dir = os.environ.get("OUT_DIR", "")
    if not out_dir:
        sys.exit("❌ OUT_DIR not set")
    W, H = int(os.environ.get("W", "3072")), int(os.environ.get("H", "4096"))
    engine = os.environ.get("ENGINE", "eevee")
    quick = os.environ.get("QUICK", "") == "1"

    model_path = os.environ.get("MODEL_PATH", "")
    if model_path:
        clean_scene()
        log(f"📦 导入模型: {model_path}")
        import_model(model_path)
    else:
        log("📦 使用 -b 传入的 .blend（清理自带相机/灯光）")
        for o in list(bpy.context.scene.objects):
            if o.type in {'CAMERA', 'LIGHT'}:
                bpy.data.objects.remove(o, do_unlink=True)

    reset_pose()
    normalize_model(float(os.environ.get("TARGET_H", "1.70")),
                    float(os.environ.get("FACING_DEG", "0")))
    add_env()
    render_all(out_dir, W, H, engine, quick)


if __name__ == "__main__":
    main()
