#!/usr/bin/env python3
"""26c_render_blend_holdout.py -- 从已有人物 .blend 生成独立静态 GT 包。

调用 Blender 后台；只读 .blend，不写资产。利用现有生成器的相机、光照和
pass 约定，保持与 N0 相同的合成测量定义。产物仅落 D:/dataset/human_fbx_gt。

Env vars:
  BLEND_FILE  输入 .blend 的 WSL 路径（必填）
  OUT_DIR     全新输出目录（必填）
  STATIC_FRAME 默认 1；在该帧烘焙网格，后续只移动相机，不播放资产动作
  N_VIEWS     默认 48
"""
import json
import os
import shutil
from pathlib import Path

import bpy


HERE = Path(__file__).resolve().parent


def main():
    source = Path(os.environ["BLEND_FILE"])
    out = Path(os.environ["OUT_DIR"])
    if out.exists():
        raise RuntimeError("Output must not exist: " + str(out))
    if not source.is_file():
        raise FileNotFoundError(source)
    bpy.ops.wm.open_mainfile(filepath=str(source))
    mesh_objects = [o for o in bpy.data.objects if o.type == "MESH"]
    report = {"source": str(source), "meshes": [{"name": o.name, "vertices": len(o.data.vertices),
              "materials": [m.name if m else None for m in o.data.materials]} for o in mesh_objects]}
    # 导入阶段只核验，不保存或覆盖资产；加载器沿用当前已修复的相机/深度实现。
    # 用脚本实际位置定位，Windows Blender 与 WSL 共用同一份只读参考实现。
    script = HERE.parent / "flash_human/01_render_gt.py"
    text = script.read_text(encoding="utf-8")
    start = text.index("# ---------------------------------------------------------------- 归一化")
    end = text.index("# ---------------------------------------------------------------- 灯光与世界")
    namespace = {"bpy": bpy, "mathutils": __import__("mathutils"), "os": os,
                 "json": __import__("json"), "math": __import__("math"), "random": __import__("random"),
                 "_re": __import__("re"), "MODE": "face", "_junk_re": __import__("re").compile(r"^(Cube|Plane|Sphere|Circle|Grid)([._]\d+)?$"),
                 "TARGET_ZFRAC": .85, "FRONT_AZIM_OVERRIDE": float(os.environ.get("FRONT_AZIM", "-1")),
                 "FBX_PATH": str(source), "OUT_DIR": str(out),
                 "N_VIEWS": int(os.environ.get("N_VIEWS", "48")), "RES": 1024,
                 "SEED": 0, "Focal_MM": 50., "DIST_MULT": 1., "ELEV_MIN": -15.,
                 "ELEV_MAX": 25., "YAW_SPAN": 150., "LIGHT_JITTER": .3,
                 "_arg": lambda name, default: default, "_arg_int": lambda name, default: default,
                 "_arg_float": lambda name, default: default}
    if namespace["FRONT_AZIM_OVERRIDE"] >= 0:
        namespace["FRONT_AZIM"] = namespace["FRONT_AZIM_OVERRIDE"]
    # 官方参考脚本会在末尾以 import 覆盖同名全局量，恢复宿主 os 后再写产物。
    # .blend 可带摄影棚/体积辅助网格，不属于人物；按原始资产显式排除。
    excluded_objects = {"Lighting Setup Generator", "Volume"}
    namespace["imported"] = [o for o in bpy.context.scene.objects if o.type == "MESH"
                            and o.name not in excluded_objects]
    for o in list(bpy.context.scene.objects):
        if o.type in ("LIGHT", "CAMERA"):
            bpy.data.objects.remove(o, do_unlink=True)
    exec(text[start:end], namespace, namespace)
    # 原参考代码修改 object.matrix_world 后未清绑定求值缓存，部分 .blend
    # 会把归一化变换施加在已经变形的网格上。重置绑定 REST→POSE 后重新求值。
    for arm in (o for o in bpy.data.objects if o.type == "ARMATURE"):
        arm.data.pose_position = "REST"
    bpy.context.view_layer.update()
    for arm in (o for o in bpy.data.objects if o.type == "ARMATURE"):
        arm.data.pose_position = "POSE"
    bpy.context.view_layer.update()
    # 相机循环的 frame_set(i+1) 会播放资产动作；先烘焙同一静态姿态，避免眼球/骨骼
    # 在中后段脱节，也保证 48 机位观测的是同一个静态几何。
    static_frame = int(os.environ.get("STATIC_FRAME", "1"))
    bpy.context.scene.frame_set(static_frame)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    baked = []
    for obj in list(bpy.context.scene.objects):
        if obj.type != "MESH" or obj.name in excluded_objects:
            continue
        evaluated = obj.evaluated_get(depsgraph)
        mesh = bpy.data.meshes.new_from_object(evaluated, depsgraph=depsgraph)
        new_obj = bpy.data.objects.new(obj.name + "__static", mesh)
        new_obj.matrix_world = obj.matrix_world.copy()
        bpy.context.scene.collection.objects.link(new_obj)
        baked.append(new_obj)
    namespace["imported"] = baked
    for obj in list(bpy.context.scene.objects):
        if obj not in baked:
            bpy.data.objects.remove(obj, do_unlink=True)
    out.mkdir(parents=True)
    namespace["static_frame"] = static_frame
    for sub in ("rgb", "depth", "normal"):
        (out / sub).mkdir()
    namespace["COLOR_LOOK"] = os.environ.get("COLOR_LOOK", "Medium High Contrast")
    namespace["baked"] = baked
    render_text = text[text.index("# ---------------------------------------------------------------- 灯光与世界"):]
    render_text = render_text.replace('scene.view_settings.look = "AgX - Medium High Contrast"',
                                      'scene.view_settings.look = COLOR_LOOK')
    exec(render_text, namespace, namespace)
    report["meta"] = namespace["meta"]
    report["static_frame"] = static_frame
    report["static_mesh_bake"] = "evaluated depsgraph meshes at one frame; armatures/actions removed before camera rendering"
    report["source_sha256"] = __import__("hashlib").sha256(source.read_bytes()).hexdigest()
    with open(out / "asset_inventory.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("BLEND_GT_DONE", out, flush=True)


if __name__ == "__main__":
    main()
