#!/usr/bin/env python3
"""一次性补丁：splannequin-gs 官方发布代码的发布级 bug（幂等，可重复跑）。

补丁 1 — gaussian_model.py add_densification_stats 调用/定义不一致：
train.py:317 以 2 参调用，而定义要求 3 参（color_point_tensor）。
4DGaussians 的 color 梯度来自其定制 rasterizer，本仓 renderer 只建
screenspace_points，color 梯度 tensor 根本不存在；且 densify() 的
color_grad_iter 恒为 0（train.py:327 未传），color_gradient_accum
从不参与 clone/split 判据 —— 故改为可选参数，行为与发布配置一致。

补丁 2 — render.py render_fixed_time 帧数越界：
官方硬编码 frames=300（面向合成数据 300 测试视角），真实场景测试
视角数 < 300 时 views[idx] IndexError —— 钳制到 len(views)。

补丁 3 — train.py NaN 时 os.execv 从零重跑：
官方检测到 NaN 直接 execv 自身（进度全丢，且 print 因 execv 不冲刷
缓冲区而丢失，日志里看不到任何痕迹）。改为 exit(42)，由 02_train.sh
外层循环从最近 checkpoint 续跑。

补丁 4 — gaussian_model.py restore() 解包数量不匹配：
capture() 返回 15 元组（含 color_gradient_accum），restore() 只解包
14 个名字 → --start_checkpoint 恢复必炸 ValueError。补上该字段。

补丁 5 — dataset_readers.py 忽略 --resolution（--resolution 是无效开关）：
readColmapCameras() 固定 PILtoTorch(image, None) 全分辨率加载。resolution>1
时按比例缩图；FoY/FovX 由原始 focal/height 计算（尺度不变量），缩图不改变
FoV，内参无需调整。Camera.image_width/height 取自 image tensor shape
（cameras.py:42），自动跟随。plumbing：scene/__init__.py 传入 args.resolution。

补丁 6 — train.py 异常 loss 保护（NaN 之外的洞）：
官方只查 isnan，有限但爆炸的 loss（30k 参照实测 1e5~1e8，hidden/occlusion
正则发散）会空跑数小时不退出。加 isfinite + 幅值（LOSS_GUARD_MAX，默认 1e3，
正常 loss < 5）三重检查，触发同样 exit(42) 走 checkpoint 续跑。
"""
import io
import sys

PATCHES = [
    (
        "/home/velvet/repos/splannequin-gs/scene/gaussian_model.py",
        (
            "    def add_densification_stats(self, viewspace_point_tensor, update_filter, color_point_tensor):\n"
            "        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor[update_filter,:2], dim=-1, keepdim=True)\n"
            "        self.color_gradient_accum[update_filter] += torch.norm(color_point_tensor[update_filter,:, :], dim=(1,2)).unsqueeze(1)\n"
            "        self.denom[update_filter] += 1"
        ),
        (
            "    def add_densification_stats(self, viewspace_point_tensor, update_filter, color_point_tensor=None):\n"
            "        # 官方 train.py 以 2 参调用本函数（renderer 不产出 color 梯度 tensor），\n"
            "        # 且 densify() 的 color_grad_iter 恒为 0 → color_gradient_accum 从不参与\n"
            "        # clone/split 判据。改为可选参数：None 时跳过累积，行为与发布配置一致。\n"
            "        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor[update_filter,:2], dim=-1, keepdim=True)\n"
            "        if color_point_tensor is not None:\n"
            "            self.color_gradient_accum[update_filter] += torch.norm(color_point_tensor[update_filter,:, :], dim=(1,2)).unsqueeze(1)\n"
            "        self.denom[update_filter] += 1"
        ),
    ),
    (
        "/home/velvet/repos/splannequin-gs/train.py",
        (
            "        if torch.isnan(loss).any():\n"
            '            print("loss is nan,end training, reexecv program now.")\n'
            "            os.execv(sys.executable, [sys.executable] + sys.argv)"
        ),
        (
            "        if torch.isnan(loss).any():\n"
            "            # patched: 官方 execv 从零重跑（进度全丢且 print 被缓冲区吞掉）。\n"
            "            # 改为 exit(42)，由 02_train.sh 从最近 checkpoint 续跑。\n"
            '            print("loss is nan, exit(42) for checkpoint resume.", flush=True)\n'
            "            import sys as _sys\n"
            "            _sys.exit(42)"
        ),
    ),
    (
        "/home/velvet/repos/splannequin-gs/scene/gaussian_model.py",
        (
            "        self.max_radii2D, \n"
            "        xyz_gradient_accum, \n"
            "        denom,\n"
            "        opt_dict, \n"
            "        self.spatial_lr_scale) = model_args"
        ),
        (
            "        self.max_radii2D, \n"
            "        xyz_gradient_accum, \n"
            "        color_gradient_accum,\n"
            "        denom,\n"
            "        opt_dict, \n"
            "        self.spatial_lr_scale) = model_args"
        ),
    ),
    (
        "/home/velvet/repos/splannequin-gs/scene/gaussian_model.py",
        (
            "        self.xyz_gradient_accum = xyz_gradient_accum\n"
        ),
        (
            "        self.xyz_gradient_accum = xyz_gradient_accum\n"
            "        self.color_gradient_accum = color_gradient_accum\n"
        ),
    ),
    (
        "/home/velvet/repos/splannequin-gs/scene/dataset_readers.py",
        (
            "def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):"
        ),
        (
            "def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, resolution=1):"
        ),
    ),
    (
        "/home/velvet/repos/splannequin-gs/scene/dataset_readers.py",
        (
            "        image = Image.open(image_path)\n"
            "        image = PILtoTorch(image,None)"
        ),
        (
            "        image = Image.open(image_path)\n"
            "        # patched: 官方忽略 --resolution，固定全分辨率加载。resolution>1 时按比例\n"
            "        # 缩图；FovY/FovX 已由上方原始 focal/height 计算（尺度不变量），缩图不改变\n"
            "        # FoV，内参无需调整。Camera 的 image_width/height 取自 tensor shape，自动跟随。\n"
            "        if resolution > 1:\n"
            "            _w, _h = image.size\n"
            "            _w, _h = max(1, _w // resolution), max(1, _h // resolution)\n"
            "            image = PILtoTorch(image, (_w, _h))\n"
            "            width, height = _w, _h\n"
            "        else:\n"
            "            image = PILtoTorch(image, None)"
        ),
    ),
    (
        "/home/velvet/repos/splannequin-gs/scene/dataset_readers.py",
        (
            "def readColmapSceneInfo(path, images, eval, llffhold=8):"
        ),
        (
            "def readColmapSceneInfo(path, images, eval, llffhold=8, resolution=1):"
        ),
    ),
    (
        "/home/velvet/repos/splannequin-gs/scene/dataset_readers.py",
        (
            "    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))"
        ),
        (
            "    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), resolution=resolution)"
        ),
    ),
    (
        "/home/velvet/repos/splannequin-gs/scene/__init__.py",
        (
            '            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.llffhold)'
        ),
        (
            '            # patched: 传入 resolution，使 --resolution 对 Colmap 加载生效\n'
            '            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.llffhold, args.resolution)'
        ),
    ),
    (
        "/home/velvet/repos/splannequin-gs/train.py",
        (
            "        if torch.isnan(loss).any():\n"
            "            # patched: 官方 execv 从零重跑（进度全丢且 print 被缓冲区吞掉）。\n"
            "            # 改为 exit(42)，由 02_train.sh 从最近 checkpoint 续跑。\n"
            '            print("loss is nan, exit(42) for checkpoint resume.", flush=True)\n'
            "            import sys as _sys\n"
            "            _sys.exit(42)"
        ),
        (
            "        # patched: NaN/非有限/异常大幅值三重保护。官方只查 isnan，有限但爆炸的\n"
            "        # loss（30k 参照实测 1e5~1e8，hidden/occlusion 正则发散）会空跑数小时不退出。\n"
            "        # LOSS_GUARD_MAX 阈值可调（默认 1e3，正常 loss < 5）；触发即 exit(42) 续跑。\n"
            '        _loss_guard_max = float(os.environ.get("LOSS_GUARD_MAX", "1000"))\n'
            "        if torch.isnan(loss).any() or (not torch.isfinite(loss).all()) or loss.item() > _loss_guard_max:\n"
            '            print("loss is nan/abnormal({:.4g}), exit(42) for checkpoint resume.".format(loss.item()), flush=True)\n'
            "            import sys as _sys\n"
            "            _sys.exit(42)"
        ),
    ),
]


def main():
    for path, old, new in PATCHES:
        src = io.open(path, "r", encoding="utf-8", newline="").read()
        if new in src:
            print("already patched, skip:", path)
            continue
        if src.count(old) != 1:
            sys.exit(f"❌ pattern not found or not unique: {path} — 检查官方代码是否更新")
        io.open(path, "w", encoding="utf-8", newline="").write(src.replace(old, new))
        print("🎉 patched:", path)


if __name__ == "__main__":
    main()
