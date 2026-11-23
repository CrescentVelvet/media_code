# Depth Pro 2 出图台账

记录 `D:/ppt/figures/depth_pro_2/` 中汇报素材的来源与再生成方式。实验结果以 `D:/output/results/depth_pro_2_results/` 为准；本目录仅改变展示，不改变深度、相机或 PLY。

## 2026-11-23 — PPT 视频封面、点云预览与三帧总览

- 一次性生成脚本：`_build/make_delivery_assets.py`，不入 git。
- 运行环境：WSL2 `Ubuntu2404`，`~/miniconda3/envs/da3_human/bin/python`；依赖 NumPy、Matplotlib、Pillow、plyfile、OpenCV。
- 输入：`human/`、`router/` 的 `pcd/00.ply～02.ply`、`fused_no_registration.ply`、`manifest.json`、`video_frames/*.jpg`、`splat/*.png`、`pcd_depth/*.png` 及两个 MP4。
- 点云：每文件最多约 85,000 点，按采集相机水平/上/前方向显示；正交 `elev=12, azim=-90`；颜色仅加亮，预览不做配准、滤波或几何修正。
- 视频：OpenCV 完整解码并取中间帧作为 poster；四个视频均 9 帧 / 1 fps / 9 秒。主视频原尺寸人像 4608×1598、路由器 3456×1598。
- 素材：`{scene}_pcd_world_views.png`、`{scene}_{video_frames,splat,pcd_depth}_triptych.jpg`、四个视频 poster。`{scene}` 为 `human` 或 `router`。
- 最终 P11/P14 直接嵌入原始 splat / PCD depth PNG，不使用 triptych 作为正式逐帧展示，避免文件名和图像过小。

再生成：

```bash
wsl -d Ubuntu2404 -- bash -lc '~/miniconda3/envs/da3_human/bin/python "/mnt/d/ppt/figures/depth_pro_2/_build/make_delivery_assets.py"'
```

## 2026-11-23 — 最终 PPT 页面验收

- `_build/render_delivery.py` 位于 `D:/ppt/_src/20261123-DepthPro2算法实验与SA管线分析/_build/`。
- 本机 PowerPoint COM 只读打开，不保存 PPTX；导出 18 张 1280×720 PNG 至 `_build/final_render/Slide1.PNG～Slide18.PNG`。
- 重点目检 P9–15：两段视频封面画幅正常；两组点云预览非空；六张 splat / PCD depth 逐帧图和文件名完整；P15 的 target/SA/Zip 三图仍存在。
- 文件级报告：`D:/output/results/depth_pro_2_results/ppt_delivery_validation.json`。两段内嵌 MP4 与源视频 SHA-256 一致，8 个 PLY 链接目标存在，页码与播放 timing 通过。
- 生成截图前必须最后一次嵌入视频；生成后不得再用 WorkBuddy 编辑器保存 PPTX。

```powershell
python "D:\ppt\_src\20261123-DepthPro2算法实验与SA管线分析\_build\render_delivery.py"
```
