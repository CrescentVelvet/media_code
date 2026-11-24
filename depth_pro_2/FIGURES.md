# Depth Pro 2 出图台账

记录 `D:/ppt/figures/depth_pro_2/` 中汇报素材的来源与再生成方式。实验结果以 `D:/output/results/depth_pro_2_results/` 为准；本目录仅改变展示，不改变深度、相机或 PLY。

## 2026-11-24 — 独立单帧 PCD 对照与 44 页 PPT

- 两组 `single_frame/own_camera_splat_depth/` 各有三帧 splat/PCD depth、相机 JSON 与全点审计；正式素材为 `assets/{scene}_single_{00..02}_{splat,pcd_depth}.png`，与实际渲染 SHA-256 一致。单帧推理与相机口径见 `README_wsl.md`、`EXPERIMENTS.md`。
- `build_single_frame_splat_depth_ppt.py` 必须从已验收 42 页备份生成，不能在 44 页上重复插入。新页为 **P14 佳佳抱枕、P31 路由器**，各位于该组三视角 10 万点投影之后；每页六张原比例图与可编辑说明，保留全部既有媒体/链接。
- `verify_single_frame_splat_depth_ppt.py` 还原临时 42 页顺序并复用原 42 页验收，再核对新增 12 图、独立 T=1 审计、源深度与页脚。最终 **44 页 / 88 图片源 / 2 视频 / 8 PLY / 零警告**。它复用 `verify_full_view_splat_depth_ppt.py` 的 SciPy 相机检查，在 Windows 下把 `PYTHONPATH` 指向同时含 PPT 依赖与 SciPy 的环境运行，不改变 PPT 内 `D:/` PLY 链接。
- 只读 PowerPoint 导出 `_build/final_render_single_frame/Slide1.PNG～Slide44.PNG`；P14/P31 实际目检通过，P9/P24 视频未重嵌或重编码。完成编辑与保存后不再由旧预览实例回写。

- `render_full_view_splat_depth.py`：两组各九视角、每视角 splat/depth，全部有效点不抽样；输出在两组 `view_splat_depth_full_9views/`。PPT 素材为 `assets/{scene}_full_{view}_{kind}.png`，共 36 张，与实际渲染 SHA-256 一致。
- `build_full_view_splat_depth_ppt.py`：从 24 页输入生成 42 页，每组九页分别插到该组 10 万点页面后。每添加一页立即移动 XML 页序元素；不重命名 slide part、不做末次文件名重排。`verify_full_view_splat_depth_ppt.py` 验证 76 个图片源、2 视频、8 PLY、相机、点数与源哈希。
- 完整构建顺序：先从 18 页备份运行 `build_view_splat_depth_ppt.py` 得 24 页，再以该文件作为 `PPTX_INPUT` 运行全点构建；不要在 42 页结果上重复插入。P9/P24 视频和 P10/P25 PLY 在最终页序确定后重嵌。`_build/render_delivery.py` 只读导出 42 页，不再经编辑器保存。
- 最终新增 P15–23 佳佳、P30–38 路由器；P39–42 保留 SA 与结论。覆盖率、抽样与虚拟相机均逐页标注，外插不作为采集真值。

WSL 渲染：

```bash
cd /mnt/c/code/media_code
for SCENE in human router; do
  env SCENE_DIR=/mnt/d/output/results/depth_pro_2_results/$SCENE \
    ~/miniconda3/envs/sharp/bin/python depth_pro_2/render_full_view_splat_depth.py
done
```



- 两组 `view_splat_depth_100k/` 各有固定 10 万点的输入/内插/外插 splat 与 PCD depth 原图、NPY、相机 JSON、抽样 PLY 与审计。三视角共用同一随机子集；内插 `t=0.5`、外插 `t=-0.5`，旋转为 SLERP 等价 rotvec，中心线性变化。PPT 原图副本 `_src/20261123-DepthPro2算法实验与SA管线分析/assets/{scene}_{view}_{kind}.png`，共 12 张。
- `depth_pro_2/render_view_splat_depth.py` 生成图像与相机；`build_view_splat_depth_ppt.py` 在 18 页输入上插入 P12–14、P18–20，统一页脚并在最后重嵌 P9/P15 视频及 P10/P16 PLY 链接。每次运行必须从已验收的 18 页备份开始，不能在 24 页结果上重复插入。
- `verify_view_splat_depth_ppt.py` 按 `presentation.xml` 的显示页序读取 slide part 与各自 rels，防止插页后把显示页号误当 `slideN.xml` 编号。最终 24 页、40 个图片源哈希、2 段视频、8 个 PLY、全部页脚、相机和抽样检查通过。
- `_build/render_delivery.py` 只读 PowerPoint 导出 24 页；P9/P15 均检测到一个媒体元素。新增六页和 P21 SA 三图已目检；覆盖不是精度，外插不对应采集图像。

WSL 中复现两组渲染（`sharp` 环境）：

```bash
cd /mnt/c/code/media_code
for SCENE in human router; do
  env SCENE_DIR=/mnt/d/output/results/depth_pro_2_results/$SCENE \
    VIEW_POINT_COUNT=100000 VIEW_SEED=20261123 VIEW_EXTRAPOLATION=-0.5 \
    ~/miniconda3/envs/sharp/bin/python depth_pro_2/render_view_splat_depth.py
done
```

Windows 下将 12 张实际 PNG 复制到上述 `assets/`（与输出逐文件 SHA-256 一致），然后：

```powershell
$env:PPTX_PATH = "D:\ppt\20261123-DepthPro2算法实验与SA管线分析.pptx"
$env:ASSET_DIR = "D:\ppt\_src\20261123-DepthPro2算法实验与SA管线分析\assets"
$env:RESULTS_DIR = "D:\output\results\depth_pro_2_results"
$env:PPTX_BACKUP = "D:\output\_backup\depth_pro_2_before_view_pages_20261123.pptx"
python C:\code\media_code\depth_pro_2\build_view_splat_depth_ppt.py
python C:\code\media_code\depth_pro_2\verify_view_splat_depth_ppt.py
python D:\ppt\_src\20261123-DepthPro2算法实验与SA管线分析\_build\render_delivery.py
```

## 2026-11-23 — 点云近景与圆盘诊断

- `_build/refine_pcd_and_splat_assets.py`：从真实 PLY 生成 P10 人物中心近景；相机投影小角度视差、有限点绘制，只改变预览，不裁切 PLY。`_build/make_delivery_assets.py` 后续全房间预览改名为 `{scene}_pcd_world_full_views.png`，不覆盖人物近景 `human_pcd_world_views.png`。
- `figures/splat_projection_audit.json`：记录三帧圆盘半径、覆盖、PCD depth/原始深度着色相同像素比例。P11 `human_splat_disc_diagnostic.png` 使用每 8 像素 1 点的诊断，不是新实验输出。
- `_build/refine_pcd_and_splat_slides.py`：一次绑定近景与诊断、恢复正确图片源、8 个 PLY 链接和两个 MP4，固定 P10/P11 不重叠布局；去重后 P11 恰为六个原始图+一个诊断图。
- 原 Z 排序 bug 归因已撤回；原始 splat/PCD depth 与备份逐字节一致，统计逐字段一致。只加 `(N,3)` 输入断言，不改变实验几何。

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

## 2026-11-23 — 图片映射修正

- 旧“图片数量齐全”不足以保证内容正确。源哈希实测发现 P1 错用 SA 图、P3/P6 架构与 RGB 对调、P15 第三张错用 target；现已按 PPT 页源码恢复六处映射，P15 恢复为 target/SA/Zip。
- `_build/repair_images_links_and_media.py`：关闭旧预览写实例后，一次写入正确图片关系、清除无效 `0000df`～`0000e2` PLY 关系、绑定 8 个实际路径，并嵌入两个视频。之后不得再让旧编辑器保存。
- `_build/verify_delivery.py`：新增 **27 个图片源 SHA-256** 检查（P1/3/5–10/12/13/15 源素材，P11/14 的 12 张逐帧原图），同时验证两个 MP4、8 个 PLY 链接、页码和 timing。
- 只读渲染复核 P1/P3/P6/P15：封面恢复真实 RGB/深度，P3 为官方架构，P6 为路由器 RGB/深度，P15 三图不再重复 target。

## 2026-11-23 — 最终 PPT 页面验收

- `_build/render_delivery.py` 位于 `D:/ppt/_src/20261123-DepthPro2算法实验与SA管线分析/_build/`。
- 本机 PowerPoint COM 只读打开，不保存 PPTX；导出 18 张 1280×720 PNG 至 `_build/final_render/Slide1.PNG～Slide18.PNG`。
- 重点目检 P9–15：两段视频封面画幅正常；两组点云预览非空；六张 splat / PCD depth 逐帧图和文件名完整；P15 的 target/SA/Zip 三图仍存在。
- 文件级报告：`D:/output/results/depth_pro_2_results/ppt_delivery_validation.json`。两段内嵌 MP4 与源视频 SHA-256 一致，8 个 PLY 链接目标存在，页码与播放 timing 通过。
- 生成截图前必须最后一次嵌入视频；生成后不得再用 WorkBuddy 编辑器保存 PPTX。

```powershell
python "D:\ppt\_src\20261123-DepthPro2算法实验与SA管线分析\_build\render_delivery.py"
```
