# media_code 项目长期约定

> 日期明细见 `.workbuddy/memory/YYYY-MM-DD.md`。本文件只留跨会话仍需遵守的结论，后续复审覆盖旧推断。

## da3_human / splatanything 当前路线（2026-11-22）

- 唯一入口 `da3_human/EXPERIMENT_PLAN.md` 顶部 **N0～N2**：修复几何测量 → SA 原生 P0 / ZipSplat FF 同协议头部对照 → 只选一个基座做人物域适配小试。三图指不同机位三张人像；DA3 是历史基线。
- **N0 测量契约已完成，N1 可执行、N2 冻结**。正式 `five_view_rigid_v28/geometry_audit_v2.json`：官方 GSN pose-align=0.9428013563、spacing=1.4887673855、总教师倍率=1.4036118984；五视角 hash、Depth 类型、K、刚体相机与坐标往返通过。测量通过≠几何/属性分支裁决。
- GT 按 manifest 文件路径/hash 对齐：id20→000021，id24→000025，id30→000031，id34→000035，id38→000039。旧 F3 错帧与强结论停用。
- **Blender Depth pass 是轴向 Z**，已用斜视/主点偏移探针验证，不做射线距离除法。ED 是 alpha 归一化的轴向 Z 期望，须配 alpha 与局部 p90。
- 官方 `forward()` 浅拷贝 geometry：PLY、未融合点云、导出相机使用 GSN 后量；`depth/*.npy` 是 GSN 前教师深度，不能共用尺度。过滤/voxel merge 后 PLY 不能按 3×756² 固定点序切片。
- source-only 相机对定义 `X_model=s R X_pkg+T`；正确相机 `R_model=R R_pkg`、`C_model=s R C_pkg+T`；PLY/ED 回米制为 `X_pkg=R.T@(X_model-T)/s`、深度 `/s`。融合前 lifting 的米制深度=`raw_teacher*total_depth_factor/spacing_factor`。`[sR|T]` 不是合法刚体相机。
- 导出相机与刚体 Sim(3) 最大差约 1.02e-7；源视角 RGB 对拍 21.3～24.4dB。五帧 look-at 目标在 prior=C@R_manifest@C 下距主点≤0.0004055px，raw manifest 不通过；官方 K=fx/fy 1050、cx/cy 377.5。target34 左 eye/brow 排除；source38 鼻翼/眼眉按遮挡/误检 QA 排除。
- 五视角诊断：原始紧脸 ED MAE target24/34=6.62/8.07mm；source38 鼻尖融合前/后 MAE=9.96/30.41mm、p90=13.52/66.45mm，平滑子集融合后=16.99/26.35mm。旧鼻翼 218.72mm 为 ROI 污染，不引用。source-only 教师尺度=0.9338547（可见性筛选紧脸）。行向量 model→pkg 必须 `(points-T)@R/s`；旧 `@R.T` 导致表面样本为 0，修复后恢复约 1.0～1.4 万点/视角。无 FBX 时表面最近距离不是独立网格或可见性真值。
- N0 要分 raw/source-only 单一尺度/每视角 oracle，报鼻尖、鼻翼、眼周 p90 和 alpha/轮廓；target 不拟合、不选参。无原始 FBX 时，深度反投影只能称“深度图派生表面”。
- **P0～P3/F1 已结束**：source 增益不迁移到独立 target；静态真人 target 仍碎裂，撤回“静态无害/微动主因”。F1-B 触顶只证明约束活跃。FLAME 本轮拟合不过门，不绑定高斯；不追加 P3/F1/SHARP，不用 P3 作伪 GT，不拼两模型 PLY。
- ZipSplat 只有“更模糊但头部更像”的观察及 29 留出整体值：Zip FF 17.64dB/100% 覆盖、TokenBA 17.27dB、SA pose 13.42dB/82.62%；`regions_holdout=null`，不是脸部几何真值。TokenBA 属测试时优化，不与 FF 混报。
- 人脸 ROI 必须逐视角目检（008 后脑勺、009 玩偶误检）；FaceMesh 原生窄带，必要时多尺度。human_fbx_gt 正确转换 `R'=C R C`、t 不变；旧生成器乘反，历史包须验证，不能凭当前生成器代码定版本。
- 原理/排错 `splatanything/NOTES.md`；结论 `EXPERIMENTS.md`；命令 `README_wsl.md`；图 `FIGURES.md`。DA3 必读 `da3_human/ERRATA.md`。

## 工程纪律与 D 盘落点

- 中文 `<项目名>: <简短描述>`，默认一行，完成即提交；只 add 具体文件并核对 staged diff。AI 不 push、不改远端；动手前读 AGENTS.md，WSL 训练前读 README_wsl.md。
- 主落点 `C:\code\media_code`；服务器 `h100:00950754@69.48.159.15:/data/00950754/code/media_code`。同一改动只在一侧提交，另一侧 `git pull --rebase`；memory 只在本机写；两边 `core.fileMode=false`。
- 删除先判 `is_symlink()`，临时文件备份搬移。只入可复用脚本/规格文档，产物与一次性脚本不入库；JSON 显式 UTF-8/LF。
- 动 D 盘前读 `D:\ppt\README.md` 和 AGENTS.md §15。成品 `D:\ppt\YYYYMMDD-标题.pptx`，源码 `_src\同名前缀\`，配图 `figures\<项目>\`，实验产物 `D:\output\results\<algo>_results\`；slide 只用相对 `assets/`。

## 其它项目已定论

- **structsplat**：GT 手性破缺是 90° 残差真因，允许反射 10.65→16.47dB；官方 2 图最佳。target 先报 alpha>0.5 覆盖率；瓶颈为覆盖、低 alpha 晕圈、几何外推与分辨率上限，核心 alpha>0.95 下曝光正常。排查顺序见 `structsplat/EXPERIMENTS.md`。
- **da3_human**：正式口径 3DGS EWA + 官方完整导出；2DGS 终止，法向 oracle 更差。前馈高斯宜作稀疏点云 init + densify，不宜整套灌入。
- **som_human 手部**：重建过平滑与两态几何冲突分开；SOM 和可微位姿精炼已证伪；按姿态分段并换运动信息来源（MANO/MHR/关键点/光流）。HiMoR 未实测，不能外推“学习方法都无效”。
- `deformable_human/BACKLOG.md` 管 A～J；`vggt_human` 三文档及 `DESIGN_face_pipeline.md` 先读；`figkit/SPEC.md` 管 PPT 规格；`tools_99/` 路径按 media_code cwd 解析，主体居中先跑 99k。
- Remy 标准包=`transforms.json + image/*.heic + pcd.ply`，`w2c=inv(M@diag(1,-1,-1,1))`，入口 `som_human/01_convert_capture.sh`。

## 环境与服务

- WSL2：RTX 3090 24GB，54GiB RAM/128GiB swap；conda 独立 env。网络会漂移先 curl 复测；国内源优先，境外必要时经 10808，国内源 no_proxy。
- colmap env 须 `openimageio=3.1`，4.x mapper 输出目录先建；直调二进制需 env lib 的 LD_LIBRARY_PATH。
- H100 ping 通而 22 timeout 优先查出口 IP 白名单及 WiFi 认证，详见 2026-11-21 日志。
