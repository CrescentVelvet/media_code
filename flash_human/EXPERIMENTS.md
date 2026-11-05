# flash_human — 实验记录(日期倒序)

> 计划与决策依据见 [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md);本节只记实验数据与结论。

---

## 2026-11-05 冒烟:Rosemary full 档 6 视角 512²

**命令**(Blender 5.2.0 LTS,EEVEE):
```bash
"<BLENDER>/blender.exe" -b --python 01_render_gt.py -- \
    --fbx "D:/dataset/高质量人物FBX模型/Rosemary/Rosemary winters 4.0.fbx" \
    --texdir "D:/dataset/高质量人物FBX模型/Rosemary/textures" \
    --out "D:/dataset/human_fbx_gt/rosemary_smoke" \
    --mode full --n_views 6 --res 512 --seed 0
```

**产出**:`D:/dataset/human_fbx_gt/rosemary_smoke/`,rgb 6 + depth 6 + normal 6 + cameras.json + meta.json。

**质检(01c_check_gt.py):PASS**
- 深度 min 3.017m(视角1)/ 3.337m(视角3)/ 3.143m(视角6),相机距离 3.910m —— 差值 ≈ 0.4~0.6m 与人体+武器半径吻合;背景为 1e10(无限远)语义正确
- 法线模长中位 0.9997(三个抽查视角一致)
- cam0→target = 3.910m 与 meta.distance_m 精确一致(相机几何正确)
- 人物 alpha 有效占比 4.2%~6.0%(全身细长体型+武器,合理)

**渲染质量(ViewImage 人工检查)**:贴图全连(粉发/角/尾巴/剑/高跟靴完整),背景透明,补 Rim 背光后背面视角无死黑。人物占画面高约 72%,武器(剑)在侧视伸出画面——GT 语义可接受(真实采集同样会出画)。

**Blender 5.2 踩坑记录(后续批量渲染不再重复踩)**:
1. **FBX 导入器 bug**:FBX 含灯光对象时 `blen_read_light` 访问已被移除的 `lamp.cycles.cast_shadow`,hasattr 检查失效直接 AttributeError。解法:导入前 `bpy.ops.preferences.addon_disable(module="cycles")`(EEVEE 渲染不受影响),导入后删除 FBX 自带 LIGHT/CAMERA 对象。
2. **Compositor 改为节点组**:`scene.node_tree` 移除 → `scene.compositing_node_group = bpy.data.node_groups.new(name, "CompositorNodeTree")`,节点加到该组。
3. **File Output 节点**:`base_path`→`directory`,`file_slots.new(name)`→`file_output_items.new(socket_type, name)`(socket_type 取 FLOAT/VECTOR/RGBA),格式**仅支持 OPEN_EXR_MULTILAYER**(8bit PNG 可视化输出不可行),帧号占位用 `node.file_name = "xxx_####"` + `scene.frame_set(i)`。
4. **Render Result 无 layers/passes**:5.2 里 `Image.passes`/`Image.layers` 均被移除,pass 落盘只能走 File Output;EXR 数值质检用 **Blender 内置 OpenImageIO**(`import OpenImageIO`,OIIO 3.1.13)读取——本机与 WSL 的 cv2 EXR 均被编译禁用,环境变量开关无效。
5. 引擎 id:5.2 为 `BLENDER_EEVEE`(`BLENDER_EEVEE_NEXT` 已改名),脚本内用 enum 自动探测兜底。

**单视角耗时**:512² EEVEE 约 2~3s(含三 sun 灯光抖动),72 视角 1024² 预估 5~8 分钟/角色,批量工厂成本可接受。

## 2026-11-05 根因修复:渲染全员偏紫 = 无图环境纹理节点输出占位品红

**现象**:峰哥检视冒烟产物发现人物呈紫色(头发/脸/金属剑最重),疑材质缺失。

**排查链**(全部实测,工具脚本留档):
1. 贴图探测(`_t_probe_tex.py`):FBX 内嵌路径全部有效,relinked=0 missing=0 → **排除贴图丢失**。
2. 贴图真值:头发 basecolor 是米白(非粉)、身体 avg(182,155,149) 正常肤色 → **排除设计如此**。
3. 干净对照渲染(`_t_probe_color.py`,无 texdir/无 compositor/无 jitter):同一 FBX 颜色全正常 → 问题在 01 脚本自身。
4. 开关矩阵二分(`_t_bisect2.py`,头部均值 RGB 定量):SKIP_CS / SKIP_DITHER / SKIP_COMP / SKIP_JITTER / SKIP_PASSES+COMP 全关仍紫 (178,109,167);FILM_OPAQUE 更紫 (207,110,209)。
5. 定位:01 脚本世界节点挂了**无图的 ShaderNodeTexEnvironment**——EEVEE 里它输出占位品红,强度 0.8 = 全场品红环境光;金属剑最敏感(紫得最深),与现象完全吻合。对照脚本没挂该节点故正常。

**修复**(01_render_gt.py):删 env 节点,世界背景改显式中性灰 (0.05)×0.8 兜底;jitter_lights 的 Strength 抖动照旧。修复后同机位头部均值 (106,99,97),白发/正常肤色/金剑全部还原(证据 `_t_probe/head_zoom{,_fixed}.png`)。

(原记录"冒烟 PASS / 贴图全连"结论修正:01c 只质检深度/法线/相机几何,**色彩无自动判据**,人工检视漏检——后续 01c 应加头部区域色彩 sanity 检查,如均值 R-B 差超阈值报警。)

**附带发现**:
- **hair 材质 Base Color 未接贴图**(FBX 只有 Alpha 连线),渲染出默认灰白——该资产设计如此(米白毛发),非缺陷。
- **头发/睫毛 Alpha 连线但渲染不透明**:DITHERED 判定从"贴图名含 opacity"扩为"或 Principled.Alpha 有连接",防发片镂空变黑块。
- **探测脚本里的 'Cube'/'Material'/'Dots Stroke' 是 Blender 默认启动场景物体**(probe 脚本没做 read_factory_settings,与 FBX 无关);01 脚本早有空场,FBX 本身无夹带。仍保留「默认名+无贴图」防御过滤给其它资产防身。
- ⚠️ `D:/dataset/human_fbx_gt/rosemary_smoke/` 的 6 视角是**污染产物**(品红环境光),已作废,正式渲染用修复后脚本重出。

**下一步**:72 视角 1024² 正式渲染(修复后脚本)+ face 特写档验证。
