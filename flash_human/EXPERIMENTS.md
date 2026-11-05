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

**下一步**:72 视角 1024² 正式渲染 + face 特写档验证。
