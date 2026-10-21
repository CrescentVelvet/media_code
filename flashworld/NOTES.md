# FlashWorld 原理与排错

LLM/AI 速查用。README 只放运行命令，机制与坑写这里。

## 1. 它是什么

FlashWorld（Xiamen Univ. + Tencent，ICLR 2026 Oral）把「多视图生成 → 重建」的两段式  
换成**生成过程中就地吐 3D 表示**：从一张图（或一段文本）出发，一次去噪链直接产出  
3D Gaussian，官方称 7 s / 单卡 A100，24 GB 卡也能跑（慢一档）。

三条关键设计（论文摘要 + 代码印证）：

| 设计       | 做法                                             | 代码位置                                            |
| -------- | ---------------------------------------------- | ----------------------------------------------- |
| 双模式预训练   | 同一个多视图扩散模型既能出图（MV-oriented）又能出 3D（3D-oriented） | `app.py: forward_generator` 的 `need_3d_mode` 分支 |
| 跨模式后训练蒸馏 | 一致但糊的 3D 模式向好看但抖的 MV 模式对齐分布                    | 训练代码未公开（需填 Google 表单）                           |
| 单视图/文本泛化 | 用海量单图和文本增强 OOD 泛化                              | 同上                                              |

## 2. 推理链到底做了什么（`app.py: GenerationSystem.generate`）

只跑 **4 步**去噪，`denoising_steps = [0, 250, 500, 750]`，scheduler 是  
`FlowMatchEulerDiscreteScheduler(shift=3)`（`app.py: __init__`）。

```
输入: cameras[N,11], image(1×3×H×W), text
  ↓ normalize_cameras: 以第 0 个相机为参考做刚体变换 + 按最大平移量归一化
  ↓ raymaps = [rays_d, rays_o - (rays_o·rays_d)rays_d]   # utils:create_raymaps
  ↓ text 前面拼 "[Static] "
  ↓ condition_latents[image_index] = VAE.encode(image)    # 只有一帧有条件
  ↓ 去噪 4 步：
      i < 3:  transformer(noisy + raymaps + condition)
                → (v_pred[48ch], feats[1024ch])
                → latents_pred = noisy - sigma * v_pred
                → recon_decoder 渲染 3D → VAE 重编码 → latents_pred_3d
                → feedback: 下一步 condition += [prev_feats, prev_latents_pred]
      i == 3: transformer 一次前向，不再回灌
                → recon_decoder(feats, unscale(latents_pred), cameras)
                → scene_params [num_gaussians, D]
  ↓ export_gaussians: opacity 剪枝 + 反 sigmoid/反 log + means·T_norm 还原尺度
输出: gaussians.ply / .spz（+ 可选 fps 插值视频）
```

要点：

- **latent 48 维、feature 1024 维、空间下采样 16×、时间下采样 4×**；24 帧 → 6 个 latent 帧。
- `scene_params` 在**归一化世界**里，`export_gaussians` 才乘回 `T_norm`；所以 PLY 的世界尺度  
  取决于相机轨迹的绝对半径（我们自己造的轨迹半径 = `RADIUS`，默认 1.0）。
- 高斯属性顺序：`xyz(3) | opacity(1) | scale(3) | rotation(4) | SH((d+1)²×3)`，  
  SH 阶数由维度反推（`utils:export_gaussians`）。
- **CLI 的 opacity 剪枝阈值是 0.000（不剪）**，web 界面是 0.00025 —— 官方 README 说 CLI  
  用未压缩参数、渲染更好，指的就是这个差别（`cli.py` vs `app.py`）。

## 3. 输入 json 与相机约定（本仓 02_make_input.py 的依据）

官方没有 Python 版轨迹生成，轨迹在前端 `index.html` 里用 three.js 造。反推出的约定：

| 项     | 约定                                             | 出处                                                                |
| ----- | ---------------------------------------------- | ----------------------------------------------------------------- |
| 位姿    | **c2w**（相机→世界）                                 | `utils:create_rays` 用 `quaternion_to_matrix` 直接当 c2w              |
| 四元数顺序 | **[w, x, y, z]**                               | `index.html: cameras.map(cam => [cam.quaternion.w, .x, .y, .z])`  |
| 朝向    | 看向 **-Z**，上方向 +Y                               | `create_rays` 里 `zs = -1`；`setFromUnitVectors((0,0,-1), lookDir)` |
| 内参    | `fx = fy = 0.5/tan(fov/2)*H`，`cx=W/2`，`cy=H/2` | `index.html: SaveAllToJson`                                       |
| 分辨率   | `resolution = [帧数, 高, 宽]`，官方只给 `24×480×704`    | `supportedResolutions`                                            |
| 关键帧   | `image_index` 指定输入图落在哪一帧                       | `masks[:, image_index] = 1`                                       |

`normalize_cameras`（`utils.py`）用**第 0 个相机**做参考系（`ref_w2c = inv(c2w[0])`），  
再除以所有相机平移量的最大值。含义有两条实践约束：

1. 第 0 个相机的位置就是世界原点的候选；我们的轨迹里它是环绕角度的起点。
2. **不要把相机全放在同一点**：`T_norm` 会趋近 0，缩放分母只有 `1e-2` 兜底，几何会炸。

## 4. 与官方 demo 的差异（本仓的默认值都是刻意的）

| 项   | 官方 demo                                                          | 本仓默认                             | 原因                                       |
| --- | ---------------------------------------------------------------- | -------------------------------- | ---------------------------------------- |
| 轨迹  | ±15° 小轨道（`totalOrbitAngle = 15°`），只点一次生成 1 个关键帧再由前端 slerp 成 24 帧 | `TRAJ=orbit`、`YAW_SPAN=40`（±20°） | demo 是「拍一段运镜」；抽 3D 资产要覆盖更宽，但跨度太大超出训练分布会崩 |
| 帧数  | 24（固定）                                                           | 24                               | 模型按 24 帧训练；改帧数只是插值，不增加信息                 |
| FOV | 60（UI 默认，与输入图真实 FOV 无关）                                          | 60                               | 单图没有内参真值，只能假设；`FOV` 可覆盖                  |
| 视频  | 默认开                                                              | 默认关（`VIDEO=1` 开）                 | 额外 VAE 解码一遍 `(N-1)*12+1` 帧，24 GB 卡很吃紧    |

## 5. 24 GB（3090）能跑吗 —— 逐条核对

| 关注点  | 结论                                                                                 | 依据                                                      |
| ---- | ---------------------------------------------------------------------------------- | ------------------------------------------------------- |
| 显存   | 官方给 24 GB 档 = `--offload_t5 --offload_transformer_during_vae`，20 s/场景              | README 表格                                               |
| FP8  | **3090 无原生 FP8**（sm_86 < 9.0）→ `native_fp8_support=False`，退化成 `F.linear(bf16 反量化)` | `quant.py: torch.cuda.get_device_capability() >= (9,0)` |
| 后果   | 能跑通，但每层 Linear 每次前向都要反量化 → 比官方标称慢，显存带宽是瓶颈                                          | 同上 `fp8_gemm` 的 else 分支                                 |
| bf16 | 3090 支持（Ampere），autocast 全程 bf16 没问题                                               | `torch.amp.autocast(dtype=torch.bfloat16)`              |
| 内存   | ckpt 用 `map_location="cpu"` 加载，19.5 GB + 基座 ≈ 需要 30 GB+ 系统内存                       | `app.py: torch.load(..., map_location="cpu")`；本机 56 GB  |

## 6. 多图输入：官方**不支持**（已被本仓补丁解锁，见 §7d）

用户提到的「单图或多图」在官方代码里只有**单图**：

- `image_prompt` 是单个字段，`image_index` 是单个整数（`cli.py` / `app.py` 一致）；
- `masks[:, image_index] = 1`、`condition_latents[:, :, image_index] = latent` 只写一帧；
- 训练阶段（论文）也是「单图 + 文本」条件。

**本仓已实现并验证多图条件**（`04_patch_multi_image.py`，2026-10-21）：把多张图各自  
VAE encode 写进多个帧槽位即可，模型零改动、推理耗时不涨，重影消除，实测数据见 §7d。  
`masks` 变量下游其实没被消费，latent 零/非零本身就是掩码——这是补丁能成立的原因。

## 7. 排错

### 装环境

| 现象                                                                    | 原因 / 处理                                                                                                                                                                                                                                                                                                                                         |
| --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `import gsplat` 失败且 pip 装不上                                           | gsplat 是**源码编译**（不同于 sharp 用的 1.5.3 纯 Python wheel）。要 `nvcc` + `g++` + `TORCH_CUDA_ARCH_LIST`。00a 已装 `cuda-nvcc` + `gxx_linux-64` 进 env 前缀，`_env.sh` 负责把 `CUDA_HOME` 指过去                                                                                                                                                                        |
| 编译报 `glm/gtc/type_ptr.hpp: No such file or directory`                 | **gsplat 的 GLM 是 git 子模块**（`gsplat/cuda/csrc/third_party/glm`）。`git fetch --depth 1 <commit>` 不会拉子模块。本机 `git submodule update --init` 会**卡死**（实测 9 min 零字节），改用 tarball：`curl -sL https://gh-proxy.com/https://github.com/g-truc/glm/archive/refs/tags/1.0.1.tar.gz -o /tmp/glm.tar.gz && tar xzf /tmp/glm.tar.gz -C <该目录> --strip-components=1` |
| `No matching distribution found for uvicorn==0.37.9`                  | 官方 requirements 里这个版本**在 PyPI 上不存在**（只有 0.37.0 → 0.38.0）。放宽成 `uvicorn>=0.37`                                                                                                                                                                                                                                                                    |
| spz 编译报 `CMakeLists.txt:14 (find_package)` + `Configuring incomplete` | 本机 pip 装到 **cmake 4.4**，而 spz 的 `cmake_minimum_required` 低于 3.5（CMake 4 已移除兼容）。装 `"cmake<4"`（实测 3.31.10 可用）                                                                                                                                                                                                                                     |
| spz 编译报 `Cannot import 'scikit_build_core.build'`                     | 用了 `--no-build-isolation`，构建后端必须在 env 里：先 `pip install scikit-build-core cmake`                                                                                                                                                                                                                                                                 |
| `import diffusers` 报 `NameError: name 'logger' is not defined`        | clone 源自带 **torchao 0.16.0** 与 torch 2.6 不兼容：diffusers 0.36.0.dev0 的 `torchao_quantizer.py` 在 import 期跑 `_update_torch_safe_globals()`，torchao 缺 `dtypes.uintx.uint4_layout` → 走进引用未定义 `logger` 的 except。本项目不用 torchao，**卸载**即可                                                                                                                 |
| 编译极慢 / symlink 报错                                                     | 在 `/mnt/c` `/mnt/d`（9p）上编译的典型症状。代码必须在 `~/repos/`                                                                                                                                                                                                                                                                                                |
| `WanTransformer3DModel` 属性/签名报错                                       | transformers 版本不对。clone 源 `minimax_h3` 是 **5.15.1**，必须降到 **4.57.0**（00a 会装）                                                                                                                                                                                                                                                                     |
| conda 把 python 换成 GraalPy                                             | `gxx_linux-64` 的老坑；00a 每条 conda install 都显式 `python=3.11`，并用 `platform.python_implementation()` 校验                                                                                                                                                                                                                                              |
| `pip install git+https://...` 卡住                                      | 官方 requirements 里三条 git 依赖已改为本地 clone（`$FLASHWORLD_DEPS`），不走网络                                                                                                                                                                                                                                                                                  |

### 下权重

| 现象                                        | 原因 / 处理                                                                                                                                                                                                                                          |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `from_pretrained` 卡住 / 报连接错误              | 没设 `HF_HOME` 或 cache 里没快照。`02_run_inference.sh` 会 export `HF_HOME` + `HF_HUB_OFFLINE=1`                                                                                                                                                          |
| hf-mirror 只有 ~2.3 MB/s                    | 本机实测；54 GB 要 7 h。走迅雷（`download_urls.md`）+ `01c_assemble_flat.py` + `01b_link_local_weights.sh`                                                                                                                                                   |
| 自动下载中途 `httpx.ReadTimeout`                | 已实测发生（约 4 GB 处）。`snapshot_download` 可断点续传，直接重跑同一条命令即可                                                                                                                                                                                            |
| 迅雷里 hf-mirror 链接下不动                       | **Windows 侧 hf-mirror 不通**（`CONNECT tunnel failed, response 502`），且 `huggingface.co` 的 `resolve` 302 到 `cdn-lfs-us-1.hf.co` 后 TLS 握手失败（本机 HTTP 代理只放行了 hf.co 主域）。迅雷只能用 **modelscope** 链接（`download_urls.txt` 就是 modelscope 版）                     |
| 迅雷批量下完是平的、结构不对                            | 迅雷按 URL 末段平铺保存。跑 `01c_assemble_flat.py --src <迅雷目录>` 归位，它会按字节数校验并自动补 5 个 KB 级小文件                                                                                                                                                                 |
| 迅雷下出 `config(1).json`                     | 三个组件的 `config.json` 重名。这些**不在** `download_urls.txt` 里，由 `01c` 按完整路径补下；源目录里同名的不会被误用（`01c` 对多重名 basename 直接忽略）                                                                                                                                     |
| `01b` 之后仍报找不到                             | 检查 `$HF_HOME/hub/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers/refs/main` 内容是否为 `local`                                                                                                                                                               |
| 迅雷里 `model.ckpt.xltd` 看着"下完了"却没变成 `.ckpt` | `.xltd` 是**未完成的临时名**，迅雷收尾时才改名。判据别只看体积：`model.ckpt` 官方大小是 **20,936,494,840** 字节；`.xltd` 可能因预分配**比官方还大几百字节**，但仍缺尾部数据——实测 20,936,495,104 的文件尾部 4 KB 与官方不一致，且**没有 zip EOCD**（正常 torch ckpt 末尾必有 `PK\x05\x06`，可用它当完整性快速判据，比 `torch.load` 省 20 GB 内存）。 |
| 迅雷 ckpt 任务卡住（大小长时间零增长）                    | 迅雷进程还在但连接已死。回迅雷点「继续/重新校验」，它走自己的通道（实测能到 19.5 GB）。WSL 侧此时往往帮不上：`huggingface.co` 的 `resolve` 会被 RST（只有 `api/` 域通），`hf-mirror` 反复超时；`modelscope` **没有** FlashWorld 镜像（已搜过）                                                                           |

### 跑推理

| 现象                                                                                          | 原因 / 处理                                                                                                                                                                                                                                                                                                                                                  |
| ------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `OfflineModeIsEnabled: Cannot reach https://.../api/models/Wan-AI/Wan2.2-TI2V-5B-Diffusers` | `WanTransformer3DModel` 是**分片**权重（5 shard + `.index.json`），diffusers 走 `_get_checkpoint_shard_files` 的「Hub 模型 ID」分支时**无条件调 `model_info()`** 联网；而 `modeling_utils.py` 里 `local_files_only` 默认是 `None` 而非 `HF_HUB_OFFLINE`，于是离线必挂。`00b_patch_deps.py` 已补：`local_files_only is None` 时回退读 `HF_HUB_OFFLINE`（只改 vendored diffusers，不碰官方代码）。重装环境后 00a/00 会自动重打 |
| CUDA OOM                                                                                    | `REALLY_LOW_VRAM=1`（加 `--offload_vae`，慢到分钟级）；或降 `N_FRAME`（会偏离训练分布）；或确认没有别的进程占卡                                                                                                                                                                                                                                                                           |
| 跑完没有产物、也没报错                                                                                 | `cli.py: process_generation_request` 出错时**只 return 不抛异常**，看 stdout 有没有 `Processing error`；json 里的 `image_prompt` 必须是**绝对路径且可读**                                                                                                                                                                                                                          |
| 输出几何被压扁/尺度异常                                                                                | 相机轨迹半径过小或所有相机几乎同点 → `T_norm ≈ 0`；保持 `RADIUS>0.1` 且轨迹有真实位移                                                                                                                                                                                                                                                                                                |
| 画面透视不对                                                                                      | 输入图真实 FOV 与 `FOV=60` 差太多；按观感调 `FOV`（30~90）                                                                                                                                                                                                                                                                                                               |
| PLY 在 viewer 里发黑                                                                            | 这是 SH 系数直出的 PLY，部分 viewer 需要 SH degree 顺序匹配；官方也是这个格式                                                                                                                                                                                                                                                                                                     |

### 读 PLY（自己渲染/分析时）

| 现象                 | 原因 / 处理                                                                                                                                                                                                          |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 自己用 gsplat 渲染全黑    | **PLY 里 opacity 存的是 logit、scale 存的是 log**（官方 `export_gaussians` 做了 inverse activation，utils.py:374-375）。渲染前必须 `sigmoid`/`exp` 还原——`03_render_ply.py` 已处理                                                         |
| 用 json 原始相机渲染全黑/偏掉 | PLY 坐标系不是原始世界系，&#x662F;**「json cameras[0] 的第一帧相机系」**：`normalize_cameras` 左乘 `ref_w2c=inv(c2w[0])`，但导出时只乘回了 `T_norm`，没乘回 `ref_w2c` 的逆。渲染要拿 `ref_w2c @ c2w_i`，且 json 是 OpenGL（看 -Z）→ gsplat 要 `diag(1,-1,-1,1)` 翻转 |
| viewer 里一团淡雾、像噪点   | cli.py 导出 `opacity_threshold=0.0`（不剪枝），而**实测生成的高斯 opacity p50≈0.006**（96% 近透明）。官方 web 版用 0.00025。想看清楚主体，读 PLY 后按 sigmoid 后 opacity>0.05~0.3 剪枝                                                                   |
| 渲染视野被放大数倍          | json 里的 fx/fy/cx/cy 是**输入原图分辨率**的（app.py/cli.py 会先 center-crop 到 H×W 并 `fx*=scale`），渲染时要重放同一套 crop 换算，不能直接拿来渲 480×704。`03_render_ply.py` 已处理                                                                     |

## 7b. 单图生成的能力边界（2026-10-19 实测，佳佳抱枕采集包）

输入：Remy 采集包 `01e954de…-佳佳抱枕` 的 000 帧（正面全身照，1440×1920），真实内参 + 真实相机轨迹（`03_capture_input.py` 生成）。

| 轨迹                                                     | 结果                                                                                         |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| **全轨迹 ~212°**（82 帧均匀抽 24，yaw 66.9°→-44°→168° 含折返）      | 输入视角布局完全正确（人/派大星/椅子/背景都在位），但**重影/幽灵**（派大星两个影子）；绕到 ~35° 场景尚存但糊、地板出现黑雾；绕到 ~100° 几乎全白只剩模糊人影   |
| **窄轨迹 ~96°**（frames 0..23 连续 24 帧，yaw 66.9°→-28.8° 单向） | 明显更好：偏离输入 45° 时结构正确（侧面坐姿可辨，细节糊）；偏离 95° 时场景结构仍在但派大星颜色变淡。对比全轨迹版在 100° 处全白，说明**容量被角度摊薄是主因之一** |

机理：24 帧里只有 `image_index` 那一帧有条件真值，其余全靠模型想象。跨大角度时模型在多个深度上各放了一份解释 → 幽灵（和 som_human 的「两态几何冲突」同族，但这里根因是**条件不足**而非运动）。单图的合理用途是**小角度新视角合成**（官方 demo 轨迹仅 ±15°），不是大角度重建。

对我们的价值定位：单图 130 秒出 4M 高斯的「先验场景」，可做 `scene_gs` 的背景初始化/补全；正面主体区域它不稳（重影），**不能**替代 COLMAP 重建。多图输入要改官方 mask 逻辑（第 6 节），未验证。

**本机实测基线（2026-10-19，RTX 3090 24 GB，480×704×24 帧，单图）**：模型加载 ≈ 2 min（含 858 个张量 fp8 量化 14 s）→ 生成 129 s →  
`gaussians.ply` 588 MB / **4,055,040 个高斯**（38 float/顶点，结构自洽）。显存未打满，`--offload_t5 --offload_transformer_during_vae` 够用。

## 7c. 重影归因实验：不是位姿问题（2026-10-19）

针对「人、背景都有明显重影，是不是位姿问题」，做了三个独立实验，结论：**位姿无错，重影是单图生成的多深度副本**。


| 实验          | 做法                                                                                                                                             | 结果                                                           | 结论                              |
| ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------- |
| 位姿独立验证      | `_t_pose_check.py`：采集包 `pcd.ply`（79,355 点）按 transforms.json 位姿 + 内参投影回真实帧 000/012/023/041/080（`w2c = inv(M @ diag(1,-1,-1,1))`，同 som_human 口径） | 投影点严格落在人/椅子/派大星的真实轮廓上，正面/侧面帧均对齐（`05_pose_check/`）            | ✅ 位姿与坐标转换正确，重影与此无关              |
| 重影副本是否为透明浮尘 | `03_render_ply.py` 加 `OPACITY_MIN`，对窄轨迹 PLY 按 sigmoid 后 opacity ≥ 0.1/0.2/0.3 剪枝重渲（`04_render_prune/`）                                         | 即使只剩 7.4% 高斯（op>0.3），输入视角仍是两个完整的人                            | ❌ 重影不是低透明度噪声，**两个副本都是实体几何**     |
| 多深度副本机理验证   | `_t_depth_hist.py`：取输入帧相机系下画面中央 ROI（人所在区域）内 op>0.3 的高斯，看深度直方图                                                                                  | 三个分离的簇：~0.95-1.05 / ~1.11-1.31 / ~1.70-1.90（单位=归一化后），簇间有明显低谷 | ✅ 同一个人被放在了**多个深度**上，单图深度歧义的典型形态 |

补充判据：重影在 `image_index` 对应帧（有条件真值的那帧）就存在——这一帧的 ray map 就是真实相机，位姿错只会造成透视扭曲/视差异常，**不可能在有条件帧上变出两份完整内容**。

可操作的缓解方向（按代价排序）：

1. **用窄轨迹**（已验证 96° 显著好于 212°，但仍残留输入视角重影）；
2. **深度分层剪枝**：按输入视线深度把副本簇分开，只保留主簇（直方图低谷可做切分点，未实现）；
3. **多图条件**：改 `app.py` 的 mask/latent 写入逻辑塞多视角真值（第 6 节，未验证）——这是治本方  
   向，位姿已验证可用，采集包 82 帧任选。

## 7d. 多图条件实测：重影治本成功（2026-10-21）

第 6 节预言的「改 mask 逻辑塞多图条件」已落地并验证**有效**。实现比预期简单：  
条件注入就是往 `condition_latents` 的帧槽位写 VAE latent（`masks` 变量下游没被消费，  
latent 零/非零本身就是掩码），所以多图 = 多张图各自 VAE encode（T=1，与官方单图  
路径逐帧一致）写进多个槽位，**模型权重零改动**。补丁：`04_patch_multi_image.py`  
（幂等，打 app.py + cli.py；还原 `git checkout app.py cli.py`）；输入 json 用  
`05_make_multi_input.py` 生成（`image_prompts: [{path, index}, ...]`）。

佳佳抱枕窄轨迹（24 帧 / ~96°）实测，逐帧 PSNR vs 采集包真值（带 * 为条件帧）：

| 帧  | 1cond（官方单图） | 2cond (0,23) | 3cond (0,11,23) |
| -- | ----------- | ------------ | --------------- |
| 0* | 18.22       | **21.29**    | 18.97           |
| 6  | 12.99       | 16.61        | **17.74**       |
| 11 | 11.72       | 14.47        | **19.60***      |
| 17 | 10.07       | 14.34        | **15.45**       |
| 23 | 9.27        | **17.96***   | **17.95***      |

结论：

- **重影消除**：1cond 在输入视角就有的「两个头」在 2cond/3cond 里完全消失，  
  单人单影；偏离条件帧的中间视角（6/17）也连贯，侧面（23）结构正确。
- **条件帧锚定 + 插值泛化都成立**：条件帧 PSNR 跳至 18~21 dB；  
  非条件帧也随条件数提升（17 帧：10.07 → 14.34 → 15.45），说明 Transformer  
  的时空注意力能把多帧真值传播到中间帧，不只是记忆条件帧。
- **2cond 已够用大半**：两端锚点（0,23）就把弧内 PSNR 提了 4~~9 dB；  
  3cond 再加中间锚点后中段（6/11/17）继续提 1~~5 dB。条件帧应**均匀铺满轨迹弧**。
- **代价**：推理耗时不变（~135 s，条件数只影响 VAE encode 几次），代价是  
  需要多视角真值图 + 一致位姿（我们有采集包，天然满足）。
- **OOD 担忧未兑现**：模型训练时只见过单帧条件，但 Wan2.2 TI2V 基座预训练见过  
  首末帧条件形态，实测多帧条件没有崩、没有颜色混乱。

产物：`06_multi_cond/{jiabao_2cond,jiabao_3cond}/gaussians.ply` +  
`render_{2,3}cond/` + `compare_psnr.png`（真值/1/2/3cond 五行对比拼图）。  
坑：03_render_ply.py 原来只认 `image_prompt` 字段，多图 json 下跳过了 crop 内参  
重放导致渲染放大 2.7 倍——已修（兼容 `image_prompts[0].path`）。

## 7e. 条件帧数扫描：5cond 后饱和，24cond 是上限对照（2026-10-21）

同一条窄轨迹（24 帧 / ~96°）扫 1/2/3/5/8/12/24 条件帧（均匀铺满弧），  
全 24 帧逐一对真值算 PSNR（`_t_cond_sweep.sh` + `_t_cond_sweep_eval.sh`，  
明细 `sweep_psnr.txt`，曲线 `sweep_curve.png`）：

| 条件数 | 1 | 2 | 3 | 5 | 8 | 12 | 24（全帧） |
| ----- | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| 全帧均值 | 11.96 | 16.04 | 17.44 | 17.59 | 17.85 | 17.86 | **18.47** |
| 条件帧均值 | 18.22 | 19.63 | 18.84 | 17.92 | 18.03 | 17.90 | 18.47 |
| 非条件帧均值 | 11.68 | 15.71 | 17.24 | 17.50 | 17.76 | 17.83 | — |

结论：

- **3cond 已把重影问题解决，5cond 后收益饱和**（17.59 → 17.85 → 17.86 几乎平）。
  实用推荐 **5~8 cond（每 3~6 帧一个锚点）**：再往上每帧边际收益 <0.1 dB。
- **24cond（全帧条件）只比 8cond 高 0.6 dB**，且全帧曲线仍停在 17.2~19.3 dB
  没有逼近真值——这是模型生成能力 / VAE 重建的上限，不是条件不够。
- **条件帧数量增加后，条件帧本身的 PSNR 反而略降**（19.63@2cond → 17.9@12cond）：
  多帧条件把容量摊到整条弧上，单帧锚定精度被「平均」掉，属正常 trade-off。
- 1cond 曲线沿弧单调衰减（18.2 → 9.3），2cond 呈 U 形（中段洼地），
  ≥3cond 曲线基本拉平——与「时空注意力把真值向中间帧传播」的机理一致。

## 8. 复现边界（诚实说明）

- 官方**只发推理**，训练代码要填表申请 → 本仓只做推理复现，不涉及双模式预训练/跨模式蒸馏。
- 权重许可 **CC BY-NC-SA 4.0**，README 写明「仅学术研究」。
- 论文声称 7 s/A100；本机 3090 + 无原生 FP8 + 双 offload，实测数字见 EXPERIMENTS.md（跑完回填）。
