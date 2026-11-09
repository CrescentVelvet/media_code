# Scene Splatter — WSL 复现指南

与 README.md 并列，记录 WSL2 + RTX 3090 24GB 本机专属事项。服务器复现看 README.md。

## 环境事实（2026-11-09 骨架阶段调研，跑通后更新）

| 项 | 值 | 来源 |
|---|---|---|
| 官方仓 | `~/repos/Scene-Splatter`（plain clone，无 submodule） | GitHub |
| conda env | `scene_splatter`，python 3.10 | 00a |
| torch | 2.1.2 + cu118（官方指定；pytorch.org 本机直连可用） | 官方 README |
| nvcc | conda `nvidia/label/cuda-11.8.0`（cuda-nvcc + cuda-cudart-dev） | 00a，WSL 无 sudo |
| 光栅器 | `gaussianSplatting/submodules/diff-gaussian-rasterization` 本地编译 | 00a |
| 权重 | 3 份 → `D:\wheel\scene_splatter\`，消费侧软链（无官方 SHA256 可核对） | 01 |
| 显存 | README 未写；全栈 2B 级（DUSt3R ViT-L + DynamiCrafter 2B + 3DGS），24GB 预计充裕【待实测】 | — |
| 耗时 | 迭代轮数 × (5000 步 3DGS + 一段 16 帧视频)；N=100/interval=2 → 3~4 轮，分钟级~小时级【待实测】 | scenesplatter.py |

## 首次准备（WSL）
```bash
cd /mnt/<code-dir>/media_code
INSTALL_DEPS=1 bash scene_splatter/00a_setup_env.sh   # clone + env + 编译 + 补丁 + import 验证
bash scene_splatter/01_download_models.sh             # 三份权重（wheel 三段式）
# HF 不通时：HF_ENDPOINT=https://aifasthub.com bash scene_splatter/01_download_models.sh
```

## 网络备忘
- huggingface.co 直连时通时断（2026-11-05 实测断）：01 会自动探测，断了回退 aifasthub；
  hf-mirror 当时证书失效，不作为默认候选。
- DUSt3R 走 `download.europe.naverlabs.com` 直连，慢/断时确认 proxy.env 代理已生效。
- pip 走阿里镜像；torch 走 download.pytorch.org（一直可用）。

## 跑完搬输出
```bash
bash scene_splatter/08_move_output.sh        # ~/output/... → /mnt/d/output/scene_splatter_results
```

## 已知坑点
1. **官方 main() 不导出 PLY**——`00a` 自动打 `patch_save_ply.py` 补丁（在 main() 末尾加
   `gaussians.save_ply(...)`）。重装/重 clone 官方仓后要重打（补丁幂等，重跑 00a 即可）。
2. **配置不能走命令行覆盖**——`scenesplatter.py` 内部 ViewCrafter 复用 `sys.argv` 做
   argparse，hydra CLI 覆盖会报 unrecognized arguments。`02_run_inference.sh` 改为直改
   `configs/config.yaml`（原文件备份为 `.media_code_bak`）。
3. **必须在官方仓根目录跑**——hydra 配置 `job.chdir: True` + `run.dir: exp`，进 `exp/`
   后代码用 `os.path.abspath('..')` 找仓根的 flash3d/viewcrafter/results，换目录会全部断链。
4. **相机轨迹格式有 README/代码矛盾**——README 说 intrinsics 是 4×4，但 `process_projs`
   按 `proj[0..3] = [fx_n, fy_n, cx_n, cy_n]` 归一化 flat-4 取值。02 生成器按 flat-4 写，
   **首跑前**先 `python scene_splatter/02_gen_traj.py --inspect-example` 核对官方示例实际结构。
5. **分辨率硬编码**——`main()` 写死 `cfg.dataset.height=576; width=1024`，输入图会被
   resize 到 1024×576；02 生成器默认内参也按这个分辨率算，横评对齐 FlashWorld 时注意口径。

## 与 FlashWorld / Gen3R 的横评口径（compare_4methods 姊妹篇）
- 输入：统一单图 + 同一组 orbit 轨迹参数（YAW_SPAN/RADIUS/FOV_Y 对齐 FlashWorld 默认）。
- 产物：Scene Splatter 出 3DGS PLY（本补丁）；FlashWorld 直出 PLY；Gen3R 出点云需
  `gen3r/03_to_colmap.sh` 转 3DGS。统一到 PLY 后复用 compare_4methods 的渲染/评估链。
- 产物图/表放 `compare_4methods/`（峰哥 2026-11-09 指定，不新建文件夹）。
