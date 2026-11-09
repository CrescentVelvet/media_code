# Scene Splatter — WSL 复现指南

与 README.md 并列，记录 WSL2 + RTX 3090 24GB 本机专属事项。服务器复现看 README.md。

## 环境事实（2026-11-09 骨架阶段调研，跑通后更新）

| 项 | 值 | 来源 |
|---|---|---|
| 官方仓 | `~/repos/Scene-Splatter`（plain clone，无 submodule） | GitHub |
| conda env | `scene_splatter`，python 3.10，主通道走 NJU 镜像 | 00a |
| torch | 2.1.2 + cu118（官方指定），curl 精确 URL 落 `D:\wheel` 后本地直装 | 00a |
| nvcc 11.8 | **借 `flame_human` env 工具链**（编译期 PATH/CUDA_HOME 借用，不装包） | 00a，见「网络备忘」 |
| 光栅器 | `gaussianSplatting/submodules/diff-gaussian-rasterization` 本地编译 | 00a |
| 权重 | 3 份 → `D:\wheel\scene_splatter\`，消费侧软链（无官方 SHA256 可核对） | 01 |
| 显存 | README 未写；全栈 2B 级（DUSt3R ViT-L + DynamiCrafter 2B + 3DGS），24GB 预计充裕【待实测】 | — |
| 耗时 | 迭代轮数 × (5000 步 3DGS + 一段 16 帧视频)；N=100/interval=2 → 3~4 轮，分钟级~小时级【待实测】 | scenesplatter.py |

## 首次准备（WSL）
```bash
cd /mnt/<code-dir>/media_code
INSTALL_DEPS=1 bash scene_splatter/00a_setup_env.sh   # clone + env + 编译 + 补丁 + import 验证
bash scene_splatter/01_download_models.sh             # 三份权重（wheel 三段式）
# HF 不通时：HF_ENDPOINT=https://hf-mirror.com bash scene_splatter/01_download_models.sh
```

## 网络备忘
- ⚠️ 2026-11-09 网络漂移实测：**repo.anaconda.com 与 download.pytorch.org 双双超时**
  （直连与代理都死；AGENTS.md 里「pytorch.org 一直可用」已过时）。替代：
  - conda 主通道：`mirrors.nju.edu.cn/anaconda/pkgs/main`（USTC 已停 anaconda 镜像并
    302 到 NJU；tuna 的 repodata 能下但 ~2MB/s 偏慢，作 fallback）；
  - torch：pytorch.org 与阿里云索引页都不可用于 pip（前者超时、后者非 PEP503），
    但阿里云**精确 wheel URL 可下**——00a 直接 curl `mirrors.aliyun.com/pytorch-wheels/cu118/`
    的 wheel 落 `D:\wheel` 再本地直装（torch 2.3GB，国内镜像速度快）；
  - pip：阿里镜像。
- **nvidia conda 通道全灭**（tuna/NJU/BFSU 无 nvidia cloud，SJTU 跳 jcloud 403），
  pip 版 nvidia-cuda-nvcc 又配不齐 host gcc——所以 00a 编译期借用 `flame_human` env 的
  nvcc 11.8 + gcc 11.4 完整工具链（flame_human 恰好也是 torch 2.1.2+cu118）。换机器时
  用 `TOOLCHAIN_ENV=<env>` 指定别的含 nvcc 11.8 的 env。
- ⚠️ **xet 存储死局（2026-11-09 定论）**：HF 已把存量 LFS 迁 xet，大文件实际走
  `cas-bridge.xethub.hf.co`（CloudFront，13.33.x.x）。本机该域 **直连+代理 TLS 均断连**
  （CONNECT 200 后 ClientHello 无响应，SNI 级阻断），`us.aws.cdn.hf.co`、`cas-server.xethub.hf.co`
  同样全灭；**hf-mirror 只镜像元数据，对 xet 文件 302 透传死 CDN**（实测 flash3d v2 与
  ViewCrafter_25 都中招）。legacy LFS 的 `cdn-lfs.hf.co` 反而可达（403=活着）。
  → 这两个权重只能**峰哥手动迅雷**（迅雷有 P2P 缓存，不看本机 SNI）落 `D:\wheel\scene_splatter\`，
  重跑 01 自动软链。
- DUSt3R 走 `download.europe.naverlabs.com` 直连，慢/断时确认 proxy.env 代理已生效。

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
4. **相机轨迹格式 README 与代码矛盾已核实**（00a 的 --inspect-example 实测）：
   intrinsics 是 flat-4 `[fx_n, fy_n, cx_n, cy_n]`（README 说的 4×4 是错的），
   poses 是 **3×4** w2c（OpenCV，末行省略；官方示例首帧 t=[0.102, 0.020, -0.488]）。
   02 生成器已按此对齐；跑完用 `--selftest` 读回核对。
5. **分辨率硬编码**——`main()` 写死 `cfg.dataset.height=576; width=1024`，输入图会被
   resize 到 1024×576；02 生成器默认内参也按这个分辨率算，横评对齐 FlashWorld 时注意口径。
6. **av==10.0.0 被剔除**——PyPI 上该版本只有 sdist（编译需 pkg-config+ffmpeg 头，
   无 sudo 装不了），且全仓仅 SIBR_viewers 预处理工具 import av，非主路径。
   00a 装 requirements 时自动 grep -v 剔除；若官方后续加 av 依赖需换 binary 版 av。
7. **scipy==1.9.1 → 1.11.4**——官方 requirements 自相矛盾（scipy 1.9.1 要求
   numpy<1.25 却 pin numpy==1.26.0），新 resolver 直接 ResolutionImpossible。
   00a 自动放宽 scipy  pin；numpy 保留官方 1.26.0。
8. **setuptools 降到 <81**——torch 2.1 的 cpp_extension.py `import pkg_resources`，
   conda 默认 setuptools 84.x 已移除该模块（2025 后上游彻底删除），编译光栅器前
   00a 自动检测并降级 setuptools，否则 metadata-generation 即挂。

## 与 FlashWorld / Gen3R 的横评口径（compare_4methods 姊妹篇）
- 输入：统一单图 + 同一组 orbit 轨迹参数（YAW_SPAN/RADIUS/FOV_Y 对齐 FlashWorld 默认）。
- 产物：Scene Splatter 出 3DGS PLY（本补丁）；FlashWorld 直出 PLY；Gen3R 出点云需
  `gen3r/03_to_colmap.sh` 转 3DGS。统一到 PLY 后复用 compare_4methods 的渲染/评估链。
- 产物图/表放 `compare_4methods/`（峰哥 2026-11-09 指定，不新建文件夹）。
