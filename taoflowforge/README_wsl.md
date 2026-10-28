# TaoFlowForge — WSL 本机复现指南（Ubuntu2404 + RTX 3090 24G）

与 README.md 并列，记录 WSL 本机特有的路径/网络/坑。命令统一用 README.md 里的。

## WSL 路径策略（与其它项目一致）
- 官方仓：`~/repos/Taobao3D`（Linux fs，clone 快；注意**代码在子目录 `TaoFlowForge/` 里**）
- 权重：`/mnt/d/wheel/taoflowforge_ms`（D 盘，推理时读一次，不占 Linux fs）
- 输出：`~/output/taoflowforge_results`（Linux fs；要长期保留就搬到 `/mnt/d/output/`）
- 以上三个值由 `proxy.env` 的 `TAOFLOWFORGE_DIR / TAOFLOWFORGE_MODEL_DIR /
  TAOFLOWFORGE_RESULTS_DIR` 注入（00a 首次运行会自动追加，已存在则不动）。

## 网络（2026-10-28 实测口径，漂移时需复测）
- GitHub 直连 ⚠️ **会漂移**：白天 clone 遇 `GnuTLS recv error -110`，经 v2rayN LAN 代理
  （`ip route show default` 网关:10808，`git -c http.proxy=...`）秒成。挂了先换代理。
- ModelScope ✅ 但**有总时限坑**：`modelscope` SDK 下载 26GB 权重 1 小时触发
  `[E1001] 下载总时限已耗尽`（部分文件，断点续传也不放大时限）。**大权重建议
  Windows 侧迅雷/curl 直下 resolve 直链**（`.../resolve/master/stageN.pt`，支持 `-C -`），
  落到 `/mnt/d/wheel/taoflowforge_ms/` 后 01 自动识别跳过下载。
- huggingface.co 直连 200 ✅ —— 但 DINOv3 / RMBG-2.0 都是 gated，要申请；
  默认链路不需要它们（checkpoint 若内嵌 DINO 则完全无 HF 依赖）。
  ⚠️ 若 stage0.pt 未内嵌 DINO，备用：ModelScope 有 facebook/dinov3-vitl16-pretrain-lvd1689m
  镜像（非 gated），`--dino-checkpoint` 指到本地目录即可。

## 环境要点
- conda env `taoflowforge`：python 3.10 + torch 2.5.1（**实际装到 +cu124**——tuna 源的
  torch 2.5.1 解析到 cu124 wheel，版本断言 `startswith('2.5.1')` 仍通过）+ torchvision 0.20.1
- **上游 requirements 有 bug**（2026-10 实测）：钉死 `Pillow==12.0.0` 与
  `gradio==5.50.0`（要求 pillow<12.0）冲突，pip 新解析器 ResolutionImpossible。
  00a 已内置兼容：拷 requirements 到 /tmp，sed 降级 `Pillow==11.3.0`，不改上游文件。
- **open3d 需 libgomp.so.1**：pip 版不捆绑；新 libgcc-ng(15.x) 也不带。WSL 无 sudo 时
  `conda install -c conda-forge libgomp`（00a 已内置该 fallback）。
- **无 CUDA 扩展编译**：open3d 0.19 / embreex 都是预编译 wheel，不需要 nvcc/gcc
  （与 sharp 的 gsplat JIT 不同，00a 不装 cuda-toolkit）
- `python -m taoflowforge.cli` 的包发现依赖 cwd = `~/repos/Taobao3D/TaoFlowForge`，
  02 脚本内部已 `cd` 过去；手动跑命令时别在仓根跑（仓根是 Taobao3D 三个项目的合集）

## 显存与性能
- 官方未公布显存/耗时。`--offload` 取值 **auto | on | off**（上游 choices，默认 auto；
  早期文档写的 "stage" 是错的，02 脚本已按 cli.py 核对修正）。
  3090 24G 首跑请 `nvidia-smi` 记录实测峰值并回填到 EXPERIMENTS.md。
- 若 OOM：先试 `OFFLOAD=on`；仍 OOM 则降输入图分辨率（预处理内部是 1024²，
  原图过大时先外部缩到 1~2k 再喂）。

## 适用边界（选品提醒）
- 输出 ≤2 万顶点 —— 道具/商品/场景摆件 OK；**人脸/人体精度不够**，
  vggt_human 链路别指望它替代 3DMM。
- 输入单图最佳形态：主体居中、白底/纯色底、无遮挡。真实照片建议配 RMBG
  （TFF_REMBG_DIR，需 HF 申请）或先外部抠图。
