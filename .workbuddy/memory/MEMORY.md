# media_code 项目长期约定

## commit 规范（用户明确要求，2026-09-01）
- 中文，格式 `<算法名>：简短描述`
- **默认只写标题行，不写正文**；正文最多 1-2 行（规范原文在 media_code/AGENTS.md 第 8 节）
- 历史教训：406153d 写了多段正文，被用户纠正

## 先读规范再动手（2026-09-05，项目工作纪律）
- **操作 git 前先读 AGENTS.md 的目录结构 / 命名 / commit 规范章节**，不要凭印象套用旧路径或旧约定
- **跑 WSL 训练前先读 README_wsl.md 的路径策略章节**（训练输出写 `~/output/` Linux fs，跑完用 08 搬到 `/mnt/d/`；不要直接写 drvfs/9p）
- 通用原则：动手前先读相关 README 章节，再下命令
- 历史教训（vggt_human）：
  - 09-05 误删整个 vggt_human 目录：没读 AGENTS.md 目录规范就 git rm + git restore，触发 WSL 9p 与 git 索引竞态，工作区整目录被识别为删除（git restore 救回）
  - 09-05 训练输出写错位置：没读 README_wsl.md 路径策略就把 RESULTS_DIR 指向 /mnt/d（drvfs），违反"训练写 Linux fs"规范（已搬运修正）

## 关键文档索引（media_code）
- `vggt_human/DESIGN_face_pipeline.md`（2026-09-08 建立）：人脸链路 v2 设计方案与决策记录
  （FLAME+DECA+468 点+重心绑定+表情驱动，八阶段全链路）。**讨论人脸链路前先读它**。
- vggt_human 三文档分工：README=运行命令；NOTES=原理/排错；EXPERIMENTS=实验结论（按日期倒序追加）。
- vggt_human 99 系列 = 重建产物的收尾/打包/搬运脚本（预设路径写在各自 main() 里，非 env 默认）：
  99a 收集 ply（压平成 `<批次>/ply/<task>.ply`）｜99b PLY→UWA MP4｜99c 收窄视角重封装（mp4_crop）
  ｜99d 复制三帧｜99e MP4→PLY｜99f 收集 ply+相机参数+front_image.jpg 到
  `<批次>/ply_viewlimit/<task>/`（三件成套，缺一即 skip）。

## 本机 WSL 网络与下载策略（2026-09-21 实测，跨会话通用）
- **本机不走代理**。`proxy.env` 里没有 `http_proxy`/`https_proxy`，只有 `HF_ENDPOINT=https://hf-mirror.com`、
  `HF_TOKEN`、`PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`。
  → `AGENTS.md` 的 `_env.sh` 代理模板在本机是空转（不报错，但也不起作用）。
- 可达性：`huggingface.co` ❌ 000｜`hf-mirror.com` ✅ 200（**下 HF 权重走这里**）｜
  `modelscope.cn` ✅ 200（备份源）｜**`github.com` ❌ 000**｜gitee 的 huggingface 镜像 ❌ 无效｜
  **`ghfast.top` ✅ 200 且 `git ls-remote` 可用（GitHub 唯一通道）**，另有 gh-proxy.com / ghproxy.net 备用。
- 结论：**任何需要 clone GitHub 的步骤都要走 `ghfast.top` 前缀**，否则直连必失败。

## 本机 WSL 硬件与 conda（2026-09-21 实测）
- WSL2，54 GiB 内存 / 128 GiB swap（`.wslconfig`：memory=56GB, swap=128GB, processors=8）；
  `/` 1007 G（可用 866 G）｜`/mnt/d` 9.1 T（可用 8.8 T，可写）；CPU i9-12900K 8 核。
- **`.wslconfig` 里的 `vhdxSize=100GB` 是无效键**，WSL 启动会报未知键警告（实际根分区 1007 G）。
- GPU：RTX 3090 24 GB，驱动 595.95，driver CUDA 13.2。
- conda：`/home/velvet/miniconda3`，env 有 4danyone / flame_human / minimax_h3 / osediff / sam3 / sharp / vggt_human。
  哪个 env 装了什么（py/torch/transformers/diffusers）见当日 memory log，新任务优先新建独立 env，别污染现有 env。
